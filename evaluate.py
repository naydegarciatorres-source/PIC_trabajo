"""Evaluation of a trained Quick, Draw! model (CONTRACT section 6).

Usage:
    python evaluate.py --checkpoint checkpoints/cnn/best.pt --data-dir data/processed \
        --split test --results-dir results

Writes into ``results/{run}/`` (run = name of the checkpoint's parent folder):
    metrics_{split}.json   global metrics + per-class metrics + 10 most confused pairs
    per_class_{split}.csv  accuracy / precision / recall / support per class
    confusion_{split}.png  row-normalised confusion matrix

All metrics are computed with torch/numpy (no sklearn).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Sequence[int] = (1, 3)) -> dict:
    """Fraction of samples whose target is among the k highest logits, for each k in ``ks``.

    Returns ``{k: accuracy_in_[0,1]}``. k larger than the number of classes is clamped.
    Empty input -> 0.0 for every k.
    """
    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets).long().view(-1)
    out = {}
    if logits.numel() == 0 or targets.numel() == 0:
        return {int(k): 0.0 for k in ks}
    num_classes = logits.shape[-1]
    maxk = min(max(ks), num_classes)
    top = logits.topk(maxk, dim=-1).indices  # (N, maxk)
    hits = top.eq(targets.view(-1, 1))  # (N, maxk)
    n = targets.numel()
    for k in ks:
        kk = min(int(k), num_classes)
        out[int(k)] = float(hits[:, :kk].any(dim=1).float().sum().item() / n)
    return out


def _unpack_batch(batch, mode: str, device):
    """Return (inputs, lengths_or_None, labels) on ``device`` for both loader formats."""
    if mode == "strokes":
        if len(batch) == 3:
            x, lengths, y = batch
        else:  # un-collated (x, y) fallback: assume full lengths
            x, y = batch
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
        return x.to(device).float(), lengths.long().cpu(), torch.as_tensor(y).long().to(device)
    x, y = batch[0], batch[-1]
    return x.to(device).float(), None, torch.as_tensor(y).long().to(device)


def _forward(model, x, lengths, mode):
    if mode == "strokes":
        # lengths must stay on CPU for pack_padded_sequence
        return model(x, lengths)
    return model(x)


def per_class_from_confusion(confusion: np.ndarray, classes: Sequence[str] | None = None) -> list:
    """Per-class accuracy (=recall), precision, recall, F1 and support from a confusion matrix
    whose rows are true labels and columns predicted labels."""
    cm = np.asarray(confusion, dtype=np.float64)
    k = cm.shape[0]
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    rows = []
    for i in range(k):
        rows.append({
            "index": i,
            "class": classes[i] if classes is not None else str(i),
            "accuracy": float(recall[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted": int(predicted[i]),
        })
    return rows


def most_confused_pairs(confusion: np.ndarray, classes: Sequence[str] | None = None, n: int = 10) -> list:
    """The ``n`` largest off-diagonal entries: (true, predicted, count, rate over true support)."""
    cm = np.asarray(confusion, dtype=np.int64).copy()
    support = cm.sum(axis=1)
    np.fill_diagonal(cm, 0)
    flat = cm.ravel()
    if flat.size == 0:
        return []
    order = np.argsort(-flat, kind="stable")
    k = cm.shape[0]
    pairs = []
    for idx in order[:n]:
        count = int(flat[idx])
        if count <= 0:
            break
        t, p = divmod(int(idx), k)
        pairs.append({
            "true": classes[t] if classes is not None else str(t),
            "predicted": classes[p] if classes is not None else str(p),
            "true_index": t,
            "predicted_index": p,
            "count": count,
            "rate": float(count / support[t]) if support[t] > 0 else 0.0,
        })
    return pairs


@torch.inference_mode()
def evaluate_model(model, loader: Iterable, num_classes: int, device="cpu", mode: str = "image",
                   classes: Sequence[str] | None = None, ks: Sequence[int] = (1, 3)) -> dict:
    """Evaluate ``model`` on every batch of ``loader``.

    ``mode``: "image" (batches ``(x, y)``) or "strokes" (batches ``(x, lengths, y)`` from
    ``collate_strokes``).

    Returns a dict with keys:
      ``loss`` (mean cross-entropy), ``top1``, ``top3`` (and ``top{k}`` for every k in ``ks``),
      ``n`` (number of samples), ``num_classes``, ``confusion`` (np.int64 array (C,C), rows = true,
      cols = predicted), ``per_class`` (list of dicts, see ``per_class_from_confusion``),
      ``per_class_accuracy`` (np.float64 (C,)), ``macro_precision``, ``macro_recall``,
      ``macro_f1``, ``most_confused`` (10 pairs), ``seconds``.
    """
    if mode not in ("image", "strokes"):
        raise ValueError(f"mode must be 'image' or 'strokes', got {mode!r}")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    ks = tuple(int(k) for k in ks)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    loss_sum = 0.0
    hits = {k: 0.0 for k in ks}
    n = 0
    t0 = time.perf_counter()
    for batch in loader:
        x, lengths, y = _unpack_batch(batch, mode, device)
        logits = _forward(model, x, lengths, mode).float()
        bsz = y.numel()
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        acc = topk_accuracy(logits, y, ks)
        for k in ks:
            hits[k] += acc[k] * bsz
        pred = logits.argmax(dim=1)
        idx = (y.cpu() * num_classes + pred.cpu())
        confusion += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)
        n += bsz
    if was_training:
        model.train()

    cm = confusion.numpy()
    per_class = per_class_from_confusion(cm, classes)
    present = [r for r in per_class if r["support"] > 0]
    result = {
        "loss": loss_sum / n if n else float("nan"),
        "n": int(n),
        "num_classes": int(num_classes),
        "confusion": cm,
        "per_class": per_class,
        "per_class_accuracy": np.array([r["accuracy"] for r in per_class], dtype=np.float64),
        "macro_precision": float(np.mean([r["precision"] for r in present])) if present else 0.0,
        "macro_recall": float(np.mean([r["recall"] for r in present])) if present else 0.0,
        "macro_f1": float(np.mean([r["f1"] for r in present])) if present else 0.0,
        "most_confused": most_confused_pairs(cm, classes, n=10),
        "seconds": time.perf_counter() - t0,
    }
    for k in ks:
        result[f"top{k}"] = hits[k] / n if n else 0.0
    return result


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------
def write_per_class_csv(path: str, per_class: list) -> None:
    fields = ["index", "class", "accuracy", "precision", "recall", "f1", "support", "predicted"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_class:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]) for k in fields})


def plot_confusion(path: str, confusion: np.ndarray, classes: Sequence[str], title: str = "") -> None:
    """Row-normalised confusion matrix. Tick labels shown up to 60 classes, hidden above."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion, dtype=np.float64)
    k = cm.shape[0]
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    show_labels = k <= 60
    side = min(30.0, max(6.0, 0.28 * k + 3.0)) if show_labels else 14.0
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    if show_labels:
        fs = 9 if k <= 30 else 7
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(classes, rotation=90, fontsize=fs)
        ax.set_yticklabels(classes, fontsize=fs)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def metrics_to_json(metrics: dict, extra: dict | None = None) -> dict:
    """JSON-serialisable view of ``evaluate_model`` output (confusion as nested list)."""
    out = {}
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            out[key] = val.item()
        else:
            out[key] = val
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_eval_loader(data_dir: str, split: str, preprocessing: dict, batch_size: int = 256,
                      num_workers: int = 0, max_per_class: int | None = None):
    """DataLoader for one split, rebuilt from the checkpoint's ``preprocessing`` dict (no augment)."""
    from torch.utils.data import DataLoader

    from data_loader import QuickDrawDataset, collate_strokes

    mode = preprocessing.get("mode", "image")
    ds = QuickDrawDataset(data_dir, split, mode=mode,
                          image_size=int(preprocessing.get("image_size", 64)),
                          max_len=int(preprocessing.get("max_len", 250)),
                          augment=False, max_per_class=max_per_class, seed=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate_strokes if mode == "strokes" else None,
                      pin_memory=False)


def _check_classes(data_dir: str, classes: Sequence[str]) -> None:
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_classes = json.load(f).get("classes")
    if meta_classes is not None and list(meta_classes) != list(classes):
        raise ValueError(
            "The classes stored in the checkpoint do not match data meta.json "
            f"({len(classes)} vs {len(meta_classes)} classes or different order). "
            "Evaluate with the dataset the model was trained on.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a Quick, Draw! checkpoint.")
    p.add_argument("--checkpoint", required=True, help="path to best.pt / last.pt")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--run-name", default=None, help="default: parent folder of the checkpoint")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    from model import load_model_from_checkpoint

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = _resolve_device(args.device)
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device=str(device))
    classes = list(ckpt["classes"])
    num_classes = int(ckpt.get("num_classes", len(classes)))
    preprocessing = dict(ckpt.get("preprocessing") or {})
    mode = preprocessing.get("mode") or ("strokes" if ckpt.get("model_name") == "bilstm" else "image")
    preprocessing["mode"] = mode
    _check_classes(args.data_dir, classes)

    loader = build_eval_loader(args.data_dir, args.split, preprocessing, args.batch_size,
                               args.num_workers, args.max_per_class)
    metrics = evaluate_model(model, loader, num_classes, device, mode, classes=classes)

    run = args.run_name or os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    out_dir = os.path.join(args.results_dir, run)
    os.makedirs(out_dir, exist_ok=True)
    extra = {
        "run": run,
        "split": args.split,
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_name": ckpt.get("model_name"),
        "epoch": ckpt.get("epoch"),
        "classes": classes,
        "preprocessing": preprocessing,
    }
    paths = {
        "metrics": os.path.join(out_dir, f"metrics_{args.split}.json"),
        "per_class": os.path.join(out_dir, f"per_class_{args.split}.csv"),
        "confusion": os.path.join(out_dir, f"confusion_{args.split}.png"),
    }
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_to_json(metrics, extra), f, indent=2, ensure_ascii=False)
    write_per_class_csv(paths["per_class"], metrics["per_class"])
    plot_confusion(paths["confusion"], metrics["confusion"], classes,
                   title=f"{run} - {args.split} (top-1 {metrics['top1']:.3f})")

    if not args.quiet:
        print(f"[{run}] split={args.split} n={metrics['n']} loss={metrics['loss']:.4f} "
              f"top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} "
              f"macroF1={metrics['macro_f1']:.4f}")
        worst = sorted((r for r in metrics["per_class"] if r["support"] > 0), key=lambda r: r["accuracy"])[:5]
        print("Worst classes: " + ", ".join(f"{r['class']} ({r['accuracy']:.2f})" for r in worst))
        print("Top-10 most confused pairs (true -> predicted):")
        for p in metrics["most_confused"]:
            print(f"  {p['true']:>15s} -> {p['predicted']:<15s} {p['count']:5d}  ({p['rate']:.1%})")
        for name, pth in paths.items():
            print(f"  wrote {pth}")
    metrics["paths"] = paths
    metrics.update(extra)
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
