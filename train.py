"""Training script for the Quick, Draw! classifiers (CNN on images, BiLSTM on stroke sequences).

Follows docs/CONTRACT.md section 5. Example:

    python train.py --model cnn --data-dir data/processed --epochs 30 --amp --resume auto
    python train.py --model bilstm --data-dir data/processed --batch-size 256 --lr 1e-3

Design choices (documented for the report):
- Optimiser: AdamW (decoupled weight decay).
- LR schedule: linear warm-up (``--warmup-epochs``, default 1) followed by cosine decay down to
  ``lr * --min-lr-ratio``. It is a per-step ``LambdaLR`` so it is exactly resumable, and if a run is
  resumed with a larger ``--epochs`` the cosine simply stretches to the new horizon.
- Loss: cross-entropy with optional label smoothing (``--label-smoothing``, default 0).
- Gradient clipping by global norm (``--grad-clip``, default 1.0, 0 disables) for both models
  (essential for the LSTM, harmless for the CNN).
- Early stopping on validation top-1 (``--patience`` epochs without improvement > ``--min-delta``).
- Mixed precision (``--amp``) only on CUDA, via ``torch.amp``.
- Train top-1/top-3 are running metrics over the epoch (with augmentation and dropout active);
  validation metrics are computed in eval mode without augmentation.
- Checkpoints: ``{out_dir}/{run}/last.pt`` every epoch and ``best.pt`` on each new best val top-1,
  written with ``model.save_checkpoint`` (atomic). ``extra`` stores scaler, early-stopping and RNG
  state so ``--resume`` continues the run where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - very old torch
            pass
    else:
        torch.backends.cudnn.benchmark = True


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict | None) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # RNG restore is best effort (e.g. CPU <-> GPU machine change)
        print(f"[train] warning: could not fully restore RNG state ({exc})")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def load_class_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"] if isinstance(data, dict) else data
    if not classes or not all(isinstance(c, str) for c in classes):
        raise ValueError(f"{path}: 'classes' must be a non-empty list of strings")
    return list(classes)


def check_dataset_classes(data_dir: str, classes: list[str]) -> None:
    """The processed dataset's meta.json classes must equal classes.json (same order)."""
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{meta_path} not found: run preprocess.py first")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_classes = list(meta.get("classes", []))
    if meta_classes != classes:
        missing = sorted(set(classes) - set(meta_classes))
        extra = sorted(set(meta_classes) - set(classes))
        raise ValueError(
            "Class mismatch between dataset meta.json and classes.json "
            f"(dataset {len(meta_classes)} vs classes.json {len(classes)}; "
            f"missing in dataset: {missing[:10]}, extra in dataset: {extra[:10]}"
            + ("; same set but different order" if not missing and not extra else "")
            + "). Re-run preprocess.py with the current classes.json."
        )


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3)) -> list[int]:
    """Number of samples whose target is within the top-k predictions, for each k."""
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    hit = pred.eq(targets.view(-1, 1))
    return [int(hit[:, : min(k, maxk)].any(dim=1).sum().item()) for k in ks]


class EarlyStopping:
    """Stop when the monitored metric (higher is better) has not improved for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -math.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, value: float, epoch: int) -> bool:
        """Update with a new value; returns True if this is a new best."""
        if value > self.best + self.min_delta:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            improved = False
        self.should_stop = self.patience > 0 and self.counter >= self.patience
        return improved

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state.get("best", -math.inf))
        self.best_epoch = int(state.get("best_epoch", 0))
        self.counter = int(state.get("counter", 0))
        # patience/min_delta come from the current CLI so they can be changed on resume.
        self.should_stop = self.patience > 0 and self.counter >= self.patience


def make_warmup_cosine(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - torch < 2.3


def unpack_batch(batch, mode: str, device: torch.device):
    """Return (inputs, lengths_or_None, labels) on device."""
    if mode == "strokes":
        x, lengths, y = batch
        return (
            x.to(device, non_blocking=True),
            lengths.to(device, non_blocking=True),
            y.to(device, non_blocking=True).long(),
        )
    x, y = batch
    return x.to(device, non_blocking=True), None, y.to(device, non_blocking=True).long()


def forward(model, x, lengths, mode: str):
    return model(x, lengths) if mode == "strokes" else model(x)


def _progress(iterable, desc: str, enabled: bool, total=None):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)


def run_epoch(model, loader, mode, device, criterion, optimizer=None, scheduler=None, scaler=None,
              use_amp=False, grad_clip=0.0, desc="", progress=True) -> dict:
    """One pass over `loader`. Trains if `optimizer` is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total, loss_sum, c1, c3 = 0, 0.0, 0, 0
    try:
        n_batches = len(loader)
    except TypeError:
        n_batches = None
    bar = _progress(loader, desc, progress, total=n_batches)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in bar:
            x, lengths, y = unpack_batch(batch, mode, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = forward(model, x, lengths, mode)
                loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            bs = y.shape[0]
            total += bs
            loss_sum += float(loss.detach().item()) * bs
            k1, k3 = topk_correct(logits.detach(), y, ks=(1, 3))
            c1 += k1
            c3 += k3
            if progress and hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss_sum / total:.3f}", top1=f"{c1 / total:.3f}")
    if total == 0:
        return {"loss": float("nan"), "top1": 0.0, "top3": 0.0, "n": 0}
    return {"loss": loss_sum / total, "top1": c1 / total, "top3": c3 / total, "n": total}


def plot_curves(history: dict, path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(epochs, history.get("train_loss", []), "o-", label="train")
    if any(v is not None for v in history.get("val_loss", [])):
        ax.plot(epochs, history.get("val_loss", []), "o-", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    for key, style in (("train_top1", "o-"), ("val_top1", "o-"), ("train_top3", "s--"), ("val_top3", "s--")):
        vals = history.get(key, [])
        if vals and any(v is not None for v in vals):
            ax.plot(epochs, vals, style, label=key.replace("_", " "))
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, path)


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def evaluate_test(model, loader, num_classes, device, mode, criterion, use_amp, progress) -> dict:
    """Use evaluate.evaluate_model if available, otherwise a simple loss/top-1/top-3."""
    try:
        from evaluate import evaluate_model  # lazy: evaluate.py is owned by another module
    except Exception:
        evaluate_model = None
    if evaluate_model is not None:
        try:
            with torch.no_grad():
                res = evaluate_model(model, loader, num_classes, device, mode)
            if isinstance(res, dict):
                # keep the summary small and JSON-friendly
                out = {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))}
                out["source"] = "evaluate.evaluate_model"
                return out
        except Exception as exc:
            print(f"[train] evaluate.evaluate_model failed ({exc}); using built-in evaluation")
    res = run_epoch(model, loader, mode, device, criterion, use_amp=use_amp, desc="test", progress=progress)
    res["source"] = "train.run_epoch"
    return res


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Quick, Draw! classifier (CNN or BiLSTM).")
    p.add_argument("--model", choices=["cnn", "bilstm"], required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--classes", default="classes.json")
    p.add_argument("--epochs", type=int, default=30, help="total number of epochs (also when resuming)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0, help="linear warm-up length (epochs)")
    p.add_argument("--min-lr-ratio", type=float, default=0.01, help="final lr = lr * ratio (cosine)")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm, 0 disables")
    p.add_argument("--dropout", type=float, default=None, help="override model dropout")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (0 disables)")
    p.add_argument("--min-delta", type=float, default=0.0, help="min val top-1 improvement")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=250)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic algorithms")
    p.add_argument("--run-name", default=None, help="default: model name")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--resume", default=None, help="'auto' (= {out_dir}/{run}/last.pt if it exists) or path")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-test", action="store_true", help="skip final test evaluation of best.pt")
    p.add_argument("--no-progress", action="store_true", help="disable tqdm bars")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    run = args.run_name or args.model
    mode = "image" if args.model == "cnn" else "strokes"
    progress = not args.no_progress

    # Imports of sibling modules are done here so `--help` works without them.
    from data_loader import make_dataloaders
    from model import build_model, load_checkpoint, save_checkpoint

    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("[train] --amp ignored: CUDA not available")

    classes = load_class_list(args.classes)
    check_dataset_classes(args.data_dir, classes)
    num_classes = len(classes)

    ckpt_dir = os.path.join(args.out_dir, run)
    res_dir = os.path.join(args.results_dir, run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    history_path = os.path.join(res_dir, "history.json")
    curves_path = os.path.join(res_dir, "curves.png")

    loaders = make_dataloaders(
        args.data_dir, mode, batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, max_len=args.max_len, augment_train=not args.no_augment,
        max_per_class=args.max_per_class, seed=args.seed,
    )
    if "train" not in loaders:
        raise RuntimeError(f"No train split found in {args.data_dir}")
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    if val_loader is None:
        print("[train] warning: no val split; early stopping/best.pt use train top-1")

    model_kwargs = {} if args.dropout is None else {"dropout": args.dropout}
    model = build_model(args.model, num_classes, **model_kwargs).to(device)
    preprocessing = {"mode": mode, "image_size": args.image_size, "max_len": args.max_len, "margin": 0.1}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = make_warmup_cosine(
        optimizer, total_steps=args.epochs * steps_per_epoch,
        warmup_steps=int(round(args.warmup_epochs * steps_per_epoch)), min_lr_ratio=args.min_lr_ratio,
    )
    scaler = make_grad_scaler(use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    stopper = EarlyStopping(args.patience, args.min_delta)
    history: dict[str, list] = {k: [] for k in (
        "epoch", "train_loss", "train_top1", "train_top3", "val_loss", "val_top1", "val_top3", "lr", "time_s")}
    start_epoch = 1
    best_val_top3 = None

    # ---------------------------------------------------------------- resume
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            resume_path = last_path if os.path.isfile(last_path) else None
            if resume_path is None:
                print(f"[train] --resume auto: no {last_path}, starting from scratch")
        else:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(f"--resume {args.resume}: file not found")
            resume_path = args.resume
    if resume_path:
        ckpt = load_checkpoint(resume_path, map_location="cpu")
        if ckpt.get("model_name") != args.model:
            raise ValueError(f"Checkpoint model {ckpt.get('model_name')!r} != --model {args.model!r}")
        if list(ckpt.get("classes", [])) != classes:
            raise ValueError("Checkpoint classes differ from classes.json; cannot resume")
        if ckpt.get("model_kwargs") and ckpt["model_kwargs"] != model_kwargs:
            model_kwargs = dict(ckpt["model_kwargs"])
            model = build_model(args.model, num_classes, **model_kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = make_warmup_cosine(
                optimizer, total_steps=args.epochs * steps_per_epoch,
                warmup_steps=int(round(args.warmup_epochs * steps_per_epoch)), min_lr_ratio=args.min_lr_ratio,
            )
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        extra = ckpt.get("extra") or {}
        if extra.get("scaler_state") and use_amp:
            scaler.load_state_dict(extra["scaler_state"])
        if extra.get("early_stopping"):
            stopper.load_state_dict(extra["early_stopping"])
        best_val_top3 = extra.get("best_val_top3")
        if ckpt.get("history"):
            history = {k: list(v) for k, v in ckpt["history"].items()}
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        set_rng_state(extra.get("rng"))
        print(f"[train] resumed from {resume_path} (completed epochs: {start_epoch - 1}, "
              f"best val top-1: {stopper.best:.4f})")

    config = dict(vars(args))
    config.update({
        "run_name": run, "mode": mode, "num_classes": num_classes, "device": str(device),
        "amp_enabled": use_amp, "model_kwargs": model_kwargs, "preprocessing": preprocessing,
        "schedule": "AdamW + linear warmup + cosine decay (per step)",
        "steps_per_epoch": steps_per_epoch, "torch_version": torch.__version__,
        "n_train": len(train_loader.dataset) if hasattr(train_loader, "dataset") else None,
        "n_val": len(val_loader.dataset) if val_loader is not None and hasattr(val_loader, "dataset") else None,
        "resumed_from": resume_path, "argv": list(argv) if argv is not None else sys.argv[1:],
    })
    write_json(os.path.join(res_dir, "config.json"), config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] run={run} model={args.model} classes={num_classes} params={n_params:,} "
          f"device={device} amp={use_amp} train_batches={steps_per_epoch}")

    def save(path: str, epoch: int) -> None:
        save_checkpoint(
            path, model, args.model, model_kwargs, classes, preprocessing,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            best_metric=stopper.best if stopper.best > -math.inf else None, history=history,
            extra={
                "scaler_state": scaler.state_dict() if use_amp else None,
                "early_stopping": stopper.state_dict(),
                "best_val_top3": best_val_top3,
                "rng": get_rng_state(),
                "args": {k: v for k, v in vars(args).items()},
                "monitor": "val_top1" if val_loader is not None else "train_top1",
            },
        )

    # ---------------------------------------------------------------- loop
    epochs_run = 0
    if stopper.should_stop and start_epoch <= args.epochs:
        print("[train] note: resumed run had already triggered early stopping; "
              "it will stop again after the next epoch unless val top-1 improves")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, mode, device, criterion, optimizer, scheduler, scaler,
                       use_amp, args.grad_clip, desc=f"epoch {epoch}/{args.epochs} train", progress=progress)
        va = None
        if val_loader is not None:
            va = run_epoch(model, val_loader, mode, device, criterion, use_amp=use_amp,
                           desc=f"epoch {epoch}/{args.epochs} val", progress=progress)
        dt = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["train_top1"].append(tr["top1"])
        history["train_top3"].append(tr["top3"])
        history["val_loss"].append(va["loss"] if va else None)
        history["val_top1"].append(va["top1"] if va else None)
        history["val_top3"].append(va["top3"] if va else None)
        history["lr"].append(lr_now)
        history["time_s"].append(dt)

        monitor = va["top1"] if va else tr["top1"]
        improved = stopper.step(monitor, epoch)
        if improved:
            best_val_top3 = va["top3"] if va else tr["top3"]
        save(last_path, epoch)
        if improved:
            save(best_path, epoch)
        epochs_run += 1

        msg = (f"[train] epoch {epoch}/{args.epochs} | train loss {tr['loss']:.4f} "
               f"top1 {tr['top1']:.4f} top3 {tr['top3']:.4f}")
        if va:
            msg += f" | val loss {va['loss']:.4f} top1 {va['top1']:.4f} top3 {va['top3']:.4f}"
        msg += f" | lr {lr_now:.2e} | {dt:.1f}s" + (" | * best" if improved else "")
        print(msg, flush=True)

        write_json(history_path, history)
        try:
            plot_curves(history, curves_path, title=f"{run} ({args.model}, {num_classes} classes)")
        except Exception as exc:  # plotting must never kill a long training run
            print(f"[train] warning: could not plot curves ({exc})")

        if stopper.should_stop:
            print(f"[train] early stopping: no val top-1 improvement for {stopper.counter} epochs "
                  f"(best {stopper.best:.4f} at epoch {stopper.best_epoch})")
            break

    # Make sure history/curves exist even if no epoch ran in this invocation (resume of a finished run).
    if not os.path.isfile(history_path):
        write_json(history_path, history)
    if not os.path.isfile(curves_path) and history["epoch"]:
        try:
            plot_curves(history, curves_path, title=run)
        except Exception as exc:
            print(f"[train] warning: could not plot curves ({exc})")

    # ---------------------------------------------------------------- test
    test_metrics = None
    if not args.no_test and "test" in loaders and os.path.isfile(best_path):
        best_ckpt = load_checkpoint(best_path, map_location="cpu")
        model.load_state_dict(best_ckpt["state_dict"])
        model.eval()
        test_metrics = evaluate_test(model, loaders["test"], num_classes, device, mode, criterion,
                                     use_amp, progress)
        test_metrics["checkpoint"] = best_path
        test_metrics["best_epoch"] = int(best_ckpt.get("epoch", 0))
        write_json(os.path.join(res_dir, "train_test_metrics.json"), test_metrics)
        t1 = test_metrics.get("top1", test_metrics.get("top1_acc"))
        t3 = test_metrics.get("top3", test_metrics.get("top3_acc"))
        print(f"[train] test (best.pt, epoch {test_metrics['best_epoch']}): top1 {t1} top3 {t3}")

    summary = {
        "run_name": run,
        "model": args.model,
        "num_classes": num_classes,
        "best_val_top1": stopper.best if stopper.best > -math.inf else None,
        "best_val_top3": best_val_top3,
        "best_epoch": stopper.best_epoch,
        "start_epoch": start_epoch,
        "last_epoch": history["epoch"][-1] if history["epoch"] else 0,
        "epochs_run": epochs_run,
        "stopped_early": bool(stopper.should_stop),
        "last_checkpoint": last_path,
        "best_checkpoint": best_path,
        "history_path": history_path,
        "curves_path": curves_path,
        "config_path": os.path.join(res_dir, "config.json"),
        "test_metrics": test_metrics,
        "device": str(device),
    }
    write_json(os.path.join(res_dir, "train_summary.json"), summary)
    return summary


if __name__ == "__main__":
    main()
