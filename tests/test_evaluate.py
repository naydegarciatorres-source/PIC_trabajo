"""Tests for evaluate.py (fast, CPU, untrained models; accuracy is never asserted)."""
import csv
import json
import os

import numpy as np
import pytest
import torch

import evaluate


def test_topk_accuracy_exact_values():
    logits = torch.tensor([
        [0.1, 0.5, 0.2, 0.9],   # rank: 3,1,2,0
        [2.0, 1.0, 0.0, -1.0],  # rank: 0,1,2,3
        [0.1, 0.0, 1.0, 0.5],   # rank: 2,3,0,1
        [0.3, 0.2, 0.1, 0.0],   # rank: 0,1,2,3
    ])
    targets = torch.tensor([3, 1, 0, 3])
    acc = evaluate.topk_accuracy(logits, targets, ks=(1, 2, 3, 4))
    assert acc[1] == pytest.approx(1 / 4)    # only sample 0
    assert acc[2] == pytest.approx(2 / 4)    # + sample 1
    assert acc[3] == pytest.approx(3 / 4)    # + sample 2
    assert acc[4] == pytest.approx(1.0)
    # k larger than num_classes is clamped
    assert evaluate.topk_accuracy(logits, targets, ks=(10,))[10] == pytest.approx(1.0)


def test_topk_accuracy_perfect_and_empty():
    logits = torch.eye(5) * 3
    targets = torch.arange(5)
    acc = evaluate.topk_accuracy(logits, targets)
    assert acc == {1: 1.0, 3: 1.0}
    assert evaluate.topk_accuracy(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long)) == {1: 0.0, 3: 0.0}


def test_per_class_and_confused_pairs():
    cm = np.array([[3, 1, 0], [2, 2, 0], [0, 0, 0]])
    rows = evaluate.per_class_from_confusion(cm, ["a", "b", "c"])
    assert rows[0]["accuracy"] == pytest.approx(0.75)
    assert rows[0]["precision"] == pytest.approx(3 / 5)
    assert rows[1]["recall"] == pytest.approx(0.5)
    assert rows[1]["precision"] == pytest.approx(2 / 3)
    assert rows[2]["support"] == 0 and rows[2]["precision"] == 0.0
    pairs = evaluate.most_confused_pairs(cm, ["a", "b", "c"])
    assert [(p["true"], p["predicted"], p["count"]) for p in pairs] == [("b", "a", 2), ("a", "b", 1)]
    assert pairs[0]["rate"] == pytest.approx(0.5)


@pytest.mark.parametrize("mode,name", [("image", "cnn"), ("strokes", "bilstm")])
def test_evaluate_model_shapes(synthetic_data_dir, classes, mode, name):
    from model import build_model

    torch.manual_seed(0)
    nc = len(classes)
    loader = evaluate.build_eval_loader(
        synthetic_data_dir, "test", {"mode": mode, "image_size": 32, "max_len": 60}, batch_size=16)
    model = build_model(name, nc)
    m = evaluate.evaluate_model(model, loader, nc, "cpu", mode, classes=classes)
    n = len(loader.dataset)
    assert m["n"] == n > 0
    assert m["confusion"].shape == (nc, nc)
    assert m["confusion"].sum() == n
    assert len(m["per_class"]) == nc
    assert m["per_class_accuracy"].shape == (nc,)
    assert sum(r["support"] for r in m["per_class"]) == n
    assert 0.0 <= m["top1"] <= m["top3"] <= 1.0
    assert np.isfinite(m["loss"]) and m["loss"] > 0
    # top-1 must agree with the confusion-matrix diagonal
    assert m["top1"] == pytest.approx(np.trace(m["confusion"]) / n)
    assert len(m["most_confused"]) <= 10


def test_cli_writes_outputs(synthetic_data_dir, classes, tmp_path):
    from model import build_model, save_checkpoint

    nc = len(classes)
    model = build_model("cnn", nc)
    ckpt = tmp_path / "checkpoints" / "myrun" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    save_checkpoint(str(ckpt), model, "cnn", {}, classes,
                    {"mode": "image", "image_size": 32, "max_len": 250, "margin": 0.1})
    results = tmp_path / "results"
    m = evaluate.main(["--checkpoint", str(ckpt), "--data-dir", synthetic_data_dir, "--split", "val",
                       "--results-dir", str(results), "--batch-size", "32", "--device", "cpu", "--quiet"])
    run_dir = results / "myrun"
    for fname in ("metrics_val.json", "per_class_val.csv", "confusion_val.png"):
        assert (run_dir / fname).is_file() and (run_dir / fname).stat().st_size > 0
    data = json.loads((run_dir / "metrics_val.json").read_text(encoding="utf-8"))
    assert data["run"] == "myrun" and data["split"] == "val"
    assert np.array(data["confusion"]).shape == (nc, nc)
    assert data["top1"] == pytest.approx(m["top1"])
    with open(run_dir / "per_class_val.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["class"] for r in rows] == list(classes)
    assert isinstance(m, dict) and "top3" in m
