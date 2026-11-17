"""Tests for train.py: tiny CPU runs on the synthetic dataset (fixture from conftest.py)."""
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

import train  # noqa: E402


def _argv(model, data_dir, project_root, tmp_path, epochs, extra=()):
    return [
        "--model", model,
        "--data-dir", data_dir,
        "--classes", os.path.join(project_root, "classes.json"),
        "--epochs", str(epochs),
        "--batch-size", "16",
        "--num-workers", "0",
        "--image-size", "32",
        "--max-len", "60",
        "--device", "cpu",
        "--seed", "0",
        "--out-dir", str(tmp_path / "ckpt"),
        "--results-dir", str(tmp_path / "results"),
        "--no-progress",
        *extra,
    ]


@pytest.mark.parametrize("model_name", ["cnn", "bilstm"])
def test_train_resume(model_name, synthetic_data_dir, project_root, tmp_path):
    summary = train.main(_argv(model_name, synthetic_data_dir, project_root, tmp_path, 2,
                               ["--patience", "0"]))
    assert summary["epochs_run"] == 2
    assert summary["last_epoch"] == 2
    for key in ("last_checkpoint", "best_checkpoint", "history_path", "curves_path", "config_path"):
        assert os.path.isfile(summary[key]), key
    assert 0.0 <= summary["best_val_top1"] <= summary["best_val_top3"] <= 1.0

    with open(summary["history_path"], encoding="utf-8") as f:
        hist = json.load(f)
    assert hist["epoch"] == [1, 2]
    for k in ("train_loss", "val_loss", "train_top1", "val_top1", "train_top3", "val_top3", "lr"):
        assert len(hist[k]) == 2 and all(v is not None and math.isfinite(v) for v in hist[k]), k

    from model import load_checkpoint, load_model_from_checkpoint

    last = load_checkpoint(summary["last_checkpoint"])
    assert last["epoch"] == 2 and last["model_name"] == model_name
    assert last["preprocessing"]["mode"] == ("image" if model_name == "cnn" else "strokes")
    assert last["optimizer_state"] is not None and last["scheduler_state"] is not None
    m, _ = load_model_from_checkpoint(summary["best_checkpoint"])
    assert not m.training

    # Resume from last.pt (explicit path) up to epoch 3: must continue at epoch 3, not restart.
    summary2 = train.main(_argv(model_name, synthetic_data_dir, project_root, tmp_path, 3,
                                ["--patience", "0", "--resume", summary["last_checkpoint"]]))
    assert summary2["start_epoch"] == 3
    assert summary2["epochs_run"] == 1
    assert summary2["last_epoch"] == 3
    with open(summary2["history_path"], encoding="utf-8") as f:
        hist2 = json.load(f)
    assert hist2["epoch"] == [1, 2, 3]
    assert hist2["train_loss"][:2] == hist["train_loss"]
    assert load_checkpoint(summary2["last_checkpoint"])["epoch"] == 3

    # --resume auto on a finished run: nothing to do, no crash.
    summary3 = train.main(_argv(model_name, synthetic_data_dir, project_root, tmp_path, 3,
                                ["--patience", "0", "--resume", "auto", "--no-test"]))
    assert summary3["epochs_run"] == 0 and summary3["last_epoch"] == 3


def test_early_stopping_helper():
    es = train.EarlyStopping(patience=2)
    assert es.step(0.5, 1) and not es.should_stop
    assert not es.step(0.5, 2) and not es.should_stop
    assert not es.step(0.4, 3) and es.should_stop
    assert es.best == 0.5 and es.best_epoch == 1
    es2 = train.EarlyStopping(patience=2)
    es2.load_state_dict(es.state_dict())
    assert es2.should_stop and es2.best == 0.5
    assert train.EarlyStopping(patience=0).step(0.1, 1)  # patience 0 never stops


def test_early_stopping_triggers(synthetic_data_dir, project_root, tmp_path):
    # min-delta 1.0 -> no improvement is possible after epoch 1 -> stop at epoch 2 (deterministic).
    summary = train.main(_argv("cnn", synthetic_data_dir, project_root, tmp_path, 5,
                               ["--patience", "1", "--min-delta", "1.0", "--no-augment", "--no-test",
                                "--run-name", "es"]))
    assert summary["stopped_early"]
    assert summary["epochs_run"] == 2
    assert summary["best_epoch"] == 1


def test_class_mismatch_raises(synthetic_data_dir, tmp_path):
    bad = tmp_path / "classes.json"
    bad.write_text(json.dumps({"classes": ["foo", "bar"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        train.check_dataset_classes(synthetic_data_dir, train.load_class_list(str(bad)))


def test_topk_correct():
    logits = torch.tensor([[0.1, 0.9, 0.0, 0.0], [0.5, 0.2, 0.3, 0.0]])
    targets = torch.tensor([1, 2])
    assert train.topk_correct(logits, targets, ks=(1, 3)) == [1, 2]
