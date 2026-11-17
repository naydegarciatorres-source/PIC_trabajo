"""Fast unit tests for model.py (no dependency on data_loader / preprocess)."""
import os

import pytest
import torch

from model import (
    MODEL_REGISTRY,
    QuickDrawCNN,
    StrokeBiLSTM,
    build_model,
    count_parameters,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)

SMALL_LSTM = dict(hidden_size=32, num_layers=2)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ----------------------------------------------------------------------------- CNN
@pytest.mark.parametrize("num_classes", [45, 345])
@pytest.mark.parametrize("size", [28, 64, 96])
def test_cnn_output_shape(num_classes, size):
    model = QuickDrawCNN(num_classes).eval()
    x = torch.rand(2, 1, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes)
    assert torch.isfinite(out).all()


def test_cnn_param_count_reasonable():
    n = count_parameters(QuickDrawCNN(345))
    assert 500_000 < n < 3_000_000


def test_cnn_backward_finite_grads():
    model = QuickDrawCNN(45).train()
    x = torch.rand(4, 1, 64, 64)
    y = torch.randint(0, 45, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# ----------------------------------------------------------------------------- BiLSTM
@pytest.mark.parametrize("num_classes", [45, 345])
def test_bilstm_output_shape_variable_lengths(num_classes):
    model = StrokeBiLSTM(num_classes, **SMALL_LSTM).eval()
    x = torch.randn(4, 20, 3)
    lengths = torch.tensor([20, 7, 1, 13])
    with torch.no_grad():
        out = model(x, lengths)
    assert out.shape == (4, num_classes)
    assert torch.isfinite(out).all()


def test_bilstm_default_contract_kwargs_and_none_lengths():
    model = StrokeBiLSTM(45).eval()  # contract defaults (3 layers, 128 hidden)
    x = torch.randn(3, 15, 3)
    with torch.no_grad():
        a = model(x)
        b = model(x, torch.full((3,), 15, dtype=torch.long))
    assert a.shape == (3, 45)
    assert torch.allclose(a, b, atol=1e-6)


def test_bilstm_single_length_one_sequence():
    model = StrokeBiLSTM(10, **SMALL_LSTM).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 3), torch.tensor([1]))
    assert out.shape == (1, 10)


@pytest.mark.parametrize("use_batchnorm", [False, True])
def test_bilstm_padding_invariance(use_batchnorm):
    model = StrokeBiLSTM(45, use_batchnorm=use_batchnorm, **SMALL_LSTM).eval()
    seqs = [torch.randn(12, 3), torch.randn(5, 3), torch.randn(1, 3)]
    with torch.no_grad():
        ref = torch.cat([model(s.unsqueeze(0), torch.tensor([len(s)])) for s in seqs])
        t_max = 30
        padded = torch.randn(len(seqs), t_max, 3) * 100.0  # garbage padding
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s
        lengths = torch.tensor([len(s) for s in seqs])
        out = model(padded, lengths)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_bilstm_backward_finite_grads():
    model = StrokeBiLSTM(45, **SMALL_LSTM).train()
    x = torch.randn(4, 25, 3)
    lengths = torch.tensor([25, 3, 1, 17])
    y = torch.randint(0, 45, (4,))
    loss = torch.nn.functional.cross_entropy(model(x, lengths), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ----------------------------------------------------------------------------- registry
def test_build_model_and_registry():
    assert set(MODEL_REGISTRY) >= {"cnn", "bilstm"}
    assert isinstance(build_model("cnn", 45), QuickDrawCNN)
    m = build_model("bilstm", 345, hidden_size=16, num_layers=1)
    assert isinstance(m, StrokeBiLSTM) and m.num_classes == 345


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("transformer", 45)


# ----------------------------------------------------------------------------- checkpoints
@pytest.mark.parametrize(
    "name,kwargs,mode",
    [("cnn", {"dropout": 0.2}, "image"), ("bilstm", dict(SMALL_LSTM, dropout=0.1), "strokes")],
)
def test_checkpoint_roundtrip(tmp_path, name, kwargs, mode):
    classes = [f"class_{i}" for i in range(7)]
    preprocessing = {"mode": mode, "image_size": 32, "max_len": 250, "margin": 0.1}
    model = build_model(name, len(classes), **kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    # one training step so optimizer/BN state is non-trivial
    model.train()
    if mode == "image":
        x = torch.rand(3, 1, 32, 32)
        call = lambda m: m(x)  # noqa: E731
    else:
        x = torch.randn(3, 10, 3)
        lengths = torch.tensor([10, 4, 1])
        call = lambda m: m(x, lengths)  # noqa: E731
    torch.nn.functional.cross_entropy(call(model), torch.tensor([0, 1, 2])).backward()
    opt.step()
    sched.step()

    path = tmp_path / "sub" / "best.pt"
    save_checkpoint(str(path), model, name, kwargs, classes, preprocessing, optimizer=opt,
                    scheduler=sched, epoch=3, best_metric=0.5, history={"loss": [1.0]},
                    extra={"note": "x"})
    assert path.exists()
    assert not [f for f in os.listdir(path.parent) if f.endswith(".tmp")]

    ckpt = load_checkpoint(str(path))
    for key in ["format_version", "model_name", "model_kwargs", "num_classes", "classes",
                "preprocessing", "state_dict", "optimizer_state", "scheduler_state", "epoch",
                "best_metric", "history", "extra"]:
        assert key in ckpt
    assert ckpt["format_version"] == 1

    loaded, ckpt2 = load_model_from_checkpoint(str(path), device="cpu")
    assert not loaded.training
    assert ckpt2["classes"] == classes
    assert ckpt2["preprocessing"] == preprocessing
    assert ckpt2["num_classes"] == len(classes)
    assert ckpt2["epoch"] == 3 and ckpt2["best_metric"] == 0.5

    model.eval()
    with torch.no_grad():
        assert torch.equal(call(model), call(loaded))

    opt2 = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
    opt2.load_state_dict(ckpt2["optimizer_state"])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    sched2.load_state_dict(ckpt2["scheduler_state"])
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
    assert sched2.last_epoch == sched.last_epoch


def test_checkpoint_overwrite_and_class_mismatch(tmp_path):
    path = str(tmp_path / "last.pt")
    model = QuickDrawCNN(3)
    prep = {"mode": "image", "image_size": 64, "max_len": 250, "margin": 0.1}
    save_checkpoint(path, model, "cnn", {}, ["a", "b", "c"], prep, epoch=1)
    save_checkpoint(path, model, "cnn", {}, ["a", "b", "c"], prep, epoch=2)  # overwrite
    assert load_checkpoint(path)["epoch"] == 2
    with pytest.raises(ValueError):
        save_checkpoint(path, model, "cnn", {}, ["a", "b"], prep)
