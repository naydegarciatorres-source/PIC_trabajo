"""Tests for classifier.py with untrained checkpoints (fast, CPU; accuracy is never asserted)."""
import numpy as np
import pytest
import torch

import classifier
from classifier import QuickDrawClassifier

PP_IMAGE = {"mode": "image", "image_size": 64, "max_len": 250, "margin": 0.1}
PP_STROKES = {"mode": "strokes", "image_size": 64, "max_len": 250, "margin": 0.1}


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, classes):
    from model import build_model, save_checkpoint

    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("ckpts")
    nc = len(classes)
    cnn_path = d / "cnn" / "best.pt"
    lstm_path = d / "bilstm" / "best.pt"
    cnn_path.parent.mkdir()
    lstm_path.parent.mkdir()
    save_checkpoint(str(cnn_path), build_model("cnn", nc), "cnn", {}, classes, PP_IMAGE)
    save_checkpoint(str(lstm_path), build_model("bilstm", nc), "bilstm", {}, classes, PP_STROKES)
    return {"cnn": str(cnn_path), "bilstm": str(lstm_path)}


@pytest.fixture(scope="module")
def cnn(ckpts):
    return QuickDrawClassifier(ckpts["cnn"], device="cpu")


@pytest.fixture(scope="module")
def lstm(ckpts):
    return QuickDrawClassifier(ckpts["bilstm"], device="cpu")


def _check(res, clf, top_k=3):
    assert isinstance(res, list) and len(res) == min(top_k, len(clf.classes))
    for item in res:
        assert isinstance(item, tuple) and len(item) == 2
        name, p = item
        assert name in clf.classes
        assert isinstance(p, float) and 0.0 <= p <= 1.0
    probs = [p for _, p in res]
    assert probs == sorted(probs, reverse=True)
    assert len({n for n, _ in res}) == len(res)


def _white_canvas_black_lines():
    img = np.full((480, 640, 3), 255, np.uint8)
    import cv2
    cv2.circle(img, (320, 240), 100, (0, 0, 0), 8)
    cv2.line(img, (200, 400), (450, 100), (0, 0, 0), 8)
    return img


def _black_canvas_coloured_lines():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(img, (150, 120), (450, 380), (0, 255, 0), 12)
    cv2.line(img, (150, 120), (300, 40), (255, 0, 255), 12)
    return img


STROKES_XY = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)],
              [(120, 150), (150, 150), (180, 150)]]
STROKES_OFFICIAL = [[[100, 200, 200, 100, 100], [100, 100, 200, 200, 100]],
                    [[120, 150, 180], [150, 150, 150]]]


def test_attributes(cnn, lstm, classes):
    assert cnn.mode == "image" and lstm.mode == "strokes"
    assert cnn.classes == list(classes) and lstm.classes == list(classes)


@pytest.mark.parametrize("make", [
    _white_canvas_black_lines,
    _black_canvas_coloured_lines,
    lambda: np.where(_white_canvas_black_lines()[..., 0] > 0, 255, 0).astype(np.uint8),  # grayscale
    lambda: _white_canvas_black_lines()[..., :1],                                         # (H,W,1)
    lambda: np.dstack([_black_canvas_coloured_lines(), np.full((480, 640), 255, np.uint8)]),  # BGRA
    lambda: _white_canvas_black_lines().astype(np.float32) / 255.0,                       # float
    lambda: np.full((480, 640, 3), 255, np.uint8),                                        # blank
    lambda: np.zeros((300, 300), np.uint8),                                               # blank black
])
def test_predict_canvas_variants(cnn, make):
    res = cnn.predict(make(), top_k=3)
    _check(res, cnn)


def test_full_probability_sums_to_one(cnn, lstm):
    p = cnn.predict_proba(_white_canvas_black_lines())
    assert p.shape == (len(cnn.classes),) and p.sum() == pytest.approx(1.0, abs=1e-4)
    p = lstm.predict_proba(strokes=STROKES_XY)
    assert p.shape == (len(lstm.classes),) and p.sum() == pytest.approx(1.0, abs=1e-4)
    allres = cnn.predict(_white_canvas_black_lines(), top_k=len(cnn.classes))
    assert sum(pr for _, pr in allres) == pytest.approx(1.0, abs=1e-4)


def test_top_k_values(cnn):
    assert len(cnn.predict(_white_canvas_black_lines(), top_k=1)) == 1
    assert len(cnn.predict(_white_canvas_black_lines(), top_k=10_000)) == len(cnn.classes)
    with pytest.raises(ValueError):
        cnn.predict(_white_canvas_black_lines(), top_k=0)


@pytest.mark.parametrize("which", ["cnn", "lstm"])
@pytest.mark.parametrize("strokes", [
    STROKES_XY,
    STROKES_OFFICIAL,
    [np.array(s, dtype=np.float32) for s in STROKES_XY],
    [[(50, 60)]],                       # single-point stroke
    [[(50, 60)], [(10, 10), (300, 20)]],
    [],                                 # empty drawing
    [[], []],                           # only empty strokes
])
def test_predict_strokes(cnn, lstm, which, strokes):
    clf = cnn if which == "cnn" else lstm
    _check(clf.predict_strokes(strokes, top_k=3), clf)


def test_stroke_formats_equivalent(lstm, cnn):
    for clf in (lstm, cnn):
        a = clf.predict_proba(strokes=STROKES_XY)
        b = clf.predict_proba(strokes=STROKES_OFFICIAL)
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_predict_on_stroke_model_raises(lstm):
    with pytest.raises(TypeError, match="stroke order"):
        lstm.predict(_white_canvas_black_lines())


def test_label_mapping_matches_checkpoint(ckpts, cnn, classes):
    """Class index i of the model output must map to classes[i]."""
    from model import load_model_from_checkpoint

    model, ckpt = load_model_from_checkpoint(ckpts["cnn"], device="cpu")
    assert ckpt["classes"] == cnn.classes == list(classes)
    with torch.inference_mode():
        logits = model(cnn.canvas_to_tensor(_white_canvas_black_lines()))
    probs = torch.softmax(logits[0], 0)
    res = cnn.predict(_white_canvas_black_lines(), top_k=len(classes))
    for name, p in res:
        assert p == pytest.approx(float(probs[classes.index(name)]), abs=1e-5)
    assert res[0][0] == classes[int(probs.argmax())]


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        QuickDrawClassifier(str(tmp_path / "nope.pt"))


def test_module_helpers_env(ckpts, monkeypatch, tmp_path):
    classifier.reset_cache()
    monkeypatch.setenv(classifier.ENV_CNN, ckpts["cnn"])
    monkeypatch.setenv(classifier.ENV_BILSTM, ckpts["bilstm"])
    try:
        r1 = classifier.predict(_white_canvas_black_lines())
        r2 = classifier.predict_strokes(STROKES_XY)
        assert len(r1) == 3 and len(r2) == 3
        a = classifier.load_classifier(ckpts["cnn"])
        assert a is classifier.load_classifier(ckpts["cnn"])  # cached singleton
        assert a.mode == "image" and classifier.load_classifier(ckpts["bilstm"]).mode == "strokes"
        monkeypatch.setenv(classifier.ENV_CNN, str(tmp_path / "missing" / "best.pt"))
        with pytest.raises(FileNotFoundError, match="best.pt"):
            classifier.predict(_white_canvas_black_lines())
    finally:
        classifier.reset_cache()


def test_demo_main(ckpts, tmp_path, capsys):
    import cv2

    img_path = tmp_path / "d.png"
    cv2.imwrite(str(img_path), _white_canvas_black_lines())
    res = classifier._main(["--checkpoint", ckpts["cnn"], "--image", str(img_path), "--device", "cpu"])
    assert len(res) == 3
    assert "1." in capsys.readouterr().out
