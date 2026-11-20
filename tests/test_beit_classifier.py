"""Tests de beit_classifier.py.

Los tests normales usan un BEiT diminuto con pesos aleatorios (sin descargar nada) para
comprobar la interfaz y la preparación de la entrada. El test con el modelo real solo se
ejecuta si BEIT_SKETCH_MODEL apunta a una carpeta con los pesos descargados.
"""
import os

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
import beit_classifier as B  # noqa: E402

N_CLASSES = 345


@pytest.fixture(scope="module")
def tiny_clf(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_beit")
    cfg = transformers.BeitConfig(image_size=224, patch_size=32, hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=64, num_labels=N_CLASSES,
                                  id2label={i: f"c{i}" for i in range(N_CLASSES)},
                                  label2id={f"c{i}": i for i in range(N_CLASSES)})
    transformers.BeitForImageClassification(cfg).save_pretrained(d)
    return B.BeitClassifier(str(d), device="cpu")


def circle(cx=320, cy=240, r=100, n=50):
    t = np.linspace(0, 2 * np.pi, n)
    return [list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))]


def test_render_strokes_shape_and_colors():
    img = B.render_strokes([np.array([[0, 0], [255, 255]], np.float32)])
    assert img.shape == (224, 224) and img.dtype == np.uint8
    assert img[0, 0] == 255 and img[8, 8] == 0 and img[112, 112] == 0  # margen 8, línea negra


def test_predict_strokes_returns_top_k_sorted(tiny_clf):
    out = tiny_clf.predict_strokes(circle(), top_k=3)
    assert len(out) == 3
    assert all(isinstance(c, str) and 0.0 <= p <= 1.0 for c, p in out)
    assert out[0][1] >= out[1][1] >= out[2][1]
    p = tiny_clf.predict_proba_strokes(circle())
    assert p.shape == (N_CLASSES,) and abs(p.sum() - 1) < 1e-4


def test_blank_inputs_return_empty(tiny_clf):
    assert tiny_clf.predict_strokes([]) == []
    assert tiny_clf.predict_strokes(None) == []
    assert tiny_clf.predict(np.full((480, 640), 255, np.uint8)) == []
    assert tiny_clf.predict(None) == []


def test_normalized_strokes_match_pixel_strokes(tiny_clf):
    px = circle()
    norm = [[(x / 640, y / 480) for x, y in s] for s in px]
    a = tiny_clf.strokes_to_image(px)
    b = tiny_clf.strokes_to_image(norm, image_size=(640, 480))
    assert np.array_equal(a, b)


def test_position_and_size_invariance(tiny_clf):
    a = tiny_clf.strokes_to_image(circle(200, 200, 50))
    b = tiny_clf.strokes_to_image(circle(400, 300, 150))
    assert (a != b).mean() < 0.02


def test_canvas_gray_and_bgr(tiny_clf):
    import cv2
    img = np.full((480, 640, 3), 255, np.uint8)
    pts = np.array(circle()[0], np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert tiny_clf.predict(img) == tiny_clf.predict(gray)
    assert len(tiny_clf.predict(gray, top_k=5)) == 5


@pytest.mark.skipif(not os.path.isdir(os.environ.get("BEIT_SKETCH_MODEL", "")),
                    reason="BEIT_SKETCH_MODEL no apunta a los pesos descargados")
def test_real_model_recognizes_circle():
    clf = B.BeitClassifier(device="cpu")
    assert len(clf.classes) == 345
    labels = [c for c, _ in clf.predict_strokes(circle(), top_k=3)]
    assert "circle" in labels
