"""Tests for the shared transforms of preprocess.py (no network)."""
import json
import os

import cv2
import numpy as np
import pytest

import preprocess as P

SQUARE = [[[0, 100, 100, 0, 0], [0, 0, 100, 100, 0]], [[20, 80], [50, 50]]]


# ---------------------------------------------------------------- classes
def test_load_classes(project_root, tmp_path):
    classes = P.load_classes(os.path.join(project_root, "classes.json"))
    with open(os.path.join(project_root, "classes.json"), encoding="utf-8") as f:
        assert classes == json.load(f)["classes"]
    assert len(set(classes)) == len(classes) > 0
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"classes": ["a", "a"]}))
    with pytest.raises(ValueError):
        P.load_classes(str(p))


# ---------------------------------------------------------------- points
def test_ndjson_strokes_to_points_and_eos():
    pts = P.ndjson_strokes_to_points(SQUARE)
    assert pts.shape == (7, 3) and pts.dtype == np.float32
    np.testing.assert_array_equal(pts[:, 2], [0, 0, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(pts[:5, :2], np.array(SQUARE[0]).T)
    assert P.ndjson_strokes_to_points([]).shape == (0, 3)


def test_single_point_stroke_and_empty_dropped():
    pts = P.ndjson_strokes_to_points([[[5], [7]], [[], []], [[1, 2, 3], [4, 5, 6]]])
    assert pts.shape == (4, 3)
    np.testing.assert_array_equal(pts[:, 2], [1, 0, 0, 1])


def test_normalize_strokes_input_formats_agree():
    official = [[[0, 10, 20, 30], [0, 5, 10, 15]], [[1, 2, 3], [4, 5, 6]]]
    as_tuples = [[(0, 0), (10, 5), (20, 10), (30, 15)], [(1, 4), (2, 5), (3, 6)]]
    as_lists = [[[0, 0], [10, 5], [20, 10], [30, 15]], [[1, 4], [2, 5], [3, 6]]]
    as_arrays = [np.array(s, dtype=np.float64) for s in as_tuples]
    ref = P.normalize_strokes_input(official)
    for other in (as_tuples, as_lists, as_arrays):
        got = P.normalize_strokes_input(other)
        assert len(got) == len(ref)
        for a, b in zip(ref, got):
            assert a.dtype == np.float32 and a.shape[1] == 2
            np.testing.assert_array_equal(a, b)
    # empty strokes are dropped
    assert len(P.normalize_strokes_input([[], [(1, 2), (3, 4), (5, 6)], np.zeros((0, 2))])) == 1
    # webcam style (x, y) tuples with exactly 2 points -> points, not official layout
    two = P.normalize_strokes_input([[(0, 1), (2, 3)]])
    np.testing.assert_array_equal(two[0], [[0, 1], [2, 3]])


def test_points_strokes_roundtrip():
    strokes = P.normalize_strokes_input(SQUARE)
    back = P.points_to_strokes(P.strokes_to_points(strokes))
    assert len(back) == 2
    for a, b in zip(strokes, back):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- sequence
def test_points_to_sequence_shape_and_normalisation():
    pts = P.ndjson_strokes_to_points(SQUARE)
    seq = P.points_to_sequence(pts)
    assert seq.shape == (len(pts) - 1, 3) and seq.dtype == np.float32
    # cumulative sum of deltas + first point (normalised) recovers coords in [0, 1]
    xy = np.cumsum(np.vstack([[0, 0], seq[:, :2]]), axis=0)  # first point is (0,0) after norm here
    assert xy.min() >= -1e-6 and xy.max() <= 1 + 1e-6
    np.testing.assert_allclose(seq[0, :2], [1.0, 0.0])
    np.testing.assert_array_equal(seq[:, 2], pts[1:, 2])
    assert np.abs(seq[:, :2]).max() <= 1.0 + 1e-6


def test_points_to_sequence_zero_range_and_truncation():
    vertical = P.ndjson_strokes_to_points([[[5, 5, 5], [0, 50, 100]]])
    seq = P.points_to_sequence(vertical)
    assert np.all(np.isfinite(seq))
    np.testing.assert_allclose(seq[:, 0], 0.0)
    np.testing.assert_allclose(seq[:, 1], [0.5, 0.5])
    long = P.ndjson_strokes_to_points([[list(range(300)), [i % 7 for i in range(300)]]])
    assert P.points_to_sequence(long, max_len=250).shape == (250, 3)
    assert P.points_to_sequence(long, max_len=10).shape == (10, 3)


def test_points_to_sequence_degenerate():
    for pts in (np.zeros((0, 3), np.float32), np.array([[3, 4, 1]], np.float32)):
        seq = P.points_to_sequence(pts)
        assert seq.shape == (1, 3)


# ---------------------------------------------------------------- render
def test_render_points_shape_dtype_range_and_centring():
    for size in (28, 64, 96):
        img = P.render_points(P.ndjson_strokes_to_points(SQUARE), image_size=size)
        assert img.shape == (size, size) and img.dtype == np.uint8
        assert img.max() == 255 and img.min() == 0
    # an off-centre, non-square drawing ends up centred with the expected extent
    pts = P.ndjson_strokes_to_points([[[200, 250, 250, 200, 200], [10, 10, 30, 30, 10]]])
    img = P.render_points(pts, image_size=64, margin=0.1)
    ys, xs = np.nonzero(img > 64)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    assert abs(cx - 31.5) <= 1.0 and abs(cy - 31.5) <= 1.0
    width = xs.max() - xs.min() + 1
    assert 64 * 0.8 <= width <= 64 * 0.8 + 6  # centre lines span image_size*(1-2*margin), + stroke width
    assert img[:3].max() == 0 and img[-3:].max() == 0 and img[:, :3].max() == 0  # margin respected


def test_render_points_thickness_and_empty():
    pts = P.ndjson_strokes_to_points(SQUARE)
    thin = P.render_points(pts, image_size=64, thickness=1)
    thick = P.render_points(pts, image_size=64, thickness=4)
    assert (thick > 0).sum() > (thin > 0).sum()
    assert P.render_points(np.zeros((0, 3), np.float32)).max() == 0
    single = P.render_points(np.array([[10, 10, 1]], np.float32), image_size=32)
    assert single.max() > 0


# ---------------------------------------------------------------- canvas
def _canvas_with_square(bg, ink, shape=(240, 320), channels=None):
    img = np.full(shape, bg, dtype=np.uint8)
    cv2.rectangle(img, (100, 60), (180, 140), int(ink), 6)
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _check_canvas_output(out, size=64):
    assert out.shape == (size, size) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.max() > 0.5
    # background dark: border is zero, centre (inside square outline) is dark too
    assert out[:3].max() < 0.05 and out[-3:].max() < 0.05
    ys, xs = np.nonzero(out > 0.3)
    assert abs((xs.min() + xs.max()) / 2 - 31.5) <= 2 and abs((ys.min() + ys.max()) / 2 - 31.5) <= 2


def test_preprocess_canvas_white_background_bgr():
    out = P.preprocess_canvas(_canvas_with_square(255, 0, channels=3))
    _check_canvas_output(out)


def test_preprocess_canvas_black_background_gray():
    gray = _canvas_with_square(0, 255)
    _check_canvas_output(P.preprocess_canvas(gray))
    _check_canvas_output(P.preprocess_canvas(gray[:, :, None]))


def test_preprocess_canvas_rgba_transparent_background():
    rgba = np.zeros((240, 320, 4), dtype=np.uint8)  # fully transparent
    mask = _canvas_with_square(0, 255)
    rgba[mask > 0] = (0, 0, 0, 255)  # opaque black ink on transparent background
    _check_canvas_output(P.preprocess_canvas(rgba))
    # opaque BGRA with white background
    bgra = cv2.cvtColor(_canvas_with_square(255, 0, channels=3), cv2.COLOR_BGR2BGRA)
    _check_canvas_output(P.preprocess_canvas(bgra))


def test_preprocess_canvas_float_input():
    f = _canvas_with_square(255, 0).astype(np.float32) / 255.0
    out_f = P.preprocess_canvas(f)
    _check_canvas_output(out_f)
    out_u8 = P.preprocess_canvas(_canvas_with_square(255, 0))
    np.testing.assert_allclose(out_f, out_u8, atol=1e-3)


def test_preprocess_canvas_blank():
    for img in (np.zeros((100, 100), np.uint8), np.full((100, 100, 3), 255, np.uint8),
                np.zeros((50, 60, 4), np.uint8), np.ones((20, 20), np.float32)):
        out = P.preprocess_canvas(img, image_size=32)
        assert out.shape == (32, 32) and out.dtype == np.float32 and out.max() == 0


def test_preprocess_canvas_matches_render_points():
    """A canvas drawn like the training data should give a similar image to render_points."""
    pts = P.ndjson_strokes_to_points(SQUARE)
    ref = P.render_points(pts, image_size=64).astype(np.float32) / 255
    canvas = np.zeros((256, 256), np.uint8)
    for s in P.points_to_strokes(pts):
        q = (s * 2 + 20).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [q], False, 255, 8, cv2.LINE_AA)
    out = P.preprocess_canvas(canvas, image_size=64)
    iou = ((out > 0.3) & (ref > 0.3)).sum() / ((out > 0.3) | (ref > 0.3)).sum()
    assert iou > 0.5


# ---------------------------------------------------------------- simplification
def test_rdp_reduces_and_keeps_endpoints():
    t = np.linspace(0, 100, 201)
    line = np.stack([t, 0.5 * t + np.sin(t * 3) * 0.3], axis=1)  # almost straight
    out = P.rdp(line, epsilon=2.0)
    assert len(out) < len(line) and len(out) >= 2
    np.testing.assert_allclose(out[0], line[0], atol=1e-4)
    np.testing.assert_allclose(out[-1], line[-1], atol=1e-4)
    corner = np.array([[0, 0], [50, 0], [50, 50]], dtype=np.float32)
    assert len(P.rdp(corner, 2.0)) == 3


def test_simplify_strokes_scales_to_255_and_reduces():
    t = np.linspace(0, 2 * np.pi, 400)
    circle = [(float(320 + 150 * np.cos(a)), float(240 + 100 * np.sin(a))) for a in t]
    line = [(10.0 + i, 400.0) for i in range(300)]
    for fmt in ([circle, line], [[[p[0] for p in circle], [p[1] for p in circle]],
                                 [[p[0] for p in line], [p[1] for p in line]]]):
        out = P.simplify_strokes(fmt, epsilon=2.0)
        assert len(out) == 2
        allp = np.concatenate(out)
        assert allp.min() >= 0 and allp.max() == 255
        assert np.all(allp == np.round(allp))
        assert len(out[0]) < len(circle) and len(out[1]) == 2  # straight line -> 2 points
    assert P.simplify_strokes([]) == []


# ---------------------------------------------------------------- split helpers
def test_assign_split_deterministic_and_ratios():
    keys = range(5_000_000_000_000_000, 5_000_000_000_020_000)
    splits = [P.assign_split(k, 42) for k in keys]
    assert splits == [P.assign_split(k, 42) for k in keys]
    frac = {s: splits.count(s) / len(splits) for s in P.SPLITS}
    assert abs(frac["train"] - 0.8) < 0.02 and abs(frac["val"] - 0.1) < 0.02
    assert P.assign_split("5000000000000001", 42) == P.assign_split(5000000000000001, 42)


def test_content_hash_detects_identical_drawings():
    assert P.content_hash(SQUARE) == P.content_hash(json.loads(json.dumps(SQUARE)))
    assert P.content_hash(SQUARE) != P.content_hash(SQUARE[:1])


def test_smooth_stroke_keeps_endpoints_and_reduces_jitter():
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 200, 120)
    clean = np.stack([xs, np.full_like(xs, 50.0)], axis=1)
    noisy = clean + rng.normal(0, 1.5, clean.shape)
    sm = P.smooth_stroke(noisy, 5)
    assert sm.shape == noisy.shape
    np.testing.assert_allclose(sm[0], noisy[0]); np.testing.assert_allclose(sm[-1], noisy[-1])
    assert np.abs(sm[2:-2, 1] - 50).mean() < np.abs(noisy[2:-2, 1] - 50).mean()
    assert P.smooth_stroke(noisy[:2], 5).shape == (2, 2)          # short strokes unchanged
    np.testing.assert_allclose(P.smooth_stroke(noisy, 1), noisy.astype(np.float32))


def test_simplify_with_smoothing_gives_fewer_points_on_jittery_input():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 2 * np.pi, 300)
    circle = np.stack([100 + 80 * np.cos(t), 100 + 80 * np.sin(t)], axis=1) + rng.normal(0, 1.5, (300, 2))
    raw = sum(len(s) for s in P.simplify_strokes([circle], 2.0, smooth_window=0))
    smooth = sum(len(s) for s in P.simplify_strokes([circle], 2.0, smooth_window=5))
    assert smooth < raw
