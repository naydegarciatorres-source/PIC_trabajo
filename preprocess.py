"""Quick, Draw! preprocessing: shared transforms + processed-dataset builder.

This module has two roles (see ``docs/CONTRACT.md`` sections 1 and 2):

1. **Shared pure transforms** (numpy / OpenCV only, no torch) used identically at training and
   inference time: stroke-format conversion, simplification (RDP), stroke -> sequence for the
   BiLSTM, stroke -> raster for the CNN, and webcam canvas -> CNN input.
2. **Dataset builder CLI** that streams the official *simplified* ndjson files, keeps only
   recognised drawings, de-duplicates them, deterministically selects ``per_class`` drawings,
   splits them 80/10/10 from a hash of ``key_id`` and writes the memory-mappable format of
   CONTRACT section 1::

       python preprocess.py --classes classes.json --out-dir data/processed --per-class 20000 \
           --raw-dir data/raw --seed 42
       python preprocess.py --categories-check      # only validates classes.json names

Point arrays everywhere are ``(P, 3)`` with columns ``x, y, end_of_stroke`` where
``end_of_stroke`` is 1 on the last point of every stroke.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import sys
import time
import urllib.parse
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

SIMPLIFIED_URL_TEMPLATE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson"
CATEGORIES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
# Rough average size of one line of the simplified ndjson (~330 B of strokes + metadata); only used
# for the download-size warning.
APPROX_BYTES_PER_DRAWING = 500

__all__ = [
    "load_classes",
    "ndjson_strokes_to_points",
    "normalize_strokes_input",
    "strokes_to_points",
    "points_to_strokes",
    "rdp",
    "simplify_strokes",
    "points_to_sequence",
    "render_points",
    "preprocess_canvas",
    "assign_split",
    "content_hash",
    "ProcessedDatasetWriter",
    "build_dataset",
    "check_categories",
    "main",
]


# ----------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------
def load_classes(path: str = "classes.json") -> list[str]:
    """Return the ordered list of class names from ``classes.json`` (label = index).

    Accepts either ``{"classes": [...]}`` or a bare JSON list. Raises ``ValueError`` on an empty
    list or duplicated names.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"] if isinstance(data, dict) else data
    classes = [str(c) for c in classes]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    if len(set(classes)) != len(classes):
        dupes = sorted({c for c in classes if classes.count(c) > 1})
        raise ValueError(f"Duplicated class names in {path}: {dupes}")
    return classes


# ----------------------------------------------------------------------------------------------
# Stroke format conversion
# ----------------------------------------------------------------------------------------------
def _stroke_layout(stroke: Any) -> str:
    """Classify one stroke as ``"official"`` ([xs, ys(, ts)]), ``"points"`` ([(x,y),...]),
    ``"ambiguous"`` (e.g. 2x2: could be either) or ``"empty"``."""
    if isinstance(stroke, np.ndarray):
        arr = stroke
        if arr.size == 0:
            return "empty"
        if arr.ndim == 1:
            return "points" if arr.shape[0] in (2, 3) else "empty"
        if arr.ndim != 2:
            raise ValueError(f"Stroke array must be 2-D, got shape {arr.shape}")
        # numpy strokes are documented as (n, 2) point arrays; only transpose when unambiguous.
        if arr.shape[1] in (2, 3):
            return "points"
        if arr.shape[0] in (2, 3):
            return "official"
        raise ValueError(f"Cannot interpret stroke array of shape {arr.shape}")
    seq = list(stroke)
    if len(seq) == 0:
        return "empty"
    first = seq[0]
    if np.isscalar(first):
        # a single bare (x, y) point given as a stroke
        return "points"
    if isinstance(first, tuple):
        return "points"  # tuples are the natural way to write (x, y) points
    lens = [len(e) for e in seq]
    official = len(seq) in (2, 3) and len(set(lens)) == 1
    strict_points = all(n == 2 for n in lens)
    if official and strict_points:
        return "ambiguous"  # e.g. [[x0, x1], [y0, y1]] vs [[x0, y0], [x1, y1]]
    if official:
        return "official"
    if all(n in (2, 3) for n in lens):
        return "points"
    raise ValueError(f"Cannot interpret stroke with element lengths {lens[:5]}...")


def normalize_strokes_input(strokes: Any) -> list[np.ndarray]:
    """Convert any supported stroke container into a list of ``(n_i, 2)`` float32 arrays.

    Supported inputs:

    * official Quick, Draw! format ``[[[x0, x1, ...], [y0, y1, ...]], ...]`` (an optional third
      list of timestamps, as in the raw data, is ignored);
    * list of strokes, each a list of ``(x, y)`` points (tuples/lists) or an ``(n, 2)``/``(n, 3)``
      numpy array (extra columns ignored).

    Empty strokes are dropped. A stroke whose shape is ambiguous (e.g. exactly 2 lists of length
    2) is resolved using the layout of the other strokes of the same drawing, defaulting to the
    official format (inner tuples are always read as points).
    """
    if strokes is None:
        return []
    if isinstance(strokes, np.ndarray) and strokes.dtype != object and strokes.ndim == 2:
        # a single stroke passed as one array
        strokes = [strokes]
    strokes = list(strokes)
    layouts = [_stroke_layout(s) for s in strokes]
    decided = {l for l in layouts if l in ("official", "points")}
    default = "points" if decided == {"points"} else "official"

    out: list[np.ndarray] = []
    for stroke, layout in zip(strokes, layouts):
        if layout == "empty":
            continue
        if layout == "ambiguous":
            layout = default
        if isinstance(stroke, np.ndarray):
            arr = stroke.astype(np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
        elif np.isscalar(list(stroke)[0]):
            arr = np.asarray([stroke], dtype=np.float32)
        else:
            arr = np.asarray([list(e) for e in stroke], dtype=np.float32)
        if layout == "official":
            arr = arr[:2].T
        arr = np.ascontiguousarray(arr[:, :2], dtype=np.float32)
        if arr.shape[0] == 0:
            continue
        out.append(arr)
    return out


def strokes_to_points(strokes: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate a list of ``(n_i, 2)`` strokes into a ``(P, 3)`` float32 points array."""
    parts = []
    for s in strokes:
        s = np.asarray(s, dtype=np.float32).reshape(-1, 2)
        if len(s) == 0:
            continue
        eos = np.zeros((len(s), 1), dtype=np.float32)
        eos[-1, 0] = 1.0
        parts.append(np.concatenate([s, eos], axis=1))
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def points_to_strokes(points: np.ndarray) -> list[np.ndarray]:
    """Inverse of :func:`strokes_to_points`: split ``(P, 3)`` points at ``end_of_stroke`` flags.

    A trailing run without a final flag is returned as the last stroke. ``(P, 2)`` input is
    treated as a single stroke.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return []
    pts = pts.reshape(len(pts), -1)
    if pts.shape[1] < 3:
        return [pts[:, :2].copy()]
    ends = np.flatnonzero(pts[:, 2] > 0.5) + 1
    bounds = [0] + [int(e) for e in ends]
    if bounds[-1] != len(pts):
        bounds.append(len(pts))
    return [pts[a:b, :2].copy() for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def ndjson_strokes_to_points(strokes: Any) -> np.ndarray:
    """Official ``[[[x...], [y...]], ...]`` drawing -> ``(P, 3)`` float32 rows ``x, y, eos``.

    ``eos`` (end_of_stroke) is 1 on the last point of each stroke, 0 otherwise. Also accepts the
    other formats of :func:`normalize_strokes_input`. Empty drawing -> shape ``(0, 3)``.
    """
    return strokes_to_points(normalize_strokes_input(strokes))


# ----------------------------------------------------------------------------------------------
# Simplification (to make live strokes look like the official simplified data)
# ----------------------------------------------------------------------------------------------
def rdp(points: np.ndarray, epsilon: float = 2.0) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification (iterative, keeps both endpoints).

    ``points``: ``(n, 2)`` array. Returns the kept subset ``(m, 2)``, ``m <= n``, in order.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n <= 2:
        return pts.astype(np.float32)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[a + 1 : b]
        p, q = pts[a], pts[b]
        d = q - p
        norm = math.hypot(d[0], d[1])
        if norm == 0.0:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / norm
        i = int(np.argmax(dist))
        if dist[i] > epsilon:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    return pts[keep].astype(np.float32)


def smooth_stroke(stroke: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing of one ``(n, 2)`` stroke; the first and last points are kept.

    Live fingertip tracking adds small jitter that RDP would otherwise keep as spurious zig-zag
    points (the stroke model is sensitive to them). ``window <= 1`` or short strokes: unchanged.
    """
    s = np.asarray(stroke, dtype=np.float32).reshape(-1, 2)
    window = int(window)
    if window <= 1 or len(s) <= 2:
        return s
    window = min(window, len(s) if len(s) % 2 == 1 else len(s) - 1)
    if window <= 1:
        return s
    half = window // 2
    padded = np.concatenate([np.repeat(s[:1], half, 0), s, np.repeat(s[-1:], half, 0)], axis=0)
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], axis=1)
    out[0], out[-1] = s[0], s[-1]
    return out.astype(np.float32)


def simplify_strokes(strokes: Any, epsilon: float = 2.0, smooth_window: int = 0) -> list[np.ndarray]:
    """Mimic the official simplification of Quick, Draw! drawings.

    Input: any format accepted by :func:`normalize_strokes_input`. The drawing is translated so its
    minimum is 0, scaled uniformly (aspect preserved) so the largest side spans 0..255, rounded to
    integer coordinates and every stroke is simplified with RDP (``epsilon`` in that 0..255 space).
    ``smooth_window > 1`` first applies :func:`smooth_stroke` (recommended for live webcam input).
    Returns a list of ``(n_i, 2)`` float32 arrays (integral values), empty strokes dropped.
    """
    strokes_n = normalize_strokes_input(strokes)
    if not strokes_n:
        return []
    if smooth_window and smooth_window > 1:
        strokes_n = [smooth_stroke(s, smooth_window) for s in strokes_n]
    allp = np.concatenate(strokes_n, axis=0)
    lo = allp.min(axis=0)
    extent = float((allp.max(axis=0) - lo).max())
    scale = 255.0 / extent if extent > 0 else 1.0
    out = []
    for s in strokes_n:
        t = np.clip(np.round((s - lo) * scale), 0, 255)
        # drop consecutive duplicates introduced by rounding, then RDP
        if len(t) > 1:
            mask = np.ones(len(t), dtype=bool)
            mask[1:] = np.any(np.diff(t, axis=0) != 0, axis=1)
            t = t[mask]
        out.append(rdp(t, epsilon).astype(np.float32))
    return out


# ----------------------------------------------------------------------------------------------
# Model inputs
# ----------------------------------------------------------------------------------------------
def points_to_sequence(points: np.ndarray, max_len: int = 250) -> np.ndarray:
    """``(P, 3)`` points -> ``(T, 3)`` float32 sequence for the stroke model (TF RNN tutorial).

    x, y are normalised to [0, 1] with the drawing bounding box (per axis; a zero range becomes 1),
    converted to deltas between consecutive points, and the first row is dropped. Column 3 keeps
    ``end_of_stroke``. The result is truncated to ``max_len`` rows. ``T >= 1`` always: drawings
    with fewer than 2 points give ``[[0, 0, 1]]``.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3) if np.size(points) else np.zeros((0, 3), np.float32)
    if len(pts) < 2:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    ink = pts.copy()
    lower = ink[:, :2].min(axis=0)
    scale = ink[:, :2].max(axis=0) - lower
    scale[scale == 0] = 1.0
    ink[:, :2] = (ink[:, :2] - lower) / scale
    ink[1:, :2] = ink[1:, :2] - ink[:-1, :2]
    seq = ink[1:]
    if max_len is not None and max_len > 0:
        seq = seq[:max_len]
    return np.ascontiguousarray(seq, dtype=np.float32)


def _default_thickness(image_size: int) -> int:
    return max(1, int(round(image_size / 32)))


def _render(
    points: np.ndarray,
    image_size: int = 64,
    thickness: int | None = None,
    margin: float = 0.1,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Implementation of :func:`render_points` with an extra pixel ``offset`` (augmentation)."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    strokes = points_to_strokes(points)
    if not strokes:
        return img
    if thickness is None:
        thickness = _default_thickness(image_size)
    thickness = max(1, int(thickness))
    allp = np.concatenate(strokes, axis=0).astype(np.float64)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    extent = float((hi - lo).max())
    box = image_size * (1.0 - 2.0 * margin)
    scale = box / extent if extent > 0 else 1.0
    centre_src = (lo + hi) / 2.0
    centre_dst = np.array([(image_size - 1) / 2.0 + offset[0], (image_size - 1) / 2.0 + offset[1]])
    shift_bits = 4
    mult = float(1 << shift_bits)
    # anti-aliased lines for thickness >= 2; 1-px AA lines never reach 255, so use 8-connected
    line_type = cv2.LINE_AA if thickness >= 2 else cv2.LINE_8
    for s in strokes:
        p = (s.astype(np.float64) - centre_src) * scale + centre_dst
        q = np.round(p * mult).astype(np.int32).reshape(-1, 1, 2)
        if len(q) == 1:
            pt = (int(q[0, 0, 0]), int(q[0, 0, 1]))
            cv2.line(img, pt, pt, 255, thickness, line_type, shift_bits)
        else:
            cv2.polylines(img, [q], False, 255, thickness, line_type, shift_bits)
    return img


def render_points(
    points: np.ndarray, image_size: int = 64, thickness: int | None = None, margin: float = 0.1
) -> np.ndarray:
    """Rasterise ``(P, 3)`` points to an ``(image_size, image_size)`` uint8 image.

    White (255) anti-aliased strokes on black (0). The drawing's bounding box is centred and
    scaled (aspect preserved) so its longest side spans ``image_size * (1 - 2 * margin)`` pixels.
    Default ``thickness = max(1, round(image_size / 32))``. Empty input -> all zeros.
    """
    return _render(points, image_size=image_size, thickness=thickness, margin=margin)


def _to_gray_float(canvas_img: np.ndarray) -> np.ndarray:
    """Any OpenCV-style image -> float32 gray in [0, 255] (alpha composited sensibly)."""
    img = np.asarray(canvas_img)
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    if np.issubdtype(img.dtype, np.floating):
        f = np.nan_to_num(img.astype(np.float32))
        if f.size and f.max() <= 1.0 + 1e-6:
            f = f * 255.0
        img_f = np.clip(f, 0, 255)
    elif img.dtype == np.uint16:
        img_f = img.astype(np.float32) / 257.0
    else:
        img_f = np.clip(img.astype(np.float32), 0, 255)

    if img_f.ndim == 3 and img_f.shape[2] == 1:
        img_f = img_f[:, :, 0]
    if img_f.ndim == 2:
        return img_f
    if img_f.ndim != 3 or img_f.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported canvas shape {img.shape}")
    gray = cv2.cvtColor(np.ascontiguousarray(img_f[:, :, :3]), cv2.COLOR_BGR2GRAY)
    if img_f.shape[2] == 3:
        return gray
    alpha = img_f[:, :, 3] / 255.0
    if float(alpha.max() - alpha.min()) < 1e-3:
        return gray  # opaque (or uniformly transparent) image: colour channels carry the drawing
    # Transparent background: composite over a background that contrasts with the ink colour.
    w = alpha.sum()
    ink_level = float((gray * alpha).sum() / w) if w > 0 else 0.0
    bg = 255.0 if ink_level < 128 else 0.0
    return gray * alpha + bg * (1.0 - alpha)


def preprocess_canvas(canvas_img: np.ndarray, image_size: int = 64, margin: float = 0.1) -> np.ndarray:
    """Webcam/game canvas -> CNN input ``(image_size, image_size)`` float32 in [0, 1].

    Accepts ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` BGR or ``(H, W, 4)`` BGRA, uint8 or float
    (floats with max <= 1 are treated as [0, 1]). Steps: grayscale; detect the background from the
    median of the image border and invert if it is light so ink is bright on dark; subtract the
    background level and stretch to 0..255; light threshold (< 20% of max -> 0) to remove noise;
    crop a square around the ink bounding box such that the stroke centre lines (bbox inset by the
    estimated half stroke width) fill ``1 - 2 * margin`` of it, exactly like
    :func:`render_points`; resize (INTER_AREA when shrinking, INTER_LINEAR when enlarging).
    A blank canvas returns all zeros.
    """
    gray = _to_gray_float(canvas_img)
    out_blank = np.zeros((image_size, image_size), dtype=np.float32)
    if gray.size == 0:
        return out_blank
    h, w = gray.shape
    bw = max(1, min(h, w) // 50)
    border = np.concatenate(
        [gray[:bw].ravel(), gray[-bw:].ravel(), gray[:, :bw].ravel(), gray[:, -bw:].ravel()]
    )
    bg = float(np.median(border))
    if bg > 127.5:
        gray = 255.0 - gray
        bg = 255.0 - bg
    ink = np.clip(gray - bg, 0, None)
    mx = float(ink.max())
    if mx < 16.0:  # no meaningful contrast -> blank
        return out_blank
    ink = ink * (255.0 / mx)
    ink[ink < 0.2 * 255.0] = 0.0
    mask = (ink > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return out_blank
    # Estimate the half stroke width (distance-transform ridge) so that the *centre lines* of the
    # strokes, not the outer edge of the ink, are mapped to the margin box like render_points does.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    half_t = float(np.median(dist[ridge])) - 0.5 if ridge.any() else 0.0
    half_t = max(0.0, half_t)
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    sk_w = max(1.0, (x1 - x0) - 2 * half_t)
    sk_h = max(1.0, (y1 - y0) - 2 * half_t)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    inner = max(1e-6, 1.0 - 2.0 * margin)
    full = int(math.ceil(max(sk_w, sk_h) / inner)) + 1
    full = max(full, 2)
    # crop a full x full square centred on the ink (zero padding outside the canvas)
    left = int(round(cx - (full - 1) / 2.0))
    top = int(round(cy - (full - 1) / 2.0))
    sq = np.zeros((full, full), dtype=np.float32)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + full), min(h, top + full)
    if sx1 > sx0 and sy1 > sy0:
        sq[sy0 - top : sy1 - top, sx0 - left : sx1 - left] = ink[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if full >= image_size else cv2.INTER_LINEAR
    out = cv2.resize(sq, (image_size, image_size), interpolation=interp)
    return np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Splitting / dedupe helpers
# ----------------------------------------------------------------------------------------------
def _hash_unit(text: str) -> float:
    """md5(text) -> float in [0, 1)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:13], 16) / float(16**13)


def assign_split(key_id: int | str, seed: int = 42, ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS) -> str:
    """Deterministic split ("train"/"val"/"test") from ``md5("split:{seed}:{key_id}")``.

    Stable across runs, machines and ``per_class`` values, so a drawing never changes split.
    """
    u = _hash_unit(f"split:{seed}:{int(key_id)}")
    total = float(sum(ratios))
    acc = 0.0
    for name, r in zip(SPLITS, ratios):
        acc += r / total
        if u < acc:
            return name
    return SPLITS[-1]


def _selection_key(key_id: int, seed: int) -> str:
    # salt differs from assign_split so selection is independent of the split assignment
    return hashlib.md5(f"select:{seed}:{int(key_id)}".encode("utf-8")).hexdigest()


def content_hash(drawing: Any) -> int:
    """64-bit hash (as Python int < 2**63) of the stroke content of an official-format drawing."""
    canon = json.dumps(drawing, separators=(",", ":"))
    return int(hashlib.md5(canon.encode("utf-8")).hexdigest()[:16], 16) & ((1 << 63) - 1)


# ----------------------------------------------------------------------------------------------
# Writer for the processed format (CONTRACT section 1), low RAM
# ----------------------------------------------------------------------------------------------
class ProcessedDatasetWriter:
    """Incrementally writes ``{split}_points/offsets/labels/key_ids.npy`` + ``meta.json``.

    Points are appended to a temporary raw ``.bin`` file per split and converted to ``.npy`` in
    chunks at :meth:`finalize`, so RAM usage stays proportional to the number of drawings (not
    points). :meth:`finalize` also verifies that key_id sets are disjoint across splits.
    """

    def __init__(self, out_dir: str, classes: Sequence[str]):
        self.out_dir = out_dir
        self.classes = list(classes)
        os.makedirs(out_dir, exist_ok=True)
        self._bin = {s: open(os.path.join(out_dir, f"{s}_points.bin.tmp"), "wb") for s in SPLITS}
        self._n_points = {s: 0 for s in SPLITS}
        self._lengths: dict[str, list[int]] = {s: [] for s in SPLITS}
        self._labels: dict[str, list[int]] = {s: [] for s in SPLITS}
        self._key_ids: dict[str, list[int]] = {s: [] for s in SPLITS}
        self.counts: dict[str, dict[str, int]] = {s: {c: 0 for c in self.classes} for s in SPLITS}

    def add(self, split: str, points: np.ndarray, label: int, key_id: int) -> None:
        """Append one drawing (``(P, 3)`` points with x, y in 0..255) to ``split``."""
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            raise ValueError(f"points must be a non-empty (P, 3) array, got {pts.shape}")
        arr = np.clip(np.round(pts), 0, 255).astype(np.uint8)
        self._bin[split].write(np.ascontiguousarray(arr).tobytes())
        self._n_points[split] += len(arr)
        self._lengths[split].append(len(arr))
        self._labels[split].append(int(label))
        self._key_ids[split].append(int(key_id))
        self.counts[split][self.classes[int(label)]] += 1

    def finalize(self, meta_extra: dict | None = None) -> dict:
        """Write all ``.npy`` files and ``meta.json``; verify no key_id leakage. Returns meta."""
        chunk_rows = 4_000_000
        for s in SPLITS:
            self._bin[s].close()
            tmp = os.path.join(self.out_dir, f"{s}_points.bin.tmp")
            n_pts = self._n_points[s]
            out = np.lib.format.open_memmap(
                os.path.join(self.out_dir, f"{s}_points.npy"), mode="w+", dtype=np.uint8, shape=(n_pts, 3)
            )
            if n_pts:
                src = np.memmap(tmp, dtype=np.uint8, mode="r", shape=(n_pts, 3))
                for a in range(0, n_pts, chunk_rows):
                    out[a : a + chunk_rows] = src[a : a + chunk_rows]
                del src
            out.flush()
            del out
            os.remove(tmp)
            offsets = np.zeros(len(self._lengths[s]) + 1, dtype=np.int64)
            np.cumsum(np.asarray(self._lengths[s], dtype=np.int64), out=offsets[1:])
            np.save(os.path.join(self.out_dir, f"{s}_offsets.npy"), offsets)
            np.save(os.path.join(self.out_dir, f"{s}_labels.npy"), np.asarray(self._labels[s], dtype=np.int16))
            np.save(os.path.join(self.out_dir, f"{s}_key_ids.npy"), np.asarray(self._key_ids[s], dtype=np.int64))
        verify_processed_dataset(self.out_dir, len(self.classes))
        meta = {
            "classes": self.classes,
            "counts": self.counts,
            "source": SIMPLIFIED_URL_TEMPLATE,
            "per_class": 0,
            "split_ratios": list(DEFAULT_SPLIT_RATIOS),
            "seed": 0,
        }
        if meta_extra:
            meta.update(meta_extra)
        tmp_meta = os.path.join(self.out_dir, "meta.json.tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, os.path.join(self.out_dir, "meta.json"))
        return meta


def verify_processed_dataset(out_dir: str, num_classes: int | None = None) -> None:
    """Check a processed directory: consistent shapes, no duplicate or cross-split key_ids.

    Raises ``RuntimeError`` on any inconsistency (in particular key_id leakage between splits).
    """
    seen: dict[str, np.ndarray] = {}
    for s in SPLITS:
        kp = os.path.join(out_dir, f"{s}_key_ids.npy")
        if not os.path.exists(kp):
            continue
        keys = np.load(kp)
        labels = np.load(os.path.join(out_dir, f"{s}_labels.npy"))
        offsets = np.load(os.path.join(out_dir, f"{s}_offsets.npy"))
        points = np.load(os.path.join(out_dir, f"{s}_points.npy"), mmap_mode="r")
        if not (len(keys) == len(labels) == len(offsets) - 1):
            raise RuntimeError(f"{s}: inconsistent lengths keys={len(keys)} labels={len(labels)} offsets={len(offsets)}")
        if offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) <= 0):
            raise RuntimeError(f"{s}: invalid offsets")
        if num_classes is not None and len(labels) and (labels.min() < 0 or labels.max() >= num_classes):
            raise RuntimeError(f"{s}: label out of range")
        if len(np.unique(keys)) != len(keys):
            raise RuntimeError(f"{s}: duplicated key_ids inside split")
        seen[s] = keys
    names = list(seen)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = np.intersect1d(seen[names[i]], seen[names[j]])
            if len(inter):
                raise RuntimeError(f"key_id leakage between {names[i]} and {names[j]}: {inter[:5].tolist()}")


# ----------------------------------------------------------------------------------------------
# Download / streaming
# ----------------------------------------------------------------------------------------------
def class_url(name: str) -> str:
    """Official simplified ndjson URL for a class (spaces -> %20)."""
    return SIMPLIFIED_URL_TEMPLATE.format(name=urllib.parse.quote(name))


def _safe_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_. ") else "_" for ch in name)


def stream_class_pool(
    name: str,
    pool_size: int,
    timeout: float = 60.0,
    retries: int = 4,
    progress: bool = True,
) -> tuple[list[dict], bool, int]:
    """Stream ``name``'s simplified ndjson and collect ``pool_size`` recognised unique drawings.

    Stops reading as soon as the pool is full (no full-file download). Drawings are de-duplicated
    by ``key_id`` and by stroke content. Returns ``(records, exhausted, lines_read)`` with records
    ``{"key_id": int, "drawing": [...]}`` in file order; ``exhausted`` is True when the whole file
    was read. Network errors are retried from the start with exponential backoff.
    """
    import requests
    from tqdm import tqdm

    url = class_url(name)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        records: list[dict] = []
        seen_keys: set[int] = set()
        seen_content: set[int] = set()
        lines = 0
        exhausted = False
        try:
            with requests.get(url, stream=True, timeout=(15, timeout)) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f"Class '{name}' not found at {url} (check the exact official name)")
                r.raise_for_status()
                bar = tqdm(total=pool_size, desc=f"{name[:18]:>18}", unit="dr", leave=False, disable=not progress)
                for raw in r.iter_lines(chunk_size=1 << 16):
                    if not raw:
                        continue
                    lines += 1
                    obj = json.loads(raw)
                    if not obj.get("recognized", False):
                        continue
                    kid = int(obj["key_id"])
                    drawing = obj["drawing"]
                    if kid in seen_keys or not drawing:
                        continue
                    ch = content_hash(drawing)
                    if ch in seen_content:
                        continue
                    seen_keys.add(kid)
                    seen_content.add(ch)
                    records.append({"key_id": kid, "drawing": drawing})
                    bar.update(1)
                    if len(records) >= pool_size:
                        break
                else:
                    exhausted = True
                bar.close()
            return records, exhausted, lines
        except FileNotFoundError:
            raise
        except Exception as e:  # network / decoding errors -> retry
            last_err = e
            wait = min(60, 2**attempt)
            print(f"  [warn] {name}: {type(e).__name__}: {e} (attempt {attempt + 1}/{retries + 1}, retry in {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download '{name}' after {retries + 1} attempts: {last_err}")


def get_class_pool(
    name: str, pool_size: int, raw_dir: str | None, timeout: float = 60.0, retries: int = 4
) -> tuple[list[dict], bool]:
    """Return the pool for ``name``, from the ``{raw_dir}/{name}.subset.ndjson`` cache if it is
    large enough (or the source was exhausted), otherwise streaming and refreshing the cache."""
    cache = side = None
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
        base = os.path.join(raw_dir, _safe_filename(name))
        cache, side = base + ".subset.ndjson", base + ".subset.json"
        if os.path.exists(cache) and os.path.exists(side):
            try:
                with open(side, "r", encoding="utf-8") as f:
                    info = json.load(f)
                if info.get("count", 0) >= pool_size or info.get("exhausted", False):
                    records = []
                    with open(cache, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))
                    if len(records) == info.get("count"):
                        return records[:pool_size], bool(info.get("exhausted", False)) and len(records) <= pool_size
            except (OSError, ValueError, KeyError):
                pass  # corrupt cache -> re-download
    records, exhausted, lines = stream_class_pool(name, pool_size, timeout=timeout, retries=retries)
    if cache:
        tmp = cache + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        os.replace(tmp, cache)
        with open(side, "w", encoding="utf-8") as f:
            json.dump({"name": name, "url": class_url(name), "count": len(records), "pool_size": pool_size,
                       "exhausted": exhausted, "lines_read": lines}, f)
    return records, exhausted


# ----------------------------------------------------------------------------------------------
# Categories check
# ----------------------------------------------------------------------------------------------
def fetch_official_categories(timeout: float = 30.0) -> list[str]:
    """Download the official list of 345 category names."""
    import requests

    r = requests.get(CATEGORIES_URL, timeout=timeout)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip()]


def check_categories(classes: Sequence[str], official: Sequence[str] | None = None) -> list[str]:
    """Return the class names that are not exact official category names (empty = all good)."""
    if official is None:
        official = fetch_official_categories()
    off = set(official)
    return [c for c in classes if c not in off]


# ----------------------------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------------------------
def build_dataset(
    classes: Sequence[str],
    out_dir: str,
    per_class: int = 20000,
    raw_dir: str | None = "data/raw",
    seed: int = 42,
    pool_factor: float = 1.25,
    timeout: float = 60.0,
    retries: int = 4,
) -> dict:
    """Build the processed dataset (CONTRACT section 1) for ``classes`` into ``out_dir``.

    For every class (in label order): obtain a pool of ``ceil(per_class * pool_factor)``
    recognised, de-duplicated drawings (streamed or cached), sort it by a seeded hash of
    ``key_id``, skip drawings whose key_id or stroke content was already used by a previous class,
    take ``per_class`` and route each one to train/val/test with :func:`assign_split`.
    Returns the written meta dict.
    """
    from tqdm import tqdm

    pool_size = int(math.ceil(per_class * pool_factor))
    writer = ProcessedDatasetWriter(out_dir, classes)
    used_keys = np.zeros(0, dtype=np.int64)
    used_content = np.zeros(0, dtype=np.int64)
    short: dict[str, int] = {}
    for label, name in enumerate(tqdm(classes, desc="classes", unit="cls")):
        records, _ = get_class_pool(name, pool_size, raw_dir, timeout=timeout, retries=retries)
        records.sort(key=lambda r: _selection_key(r["key_id"], seed))
        keys = np.fromiter((r["key_id"] for r in records), dtype=np.int64, count=len(records))
        chs = np.fromiter((content_hash(r["drawing"]) for r in records), dtype=np.int64, count=len(records))
        dup = np.isin(keys, used_keys) | np.isin(chs, used_content)
        taken_k, taken_c = [], []
        for rec, is_dup, k, c in zip(records, dup, keys, chs):
            if is_dup:
                continue
            pts = ndjson_strokes_to_points(rec["drawing"])
            if len(pts) == 0:
                continue
            writer.add(assign_split(rec["key_id"], seed), pts, label, rec["key_id"])
            taken_k.append(k)
            taken_c.append(c)
            if len(taken_k) >= per_class:
                break
        if len(taken_k) < per_class:
            short[name] = len(taken_k)
            tqdm.write(f"  [warn] {name}: only {len(taken_k)} unique recognised drawings available (< {per_class})")
        used_keys = np.union1d(used_keys, np.asarray(taken_k, dtype=np.int64))
        used_content = np.union1d(used_content, np.asarray(taken_c, dtype=np.int64))
        del records
    meta = writer.finalize(
        {
            "per_class": int(per_class),
            "seed": int(seed),
            "split_ratios": list(DEFAULT_SPLIT_RATIOS),
            "pool_factor": float(pool_factor),
            "short_classes": short,
        }
    )
    return meta


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point (see module docstring). Returns a process exit code."""
    p = argparse.ArgumentParser(description="Build the processed Quick, Draw! dataset (simplified ndjson).")
    p.add_argument("--classes", default="classes.json")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--per-class", type=int, default=20000)
    p.add_argument("--raw-dir", default="data/raw", help="cache of parsed per-class subsets ('' to disable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-factor", type=float, default=1.25)
    p.add_argument("--timeout", type=float, default=60.0, help="read timeout in seconds")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--categories-check", action="store_true",
                   help="only verify that every class name exists in the official categories.txt and exit")
    args = p.parse_args(list(argv) if argv is not None else None)

    classes = load_classes(args.classes)
    if args.categories_check:
        official = fetch_official_categories()
        missing = check_categories(classes, official)
        print(f"Official categories: {len(official)}; classes.json: {len(classes)}")
        if missing:
            print(f"MISSING ({len(missing)}) - not exact official names:")
            for m in missing:
                sugg = difflib.get_close_matches(m, official, n=3)
                print(f"  - {m!r}  (did you mean: {sugg})")
            return 1
        print("OK: all class names are valid official categories.")
        return 0

    if args.per_class <= 0:
        p.error("--per-class must be > 0")
    pool = int(math.ceil(args.per_class * args.pool_factor))
    est = len(classes) * pool / 0.9 * APPROX_BYTES_PER_DRAWING
    print("=" * 78)
    print(f"Building Quick, Draw! dataset: {len(classes)} classes x {args.per_class} drawings "
          f"(pool {pool}/class), seed {args.seed}")
    print(f"WARNING: this streams roughly {_human(est)} from Google Cloud Storage (only the first "
          f"~{pool} recognised drawings of each file; full files are 50-300 MB each).")
    print(f"Cached subsets go to '{args.raw_dir or '(no cache)'}' (reruns skip cached classes); "
          f"output -> '{args.out_dir}'.")
    print("=" * 78)
    t0 = time.time()
    meta = build_dataset(classes, args.out_dir, per_class=args.per_class, raw_dir=args.raw_dir or None,
                         seed=args.seed, pool_factor=args.pool_factor, timeout=args.timeout, retries=args.retries)
    dt = time.time() - t0
    totals = {s: sum(meta["counts"][s].values()) for s in SPLITS}
    print(f"Done in {dt:.1f}s. Drawings per split: {totals}. Output size: {_human(_dir_size(args.out_dir))}")
    if meta.get("short_classes"):
        print(f"Classes with fewer than {args.per_class} drawings: {meta['short_classes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
