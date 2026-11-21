"""Preparación de los trazos para BEiT-sketch (la usa beit_classifier.py).

Convierte los trazos del dedo (en píxeles del lienzo) a la misma geometría que los dibujos
oficiales "simplified" de Quick, Draw!: suavizado opcional (quita el temblor de la webcam),
traslación al origen, escala uniforme a 0..255 y simplificación con Ramer-Douglas-Peucker.

    simplify_strokes(trazos, epsilon=2.0, smooth_window=5) -> lista de arrays (n_i, 2)

Solo depende de numpy.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["normalize_strokes_input", "rdp", "smooth_stroke", "simplify_strokes"]


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
