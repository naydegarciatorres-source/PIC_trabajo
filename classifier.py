"""Game-side inference for the webcam Pictionary (CONTRACT section 7).

Quick use from ``vision.py``::

    import classifier
    top3 = classifier.predict(canvas_img)           # CNN, OpenCV canvas (any bg / colour)
    top3 = classifier.predict_strokes(strokes)      # BiLSTM, recorded fingertip strokes
    # -> [("cat", 0.81), ("dog", 0.07), ("bird", 0.03)]

Default checkpoints: ``checkpoints/cnn/best.pt`` and ``checkpoints/bilstm/best.pt`` (relative to
this file), overridable with the env vars ``QUICKDRAW_CNN_CKPT`` / ``QUICKDRAW_BILSTM_CKPT``.

Behaviour notes
- Probabilities are a softmax over all classes, returned sorted in descending order.
- Empty drawing (blank canvas / no strokes) never crashes: the model receives an all-zero input
  and the returned list simply has low, near-uniform confidences. Use :func:`is_blank` /
  the probability value to decide whether to show a guess.
- ``predict_strokes`` on an image model renders the strokes into an image (strokes -> image is
  lossless enough). ``predict`` on a stroke model raises: the stroke order and pen-up events cannot
  be recovered from a raster image, so the game must record the strokes (see
  docs/INTEGRATION_vision.md).

Demo::

    python classifier.py --checkpoint checkpoints/cnn/best.pt --image drawing.png
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Sequence

import numpy as np
import torch

import preprocess
from model import load_model_from_checkpoint

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CNN_CKPT = os.path.join(_HERE, "checkpoints", "cnn", "best.pt")
DEFAULT_BILSTM_CKPT = os.path.join(_HERE, "checkpoints", "bilstm", "best.pt")
ENV_CNN = "QUICKDRAW_CNN_CKPT"
ENV_BILSTM = "QUICKDRAW_BILSTM_CKPT"


def _resolve_device(device) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _missing_checkpoint_error(path: str, kind: str = "") -> FileNotFoundError:
    model = {"cnn": "cnn", "bilstm": "bilstm"}.get(kind, "cnn|bilstm")
    env = {"cnn": ENV_CNN, "bilstm": ENV_BILSTM}.get(kind, f"{ENV_CNN}/{ENV_BILSTM}")
    return FileNotFoundError(
        f"Model checkpoint not found: {path}\n"
        f"  - Train it:   python train.py --model {model}   (writes checkpoints/<run>/best.pt)\n"
        f"  - or download a trained best.pt and place it at that path,\n"
        f"  - or point the env var {env} to an existing checkpoint file.")


class QuickDrawClassifier:
    """Wraps a trained checkpoint for fast single-drawing inference.

    Attributes: ``classes`` (list[str], label i <-> classes[i]), ``mode`` ("image"|"strokes"),
    ``model_name``, ``device``, ``image_size``, ``max_len``, ``margin``.
    Thread-safe for concurrent ``predict`` calls (no mutable state after construction).
    """

    def __init__(self, checkpoint_path: str, device=None):
        checkpoint_path = os.fspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise _missing_checkpoint_error(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.device = _resolve_device(device)
        model, ckpt = load_model_from_checkpoint(checkpoint_path, device=str(self.device))
        self.model = model.eval()
        self.classes = list(ckpt["classes"])
        self.num_classes = len(self.classes)
        self.model_name = ckpt.get("model_name")
        pp = dict(ckpt.get("preprocessing") or {})
        self.mode = pp.get("mode") or ("strokes" if self.model_name == "bilstm" else "image")
        self.image_size = int(pp.get("image_size", 64))
        self.max_len = int(pp.get("max_len", 250))
        self.margin = float(pp.get("margin", 0.1))
        self.epsilon = float(pp.get("simplify_epsilon", 2.0))
        # Live-input smoothing (not applied to the already-simplified training data).
        self.smooth_window = int(pp.get("smooth_window", 5))
        self.preprocessing = pp
        if self.mode not in ("image", "strokes"):
            raise ValueError(f"Unknown preprocessing mode {self.mode!r} in {checkpoint_path}")

    def __repr__(self):
        return (f"QuickDrawClassifier({self.model_name}, mode={self.mode}, "
                f"classes={self.num_classes}, device={self.device})")

    # ---------------------------------------------------------------- input -> tensor
    def canvas_to_tensor(self, canvas_img) -> torch.Tensor:
        """OpenCV canvas -> (1,1,H,W) float tensor, using the shared ``preprocess_canvas``."""
        if canvas_img is None:
            raise ValueError("canvas_img is None (no frame/canvas given)")
        img = preprocess.preprocess_canvas(np.asarray(canvas_img), image_size=self.image_size,
                                           margin=self.margin)
        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None]

    @staticmethod
    def strokes_to_points(strokes, epsilon: float = 2.0, smooth_window: int = 5) -> np.ndarray:
        """Raw strokes (official format or lists of (x,y)) -> simplified (P,3) points
        ``x, y, end_of_stroke`` in the 0..255 box, like the official simplified data.
        ``smooth_window`` removes fingertip jitter before RDP (0 or 1 disables it).
        Empty input -> array of shape (0,3)."""
        norm = preprocess.normalize_strokes_input(strokes)
        if len(norm) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        simp = preprocess.simplify_strokes(norm, epsilon=epsilon, smooth_window=smooth_window)
        if len(simp) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # (n_i, 2) arrays are unambiguous for the shared conversion -> (P, 3) with end_of_stroke
        return np.asarray(preprocess.ndjson_strokes_to_points(simp), dtype=np.float32).reshape(-1, 3)

    def strokes_to_tensor(self, strokes):
        """Strokes -> model input. Image model: (1,1,H,W). Stroke model: ((1,T,3), lengths (1,))."""
        points = self.strokes_to_points(strokes, self.epsilon, self.smooth_window)
        if self.mode == "image":
            if len(points) == 0:
                img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                img = preprocess.render_points(points, image_size=self.image_size,
                                               margin=self.margin).astype(np.float32) / 255.0
            return torch.from_numpy(np.ascontiguousarray(img))[None, None]
        # empty / single-point drawing -> points_to_sequence returns a neutral 1-step sequence
        seq = np.asarray(preprocess.points_to_sequence(points, max_len=self.max_len), dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(seq))[None]
        return x, torch.tensor([x.shape[1]], dtype=torch.long)

    # ---------------------------------------------------------------- inference
    def _topk(self, logits: torch.Tensor, top_k: int) -> list:
        if top_k is None or int(top_k) < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        probs = torch.softmax(logits.float().view(-1), dim=0)
        k = min(int(top_k), probs.numel())
        vals, idx = torch.topk(probs, k)
        return [(self.classes[int(i)], float(v)) for v, i in zip(vals.cpu(), idx.cpu())]

    @torch.inference_mode()
    def predict_proba(self, canvas_img=None, strokes=None) -> np.ndarray:
        """Full probability vector (num_classes,) for a canvas (image models) or strokes."""
        if strokes is not None:
            logits = self._logits_strokes(strokes)
        else:
            logits = self._logits_canvas(canvas_img)
        return torch.softmax(logits.float().view(-1), 0).cpu().numpy()

    def _logits_canvas(self, canvas_img):
        if self.mode != "image":
            raise TypeError(
                f"predict() needs an image model, but '{self.checkpoint_path}' is a stroke model "
                f"({self.model_name}). The stroke order and pen-up events cannot be recovered from "
                "an image, so a stroke model can only classify recorded strokes: record the "
                "fingertip trajectory and call predict_strokes(strokes) "
                "(see docs/INTEGRATION_vision.md).")
        x = self.canvas_to_tensor(canvas_img).to(self.device)
        return self.model(x)

    def _logits_strokes(self, strokes):
        if self.mode == "image":
            x = self.strokes_to_tensor(strokes).to(self.device)
            return self.model(x)
        x, lengths = self.strokes_to_tensor(strokes)
        return self.model(x.to(self.device), lengths)  # lengths stay on CPU (packing)

    @torch.inference_mode()
    def predict(self, canvas_img, top_k: int = 3) -> list:
        """Classify an OpenCV canvas (image models). Returns ``[(class, prob), ...]`` sorted desc.

        Accepts (H,W), (H,W,1), (H,W,3) BGR, (H,W,4); uint8 or float; any background colour and
        stroke colour/thickness. A blank canvas returns low-confidence results."""
        return self._topk(self._logits_canvas(canvas_img), top_k)

    @torch.inference_mode()
    def predict_strokes(self, strokes, top_k: int = 3) -> list:
        """Classify recorded strokes. ``strokes`` = list of strokes, each a list/array of (x, y)
        points (canvas pixels), or the official ``[[xs],[ys]]`` format. Works for stroke models
        and image models (the strokes are rendered). Empty input returns low-confidence results."""
        return self._topk(self._logits_strokes(strokes), top_k)


def is_blank(canvas_img, min_ink_pixels: int = 20) -> bool:
    """True if the canvas has (almost) no ink after the shared preprocessing (cheap helper for
    the game to skip predictions on an empty canvas)."""
    img = preprocess.preprocess_canvas(np.asarray(canvas_img), image_size=64)
    return int((img > 0.2).sum()) < min_ink_pixels


# -------------------------------------------------------------------- module-level singletons
_CACHE: dict = {}
_LOCK = threading.Lock()


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def load_classifier(path: str | None = None, device=None) -> QuickDrawClassifier:
    """Return a cached :class:`QuickDrawClassifier` for ``path`` (loaded once per path).
    ``path=None`` -> the CNN default (env ``QUICKDRAW_CNN_CKPT`` or checkpoints/cnn/best.pt)."""
    if path is None:
        path = os.environ.get(ENV_CNN) or DEFAULT_CNN_CKPT
    key = _key(path)
    clf = _CACHE.get(key)
    if clf is not None:
        return clf
    with _LOCK:
        clf = _CACHE.get(key)
        if clf is None:
            if not os.path.isfile(path):
                kind = "bilstm" if "bilstm" in key else ("cnn" if "cnn" in key else "")
                raise _missing_checkpoint_error(path, kind)
            clf = QuickDrawClassifier(path, device=device)
            _CACHE[key] = clf
    return clf


def reset_cache() -> None:
    """Drop the cached classifiers (e.g. after training a new model or changing env vars)."""
    with _LOCK:
        _CACHE.clear()


def _cnn_path() -> str:
    return os.environ.get(ENV_CNN) or DEFAULT_CNN_CKPT


def _bilstm_path() -> str:
    return os.environ.get(ENV_BILSTM) or DEFAULT_BILSTM_CKPT


def predict(canvas_img, top_k: int = 3) -> list:
    """Classify an OpenCV canvas with the default CNN checkpoint (lazy-loaded once)."""
    path = _cnn_path()
    if not os.path.isfile(path):
        raise _missing_checkpoint_error(path, "cnn")
    return load_classifier(path).predict(canvas_img, top_k=top_k)


def predict_strokes(strokes, top_k: int = 3) -> list:
    """Classify recorded strokes with the default BiLSTM checkpoint (lazy-loaded once)."""
    path = _bilstm_path()
    if not os.path.isfile(path):
        raise _missing_checkpoint_error(path, "bilstm")
    return load_classifier(path).predict_strokes(strokes, top_k=top_k)


# -------------------------------------------------------------------- demo
def _main(argv=None) -> list:
    import time

    import cv2

    p = argparse.ArgumentParser(description="Classify a drawing image with a Quick, Draw! model.")
    p.add_argument("--checkpoint", default=None, help="default: env QUICKDRAW_CNN_CKPT or checkpoints/cnn/best.pt")
    p.add_argument("--image", required=True, help="path to a PNG/JPG drawing (any background)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    path = args.checkpoint or _cnn_path()
    if not os.path.isfile(path):
        raise _missing_checkpoint_error(path, "cnn")
    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    clf = QuickDrawClassifier(path, device=args.device)
    clf.predict(img, top_k=args.top_k)  # warm-up
    t0 = time.perf_counter()
    res = clf.predict(img, top_k=args.top_k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{clf}  ({ms:.1f} ms)")
    for i, (name, prob) in enumerate(res, 1):
        print(f"  {i}. {name:<15s} {prob:6.1%}")
    return res


if __name__ == "__main__":
    _main(sys.argv[1:])
