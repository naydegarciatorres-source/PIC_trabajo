"""PyTorch data loading for the processed Quick, Draw! dataset (CONTRACT section 3).

Arrays written by ``preprocess.py`` are opened with ``np.load(mmap_mode="r")`` lazily inside each
process (memory maps are never pickled), so DataLoader workers on Windows (spawn) stay cheap.
Images / sequences are built on the fly with the shared transforms of ``preprocess.py``; data
augmentation is applied to the stroke points (never to pixels).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess import (
    SPLITS,
    ProcessedDatasetWriter,
    _default_thickness,
    _render,
    assign_split,
    ndjson_strokes_to_points,
    points_to_sequence,
)

__all__ = [
    "QuickDrawDataset",
    "collate_strokes",
    "make_dataloaders",
    "make_synthetic_dataset",
    "load_meta",
    "augment_points_image",
    "augment_points_strokes",
]


def load_meta(data_dir: str) -> dict:
    """Read ``meta.json`` of a processed dataset directory."""
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------
# Augmentation (on points, before rendering / sequence building)
# ----------------------------------------------------------------------------------------------
def _affine_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a 2x2 linear map to x, y around the drawing centre; keeps the eos column."""
    out = points.astype(np.float32, copy=True)
    xy = out[:, :2]
    c = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    out[:, :2] = (xy - c) @ mat.T.astype(np.float32) + c
    return out


def augment_points_image(
    points: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 12.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Random small rotation + anisotropic scale of ``(P, 3)`` points (image mode)."""
    if len(points) == 0:
        return points
    a = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    sc = np.diag(rng.uniform(scale_range[0], scale_range[1], size=2))
    return _affine_points(points, rot @ sc)


def augment_points_strokes(
    points: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.9, 1.1),
    max_shear: float = 0.15,
    max_rotation_deg: float = 8.0,
) -> np.ndarray:
    """Random small scale/shear (+ slight rotation) of x, y before sequence normalisation.

    Note: ``points_to_sequence`` normalises each axis to [0, 1], so a pure per-axis scale mostly
    cancels out; shear and rotation are what actually change the sequence.
    """
    if len(points) == 0:
        return points
    sc = np.diag(rng.uniform(scale_range[0], scale_range[1], size=2))
    sh = np.array([[1.0, rng.uniform(-max_shear, max_shear)], [rng.uniform(-max_shear, max_shear), 1.0]])
    a = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return _affine_points(points, rot @ sh @ sc)


# ----------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------
def _subsample_indices(labels: np.ndarray, max_per_class: int | None, seed: int) -> np.ndarray:
    """Indices of the drawings kept: all, or at most ``max_per_class`` per class chosen
    deterministically (independent per class, sorted)."""
    n = len(labels)
    if max_per_class is None or n == 0:
        return np.arange(n, dtype=np.int64)
    max_per_class = int(max_per_class)
    labels = np.asarray(labels).astype(np.int64)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    uniq, starts, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
    keep = []
    for c, s, k in zip(uniq, starts, counts):
        idx = order[s : s + k]
        if k > max_per_class:
            rng = np.random.default_rng([int(seed), int(c)])
            idx = np.sort(rng.choice(idx, size=max_per_class, replace=False))
        keep.append(idx)
    return np.sort(np.concatenate(keep)).astype(np.int64) if keep else np.zeros(0, dtype=np.int64)


class QuickDrawDataset(Dataset):
    """One split of a processed dataset.

    Args:
        data_dir: directory written by ``preprocess.py`` (or :func:`make_synthetic_dataset`).
        split: ``"train"``, ``"val"`` or ``"test"``.
        mode: ``"image"`` -> ``(float32 tensor (1, H, W) in [0, 1], label)``;
            ``"strokes"`` -> ``(float32 tensor (T, 3), label)``.
        image_size, max_len: see ``render_points`` / ``points_to_sequence``.
        augment: random point-level augmentation (rotation/scale/shift/thickness for images,
            scale/shear for strokes).
        max_per_class: keep at most this many drawings per class (deterministic given ``seed``).
        seed: seed for the subsampling.

    Attributes: ``classes``, ``num_classes``, ``indices`` (kept drawing indices), ``labels``
    (``np.int64`` labels of the kept drawings).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        mode: str = "image",
        image_size: int = 64,
        max_len: int = 250,
        augment: bool = False,
        max_per_class: int | None = None,
        seed: int = 0,
        margin: float = 0.1,
    ):
        if mode not in ("image", "strokes"):
            raise ValueError(f"mode must be 'image' or 'strokes', got {mode!r}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.image_size = int(image_size)
        self.max_len = int(max_len)
        self.augment = bool(augment)
        self.max_per_class = max_per_class
        self.seed = int(seed)
        self.margin = float(margin)
        self.meta = load_meta(data_dir)
        self.classes: list[str] = list(self.meta["classes"])
        self.num_classes = len(self.classes)
        self._paths = {k: os.path.join(data_dir, f"{split}_{k}.npy") for k in ("points", "offsets", "labels", "key_ids")}
        for k in ("points", "offsets", "labels"):
            if not os.path.exists(self._paths[k]):
                raise FileNotFoundError(self._paths[k])
        all_labels = np.load(self._paths["labels"], mmap_mode="r")
        self.indices = _subsample_indices(np.asarray(all_labels), max_per_class, self.seed)
        self.labels = np.asarray(all_labels[self.indices], dtype=np.int64)
        del all_labels
        self._points = None
        self._offsets = None
        self._rng: np.random.Generator | None = None
        self._rng_pid: int | None = None

    # -- pickling: never ship memory maps / RNGs to workers ---------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_points"] = None
        state["_offsets"] = None
        state["_rng"] = None
        state["_rng_pid"] = None
        return state

    def _open(self) -> None:
        if self._points is None:
            self._points = np.load(self._paths["points"], mmap_mode="r")
            self._offsets = np.load(self._paths["offsets"], mmap_mode="r")

    def _get_rng(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            # torch.initial_seed() differs per worker (and per epoch for non-persistent workers)
            self._rng = np.random.default_rng([torch.initial_seed() % (2**63), pid, self.seed])
            self._rng_pid = pid
        return self._rng

    def __len__(self) -> int:
        return len(self.indices)

    def get_points(self, i: int) -> np.ndarray:
        """Raw ``(P, 3)`` float32 points of the i-th kept drawing (no augmentation)."""
        self._open()
        j = int(self.indices[i])
        a, b = int(self._offsets[j]), int(self._offsets[j + 1])
        return np.asarray(self._points[a:b], dtype=np.float32)

    def key_ids(self) -> np.ndarray:
        """Official key_ids of the kept drawings."""
        return np.asarray(np.load(self._paths["key_ids"], mmap_mode="r")[self.indices], dtype=np.int64)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        pts = self.get_points(i)
        label = int(self.labels[i])
        if self.mode == "image":
            thickness = _default_thickness(self.image_size)
            offset = (0.0, 0.0)
            if self.augment:
                rng = self._get_rng()
                pts = augment_points_image(pts, rng)
                thickness = max(1, thickness + int(rng.integers(-1, 2)))
                max_shift = self.image_size * self.margin * 0.5
                offset = tuple(rng.uniform(-max_shift, max_shift, size=2))
            img = _render(pts, image_size=self.image_size, thickness=thickness, margin=self.margin, offset=offset)
            x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            if self.augment:
                pts = augment_points_strokes(pts, self._get_rng())
            x = torch.from_numpy(points_to_sequence(pts, self.max_len))
        return x, label


# ----------------------------------------------------------------------------------------------
# Collate / loaders
# ----------------------------------------------------------------------------------------------
def collate_strokes(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length ``(T_i, 3)`` sequences with zeros.

    Returns ``(padded float32 (B, Tmax, 3), lengths int64 (B,), labels int64 (B,))``.
    """
    seqs = [torch.as_tensor(s, dtype=torch.float32) for s, _ in batch]
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int64)
    labels = torch.tensor([int(l) for _, l in batch], dtype=torch.int64)
    tmax = int(lengths.max()) if len(seqs) else 0
    feat = seqs[0].shape[1] if seqs else 3
    padded = torch.zeros((len(seqs), tmax, feat), dtype=torch.float32)
    for k, s in enumerate(seqs):
        padded[k, : s.shape[0]] = s
    return padded, lengths, labels


def _split_exists(data_dir: str, split: str) -> bool:
    return all(os.path.exists(os.path.join(data_dir, f"{split}_{k}.npy")) for k in ("points", "offsets", "labels"))


def make_dataloaders(
    data_dir: str,
    mode: str,
    batch_size: int = 256,
    num_workers: int = 2,
    image_size: int = 64,
    max_len: int = 250,
    augment_train: bool = True,
    max_per_class: int | None = None,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """Build ``{"train", "val", "test"}`` DataLoaders (a missing split -> key absent).

    ``max_per_class`` caps the train split; val/test are capped proportionally to the split
    ratios in ``meta.json`` (e.g. 1000 -> 125 with 80/10/10) so quick experiments stay quick.
    Train is shuffled with a ``seed``-ed generator; strokes mode uses :func:`collate_strokes`.
    ``persistent_workers`` only when ``num_workers > 0``; ``pin_memory`` when CUDA is available.
    """
    meta = load_meta(data_dir)
    ratios = meta.get("split_ratios", [0.8, 0.1, 0.1])
    ratio_map = dict(zip(SPLITS, ratios))
    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        if not _split_exists(data_dir, split):
            continue
        cap = None
        if max_per_class is not None:
            cap = int(max_per_class) if split == "train" else max(1, int(math.ceil(max_per_class * ratio_map[split] / ratio_map["train"])))
        ds = QuickDrawDataset(
            data_dir, split, mode=mode, image_size=image_size, max_len=max_len,
            augment=(augment_train and split == "train"), max_per_class=cap, seed=seed,
        )
        is_train = split == "train"
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_strokes if mode == "strokes" else None,
            generator=gen if is_train else None,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        loaders[split] = DataLoader(ds, **kwargs)
    return loaders


# ----------------------------------------------------------------------------------------------
# Synthetic data for tests
# ----------------------------------------------------------------------------------------------
def _synthetic_drawing(label: int, rng: np.random.Generator) -> list:
    """Random but class-dependent drawing in the official format (coords 0..255 ints)."""
    n_vertices = 3 + label % 6          # polygon shape depends on the class
    n_strokes = 1 + (label // 6) % 3    # extra strokes depend on the class
    phase = (label * 0.7) % (2 * math.pi)
    cx, cy = rng.uniform(100, 155, size=2)
    r = rng.uniform(60, 95)
    angles = phase + np.linspace(0, 2 * math.pi, n_vertices + 1) + rng.normal(0, 0.05, n_vertices + 1)
    xs = cx + r * np.cos(angles) + rng.normal(0, 3, n_vertices + 1)
    ys = cy + r * np.sin(angles) + rng.normal(0, 3, n_vertices + 1)
    strokes = [[xs, ys]]
    for k in range(1, n_strokes):
        a = phase + k * 1.3
        m = 2 + int(rng.integers(0, 4))
        t = np.linspace(-1, 1, m)
        strokes.append([cx + 0.6 * r * t * math.cos(a) + rng.normal(0, 2, m),
                        cy + 0.6 * r * t * math.sin(a) + rng.normal(0, 2, m)])
    return [[np.clip(np.round(x), 0, 255).astype(int).tolist(), np.clip(np.round(y), 0, 255).astype(int).tolist()]
            for x, y in strokes]


def make_synthetic_dataset(out_dir: str, classes: Sequence[str], n_per_class: int = 20, seed: int = 0) -> dict:
    """Write a tiny but valid processed dataset (CONTRACT section 1) with random strokes.

    Each class gets ``n_per_class`` drawings of a class-specific shape with random jitter and a
    unique synthetic ``key_id``; splits use the same ``assign_split`` as the real builder.
    Returns the meta dict.
    """
    rng = np.random.default_rng(seed)
    writer = ProcessedDatasetWriter(out_dir, classes)
    base = 4_000_000_000_000_000 + int(seed) * 10_000_000
    for label in range(len(classes)):
        for i in range(n_per_class):
            key_id = base + label * 100_000 + i
            pts = ndjson_strokes_to_points(_synthetic_drawing(label, rng))
            writer.add(assign_split(key_id, seed), pts, label, key_id)
    return writer.finalize({"per_class": int(n_per_class), "seed": int(seed), "source": "synthetic"})
