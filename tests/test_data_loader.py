"""Tests for data_loader.py using the synthetic dataset (no network)."""
import json
import os
import pickle

import numpy as np
import pytest
import torch

import data_loader as D
from preprocess import SPLITS, verify_processed_dataset


def _load(data_dir, split, name, mmap=False):
    return np.load(os.path.join(data_dir, f"{split}_{name}.npy"), mmap_mode="r" if mmap else None)


def test_synthetic_dataset_format(synthetic_data_dir, classes):
    meta = D.load_meta(synthetic_data_dir)
    assert meta["classes"] == classes
    assert set(meta) >= {"classes", "counts", "source", "per_class", "split_ratios", "seed"}
    total = 0
    for s in SPLITS:
        pts = _load(synthetic_data_dir, s, "points", mmap=True)
        off = _load(synthetic_data_dir, s, "offsets")
        lab = _load(synthetic_data_dir, s, "labels")
        kid = _load(synthetic_data_dir, s, "key_ids")
        assert pts.dtype == np.uint8 and pts.ndim == 2 and pts.shape[1] == 3
        assert off.dtype == np.int64 and lab.dtype == np.int16 and kid.dtype == np.int64
        assert len(off) == len(lab) + 1 == len(kid) + 1
        assert off[0] == 0 and off[-1] == len(pts) and np.all(np.diff(off) > 0)
        assert set(np.unique(pts[:, 2]).tolist()) <= {0, 1}
        # last point of every drawing ends a stroke
        assert np.all(pts[off[1:] - 1, 2] == 1)
        assert sum(meta["counts"][s].values()) == len(lab) > 0
        for c, n in meta["counts"][s].items():
            assert int((lab == classes.index(c)).sum()) == n
        total += len(lab)
    assert total == 12 * len(classes)


def test_no_key_id_overlap_between_splits(synthetic_data_dir):
    keys = {s: set(_load(synthetic_data_dir, s, "key_ids").tolist()) for s in SPLITS}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    verify_processed_dataset(synthetic_data_dir)


def test_verify_detects_leakage(tmp_path, classes):
    out = tmp_path / "leaky"
    D.make_synthetic_dataset(str(out), classes[:3], n_per_class=10, seed=1)
    tr = np.load(out / "train_key_ids.npy")
    va = np.load(out / "val_key_ids.npy")
    va[0] = tr[0]
    np.save(out / "val_key_ids.npy", va)
    with pytest.raises(RuntimeError):
        verify_processed_dataset(str(out))


def test_dataset_image_mode(synthetic_data_dir, classes):
    for augment in (False, True):
        ds = D.QuickDrawDataset(synthetic_data_dir, "train", mode="image", image_size=48, augment=augment)
        assert ds.num_classes == len(classes) and ds.classes == classes
        x, y = ds[0]
        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.shape == (1, 48, 48)
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0 and float(x.max()) > 0.5
        assert isinstance(y, int) and 0 <= y < len(classes)


def test_dataset_strokes_mode(synthetic_data_dir):
    for augment in (False, True):
        ds = D.QuickDrawDataset(synthetic_data_dir, "val", mode="strokes", max_len=5, augment=augment)
        x, y = ds[1]
        assert x.dtype == torch.float32 and x.ndim == 2 and x.shape[1] == 3
        assert 1 <= x.shape[0] <= 5
        assert set(torch.unique(x[:, 2]).tolist()) <= {0.0, 1.0}
        assert torch.isfinite(x).all()


def test_dataset_label_matches_stored_label(synthetic_data_dir):
    ds = D.QuickDrawDataset(synthetic_data_dir, "test", mode="strokes")
    labels = _load(synthetic_data_dir, "test", "labels")
    for i in range(0, len(ds), max(1, len(ds) // 10)):
        assert ds[i][1] == int(labels[ds.indices[i]])


def test_label_class_mapping_consistent_with_classes_json(synthetic_data_dir, project_root):
    with open(os.path.join(project_root, "classes.json"), encoding="utf-8") as f:
        classes_json = json.load(f)["classes"]
    meta = D.load_meta(synthetic_data_dir)
    assert meta["classes"] == classes_json
    ds = D.QuickDrawDataset(synthetic_data_dir, "train", mode="image")
    assert ds.classes == classes_json
    for c, n in meta["counts"]["train"].items():
        assert int((ds.labels == classes_json.index(c)).sum()) == n


def test_collate_strokes_padding():
    batch = [(torch.ones(3, 3), 1), (torch.full((5, 3), 2.0), 0), (torch.ones(1, 3), 4)]
    padded, lengths, labels = D.collate_strokes(batch)
    assert padded.shape == (3, 5, 3) and padded.dtype == torch.float32
    assert lengths.tolist() == [3, 5, 1] and lengths.dtype == torch.int64
    assert labels.tolist() == [1, 0, 4] and labels.dtype == torch.int64
    assert padded[0, 3:].abs().sum() == 0 and padded[2, 1:].abs().sum() == 0
    assert torch.all(padded[1] == 2.0)


@pytest.mark.parametrize("mode", ["image", "strokes"])
def test_make_dataloaders(synthetic_data_dir, classes, mode):
    loaders = D.make_dataloaders(synthetic_data_dir, mode, batch_size=16, num_workers=0, image_size=32, max_len=50)
    assert set(loaders) == set(SPLITS)
    batch = next(iter(loaders["train"]))
    if mode == "image":
        x, y = batch
        assert x.shape[1:] == (1, 32, 32) and x.dtype == torch.float32
    else:
        x, lengths, y = batch
        assert x.ndim == 3 and x.shape[2] == 3 and x.shape[1] == int(lengths.max()) <= 50
        assert lengths.dtype == torch.int64
    assert y.dtype == torch.int64 and x.shape[0] == y.shape[0] <= 16
    assert int(y.max()) < len(classes)
    n = sum(len(b[-1]) for b in loaders["val"])
    assert n == len(loaders["val"].dataset)


def test_make_dataloaders_missing_split(tmp_path, classes):
    out = tmp_path / "nosplit"
    D.make_synthetic_dataset(str(out), classes[:2], n_per_class=10, seed=0)
    for k in ("points", "offsets", "labels", "key_ids"):
        os.remove(out / f"test_{k}.npy")
    loaders = D.make_dataloaders(str(out), "image", batch_size=4, num_workers=0)
    assert set(loaders) == {"train", "val"}


def test_max_per_class_deterministic(synthetic_data_dir, classes):
    a = D.QuickDrawDataset(synthetic_data_dir, "train", max_per_class=3, seed=5)
    b = D.QuickDrawDataset(synthetic_data_dir, "train", max_per_class=3, seed=5)
    c = D.QuickDrawDataset(synthetic_data_dir, "train", max_per_class=3, seed=6)
    full = D.QuickDrawDataset(synthetic_data_dir, "train")
    np.testing.assert_array_equal(a.indices, b.indices)
    assert not np.array_equal(a.indices, c.indices)
    counts = np.bincount(a.labels, minlength=len(classes))
    full_counts = np.bincount(full.labels, minlength=len(classes))
    np.testing.assert_array_equal(counts, np.minimum(full_counts, 3))
    loaders = D.make_dataloaders(synthetic_data_dir, "image", batch_size=8, num_workers=0, max_per_class=3)
    assert len(loaders["train"].dataset) == len(a)
    assert np.bincount(loaders["val"].dataset.labels).max() <= 1  # ceil(3 * 0.1 / 0.8)


def test_dataset_is_picklable_without_mmaps(synthetic_data_dir):
    for augment in (False, True):
        ds = D.QuickDrawDataset(synthetic_data_dir, "train", mode="strokes", augment=augment)
        _ = ds[0]  # opens the memory maps / RNG
        blob = pickle.dumps(ds)
        assert len(blob) < 100_000
        ds2 = pickle.loads(blob)
        assert ds2._points is None and ds2._rng is None
        assert ds2[3][1] == ds[3][1]
        if not augment:
            torch.testing.assert_close(ds2[3][0], ds[3][0])


def test_synthetic_classes_are_distinguishable(synthetic_data_dir):
    """Class-dependent strokes: mean images of two classes should differ."""
    ds = D.QuickDrawDataset(synthetic_data_dir, "train", mode="image", image_size=32)
    means = {}
    for i in range(len(ds)):
        x, y = ds[i]
        if y in (0, 3):
            means.setdefault(y, []).append(x.numpy())
    m0, m3 = (np.mean(means[k], axis=0) for k in (0, 3))
    assert np.abs(m0 - m3).mean() > 0.01
