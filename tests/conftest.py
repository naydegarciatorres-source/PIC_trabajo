"""Shared pytest configuration: make the project root importable and provide fixtures."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def classes():
    from preprocess import load_classes

    return load_classes(os.path.join(ROOT, "classes.json"))


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory, classes):
    """Tiny dataset in the official processed format (random strokes), shared by all tests."""
    from data_loader import make_synthetic_dataset

    out = tmp_path_factory.mktemp("synthetic_processed")
    make_synthetic_dataset(str(out), classes, n_per_class=12, seed=0)
    return str(out)
