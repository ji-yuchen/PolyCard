"""PyTorch dataset / loader plumbing for PolyCard."""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .io_utils import ensure_dir, load_array, save_array

__all__ = [
    "FEATURE_FILES",
    "PolyCardDataset",
    "make_tensor_dataset",
    "make_loaders",
    "save_feature_bundle",
    "load_feature_bundle",
    "save_label_stats",
    "load_label_stats",
]


FEATURE_FILES = {
    "polygon": "polygon_vec.npy",
    "dataset": "dataset_vec.npy",
    "labels": "labels_norm.npy",
    "cardinality": "cardinality.npy",
    "meta": "meta.json",
}


class PolyCardDataset(Dataset):
    """In-memory dataset of ``(dataset_vec, polygon_vec, label)`` triples."""

    def __init__(self, dataset_vec: np.ndarray, polygon_vec: np.ndarray, labels: np.ndarray):
        self.dataset_vec = np.asarray(dataset_vec, dtype=np.float32)
        self.polygon_vec = np.asarray(polygon_vec, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        if not (len(self.dataset_vec) == len(self.polygon_vec) == len(self.labels)):
            raise ValueError(
                "length mismatch: dataset_vec=%d polygon_vec=%d labels=%d"
                % (len(self.dataset_vec), len(self.polygon_vec), len(self.labels))
            )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.dataset_vec[idx], self.polygon_vec[idx], self.labels[idx]

    @property
    def dataset_feats(self) -> int:
        return self.dataset_vec.shape[1]

    @property
    def polygon_feats(self) -> int:
        return self.polygon_vec.shape[1]


def make_tensor_dataset(ds: PolyCardDataset) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(ds.dataset_vec),
        torch.from_numpy(ds.polygon_vec),
        torch.from_numpy(ds.labels),
    )


def make_loaders(
    train: PolyCardDataset,
    test: Optional[PolyCardDataset] = None,
    batch_size: int = 128,
    eval_batch_size: int = 1024,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build training / evaluation loaders.

    The training loader shuffles; the evaluation loader does not, so that
    predictions stay aligned with the on-disk query order.
    """
    train_loader = DataLoader(
        make_tensor_dataset(train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = None
    if test is not None:
        test_loader = DataLoader(
            make_tensor_dataset(test),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return train_loader, test_loader


# ----------------------------------------------------------------------
# on-disk feature bundles
# ----------------------------------------------------------------------
def save_feature_bundle(
    out_dir: str,
    polygon_vec: np.ndarray,
    dataset_vec: np.ndarray,
    labels_norm: np.ndarray,
    cardinality: Optional[np.ndarray] = None,
    meta: Optional[dict] = None,
) -> None:
    """Persist a featurized dataset split (used by ``build_training_set.py``)."""
    ensure_dir(out_dir)
    save_array(os.path.join(out_dir, FEATURE_FILES["polygon"]), polygon_vec)
    save_array(os.path.join(out_dir, FEATURE_FILES["dataset"]), dataset_vec)
    save_array(os.path.join(out_dir, FEATURE_FILES["labels"]), labels_norm)
    if cardinality is not None:
        save_array(os.path.join(out_dir, FEATURE_FILES["cardinality"]), cardinality)
    if meta is not None:
        with open(os.path.join(out_dir, FEATURE_FILES["meta"]), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)


def load_feature_bundle(path: str) -> Tuple[PolyCardDataset, dict, Optional[np.ndarray]]:
    """Load a bundle written by :func:`save_feature_bundle`."""
    polygon_vec = load_array(os.path.join(path, FEATURE_FILES["polygon"]))
    dataset_vec = load_array(os.path.join(path, FEATURE_FILES["dataset"]))
    labels_norm = load_array(os.path.join(path, FEATURE_FILES["labels"]))
    card_path = os.path.join(path, FEATURE_FILES["cardinality"])
    cardinality = load_array(card_path) if os.path.exists(card_path) else None
    meta_path = os.path.join(path, FEATURE_FILES["meta"])
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return PolyCardDataset(dataset_vec, polygon_vec, labels_norm), meta, cardinality


def save_label_stats(path: str, stats: dict) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)


def load_label_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
