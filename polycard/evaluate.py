"""Inference and accuracy / latency measurement.

``evaluate_model`` reports both sides of the trade-off studied in the
paper: the q-error percentiles of :mod:`polycard.metrics` and the *per
query* estimation latency in microseconds (measured on batches, with
``torch.no_grad`` and ``model.eval()``).
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PolyCardDataset
from .featurize import LABEL_EPS
from .metrics import summarize_qerror
from .model import PolyCardNet

__all__ = ["predict", "evaluate_model", "load_model"]


def load_model(checkpoint_dir: str, device: Optional[str] = None) -> tuple:
    """Restore ``(model, label_stats)`` written by ``scripts/train.py``."""
    import os

    from .dataset import load_label_stats

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_label_stats(os.path.join(checkpoint_dir, "label_stats.json"))
    model = PolyCardNet(
        dataset_feats=stats["dataset_feats"],
        polygon_feats=stats["polygon_feats"],
        hid_units=stats["hid_units"],
    )
    state = torch.load(os.path.join(checkpoint_dir, "polycard.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model, stats


@torch.no_grad()
def predict(
    model: PolyCardNet,
    dataset: PolyCardDataset,
    device: Optional[str] = None,
    batch_size: int = 1024,
    warmup: int = 1,
):
    """Run inference; return ``(predictions_normalized, seconds_total)``.

    Timing covers the forward pass only (featurization is shared with the
    baselines and is reported separately by ``scripts/evaluate.py``).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.eval()
    model.to(device)

    # Build the loader directly: make_loaders() returns ``None`` for the
    # second loader when no test set is given.
    loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(dataset.dataset_vec),
            torch.from_numpy(dataset.polygon_vec),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    preds = []
    total = 0.0
    # One untimed warm-up pass: the first CUDA kernel launch includes
    # one-off initialization that would otherwise dominate a microsecond
    # measurement.
    for w in range(warmup):
        for dataset_vec, polygon_vec in loader:
            _ = model(dataset_vec.to(device), polygon_vec.to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for dataset_vec, polygon_vec in loader:
        dataset_vec = dataset_vec.to(device)
        polygon_vec = polygon_vec.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(dataset_vec, polygon_vec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total += time.perf_counter() - t0
        preds.append(out.detach().cpu().numpy().reshape(-1))

    return np.concatenate(preds) if preds else np.empty(0), total


def evaluate_model(
    model: PolyCardNet,
    dataset: PolyCardDataset,
    min_val: float,
    max_val: float,
    label_eps: float = LABEL_EPS,
    device: Optional[str] = None,
    batch_size: int = 1024,
) -> Dict:
    """Full evaluation: q-error percentiles + per-query latency."""
    preds_norm, seconds = predict(model, dataset, device=device, batch_size=batch_size)
    preds = np.exp(preds_norm * (max_val - min_val) + min_val) - label_eps
    truths = np.exp(
        np.asarray(dataset.labels, dtype=np.float64) * (max_val - min_val) + min_val
    ) - label_eps

    summary = summarize_qerror(preds, truths)
    n = max(len(dataset), 1)
    summary["est_time_us"] = seconds / n * 1e6
    summary["est_time_total_s"] = seconds
    summary["num_samples"] = n
    return summary
