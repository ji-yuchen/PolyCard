"""Featurization of queries and labels (Section 4.5 of the paper).

An intersection query is encoded as *two* vectors:

* a **dataset vector** -- one-hot of length ``M`` (number of source
  datasets), telling the network which dataset is being queried;
* a **polygon vector** -- the transformed, normalized query polygon,
  flattened to ``2n`` coordinates.

Cardinalities are log-compressed and min-max normalized before training,
because they span several orders of magnitude.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .transform import transform_polygons

__all__ = [
    "LABEL_EPS",
    "dataset_onehot",
    "dataset_matrix",
    "featurize_polygons",
    "normalize_labels",
    "unnormalize_labels",
]


# ``log(card + LABEL_EPS)`` keeps zero cardinalities finite.  1.02 matches
# the value used in the original experiments.
LABEL_EPS = 1.02


def dataset_onehot(index: int, num_datasets: int) -> np.ndarray:
    """One-hot vector of length ``num_datasets``."""
    vec = np.zeros(num_datasets, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def dataset_matrix(indices, num_datasets: int) -> np.ndarray:
    """Stack :func:`dataset_onehot` over ``indices`` -> ``(N, M)``."""
    indices = np.asarray(indices, dtype=np.int64)
    out = np.zeros((indices.shape[0], num_datasets), dtype=np.float32)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def featurize_polygons(
    polygons,
    num_vertices: int = 10,
    method: str = "adaptive",
    origin=None,
    span=None,
    dtype=np.float32,
) -> np.ndarray:
    """Fixed-size polygon vectors -> ``(N, 2 * num_vertices)``.

    ``origin`` / ``span`` override the fixed global lon/lat bounds; pass
    dataset-specific values when the data covers only a small region (see
    ``build_training_set.py --norm-from-data``).
    """
    return transform_polygons(
        polygons,
        num_vertices=num_vertices,
        method=method,
        origin=origin,
        span=span,
        dtype=dtype,
    )


def normalize_labels(
    labels,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    eps: float = LABEL_EPS,
):
    """``card -> log(card + eps) -> [0, 1]``.

    Returns ``(labels_norm, min_val, max_val)``.  Pass the returned
    bounds to :func:`unnormalize_labels` at inference time.
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    logged = np.log(labels + eps)
    if min_val is None:
        min_val = float(logged.min())
    if max_val is None:
        max_val = float(logged.max())
    denom = max_val - min_val
    if denom <= 0:
        denom = 1.0
    normed = np.clip((logged - min_val) / denom, 0.0, 1.0)
    return normed.astype(np.float32), float(min_val), float(max_val)


def unnormalize_labels(
    labels_norm,
    min_val: float,
    max_val: float,
    eps: float = LABEL_EPS,
) -> np.ndarray:
    """Inverse of :func:`normalize_labels` -> estimated cardinality."""
    labels_norm = np.asarray(labels_norm, dtype=np.float64)
    logged = labels_norm * (max_val - min_val) + min_val
    return np.exp(logged) - eps
