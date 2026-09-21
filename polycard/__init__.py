"""PolyCard -- a learned cardinality estimator for intersection queries
on spatial polygons.

Public API (in pipeline order)::

    polycard.clean          OSM ways cleaning
    polycard.augment        training-query augmentation
    polycard.transform      variable-size polygon -> fixed-size vector
    polycard.featurize      dataset/polygon vectors + label compression
    polycard.ground_truth   offline (Shapely) true cardinalities
    polycard.model          PolyCardNet
    polycard.train          training loop (q-error loss)
    polycard.evaluate       q-error percentiles + latency

The package is derived from the MSCN codebase (Kipf et al., CIDR 2019);
the polygon-specific parts (transformation, augmentation, featurization)
are original.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "PolyCardEstimator",
    "io_utils",
    "clean",
    "augment",
    "transform",
    "featurize",
    "ground_truth",
    "model",
    "dataset",
    "train",
    "evaluate",
    "estimator",
    "metrics",
]

from .estimator import PolyCardEstimator  # noqa: E402
