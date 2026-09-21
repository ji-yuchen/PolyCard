"""Ground-truth cardinality, computed without a database.

The paper obtains true cardinalities by running the generated queries
through PostGIS (``scripts/collect_cardinality.py --engine postgis``).
That is the authoritative path, but it needs a full PostGIS install and a
115 M-row import for the ``Buildings`` dataset.  This module provides an
equivalent **pure-Python** path built on Shapely's ``STRtree``:

    1. build an R-tree over the dataset polygons (MBR filter);
    2. refine every candidate with a *precise* ``intersects`` test
       (edge crossing **or** full containment -- the paper's definition).

It is fast enough for the smaller datasets (Cemetery / Sports) and is
what the bundled quickstart uses.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .io_utils import wkt_to_coords

__all__ = [
    "HAS_SHAPELY",
    "to_shapely",
    "PolygonIndex",
    "count_intersections",
    "compute_cardinalities",
]


try:  # Shapely is an optional dependency.
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.strtree import STRtree as _STRtree

    HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    _ShapelyPolygon = None
    _STRtree = None
    HAS_SHAPELY = False


def _require_shapely() -> None:
    if not HAS_SHAPELY:
        raise ImportError(
            "Shapely is required for the offline ground truth. "
            "Install it with:  pip install shapely"
        )


def to_shapely(coords):
    """Convert an ``(m, 2)`` array or a WKT string to a Shapely polygon.

    Invalid rings (self-intersecting, < 3 distinct vertices) are repaired
    with ``buffer(0)``; unrepairable ones return ``None``.
    """
    _require_shapely()
    if isinstance(coords, str):
        coords = wkt_to_coords(coords)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 3:
        return None
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[:1]])
    poly = _ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


class PolygonIndex:
    """STRtree-backed index over a set of dataset polygons."""

    def __init__(self, polygons: Sequence):
        """``polygons`` is a sequence of ``(m, 2)`` arrays or WKT strings."""
        _require_shapely()
        self.geoms: List = []
        for p in polygons:
            g = to_shapely(p)
            if g is not None:
                self.geoms.append(g)
        self.tree = _STRtree(self.geoms)

    def __len__(self) -> int:
        return len(self.geoms)

    def count(self, query) -> int:
        """Number of indexed polygons intersecting ``query``.

        The R-tree first filters on bounding boxes, then the predicate
        ``intersects`` re-checks each candidate exactly (edge crossing
        **or** containment -- the paper's definition of polygon
        intersection).  Shapely >= 2.0 performs this refinement inside
        GEOS and returns the matching *indices*.
        """
        q = to_shapely(query)
        if q is None:
            return 0
        try:
            hits = self.tree.query(q, predicate="intersects")
        except TypeError:
            # Shapely 1.x: query() yields geometries and has no predicate.
            hits = [g for g in self.tree.query(q) if g.intersects(q)]
        return int(np.size(hits))

    def count_batch(self, queries: Iterable, log_every: int = 0, log=print) -> np.ndarray:
        """Cardinalities for many queries -> ``(N,)`` int array."""
        out = []
        for i, q in enumerate(queries):
            out.append(self.count(q))
            if log_every and (i + 1) % log_every == 0:
                log(f"[ground-truth] {i + 1} queries done")
        return np.asarray(out, dtype=np.int64)


def count_intersections(dataset_polygons: Sequence, query) -> int:
    """One-shot helper: build an index and count intersections."""
    return PolygonIndex(dataset_polygons).count(query)


def compute_cardinalities(
    dataset_polygons: Sequence,
    query_polygons: Sequence,
    log_every: int = 0,
    log=print,
) -> np.ndarray:
    """True cardinalities of ``query_polygons`` against ``dataset_polygons``."""
    _require_shapely()
    index = PolygonIndex(dataset_polygons)
    log(f"[ground-truth] indexed {len(index)} dataset polygons")
    return index.count_batch(query_polygons, log_every=log_every, log=log)
