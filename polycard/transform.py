"""Polygon transformation: variable-sized polygons -> fixed-size vectors.

This module implements **Algorithm 1 (Polygon-Transformation)** of the
PolyCard paper, plus the two degenerate alternatives used as baselines in
the ablation study:

``adaptive``          the proposed zone-aware adaptive sampling (default)
``uniform_sample``    stride sampling over the vertex list
``mbr``               minimum bounding rectangle approximation
``interp``            uniform interpolation only

After resampling, coordinates are min-max normalized with the fixed
global bounds used in the paper (lon ``[-177, 180]``, lat ``[-55, 75]``),
so that the network always sees values in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .io_utils import wkt_to_coords

__all__ = [
    "NORM_ORIGIN",
    "NORM_SPAN",
    "METHODS",
    "check_zone",
    "mbr_center",
    "mbr",
    "uniform_resample",
    "adaptive_resample",
    "uniform_stride_sample",
    "mbr_approximate",
    "normalize",
    "transform_polygon",
    "transform_polygons",
]


# Fixed normalization bounds (degrees).  These cover every OSM landuse
# polygon on earth: longitude in [-180, 180], latitude in [-55, 75].
# Kept identical to the original experiments so results stay comparable.
NORM_ORIGIN = np.array([-177.0, -55.0], dtype=np.float64)
NORM_SPAN = np.array([357.0, 130.0], dtype=np.float64)

METHODS = ("adaptive", "uniform_sample", "mbr", "interp")


# ----------------------------------------------------------------------
# geometric helpers
# ----------------------------------------------------------------------
def check_zone(center_x: float, center_y: float, x: float, y: float) -> int:
    """Index of the MBR quadrant containing ``(x, y)``.

    Zones follow Algorithm 1: bottom-left = 0, top-left = 1,
    top-right = 2, bottom-right = 3.
    """
    if x < center_x:
        return 0 if y < center_y else 1
    return 3 if y < center_y else 2


def mbr_center(coords: np.ndarray) -> Tuple[float, float]:
    """Center of the minimum bounding rectangle of ``coords``."""
    coords = np.asarray(coords, dtype=np.float64)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    return (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0


def mbr(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(min_xy, max_xy)`` of the minimum bounding rectangle."""
    coords = np.asarray(coords, dtype=np.float64)
    return coords.min(axis=0), coords.max(axis=0)


# ----------------------------------------------------------------------
# resampling strategies
# ----------------------------------------------------------------------
def uniform_resample(coords: np.ndarray, n: int) -> np.ndarray:
    """Resample to exactly ``n`` vertices by linear interpolation.

    Used when the polygon has *fewer* vertices than the target size
    (the second branch of Algorithm 1).
    """
    coords = np.asarray(coords, dtype=np.float64)
    m = coords.shape[0]
    if m == n:
        return coords.copy()
    t = np.linspace(0.0, float(m - 1), n)
    out = np.empty((n, 2), dtype=np.float64)
    for d in range(2):
        out[:, d] = np.interp(t, np.arange(m), coords[:, d])
    return out


def adaptive_resample(coords: np.ndarray, n: int) -> np.ndarray:
    """Zone-aware adaptive sampling (Algorithm 1, ``n < m`` branch).

    The MBR is split into four quadrants by its mid-point.  Vertices are
    scanned sequentially and sampled on a uniform budget of ``m / n``,
    but the *first* vertex seen in each non-empty quadrant is always
    kept.  This preserves thin, far-away parts of a polygon that plain
    stride sampling would drop -- see Fig. 5 of the paper.
    """
    coords = np.asarray(coords, dtype=np.float64)
    m = coords.shape[0]
    if m <= n:
        return uniform_resample(coords, n)
    if n < 4:
        raise ValueError(
            "adaptive sampling needs n >= 4 (four MBR zones); got n=%d" % n
        )

    cx, cy = mbr_center(coords)
    interval = float(m) / float(n)
    zone_seen = np.zeros(4, dtype=bool)
    count = 0.0
    picked: list[int] = []

    for i in range(m):
        count += 1.0
        zone = check_zone(cx, cy, coords[i, 0], coords[i, 1])
        if not zone_seen[zone]:
            picked.append(i)
            count -= interval
            zone_seen[zone] = True
        elif count >= interval:
            picked.append(i)
            count -= interval
        if len(picked) == n:
            break

    out = coords[np.asarray(picked, dtype=np.int64)]

    # Safety net: with pathological vertex distributions the quadrant
    # guarantee can consume the budget before ``n`` samples are drawn.
    # Pad by repeating the last picked vertex so the tensor shape stays
    # fixed.  This never triggers for the datasets used in the paper.
    if out.shape[0] < n:
        pad = np.repeat(out[-1:], n - out.shape[0], axis=0)
        out = np.vstack([out, pad])
    return out


def uniform_stride_sample(coords: np.ndarray, n: int) -> np.ndarray:
    """Plain stride sampling (``uniform sampling`` baseline, Fig. 8)."""
    coords = np.asarray(coords, dtype=np.float64)
    m = coords.shape[0]
    if m <= n:
        return uniform_resample(coords, n)
    step = max(1, m // n)
    return coords[::step][:n]


def mbr_approximate(coords: np.ndarray, n: int) -> np.ndarray:
    """Approximate the polygon by its MBR (``MBR`` baseline, Fig. 8)."""
    coords = np.asarray(coords, dtype=np.float64)
    mn, mx = mbr(coords)
    corners = np.array(
        [
            [mn[0], mn[1]],
            [mn[0], mx[1]],
            [mx[0], mx[1]],
            [mx[0], mn[1]],
            [mn[0], mn[1]],
        ],
        dtype=np.float64,
    )
    return uniform_resample(corners, n)


# ----------------------------------------------------------------------
# normalization / public entry points
# ----------------------------------------------------------------------
def normalize(
    coords: np.ndarray,
    origin: Optional[np.ndarray] = None,
    span: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Min-max normalize with the fixed global lon/lat bounds."""
    origin = NORM_ORIGIN if origin is None else np.asarray(origin, dtype=np.float64)
    span = NORM_SPAN if span is None else np.asarray(span, dtype=np.float64)
    return (np.asarray(coords, dtype=np.float64) - origin) / span


def transform_polygon(
    polygon,
    num_vertices: int = 10,
    method: str = "adaptive",
    origin: Optional[np.ndarray] = None,
    span: Optional[np.ndarray] = None,
    flatten: bool = True,
) -> np.ndarray:
    """Transform one polygon into a fixed-size (normalized) vector.

    Parameters
    ----------
    polygon :
        Either an ``(m, 2)`` array of vertices or a ``POLYGON`` WKT string.
    num_vertices :
        Target vertex count ``n`` (10 by default, as in the paper).
    method :
        One of :data:`METHODS`.
    flatten :
        If ``True`` return a flat ``(2 * n,)`` vector, else ``(n, 2)``.
    """
    coords = wkt_to_coords(polygon) if isinstance(polygon, str) else np.asarray(
        polygon, dtype=np.float64
    )
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"expected an (m, 2) vertex array, got {coords.shape}")

    if method == "adaptive":
        resampled = adaptive_resample(coords, num_vertices)
    elif method == "uniform_sample":
        resampled = uniform_stride_sample(coords, num_vertices)
    elif method == "mbr":
        resampled = mbr_approximate(coords, num_vertices)
    elif method == "interp":
        resampled = uniform_resample(coords, num_vertices)
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    normed = normalize(resampled, origin, span)
    return normed.reshape(-1) if flatten else normed


def transform_polygons(
    polygons,
    num_vertices: int = 10,
    method: str = "adaptive",
    origin: Optional[np.ndarray] = None,
    span: Optional[np.ndarray] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Vectorized version of :func:`transform_polygon` -> ``(N, 2n)``."""
    out = [
        transform_polygon(
            p, num_vertices=num_vertices, method=method, origin=origin, span=span
        )
        for p in polygons
    ]
    return np.asarray(out, dtype=dtype)
