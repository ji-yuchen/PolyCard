"""Training-query augmentation (Section 4.3 of the paper).

PolyCard is query-driven, so it needs a large pool of *training queries*.
Generating polygons from scratch is hard (variable vertex count, complex
shapes), so the generator samples real polygons from the dataset and
applies one of three random transformations:

``perturbation``  jitter every coordinate by a small polygon-relative offset
``reflection``    point-reflect every vertex through the MBR center
``shifting``      translate the whole polygon by the same offset

.. note::
   NumPy slicing returns a view, so a naive ``new = coordinates`` followed
   by ``new[...] += ...`` would modify the caller's polygon.  All three
   functions below copy their input first and are side-effect free.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .transform import mbr_center

__all__ = [
    "AUGMENTATIONS",
    "perturbation",
    "reflection",
    "shifting",
    "augment_polygon",
    "augment_polygons",
]


AUGMENTATIONS = ("perturbation", "reflection", "shifting")

# Magnitude of the translation applied by :func:`shifting`, in degrees
# (the paper uses ``(U(0,1) - 0.5) / 2``).
SHIFT_SCALE = 0.5


def _as_coords(polygon) -> np.ndarray:
    if isinstance(polygon, str):
        from .io_utils import wkt_to_coords

        return wkt_to_coords(polygon)
    return np.array(polygon, dtype=np.float64, copy=True)


def perturbation(coords, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add a small random offset to each coordinate.

    The offset scale is derived from the polygon itself -- the smallest
    distance among its first three vertices -- so that jitter stays
    proportional to polygon size.
    """
    rng = np.random.default_rng() if rng is None else rng
    out = _as_coords(coords)
    flat = out.reshape(-1)
    if flat.size < 6:
        return out
    scale = min(abs(flat[1] - flat[3]), abs(flat[3] - flat[5]))
    flat += (rng.random(flat.size) - 0.5) * scale
    out = flat.reshape(-1, 2)
    out[-1] = out[0]  # keep the ring closed
    return out


def reflection(coords, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Point-reflect every vertex through the center of the MBR."""
    out = _as_coords(coords)
    cx, cy = mbr_center(out)
    out[:, 0] = 2.0 * cx - out[:, 0]
    out[:, 1] = 2.0 * cy - out[:, 1]
    out[-1] = out[0]
    return out


def shifting(coords, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Translate the polygon by a single random offset on both axes."""
    rng = np.random.default_rng() if rng is None else rng
    out = _as_coords(coords)
    out += (rng.random() - 0.5) * SHIFT_SCALE
    out[-1] = out[0]
    return out


def augment_polygon(
    coords,
    method: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply one augmentation, sampled uniformly when ``method is None``."""
    rng = np.random.default_rng() if rng is None else rng
    if method is None:
        method = AUGMENTATIONS[rng.integers(len(AUGMENTATIONS))]
    if method == "perturbation":
        return perturbation(coords, rng)
    if method == "reflection":
        return reflection(coords, rng)
    if method == "shifting":
        return shifting(coords, rng)
    raise ValueError(f"unknown augmentation {method!r}; expected one of {AUGMENTATIONS}")


def augment_polygons(
    polygons,
    seed: int = 0,
    method: Optional[str] = None,
) -> list:
    """Augment a list of polygons with a reproducible RNG stream."""
    rng = np.random.default_rng(seed)
    return [augment_polygon(p, method, rng) for p in polygons]
