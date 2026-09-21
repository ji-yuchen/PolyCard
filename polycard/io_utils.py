"""Reading and writing of OSM polygon datasets and derived arrays.

Two on-disk formats are used throughout PolyCard:

* **WKT TSV** -- the raw / cleaned OSM ``ways`` dump.  Each line is
  ``way_id <TAB> WKT <TAB> tags`` (see ``ways_schema.tsv``).
* **NumPy arrays** -- everything derived (query vectors, cardinalities).

Keeping all geometry in WKT until the featurization step means the
expensive parts (cleaning, PostGIS import) stay independent of the
learning code.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "GEOM_TYPES",
    "geometry_type",
    "wkt_to_coords",
    "coords_to_wkt",
    "iter_ways",
    "read_ways",
    "write_ways",
    "save_array",
    "load_array",
    "ensure_dir",
]


# Geometry types that may appear in an OSM ``ways`` dump.  Only ``POLYGON``
# is kept by PolyCard; see ``polycard.clean``.
GEOM_TYPES = (
    "POLYGON",
    "MULTIPOLYGON",
    "LINESTRING",
    "MULTILINESTRING",
    "GEOMETRYCOLLECTION",
    "POINT",
    "MULTIPOINT",
)


def geometry_type(wkt: str) -> str:
    """Return the geometry type token of a WKT string.

    ``"MULTIPOLYGON (((...)))"`` -> ``"MULTIPOLYGON"``.  A plain
    ``wkt.find("POLYGON") != -1`` test (as used during the original
    experiments) wrongly accepts MULTIPOLYGONs, so we always read the
    leading token instead.
    """
    return wkt.strip().split("(", 1)[0].strip().upper()


def wkt_to_coords(wkt: str) -> np.ndarray:
    """Parse the *outer ring* of a ``POLYGON`` WKT into an ``(m, 2)`` array.

    Interior rings (holes) are ignored -- they are rare in OSM landuse
    data and PolyCard's vertex sampling does not model them.
    """
    start = wkt.find("((")
    if start < 0:
        raise ValueError(f"not a POLYGON WKT: {wkt[:64]!r}")
    start += 2
    end = wkt.find(")", start)
    if end < 0:
        raise ValueError(f"unbalanced POLYGON WKT: {wkt[:64]!r}")
    ring = wkt[start:end]
    nums = np.array(ring.replace(",", " ").split(), dtype=np.float64)
    if nums.size < 6 or nums.size % 2 != 0:
        raise ValueError(f"malformed ring in WKT: {wkt[:64]!r}")
    return nums.reshape(-1, 2)


def coords_to_wkt(coords: np.ndarray, precision: int = 10) -> str:
    """Inverse of :func:`wkt_to_coords` (always emits a closed ring)."""
    coords = np.asarray(coords, dtype=np.float64)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[:1]])
    body = ", ".join(
        f"{x:.{precision}f} {y:.{precision}f}" for x, y in coords
    )
    return f"POLYGON (({body}))"


def iter_ways(path: str) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(way_id, wkt, tags)`` triples from a WKT TSV file."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            way_id = fields[0]
            wkt = fields[1]
            tags = fields[2] if len(fields) > 2 else ""
            yield way_id, wkt, tags


def read_ways(path: str, limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """Materialize :func:`iter_ways` into a list."""
    out = []
    for i, row in enumerate(iter_ways(path)):
        if limit is not None and i >= limit:
            break
        out.append(row)
    return out


def write_ways(path: str, rows) -> int:
    """Write ``(way_id, wkt, tags)`` triples back to a WKT TSV file."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for way_id, wkt, tags in rows:
            fh.write(f"{way_id}\t{wkt}\t{tags}\n")
            n += 1
    return n


def save_array(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.save(path, np.asarray(arr))


def load_array(path: str) -> np.ndarray:
    """Load a ``.npy`` array written by :func:`save_array`."""
    return np.load(path)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
