"""Cleaning of raw OSM ``ways`` dumps (the fiddly part).

A raw dump from https://spatialhadoop.cs.umn.edu/datasets.html is a TSV
with the schema in ``ways_schema.tsv``::

    way_id <TAB> shape(WKT) <TAB> tags

and mixes several geometry types in the same file.  PolyCard keeps only
simple polygons:

====================  ==================================================
``POLYGON``           **kept**
``MULTIPOLYGON``      dropped -- a relation of disjoint rings, no single
                      vertex list (the paper drops these)
``LINESTRING``        dropped -- roads/rivers, not areal objects
``GEOMETRYCOLLECTION`` dropped -- heterogeneous
====================  ==================================================

.. warning::
   A naive ``"POLYGON" in line`` substring test also matches
   ``MULTIPOLYGON`` and would silently keep multi-part geometries.
   :func:`polycard.io_utils.geometry_type` reads the leading WKT token
   instead, which is what the cleaning routine here uses.

The cleaned output is written both as a WKT TSV (input for query
generation) and as a ``COPY``-ready TSV for PostGIS.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .io_utils import (
    coords_to_wkt,
    ensure_dir,
    geometry_type,
    iter_ways,
    wkt_to_coords,
    write_ways,
)

__all__ = [
    "CleanStats",
    "clean_ways",
    "clean_dataset_file",
    "write_postgis_copy",
    "MIN_VERTICES",
]


# A closed ring needs at least 4 coordinate pairs (3 distinct vertices).
MIN_VERTICES = 4


@dataclass
class CleanStats:
    """Counters produced by a cleaning run."""

    total: int = 0
    kept: int = 0
    by_type: Dict[str, int] = field(default_factory=lambda: Counter())
    dropped_malformed: int = 0
    dropped_too_small: int = 0
    vertex_counts: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        vertices = np.asarray(self.vertex_counts, dtype=np.int64)
        return {
            "total_lines": self.total,
            "kept_polygons": self.kept,
            "dropped_malformed": self.dropped_malformed,
            "dropped_too_small": self.dropped_too_small,
            "by_geometry_type": dict(self.by_type),
            "avg_vertices_per_polygon": float(vertices.mean()) if vertices.size else 0.0,
            "max_vertices_per_polygon": int(vertices.max()) if vertices.size else 0,
        }

    def __str__(self) -> str:
        d = self.as_dict()
        types = "  ".join(f"{k}={v}" for k, v in sorted(d["by_geometry_type"].items()))
        return (
            f"total={d['total_lines']}  kept={d['kept_polygons']}  "
            f"malformed={d['dropped_malformed']}  too_small={d['dropped_too_small']}\n"
            f"  types: {types}\n"
            f"  avg vertices/polygon={d['avg_vertices_per_polygon']:.2f}  "
            f"max={d['max_vertices_per_polygon']}"
        )


def clean_ways(
    rows: Iterable[Tuple[str, str, str]],
    keep_types=("POLYGON",),
    min_vertices: int = MIN_VERTICES,
    dedupe: bool = True,
) -> Iterator[Tuple[str, np.ndarray, str]]:
    """Filter an iterable of ``(way_id, wkt, tags)`` rows.

    Yields ``(way_id, coords, tags)`` where ``coords`` is the ``(m, 2)``
    outer ring of a simple polygon.
    """
    seen = set()
    for way_id, wkt, tags in rows:
        gtype = geometry_type(wkt)
        if gtype not in keep_types:
            continue
        if dedupe:
            if way_id in seen:
                continue
            seen.add(way_id)
        try:
            coords = wkt_to_coords(wkt)
        except ValueError:
            continue
        if coords.shape[0] < min_vertices:
            continue
        yield way_id, coords, tags


def clean_dataset_file(
    src: str,
    dst: str,
    min_vertices: int = MIN_VERTICES,
    stats_path: Optional[str] = None,
    limit: Optional[int] = None,
    log=print,
) -> CleanStats:
    """Clean one raw dump file, writing the WKT TSV of kept polygons.

    Parameters
    ----------
    src :
        Raw ``ways`` dump (``way_id \\t WKT \\t tags``).
    dst :
        Destination TSV (same schema, polygons only).
    stats_path :
        Optional JSON report with type counts and vertex statistics.
    """
    stats = CleanStats()

    def rows() -> Iterator[Tuple[str, str, str]]:
        for i, row in enumerate(iter_ways(src)):
            if limit is not None and i >= limit:
                break
            yield row

    kept_rows = []
    for way_id, wkt, tags in rows():
        stats.total += 1
        gtype = geometry_type(wkt)
        stats.by_type[gtype] += 1
        if gtype != "POLYGON":
            continue
        try:
            coords = wkt_to_coords(wkt)
        except ValueError:
            stats.dropped_malformed += 1
            continue
        if coords.shape[0] < min_vertices:
            stats.dropped_too_small += 1
            continue
        stats.kept += 1
        stats.vertex_counts.append(int(coords.shape[0]))
        kept_rows.append((way_id, coords_to_wkt(coords), tags))

    write_ways(dst, kept_rows)
    log(f"[clean] {src} -> {dst}")
    log(str(stats))

    if stats_path:
        ensure_dir(os.path.dirname(os.path.abspath(stats_path)))
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats.as_dict(), fh, indent=2)
    return stats


def write_postgis_copy(
    src: str,
    dst: str,
    srid: int = 4326,
    log=print,
) -> int:
    """Convert a cleaned WKT TSV into a PostGIS ``\\copy`` payload.

    The emitted file has two columns (``way_id``, ``wkt``) and is loaded
    with::

        \\copy <table>(id, wkt) FROM '<dst>'
    """
    ensure_dir(os.path.dirname(os.path.abspath(dst)))
    n = 0
    with open(dst, "w", encoding="utf-8") as fh:
        for way_id, wkt, _tags in iter_ways(src):
            if geometry_type(wkt) != "POLYGON":
                continue
            escaped = wkt.replace("\t", " ").replace("\n", " ")
            fh.write(f"{way_id}\t{escaped}\n")
            n += 1
    log(f"[clean] wrote {n} rows for PostGIS COPY -> {dst}")
    return n
