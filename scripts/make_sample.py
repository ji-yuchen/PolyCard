#!/usr/bin/env python
"""Create a small dataset so that the whole pipeline can be run end-to-end.

Two modes:

**synthetic** (default, no input data needed)
    Generates simple star-shaped polygons around random centres.  Vertices
    are emitted in increasing polar angle, which guarantees a
    non-self-intersecting ring::

        python scripts/make_sample.py --synthetic --num 5000 \
            --out data/sample/toy_clean.tsv

**sample** (needs a real dump)
    Uniformly subsample a cleaned dataset::

        python scripts/make_sample.py --from-raw data/raw/cemetery \
            --num 5000 --out data/sample/cemetery_clean.tsv

Both write a WKT TSV with the same schema as ``prepare_dataset.py``, so
the rest of the pipeline works unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.clean import clean_dataset_file  # noqa: E402
from polycard.io_utils import coords_to_wkt, iter_ways, write_ways  # noqa: E402

# A 1x1 degree box, chosen to look like a dense urban landuse extract.
DEFAULT_BBOX = (135.0, 34.0, 136.0, 35.0)


def synthetic_polygon(rng: np.random.Generator, center, radius, num_vertices):
    """A simple (non-self-intersecting) star-shaped polygon."""
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, num_vertices))
    radii = radius * rng.uniform(0.45, 1.0, num_vertices)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles) * 0.8  # mild anisotropy
    coords = np.stack([x, y], axis=1)
    return np.vstack([coords, coords[:1]])  # close the ring


def generate_synthetic(
    num: int,
    seed: int,
    bbox=DEFAULT_BBOX,
    min_vertices: int = 4,
    max_vertices: int = 40,
    num_clusters: int = 8,
    cluster_sigma: float = 0.12,
) -> list:
    """Generate *clustered* polygons.

    Real OSM landuse data is heavily clustered (dense downtown, sparse
    countryside), and that spatial skew is exactly what makes cardinality
    estimation non-trivial.  A uniform random cloud would make every
    location carry the same density and the task would degenerate into
    "measure the query area", so we mimic the real skew with a Gaussian
    mixture of cluster centres.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox
    centers = np.column_stack(
        [rng.uniform(x0, x1, num_clusters), rng.uniform(y0, y1, num_clusters)]
    )
    rows = []
    for i in range(num):
        base = centers[int(rng.integers(num_clusters))]
        center = base + rng.normal(0.0, cluster_sigma, 2)
        # Heavy-tailed radii spanning two orders of magnitude, so the
        # cardinality distribution stays skewed like real OSM data.
        radius = 10 ** rng.uniform(-2.6, -1.1)
        nv = int(rng.integers(min_vertices, max_vertices + 1))
        coords = synthetic_polygon(rng, center, radius, nv)
        rows.append((str(i), coords_to_wkt(coords), "synthetic"))
    return rows


def subsample(src: str, num: int, seed: int, min_vertices: int) -> list:
    rng = np.random.default_rng(seed)
    rows = [(w, t) for _i, w, t in iter_ways(src)]
    if not rows:
        raise SystemExit(f"no polygons found in {src}")
    if len(rows) > num:
        idx = rng.choice(len(rows), size=num, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    out = []
    for i, (wkt, tags) in enumerate(rows):
        out.append((str(i), wkt, tags))
    return out


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--synthetic", action="store_true", help="generate toy polygons")
    parser.add_argument("--from-raw", default=None, help="raw/cleaned dump to sample")
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bbox", default=None, help="x0,y0,x1,y1 for synthetic mode")
    parser.add_argument("--min-vertices", type=int, default=4)
    parser.add_argument("--max-vertices", type=int, default=40)
    parser.add_argument("--num-clusters", type=int, default=8,
                        help="synthetic mode: number of cluster centres")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    out = args.out or os.path.join(path_in(data_dir, "sample"), "toy_clean.tsv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    log_header("make_sample")
    if args.from_raw:
        src = args.from_raw
        if src.endswith(".tsv"):
            rows = subsample(src, args.num, args.seed, args.min_vertices)
        else:
            tmp = os.path.join(
                os.path.dirname(os.path.abspath(out)), "_tmp_clean.tsv"
            )
            clean_dataset_file(
                src, tmp,
                min_vertices=cfg.get("clean", {}).get("min_vertices", 4),
                log=lambda *a: None,
            )
            rows = subsample(tmp, args.num, args.seed, args.min_vertices)
            os.remove(tmp)
        print(f"sampled {len(rows)} polygons from {src}")
    else:
        bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else DEFAULT_BBOX
        rows = generate_synthetic(
            args.num, args.seed, bbox, args.min_vertices, args.max_vertices,
            num_clusters=args.num_clusters,
        )
        print(f"generated {len(rows)} synthetic polygons in bbox={bbox} "
              f"({args.num_clusters} clusters)")

    n = write_ways(out, rows)
    print(f"wrote {n} polygons -> {out}")
    save_json(
        os.path.splitext(out)[0] + "_meta.json",
        {
            "source": "synthetic" if not args.from_raw else args.from_raw,
            "num": n,
            "seed": args.seed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
