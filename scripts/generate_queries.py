#!/usr/bin/env python
"""Step 2 -- generate intersection queries (Section 4.3 of the paper).

    python scripts/generate_queries.py --dataset sports --num-queries 100000

PolyCard is query-driven, so it needs a large pool of labelled queries.
Polygons are *not* generated from scratch (hard to control vertex count
and shape); instead real polygons are sampled from a source dataset and
perturbed with one of three augmentations:

    perturbation / reflection / shifting

Output :
    ``data/queries/<name>_queries.wkt``   one POLYGON WKT per line
    ``data/queries/<name>_queries.sql``   the equivalent PostGIS queries
    ``data/queries/<name>_meta.json``     generation parameters

The WKT file is the single source of truth: every later step (ground
truth, featurization, baselines) reads from it, so the SQL file can be
regenerated at any time without changing results.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json, set_seed  # noqa: E402

from polycard.augment import AUGMENTATIONS, augment_polygon  # noqa: E402
from polycard.io_utils import coords_to_wkt, iter_ways, wkt_to_coords  # noqa: E402


def load_source_polygons(path: str, limit: int | None = None) -> list:
    """Read the ``(m, 2)`` vertex arrays of every polygon in a WKT TSV."""
    out = []
    for i, (_way_id, wkt, _tags) in enumerate(iter_ways(path)):
        if limit is not None and i >= limit:
            break
        try:
            out.append(wkt_to_coords(wkt))
        except ValueError:
            continue
    return out


def write_sql(path: str, wkts, table: str, srid: int = 4326) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for wkt in wkts:
            fh.write(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE ST_Intersects(ST_GeomFromText('{wkt}', {srid}), geom);\n"
            )


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True, help="dataset being queried")
    parser.add_argument(
        "--source",
        default=None,
        help="WKT TSV of polygons used to seed queries "
             "(default: the cleaned dataset itself)",
    )
    parser.add_argument("--num-queries", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--srid", type=int, default=4326)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-sql", action="store_true", help="skip the .sql output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    gen_cfg = cfg.get("generation", {})
    seed = args.seed if args.seed is not None else gen_cfg.get("seed", 0)
    set_seed(seed)

    queries_dir = args.out_dir or path_in(data_dir, "queries")
    os.makedirs(queries_dir, exist_ok=True)

    source = args.source
    if source is None:
        source = cfg.get("query_source_clean") or ""
        if source:
            source = path_in(data_dir, source) if not os.path.isabs(source) else source
            if not os.path.exists(source):
                source = ""
    if not source:
        source = os.path.join(
            args.out_dir or path_in(data_dir, "derived"), f"{args.dataset}_clean.tsv"
        )
    if not os.path.exists(source):
        print(f"ERROR: query source not found: {source}")
        print("Run scripts/prepare_dataset.py first.")
        return 1

    log_header(f"generate_queries: {args.dataset}")
    print(f"source polygons : {source}")
    print(f"num queries     : {args.num_queries}")
    print(f"seed            : {seed}")

    pool = load_source_polygons(source)
    if not pool:
        print("ERROR: no usable polygons in the source file")
        return 1
    print(f"pool size       : {len(pool)}")

    rng = np.random.default_rng(seed)
    weights = gen_cfg.get("augmentation_weights") or None
    if weights:
        w = np.array([float(weights[k]) for k in AUGMENTATIONS], dtype=np.float64)
        w = w / w.sum()
    else:
        w = None

    wkts = []
    for i in range(args.num_queries):
        base = pool[int(rng.integers(len(pool)))]
        method = (
            AUGMENTATIONS[int(rng.choice(len(AUGMENTATIONS), p=w))]
            if w is not None
            else None
        )
        augmented = augment_polygon(base, method=method, rng=rng)
        wkts.append(coords_to_wkt(augmented))
        if (i + 1) % 100000 == 0:
            print(f"  generated {i + 1} / {args.num_queries}")

    wkt_path = os.path.join(queries_dir, f"{args.dataset}_queries.wkt")
    with open(wkt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(wkts) + "\n")
    print(f"wrote {len(wkts)} query polygons -> {wkt_path}")

    if not args.no_sql:
        sql_path = os.path.join(queries_dir, f"{args.dataset}_queries.sql")
        write_sql(sql_path, wkts, args.dataset, args.srid)
        print(f"wrote PostGIS queries        -> {sql_path}")

    save_json(
        os.path.join(queries_dir, f"{args.dataset}_meta.json"),
        {
            "dataset": args.dataset,
            "source": source,
            "num_queries": len(wkts),
            "seed": seed,
            "srid": args.srid,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
