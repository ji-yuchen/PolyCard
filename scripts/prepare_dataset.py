#!/usr/bin/env python
"""Step 1 -- clean a raw OSM ``ways`` dump into simple polygons.

    python scripts/prepare_dataset.py --dataset parks --raw data/raw/parks

Input  : ``data/raw/<name>``  (``way_id \\t WKT \\t tags``, see ways_schema.tsv)
Output :
    ``data/derived/<name>_clean.tsv``         polygons only, same schema
    ``data/derived/<name>_postgis.tsv``       COPY payload for PostGIS
    ``data/derived/<name>_clean_stats.json``  type counts + vertex stats

Dropped geometry types (Section 5.1 of the paper): MULTIPOLYGON,
LINESTRING, GEOMETRYCOLLECTION.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.clean import MIN_VERTICES, clean_dataset_file, write_postgis_copy  # noqa: E402


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True, help="dataset name (e.g. parks)")
    parser.add_argument("--raw", default=None, help="path to the raw ways dump")
    parser.add_argument("--out-dir", default=None, help="derived output directory")
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=None,
        help=f"drop polygons with fewer vertices (default {MIN_VERTICES})",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only read the first N lines (debug)"
    )
    parser.add_argument("--skip-postgis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    derived = args.out_dir or path_in(data_dir, "derived")
    os.makedirs(derived, exist_ok=True)

    dcfg = cfg.get("datasets", {}).get(args.dataset, {})
    raw = args.raw or dcfg.get("raw") or path_in(data_dir, "raw", args.dataset)
    min_vertices = args.min_vertices or cfg.get("clean", {}).get(
        "min_vertices", MIN_VERTICES
    )

    if not os.path.exists(raw):
        print(f"ERROR: raw dump not found: {raw}")
        print("Download the OSM extracts from https://spatialhadoop.cs.umn.edu/datasets.html")
        return 1

    log_header(f"prepare_dataset: {args.dataset}")

    clean_tsv = os.path.join(derived, f"{args.dataset}_clean.tsv")
    stats_json = os.path.join(derived, f"{args.dataset}_clean_stats.json")
    stats = clean_dataset_file(
        raw, clean_tsv, min_vertices=min_vertices, stats_path=stats_json,
        limit=args.limit,
    )

    if not args.skip_postgis:
        write_postgis_copy(
            clean_tsv, os.path.join(derived, f"{args.dataset}_postgis.tsv")
        )

    save_json(stats_json, stats.as_dict())
    print(f"\nDone. {stats.kept} polygons written to {clean_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
