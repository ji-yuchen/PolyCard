#!/usr/bin/env python
"""Step 3 -- compute the *true* cardinality of every generated query.

Two interchangeable engines:

``shapely``    pure Python, no database
                   python scripts/collect_cardinality.py --dataset cemetery \
                       --engine shapely

``postgis``    the path used in the paper -- run the SQL through PostGIS
                   python scripts/collect_cardinality.py --dataset cemetery \
                       --engine postgis --emit-sql
                   psql -f data/queries/cemetery_queries_explain.sql \
                        -d osm > data/queries/cemetery_explain.out
                   python scripts/collect_cardinality.py --dataset cemetery \
                       --engine postgis --parse data/queries/cemetery_explain.out

Both write ``data/queries/<name>_cards.npy`` (int64, one row per query,
in the same order as the ``.wkt`` file).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.io_utils import iter_ways, save_array, wkt_to_coords  # noqa: E402

# ``Count`` rows in a psql ``-t`` (tuples-only) output, e.g. "    12345"
_COUNT_RE = re.compile(r"^\s*(\d+)\s*$")


def read_queries(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return [wkt_to_coords(w) for w in lines]


def run_shapely(dataset_tsv: str, queries: list, log_every: int = 10000) -> np.ndarray:
    from polycard.ground_truth import PolygonIndex

    polys = []
    for _way_id, wkt, _tags in iter_ways(dataset_tsv):
        try:
            polys.append(wkt_to_coords(wkt))
        except ValueError:
            continue
    index = PolygonIndex(polys)
    print(f"indexed {len(index)} dataset polygons")
    cards = index.count_batch(queries, log_every=log_every)
    return cards


def emit_postgis_sql(
    queries_wkt: str, out_path: str, table: str, srid: int = 4326
) -> None:
    """Write a batched COUNT(*) script that psql can run in one pass."""
    with open(queries_wkt, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        fout.write("\\timing off\n")
        fout.write("\\pset tuples_only on\n")
        fout.write("\\pset format unaligned\n")
        for line in fin:
            wkt = line.strip()
            if not wkt:
                continue
            fout.write(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE ST_Intersects(ST_GeomFromText('{wkt}', {srid}), geom);\n"
            )
    print(f"wrote PostGIS script -> {out_path}")
    print(f"run it with:  psql -d <db> -f {out_path} > <output>")


def parse_postgis_output(path: str) -> np.ndarray:
    """Extract one integer per line from a tuples-only ``COUNT(*)`` run."""
    cards = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("SELECT"):
                continue
            m = _COUNT_RE.match(line)
            if m:
                cards.append(int(m.group(1)))
            elif re.match(r"^(psql:|ERROR|WARNING|SET|Timing)", line, re.I):
                continue
            else:
                # Anything else (e.g. a stray header) is not a count.
                continue
    return np.asarray(cards, dtype=np.int64)


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--engine", choices=("shapely", "postgis"), default="shapely")
    parser.add_argument("--dataset-tsv", default=None, help="cleaned dataset polygons")
    parser.add_argument("--queries", default=None, help="query WKT file")
    parser.add_argument("--out", default=None, help="output .npy of cardinalities")
    parser.add_argument("--srid", type=int, default=None)
    parser.add_argument("--emit-sql", action="store_true", help="postgis: write SQL only")
    parser.add_argument("--parse", default=None, help="postgis: parse a psql output file")
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    srid = args.srid or cfg.get("baselines", {}).get("postgresql", {}).get("srid", 4326)
    derived = path_in(data_dir, "derived")
    queries_dir = path_in(data_dir, "queries")

    queries_path = args.queries or os.path.join(queries_dir, f"{args.dataset}_queries.wkt")
    out_path = args.out or os.path.join(queries_dir, f"{args.dataset}_cards.npy")

    if not os.path.exists(queries_path):
        print(f"ERROR: query file not found: {queries_path}")
        print("Run scripts/generate_queries.py first.")
        return 1

    log_header(f"collect_cardinality: {args.dataset} [{args.engine}]")

    if args.engine == "postgis":
        if args.parse:
            cards = parse_postgis_output(args.parse)
            print(f"parsed {cards.size} cardinalities from {args.parse}")
        elif args.emit_sql:
            emit_postgis_sql(
                queries_path,
                os.path.join(queries_dir, f"{args.dataset}_count.sql"),
                args.dataset,
                srid,
            )
            return 0
        else:
            print("ERROR: for --engine postgis pass --emit-sql or --parse <file>")
            return 1
    else:
        dataset_tsv = args.dataset_tsv or os.path.join(derived, f"{args.dataset}_clean.tsv")
        if not os.path.exists(dataset_tsv):
            print(f"ERROR: cleaned dataset not found: {dataset_tsv}")
            print("Run scripts/prepare_dataset.py first.")
            return 1
        queries = read_queries(queries_path)
        print(f"loaded {len(queries)} queries")
        cards = run_shapely(dataset_tsv, queries, log_every=args.log_every)

    save_array(out_path, cards)
    nonzero = int((cards > 0).sum())
    print(
        f"cardinalities: n={cards.size} min={cards.min()} max={cards.max()} "
        f"mean={cards.mean():.2f} zero={cards.size - nonzero} "
        f"({100.0 * (cards.size - nonzero) / max(cards.size, 1):.1f}%)"
    )
    print(f"saved -> {out_path}")
    save_json(
        os.path.join(queries_dir, f"{args.dataset}_cards_stats.json"),
        {
            "dataset": args.dataset,
            "engine": args.engine,
            "num": int(cards.size),
            "min": int(cards.min()) if cards.size else 0,
            "max": int(cards.max()) if cards.size else 0,
            "mean": float(cards.mean()) if cards.size else 0.0,
            "num_zero": int(cards.size - nonzero),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
