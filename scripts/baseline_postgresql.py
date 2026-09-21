#!/usr/bin/env python
"""Baseline: PostgreSQL's own cardinality estimator.

PostgreSQL estimates the result size of ``ST_Intersects`` from the
PostGIS geometry statistics (histogram + ND statistics).  We read both
numbers out of ``EXPLAIN (ANALYZE)``:

* the planner's **estimate** -- the value to compare against the truth;
* the **actual** row count -- redundant with our ground truth, but useful
  as a sanity check that the SQL file and the WKT file are aligned.

PostgreSQL does not expose estimation time separately, so (as in the
paper) **planning time** is used as its proxy.

Usage
-----
    # 1. build the EXPLAIN script
    python scripts/baseline_postgresql.py --dataset sports --emit

    # 2. run it (see the printed psql command), then
    python scripts/baseline_postgresql.py --dataset sports \
        --parse data/baselines/sports_explain.out
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.io_utils import load_array  # noqa: E402
from polycard.metrics import summarize_qerror  # noqa: E402

DEFAULT_NODE_PATTERN = r"(Index Scan|Index Only Scan|Bitmap Heap Scan|Seq Scan|Sample Scan)"
_ROWS_RE = re.compile(r"rows=(\d+)")
_PLANNING_RE = re.compile(r"Planning Time:\s*([\d.]+)\s*ms")
_OPTIMIZATION_RE = re.compile(r"Optimization\s+([\d.]+)\s*ms")


def emit_explain(queries_wkt: str, out_path: str, table: str, srid: int = 4326) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(queries_wkt, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        fout.write("LOAD 'auto_explain';\n")
        fout.write("SET auto_explain.log_analyze = true;\n")
        fout.write("SET auto_explain.log_min_duration = 0;\n")
        for line in fin:
            wkt = line.strip()
            if not wkt:
                continue
            fout.write("EXPLAIN (ANALYZE)\n")
            fout.write(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE ST_Intersects(ST_GeomFromText('{wkt}', {srid}), geom);\n"
            )
    print(f"wrote EXPLAIN script -> {out_path}")
    print("run it with:")
    print(f"    psql -d <db> -f {out_path} > <output> 2>&1")


def parse_explain(path: str, node_pattern: str = DEFAULT_NODE_PATTERN):
    """Return ``(estimated, actual, planning_times, optimization_times)``."""
    node_re = re.compile(node_pattern)
    est, act, planning, optimization = [], [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            m_plan = _PLANNING_RE.search(line)
            if m_plan:
                planning.append(float(m_plan.group(1)))
            m_opt = _OPTIMIZATION_RE.search(line)
            if m_opt:
                optimization.append(float(m_opt.group(1)))
            if node_re.search(stripped) and "rows=" in line:
                nums = _ROWS_RE.findall(line)
                if len(nums) >= 2:
                    est.append(int(nums[0]))
                    act.append(int(nums[1]))
                elif len(nums) == 1:
                    est.append(int(nums[0]))
                    act.append(-1)
    return (
        np.asarray(est, dtype=np.int64),
        np.asarray(act, dtype=np.int64),
        np.asarray(planning, dtype=np.float64),
        np.asarray(optimization, dtype=np.float64),
    )


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--emit", action="store_true", help="write the EXPLAIN script")
    parser.add_argument("--parse", default=None, help="parse an EXPLAIN output file")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--srid", type=int, default=None)
    parser.add_argument(
        "--node-pattern", default=DEFAULT_NODE_PATTERN,
        help="regex identifying the plan node whose rows= is read")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    default_srid = (
        cfg.get("baselines", {}).get("postgresql", {}).get("srid", 4326)
    )
    queries_dir = path_in(data_dir, "queries")
    out_dir = args.out_dir or path_in(data_dir, "baselines")
    os.makedirs(out_dir, exist_ok=True)

    queries_path = args.queries or os.path.join(queries_dir, f"{args.dataset}_queries.wkt")

    log_header(f"baseline postgresql: {args.dataset}")

    if args.emit:
        if not os.path.exists(queries_path):
            print(f"ERROR: {queries_path} not found")
            return 1
        emit_explain(
            queries_path,
            os.path.join(out_dir, f"{args.dataset}_explain.sql"),
            args.dataset,
            args.srid or default_srid,
        )
        return 0

    if not args.parse:
        print("ERROR: pass --emit or --parse <file>")
        return 1

    est, act, planning, optimization = parse_explain(args.parse, args.node_pattern)
    print(f"parsed {est.size} plans from {args.parse}")
    if est.size == 0:
        print("no matching plan nodes -- try a different --node-pattern")
        return 1

    np.savetxt(os.path.join(out_dir, f"{args.dataset}_pg_est.txt"), est, fmt="%d")
    np.savetxt(os.path.join(out_dir, f"{args.dataset}_pg_act.txt"), act, fmt="%d")
    print(f"saved estimates -> {os.path.join(out_dir, f'{args.dataset}_pg_est.txt')}")

    report = {"dataset": args.dataset, "num_plans": int(est.size)}
    if planning.size:
        report["planning_time_ms_mean"] = float(planning.mean())
        report["planning_time_ms_median"] = float(np.median(planning))
        print(f"planning time   : {planning.mean():.3f} ms (mean)")
    if optimization.size:
        report["optimization_time_ms_mean"] = float(optimization.mean())
        print(f"optimization    : {optimization.mean():.3f} ms (mean)")

    cards_path = os.path.join(queries_dir, f"{args.dataset}_cards.npy")
    if os.path.exists(cards_path):
        truth = load_array(cards_path).astype(np.int64).reshape(-1)
        n = min(est.shape[0], truth.shape[0])
        s = summarize_qerror(est[:n], truth[:n])
        report["qerror"] = s
        print("\nq-error vs. our ground truth:")
        for k in ("25th", "50th", "75th", "95th", "99th"):
            print(f"  {k:>5} : {s[k]:.2f}")
    else:
        print(f"\n(no {cards_path}; skipping q-error computation)")

    save_json(os.path.join(out_dir, f"{args.dataset}_pg_report.json"), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
