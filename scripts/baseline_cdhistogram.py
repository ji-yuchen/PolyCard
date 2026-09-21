#!/usr/bin/env python
"""Baseline: CDHistogram (Jin et al., VLDB 2000).

CDHistogram is a histogram for *rectangles*.  To use it on polygons we
feed it the **minimum bounding rectangle** of each query polygon -- the
same approximation the paper applies when extending CDHistogram to the
polygon case (Section 5.2).  That approximation is exactly what costs
CDHistogram accuracy.

We ship the *input/output adapters*, not the histogram itself: the
original implementation is C++ and is not redistributed here.  See
``baselines/README.md`` for how to obtain it.

Usage
-----
    # 1. write the input file for the CDHistogram binary
    python scripts/baseline_cdhistogram.py --dataset sports --emit

    # 2. run CDHistogram, then parse its output
    python scripts/baseline_cdhistogram.py --dataset sports \
        --parse data/baselines/sports_cd_out.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.io_utils import load_array, wkt_to_coords  # noqa: E402
from polycard.metrics import summarize_qerror  # noqa: E402
from polycard.transform import mbr  # noqa: E402


def emit_input(queries_wkt: str, out_path: str) -> int:
    """Write ``<index> <TAB> POLYGON((...))`` -- one MBR per query."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = 0
    with open(queries_wkt, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            wkt = line.strip()
            if not wkt:
                continue
            coords = wkt_to_coords(wkt)
            mn, mx = mbr(coords)
            x0, y0 = mn
            x1, y1 = mx
            ring = (
                f"{x0:.10f} {y0:.10f}, {x1:.10f} {y0:.10f}, "
                f"{x1:.10f} {y1:.10f}, {x0:.10f} {y1:.10f}, "
                f"{x0:.10f} {y0:.10f}"
            )
            fout.write(f"{n + 1}\tPOLYGON (({ring}))\n")
            n += 1
    return n


def parse_output(path: str) -> np.ndarray:
    """Read one estimated cardinality per line (ignoring comment lines)."""
    vals = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept "<id> <TAB> <value>" as well as a bare "<value>".
            token = line.split("\t")[-1].split()[-1]
            try:
                vals.append(float(token))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float64)


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--parse", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--raw-wkt", action="store_true",
                        help="emit the true polygon instead of its MBR")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    bucket_size = cfg.get("baselines", {}).get("cdhistogram", {}).get("bucket_size", 10000)
    queries_dir = path_in(data_dir, "queries")
    out_dir = args.out_dir or path_in(data_dir, "baselines")
    os.makedirs(out_dir, exist_ok=True)
    queries_path = args.queries or os.path.join(queries_dir, f"{args.dataset}_queries.wkt")

    log_header(f"baseline cdhistogram: {args.dataset}")

    if args.emit:
        if not os.path.exists(queries_path):
            print(f"ERROR: {queries_path} not found")
            return 1
        out_path = os.path.join(out_dir, f"{args.dataset}_cd_in.txt")
        if args.raw_wkt:
            n = 0
            with open(queries_path, "r", encoding="utf-8") as fin, open(
                out_path, "w", encoding="utf-8"
            ) as fout:
                for line in fin:
                    if line.strip():
                        n += 1
                        fout.write(f"{n}\t{line.strip()}\n")
        else:
            n = emit_input(queries_path, out_path)
        print(f"wrote {n} query rectangles -> {out_path}")
        print(f"run CDHistogram with {bucket_size} buckets per dimension")
        return 0

    if not args.parse:
        print("ERROR: pass --emit or --parse <file>")
        return 1

    est = parse_output(args.parse)
    np.savetxt(os.path.join(out_dir, f"{args.dataset}_cd_est.txt"), est, fmt="%.6f")
    print(f"parsed {est.size} estimates -> {os.path.join(out_dir, f'{args.dataset}_cd_est.txt')}")

    report = {"dataset": args.dataset, "num": int(est.size)}
    cards_path = os.path.join(queries_dir, f"{args.dataset}_cards.npy")
    if os.path.exists(cards_path):
        truth = load_array(cards_path).astype(np.int64).reshape(-1)
        n = min(est.shape[0], truth.shape[0])
        s = summarize_qerror(est[:n], truth[:n])
        report["qerror"] = s
        print("\nq-error vs. our ground truth:")
        for k in ("25th", "50th", "75th", "95th", "99th"):
            print(f"  {k:>5} : {s[k]:.2f}")
    save_json(os.path.join(out_dir, f"{args.dataset}_cd_report.json"), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
