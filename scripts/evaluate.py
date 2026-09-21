#!/usr/bin/env python
"""Step 6 -- evaluate PolyCard (accuracy + estimation latency).

    python scripts/evaluate.py --dataset sports

Reports the q-error percentiles of Table 2 (25/50/75/95/99) together with
the per-query estimation time in microseconds, and writes
``results/<name>/metrics.json``.

Optional baseline comparison:

    python scripts/evaluate.py --dataset sports \
        --baseline-postgresql data/baselines/sports_pg_est.txt \
        --baseline-cdhistogram data/baselines/sports_cd_est.txt

Each baseline file is a plain text file with one estimated cardinality per
line, aligned with ``data/features/<name>/test/cardinality.npy``.  They
are produced by ``scripts/baseline_postgresql.py`` and
``scripts/baseline_cdhistogram.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.dataset import load_feature_bundle  # noqa: E402
from polycard.evaluate import evaluate_model, load_model  # noqa: E402
from polycard.metrics import summarize_qerror  # noqa: E402


def load_baseline_file(path: str) -> np.ndarray:
    """Read a one-estimate-per-line baseline output file."""
    vals = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float64)


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--baseline-postgresql", default=None)
    parser.add_argument("--baseline-cdhistogram", default=None)
    parser.add_argument("--baseline-ri", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    eval_cfg = cfg.get("evaluate", {})

    bundle = args.features or os.path.join(path_in(data_dir, "features"), args.dataset)
    split_dir = os.path.join(bundle, args.split)
    if not os.path.exists(split_dir):
        print(f"ERROR: feature bundle not found: {split_dir}")
        return 1

    ckpt = args.checkpoint or os.path.join(
        os.path.dirname(os.path.abspath(data_dir)), "results", args.dataset
    )
    if not os.path.exists(os.path.join(ckpt, "polycard.pt")):
        print(f"ERROR: checkpoint not found in {ckpt}. Run scripts/train.py first.")
        return 1

    log_header(f"evaluate: {args.dataset} [{args.split}]")
    dataset, meta, cardinality = load_feature_bundle(split_dir)
    device = args.device or "auto"
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, stats = load_model(ckpt, device=device)
    summary = evaluate_model(
        model,
        dataset,
        min_val=stats["min_val"],
        max_val=stats["max_val"],
        label_eps=stats.get("label_eps", 1.02),
        device=device,
        batch_size=args.batch_size or eval_cfg.get("batch_size", 1024),
    )

    print("\n--- PolyCard ---")
    print(f"  25th q-error : {summary['25th']:.2f}")
    print(f"  50th q-error : {summary['50th']:.2f}")
    print(f"  75th q-error : {summary['75th']:.2f}")
    print(f"  95th q-error : {summary['95th']:.2f}")
    print(f"  99th q-error : {summary['99th']:.2f}")
    print(f"  mean q-error : {summary['mean']:.2f}")
    print(f"  estimation   : {summary['est_time_us']:.2f} us/query")
    print(f"  queries      : {summary['num_samples']}")

    report = {"dataset": args.dataset, "split": args.split, "polycard": summary}

    # --- baselines ------------------------------------------------------
    if cardinality is None:
        cardinality = np.exp(
            np.asarray(dataset.labels, dtype=np.float64)
            * (stats["max_val"] - stats["min_val"]) + stats["min_val"]
        ) - stats.get("label_eps", 1.02)

    for name, path in (
        ("postgresql", args.baseline_postgresql),
        ("cdhistogram", args.baseline_cdhistogram),
        ("ri", args.baseline_ri),
    ):
        if not path:
            continue
        est = load_baseline_file(path)
        n = min(est.shape[0], cardinality.shape[0])
        s = summarize_qerror(est[:n], cardinality[:n])
        print(f"\n--- {name} ---")
        for k in ("25th", "50th", "75th", "95th", "99th", "mean"):
            print(f"  {k:>5} q-error : {s[k]:.2f}")
        report[name] = s

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(data_dir)), "results", args.dataset
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"metrics_{args.split}.json")
    save_json(out_path, report)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
