#!/usr/bin/env python
"""Step 5 -- train PolyCard.

    python scripts/train.py --dataset sports --epochs 100

Reads the feature bundle produced by ``build_training_set.py`` and writes
a checkpoint plus a short metrics report:

    results/<name>/polycard.pt
    results/<name>/label_stats.json
    results/<name>/train_history.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json, set_seed  # noqa: E402

from polycard.dataset import load_feature_bundle  # noqa: E402
from polycard.train import train_polycard  # noqa: E402


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", default=None, help="feature bundle directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hid-units", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-gamma", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    set_seed(seed)

    bundle = args.features or os.path.join(path_in(data_dir, "features"), args.dataset)
    train_dir = os.path.join(bundle, "train")
    test_dir = os.path.join(bundle, "test")
    if not os.path.exists(train_dir):
        print(f"ERROR: feature bundle not found: {train_dir}")
        print("Run scripts/build_training_set.py first.")
        return 1

    log_header(f"train: {args.dataset}")
    train_set, meta, _ = load_feature_bundle(train_dir)
    test_set = None
    if os.path.exists(test_dir):
        test_set, _, _ = load_feature_bundle(test_dir)
    print(f"train queries: {len(train_set)}" +
          (f"   test queries: {len(test_set)}" if test_set else ""))

    label_stats_path = os.path.join(bundle, "label_stats.json")
    label_stats = {}
    if os.path.exists(label_stats_path):
        with open(label_stats_path, "r", encoding="utf-8") as fh:
            label_stats = json.load(fh)
    min_val = float(label_stats.get("min_val", 0.0))
    max_val = float(label_stats.get("max_val", 1.0))
    label_eps = float(label_stats.get("label_eps", train_cfg.get("label_eps", 1.02)))

    device = args.device or train_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if _has_cuda() else "cpu"

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(data_dir)), "results", args.dataset
    )

    result = train_polycard(
        train_set,
        test_set,
        min_val=min_val,
        max_val=max_val,
        epochs=args.epochs or train_cfg.get("epochs", 100),
        batch_size=args.batch_size or train_cfg.get("batch_size", 128),
        hid_units=args.hid_units or model_cfg.get("hid_units", 256),
        lr=args.lr or train_cfg.get("lr", 1e-3),
        lr_gamma=args.lr_gamma or train_cfg.get("lr_gamma", 0.97),
        eval_every=args.eval_every or train_cfg.get("eval_every", 2),
        label_eps=label_eps,
        device=device,
        out_dir=out_dir,
        # Carry the featurization settings through to the checkpoint so that
        # PolyCardEstimator can reproduce them exactly at inference time.
        extra_stats={
            k: label_stats[k]
            for k in (
                "num_vertices", "method", "dataset_index",
                "norm_origin", "norm_span",
            )
            if k in label_stats
        },
    )

    save_json(
        os.path.join(out_dir, "train_history.json"),
        {
            "dataset": args.dataset,
            "history": result["history"],
            "hparams": result["hparams"],
            "meta": meta,
            "min_val": min_val,
            "max_val": max_val,
        },
    )
    final = result["history"][-1] if result["history"] else float("nan")
    print(f"\nfinal training q-error loss: {final:.4f}")
    print(f"checkpoint: {out_dir}")
    return 0


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
