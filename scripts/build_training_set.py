#!/usr/bin/env python
"""Step 4 -- turn generated queries into a training / test feature bundle.

    python scripts/build_training_set.py --dataset sports

This is the step that implements the paper's **training data generator**
(Section 4.3, Fig. 4):

1. drop queries whose true cardinality is 0 (they carry no signal);
2. bucket the remaining queries by ``log(cardinality)`` and *down-sample*
   the over-represented buckets, so the training set has an even
   cardinality distribution instead of the long tail of random queries;
3. featurize the queries -- polygon transformation (Algorithm 1) + a
   dataset one-hot vector;
4. log-compress and normalize the labels.

Output:
    ``data/features/<name>/train/``   polygon_vec.npy, dataset_vec.npy,
                                      labels_norm.npy, cardinality.npy, meta.json
    ``data/features/<name>/test/``    same
    ``data/features/<name>/label_stats.json``
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, load_config, log_header, path_in, save_json  # noqa: E402

from polycard.dataset import save_feature_bundle  # noqa: E402
from polycard.featurize import dataset_matrix, featurize_polygons, normalize_labels  # noqa: E402
from polycard.io_utils import load_array, wkt_to_coords  # noqa: E402


def balance_indices(
    cards: np.ndarray,
    num_buckets: int = 4,
    target="min",
    drop_zero: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Down-sample queries so that cardinality buckets are evenly filled.

    ``target`` is ``"min"`` (all buckets trimmed to the smallest one, the
    setting used in the paper), ``"mean"``, or an explicit integer.
    """
    rng = np.random.default_rng(seed)
    keep = np.where(cards > 0)[0] if drop_zero else np.arange(cards.shape[0])
    if keep.size == 0:
        return keep

    logged = np.log(np.asarray(cards, dtype=np.float64)[keep] + 1.0)
    lo, hi = logged.min(), logged.max()
    if hi <= lo:
        return keep

    edges = np.linspace(lo, hi, num_buckets + 1)
    bucket = np.clip(np.digitize(logged, edges[1:-1]), 0, num_buckets - 1)
    counts = np.bincount(bucket, minlength=num_buckets)
    valid = counts[counts > 0]

    if isinstance(target, str):
        size = int(valid.min()) if target == "min" else int(valid.mean())
    else:
        size = int(target)
    size = max(1, size)

    picked = []
    for k in range(num_buckets):
        idx_k = keep[bucket == k]
        if idx_k.size == 0:
            continue
        if idx_k.size > size:
            idx_k = rng.choice(idx_k, size=size, replace=False)
        picked.append(idx_k)
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def main() -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--cards", default=None)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--num-vertices", type=int, default=None)
    parser.add_argument("--method", default=None, choices=(
        "adaptive", "uniform_sample", "mbr", "interp"))
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--num-buckets", type=int, default=None)
    parser.add_argument(
        "--balance-target", default=None,
        help="min | mean | <integer> (default: min)")
    parser.add_argument("--num-datasets", type=int, default=None)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument(
        "--norm-from-data", action="store_true",
        help="normalize with the min/max of the generated queries instead of "
             "the fixed global lon/lat bounds (recommended for small/local "
             "datasets, where global normalization compresses the signal)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir
    queries_dir = path_in(data_dir, "queries")
    feat_cfg = cfg.get("featurize", {})
    bal_cfg = cfg.get("balance", {})

    num_vertices = args.num_vertices or feat_cfg.get("num_vertices", 10)
    method = args.method or feat_cfg.get("method", "adaptive")
    seed = args.seed if args.seed is not None else cfg.get("generation", {}).get("seed", 0)

    names = list(cfg.get("datasets", {}).keys())
    num_datasets = args.num_datasets or max(1, len(names))
    if args.dataset_index is not None:
        dataset_index = args.dataset_index
    else:
        dataset_index = names.index(args.dataset) if args.dataset in names else 0
    dataset_index = min(dataset_index, num_datasets - 1)

    queries_path = args.queries or os.path.join(queries_dir, f"{args.dataset}_queries.wkt")
    cards_path = args.cards or os.path.join(queries_dir, f"{args.dataset}_cards.npy")
    if not os.path.exists(queries_path) or not os.path.exists(cards_path):
        print(f"ERROR: need {queries_path} and {cards_path}")
        print("Run generate_queries.py and collect_cardinality.py first.")
        return 1

    log_header(f"build_training_set: {args.dataset}")
    with open(queries_path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    cards = load_array(cards_path).astype(np.int64).reshape(-1)
    n = min(len(lines), cards.shape[0])
    lines, cards = lines[:n], cards[:n]
    print(f"loaded {n} queries, cardinality min={cards.min()} max={cards.max()}")

    # --- 1/2. balancing ------------------------------------------------
    if bal_cfg.get("enabled", True) and not args.no_balance:
        idx = balance_indices(
            cards,
            num_buckets=args.num_buckets or bal_cfg.get("num_buckets", 4),
            target=args.balance_target or bal_cfg.get("target", "min"),
            drop_zero=bal_cfg.get("drop_zero_cardinality", True),
            seed=seed,
        )
        print(f"after balancing: {idx.size} queries (from {n})")
    else:
        idx = np.arange(n)
        print("balancing disabled")

    # --- train / test split --------------------------------------------
    num_test = min(args.num_test, max(1, idx.size - 1))
    test_idx = idx[:num_test]
    train_idx = idx[num_test:]
    if train_idx.size == 0:
        train_idx, test_idx = idx, idx[:0]
    print(f"split: train={train_idx.size}  test={test_idx.size}")

    # --- 3. featurization ----------------------------------------------
    origin = span = None
    if args.norm_from_data:
        origin, span = estimate_norm_bounds(lines, sample=20000)
        print(f"normalization from data: origin={origin} span={span}")

    def featurize(ids: np.ndarray):
        polys = [wkt_to_coords(lines[i]) for i in ids]
        vec = featurize_polygons(
            polys, num_vertices=num_vertices, method=method, origin=origin, span=span
        )
        dvec = dataset_matrix(
            np.full(ids.size, dataset_index, dtype=np.int64), num_datasets
        )
        return vec, dvec, cards[ids]

    print(f"featurizing (n={num_vertices}, method={method}) ...")
    train_poly, train_ds, train_card = featurize(train_idx)
    test_poly, test_ds, test_card = featurize(test_idx)

    # --- 4. labels ------------------------------------------------------
    train_labels, min_val, max_val = normalize_labels(train_card)
    if test_card.size:
        test_labels, _, _ = normalize_labels(test_card, min_val, max_val)
    else:
        test_labels = np.empty(0, dtype=np.float32)
    print(f"label normalization: min_val={min_val:.4f} max_val={max_val:.4f}")

    out_root = args.out_dir or os.path.join(path_in(data_dir, "features"), args.dataset)
    meta = {
        "dataset": args.dataset,
        "num_vertices": num_vertices,
        "method": method,
        "num_datasets": num_datasets,
        "dataset_index": dataset_index,
        "balanced": bool(bal_cfg.get("enabled", True) and not args.no_balance),
        "seed": seed,
    }
    save_feature_bundle(
        os.path.join(out_root, "train"), train_poly, train_ds, train_labels, train_card,
        {**meta, "split": "train", "num": int(train_idx.size)},
    )
    save_feature_bundle(
        os.path.join(out_root, "test"), test_poly, test_ds, test_labels, test_card,
        {**meta, "split": "test", "num": int(test_idx.size)},
    )
    save_json(
        os.path.join(out_root, "label_stats.json"),
        {
            "min_val": float(min_val),
            "max_val": float(max_val),
            "label_eps": cfg.get("train", {}).get("label_eps", 1.02),
            "dataset_feats": num_datasets,
            "polygon_feats": int(num_vertices * 2),
            "norm_origin": None if origin is None else [float(v) for v in origin],
            "norm_span": None if span is None else [float(v) for v in span],
            **meta,
        },
    )
    print(f"saved feature bundles -> {out_root}")
    return 0


def estimate_norm_bounds(wkt_lines, sample: int = 20000):
    """min / max coordinates of (a sample of) the query polygons.

    Sampling keeps this cheap for million-query workloads -- the bounds
    only need to be stable, not exact.
    """
    step = max(1, len(wkt_lines) // sample)
    mn = np.array([np.inf, np.inf])
    mx = np.array([-np.inf, -np.inf])
    for line in wkt_lines[::step]:
        try:
            coords = wkt_to_coords(line)
        except ValueError:
            continue
        mn = np.minimum(mn, coords.min(axis=0))
        mx = np.maximum(mx, coords.max(axis=0))
    if not np.isfinite(mn).all():
        return None, None
    span = mx - mn
    span = np.where(span <= 0, 1.0, span)
    return mn, span


if __name__ == "__main__":
    raise SystemExit(main())
