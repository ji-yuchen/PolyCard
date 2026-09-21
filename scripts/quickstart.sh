#!/usr/bin/env bash
# End-to-end PolyCard run on a synthetic toy dataset -- no downloads needed.
#
#   bash scripts/quickstart.sh
#
# Takes about one minute on a laptop (CPU only).  See README.md for the
# real OSM datasets and for what each step does.
set -euo pipefail

cd "$(dirname "$0")/.."

# Pick an interpreter: honour $PYTHON, else take the first one that exists.
if [ -z "${PYTHON:-}" ]; then
    for cand in python3 python py; do
        if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
    done
fi
PYTHON="${PYTHON:-python}"

echo "using interpreter: $PYTHON"

echo "==> 0/5  synthetic toy dataset (4,000 polygons)"
$PYTHON scripts/make_sample.py --synthetic --num 4000 \
    --out data/sample/toy_clean.tsv

echo "==> 1/5  generate 12,000 intersection queries"
$PYTHON scripts/generate_queries.py --dataset toy \
    --source data/sample/toy_clean.tsv --num-queries 12000

echo "==> 2/5  true cardinality (Shapely)"
$PYTHON scripts/collect_cardinality.py --dataset toy --engine shapely \
    --dataset-tsv data/sample/toy_clean.tsv

echo "==> 3/5  balance + featurize"
$PYTHON scripts/build_training_set.py --dataset toy --num-test 1000 \
    --num-datasets 1 --dataset-index 0 --norm-from-data

echo "==> 4/5  train (60 epochs)"
$PYTHON scripts/train.py --dataset toy --epochs 60

echo "==> 5/5  evaluate"
$PYTHON scripts/evaluate.py --dataset toy
