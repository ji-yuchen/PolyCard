# PolyCard: A Learned Cardinality Estimator for Intersection Queries on Spatial Polygons

Official implementation of **PolyCard**, a learned cardinality estimator for
intersection queries on spatial polygons.

**Paper:** Yuchen Ji, Daichi Amagata, Yuya Sasaki, Takahiro Hara.
*PolyCard: A Learned Cardinality Estimator for Intersection Queries on
Spatial Polygons.* Journal of Intelligent Information Systems **63**(3),
873--891, 2025.
[doi:10.1007/s10844-025-00921-z](https://link.springer.com/article/10.1007/s10844-025-00921-z)

> Estimating the result size of a query -- its *cardinality* -- is what a
> cost-based optimizer uses to pick a plan.  For spatial polygons the
> classical answer (histograms over minimum bounding rectangles) is fast but
> inaccurate: an MBR is a poor stand-in for a polygon.  PolyCard instead
> learns the mapping *query polygon -> cardinality* with a small MLP stack.
>
> The obstacle is that polygons have **variable vertex counts**, which neural
> networks cannot ingest directly, and that randomly generated queries have a
> **long-tailed cardinality distribution** that prevents convergence.
> PolyCard solves both: a zone-aware **adaptive sampling** transformation
> (Algorithm 1) maps any polygon to a fixed number of vertices, and a
> **training data generator** down-samples over-represented cardinality
> buckets to obtain an even label distribution.

**Properties:** accurate (up to 30% better than the baselines), fast
(~4 us per estimation), and stable across datasets of very different sizes.

---

## Table of contents

- [Quickstart (no data needed)](#quickstart-no-data-needed)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [Pipeline in detail](#pipeline-in-detail)
- [Baselines](#baselines)
- [Python API](#python-api)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

---

## Quickstart (no data needed)

Requires Python >= 3.9.

```bash
pip install -r requirements.txt
```

Then run the whole pipeline on a **synthetic toy dataset** (about a minute
on a laptop, CPU only):

```bash
bash scripts/quickstart.sh            # Linux / macOS
scripts\quickstart.bat                # Windows
```

That script is just the following six commands:

```bash
# 0. a small dataset of 4,000 synthetic polygons
python scripts/make_sample.py --synthetic --num 4000 \
       --out data/sample/toy_clean.tsv

# 1. generate 12,000 intersection queries from it
python scripts/generate_queries.py --dataset toy \
       --source data/sample/toy_clean.tsv --num-queries 12000

# 2. compute their TRUE cardinality (Shapely, no database needed)
python scripts/collect_cardinality.py --dataset toy --engine shapely \
       --dataset-tsv data/sample/toy_clean.tsv

# 3. balance the cardinality distribution + featurize
python scripts/build_training_set.py --dataset toy --num-test 1000 \
       --num-datasets 1 --dataset-index 0 --norm-from-data

# 4. train
python scripts/train.py --dataset toy --epochs 60

# 5. evaluate (q-error + estimation latency)
python scripts/evaluate.py --dataset toy
```

Expected output (numbers vary slightly with the platform):

```
--- PolyCard ---
  25th q-error : 1.38
  50th q-error : 1.96
  75th q-error : 2.86
  95th q-error : 4.69
  99th q-error : 7.62
  estimation   : 1.46 us/query
```

> **Note on `--norm-from-data`**
> The paper normalizes coordinates with fixed global lon/lat bounds
> (`[-177, -55]`, span `[357, 130]`), which is the default here.  On small or
> region-local datasets that compresses the input signal, so the quickstart
> uses dataset-specific bounds instead.  Keep the default for the real OSM
> experiments.

---

## Reproducing the paper

### 1. Get the data

Download the OSM `ways` extracts from
<https://spatialhadoop.cs.umn.edu/datasets.html> (the **OSM datasets**
section; use the *new* group) and put them in `data/raw/` under these names:

```
data/raw/cemetery      # 0.19 M polygons,  56 MB
data/raw/sports        # 1.8 M polygons,  590 MB
data/raw/parks         # 10 M polygons,    9.3 GB
data/raw/buildings     # 115 M polygons,    26 GB
```

Each file is a TSV with the schema in
[`ways_schema.tsv`](ways_schema.tsv):

```
way_id <TAB> shape(WKT) <TAB> tags
```

Optionally also download **Postal codes** (171 K polygons): the paper draws
its query polygons from that extract so that queries are not copies of the
indexed data.  Set `query_source_clean` in `configs/default.yaml` to use it;
if it is empty, query polygons are sampled from the queried dataset itself.

The data is © OpenStreetMap contributors (ODbL) and is not redistributed
here.  Exact download links and per-dataset details are in
[`docs/DATASETS.md`](docs/DATASETS.md).

### 2. Clean

```bash
for d in cemetery sports parks buildings; do
    python scripts/prepare_dataset.py --dataset $d
done
```

This drops every non-polygon row (see [Data cleaning](#data-cleaning)) and
writes `data/derived/<name>_clean.tsv` plus a PostGIS `COPY` payload.

### 3. Generate queries and their true cardinality

Either offline (Shapely, no database):

```bash
python scripts/generate_queries.py --dataset sports --num-queries 1000000
python scripts/collect_cardinality.py --dataset sports --engine shapely
```

or through PostGIS (the path used in the paper):

```bash
python scripts/generate_queries.py --dataset sports --num-queries 1000000
python scripts/collect_cardinality.py --dataset sports --engine postgis --emit-sql
psql -d osm -f data/queries/sports_count.sql > data/queries/sports_count.out
python scripts/collect_cardinality.py --dataset sports --engine postgis \
       --parse data/queries/sports_count.out
```

`Buildings` (115 M polygons) needs PostGIS -- see
[`docs/POSTGIS.md`](docs/POSTGIS.md) for the import script.

### 4. Train and evaluate

```bash
python scripts/build_training_set.py --dataset sports --num-test 1000
python scripts/train.py --dataset sports --epochs 100
python scripts/evaluate.py --dataset sports
```

Optional ablations from the paper:

```bash
# Algorithm 1 vs. uniform sampling vs. MBR (Fig. 8)
python scripts/build_training_set.py --dataset sports --method uniform_sample
python scripts/build_training_set.py --dataset sports --method mbr

# size of the transformed polygon: 4 / 10 / 25 (Fig. 9)
python scripts/build_training_set.py --dataset sports --num-vertices 4

# effect of the training data generator (Fig. 10/11)
python scripts/build_training_set.py --dataset sports --no-balance
```

---

## Repository layout

```
polycard/                 core library
    io_utils.py           WKT / TSV / npy I/O
    clean.py              OSM ways cleaning
    transform.py          Algorithm 1: polygon -> fixed-size vector
    augment.py            shifting / reflection / perturbation
    featurize.py          dataset one-hot + polygon vector, label compression
    ground_truth.py       Shapely STRtree ground truth
    model.py              PolyCardNet
    dataset.py            PyTorch Dataset / feature bundles
    train.py              training loop (q-error loss)
    evaluate.py           q-error percentiles + latency
    estimator.py          one-line inference wrapper
    metrics.py            q-error

scripts/                  executable pipeline (each step is standalone)
    prepare_dataset.py        1. clean a raw OSM dump
    generate_queries.py       2. generate intersection queries
    collect_cardinality.py    3. true cardinality (shapely | postgis)
    build_training_set.py     4. balance + featurize + split
    train.py                  5. train
    evaluate.py               6. q-error + latency
    make_sample.py            synthetic toy data / subsampling
    baseline_postgresql.py    PostgreSQL baseline (EXPLAIN)
    baseline_cdhistogram.py   CDHistogram baseline adapters
    quickstart.sh / .bat      the six commands above

configs/default.yaml      all hyperparameters in one place
tests/                    unit tests
docs/                     dataset and PostGIS notes
```

Every script supports `--help` and accepts `--config <file>`.

---

## Pipeline in detail

| Step | Input | Output | What it does |
|------|-------|--------|--------------|
| 1 | `data/raw/<name>` | `data/derived/<name>_clean.tsv` | keeps simple `POLYGON`s only |
| 2 | cleaned TSV | `data/queries/<name>_queries.wkt` | samples real polygons and augments them |
| 3 | queries + dataset | `data/queries/<name>_cards.npy` | true cardinality, Shapely or PostGIS |
| 4 | queries + cards | `data/features/<name>/{train,test}/` | balances the label distribution, runs Algorithm 1, one-hot encodes |
| 5 | features | `results/<name>/polycard.pt` | trains with the q-error loss |
| 6 | checkpoint | `results/<name>/metrics_test.json` | q-error percentiles + us/query |

### Data cleaning

A raw OSM dump mixes geometry types.  Only simple polygons are kept:

| WKT type | Action |
|----------|--------|
| `POLYGON` | **kept** |
| `MULTIPOLYGON` | dropped -- a relation of disjoint rings, no single vertex list |
| `LINESTRING` | dropped -- linear features (roads, rivers) |
| `GEOMETRYCOLLECTION` | dropped -- heterogeneous |

Note that `if "POLYGON" in line` also matches `MULTIPOLYGON`; the cleaner
reads the leading WKT token instead (`polycard.io_utils.geometry_type`).

### Polygon transformation (Algorithm 1)

`polycard/transform.py` resamples any polygon to exactly `n` vertices:

* **`n < m`** -- *adaptive sampling*.  The MBR is split into four quadrants
  and vertices are scanned with a uniform budget of `m / n`, but the first
  vertex seen in each occupied quadrant is always kept.  This preserves thin,
  distant parts of the polygon that plain stride sampling drops (Fig. 5).
* **`n >= m`** -- uniform interpolation inserts the missing vertices.

Zero padding is *not* used: it would force every polygon up to the largest
vertex count in the dataset (tens of thousands of dimensions, >150 GB for the
training set -- see Fig. 7).

### Training data generator

Randomly generated queries are dominated by low cardinalities
(a long tail).  `build_training_set.py` drops zero-cardinality queries and
then down-samples `log(cardinality)` buckets to the size of the smallest one,
producing the even distribution of Fig. 4(b) -- which is what lets training
converge (Fig. 10).

---

## Baselines

See [`baselines/README.md`](baselines/README.md) for how to obtain and run
each competitor.  In short:

```bash
# PostgreSQL -- read the planner's estimate out of EXPLAIN (ANALYZE)
python scripts/baseline_postgresql.py --dataset sports --emit
psql -d osm -f data/baselines/sports_explain.sql > out.txt 2>&1
python scripts/baseline_postgresql.py --dataset sports --parse out.txt

# CDHistogram -- we ship the MBR input adapter and the output parser
python scripts/baseline_cdhistogram.py --dataset sports --emit
#   ... run the CDHistogram binary ...
python scripts/baseline_cdhistogram.py --dataset sports --parse cd_out.txt
```

Then compare everything at once:

```bash
python scripts/evaluate.py --dataset sports \
    --baseline-postgresql data/baselines/sports_pg_est.txt \
    --baseline-cdhistogram data/baselines/sports_cd_est.txt
```

---

## Python API

```python
from polycard import PolyCardEstimator

est = PolyCardEstimator.load("results/sports")
est.estimate("POLYGON ((-120.42 34.69, -120.42 34.70, ...))")   # -> 4213
est.estimate_many([wkt1, wkt2, wkt3])                           # -> array
est.benchmark_latency(num_samples=1000)                         # -> us/query
```

Lower-level pieces are usable on their own:

```python
from polycard import transform, augment, ground_truth, metrics

transform.transform_polygon(wkt, num_vertices=10, method="adaptive")
augment.augment_polygon(coords, method="shifting")
ground_truth.PolygonIndex(dataset_polygons).count(query_polygon)
metrics.qerror_percentiles(preds, truths)
```

---

## Datasets

| Name | Polygons | Avg. vertices | Role |
|------|----------|---------------|------|
| Cemetery | 0.19 M | 10.3 | queried dataset |
| Sports | 1.8 M | 11.3 | queried dataset |
| Parks | 10 M | 35.5 | queried dataset |
| Buildings | 115 M | 7.0 | queried dataset |
| Postal codes | 171 K | -- | source of query polygons (optional) |

All from the **OSM datasets** section of
<https://spatialhadoop.cs.umn.edu/datasets.html>, extracted from
`planet.osm` by the Pigeon scripts.  See
[`docs/DATASETS.md`](docs/DATASETS.md) for download file ids, the exact TSV
schema, the geometry types that get dropped, and the disk budget.

---

## Tests

```bash
python -m unittest discover -s tests
```

26 tests cover WKT parsing, the four transformation methods, the quadrant
coverage guarantee of Algorithm 1, cleaning, augmentation, q-error, and the
ground truth against a brute-force check.

---

## Citation

If you use PolyCard, please cite the paper:

```bibtex
@article{ji2025polycard,
  title   = {PolyCard: A Learned Cardinality Estimator for Intersection
             Queries on Spatial Polygons},
  author  = {Ji, Yuchen and Amagata, Daichi and Sasaki, Yuya and Hara, Takahiro},
  journal = {Journal of Intelligent Information Systems},
  volume  = {63},
  number  = {3},
  pages   = {873--891},
  year    = {2025},
  doi     = {10.1007/s10844-025-00921-z},
  url     = {https://link.springer.com/article/10.1007/s10844-025-00921-z}
}
```

The MLP skeleton (`polycard/model.py`) follows the set-convolution model of
MSCN (Kipf et al., *Learned Cardinalities: Estimating Correlated Joins with
Deep Learning*, CIDR 2019).  Everything polygon-specific -- the
transformation, the augmentation and the featurization -- is original.

---

## License

MIT -- see [LICENSE](LICENSE).  The baseline implementations of CDHistogram
and RI are **not** included and carry their own licenses.
