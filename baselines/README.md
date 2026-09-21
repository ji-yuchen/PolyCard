# Baselines

PolyCard is compared against three competitors.  Two of them are external
C++ projects that are **not** redistributed here; this repository ships the
adapters that convert to and from their file formats, plus the evaluation
glue that turns their output into q-errors.

| Baseline | Type | Status here |
|----------|------|-------------|
| PostgreSQL 16 | ready to use | full support (EXPLAIN generation + parsing) |
| CDHistogram | external C++ | input/output adapters only |
| RI | external C++ | pointers only |

---

## 1. PostgreSQL

PostgreSQL estimates the result size of `ST_Intersects` from the PostGIS
geometry statistics.  The estimate is read out of `EXPLAIN (ANALYZE)`.

```bash
# 1. write the EXPLAIN script
python scripts/baseline_postgresql.py --dataset sports --emit

# 2. run it
psql -d osm -f data/baselines/sports_explain.sql > data/baselines/sports.out 2>&1

# 3. parse estimates, actual rows and planning time
python scripts/baseline_postgresql.py --dataset sports --parse data/baselines/sports.out
```

Output:

* `data/baselines/<name>_pg_est.txt` -- one estimate per line, aligned with
  `data/queries/<name>_queries.wkt`
* `data/baselines/<name>_pg_act.txt` -- the actual row counts (sanity check)
* `data/baselines/<name>_pg_report.json` -- planning time + q-error summary

The plan node that is read is matched by `--node-pattern`, defaulting to
`(Index Scan|Index Only Scan|Bitmap Heap Scan|Seq Scan|Sample Scan)`.  For
large tables PostgreSQL may switch to a parallel plan; pass e.g.
`--node-pattern "Parallel Seq Scan on sports"` if no rows are found.

PostgreSQL does not expose estimation time, so **planning time** is used as
its proxy, exactly as in the paper.

---

## 2. CDHistogram

Jin, An, Sivasubramaniam: *Analyzing Range Queries on Spatial Data*,
VLDB 2000.

CDHistogram is a histogram for **rectangles**.  PolyCard extends it to
polygons by feeding it each query's **minimum bounding rectangle** -- the
approximation that costs it accuracy (Section 5.2 of the paper).

```bash
# 1. write the query file (index <TAB> POLYGON((...MBR...)))
python scripts/baseline_cdhistogram.py --dataset sports --emit

# 2. run the CDHistogram binary with 10,000 buckets per dimension and
#    the dataset loaded from data/derived/sports_clean.tsv

# 3. parse its output (one estimated cardinality per line)
python scripts/baseline_cdhistogram.py --dataset sports --parse cd_out.txt
```

If your build accepts true polygons rather than rectangles, use
`--raw-wkt` to emit the unmodified query polygons.

`scripts/baseline_cdhistogram.py --parse` is tolerant: it accepts either a
bare number per line or `<id> <TAB> <value>`, and skips `#` comments.

---

## 3. RI (Raster Interval)

Georgiadis, Mamoulis, Tsitsigkos, Terrovitis: *Raster Intervals: An
Efficient, Lossless Representation for Spatial Polygon Data*, SIGMOD 2023.

RI is a lossless-ish raster approximation designed for **intersection
joins**.  Used as a cardinality estimator it is very accurate but far too
slow -- in the paper it is slower than simply executing the query, which
defeats the purpose of estimation (estimation happens *before* execution).

We therefore only report it for completeness.  Obtain the authors' C++
implementation from their repository, build it, and run it on
`data/queries/<name>_queries.wkt`; then feed the resulting estimates to
`scripts/evaluate.py --baseline-ri <file>`.

---

## Comparing everything

```bash
python scripts/evaluate.py --dataset sports \
    --baseline-postgresql data/baselines/sports_pg_est.txt \
    --baseline-cdhistogram data/baselines/sports_cd_est.txt
```

This prints the q-error percentiles of PolyCard and of every baseline whose
estimate file was supplied, and writes
`results/<name>/metrics_test.json`.

All baseline files must be **aligned** with
`data/features/<name>/test/cardinality.npy` -- i.e. one estimate per test
query, in the same order.  Since the test set is the first `num_test`
entries of the balanced query pool, generate the baseline estimates from the
full query file and let `evaluate.py` truncate to the common prefix.
