# PostGIS setup (ground truth + PostgreSQL baseline)

The paper computes true cardinalities with PostGIS.  This is optional -- the
Shapely engine in `scripts/collect_cardinality.py` needs no database and is
fine for Cemetery and Sports -- but `Buildings` (115 M polygons) needs the
database.

## 1. Create the database

```sql
CREATE DATABASE osm;
\c osm
CREATE EXTENSION postgis;
```

## 2. Import a dataset

`prepare_dataset.py` writes a `COPY`-ready TSV with two columns
(`id`, `wkt`).  Load it like this:

```sql
CREATE TABLE sports (id bigint, geom geometry(Polygon, 4326));
\copy sports(id, wkt) FROM 'data/derived/sports_postgis.tsv'
```

then build the geometry column:

```sql
ALTER TABLE sports ADD COLUMN geom geometry(Polygon, 4326);
UPDATE sports SET geom = ST_GeomFromText(wkt, 4326);
ALTER TABLE sports DROP COLUMN wkt;
```

Or, in one pass, load into a text column and convert:

```sql
CREATE TABLE sports_raw (id bigint, wkt text);
\copy sports_raw(id, wkt) FROM 'data/derived/sports_postgis.tsv'
SELECT id, ST_GeomFromText(wkt, 4326) AS geom INTO sports FROM sports_raw;
DROP TABLE sports_raw;
```

For `Buildings`, load first and index afterwards -- building the index
incrementally is much slower:

```sql
CREATE INDEX sports_gix ON sports USING GIST (geom);
VACUUM ANALYZE sports;
```

`VACUUM ANALYZE` is what populates the statistics PostgreSQL uses for its own
cardinality estimates, so it is required for the PostgreSQL baseline to be
meaningful.

## 3. Compute true cardinalities

```bash
python scripts/collect_cardinality.py --dataset sports --engine postgis --emit-sql
psql -d osm -f data/queries/sports_count.sql > data/queries/sports_count.out
python scripts/collect_cardinality.py --dataset sports --engine postgis \
    --parse data/queries/sports_count.out
```

The emitted script sets `\pset tuples_only on`, so the output is one integer
per query -- which is exactly what the parser expects.

## 4. PostgreSQL baseline

```bash
python scripts/baseline_postgresql.py --dataset sports --emit
psql -d osm -f data/baselines/sports_explain.sql > data/baselines/sports.out 2>&1
python scripts/baseline_postgresql.py --dataset sports --parse data/baselines/sports.out
```

See [`baselines/README.md`](../baselines/README.md) for details, including
how to point the parser at a different plan node.

## Notes

* `ST_Intersects` uses the GiST index automatically; without it, every query
  is a sequential scan and the ground truth step will take hours.
* The `auto_explain` extension is loaded by the emitted EXPLAIN script.  It is
  optional -- remove those three `LOAD`/`SET` lines if the extension is not
  installed; the `rows=` estimates are produced by `EXPLAIN (ANALYZE)` itself.
* Cardinalities are stored as `int64` in `data/queries/<name>_cards.npy`,
  aligned line-by-line with `data/queries/<name>_queries.wkt`.
