# Datasets

## Source

All datasets come from the **OSM datasets** section of the SpatialHadoop
project page:

<https://spatialhadoop.cs.umn.edu/datasets.html>

They are extracted from OpenStreetMap's `planet.osm` and converted to
tab-separated text by the [Pigeon](https://github.com/aseldawy/pigeon)
scripts, so no XML parsing is needed.

The page publishes two groups ("New datasets" and "Old datasets").
PolyCard uses the **new** group -- the one whose record counts match Table 3
of the paper.

## Datasets used

| Dataset | Records | Raw size | Compressed download | Role |
|---------|---------|----------|---------------------|------|
| Cemetery | 0.19 M | 56 MB | 17.4 MB | queried dataset |
| Sports | 1.8 M | 590 MB | 172 MB | queried dataset |
| Parks | 10 M | 9.3 GB | 2.9 GB | queried dataset |
| Buildings | 115 M | 26 GB | 6 GB | queried dataset |
| Postal codes | 171 K | 1.4 GB | 477 MB | **source of query polygons** |

The other extracts on the page (All Nodes, All Objects, Lakes, Roads,
Road network) are not used.

> The page lists Cemetery as "193M" records, which is a typo -- 56 MB cannot
> hold 193 M polygons.  The real count is 0.19 M, matching Table 3 of the
> paper.  `scripts/prepare_dataset.py` prints the exact count after cleaning.

### Download

The files are distributed through Google Drive:

| Dataset | Google Drive file id |
|---------|----------------------|
| Cemetery | `0B1jY75xGiy7eVEFXTTRrSUNoR3M` |
| Sports | `0B1jY75xGiy7edTI1WmtxdHF2Vm8` |
| Parks | `0B1jY75xGiy7eTExfbkNTSGp0UTA` |
| Buildings | `0B1jY75xGiy7ecW0tTFJSczdkSzQ` |
| Postal codes | `0B1jY75xGiy7eNF9SWEFJeXlVSjg` |

Fetch one with `https://drive.google.com/uc?id=<file_id>` (for example with
`gdown`), decompress it, and put it in `data/raw/` under the dataset name:

```
data/raw/cemetery
data/raw/sports
data/raw/parks
data/raw/buildings
```

`Postal codes` is optional: it is only needed to reproduce the paper's exact
query pool.  Point `query_source_clean` in `configs/default.yaml` at its
cleaned TSV; if that entry is empty, `generate_queries.py` samples query
polygons from the queried dataset itself, which is self-consistent and needs
no extra download.

## File format

Every dump is a TSV whose schema is published next to the data as
`ways_schema.tsv` (a copy is kept at the repository root for reference):

```
way_id <TAB> shape(WKT) <TAB> tags
```

* `way_id` -- OSM way identifier
* `shape`  -- the geometry as OGC **Well-Known Text**
* `tags`   -- the raw OSM tags of that way

### Geometry types

`shape` mixes geometry types, because an OSM "way" may be closed (an area)
or open (a linear feature).  PolyCard keeps only simple polygons:

| WKT type | Action |
|----------|--------|
| `POLYGON` | **kept** |
| `MULTIPOLYGON` | dropped -- an OSM relation of disjoint rings, no single vertex list |
| `LINESTRING` | dropped -- roads, rivers and other linear features |
| `GEOMETRYCOLLECTION` | dropped -- heterogeneous |

`prepare_dataset.py` prints the type histogram so the split can be checked:

```
total=191234  kept=189900  malformed=12  too_small=0
  types: POLYGON=189912  MULTIPOLYGON=45  LINESTRING=1277
  avg vertices/polygon=10.30  max=1240
```

## Licence

The data is © OpenStreetMap contributors, available under the **Open
Database License (ODbL)**.  It is therefore *not* redistributed here --
download it yourself and keep the attribution.  See
<https://www.openstreetmap.org/copyright>.

## Disk budget

| Stage | Cemetery | Sports | Parks | Buildings |
|-------|----------|--------|-------|-----------|
| raw dump | ~20 MB | ~200 MB | ~1.5 GB | ~10 GB |
| cleaned TSV | ~20 MB | ~200 MB | ~1.5 GB | ~9 GB |
| 1 M queries (WKT) | ~200 MB | ~200 MB | ~500 MB | ~150 MB |
| 1 M feature vectors (n=10) | 80 MB | 80 MB | 80 MB | 80 MB |

The last row is the point of Algorithm 1: the feature matrix has the same
size for every dataset, whereas zero padding would need
`1e6 x 2 x max_vertices x 4` bytes -- over 150 GB for `Parks` (Fig. 7).

## Generating a small dataset instead

If you only want to exercise the code:

```bash
python scripts/make_sample.py --synthetic --num 4000 \
    --out data/sample/toy_clean.tsv
```

or subsample a real dump:

```bash
python scripts/make_sample.py --from-raw data/raw/cemetery --num 5000 \
    --out data/sample/cemetery_small.tsv
```

The synthetic generator produces *clustered* polygons on purpose: uniformly
distributed data gives every location the same density, so the task
degenerates into "measure the query area", which is not what real OSM data
looks like.
