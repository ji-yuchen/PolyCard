"""Unit tests for cleaning, augmentation, metrics and the offline ground truth."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polycard import augment as A  # noqa: E402
from polycard import clean as C  # noqa: E402
from polycard import ground_truth as G  # noqa: E402
from polycard import io_utils as I  # noqa: E402
from polycard import metrics as M  # noqa: E402

POLY = "POLYGON ((0.0 0.0, 0.0 1.0, 1.0 1.0, 1.0 0.0, 0.0 0.0))"
LINE = "LINESTRING (0.0 0.0, 1.0 1.0)"
MULTI = "MULTIPOLYGON (((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 0.0)))"
COLLECTION = "GEOMETRYCOLLECTION (POINT (0.0 0.0))"


class TestClean(unittest.TestCase):
    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for i, wkt in enumerate(rows):
                fh.write(f"{i}\t{wkt}\ttag\n")
        return path

    def test_only_simple_polygons_survive(self):
        src = self._write([POLY, LINE, MULTI, COLLECTION, POLY])
        dst = src + ".clean.tsv"
        try:
            stats = C.clean_dataset_file(src, dst, log=lambda *a: None)
            kept = list(I.iter_ways(dst))
            self.assertEqual(len(kept), 2)
            self.assertEqual(stats.kept, 2)
            self.assertEqual(stats.by_type["LINESTRING"], 1)
            self.assertEqual(stats.by_type["MULTIPOLYGON"], 1)
            self.assertEqual(stats.by_type["GEOMETRYCOLLECTION"], 1)
            for _id, wkt, _tags in kept:
                self.assertEqual(I.geometry_type(wkt), "POLYGON")
        finally:
            os.remove(src)
            if os.path.exists(dst):
                os.remove(dst)

    def test_malformed_rows_are_dropped_not_fatal(self):
        src = self._write([POLY, "POLYGON ((", POLY])
        dst = src + ".clean.tsv"
        try:
            stats = C.clean_dataset_file(src, dst, log=lambda *a: None)
            self.assertEqual(stats.kept, 2)
            self.assertEqual(stats.dropped_malformed, 1)
        finally:
            os.remove(src)
            if os.path.exists(dst):
                os.remove(dst)

    def test_min_vertices_filter(self):
        tiny = "POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 0.0))"  # 4 rows
        src = self._write([tiny])
        dst = src + ".clean.tsv"
        try:
            stats = C.clean_dataset_file(src, dst, min_vertices=5,
                                         log=lambda *a: None)
            self.assertEqual(stats.kept, 0)
            self.assertEqual(stats.dropped_too_small, 1)
        finally:
            os.remove(src)
            if os.path.exists(dst):
                os.remove(dst)


class TestAugment(unittest.TestCase):
    def setUp(self):
        self.coords = I.wkt_to_coords(POLY)

    def test_no_in_place_modification(self):
        before = self.coords.copy()
        for method in A.AUGMENTATIONS:
            A.augment_polygon(self.coords, method=method)
        np.testing.assert_allclose(self.coords, before)

    def test_ring_stays_closed(self):
        for method in A.AUGMENTATIONS:
            out = A.augment_polygon(self.coords, method=method)
            np.testing.assert_allclose(out[0], out[-1])

    def test_reflection_preserves_mbr(self):
        out = A.augment_polygon(self.coords, method="reflection")
        np.testing.assert_allclose(out.min(axis=0), self.coords.min(axis=0))
        np.testing.assert_allclose(out.max(axis=0), self.coords.max(axis=0))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            A.augment_polygon(self.coords, method="nope")


class TestMetrics(unittest.TestCase):
    def test_qerror_is_symmetric(self):
        np.testing.assert_allclose(M.qerror([10], [5]), [2.0])
        np.testing.assert_allclose(M.qerror([5], [10]), [2.0])

    def test_zero_entries_are_skipped(self):
        # (0, 5) is dropped because the estimate is zero.
        self.assertEqual(M.qerror([0, 4], [5, 5]).size, 1)
        # (4, 0) is dropped because the true cardinality is zero.
        self.assertEqual(M.qerror([4, 7], [0, 5]).size, 1)
        # Nothing survives when every estimate is zero.
        self.assertEqual(M.qerror([0, 0], [5, 3]).size, 0)

    def test_percentiles(self):
        preds = np.full(1000, 10.0)
        truths = np.full(1000, 10.0)
        s = M.qerror_percentiles(preds, truths)
        for p in (25, 50, 75, 95, 99):
            self.assertAlmostEqual(s[p], 1.0)

    def test_summary_has_expected_keys(self):
        s = M.summarize_qerror([1, 2, 3], [1, 2, 4])
        for k in ("25th", "50th", "75th", "95th", "99th", "mean", "max",
                  "num_queries"):
            self.assertIn(k, s)


@unittest.skipUnless(G.HAS_SHAPELY, "shapely not installed")
class TestGroundTruth(unittest.TestCase):
    def test_count_matches_brute_force(self):
        rng = np.random.default_rng(0)
        polys = []
        for _ in range(50):
            c = rng.uniform(0, 10, 2)
            ang = np.sort(rng.uniform(0, 2 * np.pi, 6))
            r = rng.uniform(0.1, 1.0, 6)
            pts = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
            polys.append(np.vstack([pts, pts[:1]]))

        index = G.PolygonIndex(polys)
        query = np.array([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0], [4.0, 4.0]])

        expected = sum(
            1 for p in polys if G.to_shapely(p).intersects(G.to_shapely(query))
        )
        self.assertEqual(index.count(query), expected)

    def test_containment_counts_as_intersection(self):
        """A query fully inside a dataset polygon must be counted."""
        big = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
                        [0.0, 0.0]])
        small = np.array([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0],
                          [4.0, 4.0]])
        self.assertEqual(G.count_intersections([big], small), 1)


if __name__ == "__main__":
    unittest.main()
