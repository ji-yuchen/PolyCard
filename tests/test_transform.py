"""Unit tests for polygon parsing, transformation and normalization."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polycard import io_utils as I  # noqa: E402
from polycard import transform as T  # noqa: E402

SQUARE = "POLYGON ((0.0 0.0, 0.0 1.0, 1.0 1.0, 1.0 0.0, 0.0 0.0))"


def _ring(n=12, radius=1.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang) * radius, np.sin(ang) * radius])


class TestWKT(unittest.TestCase):
    def test_parse_roundtrip(self):
        coords = I.wkt_to_coords(SQUARE)
        self.assertEqual(coords.shape, (5, 2))
        out = I.wkt_to_coords(I.coords_to_wkt(coords))
        np.testing.assert_allclose(out, coords, atol=1e-9)

    def test_geometry_type_distinguishes_multipolygon(self):
        self.assertEqual(I.geometry_type(SQUARE), "POLYGON")
        self.assertEqual(
            I.geometry_type("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))"), "MULTIPOLYGON"
        )
        self.assertEqual(I.geometry_type("LINESTRING (0 0, 1 1)"), "LINESTRING")
        self.assertEqual(I.geometry_type("GEOMETRYCOLLECTION (POINT (0 0))"),
                         "GEOMETRYCOLLECTION")

    def test_open_ring_is_closed_on_write(self):
        wkt = I.coords_to_wkt(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
        coords = I.wkt_to_coords(wkt)
        np.testing.assert_allclose(coords[0], coords[-1])


class TestTransform(unittest.TestCase):
    def test_output_shape(self):
        coords = _ring(40)
        for method in T.METHODS:
            for n in (4, 10, 25):
                out = T.transform_polygon(coords, n, method, flatten=True)
                self.assertEqual(out.shape, (2 * n,))
                out2 = T.transform_polygon(coords, n, method, flatten=False)
                self.assertEqual(out2.shape, (n, 2))

    def test_short_polygon_is_interpolated(self):
        """m <= n: every method falls back to interpolation."""
        coords = _ring(5)
        self.assertEqual(T.adaptive_resample(coords, 10).shape, (10, 2))
        self.assertEqual(T.uniform_stride_sample(coords, 10).shape, (10, 2))

    def test_adaptive_is_sampling_when_m_gt_n(self):
        coords = _ring(60)
        out = T.adaptive_resample(coords, 10)
        self.assertEqual(out.shape, (10, 2))
        # every returned vertex must be an actual vertex of the input
        for v in out:
            self.assertTrue(np.any(np.all(np.isclose(coords, v), axis=1)))

    def test_adaptive_covers_all_occupied_zones(self):
        """The whole point of Algorithm 1: no non-empty quadrant is lost."""
        # A polygon with a far-away spike in the top-right quadrant.
        coords = np.array(
            [
                [0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.05],
                [0.05, 0.05], [0.05, 0.06], [0.0, 0.06],
                [10.0, 10.0], [10.1, 10.0], [10.1, 10.1],
            ]
        )
        out = T.adaptive_resample(coords, 8)
        cx, cy = T.mbr_center(coords)
        zones = {T.check_zone(cx, cy, x, y) for x, y in out}
        occupied = {T.check_zone(cx, cy, x, y) for x, y in coords}
        self.assertEqual(zones, occupied)

    def test_adaptive_rejects_n_below_four(self):
        with self.assertRaises(ValueError):
            T.adaptive_resample(_ring(40), 3)

    def test_mbr_approximation(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 2.0], [0.0, 0.0]])
        out = T.mbr_approximate(coords, 10)
        self.assertEqual(out.shape, (10, 2))
        # MBR of the input triangle
        self.assertAlmostEqual(out[:, 0].max(), 1.0)
        self.assertAlmostEqual(out[:, 1].max(), 2.0)

    def test_normalization_bounds(self):
        coords = np.array([[-177.0, -55.0], [180.0, 75.0]])
        out = T.normalize(coords)
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[1], [1.0, 1.0])

    def test_transform_accepts_wkt(self):
        out = T.transform_polygon(SQUARE, 10)
        self.assertEqual(out.shape, (20,))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            T.transform_polygon(SQUARE, 10, method="does-not-exist")


class TestZone(unittest.TestCase):
    def test_quadrant_index(self):
        self.assertEqual(T.check_zone(0, 0, -1, -1), 0)  # bottom-left
        self.assertEqual(T.check_zone(0, 0, -1, 1), 1)   # top-left
        self.assertEqual(T.check_zone(0, 0, 1, 1), 2)    # top-right
        self.assertEqual(T.check_zone(0, 0, 1, -1), 3)   # bottom-right


if __name__ == "__main__":
    unittest.main()
