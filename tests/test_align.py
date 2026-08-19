"""Tests for input grid alignment (align_inputs)."""

import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pyrweq.io import align_inputs, read_raster, write_raster


def _write_tif(path, data, crs="EPSG:4326", res=0.01, x0=100.0, y1=40.0, nodata=-9999.0):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32" if np.issubdtype(data.dtype, np.floating) else "int32",
        "crs": crs,
        "transform": from_origin(x0, y1, res, res),
        "nodata": nodata,
    }
    write_raster(data.astype(np.float32), profile, path)
    return path


class TestAlignInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_grid_no_warp(self):
        p_ref = _write_tif(os.path.join(self.dir, "ref.tif"), np.arange(25, dtype=np.float32).reshape(5, 5))
        p_other = _write_tif(os.path.join(self.dir, "other.tif"), np.ones((5, 5), dtype=np.float32))
        with self.assertLogs("pyrweq.io", level="INFO") as cm:
            data, profile = align_inputs({"wind_speed": p_ref, "precip": p_other})
        self.assertTrue(any("already on the reference grid" in m for m in cm.output))
        self.assertEqual(data["wind_speed"].shape, (5, 5))
        ref_arr, _ = read_raster(p_ref)
        self.assertTrue(np.allclose(data["wind_speed"], ref_arr))
        self.assertEqual(profile["width"], 5)

    def test_finer_resolution_warped_to_reference(self):
        # reference: 5x5 at 0.02 deg; source: 10x10 at 0.01 deg (same window)
        ref_data = np.zeros((5, 5), dtype=np.float32)
        p_ref = _write_tif(os.path.join(self.dir, "ref.tif"), ref_data, res=0.02)
        src_data = np.tile(np.array([[10.0], [20.0]], dtype=np.float32), (1, 10)).repeat(2, 0)
        p_src = _write_tif(os.path.join(self.dir, "fine.tif"), src_data, res=0.01)
        data, _ = align_inputs({"wind_speed": p_ref, "precip": p_src})
        self.assertEqual(data["precip"].shape, (5, 5))
        vals = data["precip"][np.isfinite(data["precip"])]
        # bilinear of {10,20} stays within [10,20]
        self.assertTrue(((vals >= 9.99) & (vals <= 20.01)).all())

    def test_different_crs_aligned(self):
        # reference in EPSG:4326 covering lon 100..100.05, lat 39.95..40;
        # source in web mercator covering the same window
        p_ref = _write_tif(os.path.join(self.dir, "ref.tif"), np.zeros((5, 5), dtype=np.float32))
        import math

        def merc_x(lon):
            return lon * 20037508.34 / 180.0

        def merc_y(lat):
            return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6378137.0

        x0 = merc_x(100.0)
        dx = (merc_x(100.05) - merc_x(100.0)) / 5.0
        y1 = merc_y(40.0)
        dy = (merc_y(40.0) - merc_y(39.95)) / 5.0
        profile = {
            "driver": "GTiff", "height": 5, "width": 5, "count": 1,
            "dtype": "float32", "crs": "EPSG:3857",
            "transform": from_origin(x0, y1, dx, dy), "nodata": -9999.0,
        }
        p_src = os.path.join(self.dir, "merc.tif")
        write_raster(np.full((5, 5), 7.0, dtype=np.float32), profile, p_src)
        data, _ = align_inputs({"wind_speed": p_ref, "temp": p_src})
        self.assertEqual(data["temp"].shape, (5, 5))
        valid = data["temp"][np.isfinite(data["temp"])]
        self.assertGreater(valid.size, 0)
        self.assertTrue(np.allclose(valid, 7.0))

    def test_land_use_uses_nearest_and_keeps_codes(self):
        p_ref = _write_tif(os.path.join(self.dir, "ref.tif"), np.zeros((4, 4), dtype=np.float32), res=0.01)
        lu = np.array([[10, 20], [20, 10]], dtype=np.float32)
        p_lu = _write_tif(os.path.join(self.dir, "lu.tif"), lu, res=0.02)
        data, _ = align_inputs({"wind_speed": p_ref, "land_use": p_lu})
        codes = set(np.unique(data["land_use"][np.isfinite(data["land_use"])]).astype(int))
        self.assertTrue(codes.issubset({10, 20}))
        self.assertTrue(codes)  # not all nodata

    def test_per_key_resampling_override(self):
        p_ref = _write_tif(os.path.join(self.dir, "ref.tif"), np.zeros((5, 5), dtype=np.float32), res=0.02)
        p_src = _write_tif(os.path.join(self.dir, "v.tif"), np.full((10, 10), 3.0, dtype=np.float32), res=0.01)
        # explicit per-key spec also works for a continuous layer
        data, _ = align_inputs(
            {"wind_speed": p_ref, "precip": p_src},
            resampling={"precip": "nearest"},
        )
        self.assertEqual(data["precip"].shape, (5, 5))
        vals = data["precip"][np.isfinite(data["precip"])]
        self.assertTrue(np.allclose(vals, 3.0))

    def test_unknown_resampling_rejected(self):
        p_ref = _write_tif(os.path.join(self.dir, "ref.tif"), np.zeros((4, 4), dtype=np.float32))
        p_src = _write_tif(os.path.join(self.dir, "o.tif"), np.zeros((8, 8), dtype=np.float32), res=0.005)
        with self.assertRaises(ValueError):
            align_inputs({"wind_speed": p_ref, "precip": p_src}, resampling="lanczos")

    def test_reference_without_crs_rejected(self):
        profile = {
            "driver": "GTiff", "height": 4, "width": 4, "count": 1,
            "dtype": "float32", "crs": None,
            "transform": from_origin(0, 0, 1.0, 1.0), "nodata": -9999.0,
        }
        p_ref = os.path.join(self.dir, "nocrs.tif")
        write_raster(np.zeros((4, 4), dtype=np.float32), profile, p_ref)
        with self.assertRaises(ValueError):
            align_inputs({"wind_speed": p_ref})

    def test_default_reference_is_first_input(self):
        p_a = _write_tif(os.path.join(self.dir, "a.tif"), np.ones((6, 6), dtype=np.float32), res=0.02)
        p_b = _write_tif(os.path.join(self.dir, "b.tif"), np.ones((3, 3), dtype=np.float32), res=0.04)
        data, profile = align_inputs({"wind_speed": p_a, "precip": p_b})
        self.assertEqual(profile["width"], 6)
        self.assertEqual(data["precip"].shape, (6, 6))

    def test_aligned_output_feeds_compute_rweq(self):
        # end-to-end: mixed-resolution inputs -> align -> compute
        from pyrweq.core import compute_rweq

        rng = np.random.default_rng(5)
        p_ref = _write_tif(
            os.path.join(self.dir, "ref.tif"),
            rng.random((6, 6)).astype(np.float32) * 10 + 5,
        )
        paths = {"wind_speed": p_ref}
        for name, lo, hi in [
            ("precip", 980, 985), ("temp", 10, 25), ("elevation", 0.1, 0.9),
            ("potential_et", 1000, 1005), ("snow_depth", 0, 0),
            ("sand_content", 55, 75), ("silt_content", 15, 30),
            ("clay_content", 8, 20), ("organic_matter", 0.5, 1.5),
            ("ndvi", 0.1, 0.6),
        ]:
            paths[name] = _write_tif(
                os.path.join(self.dir, f"{name}.tif"),
                (rng.random((12, 12)) * (hi - lo) + lo).astype(np.float32),
                res=0.005,
            )
        data, _ = align_inputs(paths, reference=p_ref)
        res = compute_rweq(**data, n_workers=1)
        self.assertEqual(res.sl.shape, (6, 6))
        self.assertTrue(np.isfinite(res.sl).any())


if __name__ == "__main__":
    unittest.main()
