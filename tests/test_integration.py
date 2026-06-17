"""Integration tests using synthetic GeoTIFF data."""

import os
import unittest
import tempfile
import numpy as np

from pyrweq.io import write_raster, read_raster
from pyrweq.core import compute_rweq
from pyrweq.classify import classify_erosion
from pyrweq.stats import zonal_stats


def _make_profile():
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 10,
        "height": 10,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": (0.01, 0.0, 100.0, 0.0, -0.01, 40.0),
        "nodata": -9999.0,
    }


def _write_test_raster(tmpdir, name, values):
    path = os.path.join(tmpdir, f"{name}.tif")
    write_raster(values.astype(np.float32), _make_profile(), path)
    return path


class TestComputeRWEQ(unittest.TestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape = (10, 10)
            wind = _write_test_raster(tmp, "wind", np.full(shape, 12.0))
            precip = _write_test_raster(tmp, "precip", np.full(shape, 2.0))
            temp = _write_test_raster(tmp, "temp", np.full(shape, 20.0))
            elev = _write_test_raster(tmp, "elev", np.full(shape, 1.0))
            pet = _write_test_raster(tmp, "pet", np.full(shape, 5.0))
            snow = _write_test_raster(tmp, "snow", np.zeros(shape))
            sand = _write_test_raster(tmp, "sand", np.full(shape, 70.0))
            silt = _write_test_raster(tmp, "silt", np.full(shape, 20.0))
            clay = _write_test_raster(tmp, "clay", np.full(shape, 10.0))
            om = _write_test_raster(tmp, "om", np.full(shape, 1.0))
            ndvi_r = _write_test_raster(tmp, "ndvi", np.full(shape, 0.3))

            out_dir = os.path.join(tmp, "output")
            result = compute_rweq(
                wind_speed=wind, precip=precip, temp=temp,
                elevation=elev, potential_et=pet, snow_depth=snow,
                sand_content=sand, silt_content=silt, clay_content=clay,
                organic_matter=om, ndvi=ndvi_r, output_dir=out_dir,
            )

            self.assertEqual(result.sl.shape, shape)
            self.assertTrue(np.all(result.sl >= 0))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "sl.tif")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "wf.tif")))


class TestClassify(unittest.TestCase):
    def test_china_standard(self):
        sl = np.array([[100, 300, 6000, 16000]])
        classified, labels = classify_erosion(sl)
        self.assertEqual(classified[0, 0], 1)  # 微度
        self.assertEqual(classified[0, 1], 2)  # 轻度
        self.assertEqual(classified[0, 2], 4)  # 强烈
        self.assertEqual(classified[0, 3], 6)  # 剧烈


class TestZonalStats(unittest.TestCase):
    def test_basic(self):
        sl = np.array([[100, 200], [300, 400]], dtype=np.float64)
        zones = np.array([[1, 1], [2, 2]], dtype=np.float64)
        stats = zonal_stats(sl, zones)
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["zone"], 1)
        self.assertAlmostEqual(stats[0]["mean"], 150.0, places=1)
        self.assertAlmostEqual(stats[1]["mean"], 350.0, places=1)


if __name__ == "__main__":
    unittest.main()
