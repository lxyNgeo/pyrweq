"""Tests for nodata/masking behavior."""

import unittest
import tempfile
import os

import numpy as np
import rasterio

from pyrweq.io import read_raster, read_raster_lazy, write_raster
from pyrweq.core import compute_rweq


def _make_tif(path, arr, nodata=-9999.0):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


class TestReadMasked(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "t.tif")
        arr = np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype=np.float32)
        _make_tif(self.path, arr)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_masked_default_converts_nodata_to_nan(self):
        data, profile = read_raster(self.path)
        self.assertTrue(np.isnan(data[1, 0]))
        self.assertEqual(profile["nodata"], -9999.0)

    def test_masked_false_preserves_raw(self):
        data, _ = read_raster(self.path, masked=False)
        self.assertEqual(data[1, 0].item(), -9999.0)

    def test_int_raster_promoted_to_float_when_masked(self):
        p2 = os.path.join(self.tmpdir.name, "int.tif")
        _make_tif(p2, np.array([[1, 2], [255, 4]], dtype=np.int16), nodata=255)
        data, _ = read_raster(p2)
        self.assertTrue(np.isnan(data[1, 0]))
        self.assertEqual(data.dtype, np.float64)


class TestWriteNanToNodata(unittest.TestCase):
    def test_nan_written_as_nodata(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.tif")
            profile = {"driver": "GTiff", "dtype": "float32", "height": 2,
                       "width": 2, "count": 1, "crs": None, "transform": None,
                       "nodata": -9999.0}
            arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
            write_raster(arr, profile, path)
            back, _ = read_raster(path, masked=False)
            self.assertEqual(back[0, 1].item(), -9999.0)
            self.assertEqual(back[0, 0].item(), 1.0)

    def test_nan_to_nodata_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.tif")
            profile = {"driver": "GTiff", "dtype": "float32", "height": 1,
                       "width": 1, "count": 1, "crs": None, "transform": None,
                       "nodata": -9999.0}
            write_raster(np.array([[np.nan]], dtype=np.float32), profile, path,
                         nan_to_nodata=False)
            back, _ = read_raster(path, masked=False)
            self.assertTrue(np.isnan(back[0, 0]))


class TestComputeRWEQNodata(unittest.TestCase):
    """nodata cells must not produce fake erosion values."""

    def _inputs(self, shape=(10, 10)):
        rng = np.random.default_rng(7)
        return dict(
            wind_speed=rng.random(shape).astype(np.float32) * 10 + 5,
            precip=rng.random(shape).astype(np.float32) * 5,
            temp=rng.random(shape).astype(np.float32) * 20,
            elevation=rng.random(shape).astype(np.float32),
            potential_et=rng.random(shape).astype(np.float32) * 10,
            snow_depth=np.zeros(shape, dtype=np.float32),
            sand_content=rng.random(shape).astype(np.float32) * 30 + 50,
            silt_content=rng.random(shape).astype(np.float32) * 30 + 10,
            clay_content=rng.random(shape).astype(np.float32) * 20 + 5,
            organic_matter=rng.random(shape).astype(np.float32),
            ndvi=rng.random(shape).astype(np.float32) * 0.5,
        )

    def test_nodata_cells_stay_nan_in_results(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = self._inputs()
            paths = {}
            for name, arr in inputs.items():
                arr[0, 0] = -9999.0  # one nodata cell in every raster
                p = os.path.join(d, f"{name}.tif")
                _make_tif(p, arr)
                paths[name] = p
            result = compute_rweq(**paths, n_workers=1)
            self.assertTrue(np.isnan(result.sl[0, 0]))
            self.assertTrue(np.isnan(result.wf[0, 0]))
            self.assertTrue(np.isnan(result.ef[0, 0]))
            # other cells unaffected
            self.assertTrue(np.isfinite(result.sl[5, 5]))

    def test_output_rasters_write_nodata(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = self._inputs((6, 6))
            paths = {}
            for name, arr in inputs.items():
                arr[1, 1] = -9999.0
                p = os.path.join(d, f"{name}.tif")
                _make_tif(p, arr)
                paths[name] = p
            outdir = os.path.join(d, "out")
            result = compute_rweq(**paths, n_workers=1, output_dir=outdir)
            sl, _ = read_raster(os.path.join(outdir, "sl.tif"), masked=False)
            self.assertEqual(sl[1, 1].item(), -9999.0)
            self.assertAlmostEqual(sl[2, 2].item(), float(result.sl[2, 2]), places=4)


try:
    import dask.array as da
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False


@unittest.skipUnless(_HAS_DASK, "dask not installed")
class TestLazyMasked(unittest.TestCase):
    def test_lazy_masked_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.tif")
            _make_tif(p, np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32))
            arr, _ = read_raster_lazy(p, chunks=(1, 1))
            out = arr.compute()
            self.assertTrue(np.isnan(out[0, 1]))

    def test_lazy_masked_false_raw(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.tif")
            _make_tif(p, np.array([[1.0, -9999.0]], dtype=np.float32))
            arr, _ = read_raster_lazy(p, masked=False)
            self.assertEqual(arr.compute()[0, 1].item(), -9999.0)


if __name__ == "__main__":
    unittest.main()
