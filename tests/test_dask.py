"""Tests for dask backend."""

import unittest
import tempfile
import os
import numpy as np
import rasterio

try:
    import dask.array as da
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False

from pyrweq.core import compute_rweq
from pyrweq.io import read_raster_lazy


@unittest.skipUnless(_HAS_DASK, "dask not installed")
class TestDaskCompute(unittest.TestCase):
    def setUp(self):
        shape = (50, 50)
        chunks = (25, 25)
        self.inputs = {}
        for name in ["wind_speed", "precip", "temp", "elevation", "potential_et", "snow_depth",
                      "sand_content", "silt_content", "clay_content", "organic_matter", "ndvi"]:
            arr = np.random.rand(*shape).astype(np.float32) * 10
            if name == "wind_speed":
                arr += 5
            if name == "snow_depth":
                arr = arr * 0
            self.inputs[name] = da.from_array(arr, chunks=chunks)

    def test_factors_dask_vs_numpy(self):
        result_dask = compute_rweq(**self.inputs, backend="dask", n_workers=1)
        sl_dask = result_dask.sl.compute()

        np_inputs = {k: v.compute() if hasattr(v, "compute") else v for k, v in self.inputs.items()}
        result_np = compute_rweq(**np_inputs, backend="numpy", n_workers=1)

        self.assertTrue(np.allclose(sl_dask, result_np.sl, atol=1e-4))

    def test_dask_lazy_until_compute(self):
        result = compute_rweq(**self.inputs, backend="dask", n_workers=1)
        self.assertTrue(hasattr(result.sl, "dask"), "dask backend should return lazy arrays")

    def test_dask_backend_requires_dask_arrays(self):
        np_inputs = {k: np.ones((10, 10), dtype=np.float32) for k in self.inputs}
        with self.assertRaises(TypeError):
            compute_rweq(**np_inputs, backend="dask")


@unittest.skipUnless(_HAS_DASK, "dask not installed")
class TestLazyRasterInput(unittest.TestCase):
    """GeoTIFF path inputs under backend='dask' must read lazily and agree
    with the eager numpy path."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.paths = {}
        names = ["wind_speed", "precip", "temp", "elevation", "potential_et", "snow_depth",
                 "sand_content", "silt_content", "clay_content", "organic_matter", "ndvi"]
        shape = (40, 40)
        rng = np.random.default_rng(42)
        for name in names:
            arr = rng.random(shape).astype(np.float32) * 10
            if name == "wind_speed":
                arr += 5
            if name == "snow_depth":
                arr[:] = 0
            path = os.path.join(cls.tmpdir.name, f"{name}.tif")
            with rasterio.open(
                path, "w", driver="GTiff", height=shape[0], width=shape[1],
                count=1, dtype="float32", tiled=True, blockxsize=16, blockysize=16,
            ) as dst:
                dst.write(arr, 1)
            cls.paths[name] = path

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_lazy_read_is_lazy(self):
        arr, profile = read_raster_lazy(self.paths["wind_speed"])
        self.assertTrue(hasattr(arr, "dask"), "read_raster_lazy must return a dask array")
        self.assertEqual(arr.shape, (40, 40))
        self.assertEqual(profile["dtype"], "float32")

    def test_path_input_dask_vs_numpy_agree(self):
        result_dask = compute_rweq(**self.paths, backend="dask", n_workers=1)
        sl_dask = result_dask.sl.compute()

        result_np = compute_rweq(**self.paths, backend="numpy", n_workers=1)
        self.assertTrue(
            np.allclose(sl_dask, result_np.sl, atol=1e-4),
            "dask(path) result must match numpy(path) result",
        )

    def test_custom_chunks(self):
        arr, _ = read_raster_lazy(self.paths["ndvi"], chunks=(10, 20))
        self.assertEqual(arr.chunksize, (10, 20))

    def test_bad_chunks_rejected(self):
        with self.assertRaises(ValueError):
            read_raster_lazy(self.paths["ndvi"], chunks=(0, 10))


if __name__ == "__main__":
    unittest.main()
