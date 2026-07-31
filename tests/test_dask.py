"""Tests for dask backend."""

import unittest
import numpy as np

try:
    import dask.array as da
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False

from pyrweq.core import compute_rweq


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


if __name__ == "__main__":
    unittest.main()
