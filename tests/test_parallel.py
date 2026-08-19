"""Tests for parallel execution."""

import unittest
import numpy as np

from pyrweq.core import compute_rweq


class TestParallel(unittest.TestCase):
    def _inputs(self):
        shape = (10, 10)
        return dict(
            wind_speed=np.full(shape, 12.0, dtype=np.float32),
            precip=np.full(shape, 2.0, dtype=np.float32),
            temp=np.full(shape, 20.0, dtype=np.float32),
            elevation=np.full(shape, 1.0, dtype=np.float32),
            potential_et=np.full(shape, 5.0, dtype=np.float32),
            snow_depth=np.zeros(shape, dtype=np.float32),
            sand_content=np.full(shape, 70.0, dtype=np.float32),
            silt_content=np.full(shape, 20.0, dtype=np.float32),
            clay_content=np.full(shape, 10.0, dtype=np.float32),
            organic_matter=np.full(shape, 1.0, dtype=np.float32),
            ndvi=np.full(shape, 0.3, dtype=np.float32),
        )

    def test_sequential_vs_parallel_equivalent(self):
        inputs = self._inputs()
        result_seq = compute_rweq(**inputs, n_workers=1, backend="numpy")
        result_par = compute_rweq(**inputs, n_workers=5, backend="numpy")
        self.assertTrue(np.allclose(result_seq.sl, result_par.sl))
        self.assertTrue(np.allclose(result_seq.wf, result_par.wf))
        self.assertTrue(np.allclose(result_seq.ef, result_par.ef))
        self.assertTrue(np.allclose(result_seq.scf, result_par.scf))
        self.assertTrue(np.allclose(result_seq.k_prime, result_par.k_prime))
        self.assertTrue(np.allclose(result_seq.c, result_par.c))

    def test_parallel_shape(self):
        inputs = self._inputs()
        result = compute_rweq(**inputs, n_workers=3, backend="numpy")
        self.assertEqual(result.sl.shape, (10, 10))
        self.assertTrue(np.all(result.sl >= 0))


class TestDask(unittest.TestCase):
    def test_backend_missing_package(self):
        """Simulate missing dask by temporarily removing from path."""
        import importlib
        if importlib.util.find_spec("dask") is not None:
            self.skipTest("dask is installed; cannot test missing-package path")
        shape = (10, 10)
        with self.assertRaises(ImportError) as ctx:
            compute_rweq(
                wind_speed=np.full(shape, 12.0, dtype=np.float32),
                precip=np.full(shape, 2.0, dtype=np.float32),
                temp=np.full(shape, 20.0, dtype=np.float32),
                elevation=np.full(shape, 1.0, dtype=np.float32),
                potential_et=np.full(shape, 5.0, dtype=np.float32),
                snow_depth=np.zeros(shape, dtype=np.float32),
                sand_content=np.full(shape, 70.0, dtype=np.float32),
                silt_content=np.full(shape, 20.0, dtype=np.float32),
                clay_content=np.full(shape, 10.0, dtype=np.float32),
                organic_matter=np.full(shape, 1.0, dtype=np.float32),
                ndvi=np.full(shape, 0.3, dtype=np.float32),
                backend="dask",
            )
        self.assertIn("pip install dask[array]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
