"""Tests for validation metrics and OAT sensitivity analysis."""

import os
import tempfile
import unittest

import numpy as np
from rasterio.transform import from_origin

from pyrweq.io import write_raster
from pyrweq.validate import validate, sample_points
from pyrweq.sensitivity import oat_sensitivity


class TestValidate(unittest.TestCase):
    def test_perfect_prediction(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        m = validate(obs, obs.copy())
        self.assertEqual(m["n"], 4)
        self.assertAlmostEqual(m["r2"], 1.0)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["bias"], 0.0)
        self.assertAlmostEqual(m["nash_sutcliffe"], 1.0)

    def test_known_errors(self):
        obs = np.array([2.0, 4.0])
        pred = np.array([3.0, 3.0])
        m = validate(obs, pred)
        self.assertAlmostEqual(m["rmse"], 1.0)
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["bias"], 0.0)
        self.assertAlmostEqual(m["r2"], 0.0)  # ss_res == ss_tot here

    def test_mean_prediction_scores_zero_nse(self):
        obs = np.array([1.0, 2.0, 3.0])
        m = validate(obs, np.full(3, obs.mean()))
        self.assertAlmostEqual(m["nash_sutcliffe"], 0.0, places=6)

    def test_nan_pairs_dropped(self):
        obs = np.array([1.0, np.nan, 3.0, 5.0])
        pred = np.array([1.0, 2.0, np.nan, 5.0])
        m = validate(obs, pred)
        self.assertEqual(m["n"], 2)
        # fewer than two valid pairs is invalid
        with self.assertRaises(ValueError):
            validate(np.array([np.nan]), np.array([1.0]))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            validate(np.zeros(3), np.zeros(4))

    def test_constant_observed_zero_variance(self):
        obs = np.array([5.0, 5.0, 5.0])
        m = validate(obs, np.array([5.0, 6.0, 4.0]))
        self.assertTrue(np.isnan(m["r2"]))

    def test_large_bias_warns(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([10.0, 20.0, 30.0])
        with self.assertLogs("pyrweq.validate", level="WARNING"):
            validate(obs, pred)

    def test_underestimation_negative_bias(self):
        obs = np.array([10.0, 20.0])
        m = validate(obs, np.array([5.0, 10.0]))
        self.assertLess(m["bias"], 0.0)


class TestSamplePoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "sl.tif")
        profile = {
            "driver": "GTiff", "height": 4, "width": 4, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326",
            "transform": from_origin(100.0, 40.0, 0.01, 0.01),
            "nodata": -9999.0,
        }
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        write_raster(data, profile, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_samples_known_cells(self):
        # cell centres: (100.005, 39.995) is row 0 col 0 -> value 0
        vals = sample_points(self.path, [100.005, 100.015], [39.995, 39.995])
        self.assertAlmostEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[1], 1.0)

    def test_nodata_sampled_as_nan(self):
        profile = {
            "driver": "GTiff", "height": 2, "width": 2, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326",
            "transform": from_origin(0.0, 1.0, 0.5, 0.5), "nodata": -9999.0,
        }
        p2 = os.path.join(self.tmp.name, "nodata.tif")
        write_raster(np.array([[-9999.0, 1.0], [2.0, 3.0]], dtype=np.float32), profile, p2)
        vals = sample_points(p2, [0.25, 0.75], [0.75, 0.75])
        self.assertTrue(np.isnan(vals[0]))
        self.assertAlmostEqual(vals[1], 1.0)


class TestOATSensitivity(unittest.TestCase):
    def test_linear_function_unit_elasticity(self):
        fn = lambda x: np.array([x])  # summary == x
        res = oat_sensitivity(fn, {"x": 10.0}, deltas=(0.1, -0.1))
        for v, e in res["x"]["elasticity"].items():
            self.assertAlmostEqual(e, 1.0)

    def test_quadratic_function_elasticity_two(self):
        # finite-difference (arc) elasticity of x^2 at +10 percent is 2.1
        fn = lambda x: np.array([x ** 2])
        res = oat_sensitivity(fn, {"x": 4.0}, deltas=(0.1,))
        (e,) = res["x"]["elasticity"].values()
        self.assertAlmostEqual(e, 2.1)

    def test_explicit_values_override_deltas(self):
        fn = lambda x: np.array([x])
        res = oat_sensitivity(fn, {"x": 1.0}, values={"x": [1.0, 2.0, 3.0]})
        self.assertEqual(res["x"]["values"], [1.0, 2.0, 3.0])
        self.assertEqual(res["x"]["summary"], [1.0, 2.0, 3.0])

    def test_zero_base_requires_explicit_values(self):
        with self.assertRaises(ValueError):
            oat_sensitivity(lambda x: np.array([x]), {"x": 0.0})

    def test_unknown_values_key_rejected(self):
        with self.assertRaises(ValueError):
            oat_sensitivity(lambda x: np.array([x]), {"x": 1.0}, values={"y": [1.0]})

    def test_unknown_summary_rejected(self):
        with self.assertRaises(ValueError):
            oat_sensitivity(lambda x: np.array([x]), {"x": 1.0}, summary="median")

    def test_empty_params_rejected(self):
        with self.assertRaises(ValueError):
            oat_sensitivity(lambda x: np.array([x]), {})

    def test_two_parameters_independent(self):
        fn = lambda x, y: np.array([x * y])
        res = oat_sensitivity(fn, {"x": 2.0, "y": 3.0}, deltas=(0.1,))
        # d(xy)/xy per param = 1 for both
        (ex,) = res["x"]["elasticity"].values()
        (ey,) = res["y"]["elasticity"].values()
        self.assertAlmostEqual(ex, 1.0)
        self.assertAlmostEqual(ey, 1.0)

    def test_nan_cells_ignored_in_summary(self):
        fn = lambda x: np.array([x, np.nan])
        res = oat_sensitivity(fn, {"x": 1.0}, deltas=(0.1,), summary="mean")
        (s,) = res["x"]["summary"]
        self.assertAlmostEqual(s, 1.1)


class TestOATWithComputeRweq(unittest.TestCase):
    def test_threshold_speed_reduces_erosion(self):
        from pyrweq.core import compute_rweq
        from tests.test_period import _wf_inputs
        from tests.test_yearly import _month_inputs

        m = _month_inputs(31, shape=(6, 6))
        m.update(_wf_inputs(shape=(6, 6)))

        def fn(**kw):
            return compute_rweq(**m, n_workers=1, **kw).sl

        res = oat_sensitivity(
            fn, {"threshold_speed": 5.0}, values={"threshold_speed": [4.0, 5.0, 6.0]}
        )
        s = res["threshold_speed"]["summary"]
        self.assertEqual(len(s), 3)
        self.assertGreater(s[0], s[1])  # lower threshold -> more erosion
        self.assertGreater(s[1], s[2])


if __name__ == "__main__":
    unittest.main()
