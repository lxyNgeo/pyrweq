"""Tests for wind speed distribution input to the weather factor.

RWEQ's original WF sums over wind speed observations/classes (Fryear et
al. 1998). calc_weather_factor accepts a 3D (k, rows, cols) wind speed
stack, optionally with per-pixel occurrence frequencies.
"""

import unittest

import numpy as np

from pyrweq._types import is_dask_array
from pyrweq.factors.weather import calc_weather_factor

try:
    import dask.array as da
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False


def _base_inputs(shape=(4, 5)):
    """Weather inputs with unsaturated SW so WF is comparable across runs."""
    rng = np.random.default_rng(3)
    return dict(
        precip=rng.random(shape).astype(np.float32) * 5 + 980,
        temp=rng.random(shape).astype(np.float32) * 20,
        elevation=rng.random(shape).astype(np.float32),
        potential_et=rng.random(shape).astype(np.float32) * 5 + 1000,
        snow_depth=np.zeros(shape, dtype=np.float32),
    )


class TestWindDistribution(unittest.TestCase):
    def test_constant_series_equals_2d(self):
        """2D form with n_obs=1 == k identical observations.

        In the 2D form n_obs is the inverse occurrence frequency of u:
        n_obs=1 means u already aggregates the whole period, which is what
        a constant 3D series of k observations reduces to.
        """
        inp = _base_inputs()
        u = np.full((4, 5), 8.0, dtype=np.float32)
        wf_2d = calc_weather_factor(u, n_obs=1, input_10m=False, **inp)
        wf_3d = calc_weather_factor(np.stack([u] * 10), input_10m=False, **inp)
        self.assertTrue(np.allclose(wf_2d, wf_3d))

    def test_variable_series_exceeds_mean_form(self):
        """Convexity: the true distribution beats the period-mean speed."""
        inp = _base_inputs()
        rng = np.random.default_rng(0)
        series = rng.uniform(3.0, 13.0, size=(60, 4, 5)).astype(np.float32)
        wf_dist = calc_weather_factor(series, input_10m=False, **inp)
        wf_mean = calc_weather_factor(series.mean(axis=0), input_10m=False, **inp)
        valid = np.isfinite(wf_dist) & np.isfinite(wf_mean)
        self.assertTrue((wf_dist[valid] > wf_mean[valid]).all())

    def test_frequency_weighted_matches_hand_computation(self):
        inp = _base_inputs()
        speeds = np.array([[[4.0]], [[6.0]], [[12.0]]], dtype=np.float32)  # (3,1,1)
        freqs = np.array([[[0.5]], [[0.3]], [[0.2]]], dtype=np.float32)
        wf = calc_weather_factor(speeds, wind_freq=freqs, input_10m=False, **inp)
        # below-threshold speed contributes zero
        u, f = 6.0, 0.3
        term = f * u * (u - 5.0) ** 2
        term += 0.2 * 12.0 * (12.0 - 5.0) ** 2
        rho = 348.0 * (1.013 - 0.1183 * inp["elevation"] + 0.0048 * inp["elevation"] ** 2) / (
            inp["temp"] + 273.15
        )
        sw = np.clip(
            (inp["potential_et"] - inp["precip"]) * 15.0 / inp["potential_et"], 0.0, 1.0
        )
        expected = term * 15.0 * rho / 9.8 * sw
        self.assertTrue(np.allclose(wf, expected))

    def test_below_threshold_observations_zeroed(self):
        inp = _base_inputs()
        series = np.full((2, 4, 5), 3.0, dtype=np.float32)  # all below 5 m/s
        wf = calc_weather_factor(series, input_10m=False, **inp)
        self.assertTrue(np.allclose(wf, 0.0))

    def test_freq_shape_mismatch_raises(self):
        inp = _base_inputs()
        speeds = np.ones((3, 4, 5), dtype=np.float32)
        bad_freq = np.ones((2, 4, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            calc_weather_factor(speeds, wind_freq=bad_freq, **inp)

    def test_freq_with_2d_wind_raises(self):
        inp = _base_inputs()
        u = np.full((4, 5), 8.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            calc_weather_factor(u, wind_freq=np.ones_like(u), **inp)

    def test_unnormalized_freq_warns(self):
        inp = _base_inputs()
        speeds = np.full((2, 4, 5), 8.0, dtype=np.float32)
        freq = np.full((2, 4, 5), 0.4, dtype=np.float32)  # sums to 0.8
        with self.assertLogs("pyrweq.factors.weather", level="WARNING"):
            calc_weather_factor(speeds, wind_freq=freq, input_10m=False, **inp)

    def test_n_obs_ignored_for_3d(self):
        inp = _base_inputs()
        series = np.full((10, 4, 5), 8.0, dtype=np.float32)
        wf_a = calc_weather_factor(series, input_10m=False, **inp)
        wf_b = calc_weather_factor(series, n_obs=99.0, input_10m=False, **inp)
        self.assertTrue(np.allclose(wf_a, wf_b))

    def test_equal_weight_equals_mean_of_terms(self):
        inp = _base_inputs()
        rng = np.random.default_rng(1)
        series = rng.uniform(5.5, 12.0, size=(8, 4, 5)).astype(np.float32)
        wf = calc_weather_factor(series, input_10m=False, **inp)
        terms = series * (series - 5.0) ** 2
        rho = 348.0 * (1.013 - 0.1183 * inp["elevation"] + 0.0048 * inp["elevation"] ** 2) / (
            inp["temp"] + 273.15
        )
        sw = np.clip(
            (inp["potential_et"] - inp["precip"]) * 15.0 / inp["potential_et"], 0.0, 1.0
        )
        expected = terms.mean(axis=0) * 15.0 * rho / 9.8 * sw
        self.assertTrue(np.allclose(wf, expected))


class TestComputeRweqWind3D(unittest.TestCase):
    def test_3d_wind_end_to_end(self):
        from pyrweq.core import compute_rweq
        from tests.test_yearly import _month_inputs

        m = _month_inputs(123, shape=(4, 5))
        weather = _base_inputs()
        m.update(weather)
        rng = np.random.default_rng(11)
        series = rng.uniform(4.0, 13.0, size=(20, 4, 5)).astype(np.float32)
        m["wind_speed"] = series

        res = compute_rweq(**m, n_workers=1)
        self.assertEqual(res.sl.shape, (4, 5))  # spatial output, not (k, r, c)
        self.assertEqual(res.profile["height"], 4)
        self.assertEqual(res.profile["width"], 5)

        wf_ref = calc_weather_factor(series, input_10m=True, **weather)
        self.assertTrue(np.allclose(res.wf, wf_ref))

    def test_wind_freq_end_to_end(self):
        from pyrweq.core import compute_rweq
        from tests.test_yearly import _month_inputs

        m = _month_inputs(124, shape=(4, 5))
        weather = _base_inputs()
        m.update(weather)
        speeds = np.tile(
            np.array([[[4.0, 6.0, 7.0, 9.0, 11.0]]], dtype=np.float32), (3, 4, 1)
        )
        freqs = np.full(speeds.shape, 1.0 / 3.0, dtype=np.float32)
        m["wind_speed"] = speeds
        m["wind_freq"] = freqs

        res = compute_rweq(**m, n_workers=1)
        wf_ref = calc_weather_factor(
            speeds, wind_freq=freqs, input_10m=True, **weather
        )
        self.assertTrue(np.allclose(res.wf, wf_ref))

    def test_default_slope_stays_2d_with_3d_wind(self):
        # slope=None zeros must not inherit the k axis of a 3D wind stack
        from pyrweq.core import compute_rweq
        from tests.test_yearly import _month_inputs

        m = _month_inputs(125, shape=(4, 5))
        weather = _base_inputs()
        m.update(weather)
        m["wind_speed"] = np.full((7, 4, 5), 8.0, dtype=np.float32)

        res = compute_rweq(**m, n_workers=1)
        self.assertEqual(res.k_prime.shape, (4, 5))
        self.assertEqual(res.sl.shape, (4, 5))


@unittest.skipUnless(_HAS_DASK, "dask not installed")
class TestWindDistributionDask(unittest.TestCase):
    def test_3d_dask_matches_numpy(self):
        inp = {k: da.from_array(v, chunks=2) for k, v in _base_inputs().items()}
        rng = np.random.default_rng(2)
        series_np = rng.uniform(4.0, 12.0, size=(6, 4, 5)).astype(np.float32)
        wf_np = calc_weather_factor(series_np, input_10m=False, **_base_inputs())
        wf_da = calc_weather_factor(
            da.from_array(series_np, chunks=(2, 2, 2)), input_10m=False, **inp
        )
        self.assertTrue(is_dask_array(wf_da))
        self.assertTrue(np.allclose(wf_da.compute(), wf_np))

    def test_freq_dask_matches_numpy(self):
        base = _base_inputs()
        inp = {k: da.from_array(v, chunks=2) for k, v in base.items()}
        speeds = np.tile(
            np.array([[[4.0, 6.0, 7.0, 9.0, 11.0]]], dtype=np.float32), (3, 4, 1)
        )
        freqs = np.full(speeds.shape, 1.0 / 3.0, dtype=np.float32)
        wf_np = calc_weather_factor(speeds, wind_freq=freqs, input_10m=False, **base)
        wf_da = calc_weather_factor(
            da.from_array(speeds, chunks=(1, 2, 5)),
            wind_freq=da.from_array(freqs, chunks=(1, 2, 5)),
            input_10m=False, **inp,
        )
        self.assertTrue(np.allclose(wf_da.compute(), wf_np))


if __name__ == "__main__":
    unittest.main()
