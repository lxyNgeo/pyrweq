"""Tests for period-length (nd/n_obs) handling.

RWEQ natively uses a half-month period (nd=15). Monthly runs must scale WF
with the number of days per period; the yearly wrapper infers this.
"""

import unittest

import numpy as np

from pyrweq.core import compute_rweq, compute_rweq_yearly
from pyrweq.factors.weather import calc_weather_factor
from tests.test_yearly import _month_inputs


def _wf_inputs(shape=(10, 10)):
    """Inputs where SW stays unsaturated so WF scales linearly with nd."""
    rng = np.random.default_rng(7)
    return dict(
        wind_speed=rng.random(shape).astype(np.float32) * 10 + 6,
        precip=rng.random(shape).astype(np.float32) * 5 + 980,
        temp=rng.random(shape).astype(np.float32) * 20,
        elevation=rng.random(shape).astype(np.float32),
        potential_et=rng.random(shape).astype(np.float32) * 5 + 1000,
        snow_depth=np.zeros(shape, dtype=np.float32),
    )


class TestWeatherPeriodParams(unittest.TestCase):
    def test_nd_doubles_wf_when_sw_unsaturated(self):
        inp = _wf_inputs()
        wf15 = calc_weather_factor(**inp, nd=15.0)
        wf30 = calc_weather_factor(**inp, nd=30.0)
        self.assertTrue(np.allclose(wf30, 2.0 * wf15, equal_nan=True))

    def test_n_obs_defaults_to_nd(self):
        inp = _wf_inputs()
        wf_default = calc_weather_factor(**inp, nd=20.0)
        wf_explicit = calc_weather_factor(**inp, nd=20.0, n_obs=20.0)
        self.assertTrue(np.allclose(wf_default, wf_explicit))

    def test_n_obs_halves_wf(self):
        inp = _wf_inputs()
        wf_a = calc_weather_factor(**inp, nd=15.0, n_obs=15.0)
        wf_b = calc_weather_factor(**inp, nd=15.0, n_obs=30.0)
        self.assertTrue(np.allclose(wf_b, 0.5 * wf_a))

    def test_legacy_default_unchanged(self):
        # no nd/n_obs args -> nd=15, n_obs=15 (pre-change behaviour)
        inp = _wf_inputs()
        wf = calc_weather_factor(**inp)
        wf_ref = calc_weather_factor(**inp, nd=15.0, n_obs=15.0)
        self.assertTrue(np.allclose(wf, wf_ref))


class TestComputeRweqPeriodParams(unittest.TestCase):
    def test_nd_passed_through_compute_rweq(self):
        m = _month_inputs(42)
        m.update(_wf_inputs())  # keep texture, use SW-unsaturated weather
        wf15 = compute_rweq(**m, n_workers=1, nd=15.0).wf
        wf30 = compute_rweq(**m, n_workers=1, nd=30.0).wf
        self.assertTrue(np.allclose(wf30, 2.0 * wf15))


class TestYearlyPeriodDays(unittest.TestCase):
    def test_auto_period_days_for_12_months(self):
        inputs = [_month_inputs(500 + i) for i in range(12)]
        yearly = compute_rweq_yearly(inputs, n_workers=1)
        expected_nd = 365.25 / 12
        ref = compute_rweq(**inputs[0], n_workers=1, nd=expected_nd)
        self.assertTrue(np.allclose(yearly.months[0].wf, ref.wf))

    def test_period_days_override(self):
        inputs = [_month_inputs(600 + i) for i in range(12)]
        yearly = compute_rweq_yearly(inputs, n_workers=1, period_days=30.0)
        ref = compute_rweq(**inputs[0], n_workers=1, nd=30.0)
        self.assertTrue(np.allclose(yearly.months[0].wf, ref.wf))

    def test_explicit_nd_in_factor_kwargs_wins(self):
        inputs = [_month_inputs(700 + i) for i in range(12)]
        yearly = compute_rweq_yearly(inputs, n_workers=1, period_days=30.0, nd=10.0)
        ref = compute_rweq(**inputs[0], n_workers=1, nd=10.0)
        self.assertTrue(np.allclose(yearly.months[0].wf, ref.wf))

    def test_nd_in_period_dict_respected(self):
        inputs = [_month_inputs(800 + i) for i in range(3)]
        inputs[0]["nd"] = 10.0
        yearly = compute_rweq_yearly(inputs, n_workers=1)
        ref = compute_rweq(**inputs[0], n_workers=1)  # nd comes from the dict
        self.assertTrue(np.allclose(yearly.months[0].wf, ref.wf))

    def test_kwarg_collision_rejected(self):
        inputs = [_month_inputs(900 + i) for i in range(2)]
        inputs[0]["nd"] = 10.0
        with self.assertRaises(ValueError):
            compute_rweq_yearly(inputs, n_workers=1, nd=10.0)


if __name__ == "__main__":
    unittest.main()
