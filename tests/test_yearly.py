"""Tests for multi-period (monthly/yearly) computation."""

import unittest

import numpy as np

from pyrweq.core import compute_rweq, compute_rweq_yearly


def _month_inputs(seed, scale=1.0, shape=(10, 10)):
    rng = np.random.default_rng(seed)
    return dict(
        wind_speed=(rng.random(shape).astype(np.float32) * 10 + 5) * scale,
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


class TestYearly(unittest.TestCase):
    # yearly injects nd = 365.25 / n_periods into each period; reference
    # computations must use the same value.
    ND = 365.25 / 3

    def test_yearly_equals_sum_of_months(self):
        inputs = [_month_inputs(100 + i) for i in range(3)]
        yearly = compute_rweq_yearly(inputs, n_workers=1)
        expected = np.sum(
            [compute_rweq(**m, n_workers=1, nd=self.ND).sl for m in inputs], axis=0
        )
        self.assertTrue(np.allclose(yearly.sl, expected))
        self.assertEqual(len(yearly.months), 3)

    def test_factor_means_exposed(self):
        inputs = [_month_inputs(200 + i) for i in range(3)]
        yearly = compute_rweq_yearly(inputs, n_workers=1)
        wf_mean = np.mean(
            [compute_rweq(**m, n_workers=1, nd=self.ND).wf for m in inputs], axis=0
        )
        self.assertTrue(np.allclose(yearly.wf, wf_mean))
        ef_mean = np.mean([m.ef for m in yearly.months], axis=0)
        self.assertTrue(np.allclose(yearly.ef, ef_mean))

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            compute_rweq_yearly([])

    def test_nodata_cells_excluded_from_total(self):
        inputs = [_month_inputs(300 + i) for i in range(2)]
        inputs[0]["wind_speed"] = np.where(
            np.indices((10, 10)).sum(axis=0) == 0, np.nan, inputs[0]["wind_speed"]
        )
        yearly = compute_rweq_yearly(inputs, n_workers=1)
        self.assertTrue(np.isnan(yearly.sl[0, 0]))


if __name__ == "__main__":
    unittest.main()
