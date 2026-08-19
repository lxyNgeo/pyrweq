"""Tests for the SL unit chain (g/m <-> t/(km^2*a)) and classification."""

import unittest

import numpy as np

from pyrweq.classify import classify_erosion, CHINA_STANDARD
from pyrweq.units import (
    g_per_m_to_t_per_km2,
    t_per_km2_to_g_per_m,
    cell_size_from_profile,
)


class TestUnitConversions(unittest.TestCase):
    def test_roundtrip(self):
        sl = np.array([100.0, 250.0, 1000.0])
        size = 25.0  # 25 m cells
        modulus = g_per_m_to_t_per_km2(sl, size)
        back = t_per_km2_to_g_per_m(modulus, size)
        self.assertTrue(np.allclose(back, sl))

    def test_known_value(self):
        # 500 g/m over 50 m cells -> 10 g/m^2 == 10 t/km^2/a
        self.assertAlmostEqual(g_per_m_to_t_per_km2(np.array([500.0]), 50.0).item(), 10.0)

    def test_cell_size_tuple(self):
        sl = np.array([100.0])
        self.assertAlmostEqual(g_per_m_to_t_per_km2(sl, (20.0, 20.0)).item(), 5.0)

    def test_non_square_cells_warn(self):
        with self.assertLogs("pyrweq.units", level="WARNING"):
            g_per_m_to_t_per_km2(np.array([100.0]), (20.0, 40.0))

    def test_invalid_cell_size(self):
        for bad in (0.0, -5.0, (10.0, 0.0), (1.0, 2.0, 3.0)):
            with self.assertRaises(ValueError):
                g_per_m_to_t_per_km2(np.array([1.0]), bad)

    def test_nan_propagates(self):
        out = g_per_m_to_t_per_km2(np.array([np.nan, 100.0]), 10.0)
        self.assertTrue(np.isnan(out[0]))

    def test_cell_size_from_profile(self):
        from rasterio.transform import from_origin
        profile = {"transform": from_origin(0, 0, 30.0, 30.0)}
        self.assertAlmostEqual(cell_size_from_profile(profile), 30.0)

    def test_cell_size_from_profile_no_transform(self):
        with self.assertLogs("pyrweq.units", level="WARNING"):
            size = cell_size_from_profile({})
        self.assertEqual(size, 1.0)


class TestClassifyUnits(unittest.TestCase):
    def test_modulus_input_unchanged(self):
        arr = np.array([100.0, 300.0, 3000.0, 20000.0])
        classified, _ = classify_erosion(arr)
        self.assertEqual(classified.tolist(), [1, 2, 3, 6])

    def test_native_sl_with_cell_size(self):
        # 5000 g/m over 10 m cells -> 500 t/km^2/a -> class 2 (200-2500)
        arr = np.array([5000.0])
        classified, _ = classify_erosion(arr, cell_size=10.0)
        self.assertEqual(classified.item(), 2)

    def test_same_class_from_both_conventions(self):
        # modulus M reached via native SL with cell size must classify
        # identically to feeding M directly
        modulus = np.array([150.0, 800.0, 4000.0, 20000.0])
        native = t_per_km2_to_g_per_m(modulus, 100.0)
        c_direct, _ = classify_erosion(modulus)
        c_native, _ = classify_erosion(native, cell_size=100.0)
        self.assertEqual(c_direct.tolist(), c_native.tolist())

    def test_thresholds_are_modulus_based(self):
        # sanity: class bounds live in t/(km^2*a)
        low, high, code = CHINA_STANDARD[0]
        self.assertEqual((low, high, code), (0, 200, 1))


class TestSandfixUnits(unittest.TestCase):
    def test_sandfix_passes_period_params(self):
        from pyrweq.sandfix import compute_sandfix
        from tests.test_period import _wf_inputs
        from tests.test_yearly import _month_inputs

        m = _month_inputs(77, shape=(6, 6))
        weather = _wf_inputs(shape=(6, 6))
        m.update(weather)
        g15 = compute_sandfix(**m, n_workers=1, nd=15.0)
        g30 = compute_sandfix(**m, n_workers=1, nd=30.0)
        # SL is nonlinear in the factor product, so exact scaling cannot be
        # asserted; but nd must be passed through (identical results would
        # mean it was dropped) and higher wind energy means more fixation
        self.assertFalse(np.allclose(g30, g15))
        valid = np.isfinite(g30) & np.isfinite(g15) & (g15 > 0)
        self.assertTrue((g30[valid] > g15[valid]).all())


if __name__ == "__main__":
    unittest.main()
