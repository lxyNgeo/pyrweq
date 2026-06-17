"""Unit tests for RWEQ factor calculations."""

import unittest
import numpy as np

from pyrweq.factors.helpers import wind_speed_2m, air_density, soil_moisture_factor, snow_cover_factor
from pyrweq.factors.weather import calc_weather_factor
from pyrweq.factors.erodibility import calc_erodibility
from pyrweq.factors.crust import calc_crust_factor
from pyrweq.factors.roughness import calc_roughness_simple
from pyrweq.factors.vegetation import calc_vegetation, vegetation_cover
from pyrweq.erosion import calc_sl


class TestHelpers(unittest.TestCase):
    def test_wind_speed_2m(self):
        u10 = np.array([10.0])
        u2 = wind_speed_2m(u10)
        expected = 10.0 * (2.0 / 10.0) ** (1.0 / 7.0)
        self.assertAlmostEqual(u2.item(), expected, places=2)

    def test_air_density_sea_level(self):
        rho = air_density(np.array([0.0]), np.array([288.15]))
        self.assertGreater(rho.item(), 1.1)
        self.assertLess(rho.item(), 1.4)

    def test_soil_moisture_factor(self):
        sw = soil_moisture_factor(pet=np.array([100.0]), precip=np.array([50.0]), nd=15.0)
        self.assertGreaterEqual(sw.item(), 0.0)
        self.assertLessEqual(sw.item(), 1.0)

    def test_snow_cover_factor(self):
        self.assertEqual(snow_cover_factor(np.array([50.0])).item(), 0.0)
        self.assertEqual(snow_cover_factor(np.array([5.0])).item(), 1.0)


class TestWeatherFactor(unittest.TestCase):
    def test_no_wind(self):
        wf = calc_weather_factor(
            wind_speed=np.array([3.0]), precip=np.array([0.0]),
            temp=np.array([20.0]), elevation=np.array([1.0]),
            potential_et=np.array([5.0]), snow_depth=np.array([0.0]),
        )
        self.assertEqual(wf.item(), 0.0)

    def test_high_wind(self):
        wf = calc_weather_factor(
            wind_speed=np.array([15.0]), precip=np.array([0.0]),
            temp=np.array([20.0]), elevation=np.array([1.0]),
            potential_et=np.array([5.0]), snow_depth=np.array([0.0]),
        )
        self.assertGreater(wf.item(), 0)


class TestErodibility(unittest.TestCase):
    def test_sandy_soil(self):
        ef = calc_erodibility(sand=np.array([80.0]), silt=np.array([10.0]),
                              clay=np.array([10.0]), organic_matter=np.array([0.5]))
        self.assertGreater(ef.item(), 0.5)
        self.assertLess(ef.item(), 1.0)

    def test_clay_soil(self):
        ef = calc_erodibility(sand=np.array([10.0]), silt=np.array([30.0]),
                              clay=np.array([60.0]), organic_matter=np.array([3.0]))
        self.assertLess(ef.item(), 0.5)


class TestCrustFactor(unittest.TestCase):
    def test_basic(self):
        scf = calc_crust_factor(clay=np.array([30.0]), organic_matter=np.array([2.0]))
        self.assertGreater(scf.item(), 0)
        self.assertLess(scf.item(), 1)

    def test_high_clay_low_crust(self):
        scf_low = calc_crust_factor(clay=np.array([10.0]), organic_matter=np.array([1.0]))
        scf_high = calc_crust_factor(clay=np.array([50.0]), organic_matter=np.array([3.0]))
        self.assertGreater(scf_low.item(), scf_high.item())


class TestRoughness(unittest.TestCase):
    def test_flat(self):
        k = calc_roughness_simple(np.array([0.0]))
        self.assertAlmostEqual(k.item(), 1.0, places=3)

    def test_steep(self):
        k = calc_roughness_simple(np.array([45.0]))
        self.assertAlmostEqual(k.item(), float(np.cos(np.radians(45))), places=3)


class TestVegetation(unittest.TestCase):
    def test_bare(self):
        sc = vegetation_cover(np.array([0.05]), ndvi_soil=0.05, ndvi_veg=0.8)
        self.assertEqual(sc.item(), 0.0)

    def test_full_cover(self):
        sc = vegetation_cover(np.array([0.8]), ndvi_soil=0.05, ndvi_veg=0.8)
        self.assertAlmostEqual(sc.item(), 1.0, places=3)

    def test_simplified(self):
        c = calc_vegetation(np.array([0.5]), method="simplified", ndvi_soil=0.05, ndvi_veg=0.8)
        self.assertGreater(c.item(), 0)
        self.assertLess(c.item(), 1)


class TestErosion(unittest.TestCase):
    def test_sl_formula(self):
        wf = np.array([100.0])
        ef = np.array([0.5])
        scf = np.array([0.8])
        k_prime = np.array([0.95])
        c = np.array([0.3])
        sl, s, qmax = calc_sl(wf, ef, scf, k_prime, c)
        self.assertGreater(sl.item(), 0)
        self.assertGreater(s.item(), 0)
        self.assertGreater(qmax.item(), 0)


if __name__ == "__main__":
    unittest.main()
