"""Tests for logging."""

import logging
import io
import unittest
import numpy as np

from pyrweq.core import compute_rweq
from pyrweq.factors.weather import calc_weather_factor
from pyrweq.factors.erodibility import calc_erodibility
from pyrweq.factors.vegetation import vegetation_cover, calc_vegetation


class TestLoggingSetup(unittest.TestCase):
    def test_null_handler_prevents_no_handler_warning(self):
        """The pyrweq root logger has a NullHandler; no 'No handlers' warning."""
        logger = logging.getLogger("pyrweq")
        has_null = any(isinstance(h, logging.NullHandler) for h in logger.handlers)
        self.assertTrue(has_null, "pyrweq logger should have a NullHandler")


class TestComputeRWEQLogging(unittest.TestCase):
    def test_info_emitted(self):
        logger = logging.getLogger("pyrweq")
        old_level = logger.level
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger.addHandler(handler)

        try:
            shape = (10, 10)
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
                n_workers=1,
            )
            output = buf.getvalue()
            self.assertIn("compute_rweq start", output)
            self.assertIn("compute_rweq done", output)
            self.assertIn("factor done: weather", output)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)


class TestWarningLogging(unittest.TestCase):
    def test_wind_threshold_warning(self):
        """All wind cells far above threshold -> WARNING logged."""
        logger = logging.getLogger("pyrweq.factors.weather")
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger.addHandler(handler)

        try:
            shape = (10, 10)
            calc_weather_factor(
                wind_speed=np.full(shape, 50.0, dtype=np.float32),
                precip=np.full(shape, 0.0, dtype=np.float32),
                temp=np.full(shape, 20.0, dtype=np.float32),
                elevation=np.full(shape, 1.0, dtype=np.float32),
                potential_et=np.full(shape, 5.0, dtype=np.float32),
                snow_depth=np.zeros(shape, dtype=np.float32),
                threshold_speed=5.0,
            )
            self.assertIn("exceed threshold", buf.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    def test_ndvi_degradation_warning(self):
        """Constant NDVI array triggers degredation warning in vegetation_cover."""
        logger = logging.getLogger("pyrweq.factors.vegetation")
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger.addHandler(handler)

        try:
            # NDVI with all same values -> 5th and 95th percentile will be identical
            ndvi = np.full((5, 5), 0.3, dtype=np.float32)
            # vegetation_cover with no explicit ndvi_soil/ndvi_veg:
            # 5th and 95th percentile of a constant array are both 0.3 -> degenerate
            vegetation_cover(ndvi)
            self.assertIn("ndvi_veg≈ndvi_soil", buf.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    def test_clay_zero_info(self):
        """clay==0 cells trigger INFO message."""
        logger = logging.getLogger("pyrweq.factors.erodibility")
        old_level = logger.level
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger.addHandler(handler)

        try:
            shape = (10, 10)
            calc_erodibility(
                sand=np.full(shape, 70.0, dtype=np.float32),
                silt=np.full(shape, 30.0, dtype=np.float32),
                clay=np.zeros(shape, dtype=np.float32),  # all clay==0
                organic_matter=np.full(shape, 1.0, dtype=np.float32),
            )
            self.assertIn("clay==0 cells", buf.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)


if __name__ == "__main__":
    unittest.main()
