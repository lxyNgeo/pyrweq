"""Tests for the pyrweq CLI."""

import unittest
import tempfile
import os

import numpy as np
import rasterio

from pyrweq.cli import build_parser, main


def _make_inputs(tmpdir, n=20, with_nodata=False):
    names = ["wind_speed", "precip", "temp", "elevation", "potential_et", "snow_depth",
             "sand_content", "silt_content", "clay_content", "organic_matter", "ndvi"]
    rng = np.random.default_rng(3)
    paths = {}
    for name in names:
        a = rng.random((n, n)).astype(np.float32) * 10
        if name == "wind_speed":
            a += 5
        if name == "snow_depth":
            a[:] = 0
        if with_nodata:
            a[0, 0] = -9999.0
        p = os.path.join(tmpdir, f"{name}.tif")
        with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1,
                           dtype="float32", nodata=-9999.0) as dst:
            dst.write(a, 1)
        paths[name] = p
    return paths


class TestParser(unittest.TestCase):
    def test_help_works(self):
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_compute_requires_all_inputs(self):
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["compute", "--wind", "w.tif"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_chunks_parsing(self):
        from pyrweq.cli import _parse_chunks
        self.assertEqual(_parse_chunks("auto"), "auto")
        self.assertEqual(_parse_chunks("512,256"), (512, 256))
        with self.assertRaises(Exception):
            _parse_chunks("512")

    def test_subcommand_dispatch(self):
        parser = build_parser()
        args = parser.parse_args(["stats", "--sl", "s.tif", "--zones", "z.tif"])
        self.assertEqual(args.command, "stats")
        self.assertTrue(callable(args.func))


class TestComputeCommand(unittest.TestCase):
    def test_compute_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _make_inputs(d)
            argv = ["compute", "--output-dir", os.path.join(d, "out")]
            for name, p in paths.items():
                argv += [f"--{name.replace('_', '-')}", p]
            self.assertEqual(main(argv), 0)
            outs = sorted(os.listdir(os.path.join(d, "out")))
            self.assertEqual(outs, ["c.tif", "ef.tif", "k_prime.tif", "qmax.tif",
                                    "s.tif", "scf.tif", "sl.tif", "wf.tif"])

    def test_compute_nodata_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _make_inputs(d, with_nodata=True)
            argv = ["compute", "--output-dir", os.path.join(d, "out")]
            for name, p in paths.items():
                argv += [f"--{name.replace('_', '-')}", p]
            main(argv)
            sl = rasterio.open(os.path.join(d, "out", "sl.tif")).read(1)
            self.assertEqual(sl[0, 0].item(), -9999.0)


class TestClassifyStatsCommands(unittest.TestCase):
    def test_classify_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _make_inputs(d)
            outdir = os.path.join(d, "out")
            # build argv properly
            argv = ["compute", "--output-dir", outdir]
            for name, p in paths.items():
                argv += [f"--{name.replace('_', '-')}", p]
            main(argv)
            sl = os.path.join(outdir, "sl.tif")

            cls_path = os.path.join(d, "cls.tif")
            self.assertEqual(main(["classify", "--input", sl, "--output", cls_path]), 0)
            cls = rasterio.open(cls_path).read(1)
            self.assertTrue((cls >= 0).all())

            csv_path = os.path.join(d, "stats.csv")
            zone = os.path.join(d, "zones.tif")
            zarr = np.zeros((20, 20), dtype=np.int16)
            zarr[:10, :] = 1
            zarr[10:, :] = 2
            with rasterio.open(zone, "w", driver="GTiff", height=20, width=20,
                               count=1, dtype="int16") as dst:
                dst.write(zarr, 1)
            self.assertEqual(main(["stats", "--sl", sl, "--zones", zone,
                                   "--output", csv_path]), 0)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            self.assertEqual(len(lines), 3)  # header + 2 zones


if __name__ == "__main__":
    unittest.main()
