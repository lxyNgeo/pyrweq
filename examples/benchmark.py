"""Benchmark pyrweq: numpy vs dask backends, sequential vs parallel.

Usage:
    python examples/benchmark.py            # all sizes
    python examples/benchmark.py 500 1000   # selected sizes

Reports wall time and peak RSS per configuration. Input rasters are
generated into a temp directory and cleaned up afterwards.
"""

import os
import sys
import tempfile
import time
import warnings

import numpy as np
import rasterio
import psutil

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from pyrweq.core import compute_rweq

NAMES = ["wind_speed", "precip", "temp", "elevation", "potential_et", "snow_depth",
         "sand_content", "silt_content", "clay_content", "organic_matter", "ndvi"]

CONFIGS = [
    ("numpy  seq",  dict(backend="numpy", n_workers=1)),
    ("numpy  par4", dict(backend="numpy", n_workers=4)),
    ("dask   seq",  dict(backend="dask",  n_workers=1)),
    ("dask   par4", dict(backend="dask",  n_workers=4)),
]


def make_rasters(n: int, tmpdir: str) -> dict[str, str]:
    """Generate n x n GeoTIFFs (tiled, 256x256 blocks) and return path dict."""
    paths = {}
    rng = np.random.default_rng(42)
    for name in NAMES:
        arr = rng.random((n, n), dtype=np.float32) * 10
        if name == "wind_speed":
            arr += 5
        if name == "snow_depth":
            arr[:] = 0
        path = os.path.join(tmpdir, f"{name}_{n}.tif")
        with rasterio.open(
            path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
            tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            dst.write(arr, 1)
        paths[name] = path
    return paths


def peak_rss_mb() -> float:
    """Max RSS observed during the process so far (MB)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def run_case(paths: dict, kwargs: dict, n: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    result = compute_rweq(**paths, **kwargs)
    sl = result.sl
    if hasattr(sl, "compute"):
        sl = sl.compute()
    elapsed = time.perf_counter() - t0
    rss = peak_rss_mb()
    assert sl.shape == (n, n)
    return elapsed, rss


def main() -> None:
    sizes = [int(x) for x in sys.argv[1:]] or [500, 1000, 2000]
    print(f"{'size':>6} {'config':<12} {'time (s)':>9} {'peak RSS (MB)':>13}")
    print("-" * 46)
    with tempfile.TemporaryDirectory() as tmpdir:
        for n in sizes:
            paths = make_rasters(n, tmpdir)
            rss_baseline = peak_rss_mb()
            for label, kwargs in CONFIGS:
                t, rss = run_case(paths, kwargs, n)
                print(f"{n:>6} {label:<12} {t:>9.2f} {rss - rss_baseline:>13.0f}")
                sys.stdout.flush()
            print("-" * 46)


if __name__ == "__main__":
    main()
