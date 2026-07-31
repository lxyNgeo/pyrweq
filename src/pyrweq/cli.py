"""Command-line interface for pyrweq.

Subcommands:
    compute   Run a single-period RWEQ computation (writes factor + SL rasters)
    sandfix   Compute sand fixation G = SL_pot - SL_actual (writes G.tif)
    classify  Classify an SL raster into erosion intensity classes
    stats     Zonal statistics of an SL raster

Example:
    pyrweq compute --wind wind.tif --precip prec.tif --temp temp.tif \\
        --elevation dem.tif --pet pet.tif --snow snow.tif \\
        --sand sand.tif --silt silt.tif --clay clay.tif --om om.tif \\
        --ndvi ndvi.tif --output-dir out/
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from pyrweq.core import compute_rweq
from pyrweq.classify import classify_erosion
from pyrweq.sandfix import compute_sandfix
from pyrweq.stats import zonal_stats

logger = logging.getLogger(__name__)

INPUT_HELP = {
    "wind_speed": "Wind speed raster (m/s)",
    "precip": "Precipitation raster (mm)",
    "temp": "Temperature raster (degC)",
    "elevation": "Elevation raster (km)",
    "potential_et": "Potential evapotranspiration raster (mm)",
    "snow_depth": "Snow depth raster (mm)",
    "sand_content": "Sand content raster (percent)",
    "silt_content": "Silt content raster (percent)",
    "clay_content": "Clay content raster (percent)",
    "organic_matter": "Organic matter raster (percent)",
    "ndvi": "NDVI raster",
}


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    for name, help_text in INPUT_HELP.items():
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, metavar="TIF",
                            help=help_text)
    parser.add_argument("--caco3", metavar="TIF", default=None,
                        help="Calcium carbonate raster (percent) [optional]")
    parser.add_argument("--land-use", metavar="TIF", default=None,
                        help="Land use codes raster (10/20/30/40/50/60) [optional]")
    parser.add_argument("--slope", metavar="TIF", default=None,
                        help="Slope raster (degrees) [optional, default flat]")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Threshold wind speed m/s (default 5.0)")
    parser.add_argument("--distance", type=float, default=50.0,
                        help="Downwind distance z m (default 50)")
    parser.add_argument("--veg-method", default="simplified",
                        choices=["simplified", "typed", "full_cog"],
                        help="Vegetation factor method (default simplified)")
    parser.add_argument("--wind-height", default="10m", choices=["10m", "2m"],
                        help="Input wind speed height (default 10m, converted to 2m)")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "dask"],
                        help="Computation backend (default numpy)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Threads for factor computation (default min(5, cpu))")
    parser.add_argument("--chunks", default="auto",
                        help="dask read chunk size as ROWS,COLS (default auto)")
    parser.add_argument("--no-masked", action="store_true",
                        help="Do not convert nodata to NaN in inputs")


def _collect_inputs(args: argparse.Namespace) -> dict:
    inputs = {name: getattr(args, name.replace("-", "_")) for name in INPUT_HELP}
    for opt in ("caco3", "land_use", "slope"):
        v = getattr(args, opt)
        if v is not None:
            inputs[opt] = v
    return inputs


def _parse_chunks(chunks: str) -> tuple[int, int] | str:
    if chunks == "auto":
        return "auto"
    try:
        r, c = chunks.split(",")
        return int(r), int(c)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--chunks must be 'auto' or 'ROWS,COLS', got {chunks!r}"
        ) from e


def _common_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        threshold_speed=args.threshold,
        downwind_distance=args.distance,
        veg_method=args.veg_method,
        input_10m=args.wind_height == "10m",
        backend=args.backend,
        n_workers=args.n_workers,
        chunks=_parse_chunks(args.chunks),
        masked=not args.no_masked,
    )


def cmd_compute(args: argparse.Namespace) -> None:
    inputs = _collect_inputs(args)
    result = compute_rweq(**inputs, output_dir=args.output_dir, **_common_kwargs(args))
    sl = result.sl.compute() if hasattr(result.sl, "compute") else result.sl
    print(f"SL mean={np.nanmean(sl):.4f}  max={np.nanmax(sl):.4f}  valid cells={np.isfinite(sl).sum()}")
    if args.output_dir:
        print(f"rasters written to {args.output_dir}/")


def cmd_sandfix(args: argparse.Namespace) -> None:
    inputs = _collect_inputs(args)
    g = compute_sandfix(**inputs, **_common_kwargs(args))
    if hasattr(g, "compute"):
        g = g.compute()
    print(f"G mean={np.nanmean(g):.4f}  max={np.nanmax(g):.4f}")


def cmd_classify(args: argparse.Namespace) -> None:
    classified, labels = classify_erosion(
        args.input, output_path=args.output, scheme=args.scheme
    )
    classes, counts = np.unique(classified, return_counts=True)
    for code, count in zip(classes, counts):
        label = labels.get(int(code), "nodata")
        print(f"  class {code} ({label}): {count} cells")


def cmd_stats(args: argparse.Namespace) -> None:
    rows = zonal_stats(args.sl, args.zones, output_csv=args.output)
    print(f"{'zone':>8} {'count':>8} {'mean':>12} {'sum':>12} {'min':>12} {'max':>12} {'std':>12}")
    for r in rows:
        print(f"{r['zone']:>8} {r['count']:>8} {r['mean']:>12.4f} {r['sum']:>12.2f} "
              f"{r['min']:>12.4f} {r['max']:>12.4f} {r['std']:>12.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyrweq", description="Revised Wind Erosion Equation (RWEQ) toolkit"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable INFO logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser("compute", help="run single-period RWEQ")
    _add_input_args(p_compute)
    _add_common_args(p_compute)
    p_compute.add_argument("--output-dir", default=None, help="directory for output rasters")
    p_compute.set_defaults(func=cmd_compute)

    p_sandfix = sub.add_parser("sandfix", help="compute sand fixation G")
    _add_input_args(p_sandfix)
    _add_common_args(p_sandfix)
    p_sandfix.set_defaults(func=cmd_sandfix)

    p_classify = sub.add_parser("classify", help="classify erosion intensity (SL190-2007)")
    p_classify.add_argument("--input", required=True, metavar="TIF", help="SL raster")
    p_classify.add_argument("--output", metavar="TIF", help="classified raster output")
    p_classify.add_argument("--scheme", default="china_standard")
    p_classify.set_defaults(func=cmd_classify)

    p_stats = sub.add_parser("stats", help="zonal statistics of SL by zones")
    p_stats.add_argument("--sl", required=True, metavar="TIF", help="SL raster")
    p_stats.add_argument("--zones", required=True, metavar="TIF", help="zone raster")
    p_stats.add_argument("--output", metavar="CSV", help="CSV output path")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
