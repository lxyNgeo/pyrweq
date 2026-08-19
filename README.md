# pyrweq

Python implementation of the Revised Wind Erosion Equation (RWEQ).

pyrweq is a scientific Python library for estimating wind erosion loss from raster-based geospatial data. It computes all RWEQ factors (weather, erodibility, crust, roughness, vegetation) and produces erosion amount, sand fixation, and intensity classification results following China's SL190-2007 standard.

## Features

- **Weather Factor (WF)** — wind energy available for erosion; accepts a
  single wind-speed field, a 3D observation series, or a 3D speed/frequency
  distribution (Fryear et al. 1998 original form)
- **Soil Erodibility Factor (EF)** — susceptibility based on soil texture
- **Soil Crust Factor (SCF)** — surface crusting effect
- **Roughness Factor (K')** — terrain roughness (Smith-Carson equation)
- **Vegetation Factor (C)** — vegetation protection from NDVI
- **Wind Erosion Amount (SL)** — final erosion estimate (native g/m)
- **Unit Chain** — explicit SL (g/m) to modulus (t/(km^2*a)) conversion
- **Grid Alignment** — warp mixed-resolution inputs (0.25 deg meteo, 1 km
  soil, 250 m NDVI) onto one reference grid
- **Sand Fixation** — vegetation protective effect quantification
- **Erosion Intensity Classification** — 6-level standard (SL190-2007)
- **Zonal Statistics** — per-zone summary of erosion results
- **Validation** — r2 / RMSE / MAE / bias / Nash-Sutcliffe against
  observations, with point sampling for station data
- **Sensitivity** — one-at-a-time parameter perturbation with arc
  elasticities (the standard reviewer question)
- **Parallel Factor Computation** — factors computed concurrently via threads (`n_workers`)
- **Dask Backend** — lazy array computation for large rasters (`backend="dask"`)
- **Structured Logging** — module-level loggers with info/warning messages

## Installation

pyrweq is not published on PyPI — install directly from GitHub:

```bash
pip install "git+https://github.com/lxyNgeo/pyrweq.git"
```

For development (editable install):

```bash
git clone https://github.com/lxyNgeo/pyrweq.git
cd pyrweq
pip install -e .
```

Or install from a downloaded archive (no git required):

```bash
pip install https://github.com/lxyNgeo/pyrweq/archive/refs/heads/main.zip
```

### Optional dependencies

```bash
pip install "pyrweq[geo] @ git+https://github.com/lxyNgeo/pyrweq.git"    # geopandas support
pip install "pyrweq[plot] @ git+https://github.com/lxyNgeo/pyrweq.git"   # matplotlib support
pip install "pyrweq[dask] @ git+https://github.com/lxyNgeo/pyrweq.git"   # dask backend
pip install "pyrweq[dev] @ git+https://github.com/lxyNgeo/pyrweq.git"    # dask + benchmark/notebook tools
```

(Requires pip >= 21.2 for the `[extra] @ git+...` syntax; with older pip use
`pip install "git+...#egg=pyrweq[geo]"`.)

Tests run with the standard library, no pytest needed:

```bash
cd pyrweq
PYTHONPATH=src python -m unittest discover -s tests -v
```

> **Note** — `pip install pyrweq` will NOT work: the name is not registered
> on PyPI and may point to an unrelated package if someone registers it
> later. Always install from the git URL above.

## Quick Start

```python
from pyrweq import compute_rweq, classify_erosion

result = compute_rweq(
    wind_speed="wind.tif",
    precip="precip.tif",
    temp="temp.tif",
    elevation="dem.tif",
    potential_et="pet.tif",
    snow_depth="snow.tif",
    sand_content="sand.tif",
    silt_content="silt.tif",
    clay_content="clay.tif",
    organic_matter="om.tif",
    ndvi="ndvi.tif",
    output_dir="output/",
)

print(f"Mean erosion: {result.sl.mean():.2f}")

# Classify erosion intensity
classes = classify_erosion(result.sl)
```

## Period length and units

RWEQ natively uses a half-month period. `compute_rweq` defaults to `nd=15`
(days per period) and `n_obs=nd` (wind observations per period); monthly
runs should pass `nd=30`. `compute_rweq_yearly` infers `nd=365.25/n_periods`
automatically (override with `period_days` or an explicit `nd`).

```python
# monthly periods, one year of data
yearly = compute_rweq_yearly(monthly_inputs, period_days=30.0)
```

SL comes out in RWEQ-native g/m. Convert to the reporting modulus before
classifying (the SL190-2007 thresholds are t/(km^2*a)):

```python
from pyrweq import g_per_m_to_t_per_km2, classify_erosion

modulus = g_per_m_to_t_per_km2(result.sl, cell_size=500.0)  # 500 m cells
classes = classify_erosion(modulus)
# or directly: classify_erosion(result.sl, cell_size=500.0)
```

## Wind speed distributions

`v*(v-Ut)^2` is convex, so feeding a period-mean wind speed systematically
underestimates WF. When daily (or sub-daily) wind speeds are available, pass
them as a 3D stack and WF follows the RWEQ original summation over
observations:

```python
# wind_series: (n_days, rows, cols) daily wind speeds
result = compute_rweq(wind_speed=wind_series, ...)

# or a speed-class frequency distribution (freqs sum to 1 along axis 0)
result = compute_rweq(wind_speed=speed_bins, wind_freq=freqs, ...)
```

## Mixed-resolution inputs

Meteorology, soil and NDVI rasters rarely share a grid. Warp everything onto
one reference grid first:

```python
from pyrweq import align_inputs

data, profile = align_inputs(paths, reference="ndvi.tif")  # auto: nearest
# for land_use, bilinear for continuous inputs                # for the rest
result = compute_rweq(**data)
```

## Validation and sensitivity

```python
from pyrweq import validate, sample_points, oat_sensitivity

pred = sample_points("output/sl_modulus.tif", xs, ys)  # at stations
metrics = validate(observed, pred)          # r2, rmse, mae, bias, NSE

sens = oat_sensitivity(
    lambda **kw: compute_rweq(**data, n_workers=1, **kw).sl,
    params={"threshold_speed": 5.0, "downwind_distance": 50.0},
)   # arc elasticity per parameter
```

## Performance options

By default the five RWEQ factors are computed concurrently using a thread pool
(`n_workers`, default `min(5, cpu_count)`; pass `n_workers=1` for sequential
execution):

```python
result = compute_rweq(..., n_workers=1)   # sequential
result = compute_rweq(..., n_workers=8)   # 8 threads
```

For rasters too large to fit in memory, pass `dask.array` inputs — or simply
GeoTIFF **paths** with `backend="dask"` — to keep the computation lazy until
you call `.compute()`:

```python
# Path inputs are read lazily in blocks; memory stays proportional to one chunk
result = compute_rweq(
    wind_speed="wind.tif", precip="precip.tif", ... ndvi="ndvi.tif",
    backend="dask",      # lazy windowed reads + lazy math
    chunks="auto",       # chunk size for reads (native block size), or (512, 512)
)
sl = result.sl.compute()   # triggers the full graph
```

Note: NDVI percentiles (5th/95th) are global reductions; with dask inputs they
are computed eagerly once at the start. All factor arrays remain lazy.

Rule of thumb (see `examples/benchmark.py`): numpy is 5-10x faster for
rasters that fit in RAM; dask uses a fraction of the memory and is the right
choice for rasters that do not fit.

```bash
python examples/benchmark.py 500 1000 2000
```

## Command-line interface

After installing, the `pyrweq` command is available:

```bash
# Single-period computation (writes wf/ef/scf/k_prime/c/s/qmax/sl rasters)
pyrweq compute --wind-speed wind.tif --precip prec.tif --temp temp.tif \
    --elevation dem.tif --potential-et pet.tif --snow-depth snow.tif \
    --sand-content sand.tif --silt-content silt.tif --clay-content clay.tif \
    --organic-matter om.tif --ndvi ndvi.tif --output-dir output/

# Sand fixation
pyrweq sandfix ... --output-dir output/

# Erosion intensity classification (SL190-2007)
pyrweq classify --input output/sl.tif --output output/class.tif

# Zonal statistics
pyrweq stats --sl output/sl.tif --zones landuse.tif --output stats.csv
```

Common options: `--backend numpy|dask`, `--n-workers N`, `--chunks ROWS,COLS`,
`--veg-method simplified|typed|full_cog`, `--threshold 5.0`, `--distance 50`,
`--wind-height 10m|2m`, `--no-masked`, `-v` for INFO logs.

## Logging

pyrweq uses the standard `logging` module. To see progress and warnings:

```python
import logging
logging.basicConfig(level=logging.INFO)  # or attach a handler to the "pyrweq" logger
```

## Project Structure

```
pyrweq/
├── src/pyrweq/
│   ├── core.py          # Main RWEQ computation
│   ├── erosion.py       # SL formula
│   ├── io.py            # Raster I/O
│   ├── classify.py      # Intensity classification
│   ├── sandfix.py       # Sand fixation
│   ├── stats.py         # Zonal statistics
│   ├── _types.py        # Shared type aliases & dask detection
│   └── factors/         # Factor calculations
│       ├── weather.py
│       ├── erodibility.py
│       ├── crust.py
│       ├── roughness.py
│       └── vegetation.py
└── tests/
```

## License

MIT
