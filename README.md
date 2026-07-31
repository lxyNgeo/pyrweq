# pyrweq

Python implementation of the Revised Wind Erosion Equation (RWEQ).

pyrweq is a scientific Python library for estimating wind erosion loss from raster-based geospatial data. It computes all RWEQ factors (weather, erodibility, crust, roughness, vegetation) and produces erosion amount, sand fixation, and intensity classification results following China's SL190-2007 standard.

## Features

- **Weather Factor (WF)** — wind energy available for erosion
- **Soil Erodibility Factor (EF)** — susceptibility based on soil texture
- **Soil Crust Factor (SCF)** — surface crusting effect
- **Roughness Factor (K')** — terrain roughness (Smith-Carson equation)
- **Vegetation Factor (C)** — vegetation protection from NDVI
- **Wind Erosion Amount (SL)** — final erosion estimate
- **Sand Fixation** — vegetation protective effect quantification
- **Erosion Intensity Classification** — 6-level standard (SL190-2007)
- **Zonal Statistics** — per-zone summary of erosion results
- **Parallel Factor Computation** — factors computed concurrently via threads (`n_workers`)
- **Dask Backend** — lazy array computation for large rasters (`backend="dask"`)
- **Structured Logging** — module-level loggers with info/warning messages

## Installation

```bash
pip install pyrweq
```

Or install from source:

```bash
git clone https://github.com/lxyNgeo/pyrweq.git
cd pyrweq
pip install -e .
```

### Optional dependencies

```bash
pip install "pyrweq[geo]"    # geopandas support
pip install "pyrweq[plot]"   # matplotlib support
pip install "pyrweq[dev]"    # pytest for development
```

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

## Performance options

By default the five RWEQ factors are computed concurrently using a thread pool
(`n_workers`, default `min(5, cpu_count)`; pass `n_workers=1` for sequential
execution):

```python
result = compute_rweq(..., n_workers=1)   # sequential
result = compute_rweq(..., n_workers=8)   # 8 threads
```

For rasters too large to fit in memory, pass `dask.array` inputs with
`backend="dask"` to keep the computation lazy until you call `.compute()`:

```python
import dask.array as da

wind = da.from_zarr("wind.zarr", chunks=(512, 512))
ndvi = da.from_zarr("ndvi.zarr", chunks=(512, 512))
# ... all 11 inputs as dask arrays ...
result = compute_rweq(..., backend="dask")
sl = result.sl.compute()   # triggers the full graph
```

Note: NDVI percentiles (5th/95th) are global reductions; with dask inputs they
are computed eagerly once at the start. All factor arrays remain lazy.

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
