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
