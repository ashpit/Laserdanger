# CLAUDE.md

## Overview

Laserdanger is a lidar processing pipeline for analyzing Livox Avia point cloud data (.laz files) to generate beach surface DEMs and detect wave runup. Developed for Ashton Domi's PhD thesis at Scripps Institution of Oceanography.

**Active development is in `python/` only.** The MATLAB code in the repository root is vestigial and no longer maintained.

See `python/CLAUDE.md` for detailed Python pipeline documentation.

## Quick Start

```bash
cd python
pip install -r requirements.txt
pytest tests/ -v

# L1 processing (daily beach surfaces)
python scripts/run_daily_l1.py --config ../livox_config.json

# L2 processing (wave-resolving ~2Hz)
python code/phase4.py l2 ../livox_config.json -o output.nc --start 2024-06-01 --end 2024-06-02
```

## Directory Structure

```
python/
├── code/           # Main source (phase1-4, profiles, runup, utils, validation)
├── tests/          # pytest suite (186+ tests)
├── scripts/        # CLI scripts (run_daily_l1.py, run_daily_l2.py)
├── data/           # Output data (level1/, level2/)
└── docs/           # Documentation
```

## Configuration

`livox_config.json` (in repository root):
- `dataFolder` - Path to raw .laz files
- `processFolder` - Output directory
- `transformMatrix` - 4×4 homogeneous transform (lidar → UTM)
- `LidarBoundary` - Polygon vertices (UTM) for spatial filtering

## Dependencies

numpy, scipy, pandas, laspy, lazrs, xarray, netCDF4, matplotlib, tqdm, pytest
