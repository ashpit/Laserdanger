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
python scripts/run_daily_l1.py --config configs/do_livox_config_20260112.json

# L2 processing (wave-resolving ~2Hz)
python code/phase4.py l2 configs/towr_livox_config_20260120.json -o output.nc --start 2026-01-20 --end 2026-01-21
```

## Directory Structure

```
python/
├── code/           # Main source (phase1-4, profiles, runup, utils, validation)
├── configs/        # Site/date-specific configs ({site}_livox_config_{YYYYMMDD}.json)
├── tests/          # pytest suite (186+ tests)
├── scripts/        # CLI scripts (run_daily_l1.py, run_daily_l2.py)
├── data/           # Output data (level1/, level2/)
└── docs/           # Documentation
```

## Configuration

Site/date-specific configs are stored in `python/configs/` with naming convention:
`{site}_livox_config_{YYYYMMDD}.json` (e.g., `do_livox_config_20260112.json`, `towr_livox_config_20260120.json`)

Config file fields:
- `dataFolder` - Path to raw .laz files
- `processFolder` - Output directory
- `transformMatrix` - 4×4 homogeneous transform (lidar → UTM)
- `LidarBoundary` - Polygon vertices (UTM) for spatial filtering

To add a new site, copy an existing config and update the paths and transform matrix.

## Dependencies

numpy, scipy, pandas, laspy, lazrs, xarray, netCDF4, matplotlib, tqdm, pytest
