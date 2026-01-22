# CLAUDE.md

## Overview

Laserdanger is a lidar processing pipeline for analyzing Livox Avia point cloud data (.laz files) to generate beach surface DEMs and detect wave runup. Developed for Ashton Domi's PhD thesis at Scripps Institution of Oceanography.

**Active development is in `python/` only.** The MATLAB code in the repository root is vestigial and no longer maintained.

See `python/CLAUDE.md` for detailed Python pipeline documentation.

## Processing Levels

### L1: Daily Beach Surface DEMs
- **Purpose**: Generate high-resolution gridded beach topography
- **Output**: 2D surface grids with z_mean, z_max, z_min, z_mode, z_std, count
- **Resolution**: 10cm spatial bins (default), one file per day
- **Use case**: Beach morphology monitoring, volume change analysis, dry beach mapping

### L2: Wave-Resolving Timestacks
- **Purpose**: Capture water surface dynamics for wave runup analysis
- **Output**: Z(x,t) and I(x,t) matrices along cross-shore transects
- **Resolution**: ~2Hz temporal (0.5s bins), 10cm spatial along transect
- **Use case**: Wave runup detection, swash dynamics, runup spectra

## Quick Start

```bash
cd python
pip install -r requirements.txt
pytest tests/ -v
```

### L1 Processing (Daily Beach Surfaces)

```bash
# Process all days in data folder (auto-discovers date range)
python scripts/processing/run_daily_l1.py --config configs/do_livox_config_20260112.json

# Process specific date range
python scripts/processing/run_daily_l1.py --config configs/do_livox_config_20260112.json \
    --start 2026-01-12 --end 2026-01-15

# Preview what would be processed (dry run)
python scripts/processing/run_daily_l1.py --config configs/do_livox_config_20260112.json --dry-run

# Resume interrupted processing
python scripts/processing/run_daily_l1.py --config configs/do_livox_config_20260112.json --resume
```

### L2 Processing (Wave-Resolving Timestacks)

```bash
# Process all days in data folder
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json

# Process specific date range with higher temporal resolution
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json \
    --start 2026-01-20 --end 2026-01-21 --time-bin 0.25

# Enable outlier detection (disabled by default to preserve wave signals)
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json \
    --outlier-detection

# Preview what would be processed
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json --dry-run
```

## CLI Reference

### run_daily_l1.py

```
Options:
  --config PATH       Path to config file (required)
  --output-dir DIR    Output directory (default: python/data/level1)
  --start DATE        Override start date (YYYY-MM-DD)
  --end DATE          Override end date (YYYY-MM-DD)
  --bin-size FLOAT    Spatial bin size in meters (default: 0.1)
  --resume            Resume from checkpoint if available
  --dry-run           Show what would be processed without running
  --verbose, -v       Enable debug logging
  --quiet, -q         Suppress non-error output
  --no-progress       Disable progress bars
```

### run_daily_l2.py

```
Options:
  --config PATH       Path to config file (required)
  --output-dir DIR    Output directory (default: python/data/level2)
  --start DATE        Override start date (YYYY-MM-DD)
  --end DATE          Override end date (YYYY-MM-DD)
  --time-bin FLOAT    Temporal bin size in seconds (default: 0.5 = 2Hz)
  --x-bin FLOAT       Spatial bin size along transect in meters (default: 0.1)
  --multi-transect    Extract multiple alongshore transects
  --outlier-detection Enable outlier detection (off by default)
  --resume            Resume from checkpoint if available
  --dry-run           Show what would be processed without running
  --verbose, -v       Enable debug logging
  --quiet, -q         Suppress non-error output
  --no-progress       Disable progress bars
```

## Output Files

- **L1**: `python/data/level1/L1_YYYYMMDD.nc` - Daily gridded surfaces
- **L2**: `python/data/level2/L2_YYYYMMDD.nc` - Daily timestacks

## Directory Structure

```
python/
├── code/           # Main source (phase1-4, profiles, runup, utils, validation)
├── configs/        # Site/date-specific configs ({site}_livox_config_{YYYYMMDD}.json)
├── tests/          # pytest suite (186+ tests)
├── scripts/        # CLI scripts organized by purpose
│   ├── processing/     # Data processing (run_daily_l1.py, run_daily_l2.py)
│   ├── qc/             # Quality control & validation (verify_l1.py, verify_l2.py, assess_nc.py)
│   └── visualization/  # Plotting & figures (visualize_l1.py, visualize_l2.py, gif_nc_l1.py, plot_runup.py)
├── data/           # Output data (level1/, level2/)
└── docs/           # Documentation
```

## Configuration

Site/date-specific configs are stored in `python/configs/` with naming convention:
`{site}_livox_config_{YYYYMMDD}.json` (e.g., `do_livox_config_20260112.json`, `towr_livox_config_20260120.json`)

Sites:
- `do` = Director's Office
- `towr` = Tower

Config file fields:
- `dataFolder` - Path to raw .laz files
- `processFolder` - Output directory
- `transformMatrix` - 4×4 homogeneous transform (lidar → UTM)
- `LidarBoundary` - Polygon vertices (UTM) for spatial filtering

To add a new site, copy an existing config and update the paths and transform matrix.

## Dependencies

numpy, scipy, pandas, laspy, lazrs, xarray, netCDF4, matplotlib, tqdm, pytest
