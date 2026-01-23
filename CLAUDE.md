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

# MEMORY-EFFICIENT: Process in chunks of 10 files at a time (recommended for large datasets)
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json \
    --chunk-size 10

# Resume interrupted chunked processing
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json \
    --chunk-size 10 --resume

# Enable outlier detection (disabled by default to preserve wave signals)
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json \
    --outlier-detection

# Preview what would be processed
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json --dry-run
```

### Quality Control

```bash
# Run comprehensive L1 QC diagnostics (outputs figures to plotFolder/qc/level1/)
python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json

# Run comprehensive L2 QC diagnostics (outputs figures to plotFolder/qc/level2/)
python scripts/qc/qc_level2.py --config configs/towr_livox_config_20260120.json

# QC for specific date only
python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json --date 2026-01-12
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
  --chunk-size INT    Process LAZ files in chunks (recommended: 8-10 for large datasets)
  --chunk-dir DIR     Directory for intermediate chunk files (default: temp)
  --keep-chunks       Keep intermediate chunk files after processing
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
│   ├── qc/             # Quality control (qc_level1.py, qc_level2.py, verify_l1.py, verify_l2.py)
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
- `processFolder` - Output directory for processed NetCDF files (level1/, level2/)
- `plotFolder` - Output directory for figures and QC reports (level1/, level2/, qc/)
- `transformMatrix` - 4×4 homogeneous transform (lidar → UTM)
- `LidarBoundary` - Polygon vertices (UTM) for spatial filtering
- `transect` - (optional) Cross-shore transect definition (see below)

### Transect Configuration

Both L1 and L2 processing **auto-compute** a cross-shore transect from the swath geometry if not specified. The auto-transect runs from the scanner position (extracted from `transformMatrix`) through the data centroid.

To override, add a `transect` block to your config:

```json
{
  "transect": {
    "x1": 476190.0, "y1": 3636210.0,
    "x2": 476120.0, "y2": 3636215.0,
    "resolution": 0.1,
    "tolerance": 2.0
  }
}
```

Or using origin + azimuth:
```json
{
  "transect": {
    "origin_x": 476190.0, "origin_y": 3636210.0,
    "azimuth": 265.0,
    "length": 80.0
  }
}
```

To add a new site, copy an existing config and update the paths and transform matrix.

## Visualization

All visualization scripts use `--config` (required) to auto-discover files and output to `plotFolder`.
By default they process ALL files; use `--input` for a single file.

```bash
# L1 visualizations → plotFolder/level1/
python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json
python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json --input L1_20260112.nc

# Animated L1 GIFs (processes ALL L1 files from processFolder/level1/)
python scripts/visualization/gif_nc_l1.py --config configs/do_livox_config_20260112.json
python scripts/visualization/gif_nc_l1.py --config configs/do_livox_config_20260112.json --input L1_20260112.nc

# L2 visualizations → plotFolder/level2/
python scripts/visualization/visualize_l2.py --config configs/towr_livox_config_20260120.json
python scripts/visualization/visualize_l2.py --config configs/towr_livox_config_20260120.json --input L2_20260120.nc

# Runup analysis → plotFolder/level2/
python scripts/visualization/plot_runup.py --config configs/towr_livox_config_20260120.json
python scripts/visualization/plot_runup_timestack.py --config configs/towr_livox_config_20260120.json
```

## Dependencies

numpy, scipy, pandas, laspy, lazrs, xarray, netCDF4, matplotlib, tqdm, pytest
