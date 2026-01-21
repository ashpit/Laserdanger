# CLAUDE.md

## Overview

Laserdanger is a lidar processing pipeline for analyzing Livox Avia point cloud data (.laz files) to generate beach surface DEMs and detect wave runup. Developed for Ashton Domi's PhD thesis at Scripps Institution of Oceanography.

**Note:** Python implementation (`python/`) is a translation of the original MATLAB code (repository root). Active development is Python-only.

## Directory Structure

```
python/
├── code/
│   ├── phase1.py      # Config, file discovery, transforms, filtering
│   ├── phase2.py      # Point cloud binning, SNR filtering, ground extraction
│   ├── phase3.py      # Data models (xarray/NetCDF)
│   ├── phase4.py      # Pipeline orchestration, batch processing, CLI
│   ├── profiles.py    # Cross-shore profile extraction
│   ├── runup.py       # Runup detection and spectral analysis
│   ├── utils.py       # Outlier detection, intensity contours
│   └── validation.py  # MATLAB comparison utilities
├── tests/             # pytest suite
├── scripts/           # Visualization scripts
└── figures/           # Output figures
```

## Pipeline Architecture

### Phase 1: Infrastructure (`phase1.py`)
- `load_config()` - Load/validate `livox_config.json`
- `discover_laz_files()` - Find LAZ files by POSIX timestamp
- `transform_points()` - Apply 4×4 homogeneous transform (lidar → UTM)
- `filter_by_polygon()` - Boundary filtering via ray casting
- `filter_points()` - Combined polygon + intensity (<100) filtering

### Phase 2: Processing (`phase2.py`)
- `bin_point_cloud()` - Rasterize to grid, compute stats (mean/min/max/std/mode), SNR filtering
- `bin_point_cloud_temporal()` - L2 temporal binning (~2Hz Z(x,t) matrices)
- `residual_kernel_filter_delaunay()` - Delaunay triangulation + plane fitting for ground extraction
- `residual_kernel_filter_two_stage()` - Two-pass: 10m/0.2m then 3m/0.1m

### Phase 3: Data Models (`phase3.py`)
- `BinnedGrid` - 2D spatial grid dataclass
- `TimeResolvedGrid` - 2D temporal grid (t × x)
- `TimeResolvedDataset` - L2 container with Z(x,t), I(x,t), multi-transect support
- `to_xarray()`, `to_netcdf()` - Export functions

### Phase 4: Orchestration (`phase4.py`)
- `process_l1()` - Full L1 pipeline (beach surfaces)
- `process_l2()` - Full L2 pipeline (wave-resolving)
- `process_l1_batch()`, `process_l2_batch()` - Batch with checkpoint/resume
- `load_laz_points()` - LAZ file reading via laspy

### Supporting Modules
- **`profiles.py`**: `extract_transects()` - Shore-normal profile extraction
- **`runup.py`**: `compute_runup_stats()` - Runup detection, spectral analysis (IG/incident bands)
- **`utils.py`**: `detect_outliers_conv2d()`, `get_intensity_contours()`

## Running the Code

```bash
# Setup
cd python
pip install numpy scipy pandas laspy xarray netCDF4 matplotlib tqdm pytest

# L1 processing
python code/phase4.py l1 ../livox_config.json -o output.nc --start 2024-06-01 --end 2024-06-02

# L2 processing
python code/phase4.py l2 ../livox_config.json -o output.nc --time-bin 0.5 --multi-transect

# Batch with resume
python code/phase4.py batch ../livox_config.json -o output_dir/ --start 2024-06-01 --end 2024-06-30 --resume

# Tests
pytest tests/ -v
```

## Configuration

`livox_config.json` fields:
- `dataFolder` - Path to raw .laz files
- `processFolder` - Output directory
- `transformMatrix` - 4×4 homogeneous transform (lidar → UTM)
- `LidarBoundary` - Polygon vertices (UTM) for spatial filtering

## Key Parameters

| Parameter | Default | Location |
|-----------|---------|----------|
| bin_size | 0.10m | `bin_point_cloud()` |
| SNR threshold | 100 | `bin_point_cloud()` |
| min_count | 10 | `bin_point_cloud()` |
| time_bin_size | 0.5s (~2Hz) | `bin_point_cloud_temporal()` |
| residual threshold | 0.2m/0.1m | `residual_kernel_filter_two_stage()` |
| runup threshold | 0.1m | `compute_runup_stats()` |
| IG window | 100s | `compute_runup_stats()` |

## Dependencies

numpy, scipy, pandas, laspy, xarray, netCDF4, matplotlib, tqdm, pytest
