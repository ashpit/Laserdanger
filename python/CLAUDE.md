# CLAUDE.md - Python Lidar Pipeline

This file provides guidance to Claude Code when working with the Python lidar processing pipeline.

## Overview

This is a Python implementation of the lidar processing pipeline for analyzing Livox Avia point cloud data. It processes .laz format scans to generate time-resolved beach surface DEMs and wave runup statistics.

## Quick Start

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Process L1 data (single day)
python code/phase4.py l1 livox_config.json -o output.nc --start 2025-05-03 --end 2025-05-04

# Process L2 data (wave-resolving)
python code/phase4.py l2 livox_config.json -o l2_output.nc --start 2025-05-03 --end 2025-05-04

# Batch processing with resume capability
python code/phase4.py batch livox_config.json -o output_dir/ --start 2025-05-01 --end 2025-05-31
```

## Project Structure

```
python/
├── code/                    # Main source code
│   ├── phase1.py           # Config loading, file discovery, transforms
│   ├── phase2.py           # Point cloud binning, filtering
│   ├── phase3.py           # Data models (BinnedGrid, TimeResolvedDataset)
│   ├── phase4.py           # Pipeline orchestration (process_l1, process_l2)
│   ├── profiles.py         # Profile extraction utilities
│   ├── runup.py            # Runup detection and statistics
│   ├── utils.py            # General utilities
│   └── validation.py       # MATLAB comparison utilities
├── tests/                   # Test suite (pytest)
├── scripts/                 # Verification and utility scripts
├── examples/                # Usage examples
├── docs/                    # Documentation
└── requirements.txt         # Python dependencies
```

## Key Modules

### phase1.py - Core Infrastructure
- `load_config(path)` - Load and validate livox_config.json
- `discover_laz_files(folder, start_date, end_date)` - Find LAZ files in date range
- `transform_points(x, y, z, matrix)` - Apply 4x4 transformation matrix
- `filter_by_polygon(x, y, polygon)` - Filter points within boundary

### phase2.py - Point Cloud Processing
- `bin_point_cloud(x, y, z, bin_size)` - Spatial binning with statistics
- `apply_snr_filter(grid, snr_threshold, min_count)` - SNR-based filtering
- `residual_kernel_filter_two_stage(x, y, z)` - Two-stage ground filtering
- `bin_point_cloud_temporal(x, y, z, t, ...)` - Time-resolved binning (L2)

### phase3.py - Data Models
- `BinnedGrid` - 2D statistics grid (z_mean, z_max, z_min, z_mode, z_std, count)
- `TimeResolvedDataset` - L2 data container with Z(x,t), I(x,t) matrices
- `grid_to_dataset()`, `build_dataset()` - xarray conversion utilities

### phase4.py - Pipeline Orchestration
- `process_l1(config, ...)` - Full L1 processing pipeline
- `process_l2(config, ...)` - Full L2 processing pipeline
- `process_l1_batch(config, ...)` - Multi-day L1 with checkpointing
- `process_l2_batch(config, ...)` - Multi-day L2 with checkpointing

### profiles.py - Profile Extraction
- `extract_transects(grid, x_edges, y_edges, config, alongshore_spacings)`
- `inpaint_nans(z, x, max_gap)` - Gap interpolation
- `TransectConfig` - Transect configuration dataclass

### runup.py - Runup Detection
- `compute_runup_stats(Z_xt, I_xt, x1d, time_vec, ...)` - Full runup analysis
- `compute_dry_beach_reference(Z_xt, dt, window_s)` - Moving minimum filter
- `detect_runup_line(Z_xt, dry_ref, x1d, ...)` - Threshold crossing detection
- `compute_runup_spectrum(...)` - Spectral analysis (Welch's method)

### validation.py - MATLAB Comparison
- `compare_l1_outputs(python_path, matlab_path)` - Compare L1 outputs
- `compare_l2_outputs(python_path, matlab_path)` - Compare L2 outputs
- `ValidationReport` - Comparison results with RMSE, correlation, bias

## CLI Reference

```bash
# L1 Processing
python code/phase4.py l1 CONFIG [OPTIONS]
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  -o, --output PATH     Output file (.nc)
  --bin-size FLOAT      Spatial bin size in meters (default: 0.10)
  --mode-bin FLOAT      Mode histogram bin size (default: 0.05)
  --verbose             Enable debug logging
  --quiet               Suppress console output except errors
  --log-file PATH       Write logs to file
  --no-progress         Disable progress bars

# L2 Processing
python code/phase4.py l2 CONFIG [OPTIONS]
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  -o, --output PATH     Output file (.nc)
  --time-bin FLOAT      Time bin size in seconds (default: 0.5)
  --multi-transect      Extract multiple alongshore transects
  --no-outlier          Disable outlier detection
  --intensity-contours  Extract intensity contours

# Batch Processing
python code/phase4.py batch CONFIG [OPTIONS]
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  -o, --output DIR      Output directory
  --resume              Resume from checkpoint
  --parallel N          Use N parallel workers
```

## Configuration File Format

```json
{
  "dataFolder": "/path/to/raw/laz/files",
  "processFolder": "/path/to/processed/output",
  "plotFolder": "/path/to/visualization/output",
  "transformMatrix": [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0, 0, 0, 1]
  ],
  "LidarBoundary": [[x1, y1], [x2, y2], ..., [xn, yn]]
}
```

## Output Formats

### L1 NetCDF Output
- Coordinates: `x_edge`, `y_edge`, `time`
- Variables: `z_mean`, `z_max`, `z_min`, `z_mode`, `z_std`, `count`

### L2 NetCDF Output
- Coordinates: `x`, `time`
- Variables: `Z` (elevation), `I` (intensity), `outlier_mask` (optional)

### Validation Report JSON
```json
{
  "name": "L1 Comparison",
  "passed": true,
  "overall_rmse": 0.015,
  "overall_correlation": 0.995,
  "field_comparisons": [...]
}
```

## MATLAB Equivalents

| MATLAB Function | Python Function |
|-----------------|-----------------|
| `accumpts.m` | `phase2.bin_point_cloud()` |
| `accumpts_L2.m` | `phase2.bin_point_cloud_temporal()` |
| `ResidualKernelFilter.m` | `phase2.residual_kernel_filter_delaunay()` |
| `Get3_1Dprofiles.m` | `profiles.extract_transects()` |
| `get_runupStats_L2.m` | `runup.compute_runup_stats()` |
| `L1_pipeline.m` | `phase4.process_l1()` |
| `L2_pipeline.m` | `phase4.process_l2()` |
| `L1_batchpipeline.m` | `phase4.process_l1_batch()` |
| `L2_batchpipeline.m` | `phase4.process_l2_batch()` |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_phase2.py -v

# Run with coverage
pytest tests/ --cov=code --cov-report=html

# Run edge case tests
pytest tests/test_edge_cases.py -v
```

## Validation Against MATLAB

```bash
# Compare L1 outputs
python scripts/verify_l1.py python_output.nc matlab_output.mat

# Compare L2 outputs
python scripts/verify_l2.py python_output.nc matlab_output.mat

# Batch validation
python scripts/verify_l1.py python_dir/ matlab_dir/ --batch
```

## Common Tasks

### Process a single day of L1 data
```python
from phase1 import load_config
from phase4 import process_l1

config = load_config("livox_config.json")
result = process_l1(
    config,
    start_date="2025-05-03",
    end_date="2025-05-04",
    bin_size=0.10,
)
result.dataset.to_netcdf("output.nc")
```

### Extract profiles from L1 result
```python
from profiles import extract_transects, TransectConfig

config = TransectConfig(
    origin_x=500000.0,
    origin_y=3700000.0,
    azimuth=270.0,  # West
)

x1d, profiles = extract_transects(
    result.profiles[0].z_mode,
    result.x_edges,
    result.y_edges,
    config,
    alongshore_spacings=[-4, -2, 0, 2, 4],
)
```

### Compute runup statistics
```python
from runup import compute_runup_stats

Spec, Info, Bulk, Tseries = compute_runup_stats(
    Z_xt, I_xt, x1d, time_vec,
    threshold=0.1,
    windowlength=5,
)

print(f"Sig (IG): {Bulk.swashparams[0]:.3f} m")
print(f"Sinc: {Bulk.swashparams[1]:.3f} m")
print(f"Beach slope: {Bulk.beta:.4f}")
```

## Error Handling

The pipeline defines custom exceptions:
- `LidarProcessingError` - Base class for all pipeline errors
- `CorruptFileError` - Raised when a LAZ file cannot be read
- `NoDataError` - Raised when no valid data remains after filtering
- `ConfigurationError` - Raised for invalid configuration

Use `skip_corrupt=True` in batch processing to continue past corrupt files:
```python
result = process_l1_batch(
    config,
    output_dir,
    start_date="2025-05-01",
    end_date="2025-05-31",
    skip_corrupt=True,  # Continue on file errors
)
```

## Dependencies

- numpy>=1.21.0
- scipy>=1.7.0
- pandas>=1.3.0
- laspy>=2.0.0
- lazrs>=0.5.0 (fast Rust-based LAZ decompression)
- xarray>=0.19.0
- netCDF4>=1.5.0
- matplotlib>=3.4.0
- tqdm>=4.62.0
- pytest>=7.0.0 (for testing)

## Performance Optimizations

The pipeline includes several optimizations for processing large point cloud datasets:

### LAZ Decompression (lazrs)
- Uses `lazrs` Rust backend for 1.5-2x faster LAZ file reading
- Automatic fallback to laszip if lazrs unavailable
- Warning logged if lazrs not installed

### Polygon Filtering (matplotlib.path)
- Uses `matplotlib.path.Path.contains_points()` for C-optimized point-in-polygon testing
- 2-5x faster than pure Python ray casting
- Pure Python fallback available if matplotlib not installed

### Percentile Binning (np.argpartition)
- Uses `np.argpartition()` for O(n) percentile selection vs O(n log n) sorting
- Uses `np.bincount()` for fast mode calculation
- Optimized variance: `std = sqrt(mean(x²) - mean(x)²)`

### Future Optimization Opportunities
- Parallel file loading with multiprocessing (not yet implemented)
- `--fast` mode to skip residual kernel filtering
- Numba JIT compilation for hot loops (optional dependency)
