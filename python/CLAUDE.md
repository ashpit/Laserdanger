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

# Process L1 data (batch) - use site/date-specific config from configs/
python scripts/processing/run_daily_l1.py --config configs/do_livox_config_20260112.json --start 2026-01-12 --end 2026-01-13

# Process L2 data (wave-resolving)
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json --start 2026-01-20 --end 2026-01-21

# Visualize all L1 outputs
python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json

# Create animated GIFs for all L1 files
python scripts/visualization/gif_nc_l1.py --config configs/do_livox_config_20260112.json

# Visualize all L2 outputs
python scripts/visualization/visualize_l2.py --config configs/towr_livox_config_20260120.json

# Validate against MATLAB
python scripts/qc/verify_l1.py python_output.nc matlab_output.mat
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
├── configs/                 # Site/date-specific configuration files
│   ├── do_livox_config_YYYYMMDD.json    # Director's Office site
│   └── towr_livox_config_YYYYMMDD.json  # Tower site
├── tests/                   # Test suite (pytest)
├── scripts/                 # CLI scripts organized by purpose
│   ├── processing/         # Data processing scripts
│   │   ├── run_daily_l1.py     # Batch L1 processing
│   │   └── run_daily_l2.py     # Batch L2 processing
│   ├── qc/                 # Quality control & validation
│   │   ├── qc_level1.py        # Comprehensive L1 diagnostics (config-driven)
│   │   ├── qc_level2.py        # Comprehensive L2 diagnostics (config-driven)
│   │   ├── verify_l1.py        # Compare Python L1 to MATLAB
│   │   └── verify_l2.py        # Compare Python L2 to MATLAB
│   └── visualization/      # Plotting & figure generation
│       ├── visualize_l1.py     # Standard L1 figures
│       ├── visualize_l2.py     # Standard L2 figures
│       ├── gif_nc_l1.py        # Animated L1 GIF with slope
│       ├── plot_runup.py       # Full runup analysis figures
│       └── plot_runup_timestack.py  # Publication-style runup timestack
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
- `get_scanner_position(transform_matrix)` - Extract scanner position from 4x4 transform
- `compute_transect_from_swath(X, Y, transform_matrix)` - Auto-compute transect from swath geometry
- `transect_config_from_dict(config_dict)` - Create TransectConfig from JSON dict

### mop.py - MOP Transect Integration
- `MopTable.load(path)` - Load MOP table from CSV
- `MopTable.get_mop(num)` - Get integer MOP transect
- `MopTable.get_fractional_mop(num)` - Get interpolated fractional MOP (e.g., 456.3)
- `MopTable.find_mops_in_bounds(...)` - Find MOPs within UTM bounds
- `select_best_mop(X, Y, table, ...)` - Auto-select optimal MOP for data
- `MopTransect.to_transect_config()` - Convert MOP to TransectConfig for processing
- `get_mop_transect(num, ...)` - Convenience function to get TransectConfig from MOP number
- `format_mop_filename_suffix(num)` - Generate filename suffix (e.g., "_MOP456")

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

### Processing Scripts

```bash
# L1 Processing (daily beach surfaces)
python scripts/processing/run_daily_l1.py --config CONFIG [OPTIONS]
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  --output-dir DIR      Output directory
  --bin-size FLOAT      Spatial bin size in meters (default: 0.10)
  --resume              Resume from checkpoint
  --dry-run             Preview without processing
  --verbose             Enable debug logging
  --no-progress         Disable progress bars

# L2 Processing (wave-resolving timestacks)
python scripts/processing/run_daily_l2.py --config CONFIG [OPTIONS]
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  --output-dir DIR      Output directory
  --time-bin FLOAT      Time bin size in seconds (default: 0.5)
  --x-bin FLOAT         Spatial bin size (default: 0.1)
  --tolerance FLOAT     Base transect tolerance in meters (default: 1.0)
  --expansion-rate FLOAT  Adaptive tolerance expansion (m/m), e.g., 0.02
  --multi-transect      Extract multiple alongshore transects
  --outlier-detection   Enable outlier detection
  --chunk-size INT      Process LAZ files in chunks (recommended: 8-10)
  --chunk-dir DIR       Directory for intermediate chunk files
  --keep-chunks         Keep intermediate chunks after processing
  --resume              Resume from checkpoint
  --mop FLOAT           Use MOP transect (supports fractional, e.g., 456.3)
  --mop-table PATH      Path to MOP CSV (default: mop_data/MopTable.csv)
  --auto-mop            Auto-select best MOP for data
  --mop-method STR      MOP selection: centroid (default), coverage, nearest_scanner
```

### QC Scripts

```bash
# Comprehensive L1 QC (config-driven, outputs to plotFolder/qc/level1/)
python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json
python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json --date 2026-01-12
python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json --no-figures

# Comprehensive L2 QC (config-driven, outputs to plotFolder/qc/level2/)
python scripts/qc/qc_level2.py --config configs/towr_livox_config_20260120.json
python scripts/qc/qc_level2.py --config configs/towr_livox_config_20260120.json --date 2026-01-20

# Verify L1 against MATLAB
python scripts/qc/verify_l1.py python.nc matlab.mat [-o report.json]
python scripts/qc/verify_l1.py python_dir/ matlab_dir/ --batch

# Verify L2 against MATLAB
python scripts/qc/verify_l2.py python.nc matlab.mat [-o report.json]
```

### Visualization Scripts

All visualization scripts use `--config` (required) to determine input/output directories.
They process ALL files by default, or use `--input` for a single file.

```bash
# L1 visualization (DEM, stats, profiles) → plotFolder/level1/
# Processes ALL L1 files from processFolder/level1/ by default
python scripts/visualization/visualize_l1.py --config CONFIG
python scripts/visualization/visualize_l1.py --config CONFIG --input L1_20260112.nc  # single file

# Animated L1 GIFs with slope calculation → plotFolder/level1/
# Uses auto-computed transect for cross-shore profiles (seaward at x=0)
# Processes ALL L1 files from processFolder/level1/ by default
python scripts/visualization/gif_nc_l1.py --config CONFIG
python scripts/visualization/gif_nc_l1.py --config CONFIG --input L1_20260112.nc  # single file
python scripts/visualization/gif_nc_l1.py --config CONFIG [--fps 2] [--save-slopes]

# L2 visualization (timestacks, intensity, wave detection) → plotFolder/level2/
# Processes ALL L2 files from processFolder/level2/ by default
python scripts/visualization/visualize_l2.py --config CONFIG
python scripts/visualization/visualize_l2.py --config CONFIG --input L2_20260120.nc  # single file

# Runup analysis figures → plotFolder/level2/
# Processes ALL L2 files from processFolder/level2/ by default
python scripts/visualization/plot_runup.py --config CONFIG
python scripts/visualization/plot_runup.py --config CONFIG --input L2_20260120.nc  # single file

# Publication-style runup timestack → plotFolder/level2/
# Processes ALL L2 files from processFolder/level2/ by default
python scripts/visualization/plot_runup_timestack.py --config CONFIG
python scripts/visualization/plot_runup_timestack.py --config CONFIG --input L2_20260120.nc
python scripts/visualization/plot_runup_timestack.py --config CONFIG --t-start 200 --t-end 360
```

Output path priority: `--output` > `--config` (plotFolder)

## Configuration

Site and date-specific configs are stored in `configs/` with naming convention:
`{site}_livox_config_{YYYYMMDD}.json`

- `do` = Director's Office site
- `towr` = Tower site
- Date indicates when the transform matrix was calibrated (use config valid for your data's date range)

To add a new site or update for a new calibration:
```bash
cp configs/do_livox_config_20260112.json configs/newsite_livox_config_20260122.json
# Edit with new paths and transform matrix
```

### Config File Format

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
  "LidarBoundary": [[x1, y1], [x2, y2], ..., [xn, yn]],
  "transect": {
    "x1": 476190.0, "y1": 3636210.0,
    "x2": 476120.0, "y2": 3636215.0,
    "resolution": 0.1,
    "tolerance": 2.0
  }
}
```

### Transect Auto-Computation

If no `transect` is specified in the config, both L1 and L2 processing **auto-compute** a cross-shore transect from the lidar swath geometry:

1. Scanner position is extracted from `transformMatrix` (the translation components tx, ty)
2. Data centroid is computed from the point cloud
3. Transect runs from scanner position through centroid, covering full data extent
4. Profile orientation is automatically flipped so x=0 is at the seaward (low elevation) end

This ensures consistent cross-shore profiles for slope calculation and wave runup analysis.

## Output Directory Structure

Config-driven scripts organize outputs as follows:

```
processFolder/                  # NetCDF data outputs
├── level1/
│   └── L1_YYYYMMDD.nc         # Daily gridded surfaces
└── level2/
    └── L2_YYYYMMDD.nc         # Daily timestacks

plotFolder/                     # Figures and QC reports
├── level1/                    # L1 visualization outputs
├── level2/                    # L2 visualization outputs
└── qc/
    ├── level1/                # L1 QC diagnostics
    │   ├── qc_report.json
    │   └── YYYYMMDD/          # Date-specific figures
    └── level2/                # L2 QC diagnostics
        ├── qc_report.json
        └── YYYYMMDD/          # Date-specific figures
```

## Output Formats

### L1 NetCDF Output
- Coordinates: `x_edge`, `y_edge`, `time`
- Variables: `z_mean`, `z_max`, `z_min`, `z_mode`, `z_std`, `count`

### L2 NetCDF Output
- Coordinates: `x` (cross-shore distance along transect), `time`
- Variables: `elevation`, `intensity`, `count`, `elevation_min`, `elevation_max`, `elevation_std`, `outlier_mask` (optional)
- Note: `x` is cross-shore distance (0 = seaward end) when processed with auto-transect

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
| (auto-transect) | `profiles.compute_transect_from_swath()` |
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
python scripts/qc/verify_l1.py python_output.nc matlab_output.mat

# Compare L2 outputs
python scripts/qc/verify_l2.py python_output.nc matlab_output.mat

# Batch validation
python scripts/qc/verify_l1.py python_dir/ matlab_dir/ --batch
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
from profiles import extract_transects, TransectConfig, compute_transect_from_swath
from phase1 import load_config

# Option 1: Auto-compute transect from swath geometry
cfg = load_config("livox_config.json")
transect = compute_transect_from_swath(
    X, Y, transform_matrix=cfg.transform_matrix
)

# Option 2: Manual transect config
transect = TransectConfig(
    x1=476190.0, y1=3636210.0,  # Landward end (near scanner)
    x2=476120.0, y2=3636215.0,  # Seaward end
    resolution=0.1,
)

x1d, profiles = extract_transects(
    result.profiles[0].z_mode,
    result.x_edges,
    result.y_edges,
    transect,
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
