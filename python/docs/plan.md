# Python Lidar Pipeline - Implementation Plan

This document tracks the conversion of the MATLAB lidar processing pipeline to Python, providing an agentic checklist for completing the project.

**Goal**: A production-ready Python pipeline that replicates (or improves upon) the MATLAB codebase for processing Livox Avia lidar data into beach DEMs and wave runup statistics.

---

## Current Status Summary

| Component | Status | Tests |
|-----------|--------|-------|
| Phase 1: Config & File I/O | Complete | 88 lines |
| Phase 2: Point Cloud Processing | Complete | 506 lines |
| Phase 3: Data Models | Complete | 72 lines |
| Phase 4: L1/L2 Orchestration | Complete | 100 lines |
| Profile Extraction | Complete | 422 lines |
| Runup Detection | Complete | 328 lines |
| Utilities | Complete | 374 lines |
| L2 Enhancements (Phase 8.1) | Complete | 350 lines |
| Batch/Logging/Errors (Phase 8.2-8.4) | Complete | 27 tests |
| Validation Infrastructure (Phase 9) | Complete | 36 tests |
| Documentation (Phase 10) | Complete | - |
| Performance Optimizations (Phase 11) | Complete | - |
| **Total Test Coverage** | | **186 tests passing** |

---

## Phase 1: Core Infrastructure

### 1.1 Configuration Loading
- [x] `load_config()` - Parse `livox_config.json` with validation
- [x] Support all config fields: dataFolder, processFolder, plotFolder, transformMatrix, LidarBoundary
- [x] Unit tests for config loading

### 1.2 File Discovery
- [x] `discover_laz_files()` - Find `.laz` files with `do-lidar_<timestamp>.laz` pattern
- [x] Parse POSIX timestamp from filename
- [x] Date range filtering (start_date, end_date)
- [x] Return sorted list by timestamp
- [x] Unit tests for file discovery

### 1.3 Coordinate Transformation
- [x] `transform_points()` - Apply 4x4 homogeneous transformation matrix
- [x] Convert lidar native coordinates to UTM (NAD83 Zone 11N, NAVD88)
- [x] Unit tests for transformation

### 1.4 Spatial Filtering
- [x] `filter_by_polygon()` - Ray casting point-in-polygon test
- [x] Handle edge cases (points on boundary)
- [x] `filter_points()` - Combined polygon + intensity + time filtering
- [x] Unit tests for filtering

---

## Phase 2: Point Cloud Processing

### 2.1 Spatial Binning
- [x] `bin_point_cloud()` - Accumarray-style binning to regular grid
- [x] Compute statistics per bin: mean, max, min, std, count
- [x] Mode computation using histogram method
- [x] Configurable bin_size (default 0.10m)
- [x] Unit tests for binning

### 2.2 Percentile Filtering
- [x] 50th percentile pre-filtering within bins (matches MATLAB)
- [x] Configurable percentile parameter
- [x] Unit tests for percentile filtering

### 2.3 SNR Filtering
- [x] `compute_snr()` - Signal-to-noise ratio = mean / (std / sqrt(count))
- [x] SNR threshold filtering (default: 100)
- [x] Minimum count filtering (default: 10)
- [x] `apply_snr_filter()` - Extract valid bins as 1D point arrays
- [x] Unit tests for SNR filtering

### 2.4 Residual Kernel Filtering
- [x] `residual_kernel_filter()` - KD-tree based local plane fitting
- [x] `residual_kernel_filter_delaunay()` - Delaunay triangulation method (MATLAB equivalent)
- [x] `_fit_plane_svd()` - SVD-based plane fitting with outlier trimming
- [x] Remove points above fitted plane by threshold
- [x] Unit tests for residual filtering

### 2.5 Two-Stage Filtering
- [x] `residual_kernel_filter_two_stage()` - Iterative filtering
- [x] Pass 1: window=10m, threshold=0.2m
- [x] Pass 2: window=3m, threshold=0.1m
- [x] Unit tests for two-stage filtering

### 2.6 Temporal Binning (L2)
- [x] `bin_point_cloud_temporal()` - Time-resolved binning
- [x] Z(x,t), I(x,t) matrix output
- [x] Configurable time bin size (~0.5s for 2Hz)
- [x] Unit tests for temporal binning

---

## Phase 3: Data Models

### 3.1 Grid Structures
- [x] `BinnedGrid` dataclass - 2D statistics grid
- [x] `TimeResolvedGrid` dataclass - 3D (x, y, t) grid
- [x] `GridWithTime` - Grid + timestamp metadata
- [x] Unit tests for data structures

### 3.2 Dataset Building
- [x] `grid_to_dataset()` - Single grid to xarray.Dataset
- [x] `build_dataset()` - Stack multiple grids by time
- [x] Proper coordinate handling (x_edge, y_edge, time)
- [x] Unit tests for dataset building

### 3.3 L2 Dataset
- [x] `TimeResolvedDataset` class - L2 wave-resolving data container
- [x] Properties: x1d, time_vec, Z_xt, I_xt, dt, dx
- [x] `to_xarray()` method
- [x] `to_netcdf()` method
- [x] Unit tests for L2 dataset

---

## Phase 4: Pipeline Orchestration

### 4.1 L1 Processing
- [x] `process_l1()` - Full L1 pipeline
- [x] Discover files -> Load LAZ -> Transform -> Filter -> Bin -> Stack
- [x] Integration with residual filtering
- [x] Return `L1Result` with dataset + profiles
- [x] Unit tests for L1 pipeline

### 4.2 L2 Processing
- [x] `process_l2()` - Time-resolved L2 pipeline
- [x] Temporal binning at ~2Hz
- [x] Central transect extraction
- [x] Return `TimeResolvedDataset`
- [x] Unit tests for L2 pipeline

### 4.3 Batch Processing
- [x] `process_l1_batch()` - Multi-day L1 processing
- [x] Loop over date range with daily chunks
- [x] Progress reporting
- [x] `process_l2_batch()` - Multi-day L2 processing
- [x] Resume capability for interrupted runs (checkpoint saving)

### 4.4 File I/O
- [x] `load_laz_points()` - Read LAZ files via laspy
- [x] `save_dataset()` - Export to NetCDF
- [x] `export_profiles_json()` - JSON for web visualization (lidar_plot_data.json format)

### 4.5 CLI Interface
- [x] argparse CLI with subcommands: `l1`, `l2`, `batch`
- [x] `--config`, `--start`, `--end`, `--output` flags
- [x] `--bin-size`, `--mode-bin` parameters
- [x] `--verbose` flag with proper logging
- [x] Progress bar for long operations (tqdm)

---

## Phase 5: Profile Extraction

### 5.1 Transect Extraction
- [x] `extract_transects()` - Multi-transect profile extraction
- [x] Shore-normal projection
- [x] Configurable alongshore spacing (default: -8m to +10m)
- [x] Unit tests

### 5.2 Profile Cleaning
- [x] `_fit_quadratic_and_remove_outliers()` - Iterative outlier removal
- [x] Configurable threshold (default: 0.4m)
- [x] Unit tests

### 5.3 Gap Interpolation
- [x] `inpaint_nans()` - Linear interpolation of gaps
- [x] `gapsize()` - Gap size detection
- [x] Max gap limit (default: 4m)
- [x] Unit tests

### 5.4 Coordinate Utilities
- [x] `transect_to_utm()` - Cross-shore distance to UTM coordinates
- [x] `TransectConfig` dataclass
- [x] Unit tests

---

## Phase 6: Runup Detection

### 6.1 Dry Beach Reference
- [x] `compute_dry_beach_reference()` - Moving minimum filter
- [x] Configurable window (default: 100s)
- [x] Gap filling and smoothing
- [x] Unit tests

### 6.2 Runup Line Detection
- [x] `detect_runup_line()` - Threshold crossing detection
- [x] Adaptive search window (±0.5m from previous)
- [x] Configurable threshold (default: 0.1m)
- [x] Unit tests

### 6.3 Time Series Smoothing
- [x] `smooth_runup_timeseries()` - Median filter (1s window)
- [x] Gap interpolation
- [x] Unit tests

### 6.4 Spectral Analysis
- [x] `compute_runup_spectrum()` - Welch's method FFT
- [x] IG band: 0.004-0.04 Hz
- [x] Incident band: 0.04-0.25 Hz
- [x] Confidence bounds (chi-squared)
- [x] Unit tests

### 6.5 Bulk Statistics
- [x] `compute_bulk_stats()` - Summary statistics
- [x] Significant wave heights (Sig, Sinc)
- [x] Mean water level (eta)
- [x] 2% exceedance (R2)
- [x] Unit tests

### 6.6 Beach Slope
- [x] `estimate_beach_slope()` - Linear fit to stable regions
- [x] Outlier rejection
- [x] Unit tests

### 6.7 Output Structures
- [x] `RunupSpectrum` dataclass
- [x] `RunupBulk` dataclass
- [x] `RunupTimeseries` dataclass
- [x] `RunupResult` composite result

---

## Phase 7: Utilities

### 7.1 2D Outlier Detection
- [x] `detect_outliers_conv2d()` - Gradient/Laplacian spike detection
- [x] Configurable gradient threshold
- [x] NaN handling
- [x] Unit tests

### 7.2 Intensity Contours
- [x] `get_intensity_contours()` - Water edge from intensity
- [x] Single and multi-threshold support
- [x] Statistics computation (mean, std, valid fraction)
- [x] Unit tests

### 7.3 Time Utilities
- [x] `round_to_half_hour()` - 30-minute binning
- [x] `round_to_interval()` - Configurable interval rounding
- [x] `split_by_day()` - Group data by calendar day
- [x] `deduplicate_by_interval()` - Remove time duplicates
- [x] Unit tests

### 7.4 Gap Handling
- [x] `gapsize()` - Gap size array
- [x] `inpaint_nans_1d()` - 1D interpolation
- [x] `inpaint_nans_2d()` - 2D interpolation
- [x] Unit tests

---

## Phase 8: Integration & Polish

### 8.1 L2 Pipeline Enhancements ✅ COMPLETED
- [x] Integrate `detect_outliers_conv2d()` into L2 processing
  - Added `apply_outlier_detection` parameter (default True)
  - Configurable via `outlier_params` dict (ig_length, gradient_threshold_std, laplacian_threshold_std)
  - Results stored in `TimeResolvedDataset.outlier_mask` and `Z_filtered`
- [x] Integrate `get_intensity_contours()` into L2 processing
  - Added `extract_intensity_contours` parameter (default False)
  - Configurable thresholds via `intensity_contour_thresholds`
  - Results stored in `TimeResolvedDataset.intensity_contours`
- [x] Multi-transect L2 extraction
  - Added `multi_transect` parameter (default False for backward compat)
  - When True, processes all `alongshore_spacings` from profile_config
  - Results stored in `TimeResolvedDataset.transect_grids` dict
  - Access via `get_transect(offset)` and `get_transect_Z_xt(offset)`
- [x] Updated `TimeResolvedDataset` with new properties and xarray export
  - `n_transects`, `alongshore_offsets` properties
  - `Z_xt` returns filtered data when outlier detection applied
  - `Z_xt_raw` returns unfiltered data
  - 3D xarray export with alongshore dimension when grids have matching shapes
- [x] 15 new tests in `test_l2_enhancements.py` (all passing)

### 8.2 Batch Processing ✅ COMPLETED
- [x] Implement `process_l2_batch()` for multi-day L2
  - Mirrors `process_l1_batch()` with L2-specific parameters
  - Supports outlier detection, intensity contours, multi-transect options
- [x] Add resume capability (checkpoint saving)
  - `Checkpoint` dataclass for state persistence
  - JSON-based checkpoint files with timestamps
  - Auto-resume from last completed date
- [x] Parallel processing option (multiprocessing)
  - `ProcessPoolExecutor` for file-level parallelism
  - Configurable `max_workers` parameter
  - Progress tracking across parallel tasks

### 8.3 Logging & Progress ✅ COMPLETED
- [x] Add proper logging throughout orchestration layer
  - `configure_logging()` function with verbose/quiet/file options
  - Structured log messages at DEBUG/INFO/WARNING/ERROR levels
- [x] Progress bars for long operations (tqdm)
  - Added tqdm to requirements.txt
  - Optional `show_progress` parameter on all processing functions
- [x] `--verbose` CLI flag
  - Added `--verbose`, `--quiet`, `--log-file`, `--no-progress` flags
  - Verbose enables DEBUG logging, quiet shows only errors

### 8.4 Error Handling ✅ COMPLETED
- [x] Add structured error messages for common failures
  - `LidarProcessingError` base exception class
  - `CorruptFileError`, `NoDataError`, `ConfigurationError` subclasses
  - Detailed error messages with context
- [x] Graceful handling of corrupt LAZ files
  - `load_laz_points_safe()` wrapper with error catching
  - `skip_corrupt` parameter to continue on file errors
  - Failed files tracked in checkpoint
- [x] Memory-efficient processing for large datasets
  - `clear_memory()` function for garbage collection
  - `estimate_memory_usage()` for point cloud arrays
  - `chunked_file_iterator()` for memory-bounded batch processing
- [x] 27 new tests in `test_phase8_enhancements.py` (all passing)

---

## Phase 9: Validation Infrastructure ✅ COMPLETED

### 9.1 MATLAB Comparison ✅ COMPLETED
- [x] Complete verification scripts
  - `scripts/verify_l1.py` - L1 output comparison
  - `scripts/verify_l2.py` - L2 output comparison
- [x] Quantitative comparison utilities in `code/validation.py`
  - `compute_rmse()`, `compute_correlation()`, `compute_bias()`
  - `compare_arrays()`, `compare_l1_outputs()`, `compare_l2_outputs()`
  - `ValidationReport` and `FieldComparison` dataclasses
  - JSON export/import for reports
- [x] Document intentional differences
  - `get_intentional_differences()` returns documentation
  - Covers: data formats, coordinate handling, NaN handling, floating point precision
  - Defines acceptable thresholds (RMSE < 0.01m, correlation > 0.99)

### 9.2 Real Data Testing (Ready for Use)
- [ ] Test on real LAZ files at `/Volumes/camera/DO-lidar/data-lvx/`
- [ ] Validate L1 output against existing MATLAB results
- [ ] Validate L2 runup statistics against MATLAB
- [ ] Performance benchmarking

*Note: Infrastructure is in place. Run validation when MATLAB outputs are available.*

### 9.3 Edge Cases ✅ COMPLETED
- [x] Test with sparse point clouds - `test_edge_cases.py`
- [x] Test with large gaps in data - `test_edge_cases.py`
- [x] Test with outlier-heavy datasets - `test_edge_cases.py`
- [x] Test date range edge cases (midnight crossing, etc.) - `test_edge_cases.py`
- [x] 36 new tests in `test_validation.py` and `test_edge_cases.py`

---

## Phase 10: Documentation ✅ COMPLETED

### 10.1 User Documentation ✅ COMPLETED
- [x] Update CLAUDE.md with Python pipeline instructions
  - Created `python/CLAUDE.md` with full documentation
  - Quick start, CLI reference, API overview
- [x] Create `examples/` directory with usage examples
  - `basic_l1_processing.py` - L1 pipeline example
  - `l2_runup_analysis.py` - L2 and runup analysis example
  - `batch_processing.py` - Batch processing with checkpointing
- [x] Document config file format - `docs/formats.md`
- [x] Document output file formats - `docs/formats.md`
  - L1 NetCDF, L2 NetCDF, validation JSON, profile JSON, checkpoint JSON

### 10.2 Developer Documentation ✅ COMPLETED
- [x] MATLAB → Python mapping table in `plan.md`
- [x] Algorithm descriptions in function docstrings
- [x] Intentional differences documented in `validation.py`

### 10.3 Deployment ✅ COMPLETED
- [x] Create pyproject.toml for installation
  - Package metadata, dependencies, optional extras
  - Script entry points for CLI tools
  - pytest, black, isort, mypy configuration
- [ ] Test on Python 3.9, 3.10, 3.11 (manual verification needed)
- [ ] Docker container option (optional future enhancement)

---

## Phase 11: Performance Optimizations ✅ COMPLETED

### 11.1 LAZ Decompression ✅ COMPLETED
- [x] Add `lazrs` dependency for Rust-based LAZ decompression (1.5-2x faster)
- [x] Add runtime detection and warning if lazrs not installed
- [x] Automatic fallback to laszip backend

### 11.2 Polygon Filtering ✅ COMPLETED
- [x] Replace pure Python ray casting with `matplotlib.path.Path.contains_points()`
- [x] C-optimized point-in-polygon testing (2-5x faster)
- [x] Preserve pure Python fallback in `_filter_by_polygon_python()`

### 11.3 Percentile Binning ✅ COMPLETED
- [x] Replace `np.percentile()` with `np.argpartition()` for O(n) selection
- [x] Use `np.bincount()` for fast mode histogram calculation
- [x] Optimize variance calculation: `std = sqrt(mean(x²) - mean(x)²)`

### 11.4 Future Optimizations (Not Yet Implemented)
- [ ] Parallel file loading with `multiprocessing.ProcessPoolExecutor`
- [ ] `--fast` CLI mode to skip residual kernel filtering
- [ ] Optional Numba JIT compilation for hot loops
- [ ] Memory-mapped file reading for very large datasets

---

## Quick Reference: MATLAB → Python Mapping

| MATLAB Function | Python Function | Status |
|-----------------|-----------------|--------|
| `accumpts.m` | `phase2.bin_point_cloud()` | Complete |
| `accumpts_L2.m` | `phase2.bin_point_cloud_temporal()` | Complete |
| `fitPlane.m` | `phase2._fit_plane_svd()` | Complete |
| `ResidualKernelFilter.m` | `phase2.residual_kernel_filter_delaunay()` | Complete |
| Two-stage filtering | `phase2.residual_kernel_filter_two_stage()` | Complete |
| SNR filtering | `phase2.apply_snr_filter()` | Complete |
| `lidar_config.m` | `phase1.load_config()` | Complete |
| `Get3_1Dprofiles.m` | `profiles.extract_transects()` | Complete |
| `Get1D_profiles_swash_accum.m` | (use `extract_transects` with L2) | Partial |
| `gapsize.m` | `profiles.gapsize()` / `utils.gapsize()` | Complete |
| `get_runupStats_L2.m` | `runup.compute_runup_stats()` | Complete |
| `Get_intensity_contours.m` | `utils.get_intensity_contours()` | Complete (integrated in L2) |
| `detect_outliers_conv2D.m` | `utils.detect_outliers_conv2d()` | Complete (integrated in L2) |
| `inpaint_nans.m` | `profiles.inpaint_nans()` / `utils.inpaint_nans_*()` | Complete |
| `roundToHalfHour.m` | `utils.round_to_half_hour()` | Complete |
| `split_L1_days.m` | `utils.split_by_day()` | Complete |
| `L1_pipeline.m` | `phase4.process_l1()` | Complete |
| `L1_batchpipeline.m` | `phase4.process_l1_batch()` | Complete |
| `L2_pipeline.m` | `phase4.process_l2()` | Complete |
| `L2_batchpipeline.m` | `phase4.process_l2_batch()` | Complete |

---

## Next Steps (Priority Order)

1. **Test on real data with performance optimizations**
   ```bash
   # Ensure lazrs is installed for fastest LAZ decompression
   pip install lazrs

   # Run L1 batch processing
   python scripts/run_daily_l1.py --config ../livox_config.json --verbose
   ```

2. **Validate against MATLAB outputs** (Phase 9.2)
   - Side-by-side comparison with known good MATLAB results
   - Performance benchmarking with optimizations enabled

3. **Implement parallel file loading** (Phase 11.4)
   - Add `--parallel N` flag to use multiple workers
   - Target: 2-4x speedup on multi-core systems

4. **Add `--fast` mode** (Phase 11.4)
   - Skip residual kernel filtering for quick previews
   - Useful for initial data exploration

---

## Notes

- Python uses xarray/NetCDF instead of MATLAB .mat files for better interoperability
- All coordinate systems match MATLAB: UTM NAD83 Zone 11N, NAVD88 elevation
- Test data should be version-controlled separately (large files)
- Dependencies: numpy, scipy, pandas, laspy, lazrs, xarray, netCDF4, matplotlib, pytest
- Performance: lazrs provides 1.5-2x faster LAZ decompression than laszip
