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
| **Total Test Coverage** | | **119 tests passing** |

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
- [ ] `process_l2_batch()` - Multi-day L2 processing (not implemented)
- [ ] Resume capability for interrupted runs

### 4.4 File I/O
- [x] `load_laz_points()` - Read LAZ files via laspy
- [x] `save_dataset()` - Export to NetCDF
- [x] `export_profiles_json()` - JSON for web visualization (lidar_plot_data.json format)

### 4.5 CLI Interface
- [x] argparse CLI with subcommands: `l1`, `l2`, `batch`
- [x] `--config`, `--start`, `--end`, `--output` flags
- [x] `--bin-size`, `--mode-bin` parameters
- [ ] `--verbose` flag with proper logging
- [ ] Progress bar for long operations

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

### 8.2 Batch Processing
- [ ] Implement `process_l2_batch()` for multi-day L2
- [ ] Add resume capability (checkpoint saving)
- [ ] Parallel processing option (multiprocessing)

### 8.3 Logging & Progress
- [ ] Add proper logging throughout orchestration layer
- [ ] Progress bars for long operations (tqdm)
- [ ] `--verbose` CLI flag

### 8.4 Error Handling
- [ ] Add structured error messages for common failures
- [ ] Graceful handling of corrupt LAZ files
- [ ] Memory-efficient processing for large datasets

---

## Phase 9: Validation (TODO)

### 9.1 MATLAB Comparison
- [ ] Complete verification scripts (verify_phase1.py, verify_phase2.py)
- [ ] Create test dataset processed by both MATLAB and Python
- [ ] Quantitative comparison (RMSE, correlation)
- [ ] Document intentional differences

### 9.2 Real Data Testing
- [ ] Test on real LAZ files at `/Volumes/camera/DO-lidar/data-lvx/`
- [ ] Validate L1 output against existing MATLAB results
- [ ] Validate L2 runup statistics against MATLAB
- [ ] Performance benchmarking

### 9.3 Edge Cases
- [ ] Test with sparse point clouds
- [ ] Test with large gaps in data
- [ ] Test with outlier-heavy datasets
- [ ] Test date range edge cases (partial days, DST transitions)

---

## Phase 10: Documentation (TODO)

### 10.1 User Documentation
- [ ] Update CLAUDE.md with Python pipeline instructions
- [ ] Create `examples/` directory with usage examples
- [ ] Document config file format
- [ ] Document output file formats

### 10.2 Developer Documentation
- [ ] Add type stubs if needed
- [ ] Document internal algorithms
- [ ] Add architecture diagram

### 10.3 Deployment
- [ ] Test on Python 3.9, 3.10, 3.11
- [ ] Create setup.py or pyproject.toml for installation
- [ ] Docker container option

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
| `Get_intensity_contours.m` | `utils.get_intensity_contours()` | Complete (not integrated) |
| `detect_outliers_conv2D.m` | `utils.detect_outliers_conv2d()` | Complete (not integrated) |
| `inpaint_nans.m` | `profiles.inpaint_nans()` / `utils.inpaint_nans_*()` | Complete |
| `roundToHalfHour.m` | `utils.round_to_half_hour()` | Complete |
| `split_L1_days.m` | `utils.split_by_day()` | Complete |
| `L1_pipeline.m` | `phase4.process_l1()` | Complete |
| `L1_batchpipeline.m` | `phase4.process_l1_batch()` | Complete |
| `L2_pipeline.m` | `phase4.process_l2()` | Complete |
| `L2_batchpipeline.m` | `phase4.process_l2_batch()` | **TODO** |

---

## Next Steps (Priority Order)

1. **Install dependencies and run tests**
   ```bash
   pip install -r requirements.txt
   pytest tests/ -v
   ```

2. **Test on real data**
   ```bash
   python code/phase4.py l1 livox_config.json -o test_output.nc --start 2025-05-03 --end 2025-05-04
   ```

3. **Integrate outlier detection into L2** (Phase 8.1)
   - Add `detect_outliers_conv2d()` call in `process_l2()`

4. **Add multi-transect L2 extraction** (Phase 8.1)
   - Process multiple alongshore positions like MATLAB

5. **Complete MATLAB validation** (Phase 9.1)
   - Side-by-side comparison with known good MATLAB outputs

---

## Notes

- Python uses xarray/NetCDF instead of MATLAB .mat files for better interoperability
- All coordinate systems match MATLAB: UTM NAD83 Zone 11N, NAVD88 elevation
- Test data should be version-controlled separately (large files)
- Dependencies: numpy, scipy, pandas, laspy, xarray, netCDF4, matplotlib, pytest
