# Python Pipeline Conversion Plan

This document tracks the conversion of the MATLAB lidar processing pipeline to Python.

## Current Status

**Core algorithms (Phase 1-2):** ✅ 100% complete
**Data models (Phase 3):** ✅ 100% complete
**Orchestration (Phase 4):** ✅ 100% complete
**Profile extraction:** ✅ 100% complete
**Runup analysis:** ✅ 100% complete
**Utilities:** ✅ 100% complete
**CLI:** ✅ 100% complete

### Ready for Testing on Real Data
All core pipeline components are implemented. To run:
```bash
pip install -r requirements.txt
python code/phase4.py l1 livox_config.json -o output.nc --start 2024-06-15 --end 2024-06-16
```

---

## Phase 1: Complete Core L1 Pipeline ✅ COMPLETED

### 1.1 SNR Filtering Integration
- [x] Add SNR threshold filtering to `phase2.py` (threshold=100, min_count=10)
- [x] Integrate SNR mask into `bin_point_cloud()` output via `valid_mask` field
- [x] Add unit tests for SNR filtering edge cases
- [x] Add `apply_snr_filter()` function to extract valid data as 1D vectors (matches MATLAB output)
- [x] Add `grid_to_points()` utility function

### 1.2 Two-Stage Residual Kernel Filtering
- [x] Update `residual_kernel_filter()` to support iterative passes
- [x] Implement MATLAB's two-stage approach via `residual_kernel_filter_two_stage()`:
  - Pass 1: window=10m, threshold=0.2m
  - Pass 2: window=3m, threshold=0.1m
- [x] Add Delaunay triangulation option via `residual_kernel_filter_delaunay()` (matches MATLAB's `ResidualKernelFilter.m`)
- [x] Add robust plane fitting with outlier trimming via `_fit_plane_svd()`
- [x] Add `_points_in_triangle()` helper for barycentric point-in-triangle testing
- [x] Add tests for Delaunay filter and two-stage filtering

### 1.3 50th Percentile Pre-filtering
- [x] Add percentile filtering option to `bin_point_cloud()` (MATLAB uses 50th percentile)
- [x] Make percentile configurable (default=50)
- [x] Implement `_bin_with_percentile_filter()` helper function
- [x] Add tests for percentile filtering behavior

---

## Phase 2: Profile Extraction (`Get3_1Dprofiles`) ✅ COMPLETED

### 2.1 Core Profile Extraction
- [x] Create `profiles.py` module
- [x] Implement `extract_transects()` function:
  - Project 3D points onto shore-normal transects
  - Support configurable alongshore spacing via `TransectConfig`
  - Return `TransectResult` with Z3D matrix (n_transects × n_crossshore_positions)
- [x] Add `extract_single_transect()` convenience function
- [x] Add `transect_to_utm()` for coordinate conversion

### 2.2 Profile Cleaning
- [x] Implement quadratic outlier removal via `_fit_quadratic_and_remove_outliers()`
  - Iterative robust fitting (up to 3 iterations)
  - Configurable threshold (default 0.4m)
- [x] Implement gap interpolation via `inpaint_nans()`
  - Only fills gaps smaller than `max_gap` (default 4m)
- [x] Add `gapsize()` utility function (matches MATLAB)

### 2.3 Swash Zone Profiles
- [ ] Port `Get1D_profiles_swash_accum.m` logic (deferred to L2 pipeline work)

### 2.4 Tests (25 tests, all passing)
- [x] Unit tests for `gapsize()` (6 tests)
- [x] Unit tests for `inpaint_nans()` (4 tests)
- [x] Unit tests for quadratic outlier removal (3 tests)
- [x] Unit tests for point projection geometry (2 tests)
- [x] Integration tests for `extract_transects()` (5 tests)
- [x] Input validation tests (3 tests)
- [x] UTM conversion tests (2 tests)

---

## Phase 3: L2 Wave-Resolving Pipeline ✅ COMPLETED

### 3.1 Temporal Binning Enhancements
- [x] Verify `bin_point_cloud_temporal()` matches MATLAB's `accumpts_L2.m`
- [x] Add centered transect extraction for Z(x,t) output
- [x] Add intensity array output I(x,t)

### 3.2 L2 Orchestration
- [x] Add `process_l2()` function to `phase4.py`
- [x] Implement ~2Hz temporal resolution (configurable via `time_bin_size`)
- [x] Output Z(x,t), X(x,t), I(x,t) arrays

### 3.3 L2 Data Model
- [x] Add `TimeResolvedDataset` to `phase3.py`
- [x] Store time-resolved grids in xarray with proper dimensions
- [x] Add metadata for spectral analysis downstream

---

## Phase 4: Runup Detection (`get_runupStats_L2`) ✅ COMPLETED

### 4.1 Core Runup Detection
- [x] Create `runup.py` module
- [x] Implement moving minimum filter for dry beach reference (window=100s)
- [x] Implement water level computation: `water_level = Z - dry_beach`
- [x] Implement runup line detection via threshold crossing (default=0.1m)

### 4.2 Adaptive Search Window
- [x] Track runup position across time with ±0.5m search window
- [x] Implement median filtering (1s window) for smoothing
- [x] Implement gap interpolation for missing detections

### 4.3 Spectral Analysis
- [x] Compute frequency spectra of runup time series
- [x] Separate IG band (0.004-0.04 Hz) and incident band (0.04-0.25 Hz)
- [x] Compute significant wave heights: Sig (IG), Sinc (incident)
- [x] Estimate mean water level (eta)

### 4.4 Beach Slope Estimation
- [x] Implement foreshore slope (beta) estimation from stable beach regions
- [x] Add slope validation and outlier rejection

### 4.5 Output Structure
- [x] Return dataclass with:
  - `Spec`: frequency spectra (RunupSpectrum)
  - `Bulk`: Sig, Sinc, eta, beta, R2 (RunupBulk)
  - `Tseries`: Xrunup, Zrunup time series (RunupTimeseries)

### 4.6 Tests
- [x] Unit tests for moving minimum filter
- [x] Unit tests for threshold crossing detection
- [x] Integration tests with synthetic wave data
- [ ] Validation against MATLAB outputs (requires real data)

---

## Phase 5: Utility Functions ✅ COMPLETED

### 5.1 Gap/Interpolation Utilities
- [x] `inpaint_nans_1d()` and `inpaint_nans_2d()` - NaN interpolation using scipy
- [x] `gapsize()` - Detect gap sizes in 1D arrays
- [x] `detect_outliers_conv2d()` - 2D convolution-based outlier detection

### 5.2 Time Utilities
- [x] `round_to_half_hour()` - Time binning utility
- [x] `round_to_interval()` - Configurable interval rounding
- [x] `split_by_day()` - Split datasets by calendar day
- [x] `deduplicate_by_interval()` - Remove duplicates by time interval

### 5.3 Intensity Contours
- [x] `get_intensity_contours()` - Water edge detection from intensity

---

## Phase 6: CLI and Batch Processing ✅ COMPLETED

### 6.1 Command-Line Interface
- [x] Add `argparse` CLI to `phase4.py`
- [x] Support commands: `l1`, `l2`, `batch`
- [x] Add `--config`, `--start`, `--end`, `--output` flags

### 6.2 Batch Processing
- [x] Implement `process_l1_batch()` for multiple days
- [x] Add progress reporting and logging
- [ ] Add resume capability for interrupted runs (deferred)

### 6.3 JSON Export
- [x] Export 1D profiles to JSON via `export_profiles_json()`
- [x] Match MATLAB's `lidar_plot_data.json` format

---

## Phase 7: Validation and Documentation

### 7.1 Cross-Validation
- [ ] Create test dataset processed by both MATLAB and Python
- [ ] Compare outputs quantitatively (RMSE, correlation)
- [ ] Document any intentional differences

### 7.2 Documentation
- [x] Add docstrings to all public functions
- [ ] Create usage examples in `examples/` directory
- [ ] Update CLAUDE.md with Python pipeline instructions

### 7.3 Dependencies
- [x] Create `requirements.txt` with pinned versions
- [ ] Test on Python 3.9, 3.10, 3.11
- [x] Document laspy installation requirements

---

## Quick Reference: MATLAB → Python Mapping

| MATLAB | Python | Status |
|--------|--------|--------|
| `accumpts.m` | `phase2.bin_point_cloud()` | ✅ Done (with percentile + SNR filtering) |
| `accumpts_L2.m` | `phase2.bin_point_cloud_temporal()` | ✅ Done |
| `fitPlane.m` | `phase2._fit_plane_svd()` | ✅ Done (with robust outlier trimming) |
| `ResidualKernelFilter.m` | `phase2.residual_kernel_filter_delaunay()` | ✅ Done |
| Two-stage filtering | `phase2.residual_kernel_filter_two_stage()` | ✅ Done |
| SNR filtering | `phase2.apply_snr_filter()` | ✅ Done |
| `lidar_config.m` | `phase1.load_config()` | ✅ Done |
| `Get3_1Dprofiles.m` | `profiles.extract_transects()` | ✅ Done |
| `Get1D_profiles_swash_accum.m` | `profiles.extract_swash_transects()` | ⚠️ Deferred (use extract_transects with L2 data) |
| `gapsize.m` | `profiles.gapsize()` + `utils.gapsize()` | ✅ Done |
| `get_runupStats_L2.m` | `runup.compute_runup_stats()` | ✅ Done |
| `Get_intensity_contours.m` | `utils.get_intensity_contours()` | ✅ Done |
| `detect_outliers_conv2D.m` | `utils.detect_outliers_conv2d()` | ✅ Done |
| `inpaint_nans.m` | `profiles.inpaint_nans()` + `utils.inpaint_nans_1d/2d()` | ✅ Done |
| `roundToHalfHour.m` | `utils.round_to_half_hour()` | ✅ Done |
| `split_L1_days.m` | `utils.split_by_day()` + `utils.deduplicate_by_interval()` | ✅ Done |
| `L1_pipeline.m` | `phase4.process_l1()` | ✅ Done |
| `L1_batchpipeline.m` | `phase4.process_l1_batch()` | ✅ Done |
| `L2_pipeline.m` | `phase4.process_l2()` | ✅ Done |

---

## Suggested Order of Implementation

1. **Phase 1.1-1.3** - Complete core L1 (SNR + two-stage filtering)
2. **Phase 2** - Profile extraction (critical for analysis)
3. **Phase 5.1** - Gap/interpolation utilities (needed by profiles)
4. **Phase 3** - L2 temporal pipeline
5. **Phase 4** - Runup detection
6. **Phase 6** - CLI and batch processing
7. **Phase 7** - Validation and docs

---

## Notes

- Python uses xarray/NetCDF instead of MATLAB .mat files for better interoperability
- All coordinate systems match MATLAB: UTM NAD83 Zone 11N, NAVD88 elevation
- Test data should be version-controlled separately (large files)
