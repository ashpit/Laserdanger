# Repository Guidelines

## Project Structure & Module Organization
- `matlab/`: Core LiDAR pipelines. `L1_pipeline.m` builds 30-minute DEM-like outputs; `L2_pipeline_testing.m` experiments with higher-frequency swash/runup edges; helpers include `process_lidar_L1.m`, `ResidualKernelFilter.m`, and profile utilities.
- `data/`: Sample data (`plot_test05_17.mat`) for dry runs and visualization checks.
- `python/`: Python port of the LiDAR processing pipeline.
    - `code/`: Reusable modules (`phase1.py`, `phase2.py`, `phase3.py`).
    - `tests/`: Pytest suite mirroring MATLAB logic.

## Build, Test, and Development Commands
- MATLAB batch runs from `matlab/` with a valid `livox_config.json` there: `matlab -batch "L1_pipeline"` for L1 processing; `matlab -batch "L2_pipeline_testing"` for swash/runup trials. Run from `matlab/` so relative paths resolve.
- MATLAB unit tests (once added): `matlab -batch "runtests"` with tests on the MATLAB path or inside `matlab/`.
- Python tests: `python3 -m pytest` from `python/`.

## Coding Style & Naming Conventions
- MATLAB: 4-space indentation, vectorize when possible, preallocate arrays, and keep file names matching function names (lowerCamelCase with stage tags like `_L1`, `_L2`). Note config requirements at the top of scripts.
- Python: Follow PEP 8 with type hints. Place reusable modules in `python/code/`.
- Logging/prints: favor concise progress messages; avoid hard-coded local paths.

## Security & Configuration Tips
- `livox_config.json` is required but not versioned; include paths (`dataFolder`, `processFolder`, `plotFolder`), `transformMatrix`, and `LidarBoundary`. Keep credentials and raw data paths out of git; if sharing config, provide a redacted template only.

---

# Python Port Architecture

The Python implementation (`python/code/`) is designed as a modular, functional pipeline that mirrors the MATLAB L1/L2 logic but uses modern Python data science stacks (`numpy`, `scipy`, `xarray`, `pandas`).

## Phase 1: Ingestion & Preprocessing (`phase1.py`)
**Role:** Pure, side-effect-free data loading and coordinate transformation.
*   **Config Management:**
    *   `load_config(path)`: Parses `livox_config.json`, validating keys and enforcing strong types for paths and matrices.
*   **File Discovery:**
    *   `discover_laz_files(folder, start, end)`: Scans for `do-lidar_*.laz` files, extracting timestamps from filenames and filtering by date range.
*   **Coordinate Math:**
    *   `transform_points(points, tmatrix)`: Applies the 4x4 homogeneous transformation matrix to convert raw LiDAR coords to UTM.
*   **Filtering:**
    *   `filter_by_polygon(points, polygon)`: Efficient ray-casting implementation to mask points outside the defined beach boundary.
    *   `filter_points(...)`: Composite filter combining spatial polygon, intensity thresholds (retro-reflection removal), and time windowing.

## Phase 2: Core Algorithms (`phase2.py`)
**Role:** Heavy computational lifting using pure NumPy/SciPy. No I/O.
*   **Spatial Binning:**
    *   `bin_point_cloud(points, bin_size)`: Rasterizes 3D points into a 2D grid.
    *   **Statistics:** Computes Mean, Min, Max, Std, and Count per bin using `scipy.stats.binned_statistic_2d`.
    *   **Mode Calculation:** Quantizes elevations (e.g., 5cm steps) to compute the statistical mode, robust against outliers.
    *   **SNR:** Calculates Signal-to-Noise Ratio for each bin (`mean / (std/sqrt(count))`).
*   **Ground Filtering:**
    *   `residual_kernel_filter(points, window_radius, ...)`: Implements the "Residual Kernel Filter".
    *   **Logic:** Fits a local plane to a point's neighbors (using KD-tree). Points deviating > `max_residual` from this plane are discarded. This is critical for removing birds, spray, and temporary obstructions.

## Phase 3: Data Modeling (`phase3.py`)
**Role:** High-level data structures for temporal analysis.
*   **Data Structures:**
    *   `BinnedGrid`: Dataclass holding raw numpy arrays from Phase 2.
    *   `GridWithTime`: Binds a `BinnedGrid` to a specific UTC timestamp.
*   **Xarray Integration:**
    *   `grid_to_dataset(grid, timestamp)`: Converts a single grid into a labeled `xarray.Dataset` with coordinates `(y, x)` and variables (`elevation`, `snr`, etc.).
    *   `build_dataset(grids)`: Stacks multiple processed grids along a `time` dimension.
*   **Outcome:**
    *   Produces a 3D `(time, y, x)` DataCube, enabling powerful slicing (e.g., "get elevation profile at x=100 for all times") and NetCDF export.

## Phase 4: Orchestration (`phase4.py`)
**Role:** The "Driver" layer. Handles I/O and pipeline execution.
*   **Data Loading:**
    *   `load_laz_points(path)`: Uses `laspy` to read binary `.laz` files, extracting coordinates, intensity, and GPS time. Handles header scaling automatically.
*   **Pipeline Driver:**
    *   `process_l1(config_path, ...)`: The main entry point for L1 processing.
    *   **Workflow:**
        1.  Loads Config.
        2.  Discovers files within date range.
        3.  **Batch Loop:** Loads -> Transforms -> Filters (Phase 1) for each file.
        4.  **Alignment:** Computes global grid edges across *all* files to ensure spatial alignment.
        5.  **Binning:** Rasterizes each file onto the common grid (Phase 2).
        6.  **Assembly:** Stacks grids into an xarray Dataset (Phase 3).
*   **Export:**
    *   `save_dataset(ds, path)`: Saves the processed DataCube to NetCDF.