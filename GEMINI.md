# Laserdanger Project Context

## Project Overview
Laserdanger is a MATLAB-based LiDAR data processing pipeline designed to analyze point cloud data from a Livox Avia sensor. The system processes raw `.laz` files to generate:
1.  **Level 1 (L1):** Hourly/Daily beach surface Digital Elevation Models (DEMs).
2.  **Level 2 (L2):** High-frequency (2Hz) wave runup analysis.

The project was developed by Ashton Domi (Scripps Institution of Oceanography) for coastal process analysis.

## Architecture

The system is organized into two primary processing levels:

### Level 1: Beach Surface Morphology (L1)
*   **Goal:** Create clean, 30-minute average beach surfaces.
*   **Entry Point:** `matlab/L1_pipeline.m` (or `matlab/L1_batchpipeline.m` for batch mode).
*   **Key Steps:**
    1.  **Ingestion:** Reads `.laz` files based on time range.
    2.  **Transformation:** Applies homogeneous transformation matrix (LiDAR -> UTM coordinates) defined in `livox_config.json`.
    3.  **Binning:** Rasterizes point clouds into 0.10m bins using `accumpts.m`.
    4.  **Filtering:**
        *   **Intensity Filter:** Removes points with intensity > 100.
        *   **SNR Filter:** Removes noisy bins based on signal-to-noise ratio.
        *   **Residual Kernel Filter:** Uses `ResidualKernelFilter.m` (Delaunay triangulation + planar fitting) to remove off-ground noise (2 passes).
    5.  **Output:** `L1` struct containing `X`, `Y`, `Zmean`, `Zmin`, `Zmode`, etc., organized by date.
    6.  **Visualization:** Extracts 1D cross-shore profiles (`Get3_1Dprofiles.m`) and exports to `lidar_plot_data.json`.

### Level 2: Wave Runup Analysis (L2)
*   **Goal:** Resolve wave runup dynamics at 2Hz.
*   **Entry Point:** `matlab/L2_pipeline.m` (or `L2_pipeline_testing.m`).
*   **Key Steps:**
    1.  **Temporal Accumulation:** Uses `accumpts_L2.m` to bin data in both space and time (Z(x,t)).
    2.  **Runup Detection:** `get_runupStats_L2.m` identifies the water edge/runup line.
    3.  **Analysis:** Computes spectral stats (IG vs. Incident band), beach slope (beta), and runup time series.

## Key Files & Directories

### `matlab/`
*   **`L1_pipeline.m`**: Core script for daily beach surface processing.
*   **`L2_pipeline.m`**: Core script for wave runup processing.
*   **`lidar_config.m`**: Generates the JSON configuration files (e.g., `livox_config.json`) containing paths, transformation matrices, and boundary polygons.
*   **`accumpts.m`**: Function for spatial binning and statistical calculation of point cloud data.
*   **`ResidualKernelFilter.m`**: Sophisticated noise removal algorithm using iterative planar fitting.
*   **`Get3_1Dprofiles.m`**: Extracts 1D elevation profiles along specific shore-normal transects.
*   **`get_runupStats_L2.m`**: Computes runup statistics from time-stacked elevation matrices.

### `python/`
*   Currently contains empty scaffolding directories (`code`, `figures`, `tests`).

### `data/`
*   Contains sample data or test artifacts (e.g., `plot_test05_17.mat`).

## Usage

### 1. Configuration
Modify `matlab/lidar_config.m` to set:
*   `dataFolder`: Path to raw `.laz` files.
*   `processFolder`: Output directory.
*   `transformMatrix`: 4x4 matrix for coordinate conversion (specific to deployment period).
*   `LidarBoundary`: Polygon defining the valid beach area.

Run the script to generate `livox_config.json`.

### 2. Running L1 Pipeline
```matlab
% In MATLAB
run lidar_config.m  % Generate config
L1_pipeline         % Run processing
```

### 3. Running L2 Pipeline
```matlab
% In MATLAB
config = jsondecode(fileread('livox_config.json'));
L2_pipeline
```

## Development Conventions

*   **Language:** Primarily MATLAB.
*   **Data Structure:** Structured arrays (`struct`) are used extensively for passing data (e.g., `L1` struct, `DO` struct).
*   **Coordinate System:** Output is in UTM (NAD83, Zone 11N) with NAVD88 elevation.
*   **Filtering:** Multi-stage filtering is standard (Intensity -> SNR -> Spatial Boundary -> Residual Kernel).
*   **Testing:** No formal test suite observed; manual verification via plotting (`L2_pipeline_testing.m`) is implied.
