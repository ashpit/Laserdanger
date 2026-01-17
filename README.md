# Laserdanger

**Laserdanger** is a MATLAB-based LiDAR data processing pipeline designed to analyze point cloud data from a Livox Avia sensor. The system processes raw `.laz` files to generate time-resolved beach surface Digital Elevation Models (DEMs) and resolve high-frequency wave runup dynamics.

This project was developed by **Ashton Domi** for his PhD thesis at the **Scripps Institution of Oceanography** (Coastal Processes Group).

---

## Project Overview

The repository implements a two-level processing pipeline to analyze coastal morphology and wave dynamics:

1.  **Level 1 (L1): Beach Surface Morphology**
    *   Generates clean, 30-minute average beach surface DEMs.
    *   Used for analyzing morphological changes over hours or days.
2.  **Level 2 (L2): Wave Runup Analysis**
    *   Resolves wave runup dynamics at high frequency (2Hz).
    *   Used for studying swash zone processes and wave runup statistics.

## Architecture & Pipelines

### Level 1 (L1): Hourly/Daily Beach Surface
*   **Goal:** Create vectorized beach surfaces (min, max, mode, mean) organized by date.
*   **Entry Point:** `matlab/L1_pipeline.m` (or `matlab/L1_batchpipeline.m` for batch mode).
*   **Key Steps:**
    1.  **Ingestion:** Reads `.laz` files based on a specified time range.
    2.  **Transformation:** Applies a homogeneous transformation matrix (LiDAR frame → UTM coordinates) defined in `livox_config.json`.
    3.  **Binning:** Rasterizes point clouds into 0.10m spatial bins using `accumpts.m`.
    4.  **Filtering:**
        *   **Intensity Filter:** Removes points with intensity > 100.
        *   **SNR Filter:** Removes noisy bins based on Signal-to-Noise Ratio (SNR = mean/standard_error).
        *   **Residual Kernel Filter:** Uses `ResidualKernelFilter.m` (Delaunay triangulation + planar fitting) to remove off-ground noise (two passes: 10m/0.2m and 3m/0.1m).
    5.  **Output:** `L1` struct containing `X`, `Y`, `Zmean`, `Zmin`, `Zmode`, etc.
    6.  **Visualization:** Extracts 1D cross-shore profiles (`Get3_1Dprofiles.m`) and exports to `lidar_plot_data.json`.

### Level 2 (L2): Wave Runup Analysis
*   **Goal:** Create time-stacked elevation matrices `Z(x,t)` and detect the water runup edge.
*   **Entry Point:** `matlab/L2_pipeline.m` (or `matlab/L2_pipeline_testing.m`).
*   **Key Steps:**
    1.  **Temporal Accumulation:** Uses `accumpts_L2.m` to bin data in both space and time (~2Hz).
    2.  **Profile Extraction:** Generates `Z(x,t)`, `X(x,t)`, and `I(x,t)` arrays along a centered transect.
    3.  **Runup Detection:** `get_runupStats_L2.m` identifies the instantaneous water edge.
    *   Analysis: Computes spectral stats (IG vs. Incident band), beach slope (beta), and runup time series.

### Python Port (New)
A modern, modular Python implementation of the pipeline is currently in development under `python/code/`.
*   **Goal:** Replicate MATLAB L1 functionality using a robust, open-source stack (numpy, scipy, xarray).
*   **Structure:**
    *   **Phase 1:** Pure ingestion & preprocessing.
    *   **Phase 2:** Core algorithms (binning, filtering).
    *   **Phase 3:** Data modeling with xarray.
    *   **Phase 4:** Pipeline orchestration.
*   **Documentation:** See [AGENTS.md](AGENTS.md) for detailed architecture and usage of the Python modules.

---

## Key Functions

### Core Processing
*   **`accumpts.m`**: Bins 3D points to a grid. Applies a 50th percentile filter within bins and computes statistics (mean, max, min, mode, std). Filters unreliable bins based on SNR.
*   **`accumpts_L2.m`**: Extends `accumpts.m` to include the time dimension for L2 processing.
*   **`ResidualKernelFilter.m`**: Advanced noise removal. Creates a Delaunay triangulation of the point cloud, fits a plane to each triangle using SVD, and removes points exceeding a threshold distance from the plane.

### Profile & Runup
*   **`Get3_1Dprofiles.m`**: Projects the 3D point cloud onto multiple shore-normal transects (e.g., at -8m to +10m alongshore spacing). Uses quadratic fitting to remove residual outliers.
*   **`get_runupStats_L2.m`**: Detects the runup line from `Z(x,t)` matrices.
    *   Constructs a "dry beach" reference surface using a moving minimum filter (default 100s window).
    *   Identifies the water edge where elevation exceeds the dry beach by a threshold (default 0.1m).
    *   Computes spectral analysis (IG and Incident bands) and foreshore slope.

### Utilities
*   **`lidar_config.m`**: Generates JSON configuration files (e.g., `livox_config.json`) containing paths, transformation matrices, and boundary polygons.
*   **`fitPlane.m`**: SVD-based plane fitting helper.
*   **`inpaint_nans.m`**: Interpolates NaN values to fill gaps in data.
*   **`roundToHalfHour.m`**: Utility for time binning.

---

## Configuration

The system is configured via `matlab/lidar_config.m`, which generates the JSON config file used by the pipelines.

**Key Configuration Fields:**
*   **`dataFolder`**: Path to raw `.laz` files.
*   **`processFolder`**: Output directory for processed data.
*   **`transformMatrix`**: 4x4 homogeneous matrix to convert from LiDAR sensor coordinates to UTM (NAD83, Zone 11N). *Note: Different matrices may be required for different deployment periods.*
*   **`LidarBoundary`**: Polygon vertices defining the valid spatial extent (beach area) in UTM coordinates.

---

## Usage

### 1. Setup Configuration
Open `matlab/lidar_config.m`, adjust the paths and transformation matrix for your dataset, and run it to generate `livox_config.json`.

```matlab
run matlab/lidar_config.m
```

### 2. Run L1 Pipeline (Beach Surface)
To process daily beach surfaces:

```matlab
% In MATLAB
L1_pipeline
% Or for batch processing:
L1_batchpipeline
```

### 3. Run L2 Pipeline (Wave Runup)
To process wave runup data:

```matlab
% In MATLAB
config = jsondecode(fileread('livox_config.json'));
L2_pipeline
```

### 4. Visualize Results
To plot a cross-shore profile from the L1 structure:

```matlab
% Assuming 'L1' struct exists in workspace
[x1d, Z3D] = Get3_1Dprofiles(L1(i).X, L1(i).Y, L1(i).Zmode);
plot(x1d, Z3D(5,:)); % Plot central transect
```

---

## Data Coordinate Systems
*   **Input:** Livox LiDAR native coordinates (Sensor Frame XYZ).
*   **Intermediate:** Homogeneous coordinates for matrix transformation.
*   **Output:** UTM Coordinates (NAD83, Zone 11N) with NAVD88 elevation.

## Requirements
*   MATLAB (with Image Processing Toolbox, Statistics Toolbox)
*   `lasFileReader` (for reading `.laz` files)
*   Point Cloud Processing Toolbox
