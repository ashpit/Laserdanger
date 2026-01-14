# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Laserdanger is a MATLAB-based lidar processing pipeline for analyzing point cloud data from a Livox Avia lidar sensor. The codebase processes .laz format point cloud scans to generate time-resolved beach surface Digital Elevation Models (DEMs) and detect wave runup on beaches. Developed by Ashton Domi for his PhD thesis at Scripps Institution of Oceanography's Coastal Processes Group.

## Processing Pipeline Architecture

The repository implements a two-level processing pipeline:

### Level 1 (L1): Hourly/Daily Beach Surface Generation
- **Entry point**: `L1_pipeline.m`
- **Purpose**: Produces 30-minute average DEM-like surfaces from point clouds
- **Batch processing**: `L1_batchpipeline.m`
- **Output**: Structured data organized by day containing vectorized surfaces (min, max, mode, mean)

**L1 Processing Steps** (documented in L1_pipeline.m lines 7-64):
1. Load configuration from `livox_config.json` (paths, transformation matrix, boundary polygon)
2. Identify and filter valid lidar files within date range
3. Read point clouds and apply homogeneous transformation matrix to convert from lidar coordinates to UTM
4. Filter by intensity (<100) and spatial boundary polygon
5. Bin/rasterize using `accumpts.m` with configurable resolution (default 0.10m)
6. Apply signal-to-noise ratio filtering (SNR = mean/standard_error, threshold 100, min count 10)
7. Residual noise filtering using `ResidualKernelFilter.m` with Delaunay triangulation and planar fitting
8. Two-stage filtering: 10m window with 0.2m threshold, then 3m window with 0.1m threshold
9. Store results in L1 struct with fields: Dates, X, Y, Zmean, Zmax, Zmin, Zstdv, Zmode
10. Generate 1D cross-shore profiles using `Get3_1Dprofiles.m` and export as JSON

### Level 2 (L2): Wave-Resolving Analysis
- **Entry point**: `L2_pipeline.m` (testing version: `L2_pipeline_testing.m`)
- **Purpose**: Creates 2Hz time-resolved Z(x,t) matrices and detects wave runup edges
- **Key difference**: Processes raw point clouds with temporal accumulation via `accumpts_L2.m`

**L2 Processing Steps** (documented in L2_pipeline.m lines 7-64):
1. Similar setup and filtering as L1
2. Temporal binning at ~2Hz frequency (configurable via `ff` variable)
3. Outputs Z(x,t), X(x,t), I(x,t) arrays along centered transect
4. Runup detection via `get_runupStats_L2.m`

## Key Functions and Their Roles

### Core Processing Functions

**`accumpts.m`** - Point cloud binning and statistics
- Bins 3D points (X,Y,Z) to configurable grid resolution
- Applies 50th percentile filtering within bins
- Computes statistics: mean, max, min, mode, std, count per bin
- SNR-based filtering removes unreliable bins

**`accumpts_L2.m`** - Temporal point cloud accumulation (L2 specific)
- Extends accumpts to include time dimension
- Returns time-indexed arrays for wave-resolving analysis

**`ResidualKernelFilter.m`** - Noise removal via planar fitting
- Uses Delaunay triangulation to partition point cloud
- Fits plane to each triangle, removes points >threshold distance above plane
- Iterative application with decreasing window sizes improves results

### Profile Extraction

**`Get3_1Dprofiles.m`** - Extract cross-shore transects
- Projects 3D point cloud onto multiple shore-normal transects (10 transects at -8m to +10m alongshore spacing)
- Uses quadratic fitting to remove residual outliers (threshold: 0.4m)
- Interpolates gaps up to 4m
- Returns Z3D matrix (n_transects × n_crossshore_positions)

**`Get1D_profiles_swash_accum.m`** - Similar to Get3_1Dprofiles but optimized for swash zone

### Runup Analysis

**`get_runupStats_L2.m`** - Comprehensive runup statistics from Z(x,t) matrices
- Detects runup line by intersecting water level with threshold (default 0.1m)
- Creates reference beach surface using moving minimum filter (IGlength=100s)
- Adaptive search window tracks runup position across time
- Computes spectral analysis: IG (0.004-0.04 Hz) and incident (0.04-0.25 Hz) bands
- Estimates foreshore slope (beta) from stable regions
- Returns: Spec (frequency spectra), Bulk (Sig, Sinc, eta, beta), Tseries (Xrunup, Zrunup)

### Utility Functions

- `fitPlane.m` - SVD-based plane fitting
- `inpaint_nans.m` - NaN interpolation for gap filling
- `gapsize.m` - Detects gap sizes in 1D arrays
- `roundToHalfHour.m` - Time binning utility
- `split_L1_days.m` - Splits L1 data by day
- `detect_outliers_conv2D.m` - 2D outlier detection
- `Get_intensity_contours.m` - Extract intensity-based contours for water edge detection

## Configuration System

**`lidar_config.m`** - Configuration generator
- Creates JSON configuration files with different transformation matrices for different deployment periods
- Key configuration fields:
  - `dataFolder`: Path to raw .laz files
  - `processFolder`: Output directory for daily processed data
  - `processHFolder`: Output directory for hourly processed data (L1)
  - `plotFolder`: Output directory for visualization JSON
  - `transformMatrix`: 4×4 homogeneous transformation matrix (lidar → UTM coordinates)
  - `LidarBoundary`: Polygon vertices defining valid spatial extent (UTM coordinates)

**Transformation Matrices**: Multiple matrices stored for different time periods (June, July, August) due to equipment recalibration or repositioning. Selection depends on data timestamp.

## Data Coordinate Systems

- **Input**: Livox lidar native coordinates (point cloud XYZ in sensor frame)
- **Intermediate**: Homogeneous coordinates (4×1 vectors) for matrix transformation
- **Output**: UTM coordinates (NAD83, Zone 11N) with NAVD88 elevation datum
- **Boundary filtering**: Uses polygon intersection to limit processing to beach region

## File Naming Conventions

- Raw point cloud files: `do-lidar_<POSIX_timestamp>.laz`
- Configuration files: `livox_config.json`, `livox_config2.json`, `livox_config_L2.json`
- Output structures: Saved as .mat files (L1, L2, DO structs)
- Visualization data: `lidar_plot_data.json` with profiles array

## Running the Code

### Environment Requirements
- MATLAB (version with Image Processing Toolbox, Statistics Toolbox)
- `lasFileReader` function for .laz file reading
- Point Cloud Processing Toolbox

### Processing L1 Data (Beach Surveys)

```matlab
% 1. Edit lidar_config.m to set correct paths and transformation matrix
% 2. Run section to generate livox_config.json
run lidar_config.m

% 3. Run L1 pipeline (processes new data since last run)
L1_pipeline

% 4. For batch processing multiple days
L1_batchpipeline
```

### Processing L2 Data (Wave Runup)

```matlab
% 1. Load L2 configuration
config = jsondecode(fileread('livox_config.json'));

% 2. Run L2 pipeline
L2_pipeline

% 3. Analyze specific file (modify 'n' variable in script)
% Set n = <file_index> in L2_pipeline.m and run relevant sections
```

### Extracting 1D Profiles

```matlab
% From L1 structure
[x1d, Z3D] = Get3_1Dprofiles(L1(i).X, L1(i).Y, L1(i).Zmode);

% Z3D contains multiple transects (rows)
% Plot central transect
plot(x1d, Z3D(5,:))
```

### Runup Analysis

```matlab
% Assuming you have Z_xt matrix from L2 processing
[Spec, Info, Bulk, Tseries] = get_runupStats_L2(Z_xt, I_xt, x1d, time_vec, ...
    'threshold', 0.1, 'windowlength', 5, 'plot', true);

% Access results
Sig = Bulk.swashparams(1);    % IG significant wave height
Sinc = Bulk.swashparams(2);   % Incident significant wave height
eta = Bulk.swashparams(3);    % Mean water level
beta = Bulk.beta;              % Beach slope
```

## Important Implementation Details

### Accumarray Binning Strategy (accumpts.m)
- Uses MATLAB's `accumarray` with cell function to collect all Z values per bin
- 50th percentile filter applied before computing statistics (focuses on minimum surface)
- SNR = mean / (std/sqrt(count)) filters bins with high uncertainty
- Invalid bins (SNR < 100 or count ≤ 10) are set to NaN and filtered out

### Residual Kernel Filtering (ResidualKernelFilter.m)
- Creates Delaunay triangulation of 2D point cloud (X,Y plane)
- For each triangle, fits plane using SVD decomposition
- Removes points with residual > threshold above the plane
- Two-pass filtering with decreasing window/threshold improves ground extraction

### Runup Detection Algorithm (get_runupStats_L2.m)
- Moving minimum filter (100s window) creates "dry beach" reference surface
- Water level = instantaneous elevation - dry beach surface
- Runup line = intersection of water level curve with threshold (0.1m)
- Adaptive search window (±0.5m around previous position) improves temporal continuity
- Median filtering (1s window) and gap interpolation clean up detection

### Coordinate Transformation
Point transformation: `P_utm = T * [x_lidar; y_lidar; z_lidar; 1]`
- Transformation matrix T combines rotation and translation in homogeneous coordinates
- Different matrices used for different deployment periods (stored in lidar_config.m)

## Typical Modifications

### Changing spatial resolution
Modify binsize parameter in L1_pipeline.m (line 135):
```matlab
binsize = 0.10; % change from default 0.10m to desired resolution
```

### Adjusting filtering thresholds
In L1_pipeline.m:
```matlab
windowSize = 10; thresh = 0.2;  % First pass (line 138)
windowSize = 3; thresh = 0.1;   % Second pass (line 149)
```

### Modifying runup detection sensitivity
In get_runupStats_L2.m:
```matlab
options.threshold = 0.1;        % Water depth threshold (line 18)
options.IGlength = 100;         % Moving min window (line 22)
```

### Adding new transect positions
In Get3_1Dprofiles.m (line 34):
```matlab
alongshore_spacings = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10]; % modify as needed
```
