# Data Formats Documentation

This document describes the input and output data formats used by the Python lidar pipeline.

## Input Formats

### Configuration File (livox_config.json)

The configuration file is a JSON document that specifies paths, transformation matrices, and boundary polygons.

```json
{
  "dataFolder": "/Volumes/camera/DO-lidar/data-lvx",
  "processFolder": "/path/to/processed/daily",
  "processHFolder": "/path/to/processed/hourly",
  "plotFolder": "/path/to/plots",
  "transformMatrix": [
    [0.9998, -0.0175, 0.0087, 500123.456],
    [0.0174, 0.9998, 0.0052, 3712345.678],
    [-0.0088, -0.0050, 0.9999, 42.123],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "LidarBoundary": [
    [500100.0, 3712300.0],
    [500150.0, 3712300.0],
    [500150.0, 3712400.0],
    [500100.0, 3712400.0]
  ]
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `dataFolder` | string | Path to directory containing raw .laz files |
| `processFolder` | string | Output directory for daily processed data |
| `processHFolder` | string | Output directory for hourly processed data |
| `plotFolder` | string | Output directory for visualization files |
| `transformMatrix` | 4x4 array | Homogeneous transformation matrix (lidar → UTM) |
| `LidarBoundary` | array of [x, y] | Polygon vertices defining valid processing area |

#### Transformation Matrix

The 4x4 transformation matrix converts from lidar sensor coordinates to UTM coordinates:

```
| r11  r12  r13  tx |   | x_lidar |   | x_utm |
| r21  r22  r23  ty | × | y_lidar | = | y_utm |
| r31  r32  r33  tz |   | z_lidar |   | z_utm |
| 0    0    0    1  |   | 1       |   | 1     |
```

- Top-left 3x3: Rotation matrix
- Right column: Translation vector (tx, ty, tz)
- Bottom row: Always [0, 0, 0, 1]

### LAZ Files

Input point cloud files follow the naming convention:

```
do-lidar_<POSIX_TIMESTAMP>.laz
```

Example: `do-lidar_1714742400.laz`

The POSIX timestamp is seconds since Unix epoch (1970-01-01 00:00:00 UTC).

#### Required Fields in LAZ

| Field | Description |
|-------|-------------|
| X, Y, Z | Point coordinates (lidar frame) |
| intensity | Return intensity (0-255 typical) |
| gps_time | GPS timestamp for each point |

---

## Output Formats

### L1 NetCDF Output

L1 outputs are stored in NetCDF4 format (.nc) compatible with xarray.

#### Dimensions

| Dimension | Description |
|-----------|-------------|
| `x_edge` | X coordinate bin edges (N+1 values for N bins) |
| `y_edge` | Y coordinate bin edges (M+1 values for M bins) |
| `time` | Timestamp for each processed file |

#### Variables

| Variable | Dimensions | Type | Description |
|----------|------------|------|-------------|
| `z_mean` | (time, y, x) | float32 | Mean elevation per bin |
| `z_max` | (time, y, x) | float32 | Maximum elevation per bin |
| `z_min` | (time, y, x) | float32 | Minimum elevation per bin |
| `z_mode` | (time, y, x) | float32 | Mode elevation per bin |
| `z_std` | (time, y, x) | float32 | Standard deviation per bin |
| `count` | (time, y, x) | int32 | Point count per bin |

#### Attributes

| Attribute | Description |
|-----------|-------------|
| `bin_size` | Spatial bin size in meters |
| `mode_bin` | Mode histogram bin size |
| `snr_threshold` | SNR threshold used |
| `min_count` | Minimum count threshold |
| `crs` | Coordinate reference system (e.g., "EPSG:26911") |

#### Example: Reading L1 Data

```python
import xarray as xr

ds = xr.open_dataset("l1_output.nc")

# Get mean elevation grid for first timestamp
z_mean = ds["z_mean"].isel(time=0).values

# Get coordinates
x = ds["x_edge"].values[:-1] + ds.attrs.get("bin_size", 0.1) / 2
y = ds["y_edge"].values[:-1] + ds.attrs.get("bin_size", 0.1) / 2
```

### L2 NetCDF Output

L2 outputs contain time-resolved data along cross-shore transects.

#### Dimensions

| Dimension | Description |
|-----------|-------------|
| `x` | Cross-shore distance (m from origin) |
| `time` | Time in seconds from start |
| `alongshore` | Alongshore offset (optional, for multi-transect) |

#### Variables

| Variable | Dimensions | Type | Description |
|----------|------------|------|-------------|
| `Z` | (time, x) or (alongshore, time, x) | float32 | Elevation time stack |
| `I` | (time, x) or (alongshore, time, x) | float32 | Intensity time stack |
| `outlier_mask` | (time, x) | bool | True where outliers detected |
| `intensity_contours` | (time, threshold) | float32 | Intensity contour positions |

#### Attributes

| Attribute | Description |
|-----------|-------------|
| `dt` | Time step in seconds |
| `dx` | Cross-shore spacing in meters |
| `origin_x` | UTM X of transect origin |
| `origin_y` | UTM Y of transect origin |
| `azimuth` | Transect azimuth in degrees |
| `time_bin_size` | Temporal bin size in seconds |

#### Example: Reading L2 Data

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset("l2_output.nc")

# Get Z(x,t) matrix
Z_xt = ds["Z"].values

# Get coordinates
x1d = ds["x"].values
time_vec = ds["time"].values

# Sampling rate
dt = float(ds.attrs.get("dt", 0.5))
print(f"Sampling rate: {1/dt:.1f} Hz")
```

### Validation Report JSON

Validation reports from comparing Python and MATLAB outputs.

```json
{
  "name": "L1 Comparison",
  "python_source": "/path/to/python_output.nc",
  "matlab_source": "/path/to/matlab_output.mat",
  "passed": true,
  "overall_rmse": 0.0152,
  "overall_correlation": 0.9987,
  "field_comparisons": [
    {
      "field_name": "z_mean",
      "python_shape": [10, 100, 100],
      "matlab_shape": [10, 100, 100],
      "rmse": 0.0123,
      "correlation": 0.9992,
      "bias": 0.0015,
      "max_abs_diff": 0.0456,
      "n_valid_points": 85000,
      "n_total_points": 100000,
      "valid_fraction": 0.85,
      "shapes_match": true,
      "notes": ""
    }
  ],
  "metadata": {
    "python_version": "3.11.0",
    "validation_date": "2025-01-15T10:30:00"
  },
  "warnings": [],
  "errors": []
}
```

### Profile JSON (lidar_plot_data.json)

Web visualization format for beach profiles.

```json
{
  "profiles": [
    {
      "timestamp": "2025-05-03T12:00:00",
      "x1d": [0.0, 0.1, 0.2, ...],
      "z_mode": [2.5, 2.4, 2.3, ...],
      "z_mean": [2.5, 2.4, 2.3, ...],
      "z_min": [2.4, 2.3, 2.2, ...],
      "z_max": [2.6, 2.5, 2.4, ...]
    }
  ],
  "metadata": {
    "origin_x": 500123.456,
    "origin_y": 3712345.678,
    "azimuth": 270.0,
    "bin_size": 0.1
  }
}
```

### Checkpoint JSON

Checkpoint files for resumable batch processing.

```json
{
  "config_path": "/path/to/livox_config.json",
  "output_dir": "/path/to/output",
  "start_date": "2025-05-01",
  "end_date": "2025-05-31",
  "completed_dates": [
    "2025-05-01",
    "2025-05-02",
    "2025-05-03"
  ],
  "failed_dates": [],
  "kwargs": {
    "bin_size": 0.1,
    "mode_bin": 0.05
  },
  "timestamp": "2025-05-03T15:30:00"
}
```

---

## Coordinate Systems

### Input Coordinates

- **Lidar Frame**: Native sensor coordinates (X forward, Y left, Z up)
- **Units**: Meters

### Output Coordinates

- **Horizontal**: UTM NAD83 Zone 11N (EPSG:26911)
- **Vertical**: NAVD88 (meters above geoid)

### Cross-shore Distance

For L2 profiles, cross-shore distance is measured from the transect origin:
- Positive values: Seaward (offshore)
- Negative values: Landward (onshore)
- Zero: Transect origin

---

## Units

| Quantity | Unit |
|----------|------|
| Horizontal coordinates | Meters (UTM) |
| Elevation | Meters (NAVD88) |
| Time | Seconds (POSIX or relative) |
| Intensity | Dimensionless (0-255) |
| Frequency | Hertz (Hz) |
| Spectral density | m²/Hz |
| Slope | Dimensionless (rise/run) |

---

## Data Quality Flags

### SNR Filter
- Points with SNR < threshold are marked as NaN
- Default threshold: 100

### Outlier Detection
- `outlier_mask = True` indicates detected outliers
- Outliers are typically water surface spikes or multi-path returns

### Gap Handling
- Gaps < `max_gap` meters are interpolated
- Larger gaps remain as NaN
