# Laserdanger

**Laserdanger** is a LiDAR data processing pipeline for analyzing point cloud data from Livox Avia sensors. The system processes raw `.laz` files to generate beach surface Digital Elevation Models (DEMs) and resolve high-frequency wave runup dynamics.

Developed by **Ashton Domi** for his PhD thesis at **Scripps Institution of Oceanography** (Coastal Processes Group).

> **Note**: Active development is in `python/` only. The MATLAB code is legacy and no longer maintained.

---

## Processing Levels

| Level | Purpose | Output | Temporal Resolution |
|-------|---------|--------|---------------------|
| **L1** | Beach surface morphology | 2D gridded DEMs (z_mean, z_mode, etc.) | Daily aggregates |
| **L2** | Wave runup dynamics | Z(x,t) timestacks along transects | ~2 Hz (0.5s bins) |

---

## Quick Start (Python)

### 1. Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Create a Configuration File

Create a JSON config file for your site. Copy an existing template and modify:

```bash
cp configs/do_livox_config_20260112.json configs/mysite_livox_config_20260122.json
```

Edit the config with your site-specific settings:

```json
{
  "dataFolder": "/path/to/your/laz/files",
  "processFolder": "/path/to/output",
  "transformMatrix": [
    [0.999, 0.012, 0.003, 476543.21],
    [-0.012, 0.999, 0.001, 3628901.45],
    [-0.003, -0.001, 1.000, 42.15],
    [0, 0, 0, 1]
  ],
  "LidarBoundary": [
    [476500, 3628850],
    [476600, 3628850],
    [476600, 3628950],
    [476500, 3628950]
  ]
}
```

**Config fields explained:**

| Field | Description |
|-------|-------------|
| `dataFolder` | Directory containing raw `.laz` files (named with timestamps) |
| `processFolder` | Where to write intermediate outputs |
| `transformMatrix` | 4×4 homogeneous matrix: LiDAR sensor coords → UTM (NAD83) |
| `LidarBoundary` | Polygon vertices (UTM) defining the valid beach area |

### 3. Run L1 Processing (Beach Surfaces)

```bash
cd python

# Process all days found in dataFolder
python scripts/run_daily_l1.py --config configs/mysite_livox_config_20260122.json

# Process specific date range
python scripts/run_daily_l1.py --config configs/mysite_livox_config_20260122.json \
    --start 2026-01-15 --end 2026-01-20

# Preview without processing (dry run)
python scripts/run_daily_l1.py --config configs/mysite_livox_config_20260122.json --dry-run
```

**Output**: `python/data/level1/L1_YYYYMMDD.nc` (one NetCDF per day)

### 4. Run L2 Processing (Wave Runup)

```bash
# Process all days
python scripts/run_daily_l2.py --config configs/mysite_livox_config_20260122.json

# Higher temporal resolution (4 Hz)
python scripts/run_daily_l2.py --config configs/mysite_livox_config_20260122.json --time-bin 0.25
```

**Output**: `python/data/level2/L2_YYYYMMDD.nc` (one NetCDF per day)

---

## CLI Reference

### L1 Processing: `run_daily_l1.py`

```
python scripts/run_daily_l1.py --config PATH [OPTIONS]

Required:
  --config PATH       Path to JSON config file

Options:
  --output-dir DIR    Output directory (default: python/data/level1)
  --start DATE        Start date as YYYY-MM-DD (default: auto-detect from files)
  --end DATE          End date as YYYY-MM-DD (default: auto-detect from files)
  --bin-size FLOAT    Spatial bin size in meters (default: 0.1)
  --resume            Resume from checkpoint if interrupted
  --dry-run           Show what would be processed without running
  --verbose, -v       Enable debug logging
  --quiet, -q         Suppress non-error output
  --no-progress       Disable progress bars
```

### L2 Processing: `run_daily_l2.py`

```
python scripts/run_daily_l2.py --config PATH [OPTIONS]

Required:
  --config PATH       Path to JSON config file

Options:
  --output-dir DIR    Output directory (default: python/data/level2)
  --start DATE        Start date as YYYY-MM-DD
  --end DATE          End date as YYYY-MM-DD
  --time-bin FLOAT    Temporal bin size in seconds (default: 0.5 = 2Hz)
  --x-bin FLOAT       Spatial bin size along transect in meters (default: 0.1)
  --multi-transect    Extract multiple alongshore transects
  --outlier-detection Enable outlier detection (off by default)
  --resume            Resume from checkpoint if interrupted
  --dry-run           Show what would be processed without running
  --verbose, -v       Enable debug logging
```

---

## Output Formats

### L1 NetCDF Variables

| Variable | Description |
|----------|-------------|
| `z_mean` | Mean elevation per bin |
| `z_mode` | Modal elevation (most common) |
| `z_min` | Minimum elevation |
| `z_max` | Maximum elevation |
| `z_std` | Standard deviation |
| `count` | Point count per bin |

### L2 NetCDF Variables

| Variable | Description |
|----------|-------------|
| `Z` | Elevation timestack Z(x, t) |
| `I` | Intensity timestack I(x, t) |
| `x` | Cross-shore distance coordinate |
| `time` | Time coordinate |

---

## Example: Working with Output Data

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load L1 daily surface
ds = xr.open_dataset("python/data/level1/L1_20260115.nc")
ds.z_mode.plot()
plt.title("Beach Surface (Mode Elevation)")
plt.show()

# Load L2 timestack
ds2 = xr.open_dataset("python/data/level2/L2_20260115.nc")
ds2.Z.plot(x="time", y="x")
plt.title("Elevation Timestack")
plt.show()
```

---

## Transformation Matrix

The `transformMatrix` converts from LiDAR sensor coordinates to UTM. This is a 4×4 homogeneous transformation matrix:

```
| R11  R12  R13  Tx |     R = rotation matrix (3×3)
| R21  R22  R23  Ty |     T = translation vector (Tx, Ty, Tz)
| R31  R32  R33  Tz |
|  0    0    0   1  |
```

To determine your transformation matrix:
1. Collect ground control points (GCPs) with known UTM coordinates
2. Identify corresponding points in the LiDAR data
3. Solve for the rigid transformation (rotation + translation)

---

## Directory Structure

```
python/
├── code/           # Source modules (phase1-4, profiles, runup, utils)
├── configs/        # Site/date-specific JSON configs
├── scripts/        # CLI entry points (run_daily_l1.py, run_daily_l2.py)
├── tests/          # pytest suite (186+ tests)
├── data/           # Output data
│   ├── level1/     # L1 NetCDF outputs
│   └── level2/     # L2 NetCDF outputs
└── docs/           # Documentation
```

---

## Requirements

**Python 3.10+** with:

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
laspy>=2.0.0
lazrs>=0.5.0
xarray>=0.19.0
netCDF4>=1.5.0
matplotlib>=3.4.0
tqdm>=4.62.0
```

Install all dependencies:

```bash
cd python
pip install -r requirements.txt
```

---

## Running Tests

```bash
cd python
pytest tests/ -v

# With coverage
pytest tests/ --cov=code --cov-report=html
```

---

## Troubleshooting

### No LAZ files found
- Verify `dataFolder` path in your config is correct
- Check that LAZ files have timestamps in their filenames
- Use `--dry-run` to see what files would be discovered

### Transform matrix issues
- Ensure the matrix is 4×4 with bottom row `[0, 0, 0, 1]`
- Verify UTM zone matches your GCPs
- Check that output coordinates fall within expected bounds

### Memory errors on large datasets
- Process smaller date ranges with `--start` and `--end`
- Use `--resume` to continue if processing is interrupted

---

## Legacy MATLAB Code

The original MATLAB implementation is in the repository root (`matlab/`). It is no longer actively maintained but remains available for reference.

Key MATLAB entry points:
- `L1_pipeline.m` - Single-day L1 processing
- `L2_pipeline.m` - Single-day L2 processing
- `lidar_config.m` - Config file generator

---

## License

This project was developed for academic research at Scripps Institution of Oceanography.

## Contact

For questions about this codebase, contact Ashton Domi at the Scripps Institution of Oceanography.
