#!/usr/bin/env python3
"""
Create animated GIFs from L1 NetCDF files showing beach surface grids and
cross-shore profiles with slope calculation.

Two-panel figure:
  - Left: 2D DEM/grid view with colorbar
  - Right: Cross-shore profile with foreshore slope fit

Slope Calculation:
  By default, slope is calculated between fixed tidal datum elevations:
    - MSL (Mean Sea Level): 0.744m NAVD88
    - MHW (Mean High Water): 1.34m NAVD88
  This produces the foreshore slope between these two elevations, which is
  more oceanographically meaningful than distance-based fitting.

  Use --no-tidal-slope to revert to distance-based fitting (first X meters
  from seaward edge).

Usage:
    # Process all L1 files from config (recommended)
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json

    # Process single file
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --input L1_20260112.nc

Options:
    --config PATH       Path to config file (required)
    --input FILE        Process single file (filename only, looked up in processFolder/level1/)
    --output PATH       Output GIF path (default: plotFolder/level1/)
    --variable VAR      Variable to plot (default: elevation_mode)
    --cmap COLORMAP     Matplotlib colormap (default: terrain)
    --fps FPS           Frames per second (default: 2)
    --dpi DPI           Resolution (default: 150)
    --vmin FLOAT        Minimum value for colorbar (default: auto)
    --vmax FLOAT        Maximum value for colorbar (default: auto)
    --y-index INT       Y index for profile extraction (default: middle)
    --z-msl FLOAT       Mean Sea Level in NAVD88 (default: 0.744m)
    --z-mhw FLOAT       Mean High Water in NAVD88 (default: 1.34m)
    --no-tidal-slope    Use distance-based slope fit instead of tidal datums
    --x-max FLOAT       Max cross-shore distance for slope fit (default: 20m, used with --no-tidal-slope)
    --no-colorbar       Disable colorbar
    --no-profile        Disable profile panel (single panel DEM only)
    --save-slopes       Save slopes to JSON file (in processFolder/slopes/)

Examples:
    # Process all L1 files using MSL→MHW slope (default)
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json

    # Use custom tidal datums
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --z-msl 0.5 --z-mhw 1.2

    # Use distance-based slope calculation instead
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --no-tidal-slope --x-max 25

    # Process specific file
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --input L1_20260115.nc
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import xarray as xr

# Add code directory to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

import json
import profiles


def print_metadata(ds: xr.Dataset, nc_path: Path) -> dict:
    """
    Print comprehensive metadata from L1 NetCDF file.

    Returns dict with extracted metadata for use in plotting.
    """
    print("=" * 70)
    print(f"L1 NetCDF Metadata: {nc_path.name}")
    print("=" * 70)

    # File info
    print(f"\nFile: {nc_path}")
    print(f"Size: {nc_path.stat().st_size / (1024*1024):.2f} MB")

    # Dimensions
    print(f"\n--- Dimensions ---")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")

    # Coordinates
    print(f"\n--- Coordinates ---")
    if 'x' in ds.coords:
        x = ds.coords['x'].values
        print(f"  x: {len(x)} values, range [{x.min():.2f}, {x.max():.2f}] m")
        dx = np.median(np.diff(x)) if len(x) > 1 else 0
        print(f"     resolution: {dx:.3f} m")

    if 'y' in ds.coords:
        y = ds.coords['y'].values
        print(f"  y: {len(y)} values, range [{y.min():.2f}, {y.max():.2f}] m")
        dy = np.median(np.diff(y)) if len(y) > 1 else 0
        print(f"     resolution: {dy:.3f} m")

    # Time information
    timestamps = []
    if 'time' in ds.coords:
        time = ds.coords['time'].values
        n_times = len(time)
        print(f"  time: {n_times} timesteps")

        if n_times > 0:
            # Convert numpy datetime64 to Python datetime
            time_start = np.datetime_as_string(time[0], unit='s')
            time_end = np.datetime_as_string(time[-1], unit='s')
            print(f"     start: {time_start}")
            print(f"     end:   {time_end}")

            if n_times > 1:
                # Calculate time step
                dt = (time[1] - time[0]) / np.timedelta64(1, 's')
                print(f"     step:  {dt:.1f} s")

            # Store timestamps for return
            timestamps = [np.datetime_as_string(t, unit='s') for t in time]

    # Variables
    print(f"\n--- Variables ---")
    data_vars = list(ds.data_vars)
    for var in sorted(data_vars):
        da = ds[var]
        dims = ", ".join(da.dims)

        # Calculate stats (ignoring NaN)
        values = da.values
        valid = ~np.isnan(values)
        n_valid = valid.sum()
        n_total = values.size
        pct_valid = 100 * n_valid / n_total if n_total > 0 else 0

        print(f"  {var}")
        print(f"     dims: ({dims})")
        print(f"     dtype: {da.dtype}")
        print(f"     valid: {n_valid:,} / {n_total:,} ({pct_valid:.1f}%)")

        if n_valid > 0:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            vmean = float(np.nanmean(values))
            vstd = float(np.nanstd(values))
            print(f"     range: [{vmin:.3f}, {vmax:.3f}]")
            print(f"     mean:  {vmean:.3f} ± {vstd:.3f}")

    # Global attributes
    if ds.attrs:
        print(f"\n--- Attributes ---")
        for key, val in ds.attrs.items():
            val_str = str(val)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            print(f"  {key}: {val_str}")

    # Grid coverage
    print(f"\n--- Grid Coverage ---")
    if 'x' in ds.coords and 'y' in ds.coords:
        x_range = float(x.max() - x.min())
        y_range = float(y.max() - y.min())
        area = x_range * y_range
        print(f"  X extent: {x_range:.1f} m")
        print(f"  Y extent: {y_range:.1f} m")
        print(f"  Area: {area:.1f} m² ({area/1e4:.2f} ha)")

    # Data quality summary
    print(f"\n--- Data Quality ---")
    if 'elevation_mode' in ds:
        elev = ds['elevation_mode'].values
    elif 'elevation' in ds:
        elev = ds['elevation'].values
    else:
        elev = None

    if elev is not None:
        valid_pct = 100 * np.sum(~np.isnan(elev)) / elev.size
        print(f"  Elevation coverage: {valid_pct:.1f}%")

    if 'count' in ds:
        count = ds['count'].values
        total_points = np.nansum(count)
        mean_density = np.nanmean(count[count > 0]) if np.any(count > 0) else 0
        print(f"  Total points: {int(total_points):,}")
        print(f"  Mean point density: {mean_density:.1f} pts/bin")

    if 'snr' in ds:
        snr = ds['snr'].values
        mean_snr = np.nanmean(snr[~np.isnan(snr)]) if np.any(~np.isnan(snr)) else 0
        print(f"  Mean SNR: {mean_snr:.2f}")

    print("=" * 70)

    return {
        'timestamps': timestamps,
        'n_times': len(timestamps),
        'filename': nc_path.name,
    }


def compute_frame_stats(da: xr.DataArray, time_idx: int) -> dict:
    """Compute statistics for a single frame."""
    if 'time' in da.dims:
        data = da.isel(time=time_idx).values
    else:
        data = da.values

    valid = ~np.isnan(data)
    n_valid = valid.sum()

    if n_valid == 0:
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'coverage': 0.0,
        }

    return {
        'min': float(np.nanmin(data)),
        'max': float(np.nanmax(data)),
        'mean': float(np.nanmean(data)),
        'std': float(np.nanstd(data)),
        'coverage': 100 * n_valid / data.size,
    }


def calculate_slope(
    x: np.ndarray,
    z: np.ndarray,
    x_max_relative: float = 20.0,
    z_min_threshold: Optional[float] = None,
    z_msl: Optional[float] = None,
    z_mhw: Optional[float] = None,
    use_robust: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate foreshore slope using linear regression.

    Two modes of operation:
    1. Elevation-based (preferred): If z_msl and z_mhw are provided, fits slope
       between these two tidal datum elevations (MSL to MHW).
    2. Distance-based (fallback): Uses x_max_relative from seaward edge.

    Parameters
    ----------
    x : array
        Cross-shore positions (m), can be UTM coordinates
    z : array
        Elevation values (m) in NAVD88
    x_max_relative : float
        Maximum cross-shore distance from the seaward edge to include in fit (m).
        Only used if z_msl/z_mhw are not provided.
    z_min_threshold : float, optional
        Minimum elevation threshold - exclude points below this (rejects water noise).
        If None, uses 10th percentile of elevations as threshold.
        Only used if z_msl/z_mhw are not provided.
    z_msl : float, optional
        Mean Sea Level elevation in NAVD88 (m). Default: None.
        If provided along with z_mhw, slope is calculated between these elevations.
    z_mhw : float, optional
        Mean High Water elevation in NAVD88 (m). Default: None.
        If provided along with z_msl, slope is calculated between these elevations.
    use_robust : bool
        If True, use Theil-Sen robust regression (median-based, outlier resistant).
        If False, use ordinary least squares.

    Returns
    -------
    slope : float
        Beach slope (dz/dx, positive = landward-sloping up)
    x_fit : array
        X positions used for fit (for plotting fit line)
    z_fit : array
        Fitted Z values (for plotting fit line)
    """
    from scipy import stats

    # Remove NaN values
    valid = ~np.isnan(z)
    x_valid = x[valid]
    z_valid = z[valid]

    if len(z_valid) < 3:
        return np.nan, np.array([]), np.array([])

    # Elevation-based filtering (MSL to MHW)
    if z_msl is not None and z_mhw is not None:
        # Filter points between MSL and MHW elevations
        in_range = (z_valid >= z_msl) & (z_valid <= z_mhw)
        x_fit_pts = x_valid[in_range]
        z_fit_pts = z_valid[in_range]

        if len(x_fit_pts) < 2:
            # Not enough points in elevation range
            return np.nan, np.array([]), np.array([])
    else:
        # Distance-based filtering (original behavior)
        # Convert to relative cross-shore distance from seaward edge
        x_min = x_valid.min()
        x_relative = x_valid - x_min

        # Filter by cross-shore distance (swash zone constraint)
        in_swash = x_relative <= x_max_relative

        # Determine elevation threshold for rejecting water/noise at seaward edge
        if z_min_threshold is None and np.any(in_swash):
            # Use 10th percentile as automatic threshold to exclude low outliers
            z_min_threshold = np.percentile(z_valid[in_swash], 10)

        # Apply both spatial and elevation filters
        if z_min_threshold is not None:
            in_range = in_swash & (z_valid >= z_min_threshold)
        else:
            in_range = in_swash

        x_fit_pts = x_valid[in_range]
        z_fit_pts = z_valid[in_range]

        if len(x_fit_pts) < 3:
            # Fall back to just swash zone filter
            x_fit_pts = x_valid[in_swash]
            z_fit_pts = z_valid[in_swash]

    if len(x_fit_pts) < 2:
        return np.nan, np.array([]), np.array([])

    # Sort by x for proper fitting
    sort_idx = np.argsort(x_fit_pts)
    x_fit_pts = x_fit_pts[sort_idx]
    z_fit_pts = z_fit_pts[sort_idx]

    # Regression
    if use_robust and len(x_fit_pts) >= 10:
        # Theil-Sen regression: median-based, resistant to up to 29% outliers
        result = stats.theilslopes(z_fit_pts, x_fit_pts)
        slope = result.slope
        intercept = result.intercept
    else:
        # Ordinary least squares
        coeffs = np.polyfit(x_fit_pts, z_fit_pts, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

    # Generate fit line for plotting (only within the fitted region)
    x_line = np.array([x_fit_pts.min(), x_fit_pts.max()])
    z_line = slope * x_line + intercept

    return slope, x_line, z_line


def extract_profile_and_slope(
    ds: xr.Dataset,
    time_idx: int,
    y_index: int,
    x_max_relative: float = 20.0,
    elevation_var: str = 'elevation',
    z_msl: Optional[float] = None,
    z_mhw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Extract cross-shore profile and calculate slope for a single timestep.

    Returns
    -------
    x : array
        Cross-shore positions
    profile : array
        Elevation profile
    slope : float
        Foreshore slope
    x_fit : array
        X positions for fit line
    z_fit : array
        Z values for fit line
    """
    x = ds.x.values

    # Handle different variable names
    if elevation_var in ds:
        da = ds[elevation_var]
    elif 'elevation' in ds:
        da = ds['elevation']
    elif 'elevation_mode' in ds:
        da = ds['elevation_mode']
    else:
        raise ValueError("No elevation variable found in dataset")

    if 'time' in da.dims:
        profile = da.isel(time=time_idx, y=y_index).values
    else:
        profile = da.isel(y=y_index).values

    slope, x_fit, z_fit = calculate_slope(
        x, profile, x_max_relative=x_max_relative,
        z_msl=z_msl, z_mhw=z_mhw
    )

    return x, profile, slope, x_fit, z_fit


def extract_transect_profile(
    ds: xr.Dataset,
    time_idx: int,
    transect_config: 'profiles.TransectConfig',
    x_max_relative: float = 20.0,
    elevation_var: str = 'elevation',
    flip_profile: Optional[bool] = None,
    z_msl: Optional[float] = None,
    z_mhw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Extract profile along a transect line from 2D gridded data.

    Interpolates the grid data along the transect centerline.
    The profile is oriented so x=0 is at the SEAWARD (low elevation) end,
    which is required for proper foreshore slope calculation.

    Parameters
    ----------
    ds : xr.Dataset
        L1 dataset with x, y coordinates and elevation data
    time_idx : int
        Time index to extract
    transect_config : TransectConfig
        Transect configuration
    x_max_relative : float
        Max cross-shore distance for slope fit (m). Only used if z_msl/z_mhw not provided.
    elevation_var : str
        Name of elevation variable
    flip_profile : bool, optional
        If True, flip profile so seaward is at x=0.
        If False, don't flip.
        If None (default), auto-detect from elevation data.
    z_msl : float, optional
        Mean Sea Level elevation in NAVD88 (m) for slope calculation.
    z_mhw : float, optional
        Mean High Water elevation in NAVD88 (m) for slope calculation.

    Returns
    -------
    x1d : array
        Cross-shore distance along transect (0 = seaward end)
    profile : array
        Elevation profile
    slope : float
        Foreshore slope
    x_fit : array
        X positions for fit line (in cross-shore distance)
    z_fit : array
        Z values for fit line
    transect_x_utm : array
        UTM X coordinates of transect (ordered seaward to landward)
    transect_y_utm : array
        UTM Y coordinates of transect (ordered seaward to landward)
    did_flip : bool
        Whether the profile was flipped (useful for caching this decision)
    """
    from scipy.interpolate import RegularGridInterpolator

    # Get grid coordinates
    x_grid = ds.x.values
    y_grid = ds.y.values

    # Handle different variable names
    if elevation_var in ds:
        da = ds[elevation_var]
    elif 'elevation' in ds:
        da = ds['elevation']
    elif 'elevation_mode' in ds:
        da = ds['elevation_mode']
    else:
        raise ValueError("No elevation variable found in dataset")

    # Get data for this timestep
    if 'time' in da.dims:
        data = da.isel(time=time_idx).values
    else:
        data = da.values

    # Create interpolator (handles NaN by returning NaN)
    # Data is (y, x) indexed
    interpolator = RegularGridInterpolator(
        (y_grid, x_grid), data,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # Generate points along transect
    transect_length = np.sqrt(
        (transect_config.x2 - transect_config.x1)**2 +
        (transect_config.y2 - transect_config.y1)**2
    )
    n_points = int(transect_length / transect_config.resolution) + 1
    x1d_raw = np.linspace(0, transect_length, n_points)

    # UTM coordinates along transect (from x1,y1 to x2,y2)
    t = x1d_raw / transect_length
    transect_x_utm_raw = transect_config.x1 + t * (transect_config.x2 - transect_config.x1)
    transect_y_utm_raw = transect_config.y1 + t * (transect_config.y2 - transect_config.y1)

    # Interpolate elevations along transect
    points = np.column_stack([transect_y_utm_raw, transect_x_utm_raw])
    profile_raw = interpolator(points)

    # Determine flip direction
    if flip_profile is None:
        # Auto-detect: compare mean elevation of first 10% vs last 10%
        n_check = max(1, n_points // 10)
        elev_start = np.nanmean(profile_raw[:n_check])
        elev_end = np.nanmean(profile_raw[-n_check:])
        # Flip if start has higher elevation than end
        # (we want x=0 at the seaward/low elevation end)
        flip_profile = elev_start > elev_end

    if flip_profile:
        # Flip everything so seaward end is at x=0
        x1d = x1d_raw  # Keep same distances
        profile = profile_raw[::-1]
        transect_x_utm = transect_x_utm_raw[::-1]
        transect_y_utm = transect_y_utm_raw[::-1]
    else:
        # Already correct orientation
        x1d = x1d_raw
        profile = profile_raw
        transect_x_utm = transect_x_utm_raw
        transect_y_utm = transect_y_utm_raw

    # Calculate slope (now x=0 is at seaward end)
    # If z_msl and z_mhw provided, uses elevation-based fitting
    # Otherwise uses distance-based fitting (first x_max_relative meters)
    slope, x_fit, z_fit = calculate_slope(
        x1d, profile, x_max_relative=x_max_relative,
        z_msl=z_msl, z_mhw=z_mhw
    )

    return x1d, profile, slope, x_fit, z_fit, transect_x_utm, transect_y_utm, flip_profile


def get_transect_config(config_path: Path, x_grid: np.ndarray, y_grid: np.ndarray) -> Optional['profiles.TransectConfig']:
    """
    Load transect config from JSON file or auto-compute from grid extent.

    Parameters
    ----------
    config_path : Path
        Path to config JSON file
    x_grid, y_grid : array
        Grid coordinates (for auto-computation if needed)

    Returns
    -------
    TransectConfig or None
    """
    # Try loading from config file
    try:
        with open(config_path, 'r') as f:
            raw_config = json.load(f)

        if 'transect' in raw_config:
            return profiles.transect_config_from_dict(raw_config['transect'])

        # Auto-compute from grid and transform matrix
        if 'transformMatrix' in raw_config:
            transform = np.array(raw_config['transformMatrix'])
            scanner_x, scanner_y = profiles.get_scanner_position(transform)

            # Create synthetic points from grid corners for transect computation
            X_corners = np.array([x_grid.min(), x_grid.max(), x_grid.min(), x_grid.max()])
            Y_corners = np.array([y_grid.min(), y_grid.min(), y_grid.max(), y_grid.max()])

            # Also add points along the edges
            X_edge = np.concatenate([
                np.linspace(x_grid.min(), x_grid.max(), 20),
                np.linspace(x_grid.min(), x_grid.max(), 20),
                np.full(20, x_grid.min()),
                np.full(20, x_grid.max()),
            ])
            Y_edge = np.concatenate([
                np.full(20, y_grid.min()),
                np.full(20, y_grid.max()),
                np.linspace(y_grid.min(), y_grid.max(), 20),
                np.linspace(y_grid.min(), y_grid.max(), 20),
            ])

            X_all = np.concatenate([X_corners, X_edge])
            Y_all = np.concatenate([Y_corners, Y_edge])

            config = profiles.compute_transect_from_swath(
                X_all, Y_all,
                scanner_position=(scanner_x, scanner_y),
                padding=2.0,
            )
            print(f"Auto-computed transect: ({config.x1:.1f}, {config.y1:.1f}) -> ({config.x2:.1f}, {config.y2:.1f})")
            return config

    except Exception as e:
        print(f"Warning: Could not load/compute transect config: {e}")

    return None


def create_gif(
    ds: xr.Dataset,
    output_path: Path,
    variable: str = 'elevation_mode',
    cmap: str = 'terrain',
    fps: float = 2.0,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    show_profile: bool = True,
    y_index: Optional[int] = None,
    x_max_relative: float = 20.0,
    metadata: dict = None,
    transect_config: Optional['profiles.TransectConfig'] = None,
    z_msl: Optional[float] = None,
    z_mhw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create animated GIF from L1 NetCDF dataset with two-panel display.

    Left panel: 2D DEM/grid
    Right panel: Cross-shore profile with slope fit

    Parameters
    ----------
    ds : xr.Dataset
        L1 dataset
    output_path : Path
        Output GIF path
    variable : str
        Variable to plot
    cmap : str
        Matplotlib colormap name
    fps : float
        Frames per second
    dpi : int
        Image resolution
    vmin, vmax : float, optional
        Colorbar limits (auto if None)
    show_colorbar : bool
        Whether to show colorbar
    show_profile : bool
        Whether to show profile panel
    y_index : int, optional
        Y index for profile extraction (default: middle)
    x_max_relative : float
        Max cross-shore distance from seaward edge for slope fit (m).
        Only used if z_msl/z_mhw are not provided.
    metadata : dict
        Metadata from print_metadata()
    z_msl : float, optional
        Mean Sea Level elevation in NAVD88 (m). If provided with z_mhw,
        slope is calculated between these tidal datums.
    z_mhw : float, optional
        Mean High Water elevation in NAVD88 (m). If provided with z_msl,
        slope is calculated between these tidal datums.

    Returns
    -------
    slopes : array
        Slope values for each timestep
    times : array
        Time values
    y_pos : float
        Y position of extracted profile
    """
    # Check variable exists
    if variable not in ds:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{variable}' not found. Available: {available}")

    da = ds[variable]

    # Check dimensions
    if 'time' not in da.dims:
        print(f"Warning: Variable '{variable}' has no time dimension. Creating single-frame GIF.")
        n_frames = 1
        has_time = False
    else:
        n_frames = ds.dims['time']
        has_time = True

    if n_frames == 0:
        raise ValueError("Dataset has no time steps")

    print(f"\nCreating GIF with {n_frames} frames...")

    # Determine color limits
    if vmin is None or vmax is None:
        all_data = da.values
        if vmin is None:
            vmin = float(np.nanpercentile(all_data, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(all_data, 98))

    # Get coordinates
    x = ds.coords['x'].values if 'x' in ds.coords else np.arange(da.shape[-1])
    y = ds.coords['y'].values if 'y' in ds.coords else np.arange(da.shape[-2])

    # Determine profile extraction method
    use_transect = transect_config is not None

    if use_transect:
        print(f"  Using transect: ({transect_config.x1:.1f}, {transect_config.y1:.1f}) -> "
              f"({transect_config.x2:.1f}, {transect_config.y2:.1f})")
        y_pos = None  # Not used for transect mode
        transect_x_utm = None
        transect_y_utm = None
    else:
        # Determine y_index for profile (fallback to horizontal slice)
        if y_index is None:
            y_index = len(y) // 2
        y_pos = float(y[y_index])

    # Pre-calculate all profiles and slopes
    profile_data = []  # Store (x1d, profile) tuples
    slopes = []
    fit_lines = []

    # For transect profiles, determine flip direction ONCE from first frame
    # to ensure consistent orientation across all frames
    flip_direction = None

    for i in range(n_frames):
        if use_transect:
            x1d, profile, slope, x_fit, z_fit, tx, ty, did_flip = extract_transect_profile(
                ds, i, transect_config, x_max_relative=x_max_relative, elevation_var=variable,
                flip_profile=flip_direction,  # Use cached direction after first frame
                z_msl=z_msl, z_mhw=z_mhw,
            )
            if i == 0:
                # Cache the flip direction from first frame for all subsequent frames
                flip_direction = did_flip
                transect_x_utm = tx
                transect_y_utm = ty
                print(f"  Profile orientation: {'flipped' if did_flip else 'not flipped'} (seaward at x=0)")
                if z_msl is not None and z_mhw is not None:
                    print(f"  Slope fit: MSL ({z_msl:.3f}m) to MHW ({z_mhw:.3f}m) NAVD88")
                else:
                    print(f"  Slope fit: 0-{x_max_relative:.0f}m from seaward edge")
            profile_data.append((x1d, profile))
        else:
            _, profile, slope, x_fit, z_fit = extract_profile_and_slope(
                ds, i, y_index, x_max_relative=x_max_relative, elevation_var=variable,
                z_msl=z_msl, z_mhw=z_mhw,
            )
            profile_data.append((x, profile))

        slopes.append(slope)
        fit_lines.append((x_fit, z_fit))

    slopes = np.array(slopes)

    # Print slope summary
    valid_slopes = slopes[~np.isnan(slopes)]
    if len(valid_slopes) > 0:
        print(f"\n--- Slope Summary ---")
        mean_slope = np.mean(valid_slopes)
        print(f"  Mean slope: {mean_slope:.4f} ({np.degrees(np.arctan(mean_slope)):.2f}°)")
        print(f"  Std slope:  {np.std(valid_slopes):.4f}")
        print(f"  Min slope:  {np.min(valid_slopes):.4f} ({np.degrees(np.arctan(np.min(valid_slopes))):.2f}°)")
        print(f"  Max slope:  {np.max(valid_slopes):.4f} ({np.degrees(np.arctan(np.max(valid_slopes))):.2f}°)")
        if use_transect:
            transect_length = np.sqrt(
                (transect_config.x2 - transect_config.x1)**2 +
                (transect_config.y2 - transect_config.y1)**2
            )
            print(f"  Transect length: {transect_length:.1f} m")
        else:
            print(f"  Profile Y position: {y_pos:.1f} m (index {y_index})")
        if z_msl is not None and z_mhw is not None:
            print(f"  Slope reference: MSL ({z_msl:.3f}m) to MHW ({z_mhw:.3f}m) NAVD88")
        else:
            print(f"  Slope fit range: 0-{x_max_relative:.0f} m from seaward edge")

    # Create figure
    if show_profile:
        fig, (ax_dem, ax_profile) = plt.subplots(1, 2, figsize=(16, 7))
    else:
        fig, ax_dem = plt.subplots(figsize=(12, 8))
        ax_profile = None

    # === DEM Panel Setup ===
    if has_time:
        data = da.isel(time=0).values
    else:
        data = da.values

    # Handle dimension order (y, x) vs (x, y)
    if da.dims[-2:] == ('y', 'x'):
        extent = [x.min(), x.max(), y.min(), y.max()]
        aspect = 'equal'
    else:
        extent = [y.min(), y.max(), x.min(), x.max()]
        data = data.T
        aspect = 'equal'

    im = ax_dem.imshow(
        data,
        extent=extent,
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
    )

    # Add line showing profile location
    if use_transect and transect_x_utm is not None:
        # Draw transect line
        profile_line, = ax_dem.plot(
            transect_x_utm, transect_y_utm,
            color='red', linestyle='--', linewidth=2, alpha=0.7
        )
    else:
        # Horizontal line at constant Y
        profile_line = ax_dem.axhline(y=y_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax_dem, label=f'{variable} (m)', shrink=0.8)

    ax_dem.set_xlabel('X (m)', fontsize=11)
    ax_dem.set_ylabel('Y (m)', fontsize=11)

    # Title with metadata
    title_base = metadata.get('filename', 'L1 Grid') if metadata else 'L1 Grid'
    title_dem = ax_dem.set_title('')

    # Stats text box on DEM
    stats_text = ax_dem.text(
        0.02, 0.98, '', transform=ax_dem.transAxes,
        fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # === Profile Panel Setup ===
    if show_profile:
        # Extract all profiles for axis limits
        all_profiles = [p[1] for p in profile_data]

        # Determine profile axis limits
        all_z = np.concatenate([p[~np.isnan(p)] for p in all_profiles if np.any(~np.isnan(p))])
        if len(all_z) > 0:
            z_min_plot = np.nanmin(all_z) - 0.5
            z_max_plot = np.nanmax(all_z) + 0.5
        else:
            z_min_plot, z_max_plot = -2, 5

        # Get x-axis limits from first profile
        x1d_first = profile_data[0][0]

        line_profile, = ax_profile.plot([], [], 'b-', linewidth=2, label='Profile')
        line_fit, = ax_profile.plot([], [], 'r--', linewidth=2.5, label='Slope fit')

        ax_profile.set_xlim(x1d_first.min(), x1d_first.max())
        ax_profile.set_ylim(z_min_plot, z_max_plot)
        ax_profile.set_xlabel('Cross-shore distance (m)', fontsize=11)
        ax_profile.set_ylabel('Elevation (m NAVD88)', fontsize=11)
        if use_transect:
            ax_profile.set_title('Cross-shore Profile (along transect)', fontsize=12)
        else:
            ax_profile.set_title(f'Cross-shore Profile (Y = {y_pos:.1f} m)', fontsize=12)
        ax_profile.grid(True, alpha=0.3)

        # Add horizontal reference lines for tidal datums if using elevation-based slope
        if z_msl is not None and z_mhw is not None:
            ax_profile.axhline(y=z_msl, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'MSL ({z_msl:.2f}m)')
            ax_profile.axhline(y=z_mhw, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'MHW ({z_mhw:.2f}m)')

        ax_profile.legend(loc='upper right')

        # Slope info text box
        slope_text = ax_profile.text(
            0.02, 0.98, '', transform=ax_profile.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        )

        # Time/frame text
        time_text = ax_profile.text(
            0.02, 0.78, '', transform=ax_profile.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.tight_layout()

    def update(frame_idx):
        """Update function for animation."""
        artists = []

        # Update DEM
        if has_time:
            data = da.isel(time=frame_idx).values
            if da.dims[-2:] != ('y', 'x'):
                data = data.T
            time_val = ds.coords['time'].values[frame_idx]
            timestamp = np.datetime_as_string(time_val, unit='s')
        else:
            data = da.values
            if da.dims[-1] != 'x':
                data = data.T
            timestamp = "Static"

        im.set_array(data)
        artists.append(im)

        # Update DEM title
        title_dem.set_text(f"{title_base}\n{timestamp}")
        artists.append(title_dem)

        # Compute and display DEM stats
        stats = compute_frame_stats(da, frame_idx)
        stats_str = (
            f"Frame {frame_idx + 1}/{n_frames}\n"
            f"Min:  {stats['min']:.3f} m\n"
            f"Max:  {stats['max']:.3f} m\n"
            f"Mean: {stats['mean']:.3f} m\n"
            f"Coverage: {stats['coverage']:.1f}%"
        )
        stats_text.set_text(stats_str)
        artists.append(stats_text)

        # Update profile panel
        if show_profile:
            x1d, profile = profile_data[frame_idx]
            slope = slopes[frame_idx]
            x_fit, z_fit = fit_lines[frame_idx]

            # Update profile line
            line_profile.set_data(x1d, profile)
            artists.append(line_profile)

            # Update fit line
            if len(x_fit) > 0:
                line_fit.set_data(x_fit, z_fit)
            else:
                line_fit.set_data([], [])
            artists.append(line_fit)

            # Update slope text
            if not np.isnan(slope):
                angle = np.degrees(np.arctan(slope))
                if z_msl is not None and z_mhw is not None:
                    slope_str = (
                        f"Slope: {slope:.4f}\n"
                        f"Angle: {angle:.2f}°\n"
                        f"MSL→MHW\n"
                        f"({z_msl:.2f}→{z_mhw:.2f}m)"
                    )
                else:
                    slope_str = (
                        f"Slope: {slope:.4f}\n"
                        f"Angle: {angle:.2f}°\n"
                        f"tan(β) = {abs(slope):.4f}"
                    )
            else:
                slope_str = "Slope: N/A"
            slope_text.set_text(slope_str)
            artists.append(slope_text)

            # Update time text
            time_str = f"Time: {timestamp}\nFrame: {frame_idx + 1}/{n_frames}"
            time_text.set_text(time_str)
            artists.append(time_text)

        return artists

    # Create animation
    interval = 1000 / fps  # milliseconds per frame
    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval,
        blit=False,
    )

    # Save GIF
    print(f"Saving to {output_path}...")

    # Use Pillow writer for GIF
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)

    # Print summary
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nGIF created successfully:")
    print(f"  Output: {output_path}")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {fps}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Resolution: {dpi} DPI")

    times = ds.coords['time'].values if 'time' in ds.coords else np.array([0])
    return slopes, times, y_pos


def save_slopes_to_json(
    slopes: np.ndarray,
    times: np.ndarray,
    y_pos: float,
    x_max_relative: float,
    output_path: Path,
    source_file: str,
    z_msl: Optional[float] = None,
    z_mhw: Optional[float] = None,
) -> None:
    """Save slope timeseries to JSON file."""
    import json

    # Convert slopes to angles
    angles = np.degrees(np.arctan(slopes))

    # Convert numpy datetime64 to ISO strings
    time_strings = [np.datetime_as_string(t, unit='s') for t in times]

    # Build JSON structure
    metadata = {
        'title': 'L1 Beach Foreshore Slope Timeseries',
        'description': 'Foreshore slopes calculated for Stockdon runup',
        'source_file': source_file,
        'slope_method': 'Theil-Sen robust regression',
        'created_by': 'gif_nc_l1.py',
    }

    # Add slope calculation method details
    if z_msl is not None and z_mhw is not None:
        metadata['slope_reference'] = 'tidal_datums'
        metadata['z_msl_navd88_m'] = float(z_msl)
        metadata['z_mhw_navd88_m'] = float(z_mhw)
        metadata['description'] = 'Foreshore slope between MSL and MHW (tidal datums)'
    else:
        metadata['slope_reference'] = 'distance_based'
        metadata['x_max_relative_m'] = float(x_max_relative)
        if y_pos is not None:
            metadata['y_position_m'] = float(y_pos)

    data = {
        'metadata': metadata,
        'statistics': {
            'mean_slope': float(np.nanmean(slopes)),
            'std_slope': float(np.nanstd(slopes)),
            'min_slope': float(np.nanmin(slopes)),
            'max_slope': float(np.nanmax(slopes)),
            'mean_angle_deg': float(np.nanmean(angles)),
            'n_valid': int(np.sum(~np.isnan(slopes))),
            'n_total': len(slopes),
        },
        'timeseries': [
            {
                'time': t,
                'slope': float(s) if not np.isnan(s) else None,
                'angle_deg': float(a) if not np.isnan(a) else None,
            }
            for t, s, a in zip(time_strings, slopes, angles)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved slopes to: {output_path}')


def process_single_file(
    nc_path: Path,
    output_path: Path,
    slopes_dir: Path,
    args,
    transect_config: Optional['profiles.TransectConfig'] = None,
) -> bool:
    """
    Process a single NC file and create GIF.

    Returns True on success, False on failure.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {nc_path.name}")
    print(f"{'='*70}")

    # Load dataset
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"Error: Failed to open NetCDF file: {e}", file=sys.stderr)
        return False

    # List variables mode
    if args.list_variables:
        print(f"Variables in {nc_path.name}:")
        for var in sorted(ds.data_vars):
            dims = ", ".join(ds[var].dims)
            print(f"  {var} ({dims})")
        ds.close()
        return True

    # Print metadata
    metadata = print_metadata(ds, nc_path)

    # Exit if metadata-only mode
    if args.metadata_only:
        ds.close()
        return True

    # Auto-compute transect config if not provided
    if transect_config is None and not args.y_index:
        x_grid = ds.x.values if 'x' in ds.coords else None
        y_grid = ds.y.values if 'y' in ds.coords else None
        if x_grid is not None and y_grid is not None:
            transect_config = get_transect_config(args.config, x_grid, y_grid)

    # Create GIF
    try:
        slopes, times, y_pos = create_gif(
            ds,
            output_path,
            variable=args.variable,
            transect_config=transect_config,
            cmap=args.cmap,
            fps=args.fps,
            dpi=args.dpi,
            vmin=args.vmin,
            vmax=args.vmax,
            show_colorbar=not args.no_colorbar,
            show_profile=not args.no_profile,
            y_index=args.y_index,
            x_max_relative=args.x_max,
            metadata=metadata,
            z_msl=args.z_msl,
            z_mhw=args.z_mhw,
        )

        # Save slopes if requested
        if args.save_slopes:
            slopes_path = slopes_dir / (nc_path.stem + '_slopes.json')
            save_slopes_to_json(
                slopes, times, y_pos, args.x_max, slopes_path, nc_path.name,
                z_msl=args.z_msl, z_mhw=args.z_mhw
            )

        ds.close()
        return True

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        ds.close()
        return False
    except Exception as e:
        print(f"Error creating GIF: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        ds.close()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIFs from L1 NetCDF files (DEM + profile with slope)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all L1 files from config
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json

    # Process specific file only
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --input L1_20260115.nc

    # Just print metadata for all files
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --metadata-only

    # Custom settings
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json -v elevation --cmap viridis --fps 1

    # Specify profile location and slope fit range
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --y-index 50 --x-max 30

    # DEM only (no profile panel)
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --no-profile

    # Save slopes to NetCDF for all files
    python scripts/gif_nc_l1.py --config configs/do_livox_config_20260112.json --save-slopes
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Config file (required) - reads NC files from processFolder/level1/"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Process single file only (filename or full path)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for GIFs (default: plotFolder/level1/)"
    )
    parser.add_argument(
        "--variable", "-v",
        type=str,
        default="elevation_mode",
        help="Variable to plot (default: elevation_mode)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="terrain",
        help="Matplotlib colormap (default: terrain)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second (default: 2)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in DPI (default: 150)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for colorbar (default: auto from 2nd percentile)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for colorbar (default: auto from 98th percentile)"
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable colorbar"
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable profile panel (single panel DEM only)"
    )
    parser.add_argument(
        "--y-index",
        type=int,
        default=None,
        help="Y index for profile extraction (default: middle of grid)"
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=20.0,
        help="Max cross-shore distance from seaward edge for slope fit in meters (default: 20). "
             "Only used if --z-msl and --z-mhw are not provided."
    )
    parser.add_argument(
        "--z-msl",
        type=float,
        default=0.744,
        help="Mean Sea Level elevation in NAVD88 (m) for slope calculation (default: 0.744). "
             "Slope is calculated between MSL and MHW elevations."
    )
    parser.add_argument(
        "--z-mhw",
        type=float,
        default=1.34,
        help="Mean High Water elevation in NAVD88 (m) for slope calculation (default: 1.34). "
             "Slope is calculated between MSL and MHW elevations."
    )
    parser.add_argument(
        "--no-tidal-slope",
        action="store_true",
        help="Disable tidal datum slope calculation (MSL to MHW) and use distance-based fit instead"
    )
    parser.add_argument(
        "--save-slopes",
        action="store_true",
        help="Save slope timeseries to JSON file (in processFolder/slopes/)"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only print metadata, don't create GIF"
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List available variables and exit"
    )

    args = parser.parse_args()

    # Handle --no-tidal-slope flag: disable elevation-based slope calculation
    if args.no_tidal_slope:
        args.z_msl = None
        args.z_mhw = None

    # Load config
    from phase1 import load_config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine input directory
    input_dir = config.process_folder / "level1"
    if not input_dir.exists():
        print(f"Error: Level1 directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = config.plot_folder / "level1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover NC files to process
    if args.input is not None:
        # Single file mode
        input_path = Path(args.input)
        if input_path.is_absolute():
            nc_files = [input_path]
        else:
            # Treat as filename, look in input_dir
            nc_files = [input_dir / input_path]

        if not nc_files[0].exists():
            print(f"Error: Input file not found: {nc_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        # Discover all NC files in level1 directory
        nc_files = sorted(input_dir.glob("L1_*.nc"))
        if not nc_files:
            print(f"No L1 NetCDF files found in: {input_dir}", file=sys.stderr)
            sys.exit(1)

    # Slopes output directory (processFolder/slopes/)
    slopes_dir = config.process_folder / "slopes"
    if args.save_slopes:
        slopes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(nc_files)} L1 file(s) to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if args.save_slopes:
        print(f"Slopes directory: {slopes_dir}")

    # Process each file
    success_count = 0
    fail_count = 0

    for nc_path in nc_files:
        output_path = output_dir / nc_path.with_suffix('.gif').name
        if process_single_file(nc_path, output_path, slopes_dir, args):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Processed: {success_count}/{len(nc_files)} files")
    if fail_count > 0:
        print(f"  Failed: {fail_count} files")
    print(f"  Output directory: {output_dir}")
    if args.save_slopes:
        print(f"  Slopes directory: {slopes_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
