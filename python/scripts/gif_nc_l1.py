#!/usr/bin/env python3
"""
Create animated GIF from L1 NetCDF file showing beach surface grids and
cross-shore profiles with slope calculation.

Two-panel figure:
  - Left: 2D DEM/grid view with colorbar
  - Right: Cross-shore profile with foreshore slope fit

Displays metadata about timestamps and grid statistics both in the animation
frames and printed to the command line.

Usage:
    python scripts/gif_nc_l1.py path/to/L1_YYYYMMDD.nc [options]

Options:
    --output PATH       Output GIF path (default: same name as input with .gif)
    --variable VAR      Variable to plot (default: elevation_mode)
    --cmap COLORMAP     Matplotlib colormap (default: terrain)
    --fps FPS           Frames per second (default: 2)
    --dpi DPI           Resolution (default: 150)
    --vmin FLOAT        Minimum value for colorbar (default: auto)
    --vmax FLOAT        Maximum value for colorbar (default: auto)
    --y-index INT       Y index for profile extraction (default: middle)
    --x-max FLOAT       Max cross-shore distance for slope fit in meters (default: 20)
    --no-colorbar       Disable colorbar
    --no-profile        Disable profile panel (single panel DEM only)
    --save-slopes       Save slopes to NetCDF file

Examples:
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --variable elevation --cmap viridis
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --output beach_surface.gif --fps 1
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --y-index 50 --x-max 30
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
    use_robust: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate foreshore slope using linear regression.

    Designed for Stockdon runup calculations - fits slope in the swash zone
    from the seaward edge of the profile up to x_max_relative meters inland.

    Parameters
    ----------
    x : array
        Cross-shore positions (m), can be UTM coordinates
    z : array
        Elevation values (m)
    x_max_relative : float
        Maximum cross-shore distance from the seaward edge to include in fit (m).
        Default 20m captures the typical swash/foreshore zone.
    z_min_threshold : float, optional
        Minimum elevation threshold - exclude points below this (rejects water noise).
        If None, uses 10th percentile of elevations as threshold.
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
    elevation_var: str = 'elevation'
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

    slope, x_fit, z_fit = calculate_slope(x, profile, x_max_relative=x_max_relative)

    return x, profile, slope, x_fit, z_fit


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
        Max cross-shore distance from seaward edge for slope fit (m)
    metadata : dict
        Metadata from print_metadata()

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

    # Determine y_index for profile
    if y_index is None:
        y_index = len(y) // 2
    y_pos = float(y[y_index])

    # Pre-calculate all profiles and slopes
    profiles = []
    slopes = []
    fit_lines = []

    for i in range(n_frames):
        _, profile, slope, x_fit, z_fit = extract_profile_and_slope(
            ds, i, y_index, x_max_relative=x_max_relative, elevation_var=variable
        )
        profiles.append(profile)
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
        print(f"  Profile Y position: {y_pos:.1f} m (index {y_index})")
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

    # Add horizontal line showing profile location
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
        # Determine profile axis limits
        all_z = np.concatenate([p[~np.isnan(p)] for p in profiles if np.any(~np.isnan(p))])
        if len(all_z) > 0:
            z_min_plot = np.nanmin(all_z) - 0.5
            z_max_plot = np.nanmax(all_z) + 0.5
        else:
            z_min_plot, z_max_plot = -2, 5

        line_profile, = ax_profile.plot([], [], 'b-', linewidth=2, label='Profile')
        line_fit, = ax_profile.plot([], [], 'r--', linewidth=2.5, label='Slope fit')

        ax_profile.set_xlim(x.min(), x.max())
        ax_profile.set_ylim(z_min_plot, z_max_plot)
        ax_profile.set_xlabel('Cross-shore distance (m)', fontsize=11)
        ax_profile.set_ylabel('Elevation (m)', fontsize=11)
        ax_profile.set_title(f'Cross-shore Profile (Y = {y_pos:.1f} m)', fontsize=12)
        ax_profile.grid(True, alpha=0.3)
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
            profile = profiles[frame_idx]
            slope = slopes[frame_idx]
            x_fit, z_fit = fit_lines[frame_idx]

            # Update profile line
            line_profile.set_data(x, profile)
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


def save_slopes_to_nc(
    slopes: np.ndarray,
    times: np.ndarray,
    y_pos: float,
    x_max_relative: float,
    output_path: Path
) -> None:
    """Save slope timeseries to NetCDF file."""
    # Convert slopes to angles
    angles = np.degrees(np.arctan(slopes))

    # Create xarray dataset
    ds_out = xr.Dataset(
        {
            'slope': (['time'], slopes, {
                'units': 'dimensionless',
                'long_name': 'Foreshore slope (tan beta)',
                'description': 'Beach foreshore slope dz/dx',
            }),
            'slope_angle': (['time'], angles, {
                'units': 'degrees',
                'long_name': 'Foreshore slope angle',
                'description': 'Beach foreshore slope in degrees',
            }),
        },
        coords={
            'time': times,
        },
        attrs={
            'title': 'L1 Beach Foreshore Slope Timeseries',
            'description': 'Foreshore slopes calculated for Stockdon runup',
            'y_position_m': y_pos,
            'x_max_relative_m': x_max_relative,
            'slope_method': 'Theil-Sen robust regression',
            'created_by': 'gif_nc_l1.py',
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_path)
    print(f'Saved slopes to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF from L1 NetCDF file (DEM + profile with slope)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic two-panel animation (DEM + profile with slope)
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc

    # Just print metadata
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --metadata-only

    # Custom settings
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc -v elevation -c viridis --fps 1

    # Specify profile location and slope fit range
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --y-index 50 --x-max 30

    # DEM only (no profile panel)
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --no-profile

    # Save slopes to NetCDF
    python scripts/gif_nc_l1.py data/level1/L1_20260115.nc --save-slopes
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input L1 NetCDF file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output GIF path (default: same name with .gif extension)"
    )
    parser.add_argument(
        "--variable", "-v",
        type=str,
        default="elevation_mode",
        help="Variable to plot (default: elevation_mode)"
    )
    parser.add_argument(
        "--cmap", "-c",
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
        help="Max cross-shore distance from seaward edge for slope fit in meters (default: 20)"
    )
    parser.add_argument(
        "--save-slopes",
        action="store_true",
        help="Save slope timeseries to NetCDF file"
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

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    try:
        ds = xr.open_dataset(args.input)
    except Exception as e:
        print(f"Error: Failed to open NetCDF file: {e}", file=sys.stderr)
        sys.exit(1)

    # List variables mode
    if args.list_variables:
        print(f"Variables in {args.input.name}:")
        for var in sorted(ds.data_vars):
            dims = ", ".join(ds[var].dims)
            print(f"  {var} ({dims})")
        ds.close()
        sys.exit(0)

    # Print metadata
    metadata = print_metadata(ds, args.input)

    # Exit if metadata-only mode
    if args.metadata_only:
        ds.close()
        sys.exit(0)

    # Determine output path
    if args.output is None:
        output_path = args.input.with_suffix('.gif')
    else:
        output_path = args.output

    # Create GIF
    try:
        slopes, times, y_pos = create_gif(
            ds,
            output_path,
            variable=args.variable,
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
        )

        # Save slopes if requested
        if args.save_slopes:
            slopes_path = args.input.with_name(args.input.stem + '_slopes.nc')
            save_slopes_to_nc(slopes, times, y_pos, args.x_max, slopes_path)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        ds.close()
        sys.exit(1)
    except Exception as e:
        print(f"Error creating GIF: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        ds.close()
        sys.exit(1)

    ds.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
