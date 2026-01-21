#!/usr/bin/env python
"""
Animate L1 cross-shore profiles with slope calculation.

Creates a GIF animation showing the beach profile at each timestep
with the foreshore slope displayed.

Usage:
    python scripts/animate_l1_slope.py tests/data/test_l1.nc
    python scripts/animate_l1_slope.py tests/data/test_l1.nc -o figures/l1/
    python scripts/animate_l1_slope.py tests/data/test_l1.nc --fps 2
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_l1_dataset(path: Path) -> xr.Dataset:
    """Load L1 NetCDF dataset."""
    return xr.open_dataset(path)


def calculate_slope(x: np.ndarray, z: np.ndarray,
                    x_max_relative: float = 60.0,
                    z_min_threshold: Optional[float] = None,
                    use_robust: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
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
        Default 60m captures the typical swash/foreshore zone.
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
    if z_min_threshold is None:
        # Use 10th percentile as automatic threshold to exclude low outliers
        z_min_threshold = np.percentile(z_valid[in_swash], 10)

    # Apply both spatial and elevation filters
    in_range = in_swash & (z_valid >= z_min_threshold)
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


def create_animation(ds: xr.Dataset, output_path: Path, fps: int = 2,
                     y_index: Optional[int] = None, x_max_relative: float = 20.0):
    """
    Create animated GIF of profiles with slope.

    Parameters
    ----------
    ds : xr.Dataset
        L1 dataset
    output_path : Path
        Output path for GIF
    fps : int
        Frames per second
    y_index : int, optional
        Y index for profile extraction. If None, uses middle.
    x_max_relative : float
        Max cross-shore distance from seaward edge for slope fit (m).
        Default 20m for typical swash zone / Stockdon foreshore slope.
    """
    x = ds.x.values
    n_times = ds.sizes["time"]

    # Use middle y index if not specified
    if y_index is None:
        y_index = len(ds.y) // 2
    y_pos = float(ds.y[y_index])

    # Extract profiles and calculate slopes for all timesteps
    profiles = []
    slopes = []
    fit_lines = []

    for i in range(n_times):
        profile = ds.elevation.isel(time=i, y=y_index).values
        profiles.append(profile)

        slope, x_fit, z_fit = calculate_slope(x, profile, x_max_relative=x_max_relative)
        slopes.append(slope)
        fit_lines.append((x_fit, z_fit))

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine axis limits
    all_z = np.concatenate([p[~np.isnan(p)] for p in profiles if np.any(~np.isnan(p))])
    z_min = np.nanmin(all_z) - 0.5
    z_max = np.nanmax(all_z) + 0.5

    # Initialize plot elements
    line_profile, = ax.plot([], [], 'b-', linewidth=2, label='Profile')
    line_fit, = ax.plot([], [], 'r--', linewidth=2, label='Slope fit')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    slope_text = ax.text(0.02, 0.88, '', transform=ax.transAxes,
                         fontsize=14, verticalalignment='top',
                         fontfamily='monospace', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('Cross-shore distance (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title(f'L1 Beach Profile Animation (Y = {y_pos:.1f}m)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    def init():
        line_profile.set_data([], [])
        line_fit.set_data([], [])
        time_text.set_text('')
        slope_text.set_text('')
        return line_profile, line_fit, time_text, slope_text

    def animate(frame):
        profile = profiles[frame]
        slope = slopes[frame]
        x_fit, z_fit = fit_lines[frame]

        # Update profile line
        line_profile.set_data(x, profile)

        # Update fit line
        if len(x_fit) > 0:
            line_fit.set_data(x_fit, z_fit)
        else:
            line_fit.set_data([], [])

        # Update text
        time_val = ds.time.values[frame]
        time_str = str(time_val)[:19]
        time_text.set_text(f'Time: {time_str}\nFrame: {frame+1}/{n_times}')

        if not np.isnan(slope):
            # Convert slope to angle in degrees
            angle = np.degrees(np.arctan(slope))
            slope_text.set_text(f'Slope: {slope:.4f} ({angle:.2f}°)\ntan(β) = {abs(slope):.4f}')
        else:
            slope_text.set_text('Slope: N/A')

        return line_profile, line_fit, time_text, slope_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_times,
                         interval=1000/fps, blit=True)

    # Save as GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f'Saved animation: {output_path}')

    # Print slope summary
    valid_slopes = [s for s in slopes if not np.isnan(s)]
    if valid_slopes:
        print(f'\nSlope Summary:')
        print(f'  Mean slope: {np.mean(valid_slopes):.4f} ({np.degrees(np.arctan(np.mean(valid_slopes))):.2f}°)')
        print(f'  Std slope:  {np.std(valid_slopes):.4f}')
        print(f'  Min slope:  {np.min(valid_slopes):.4f}')
        print(f'  Max slope:  {np.max(valid_slopes):.4f}')

    return np.array(slopes), ds.time.values, y_pos, x_max_relative


def save_slopes_to_nc(slopes: np.ndarray, times: np.ndarray, y_pos: float,
                      x_max_relative: float, output_path: Path):
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
            'created_by': 'animate_l1_slope.py',
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_path)
    print(f'Saved slopes to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description="Animate L1 profiles with slope calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input L1 NetCDF file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory for GIF (default: figures/l1)")
    parser.add_argument("--fps", type=int, default=2,
                        help="Frames per second (default: 2)")
    parser.add_argument("--y-index", type=int, default=None,
                        help="Y index for profile (default: middle)")
    parser.add_argument("--x-max", type=float, default=20.0,
                        help="Max cross-shore distance from seaward edge for slope fit in meters (default: 20)")
    parser.add_argument("--save-nc", action="store_true",
                        help="Save slopes to NetCDF file")
    parser.add_argument("--nc-only", action="store_true",
                        help="Only save NC file, skip GIF animation")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set output directory
    if args.output is None:
        output_dir = Path("figures/l1")
    else:
        output_dir = args.output

    gif_path = output_dir / "l1_profile_slope_animation.gif"
    nc_path = output_dir / "l1_slopes.nc"

    # Load data
    print(f"Loading: {args.input}")
    ds = load_l1_dataset(args.input)

    print(f"Dataset has {ds.sizes['time']} timesteps")
    print(f"Fitting slope using first {args.x_max}m from seaward edge (swash zone)")

    if args.nc_only:
        # Calculate slopes without animation
        x = ds.x.values
        y_index = args.y_index if args.y_index is not None else len(ds.y) // 2
        y_pos = float(ds.y[y_index])

        slopes = []
        for i in range(ds.sizes["time"]):
            profile = ds.elevation.isel(time=i, y=y_index).values
            slope, _, _ = calculate_slope(x, profile, x_max_relative=args.x_max)
            slopes.append(slope)

        slopes = np.array(slopes)
        times = ds.time.values

        # Print summary
        valid_slopes = slopes[~np.isnan(slopes)]
        if len(valid_slopes) > 0:
            print(f'\nSlope Summary:')
            print(f'  Mean slope: {np.mean(valid_slopes):.4f} ({np.degrees(np.arctan(np.mean(valid_slopes))):.2f}°)')
            print(f'  Std slope:  {np.std(valid_slopes):.4f}')
            print(f'  Min slope:  {np.min(valid_slopes):.4f}')
            print(f'  Max slope:  {np.max(valid_slopes):.4f}')

        save_slopes_to_nc(slopes, times, y_pos, args.x_max, nc_path)
    else:
        # Create animation (and optionally save NC)
        print(f"Creating animation at {args.fps} fps...")
        slopes, times, y_pos, x_max = create_animation(
            ds, gif_path, fps=args.fps, y_index=args.y_index,
            x_max_relative=args.x_max
        )

        if args.save_nc:
            save_slopes_to_nc(slopes, times, y_pos, x_max, nc_path)

    ds.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
