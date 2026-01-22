#!/usr/bin/env python
"""
Visualize L1 (beach surface DEM) processing results.

Usage:
    python scripts/visualize_l1.py tests/data/test_l1.nc
    python scripts/visualize_l1.py tests/data/test_l1.nc -o output_dir/
    python scripts/visualize_l1.py tests/data/test_l1.nc --show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add code directory to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))


def load_l1_dataset(path: Path) -> xr.Dataset:
    """Load L1 NetCDF dataset."""
    return xr.open_dataset(path)


def plot_dem_timesteps(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot DEM for each timestep side by side."""
    n_times = ds.sizes["time"]
    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4))

    if n_times == 1:
        axes = [axes]

    for i, t in enumerate(ds.time.values):
        ax = axes[i]
        elev = ds.elevation.sel(time=t)
        im = ax.pcolormesh(ds.x, ds.y, elev, cmap='terrain', vmin=-2, vmax=3)
        ax.set_title(f'Time {i+1}\n{str(t)[:19]}', fontsize=9)
        ax.set_xlabel('X (m)')
        if i == 0:
            ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

    plt.colorbar(im, ax=axes, label='Elevation (m)', shrink=0.8)
    plt.suptitle('L1 Beach Surface DEMs', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'l1_dem_timesteps.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_statistics(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot L1 statistics: mean, std, count, SNR."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean elevation
    ax = axes[0, 0]
    mean_elev = ds.elevation.mean(dim='time')
    im = ax.pcolormesh(ds.x, ds.y, mean_elev, cmap='terrain', vmin=-2, vmax=3)
    ax.set_title('Mean Elevation')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Temporal std (variability)
    ax = axes[0, 1]
    std_elev = ds.elevation.std(dim='time')
    im = ax.pcolormesh(ds.x, ds.y, std_elev, cmap='Reds', vmin=0, vmax=0.5)
    ax.set_title('Temporal Std Dev (variability)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Std Dev (m)')

    # Mean count
    ax = axes[1, 0]
    mean_count = ds['count'].mean(dim='time')
    im = ax.pcolormesh(ds.x, ds.y, mean_count, cmap='viridis')
    ax.set_title('Mean Point Count per Bin')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Count')

    # Mean SNR
    ax = axes[1, 1]
    mean_snr = ds['snr'].mean(dim='time')
    im = ax.pcolormesh(ds.x, ds.y, np.log10(mean_snr + 1), cmap='plasma')
    ax.set_title('Mean SNR (log10)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='log10(SNR)')

    plt.suptitle('L1 Processing Statistics', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'l1_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_profiles(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot cross-shore and alongshore profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cross-shore profile (middle Y)
    y_mid_idx = len(ds.y) // 2
    y_mid = float(ds.y[y_mid_idx])

    ax = axes[0]
    for i, t in enumerate(ds.time.values):
        profile = ds.elevation.sel(time=t).isel(y=y_mid_idx)
        ax.plot(ds.x, profile, label=f'T{i+1}', alpha=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Cross-shore Profiles at Y={y_mid:.1f}m')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Alongshore profile (middle X)
    x_mid_idx = len(ds.x) // 2
    x_mid = float(ds.x[x_mid_idx])

    ax = axes[1]
    for i, t in enumerate(ds.time.values):
        profile = ds.elevation.sel(time=t).isel(x=x_mid_idx)
        ax.plot(ds.y, profile, label=f'T{i+1}', alpha=0.8)
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Alongshore Profiles at X={x_mid:.1f}m')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('L1 Profile Comparison Across Time Steps', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'l1_profiles.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_elevation_histogram(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot elevation distribution histograms."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, t in enumerate(ds.time.values):
        elev = ds.elevation.sel(time=t).values.flatten()
        elev = elev[~np.isnan(elev)]
        ax.hist(elev, bins=50, alpha=0.5, label=f'T{i+1}', density=True)

    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Density')
    ax.set_title('Elevation Distribution Across Time Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'l1_elevation_hist.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def print_summary(ds: xr.Dataset):
    """Print dataset summary."""
    print("=" * 50)
    print("L1 Dataset Summary")
    print("=" * 50)
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Time steps: {ds.sizes['time']}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"X range: {float(ds.x.min()):.2f} to {float(ds.x.max()):.2f} m")
    print(f"Y range: {float(ds.y.min()):.2f} to {float(ds.y.max()):.2f} m")
    print(f"Elevation range: {float(ds.elevation.min()):.2f} to {float(ds.elevation.max()):.2f} m")
    print(f"Variables: {list(ds.data_vars)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize L1 beach surface DEM results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input L1 NetCDF file")
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help="Config file to determine output directory (uses plot_folder/level1/)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory for figures (default: config plot_folder/level1/ or input dir)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--no-summary", action="store_true", help="Skip printing summary")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set output directory
    if args.output is not None:
        output_dir = args.output
    elif args.config is not None:
        # Use config's plot_folder with level1/ subfolder
        from phase1 import load_config
        config = load_config(args.config)
        output_dir = config.plot_folder / "level1"
    else:
        # Fallback: figures/ subdirectory in input dir
        output_dir = args.input.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading: {args.input}")
    ds = load_l1_dataset(args.input)

    # Print summary
    if not args.no_summary:
        print_summary(ds)

    # Generate plots
    print(f"\nGenerating figures in: {output_dir}")
    plot_dem_timesteps(ds, output_dir, args.show)
    plot_statistics(ds, output_dir, args.show)
    plot_profiles(ds, output_dir, args.show)
    plot_elevation_histogram(ds, output_dir, args.show)

    ds.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
