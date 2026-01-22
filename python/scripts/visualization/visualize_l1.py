#!/usr/bin/env python
"""
Visualize L1 (beach surface DEM) processing results.

Processes all L1 NetCDF files from config's processFolder/level1/ directory.
Outputs figures to config's plotFolder/level1/ directory.

Usage:
    # Process ALL L1 files from processFolder/level1/
    python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json

    # Process a single file
    python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json --input L1_20260112.nc

    # Show plots interactively
    python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json --show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add code directory to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from phase1 import load_config


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


def print_summary(ds: xr.Dataset, filename: str = ""):
    """Print dataset summary."""
    print("=" * 50)
    print(f"L1 Dataset Summary: {filename}")
    print("=" * 50)
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Time steps: {ds.sizes['time']}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"X range: {float(ds.x.min()):.2f} to {float(ds.x.max()):.2f} m")
    print(f"Y range: {float(ds.y.min()):.2f} to {float(ds.y.max()):.2f} m")
    print(f"Elevation range: {float(ds.elevation.min()):.2f} to {float(ds.elevation.max()):.2f} m")
    print(f"Variables: {list(ds.data_vars)}")
    print("=" * 50)


def process_single_file(
    nc_path: Path,
    output_dir: Path,
    show: bool = False,
    no_summary: bool = False,
) -> bool:
    """
    Process a single L1 NetCDF file and generate visualization figures.

    Parameters
    ----------
    nc_path : Path
        Path to the L1 NetCDF file
    output_dir : Path
        Output directory for figures
    show : bool
        Show plots interactively
    no_summary : bool
        Skip printing summary

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Create date-specific output directory
        date_str = nc_path.stem.replace("L1_", "")[:8]
        date_output_dir = output_dir / date_str
        date_output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"\nLoading: {nc_path.name}")
        ds = load_l1_dataset(nc_path)

        # Print summary
        if not no_summary:
            print_summary(ds, nc_path.name)

        # Generate plots
        print(f"Generating figures in: {date_output_dir}")
        plot_dem_timesteps(ds, date_output_dir, show)
        plot_statistics(ds, date_output_dir, show)
        plot_profiles(ds, date_output_dir, show)
        plot_elevation_histogram(ds, date_output_dir, show)

        ds.close()
        return True

    except Exception as e:
        print(f"Error processing {nc_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize L1 beach surface DEM results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process ALL L1 files from processFolder/level1/
    python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json

    # Process a single file
    python scripts/visualization/visualize_l1.py --config configs/do_livox_config_20260112.json --input L1_20260112.nc
        """,
    )
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="Config file (required) - determines input/output directories")
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Single input filename to process (default: process ALL L1_*.nc files)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Override output directory (default: config's plotFolder/level1/)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--no-summary", action="store_true", help="Skip printing summary")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set input directory (processFolder/level1/)
    input_dir = config.process_folder / "level1"
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Set output directory
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = config.plot_folder / "level1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover L1 files
    if args.input is not None:
        # Single file mode
        nc_path = input_dir / args.input
        if not nc_path.exists():
            print(f"Error: Input file not found: {nc_path}")
            return 1
        nc_files = [nc_path]
    else:
        # Batch mode - find all L1_*.nc files
        nc_files = sorted(input_dir.glob("L1_*.nc"))

    if not nc_files:
        print(f"No L1 NetCDF files found in: {input_dir}")
        return 1

    print(f"Found {len(nc_files)} L1 file(s) in: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process each file
    success_count = 0
    for nc_path in nc_files:
        if process_single_file(nc_path, output_dir, args.show, args.no_summary):
            success_count += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Processed {success_count}/{len(nc_files)} files successfully")
    print("=" * 50)

    return 0 if success_count == len(nc_files) else 1


if __name__ == "__main__":
    exit(main())
