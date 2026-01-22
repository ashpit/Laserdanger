#!/usr/bin/env python
"""
Visualize L2 (wave-resolving timestack) processing results.

Usage:
    python scripts/visualize_l2.py tests/data/test_l2.nc
    python scripts/visualize_l2.py tests/data/test_l2.nc -o output_dir/
    python scripts/visualize_l2.py tests/data/test_l2.nc --show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add code directory to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))


def load_l2_dataset(path: Path) -> xr.Dataset:
    """Load L2 NetCDF dataset."""
    return xr.open_dataset(path)


def plot_timestack(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot elevation timestack Z(x,t) - raw and filtered."""
    Z = ds.elevation.values
    Z_raw = ds.elevation_raw.values
    x = ds.x.values
    t = ds.time_seconds.values

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Raw timestack
    ax = axes[0]
    im = ax.pcolormesh(t, x, Z_raw, cmap='terrain', vmin=-1, vmax=3, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-shore distance (m)')
    ax.set_title('L2 Timestack - Raw Elevation Z(x,t)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Filtered timestack
    ax = axes[1]
    im = ax.pcolormesh(t, x, Z, cmap='terrain', vmin=-1, vmax=3, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-shore distance (m)')
    ax.set_title('L2 Timestack - After Outlier Filtering')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    plt.tight_layout()

    output_path = output_dir / 'l2_timestack.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_profiles_and_timeseries(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot cross-shore profiles and time series."""
    Z = ds.elevation.values
    x = ds.x.values
    t = ds.time_seconds.values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Profiles at different times
    ax = axes[0]
    time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    for i, idx in enumerate(time_indices):
        ax.plot(x, Z[:, idx], color=colors[i], label=f't={t[idx]:.0f}s', alpha=0.8)
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Cross-shore Profiles at Different Times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time series at different x positions
    ax = axes[1]
    x_indices = [len(x)//5, 2*len(x)//5, 3*len(x)//5, 4*len(x)//5]
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_indices)))
    for i, idx in enumerate(x_indices):
        ax.plot(t, Z[idx, :], color=colors[i], label=f'x={x[idx]:.0f}m', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Time Series at Different X Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'l2_profiles_timeseries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_intensity(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot intensity timestack."""
    I = ds.intensity.values
    x = ds.x.values
    t = ds.time_seconds.values

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(t, x, I, cmap='gray_r', vmin=0, vmax=100, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-shore distance (m)')
    ax.set_title('L2 Intensity Timestack I(x,t)')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()

    output_path = output_dir / 'l2_intensity.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_statistics(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot L2 statistics."""
    Z = ds.elevation.values
    x = ds.x.values
    t = ds.time_seconds.values
    count = ds['count'].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Temporal mean profile
    ax = axes[0, 0]
    ax.plot(x, np.nanmean(Z, axis=1))
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Mean elevation (m)')
    ax.set_title('Temporal Mean Profile')
    ax.grid(True, alpha=0.3)

    # Temporal std (wave activity)
    ax = axes[0, 1]
    ax.plot(x, np.nanstd(Z, axis=1))
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Std elevation (m)')
    ax.set_title('Temporal Std Dev (Wave Activity)')
    ax.grid(True, alpha=0.3)

    # Data density
    ax = axes[1, 0]
    im = ax.pcolormesh(t, x, count, cmap='viridis', shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-shore distance (m)')
    ax.set_title('Point Count per Bin')
    plt.colorbar(im, ax=ax, label='Count')

    # Valid data percentage over time
    ax = axes[1, 1]
    valid_pct = 100 * np.sum(~np.isnan(Z), axis=0) / Z.shape[0]
    ax.plot(t, valid_pct)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Valid data (%)')
    ax.set_title('Data Coverage Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.suptitle('L2 Processing Statistics', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'l2_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def plot_wave_detection(ds: xr.Dataset, output_dir: Path, show: bool = False):
    """Plot wave detection visualization - shows temporal variability patterns."""
    Z = ds.elevation.values
    x = ds.x.values
    t = ds.time_seconds.values

    # Compute "dry beach" reference using rolling minimum
    window_size = min(100, Z.shape[1] // 2)  # ~50s window at 2Hz
    if window_size > 1:
        from scipy.ndimage import minimum_filter1d
        Z_dry = minimum_filter1d(Z, size=window_size, axis=1, mode='nearest')
    else:
        Z_dry = np.nanmin(Z, axis=1, keepdims=True)

    # Water level = instantaneous - dry reference
    water_level = Z - Z_dry

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Raw elevation
    ax = axes[0]
    im = ax.pcolormesh(t, x, Z, cmap='terrain', vmin=-1, vmax=3, shading='auto')
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title('Elevation Z(x,t)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Dry beach reference
    ax = axes[1]
    im = ax.pcolormesh(t, x, Z_dry, cmap='terrain', vmin=-1, vmax=3, shading='auto')
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title(f'Dry Beach Reference (rolling min, {window_size*0.5:.0f}s window)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Water level (wave signal)
    ax = axes[2]
    im = ax.pcolormesh(t, x, water_level, cmap='Blues', vmin=0, vmax=0.5, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title('Water Level (Z - dry reference)')
    plt.colorbar(im, ax=ax, label='Water depth (m)')

    plt.tight_layout()

    output_path = output_dir / 'l2_wave_detection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    if show:
        plt.show()
    plt.close()


def print_summary(ds: xr.Dataset):
    """Print dataset summary."""
    print("=" * 50)
    print("L2 Dataset Summary")
    print("=" * 50)
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Grid shape: {ds.sizes['x']} x {ds.sizes['time']} (x × time)")
    print(f"Time range: {float(ds.time_seconds.min()):.1f}s to {float(ds.time_seconds.max()):.1f}s")
    print(f"X range: {float(ds.x.min()):.1f}m to {float(ds.x.max()):.1f}m")
    print(f"dt: {ds.attrs.get('dt', 'N/A')}s")
    print(f"dx: {ds.attrs.get('dx', 'N/A')}m")
    print(f"Outlier detection: {'Yes' if ds.attrs.get('outlier_detection_applied', 0) else 'No'}")
    print(f"Variables: {list(ds.data_vars)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize L2 wave-resolving timestack results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input L2 NetCDF file")
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help="Config file to determine output directory (uses plot_folder/level2/)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory for figures (default: config plot_folder/level2/ or input dir)")
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
        # Use config's plot_folder with level2/ subfolder
        from phase1 import load_config
        config = load_config(args.config)
        output_dir = config.plot_folder / "level2"
    else:
        # Fallback: figures/ subdirectory in input dir
        output_dir = args.input.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading: {args.input}")
    ds = load_l2_dataset(args.input)

    # Print summary
    if not args.no_summary:
        print_summary(ds)

    # Generate plots
    print(f"\nGenerating figures in: {output_dir}")
    plot_timestack(ds, output_dir, args.show)
    plot_profiles_and_timeseries(ds, output_dir, args.show)
    plot_intensity(ds, output_dir, args.show)
    plot_statistics(ds, output_dir, args.show)
    plot_wave_detection(ds, output_dir, args.show)

    ds.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
