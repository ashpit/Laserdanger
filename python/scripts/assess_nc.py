#!/usr/bin/env python3
"""
Assess L2 NetCDF files - show dimensions, fields, statistics, and generate summary figures.

Usage:
    python scripts/assess_nc.py /path/to/L2_file.nc
    python scripts/assess_nc.py /path/to/level2/  # Process all NC files in directory
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def assess_nc_file(nc_path: Path, output_dir: Path, verbose: bool = True):
    """
    Assess a single NetCDF file and generate summary figures.

    Parameters
    ----------
    nc_path : Path
        Path to the NetCDF file
    output_dir : Path
        Directory for output figures
    verbose : bool
        Print detailed info to console
    """
    print(f"\n{'='*60}")
    print(f"Assessing: {nc_path.name}")
    print(f"{'='*60}")

    ds = xr.open_dataset(nc_path)

    # Extract date from filename or data
    if 'time' in ds.coords:
        data_date = str(ds.time.values[0])[:10]
    else:
        # Try to parse from filename
        data_date = nc_path.stem.replace('L2_', '')[:8]
        data_date = f"{data_date[:4]}-{data_date[4:6]}-{data_date[6:8]}"

    # Create output directory for this date
    date_dir = output_dir / data_date.replace('-', '')
    date_dir.mkdir(parents=True, exist_ok=True)

    # --- Print Dataset Info ---
    print(f"\nFile: {nc_path}")
    print(f"Size: {nc_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Date: {data_date}")

    print(f"\n--- Dimensions ---")
    for dim in ds.sizes:
        print(f"  {dim}: {ds.sizes[dim]}")

    print(f"\n--- Coordinates ---")
    for name, coord in ds.coords.items():
        if coord.size > 0:
            if np.issubdtype(coord.dtype, np.datetime64):
                print(f"  {name}: {coord.dtype}, [{coord.values[0]} ... {coord.values[-1]}]")
            else:
                print(f"  {name}: {coord.dtype}, range=[{float(coord.min()):.2f}, {float(coord.max()):.2f}]")
        else:
            print(f"  {name}: {coord.dtype}, empty")

    print(f"\n--- Data Variables ---")
    for name, var in ds.data_vars.items():
        valid_count = np.sum(~np.isnan(var.values))
        total = var.size
        valid_pct = 100 * valid_count / total if total > 0 else 0

        if valid_count > 0:
            vmin = float(np.nanmin(var.values))
            vmax = float(np.nanmax(var.values))
            vmean = float(np.nanmean(var.values))
            print(f"  {name}: {var.dims}, dtype={var.dtype}")
            print(f"       range=[{vmin:.3f}, {vmax:.3f}], mean={vmean:.3f}, valid={valid_pct:.1f}%")
        else:
            print(f"  {name}: {var.dims}, dtype={var.dtype}, all NaN")

    print(f"\n--- Attributes ---")
    for attr, val in ds.attrs.items():
        print(f"  {attr}: {val}")

    # --- Generate Figures ---
    print(f"\n--- Generating Figures ---")

    # Figure 1: Timestack plot of elevation
    fig1_path = date_dir / "elevation_timestack.png"
    create_timestack_figure(ds, fig1_path)
    print(f"  Saved: {fig1_path}")

    # Figure 2: Statistics summary
    fig2_path = date_dir / "statistics_summary.png"
    create_statistics_figure(ds, fig2_path)
    print(f"  Saved: {fig2_path}")

    # Figure 3: Data coverage plot
    fig3_path = date_dir / "data_coverage.png"
    create_coverage_figure(ds, fig3_path)
    print(f"  Saved: {fig3_path}")

    # Save text report
    report_path = date_dir / "assessment_report.txt"
    save_text_report(ds, nc_path, report_path)
    print(f"  Saved: {report_path}")

    ds.close()

    return date_dir


def create_timestack_figure(ds: xr.Dataset, output_path: Path):
    """Create timestack visualization of elevation data."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Subsample for visualization if needed
    x = ds.x.values
    time = ds.time.values

    # Determine subsampling factor for manageable plot
    max_time_points = 5000
    time_step = max(1, len(time) // max_time_points)

    elev = ds.elevation.values[:, ::time_step]
    time_sub = time[::time_step]

    # Plot 1: Elevation timestack
    ax1 = axes[0]
    vmin, vmax = np.nanpercentile(elev, [2, 98])
    im1 = ax1.pcolormesh(time_sub, x, elev,
                         vmin=vmin, vmax=vmax,
                         cmap='viridis', shading='auto')
    ax1.set_ylabel('Cross-shore position (m)')
    ax1.set_title('Elevation Z(x,t)')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')

    # Plot 2: Intensity timestack (if available)
    ax2 = axes[1]
    if 'intensity' in ds:
        inten = ds.intensity.values[:, ::time_step]
        vmin_i, vmax_i = np.nanpercentile(inten[~np.isnan(inten)], [2, 98]) if np.any(~np.isnan(inten)) else (0, 1)
        im2 = ax2.pcolormesh(time_sub, x, inten,
                             vmin=vmin_i, vmax=vmax_i,
                             cmap='gray', shading='auto')
        ax2.set_ylabel('Cross-shore position (m)')
        ax2.set_title('Intensity I(x,t)')
        plt.colorbar(im2, ax=ax2, label='Intensity')
    else:
        ax2.text(0.5, 0.5, 'No intensity data', transform=ax2.transAxes, ha='center')

    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_statistics_figure(ds: xr.Dataset, output_path: Path):
    """Create statistics summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = ds.x.values

    # Row 1: Temporal statistics along x
    # Mean elevation profile
    ax = axes[0, 0]
    z_mean = np.nanmean(ds.elevation.values, axis=1)
    z_std = np.nanstd(ds.elevation.values, axis=1)
    ax.fill_between(x, z_mean - z_std, z_mean + z_std, alpha=0.3)
    ax.plot(x, z_mean, 'b-', lw=1.5)
    ax.set_xlabel('Cross-shore (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Mean Elevation Profile')
    ax.grid(True, alpha=0.3)

    # Min/Max range
    ax = axes[0, 1]
    z_min = np.nanmin(ds.elevation.values, axis=1)
    z_max = np.nanmax(ds.elevation.values, axis=1)
    ax.fill_between(x, z_min, z_max, alpha=0.3, label='Min-Max range')
    ax.plot(x, z_mean, 'b-', lw=1, label='Mean')
    ax.set_xlabel('Cross-shore (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Range')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Standard deviation profile
    ax = axes[0, 2]
    ax.plot(x, z_std, 'r-', lw=1.5)
    ax.set_xlabel('Cross-shore (m)')
    ax.set_ylabel('Std Dev (m)')
    ax.set_title('Temporal Variability')
    ax.grid(True, alpha=0.3)

    # Row 2: Histograms and time series
    # Elevation histogram
    ax = axes[1, 0]
    elev_flat = ds.elevation.values.flatten()
    elev_valid = elev_flat[~np.isnan(elev_flat)]
    if len(elev_valid) > 0:
        ax.hist(elev_valid, bins=100, density=True, alpha=0.7)
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Density')
    ax.set_title('Elevation Distribution')
    ax.grid(True, alpha=0.3)

    # Point count over time
    ax = axes[1, 1]
    time = ds.time.values
    count_per_time = np.nansum(ds['count'].values, axis=0) if 'count' in ds else np.sum(~np.isnan(ds.elevation.values), axis=0)
    ax.plot(time, count_per_time, 'g-', lw=0.5, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Point count')
    ax.set_title('Data Density Over Time')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    # Intensity histogram (if available)
    ax = axes[1, 2]
    if 'intensity' in ds:
        inten_flat = ds.intensity.values.flatten()
        inten_valid = inten_flat[~np.isnan(inten_flat)]
        if len(inten_valid) > 0:
            ax.hist(inten_valid, bins=100, density=True, alpha=0.7, color='gray')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title('Intensity Distribution')
    else:
        ax.text(0.5, 0.5, 'No intensity data', transform=ax.transAxes, ha='center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_coverage_figure(ds: xr.Dataset, output_path: Path):
    """Create data coverage/validity figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = ds.x.values
    time = ds.time.values

    # Subsample for visualization
    max_time_points = 5000
    time_step = max(1, len(time) // max_time_points)

    elev = ds.elevation.values[:, ::time_step]
    time_sub = time[::time_step]

    # Valid data mask
    ax = axes[0]
    valid_mask = (~np.isnan(elev)).astype(float)
    im = ax.pcolormesh(time_sub, x, valid_mask,
                       cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cross-shore position (m)')
    ax.set_title('Data Validity (green=valid, red=NaN)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.colorbar(im, ax=ax, label='Valid')

    # Coverage percentage by x position
    ax = axes[1]
    valid_pct_x = 100 * np.sum(~np.isnan(ds.elevation.values), axis=1) / ds.elevation.shape[1]
    ax.barh(x, valid_pct_x, height=np.diff(x)[0] if len(x) > 1 else 0.1, alpha=0.7)
    ax.set_xlabel('Valid data (%)')
    ax.set_ylabel('Cross-shore position (m)')
    ax.set_title('Data Coverage by Position')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_text_report(ds: xr.Dataset, nc_path: Path, output_path: Path):
    """Save detailed text report of the NetCDF file."""
    with open(output_path, 'w') as f:
        f.write(f"L2 NetCDF Assessment Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write(f"File: {nc_path}\n")
        f.write(f"Size: {nc_path.stat().st_size / (1024**3):.2f} GB\n\n")

        f.write(f"Dimensions\n")
        f.write(f"{'-'*40}\n")
        for dim in ds.sizes:
            f.write(f"  {dim}: {ds.sizes[dim]}\n")

        f.write(f"\nCoordinates\n")
        f.write(f"{'-'*40}\n")
        for name, coord in ds.coords.items():
            if coord.size > 0:
                if np.issubdtype(coord.dtype, np.datetime64):
                    f.write(f"  {name}:\n")
                    f.write(f"    dtype: {coord.dtype}\n")
                    f.write(f"    start: {coord.values[0]}\n")
                    f.write(f"    end:   {coord.values[-1]}\n")
                else:
                    f.write(f"  {name}:\n")
                    f.write(f"    dtype: {coord.dtype}\n")
                    f.write(f"    min:   {float(coord.min()):.6f}\n")
                    f.write(f"    max:   {float(coord.max()):.6f}\n")

        f.write(f"\nData Variables\n")
        f.write(f"{'-'*40}\n")
        for name, var in ds.data_vars.items():
            valid_count = np.sum(~np.isnan(var.values))
            total = var.size
            valid_pct = 100 * valid_count / total if total > 0 else 0

            f.write(f"  {name}:\n")
            f.write(f"    shape: {var.dims} = {var.shape}\n")
            f.write(f"    dtype: {var.dtype}\n")
            f.write(f"    valid: {valid_count:,} / {total:,} ({valid_pct:.1f}%)\n")

            if valid_count > 0:
                f.write(f"    min:   {float(np.nanmin(var.values)):.6f}\n")
                f.write(f"    max:   {float(np.nanmax(var.values)):.6f}\n")
                f.write(f"    mean:  {float(np.nanmean(var.values)):.6f}\n")
                f.write(f"    std:   {float(np.nanstd(var.values)):.6f}\n")

        f.write(f"\nAttributes\n")
        f.write(f"{'-'*40}\n")
        for attr, val in ds.attrs.items():
            f.write(f"  {attr}: {val}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Assess L2 NetCDF files and generate summary figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to NC file or directory containing NC files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for figures (default: figures/l2)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed info"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / "figures" / "l2"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find NC files
    input_path = args.input
    if input_path.is_file():
        nc_files = [input_path]
    elif input_path.is_dir():
        nc_files = sorted(input_path.glob("*.nc"))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not nc_files:
        print(f"No NC files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(nc_files)} NC file(s)")
    print(f"Output directory: {output_dir}")

    for nc_path in nc_files:
        try:
            assess_nc_file(nc_path, output_dir, verbose=args.verbose)
        except Exception as e:
            print(f"Error processing {nc_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
