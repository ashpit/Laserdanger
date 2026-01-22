#!/usr/bin/env python3
"""
Comprehensive QC diagnostics for L2 (wave-resolving timestack) NetCDF files.

Reads processFolder from config, finds L2_*.nc files, generates diagnostic
figures and reports to plotFolder/qc/level2/.

Usage:
    python scripts/qc/qc_level2.py --config configs/towr_livox_config_20260120.json
    python scripts/qc/qc_level2.py --config configs/towr_livox_config_20260120.json --date 2026-01-20
    python scripts/qc/qc_level2.py --config configs/towr_livox_config_20260120.json --verbose
"""
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add code directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from phase1 import load_config, Config


@dataclass
class L2QCResult:
    """Results from L2 QC assessment."""
    file_path: Path
    date: str
    passed: bool
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, float]


def find_l2_files(process_folder: Path, date_filter: Optional[str] = None) -> List[Path]:
    """Find L2 NetCDF files in process folder."""
    # L2 files could be in process_folder directly or in a level2 subfolder
    search_paths = [process_folder, process_folder / "level2"]

    nc_files = []
    for search_path in search_paths:
        if search_path.exists():
            nc_files.extend(search_path.glob("L2_*.nc"))

    nc_files = sorted(set(nc_files))

    if date_filter:
        date_str = date_filter.replace("-", "")
        nc_files = [f for f in nc_files if date_str in f.name]

    return nc_files


def assess_l2_file(nc_path: Path, verbose: bool = False) -> Tuple[xr.Dataset, L2QCResult]:
    """
    Assess a single L2 NetCDF file for quality issues.

    Returns the dataset and QC results.
    """
    warnings = []
    errors = []
    stats = {}

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        return None, L2QCResult(
            file_path=nc_path,
            date=nc_path.stem.replace("L2_", "")[:8],
            passed=False,
            warnings=[],
            errors=[f"Failed to open file: {e}"],
            stats={},
        )

    # Extract date
    date_str = nc_path.stem.replace("L2_", "")[:8]

    # File size
    stats["file_size_gb"] = nc_path.stat().st_size / (1024**3)

    # Check required variables
    required_vars = ["elevation"]
    alt_names = {"elevation": ["Z", "z_mean"]}

    for var in required_vars:
        if var not in ds:
            found = False
            for alt in alt_names.get(var, []):
                if alt in ds:
                    found = True
                    break
            if not found:
                errors.append(f"Missing required variable: {var}")

    # Check coordinates
    if "x" not in ds.coords:
        errors.append("Missing x coordinate")
    if "time" not in ds.coords:
        errors.append("Missing time coordinate")

    # Get elevation data
    elev_var = None
    for name in ["elevation", "Z", "z_mean"]:
        if name in ds:
            elev_var = name
            break

    if elev_var:
        elev = ds[elev_var].values

        # Shape should be (n_x, n_t) or (n_t, n_x)
        stats["shape"] = list(elev.shape)

        # Flatten for stats
        elev_flat = elev.flatten()
        valid_mask = ~np.isnan(elev_flat)
        valid_count = np.sum(valid_mask)
        total_count = len(elev_flat)
        valid_pct = 100 * valid_count / total_count if total_count > 0 else 0

        stats["valid_percent"] = valid_pct
        stats["total_cells"] = total_count
        stats["valid_cells"] = int(valid_count)

        if valid_count > 0:
            valid_elev = elev_flat[valid_mask]
            stats["z_min"] = float(np.min(valid_elev))
            stats["z_max"] = float(np.max(valid_elev))
            stats["z_mean"] = float(np.mean(valid_elev))
            stats["z_std"] = float(np.std(valid_elev))

            # QC checks
            if valid_pct < 10:
                warnings.append(f"Low data coverage: {valid_pct:.1f}%")

            if stats["z_max"] > 20:
                warnings.append(f"Suspiciously high elevation: {stats['z_max']:.1f}m")

            if stats["z_min"] < -10:
                warnings.append(f"Suspiciously low elevation: {stats['z_min']:.1f}m")

            # Check for excessive NaN gaps in time
            nan_pct_per_time = 100 * np.sum(np.isnan(elev), axis=0) / elev.shape[0]
            if np.max(nan_pct_per_time) > 90:
                warnings.append("Some time steps have >90% NaN values")
        else:
            errors.append("No valid elevation data")

    # Check intensity
    if "intensity" in ds:
        intensity = ds["intensity"].values
        valid_int = intensity[~np.isnan(intensity)]
        if len(valid_int) > 0:
            stats["intensity_min"] = float(np.min(valid_int))
            stats["intensity_max"] = float(np.max(valid_int))
            stats["intensity_mean"] = float(np.mean(valid_int))
    else:
        warnings.append("No intensity data available")

    # Time coordinate stats
    if "time" in ds.coords:
        time = ds.time.values
        stats["n_time_steps"] = len(time)

        if len(time) > 1:
            # Calculate temporal resolution
            if np.issubdtype(time.dtype, np.datetime64):
                dt = np.diff(time).astype('timedelta64[ms]').astype(float) / 1000.0
            else:
                dt = np.diff(time)

            stats["dt_median"] = float(np.median(dt))
            stats["dt_min"] = float(np.min(dt))
            stats["dt_max"] = float(np.max(dt))

            # Check for irregular sampling
            if stats["dt_max"] > 2 * stats["dt_median"]:
                warnings.append(f"Irregular time sampling detected (max gap: {stats['dt_max']:.1f}s)")

            # Duration
            if np.issubdtype(time.dtype, np.datetime64):
                duration = (time[-1] - time[0]).astype('timedelta64[s]').astype(float)
            else:
                duration = time[-1] - time[0]
            stats["duration_hours"] = duration / 3600

    # X coordinate stats
    if "x" in ds.coords:
        x = ds.x.values
        stats["n_x_bins"] = len(x)
        stats["x_min"] = float(np.min(x))
        stats["x_max"] = float(np.max(x))
        stats["x_range"] = float(np.max(x) - np.min(x))

        if len(x) > 1:
            stats["dx"] = float(np.median(np.diff(x)))

    # Check for outlier mask
    if "outlier_mask" in ds:
        mask = ds["outlier_mask"].values
        outlier_pct = 100 * np.sum(mask) / mask.size
        stats["outlier_percent"] = outlier_pct
        if outlier_pct > 10:
            warnings.append(f"High outlier percentage: {outlier_pct:.1f}%")

    passed = len(errors) == 0

    return ds, L2QCResult(
        file_path=nc_path,
        date=date_str,
        passed=passed,
        warnings=warnings,
        errors=errors,
        stats=stats,
    )


def create_l2_timestack_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create timestack visualization figure."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x = ds.x.values
    time = ds.time.values

    # Subsample for visualization if needed
    max_time_points = 5000
    time_step = max(1, len(time) // max_time_points)

    # Get elevation
    elev_var = None
    for name in ["elevation", "Z", "z_mean"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values[:, ::time_step]
    time_sub = time[::time_step]

    # Plot 1: Elevation timestack
    ax = axes[0]
    vmin, vmax = np.nanpercentile(elev, [2, 98])
    im = ax.pcolormesh(time_sub, x, elev, vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
    ax.set_ylabel('Cross-shore position (m)')
    ax.set_title(f'{title_prefix}Elevation Z(x,t)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Plot 2: Intensity (if available)
    ax = axes[1]
    if "intensity" in ds:
        inten = ds["intensity"].values[:, ::time_step]
        valid_inten = inten[~np.isnan(inten)]
        if len(valid_inten) > 0:
            vmin_i, vmax_i = np.nanpercentile(valid_inten, [2, 98])
            im = ax.pcolormesh(time_sub, x, inten, vmin=vmin_i, vmax=vmax_i, cmap='gray', shading='auto')
            ax.set_ylabel('Cross-shore position (m)')
            ax.set_title('Intensity I(x,t)')
            plt.colorbar(im, ax=ax, label='Intensity')
        else:
            ax.text(0.5, 0.5, 'No valid intensity data', transform=ax.transAxes, ha='center')
    else:
        ax.text(0.5, 0.5, 'Intensity not available', transform=ax.transAxes, ha='center')

    ax.set_xlabel('Time')
    if np.issubdtype(time_sub.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_l2_statistics_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create statistics summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = ds.x.values
    time = ds.time.values

    # Get elevation
    elev_var = None
    for name in ["elevation", "Z", "z_mean"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values

    # Row 1: Temporal statistics along x
    # Mean elevation profile
    ax = axes[0, 0]
    z_mean = np.nanmean(elev, axis=1)
    z_std = np.nanstd(elev, axis=1)
    ax.fill_between(x, z_mean - z_std, z_mean + z_std, alpha=0.3)
    ax.plot(x, z_mean, 'b-', lw=1.5)
    ax.set_xlabel('Cross-shore (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Mean Elevation Profile')
    ax.grid(True, alpha=0.3)

    # Min/Max range
    ax = axes[0, 1]
    z_min = np.nanmin(elev, axis=1)
    z_max = np.nanmax(elev, axis=1)
    ax.fill_between(x, z_min, z_max, alpha=0.3, label='Min-Max range')
    ax.plot(x, z_mean, 'b-', lw=1, label='Mean')
    ax.set_xlabel('Cross-shore (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Range')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Standard deviation profile (temporal variability)
    ax = axes[0, 2]
    ax.plot(x, z_std, 'r-', lw=1.5)
    ax.set_xlabel('Cross-shore (m)')
    ax.set_ylabel('Std Dev (m)')
    ax.set_title('Temporal Variability (wave signal)')
    ax.grid(True, alpha=0.3)

    # Row 2: Distributions and time series
    # Elevation histogram
    ax = axes[1, 0]
    elev_flat = elev.flatten()
    elev_valid = elev_flat[~np.isnan(elev_flat)]
    if len(elev_valid) > 0:
        ax.hist(elev_valid, bins=100, density=True, alpha=0.7)
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Density')
    ax.set_title('Elevation Distribution')
    ax.grid(True, alpha=0.3)

    # Valid data count over time
    ax = axes[1, 1]
    count_per_time = np.sum(~np.isnan(elev), axis=0)
    ax.plot(time, count_per_time, 'g-', lw=0.5, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Valid cell count')
    ax.set_title('Data Density Over Time')
    if np.issubdtype(time.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    # Intensity histogram (if available)
    ax = axes[1, 2]
    if "intensity" in ds:
        inten_flat = ds["intensity"].values.flatten()
        inten_valid = inten_flat[~np.isnan(inten_flat)]
        if len(inten_valid) > 0:
            ax.hist(inten_valid, bins=100, density=True, alpha=0.7, color='gray')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title('Intensity Distribution')
    else:
        ax.text(0.5, 0.5, 'Intensity not available', transform=ax.transAxes, ha='center')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix}Statistics Summary', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_l2_coverage_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create data coverage/validity figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = ds.x.values
    time = ds.time.values

    # Subsample for visualization
    max_time_points = 5000
    time_step = max(1, len(time) // max_time_points)

    # Get elevation
    elev_var = None
    for name in ["elevation", "Z", "z_mean"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values[:, ::time_step]
    time_sub = time[::time_step]

    # Valid data mask
    ax = axes[0]
    valid_mask = (~np.isnan(elev)).astype(float)
    im = ax.pcolormesh(time_sub, x, valid_mask, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cross-shore position (m)')
    ax.set_title('Data Validity (green=valid, red=NaN)')
    if np.issubdtype(time_sub.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.colorbar(im, ax=ax, label='Valid')

    # Coverage percentage by x position
    ax = axes[1]
    full_elev = ds[elev_var].values
    valid_pct_x = 100 * np.sum(~np.isnan(full_elev), axis=1) / full_elev.shape[1]
    ax.barh(x, valid_pct_x, height=np.diff(x)[0] if len(x) > 1 else 0.1, alpha=0.7)
    ax.set_xlabel('Valid data (%)')
    ax.set_ylabel('Cross-shore position (m)')
    ax.set_title('Coverage by Position')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # Coverage percentage over time
    ax = axes[2]
    valid_pct_t = 100 * np.sum(~np.isnan(full_elev), axis=0) / full_elev.shape[0]
    # Subsample for plotting
    ax.plot(time[::max(1, len(time)//1000)], valid_pct_t[::max(1, len(time)//1000)], 'b-', lw=0.5, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Valid data (%)')
    ax.set_title('Coverage Over Time')
    ax.set_ylim(0, 100)
    if np.issubdtype(time.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix}Data Coverage', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_l2_sample_traces_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create sample time series traces at different cross-shore positions."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    x = ds.x.values
    time = ds.time.values

    # Get elevation
    elev_var = None
    for name in ["elevation", "Z", "z_mean"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values

    # Select a time window (first 10 minutes if datetime)
    if np.issubdtype(time.dtype, np.datetime64):
        window_ns = 10 * 60 * 1e9  # 10 minutes in nanoseconds
        t_start = time[0]
        t_end = t_start + np.timedelta64(int(window_ns), 'ns')
        time_mask = time <= t_end
    else:
        # Use first 600 seconds
        time_mask = time <= (time[0] + 600)

    time_window = time[time_mask]
    elev_window = elev[:, time_mask]

    # Select positions: seaward, mid, landward
    n_x = len(x)
    positions = [
        ("Seaward", int(n_x * 0.2)),
        ("Mid-beach", int(n_x * 0.5)),
        ("Landward", int(n_x * 0.8)),
    ]

    for ax, (label, idx) in zip(axes, positions):
        if idx < n_x:
            trace = elev_window[idx, :]
            ax.plot(time_window, trace, 'b-', lw=0.5)
            ax.set_ylabel('Elevation (m)')
            ax.set_title(f'{label} (x={x[idx]:.1f}m)')
            ax.grid(True, alpha=0.3)

            # Add mean line
            mean_val = np.nanmean(trace)
            ax.axhline(mean_val, color='r', linestyle='--', alpha=0.5, label=f'mean={mean_val:.2f}m')
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time')
    if np.issubdtype(time_window.dtype, np.datetime64):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.suptitle(f'{title_prefix}Sample Time Series (first 10 min)', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_qc_report(results: List[L2QCResult], output_path: Path):
    """Save QC results to JSON report."""
    report = {
        "generated": datetime.now().isoformat(),
        "level": "L2",
        "summary": {
            "total_files": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "with_warnings": sum(1 for r in results if r.warnings),
        },
        "files": [],
    }

    for r in results:
        report["files"].append({
            "file": str(r.file_path.name),
            "date": r.date,
            "passed": r.passed,
            "warnings": r.warnings,
            "errors": r.errors,
            "stats": r.stats,
        })

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def save_text_report(results: List[L2QCResult], ds_list: List[xr.Dataset], output_path: Path):
    """Save detailed text report."""
    with open(output_path, 'w') as f:
        f.write("L2 NetCDF QC Assessment Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for r, ds in zip(results, ds_list):
            f.write(f"\nFile: {r.file_path.name}\n")
            f.write(f"{'-'*40}\n")

            if ds is None:
                f.write("  ERROR: Could not open file\n")
                for err in r.errors:
                    f.write(f"  {err}\n")
                continue

            f.write(f"Status: {'PASS' if r.passed else 'FAIL'}\n")
            f.write(f"Date: {r.date}\n")
            f.write(f"Size: {r.stats.get('file_size_gb', 0):.2f} GB\n\n")

            f.write("Dimensions:\n")
            for dim in ds.sizes:
                f.write(f"  {dim}: {ds.sizes[dim]}\n")

            f.write("\nCoordinates:\n")
            for name, coord in ds.coords.items():
                if coord.size > 0:
                    if np.issubdtype(coord.dtype, np.datetime64):
                        f.write(f"  {name}: {coord.values[0]} to {coord.values[-1]}\n")
                    else:
                        f.write(f"  {name}: [{float(coord.min()):.2f}, {float(coord.max()):.2f}]\n")

            f.write("\nStatistics:\n")
            for key, val in r.stats.items():
                if isinstance(val, float):
                    f.write(f"  {key}: {val:.4f}\n")
                else:
                    f.write(f"  {key}: {val}\n")

            if r.warnings:
                f.write("\nWarnings:\n")
                for warn in r.warnings:
                    f.write(f"  - {warn}\n")

            if r.errors:
                f.write("\nErrors:\n")
                for err in r.errors:
                    f.write(f"  - {err}\n")

            f.write("\n")


def print_qc_summary(results: List[L2QCResult]):
    """Print QC summary to console."""
    print(f"\n{'='*60}")
    print("L2 QC Summary")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    with_warnings = sum(1 for r in results if r.warnings)

    print(f"Total files:    {len(results)}")
    print(f"Passed:         {passed}")
    print(f"Failed:         {failed}")
    print(f"With warnings:  {with_warnings}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        warn_str = f" ({len(r.warnings)} warnings)" if r.warnings else ""
        print(f"\n  {r.file_path.name}: {status}{warn_str}")

        if r.errors:
            for err in r.errors:
                print(f"    ERROR: {err}")
        if r.warnings:
            for warn in r.warnings:
                print(f"    WARN:  {warn}")

        if r.stats:
            print(f"    Coverage: {r.stats.get('valid_percent', 0):.1f}%")
            print(f"    Duration: {r.stats.get('duration_hours', 0):.1f} hours")
            if "z_mean" in r.stats:
                print(f"    Z range: [{r.stats['z_min']:.2f}, {r.stats['z_max']:.2f}]m")
            if "dt_median" in r.stats:
                print(f"    Temporal res: {r.stats['dt_median']:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive QC diagnostics on L2 NetCDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to project config file (JSON)",
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Process only this date (YYYY-MM-DD or YYYYMMDD)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Override output directory (default: plotFolder/qc/level2)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (report only)",
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.plot_folder / "qc" / "level2"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find L2 files
    nc_files = find_l2_files(config.process_folder, args.date)

    if not nc_files:
        print(f"No L2 NetCDF files found in {config.process_folder}")
        sys.exit(1)

    print(f"Found {len(nc_files)} L2 file(s)")
    print(f"Output directory: {output_dir}")

    results = []
    datasets = []

    for nc_path in nc_files:
        print(f"\nProcessing: {nc_path.name}")

        ds, qc_result = assess_l2_file(nc_path, verbose=args.verbose)
        results.append(qc_result)
        datasets.append(ds)

        if ds is None:
            print(f"  ERROR: Could not open file")
            continue

        # Create output subdirectory for this date
        date_dir = output_dir / qc_result.date
        date_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_figures:
            title_prefix = f"L2 {qc_result.date} - "

            # Generate figures
            timestack_path = date_dir / "timestack.png"
            create_l2_timestack_figure(ds, timestack_path, title_prefix)
            print(f"  Saved: {timestack_path.name}")

            stats_path = date_dir / "statistics.png"
            create_l2_statistics_figure(ds, stats_path, title_prefix)
            print(f"  Saved: {stats_path.name}")

            coverage_path = date_dir / "coverage.png"
            create_l2_coverage_figure(ds, coverage_path, title_prefix)
            print(f"  Saved: {coverage_path.name}")

            traces_path = date_dir / "sample_traces.png"
            create_l2_sample_traces_figure(ds, traces_path, title_prefix)
            print(f"  Saved: {traces_path.name}")

        ds.close()

    # Save consolidated JSON report
    report_path = output_dir / "qc_report.json"
    save_qc_report(results, report_path)
    print(f"\nSaved JSON report: {report_path}")

    # Save text report
    text_report_path = output_dir / "qc_report.txt"
    # Re-open datasets for text report
    datasets_for_report = []
    for nc_path in nc_files:
        try:
            datasets_for_report.append(xr.open_dataset(nc_path))
        except:
            datasets_for_report.append(None)
    save_text_report(results, datasets_for_report, text_report_path)
    for ds in datasets_for_report:
        if ds is not None:
            ds.close()
    print(f"Saved text report: {text_report_path}")

    # Print summary
    print_qc_summary(results)

    # Exit with appropriate code
    if all(r.passed for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
