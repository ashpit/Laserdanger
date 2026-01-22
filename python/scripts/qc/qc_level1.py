#!/usr/bin/env python3
"""
Comprehensive QC diagnostics for L1 (daily beach surface) NetCDF files.

Reads processFolder from config, finds L1_*.nc files, generates diagnostic
figures and reports to plotFolder/qc/level1/.

Usage:
    python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json
    python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json --date 2026-01-12
    python scripts/qc/qc_level1.py --config configs/do_livox_config_20260112.json --verbose
"""
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add code directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from phase1 import load_config, Config


@dataclass
class L1QCResult:
    """Results from L1 QC assessment."""
    file_path: Path
    date: str
    passed: bool
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, float]


def find_l1_files(process_folder: Path, date_filter: Optional[str] = None) -> List[Path]:
    """Find L1 NetCDF files in process folder."""
    # L1 files could be in process_folder directly or in a level1 subfolder
    search_paths = [process_folder, process_folder / "level1"]

    nc_files = []
    for search_path in search_paths:
        if search_path.exists():
            nc_files.extend(search_path.glob("L1_*.nc"))

    nc_files = sorted(set(nc_files))

    if date_filter:
        date_str = date_filter.replace("-", "")
        nc_files = [f for f in nc_files if date_str in f.name]

    return nc_files


def assess_l1_file(nc_path: Path, verbose: bool = False) -> Tuple[xr.Dataset, L1QCResult]:
    """
    Assess a single L1 NetCDF file for quality issues.

    Returns the dataset and QC results.
    """
    warnings = []
    errors = []
    stats = {}

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        return None, L1QCResult(
            file_path=nc_path,
            date=nc_path.stem.replace("L1_", "")[:8],
            passed=False,
            warnings=[],
            errors=[f"Failed to open file: {e}"],
            stats={},
        )

    # Extract date
    date_str = nc_path.stem.replace("L1_", "")[:8]

    # Check required variables
    required_vars = ["z_mean", "z_min", "z_max", "count"]
    for var in required_vars:
        if var not in ds:
            # Try alternate names
            alt_names = {"z_mean": "elevation", "z_min": "z_min", "z_max": "z_max"}
            if var in alt_names and alt_names[var] in ds:
                continue
            errors.append(f"Missing required variable: {var}")

    # Check coordinates
    if "x" not in ds.coords and "x_edge" not in ds.coords:
        errors.append("Missing x coordinate")
    if "y" not in ds.coords and "y_edge" not in ds.coords:
        errors.append("Missing y coordinate")

    # Get elevation data (try multiple names)
    elev_var = None
    for name in ["z_mean", "elevation", "z"]:
        if name in ds:
            elev_var = name
            break

    if elev_var:
        elev = ds[elev_var].values

        # Flatten for stats (handle 2D or 3D with time)
        if elev.ndim == 3:
            elev_flat = elev[0].flatten()  # First time step
        else:
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
            stats["z_median"] = float(np.median(valid_elev))

            # QC checks
            if valid_pct < 10:
                warnings.append(f"Low data coverage: {valid_pct:.1f}%")

            if stats["z_max"] > 50:
                warnings.append(f"Suspiciously high elevation: {stats['z_max']:.1f}m")

            if stats["z_min"] < -20:
                warnings.append(f"Suspiciously low elevation: {stats['z_min']:.1f}m")

            if stats["z_std"] > 5:
                warnings.append(f"High elevation variability: std={stats['z_std']:.2f}m")
        else:
            errors.append("No valid elevation data")

    # Check count variable
    if "count" in ds:
        count_data = ds["count"].values
        if count_data.ndim == 3:
            count_data = count_data[0]

        stats["max_point_count"] = int(np.nanmax(count_data))
        stats["mean_point_count"] = float(np.nanmean(count_data[count_data > 0])) if np.any(count_data > 0) else 0

        if stats["max_point_count"] < 10:
            warnings.append(f"Low point counts: max={stats['max_point_count']}")

    # Get grid dimensions
    x_coord = ds.coords.get("x", ds.coords.get("x_edge"))
    y_coord = ds.coords.get("y", ds.coords.get("y_edge"))

    if x_coord is not None and y_coord is not None:
        stats["x_range"] = float(x_coord.max() - x_coord.min())
        stats["y_range"] = float(y_coord.max() - y_coord.min())
        stats["n_x"] = len(x_coord)
        stats["n_y"] = len(y_coord)

        # Estimate bin size
        if len(x_coord) > 1:
            stats["bin_size_x"] = float(np.median(np.diff(x_coord.values)))
        if len(y_coord) > 1:
            stats["bin_size_y"] = float(np.median(np.diff(y_coord.values)))

    passed = len(errors) == 0

    return ds, L1QCResult(
        file_path=nc_path,
        date=date_str,
        passed=passed,
        warnings=warnings,
        errors=errors,
        stats=stats,
    )


def create_l1_dem_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create DEM visualization figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Get coordinates
    x = ds.coords.get("x", ds.coords.get("x_edge")).values
    y = ds.coords.get("y", ds.coords.get("y_edge")).values

    # Get elevation data
    elev_var = None
    for name in ["z_mean", "elevation", "z"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values
    if elev.ndim == 3:
        elev = elev[0]  # First time step

    # Plot 1: Elevation DEM
    ax = axes[0, 0]
    vmin, vmax = np.nanpercentile(elev, [2, 98])
    im = ax.pcolormesh(x, y, elev.T, vmin=vmin, vmax=vmax, cmap='terrain', shading='auto')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title_prefix}Elevation (z_mean)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Plot 2: z_max - z_min (roughness)
    ax = axes[0, 1]
    if "z_max" in ds and "z_min" in ds:
        z_max = ds["z_max"].values
        z_min = ds["z_min"].values
        if z_max.ndim == 3:
            z_max = z_max[0]
            z_min = z_min[0]
        z_range = z_max - z_min
        vmax_r = np.nanpercentile(z_range, 98)
        im = ax.pcolormesh(x, y, z_range.T, vmin=0, vmax=vmax_r, cmap='hot', shading='auto')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Elevation Range (z_max - z_min)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Range (m)')
    else:
        ax.text(0.5, 0.5, 'z_max/z_min not available', transform=ax.transAxes, ha='center')

    # Plot 3: Point count
    ax = axes[1, 0]
    if "count" in ds:
        count = ds["count"].values
        if count.ndim == 3:
            count = count[0]
        vmax_c = np.nanpercentile(count[count > 0], 98) if np.any(count > 0) else 1
        im = ax.pcolormesh(x, y, count.T, vmin=0, vmax=vmax_c, cmap='viridis', shading='auto')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Point Count per Cell')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Count')
    else:
        ax.text(0.5, 0.5, 'Count not available', transform=ax.transAxes, ha='center')

    # Plot 4: Standard deviation
    ax = axes[1, 1]
    if "z_std" in ds:
        z_std = ds["z_std"].values
        if z_std.ndim == 3:
            z_std = z_std[0]
        vmax_s = np.nanpercentile(z_std, 98)
        im = ax.pcolormesh(x, y, z_std.T, vmin=0, vmax=vmax_s, cmap='plasma', shading='auto')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Elevation Std Dev')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Std (m)')
    else:
        ax.text(0.5, 0.5, 'z_std not available', transform=ax.transAxes, ha='center')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_l1_profile_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create cross-shore and alongshore profile figures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get coordinates
    x = ds.coords.get("x", ds.coords.get("x_edge")).values
    y = ds.coords.get("y", ds.coords.get("y_edge")).values

    # Get elevation
    elev_var = None
    for name in ["z_mean", "elevation", "z"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values
    if elev.ndim == 3:
        elev = elev[0]

    # Cross-shore profiles (at different y positions)
    ax = axes[0, 0]
    n_profiles = 5
    y_indices = np.linspace(0, len(y) - 1, n_profiles, dtype=int)
    for i, yi in enumerate(y_indices):
        profile = elev[:, yi]
        ax.plot(x, profile, label=f'y={y[yi]:.0f}m', alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Cross-shore Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Alongshore profiles (at different x positions)
    ax = axes[0, 1]
    x_indices = np.linspace(0, len(x) - 1, n_profiles, dtype=int)
    for i, xi in enumerate(x_indices):
        profile = elev[xi, :]
        ax.plot(y, profile, label=f'x={x[xi]:.0f}m', alpha=0.7)
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Alongshore Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mean cross-shore profile
    ax = axes[1, 0]
    mean_profile = np.nanmean(elev, axis=1)
    std_profile = np.nanstd(elev, axis=1)
    ax.fill_between(x, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
    ax.plot(x, mean_profile, 'b-', lw=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Mean Cross-shore Profile (±1σ)')
    ax.grid(True, alpha=0.3)

    # Elevation histogram
    ax = axes[1, 1]
    elev_flat = elev.flatten()
    valid = elev_flat[~np.isnan(elev_flat)]
    if len(valid) > 0:
        ax.hist(valid, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(valid), color='r', linestyle='--', label=f'Mean: {np.mean(valid):.2f}m')
        ax.axvline(np.median(valid), color='g', linestyle='--', label=f'Median: {np.median(valid):.2f}m')
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Density')
    ax.set_title('Elevation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix}Profile Analysis', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_l1_coverage_figure(ds: xr.Dataset, output_path: Path, title_prefix: str = ""):
    """Create data coverage and quality figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get coordinates
    x = ds.coords.get("x", ds.coords.get("x_edge")).values
    y = ds.coords.get("y", ds.coords.get("y_edge")).values

    # Get elevation
    elev_var = None
    for name in ["z_mean", "elevation", "z"]:
        if name in ds:
            elev_var = name
            break

    if elev_var is None:
        plt.close(fig)
        return

    elev = ds[elev_var].values
    if elev.ndim == 3:
        elev = elev[0]

    # Data coverage map
    ax = axes[0]
    valid_mask = (~np.isnan(elev)).astype(float)
    im = ax.pcolormesh(x, y, valid_mask.T, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Data Coverage (green=valid)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Valid')

    # Coverage by cross-shore position
    ax = axes[1]
    valid_pct_x = 100 * np.sum(~np.isnan(elev), axis=1) / elev.shape[1]
    ax.barh(x, valid_pct_x, height=np.diff(x)[0] if len(x) > 1 else 0.1, alpha=0.7)
    ax.set_xlabel('Valid data (%)')
    ax.set_ylabel('X position (m)')
    ax.set_title('Coverage by Cross-shore Position')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # Coverage by alongshore position
    ax = axes[2]
    valid_pct_y = 100 * np.sum(~np.isnan(elev), axis=0) / elev.shape[0]
    ax.bar(y, valid_pct_y, width=np.diff(y)[0] if len(y) > 1 else 0.1, alpha=0.7)
    ax.set_xlabel('Y position (m)')
    ax.set_ylabel('Valid data (%)')
    ax.set_title('Coverage by Alongshore Position')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix}Data Coverage', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_qc_report(results: List[L1QCResult], output_path: Path):
    """Save QC results to JSON report."""
    report = {
        "generated": datetime.now().isoformat(),
        "level": "L1",
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


def print_qc_summary(results: List[L1QCResult]):
    """Print QC summary to console."""
    print(f"\n{'='*60}")
    print("L1 QC Summary")
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
            if "z_mean" in r.stats:
                print(f"    Z range: [{r.stats['z_min']:.2f}, {r.stats['z_max']:.2f}]m, mean={r.stats['z_mean']:.2f}m")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive QC diagnostics on L1 NetCDF files",
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
        help="Override output directory (default: plotFolder/qc/level1)",
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
        output_dir = config.plot_folder / "qc" / "level1"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find L1 files
    nc_files = find_l1_files(config.process_folder, args.date)

    if not nc_files:
        print(f"No L1 NetCDF files found in {config.process_folder}")
        sys.exit(1)

    print(f"Found {len(nc_files)} L1 file(s)")
    print(f"Output directory: {output_dir}")

    results = []

    for nc_path in nc_files:
        print(f"\nProcessing: {nc_path.name}")

        ds, qc_result = assess_l1_file(nc_path, verbose=args.verbose)
        results.append(qc_result)

        if ds is None:
            print(f"  ERROR: Could not open file")
            continue

        # Create output subdirectory for this date
        date_dir = output_dir / qc_result.date
        date_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_figures:
            title_prefix = f"L1 {qc_result.date} - "

            # Generate figures
            dem_path = date_dir / "dem_overview.png"
            create_l1_dem_figure(ds, dem_path, title_prefix)
            print(f"  Saved: {dem_path.name}")

            profile_path = date_dir / "profiles.png"
            create_l1_profile_figure(ds, profile_path, title_prefix)
            print(f"  Saved: {profile_path.name}")

            coverage_path = date_dir / "coverage.png"
            create_l1_coverage_figure(ds, coverage_path, title_prefix)
            print(f"  Saved: {coverage_path.name}")

        ds.close()

    # Save consolidated report
    report_path = output_dir / "qc_report.json"
    save_qc_report(results, report_path)
    print(f"\nSaved report: {report_path}")

    # Print summary
    print_qc_summary(results)

    # Exit with appropriate code
    if all(r.passed for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
