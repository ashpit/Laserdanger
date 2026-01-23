#!/usr/bin/env python3
"""
Create runup timestack plots from L2 NetCDF files.

For each burst (contiguous region with valid data), creates a scatter plot showing:
  - X-axis: time in seconds
  - Y-axis: runup cross-shore position (m)
  - Points colored by runup elevation

Usage:
    # Process all L2 files from config
    python scripts/visualization/plot_runup_timestack.py --config configs/towr_livox_config_20260120.json

    # Process single file
    python scripts/visualization/plot_runup_timestack.py --config configs/towr_livox_config_20260120.json --input L2_20260120.nc

Options:
    --config PATH       Path to config file (required)
    --input FILE        Process single file (filename only, looked up in processFolder/level2/)
    --output PATH       Output directory for PNGs (default: plotFolder/level2/pngs/)
    --dpi DPI           Resolution (default: 150)
    --figsize W H       Figure size in inches (default: 12 6)
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from phase1 import load_config
import runup


def find_valid_bursts(
    Z_xt: np.ndarray,
    time_sec: np.ndarray,
    min_duration_sec: float = 60.0,
) -> List[dict]:
    """
    Find contiguous regions with valid data (bursts).

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation timestack
    time_sec : array (n_t,)
        Time in seconds
    min_duration_sec : float
        Minimum burst duration to include

    Returns
    -------
    List of burst dictionaries with start/end indices and times
    """
    # A column is valid if it has at least some non-NaN values
    valid_cols = ~np.all(np.isnan(Z_xt), axis=0)

    if not valid_cols.any():
        return []

    # Find transitions
    diff = np.diff(valid_cols.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if valid_cols[0]:
        starts = np.concatenate([[0], starts])
    if valid_cols[-1]:
        ends = np.concatenate([ends, [len(valid_cols)]])

    bursts = []
    for s, e in zip(starts, ends):
        duration = time_sec[e-1] - time_sec[s]
        if duration >= min_duration_sec:
            bursts.append({
                'start_idx': s,
                'end_idx': e,
                'start_time': time_sec[s],
                'end_time': time_sec[e-1],
                'duration_sec': duration,
                'n_frames': e - s,
            })
    return bursts


def detect_runup_for_burst(
    Z_xt: np.ndarray,
    x1d: np.ndarray,
    time_sec: np.ndarray,
    I_xt: Optional[np.ndarray] = None,
    threshold: float = 0.1,
    ig_length: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect runup position for each timestep.

    Returns
    -------
    X_runup : array (n_t,)
        Cross-shore runup position
    Z_runup : array (n_t,)
        Runup elevation
    """
    result = runup.compute_runup_stats(
        Z_xt, x1d, time_sec,
        I_xt=I_xt,
        threshold=threshold,
        ig_length=ig_length,
    )

    return result.timeseries.X_runup, result.timeseries.Z_runup


def create_runup_timestack_plot(
    time_burst: np.ndarray,
    X_runup: np.ndarray,
    Z_runup: np.ndarray,
    output_path: Path,
    burst_info: dict,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 6),
) -> bool:
    """
    Create runup timestack scatter plot for a single burst.

    Parameters
    ----------
    time_burst : array (n_frames,)
        Time values for this burst (seconds)
    X_runup : array (n_frames,)
        Cross-shore runup position
    Z_runup : array (n_frames,)
        Runup elevation
    output_path : Path
        Output PNG path
    burst_info : dict
        Burst metadata
    dpi : int
        Resolution
    figsize : tuple
        Figure size (width, height) in inches

    Returns
    -------
    bool : True on success
    """
    # Filter to valid points
    valid = ~np.isnan(X_runup) & ~np.isnan(Z_runup)
    if valid.sum() < 5:
        return False

    t_valid = time_burst[valid]
    x_valid = X_runup[valid]
    z_valid = Z_runup[valid]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot with elevation colormap
    scatter = ax.scatter(
        t_valid, x_valid,
        c=z_valid,
        cmap='viridis',
        s=10,
        alpha=0.7,
        edgecolors='none',
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Runup Elevation (m)', fontsize=11)

    # Labels and title
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Runup Position (m)', fontsize=12)

    burst_start_min = burst_info['start_time'] / 60
    duration_min = burst_info['duration_sec'] / 60
    ax.set_title(
        f"Runup Timestack | Burst at {burst_start_min:.0f} min "
        f"({duration_min:.1f} min duration)",
        fontsize=12,
        fontweight='bold',
    )

    ax.grid(True, alpha=0.3)

    # Add stats text box
    stats_text = (
        f"N points: {valid.sum()}\n"
        f"X range: [{x_valid.min():.1f}, {x_valid.max():.1f}] m\n"
        f"Z range: [{z_valid.min():.2f}, {z_valid.max():.2f}] m\n"
        f"Z mean: {z_valid.mean():.2f} m"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return True


def process_l2_file(
    l2_path: Path,
    output_dir: Path,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 6),
    verbose: bool = True,
) -> int:
    """
    Process a single L2 file and create PNG plots for each burst.

    Returns number of PNGs created.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {l2_path.name}")
        print(f"{'='*60}")

    ds = xr.open_dataset(l2_path)

    # Extract elevation data
    if "elevation" in ds.data_vars:
        Z_xt = ds["elevation"].values
    elif "Z" in ds.data_vars:
        Z_xt = ds["Z"].values
    else:
        print(f"  ERROR: No elevation variable found")
        ds.close()
        return 0

    x1d = ds["x"].values

    # Get time in seconds
    if "time_seconds" in ds.coords:
        time_sec = ds["time_seconds"].values
    elif "time_seconds" in ds.data_vars:
        time_sec = ds["time_seconds"].values
    else:
        time_vec = ds["time"].values
        if np.issubdtype(time_vec.dtype, np.datetime64):
            time_sec = (time_vec - time_vec[0]) / np.timedelta64(1, 's')
        else:
            time_sec = time_vec.astype(float)

    # Get intensity if available
    I_xt = ds["intensity"].values if "intensity" in ds.data_vars else None

    if verbose:
        print(f"  Data shape: Z_xt={Z_xt.shape}")
        print(f"  Time range: {time_sec[0]:.1f}s to {time_sec[-1]:.1f}s ({(time_sec[-1]-time_sec[0])/60:.1f} min)")
        print(f"  Intensity: {'available' if I_xt is not None else 'not available'}")

    # Find valid bursts
    bursts = find_valid_bursts(Z_xt, time_sec, min_duration_sec=60.0)

    if not bursts:
        print(f"  WARNING: No valid bursts found")
        ds.close()
        return 0

    if verbose:
        print(f"  Found {len(bursts)} burst(s)")

    # Get date string for output directory
    date_str = l2_path.stem.replace("L2_", "")
    date_output_dir = output_dir / date_str
    date_output_dir.mkdir(parents=True, exist_ok=True)

    n_created = 0

    for i, burst in enumerate(bursts):
        s, e = burst['start_idx'], burst['end_idx']

        if verbose:
            burst_min = burst['start_time'] / 60
            print(f"  Burst {i+1}: {burst_min:.0f} min, {burst['n_frames']} frames, {burst['duration_sec']:.0f}s")

        # Extract burst data
        Z_burst = Z_xt[:, s:e]
        time_burst = time_sec[s:e]
        I_burst = I_xt[:, s:e] if I_xt is not None else None

        # Detect runup for this burst
        try:
            X_runup, Z_runup = detect_runup_for_burst(
                Z_burst, x1d, time_burst, I_xt=I_burst
            )
        except Exception as ex:
            print(f"    WARNING: Runup detection failed: {ex}")
            continue

        # Generate output filename (same naming as gif_nc_l2.py)
        burst_time_min = int(burst['start_time'] / 60)
        output_path = date_output_dir / f"{burst_time_min:04d}min.png"

        # Create plot
        try:
            success = create_runup_timestack_plot(
                time_burst, X_runup, Z_runup,
                output_path, burst,
                dpi=dpi, figsize=figsize,
            )
            if success:
                n_created += 1
                if verbose:
                    print(f"    Saved: {output_path.name}")
        except Exception as ex:
            print(f"    ERROR creating plot: {ex}")

    ds.close()
    return n_created


def main():
    parser = argparse.ArgumentParser(
        description="Create runup timestack plots from L2 NetCDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Config file (required)"
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
        help="Output directory for PNGs (default: plotFolder/level2/pngs/)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in DPI (default: 150)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6],
        metavar=('W', 'H'),
        help="Figure size in inches (default: 12 6)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine input directory
    input_dir = config.process_folder / "level2"
    if not input_dir.exists():
        print(f"Error: Level2 directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory (pngs subfolder within level2)
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = config.plot_folder / "level2" / "pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover L2 files
    if args.input is not None:
        input_path = Path(args.input)
        if input_path.is_absolute():
            l2_files = [input_path]
        else:
            l2_files = [input_dir / input_path]

        if not l2_files[0].exists():
            print(f"Error: Input file not found: {l2_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        l2_files = sorted(input_dir.glob("L2_*.nc"))
        if not l2_files:
            print(f"No L2 NetCDF files found in: {input_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(l2_files)} L2 file(s) to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process each file
    total_pngs = 0
    for l2_path in l2_files:
        n_pngs = process_l2_file(
            l2_path, output_dir,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
            verbose=args.verbose or len(l2_files) == 1,
        )
        total_pngs += n_pngs

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Files processed: {len(l2_files)}")
    print(f"  PNGs created: {total_pngs}")
    print(f"  Output directory: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
