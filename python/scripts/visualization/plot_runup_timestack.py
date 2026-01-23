#!/usr/bin/env python3
"""
Create runup timestack plots from L2 NetCDF files.

For each burst (contiguous region with valid data), creates a timestack figure showing:
  - Background: Z(x,t) elevation heatmap
  - Overlay: runup line (black) with markers (red dots)
  - Optional: smoothed runup line (white dashed)

Usage:
    # Process all L2 files from config
    python scripts/visualization/plot_runup_timestack.py --config configs/towr_livox_config_20260120.json

    # Process single file
    python scripts/visualization/plot_runup_timestack.py --config configs/towr_livox_config_20260120.json --input L2_20260120.nc

    # Custom time window
    python scripts/visualization/plot_runup_timestack.py --config configs/towr_livox_config_20260120.json --t-start 200 --t-end 360

Options:
    --config PATH       Path to config file (required)
    --input FILE        Process single file (filename only, looked up in processFolder/level2/)
    --output PATH       Output directory for PNGs (default: plotFolder/level2/pngs/)
    --dpi DPI           Resolution (default: 150)
    --figsize W H       Figure size in inches (default: 12 4)
    --t-start FLOAT     Start time in seconds (default: auto)
    --t-end FLOAT       End time in seconds (default: auto)
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import uniform_filter1d
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
    Z_burst: np.ndarray,
    x1d: np.ndarray,
    time_burst: np.ndarray,
    X_runup: np.ndarray,
    Z_runup: np.ndarray,
    output_path: Path,
    burst_info: dict,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 4),
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> bool:
    """
    Create runup timestack plot with elevation heatmap background.

    Parameters
    ----------
    Z_burst : array (n_x, n_t)
        Elevation timestack
    x1d : array (n_x,)
        Cross-shore positions
    time_burst : array (n_t,)
        Time values for this burst (seconds)
    X_runup : array (n_t,)
        Cross-shore runup position
    Z_runup : array (n_t,)
        Runup elevation
    output_path : Path
        Output PNG path
    burst_info : dict
        Burst metadata
    dpi : int
        Resolution
    figsize : tuple
        Figure size (width, height) in inches
    t_start, t_end : float, optional
        Time window to plot (default: full burst)

    Returns
    -------
    bool : True on success
    """
    # Filter to valid points
    valid = ~np.isnan(X_runup) & ~np.isnan(Z_runup)
    if valid.sum() < 5:
        return False

    # Apply time window if specified
    if t_start is not None or t_end is not None:
        t0 = t_start if t_start is not None else time_burst[0]
        t1 = t_end if t_end is not None else time_burst[-1]
        time_mask = (time_burst >= t0) & (time_burst <= t1)

        Z_burst = Z_burst[:, time_mask]
        time_burst = time_burst[time_mask]
        X_runup = X_runup[time_mask]
        Z_runup = Z_runup[time_mask]
        valid = valid[time_mask]

    if len(time_burst) < 5:
        return False

    # Determine x-axis limits from runup range (with padding)
    runup_valid = X_runup[~np.isnan(X_runup)]
    if len(runup_valid) < 5:
        return False

    x_margin = 5.0  # meters padding
    x_min = max(x1d.min(), runup_valid.min() - x_margin)
    x_max = min(x1d.max(), runup_valid.max() + x_margin)

    # Subset x to focus on runup region
    x_mask = (x1d >= x_min) & (x1d <= x_max)
    x_subset = x1d[x_mask]
    Z_subset = Z_burst[x_mask, :]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Background: elevation heatmap using pcolormesh
    # Need to create mesh edges for pcolormesh
    dt = np.median(np.diff(time_burst))
    dx = np.median(np.diff(x_subset)) if len(x_subset) > 1 else 0.1

    t_edges = np.concatenate([
        [time_burst[0] - dt/2],
        (time_burst[:-1] + time_burst[1:]) / 2,
        [time_burst[-1] + dt/2]
    ])
    x_edges = np.concatenate([
        [x_subset[0] - dx/2],
        (x_subset[:-1] + x_subset[1:]) / 2,
        [x_subset[-1] + dx/2]
    ])

    # Determine color limits from data
    z_valid = Z_subset[~np.isnan(Z_subset)]
    if len(z_valid) > 0:
        vmin = np.percentile(z_valid, 2)
        vmax = np.percentile(z_valid, 98)
    else:
        vmin, vmax = -1, 1

    # Plot heatmap (note: pcolormesh expects Z_subset as (n_x, n_t))
    mesh = ax.pcolormesh(
        t_edges, x_edges, Z_subset,
        cmap='RdYlBu_r',  # Blue (low/water) to red (high/land)
        vmin=vmin, vmax=vmax,
        shading='flat',
        rasterized=True,
    )

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label('Elevation (m)', fontsize=11)

    # Compute smoothed runup for white dashed line
    X_smooth = X_runup.copy()
    valid_idx = ~np.isnan(X_smooth)
    if valid_idx.sum() > 10:
        # Apply uniform filter (moving average) to valid data
        kernel_size = min(11, valid_idx.sum() // 3)
        if kernel_size >= 3:
            X_smooth[valid_idx] = uniform_filter1d(
                X_smooth[valid_idx], kernel_size, mode='nearest'
            )

    # Overlay: smoothed runup line (white dashed, plotted first so it's behind)
    ax.plot(
        time_burst, X_smooth,
        'w--', linewidth=1.5, alpha=0.8,
        label='Smoothed',
    )

    # Overlay: raw runup line (black solid)
    ax.plot(
        time_burst, X_runup,
        'k-', linewidth=1.0, alpha=0.9,
        label='Runup',
    )

    # Overlay: red dots at runup positions (subsample if too many)
    n_points = valid.sum()
    if n_points > 200:
        # Subsample to ~100 points
        step = n_points // 100
        plot_mask = np.zeros_like(valid)
        valid_indices = np.where(valid)[0]
        plot_mask[valid_indices[::step]] = True
    else:
        plot_mask = valid

    ax.scatter(
        time_burst[plot_mask], X_runup[plot_mask],
        c='red', s=15, alpha=0.8,
        edgecolors='darkred', linewidths=0.5,
        zorder=5,
    )

    # Labels and formatting
    ax.set_xlabel('time (s)', fontsize=12)
    ax.set_ylabel('x-shore distance (m)', fontsize=12)

    # Set axis limits
    ax.set_xlim(time_burst[0], time_burst[-1])
    ax.set_ylim(x_max, x_min)  # Inverted so land (high x) is at bottom

    # Legend (compact, in corner)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return True


def process_l2_file(
    l2_path: Path,
    output_dir: Path,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 4),
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
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
                Z_burst, x1d, time_burst,
                X_runup, Z_runup,
                output_path, burst,
                dpi=dpi, figsize=figsize,
                t_start=t_start, t_end=t_end,
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
        default=[12, 4],
        metavar=('W', 'H'),
        help="Figure size in inches (default: 12 4)"
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Start time in seconds (default: auto)"
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="End time in seconds (default: auto)"
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
            t_start=args.t_start,
            t_end=args.t_end,
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
