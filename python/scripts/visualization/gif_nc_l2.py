#!/usr/bin/env python3
"""
Create animated GIFs from L2 NetCDF files showing cross-shore profiles
with runup detection.

For each burst (contiguous region with valid data), creates a GIF showing:
  - Cross-shore elevation profile Z(x) for each timestep
  - Detected runup position marked with a red dot
  - Dry beach reference line

Usage:
    # Process all L2 files from config
    python scripts/visualization/gif_nc_l2.py --config configs/towr_livox_config_20260120.json

    # Process single file
    python scripts/visualization/gif_nc_l2.py --config configs/towr_livox_config_20260120.json --input L2_20260120.nc

Options:
    --config PATH       Path to config file (required)
    --input FILE        Process single file (filename only, looked up in processFolder/level2/)
    --output PATH       Output directory for GIFs (default: plotFolder/level2/)
    --fps FPS           Frames per second (default: 4)
    --dpi DPI           Resolution (default: 150)
    --max-frames INT    Max frames per burst GIF (default: 500, ~2 min at 4fps)
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import xarray as xr

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from phase1 import adapt_path_for_os, load_config
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect runup position for each timestep (same method as plot_runup_multisignal.py).

    Returns
    -------
    X_runup : array (n_t,)
        Cross-shore runup position
    Z_runup : array (n_t,)
        Runup elevation
    dry_beach : array (n_x, n_t)
        Dry beach reference surface (2D, varies with time)
    """
    dt = time_sec[1] - time_sec[0] if len(time_sec) > 1 else 0.5

    # Compute runup stats (same as plot_runup_multisignal.py)
    result = runup.compute_runup_stats(
        Z_xt, x1d, time_sec,
        I_xt=I_xt,
        threshold=threshold,
        ig_length=ig_length,
    )

    # Compute dry beach reference for plotting (returns n_x, n_t)
    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt, ig_length=ig_length)

    return result.timeseries.X_runup, result.timeseries.Z_runup, dry_beach


def create_burst_gif(
    Z_burst: np.ndarray,
    x1d: np.ndarray,
    time_burst: np.ndarray,
    X_runup: np.ndarray,
    Z_runup: np.ndarray,
    dry_beach: np.ndarray,
    output_path: Path,
    burst_info: dict,
    fps: float = 4.0,
    dpi: int = 150,
    max_frames: int = 500,
) -> bool:
    """
    Create animated GIF for a single burst showing profile + runup.

    Two-panel figure:
      - Left: Zoomed view centered on runup region
      - Right: Full profile view

    Parameters
    ----------
    Z_burst : array (n_x, n_frames)
        Elevation data for this burst
    x1d : array (n_x,)
        Cross-shore positions
    time_burst : array (n_frames,)
        Time values for this burst
    X_runup, Z_runup : arrays (n_frames,)
        Detected runup positions and elevations
    dry_beach : array (n_x, n_frames)
        Dry beach reference (2D, varies with time)
    output_path : Path
        Output GIF path
    burst_info : dict
        Burst metadata
    fps : float
        Frames per second
    dpi : int
        Resolution
    max_frames : int
        Maximum number of frames (subsample if exceeded)

    Returns
    -------
    bool : True on success
    """
    n_x, n_frames = Z_burst.shape

    # Subsample if too many frames
    if n_frames > max_frames:
        step = n_frames // max_frames
        frame_indices = np.arange(0, n_frames, step)[:max_frames]
    else:
        frame_indices = np.arange(n_frames)

    n_plot_frames = len(frame_indices)

    # Determine plot limits
    z_valid = Z_burst[~np.isnan(Z_burst)]
    if len(z_valid) == 0:
        return False

    z_min = np.percentile(z_valid, 1) - 0.3
    z_max = np.percentile(z_valid, 99) + 0.3

    # Full profile limits
    x_full_min = x1d.min()
    x_full_max = x1d.max()

    # Determine runup range for zoomed view
    runup_valid = X_runup[~np.isnan(X_runup)]
    if len(runup_valid) > 0:
        x_runup_min = np.percentile(runup_valid, 5)
        x_runup_max = np.percentile(runup_valid, 95)
        # Add padding
        x_range = x_runup_max - x_runup_min
        x_zoom_min = max(x1d.min(), x_runup_min - x_range * 0.5 - 5)
        x_zoom_max = min(x1d.max(), x_runup_max + x_range * 0.5 + 5)
    else:
        # Fallback to middle portion
        x_zoom_min = x1d.min()
        x_zoom_max = x1d.max()

    # Create two-panel figure
    fig, (ax_zoom, ax_full) = plt.subplots(1, 2, figsize=(16, 6))

    # === Left panel: Zoomed view ===
    line_profile_zoom, = ax_zoom.plot([], [], 'b-', linewidth=1.5, label='Profile')
    line_dry_zoom, = ax_zoom.plot([], [], 'g--', linewidth=1, alpha=0.7, label='Dry beach')
    point_runup_zoom, = ax_zoom.plot([], [], 'ro', markersize=12, markeredgecolor='darkred',
                                      markeredgewidth=2, label='Runup', zorder=5)
    ax_zoom.axhline(y=0, color='cyan', linestyle='-', linewidth=1, alpha=0.5, label='MWL')

    ax_zoom.set_xlim(x_zoom_min, x_zoom_max)
    ax_zoom.set_ylim(z_min, z_max)
    ax_zoom.set_xlabel('Cross-shore distance (m)', fontsize=11)
    ax_zoom.set_ylabel('Elevation (m)', fontsize=11)
    ax_zoom.set_title('Zoomed View (Runup Region)')
    ax_zoom.legend(loc='upper right', fontsize=9)
    ax_zoom.grid(True, alpha=0.3)

    # Info text on zoomed panel
    info_text = ax_zoom.text(
        0.02, 0.98, '', transform=ax_zoom.transAxes,
        fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    )

    # === Right panel: Full profile ===
    line_profile_full, = ax_full.plot([], [], 'b-', linewidth=1.5, label='Profile')
    line_dry_full, = ax_full.plot([], [], 'g--', linewidth=1, alpha=0.7, label='Dry beach')
    point_runup_full, = ax_full.plot([], [], 'ro', markersize=10, markeredgecolor='darkred',
                                      markeredgewidth=2, label='Runup', zorder=5)
    ax_full.axhline(y=0, color='cyan', linestyle='-', linewidth=1, alpha=0.5, label='MWL')

    # Add shaded region showing zoom extent
    ax_full.axvspan(x_zoom_min, x_zoom_max, alpha=0.15, color='yellow', label='Zoom region')

    ax_full.set_xlim(x_full_min, x_full_max)
    ax_full.set_ylim(z_min, z_max)
    ax_full.set_xlabel('Cross-shore distance (m)', fontsize=11)
    ax_full.set_ylabel('Elevation (m)', fontsize=11)
    ax_full.set_title('Full Profile')
    ax_full.legend(loc='upper right', fontsize=9)
    ax_full.grid(True, alpha=0.3)

    # Overall title
    burst_start_min = burst_info['start_time'] / 60
    suptitle = fig.suptitle('', fontsize=12, fontweight='bold')

    plt.tight_layout()

    def update(frame_num):
        """Update animation frame."""
        idx = frame_indices[frame_num]

        # Get data for this frame
        profile = Z_burst[:, idx]
        dry_ref = dry_beach[:, idx]
        x_r = X_runup[idx]
        z_r = Z_runup[idx]
        t_sec = time_burst[idx]
        t_min = t_sec / 60

        # Update zoomed panel
        line_profile_zoom.set_data(x1d, profile)
        line_dry_zoom.set_data(x1d, dry_ref)
        if not np.isnan(x_r) and not np.isnan(z_r):
            point_runup_zoom.set_data([x_r], [z_r])
        else:
            point_runup_zoom.set_data([], [])

        # Update full panel
        line_profile_full.set_data(x1d, profile)
        line_dry_full.set_data(x1d, dry_ref)
        if not np.isnan(x_r) and not np.isnan(z_r):
            point_runup_full.set_data([x_r], [z_r])
        else:
            point_runup_full.set_data([], [])

        # Update title
        suptitle.set_text(f'Burst at {burst_start_min:.0f} min | t = {t_sec:.1f}s ({t_min:.2f} min)')

        # Update info text
        if not np.isnan(x_r):
            info_str = (
                f"Frame: {frame_num + 1}/{n_plot_frames}\n"
                f"Time: {t_sec:.1f} s\n"
                f"Runup X: {x_r:.2f} m\n"
                f"Runup Z: {z_r:.3f} m"
            )
        else:
            info_str = (
                f"Frame: {frame_num + 1}/{n_plot_frames}\n"
                f"Time: {t_sec:.1f} s\n"
                f"Runup: not detected"
            )
        info_text.set_text(info_str)

        return [line_profile_zoom, line_dry_zoom, point_runup_zoom,
                line_profile_full, line_dry_full, point_runup_full,
                suptitle, info_text]

    # Create animation
    interval = 1000 / fps
    anim = animation.FuncAnimation(
        fig, update,
        frames=n_plot_frames,
        interval=interval,
        blit=False,
    )

    # Save GIF
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    return True


def process_l2_file(
    l2_path: Path,
    output_dir: Path,
    fps: float = 4.0,
    dpi: int = 150,
    max_frames: int = 500,
    verbose: bool = True,
) -> int:
    """
    Process a single L2 file and create GIFs for each burst.

    Returns number of GIFs created.
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

        # Detect runup for this burst (same method as plot_runup_multisignal.py)
        try:
            X_runup, Z_runup, dry_beach = detect_runup_for_burst(
                Z_burst, x1d, time_burst, I_xt=I_burst
            )
        except Exception as ex:
            print(f"    WARNING: Runup detection failed: {ex}")
            continue

        # Generate output filename
        burst_time_min = int(burst['start_time'] / 60)
        output_path = date_output_dir / f"{burst_time_min:04d}min.gif"

        # Create GIF
        try:
            success = create_burst_gif(
                Z_burst, x1d, time_burst,
                X_runup, Z_runup, dry_beach,
                output_path, burst,
                fps=fps, dpi=dpi, max_frames=max_frames,
            )
            if success:
                n_created += 1
                if verbose:
                    print(f"    Saved: {output_path.name}")
        except Exception as ex:
            print(f"    ERROR creating GIF: {ex}")

    ds.close()
    return n_created


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIFs from L2 NetCDF files (profile + runup)",
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
        help="Output directory for GIFs (default: plotFolder/level2/)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second (default: 4)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in DPI (default: 150)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Max frames per burst GIF (default: 500)"
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

    # Determine output directory
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = config.plot_folder / "level2"
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
    total_gifs = 0
    for l2_path in l2_files:
        n_gifs = process_l2_file(
            l2_path, output_dir,
            fps=args.fps,
            dpi=args.dpi,
            max_frames=args.max_frames,
            verbose=args.verbose or len(l2_files) == 1,
        )
        total_gifs += n_gifs

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Files processed: {len(l2_files)}")
    print(f"  GIFs created: {total_gifs}")
    print(f"  Output directory: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
