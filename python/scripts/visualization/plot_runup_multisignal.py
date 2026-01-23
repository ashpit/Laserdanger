#!/usr/bin/env python3
"""
Plot multi-signal runup detection results for L2 data.

Uses config file to determine input/output paths.
Processes all L2 files by default, or a single file with --input.

Usage:
    python scripts/visualization/plot_runup_multisignal.py --config configs/towr_livox_config_20260120.json
    python scripts/visualization/plot_runup_multisignal.py --config configs/towr_livox_config_20260120.json --input L2_20260120.nc
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from phase1 import adapt_path_for_os
import runup


def find_valid_regions(Z_xt, time_sec):
    """Find contiguous regions with valid data."""
    valid_cols = ~np.all(np.isnan(Z_xt), axis=0)

    if not valid_cols.any():
        return []

    diff = np.diff(valid_cols.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if valid_cols[0]:
        starts = np.concatenate([[0], starts])
    if valid_cols[-1]:
        ends = np.concatenate([ends, [len(valid_cols)]])

    regions = []
    for s, e in zip(starts, ends):
        regions.append({
            'start_idx': s,
            'end_idx': e,
            'start_time': time_sec[s],
            'end_time': time_sec[e-1],
            'n_timesteps': e - s,
        })
    return regions


def process_l2_file(l2_path: Path, output_dir: Path, verbose: bool = False):
    """Process a single L2 file and generate runup plots."""

    if verbose:
        print(f"Processing {l2_path.name}...")

    ds = xr.open_dataset(l2_path)

    # Extract arrays
    if "elevation" in ds.data_vars:
        Z_xt = ds["elevation"].values
    elif "Z" in ds.data_vars:
        Z_xt = ds["Z"].values
    else:
        print(f"  ERROR: No elevation variable found in {l2_path.name}")
        ds.close()
        return None

    x1d = ds["x"].values

    if "time_seconds" in ds.coords:
        time_sec = ds["time_seconds"].values
    elif "time_seconds" in ds.data_vars:
        time_sec = ds["time_seconds"].values
    else:
        time_vec = ds["time"].values
        if np.issubdtype(time_vec.dtype, np.datetime64):
            time_sec = (time_vec - time_vec[0]) / np.timedelta64(1, 's')
        else:
            time_sec = time_vec - time_vec[0]

    I_xt = ds["intensity"].values if "intensity" in ds.data_vars else None

    # Check if transect needs to be flipped (seaward should be at x=0)
    z_mean_along_x = np.nanmean(Z_xt, axis=1)
    valid_z = ~np.isnan(z_mean_along_x)
    if valid_z.sum() > 10:
        first_valid = np.where(valid_z)[0][0]
        last_valid = np.where(valid_z)[0][-1]
        if z_mean_along_x[first_valid] > z_mean_along_x[last_valid]:
            if verbose:
                print("  Flipping transect orientation (seaward -> x=0)")
            x1d = x1d[::-1]
            Z_xt = Z_xt[::-1, :]
            if I_xt is not None:
                I_xt = I_xt[::-1, :]

    # Find valid regions
    regions = find_valid_regions(Z_xt, time_sec)

    if not regions:
        print(f"  WARNING: No valid data regions in {l2_path.name}")
        ds.close()
        return None

    if verbose:
        print(f"  Found {len(regions)} valid data bursts")

    # Get date string for directory
    date_str = l2_path.stem.replace("L2_", "")

    # Create date-specific output directory
    date_output_dir = output_dir / date_str
    date_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each region
    all_results = []

    for i, region in enumerate(regions):
        s, e = region['start_idx'], region['end_idx']

        Z_sub = Z_xt[:, s:e]
        I_sub = I_xt[:, s:e] if I_xt is not None else None
        time_sub = time_sec[s:e]

        # Run multi-signal detection
        result = runup.compute_runup_stats(
            Z_sub, x1d, time_sub,
            I_xt=I_sub,
            threshold=0.1,
            ig_length=100.0,
        )

        data = {
            'region': region,
            'result': result,
            'Z_sub': Z_sub,
            'I_sub': I_sub,
            'time_sub': time_sub,
            'burst_num': i + 1,
        }
        all_results.append(data)

        # Generate burst time string (minutes from start of day)
        burst_time_min = int(region['start_time'] / 60)
        burst_time_str = f"{burst_time_min:04d}min"

        # Plot individual burst
        _plot_single_burst(data, x1d, date_output_dir / f"{burst_time_str}.png")

        if verbose:
            valid_frac = result.info['n_valid'] / len(time_sub) * 100
            print(f"  Burst {i+1}: {burst_time_str} ({valid_frac:.0f}% detection, Sig={result.bulk.Sig:.3f}m)")

    # Also save summary statistics plot
    _plot_statistics_summary(all_results, date_output_dir / "summary.png")

    if verbose:
        print(f"  Saved {len(all_results)} burst plots + summary to {date_output_dir}")

    ds.close()
    return all_results


def _plot_single_burst(data, x1d, output_path):
    """Plot a single burst with runup detection."""
    region = data['region']
    result = data['result']
    Z_sub = data['Z_sub']
    I_sub = data['I_sub']
    time_sub = data['time_sub']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Determine zoom range based on runup positions
    valid = ~np.isnan(result.timeseries.X_runup)
    if valid.any():
        runup_min = np.nanmin(result.timeseries.X_runup)
        runup_max = np.nanmax(result.timeseries.X_runup)
        # Add padding around runup range (20m on each side)
        y_zoom_min = runup_min - 20
        y_zoom_max = runup_max + 20
    else:
        # Fallback to middle portion of x range
        y_zoom_min = x1d[len(x1d)//4]
        y_zoom_max = x1d[3*len(x1d)//4]

    # Top-left: Elevation timestack with runup
    ax = axes[0, 0]
    extent = [time_sub[0], time_sub[-1], x1d[-1], x1d[0]]
    im = ax.imshow(Z_sub, aspect='auto', extent=extent, cmap='terrain', vmin=-1, vmax=3)

    if valid.any():
        ax.plot(time_sub[valid], result.timeseries.X_runup[valid], 'b-', lw=1.5, label='Runup')
        ax.legend(loc='upper right')

    # Zoom in on runup region
    ax.set_ylim([y_zoom_max, y_zoom_min])  # Inverted for imshow orientation
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title('Elevation Timestack')
    plt.colorbar(im, ax=ax, label='Z (m)')

    # Top-right: Intensity timestack (if available)
    ax = axes[0, 1]
    if I_sub is not None:
        im = ax.imshow(I_sub, aspect='auto', extent=extent, cmap='gray', vmin=0, vmax=100)
        if valid.any():
            ax.plot(time_sub[valid], result.timeseries.X_runup[valid], 'c-', lw=1.5)
        ax.set_title('Intensity Timestack')
        plt.colorbar(im, ax=ax, label='Intensity')
    else:
        ax.text(0.5, 0.5, 'No intensity data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Intensity (N/A)')
    # Zoom in on runup region
    ax.set_ylim([y_zoom_max, y_zoom_min])  # Inverted for imshow orientation
    ax.set_ylabel('Cross-shore (m)')

    # Bottom-left: Runup position time series
    ax = axes[1, 0]
    if valid.any():
        ax.plot(time_sub, result.timeseries.X_runup, 'b-', lw=1)
        # Use same zoom range as top plots, inverted to match orientation
        ax.set_ylim([y_zoom_max, y_zoom_min])
    ax.set_ylabel('Runup X (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Runup Position')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Adaptive weights (if available)
    ax = axes[1, 1]
    has_weights = result.info.get('multisignal_enabled', False)
    if has_weights and result.timeseries.weights_used.size > 0:
        w = result.timeseries.weights_used
        ax.fill_between(time_sub, 0, w[0, :], alpha=0.7, label='Elevation', color='C0')
        ax.fill_between(time_sub, w[0, :], w[0, :]+w[1, :], alpha=0.7, label='Intensity', color='C1')
        ax.fill_between(time_sub, w[0, :]+w[1, :], 1, alpha=0.7, label='Variance', color='C2')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Signal Weights')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Single-signal mode', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Adaptive Weights (N/A)')
    ax.set_xlabel('Time (s)')

    # Add overall title with stats
    burst_time = region['start_time'] / 60
    stats_str = f"Sig={result.bulk.Sig:.3f}m, Sinc={result.bulk.Sinc:.3f}m, β={result.bulk.beta:.4f}"
    detect_rate = result.info['n_valid'] / len(time_sub) * 100
    fig.suptitle(f"Burst at {burst_time:.0f} min | {stats_str} | {detect_rate:.0f}% detection",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_statistics_summary(all_results, output_path):
    """Plot summary statistics across all bursts."""
    if not all_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract stats
    times = [r['region']['start_time']/60 for r in all_results]
    sigs = [r['result'].bulk.Sig for r in all_results]
    sincs = [r['result'].bulk.Sinc for r in all_results]
    betas = [r['result'].bulk.beta for r in all_results]
    detection_rates = [r['result'].info['n_valid'] / len(r['time_sub']) * 100 for r in all_results]

    # Get weights if available
    has_weights = all_results[0]['result'].info.get('multisignal_enabled', False)
    if has_weights:
        w_elev = [r['result'].info.get('mean_weight_elevation', np.nan) for r in all_results]
        w_int = [r['result'].info.get('mean_weight_intensity', np.nan) for r in all_results]
        w_var = [r['result'].info.get('mean_weight_variance', np.nan) for r in all_results]

    # Plot 1: Wave heights over time
    ax = axes[0, 0]
    ax.plot(times, sigs, 'o-', label='Sig (IG)', color='C0')
    ax.plot(times, sincs, 's-', label='Sinc', color='C1')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Wave Height (m)')
    ax.set_title('Significant Wave Heights')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Beach slope over time
    ax = axes[0, 1]
    ax.plot(times, betas, 'o-', color='C2')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Beach Slope (β)')
    ax.set_title('Foreshore Slope')
    ax.grid(True, alpha=0.3)

    # Plot 3: Detection rate
    ax = axes[1, 0]
    ax.bar(times, detection_rates, width=5, color='C3', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Runup Detection Rate')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Plot 4: Adaptive weights (if available)
    ax = axes[1, 1]
    if has_weights:
        width = 5
        ax.bar(times, w_elev, width, label='Elevation', color='C0', alpha=0.8)
        ax.bar(times, w_int, width, bottom=w_elev, label='Intensity', color='C1', alpha=0.8)
        bottoms = [e + i for e, i in zip(w_elev, w_int)]
        ax.bar(times, w_var, width, bottom=bottoms, label='Variance', color='C2', alpha=0.8)
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Signal Weights')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, 'Single-signal mode\n(no intensity data)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Adaptive Weights (N/A)')
    ax.set_xlabel('Time (min)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Runup Statistics Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot multi-signal runup detection results for L2 data.'
    )
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    parser.add_argument('--input', help='Specific L2 file to process (default: all files)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Determine paths from config (adapt for OS)
    process_folder = adapt_path_for_os(config['processFolder'])
    plot_folder = adapt_path_for_os(config['plotFolder'])

    l2_input_dir = process_folder / 'level2'
    l2_output_dir = plot_folder / 'level2' / 'pngs'

    if args.verbose:
        print(f"Config: {config_path.name}")
        print(f"L2 input: {l2_input_dir}")
        print(f"L2 output: {l2_output_dir}")

    # Find L2 files to process
    if args.input:
        # Single file
        if Path(args.input).is_absolute():
            l2_files = [Path(args.input)]
        else:
            l2_files = [l2_input_dir / args.input]
    else:
        # All L2 files
        l2_files = sorted(l2_input_dir.glob('L2_*.nc'))

    if not l2_files:
        print(f"ERROR: No L2 files found in {l2_input_dir}")
        sys.exit(1)

    print(f"Processing {len(l2_files)} L2 file(s)...")

    for l2_path in l2_files:
        if not l2_path.exists():
            print(f"WARNING: File not found: {l2_path}")
            continue

        process_l2_file(l2_path, l2_output_dir, verbose=args.verbose)

    print(f"\nDone! Plots saved to: {l2_output_dir}")


if __name__ == "__main__":
    main()
