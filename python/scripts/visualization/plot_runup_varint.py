#!/usr/bin/env python3
"""
Plot runup detection using ONLY variance and intensity signals (no elevation).

This is an experimental approach that may work better when elevation signal
is noisy or when water/sand have similar elevations.

Uses config file to determine input/output paths.
Processes all L2 files by default, or a single file with --input.

Usage:
    python scripts/visualization/plot_runup_varint.py --config configs/towr_livox_config_20260120.json
    python scripts/visualization/plot_runup_varint.py --config configs/towr_livox_config_20260120.json --input L2_20260120.nc
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from phase1 import adapt_path_for_os
import runup


@dataclass
class VarIntResult:
    """Results from variance+intensity runup detection."""
    X_runup: np.ndarray
    Z_runup: np.ndarray
    confidence: np.ndarray
    w_intensity: np.ndarray
    w_variance: np.ndarray
    p_intensity: np.ndarray
    p_variance: np.ndarray
    P_water: np.ndarray


def compute_runup_varint(
    Z_xt: np.ndarray,
    I_xt: np.ndarray,
    x1d: np.ndarray,
    time_vec: np.ndarray,
    ig_length: float = 100.0,
    search_window: float = 0.5,
    variance_window_seconds: float = 2.5,
    intensity_scale: float = 20.0,
    variance_scale: float = 0.002,
) -> VarIntResult:
    """
    Compute runup using only variance and intensity signals.

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix (used for variance calculation and final Z lookup)
    I_xt : array (n_x, n_t)
        Intensity matrix
    x1d : array (n_x,)
        Cross-shore positions
    time_vec : array (n_t,)
        Time values
    ig_length : float
        Window for dry reference calculation
    search_window : float
        Search window for runup tracking (meters)
    variance_window_seconds : float
        Window for variance calculation
    intensity_scale : float
        Sigmoid scale for intensity
    variance_scale : float
        Sigmoid scale for variance

    Returns
    -------
    VarIntResult
        Detection results with runup positions and signal diagnostics
    """
    n_x, n_t = Z_xt.shape
    dt = np.median(np.diff(time_vec))
    dx = np.median(np.diff(x1d))

    # Compute intensity signal
    I_dry_ref = runup.compute_dry_intensity_reference(I_xt, dt, ig_length)
    p_intensity = runup.compute_intensity_signal(I_xt, I_dry_ref, intensity_scale)

    # Compute variance signal
    p_variance = runup.compute_variance_signal(
        Z_xt, dt, variance_window_seconds,
        var_threshold=0.001, scale=variance_scale
    )

    # Estimate SNR for each signal (use variance as proxy for water regions)
    # Since we don't have elevation, use variance > 0.5 as water proxy
    snr_intensity = _estimate_intensity_snr_noelevation(I_xt, I_dry_ref, p_variance)
    snr_variance = _estimate_variance_snr_noelevation(p_variance)

    # Compute adaptive weights (just two signals)
    w_int, w_var = _compute_two_signal_weights(snr_intensity, snr_variance)

    # Fuse signals (only intensity + variance)
    P_water = (
        w_int[np.newaxis, :] * p_intensity +
        w_var[np.newaxis, :] * p_variance
    )

    # Detect runup using fused probability
    X_runup, Z_runup, idx_runup, confidence = runup.detect_runup_multisignal(
        Z_xt, P_water, x1d, search_window, dx
    )

    # Smooth runup timeseries
    X_runup, Z_runup = runup.smooth_runup_timeseries(X_runup, Z_runup, dt)

    return VarIntResult(
        X_runup=X_runup,
        Z_runup=Z_runup,
        confidence=confidence,
        w_intensity=w_int,
        w_variance=w_var,
        p_intensity=p_intensity,
        p_variance=p_variance,
        P_water=P_water,
    )


def _estimate_intensity_snr_noelevation(
    I_xt: np.ndarray,
    I_dry_ref: np.ndarray,
    p_variance: np.ndarray,
) -> np.ndarray:
    """Estimate intensity SNR using variance as water proxy."""
    n_x, n_t = I_xt.shape
    intensity_drop = I_dry_ref - I_xt

    snr = np.zeros(n_t)
    for t in range(n_t):
        col = intensity_drop[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            snr[t] = 0.1
            continue

        # Use variance probability as water proxy
        p_col = p_variance[:, t]
        water_mask = p_col > 0.5
        sand_mask = p_col < 0.3

        if water_mask.sum() > 2 and sand_mask.sum() > 2:
            water_drop = np.nanmean(col[water_mask])
            sand_drop = np.nanmean(col[sand_mask])
            contrast = water_drop - sand_drop
            noise = np.nanstd(col[sand_mask])
        else:
            contrast = np.nanmax(col) - np.nanmedian(col)
            noise = np.nanstd(col) * 0.5

        if noise < 1e-6:
            noise = 1e-6

        snr[t] = max(0.1, contrast / noise)

    return snr


def _estimate_variance_snr_noelevation(p_variance: np.ndarray) -> np.ndarray:
    """Estimate variance SNR from signal contrast."""
    n_x, n_t = p_variance.shape

    snr = np.zeros(n_t)
    for t in range(n_t):
        p_col = p_variance[:, t]
        valid = ~np.isnan(p_col)
        if valid.sum() < 5:
            snr[t] = 0.1
            continue

        # Use high/low variance regions
        high_var = p_col > 0.6
        low_var = p_col < 0.3

        if high_var.sum() > 2 and low_var.sum() > 2:
            contrast = np.nanmean(p_col[high_var]) - np.nanmean(p_col[low_var])
            noise = np.nanstd(p_col[low_var])
        else:
            contrast = np.nanmax(p_col) - np.nanmin(p_col)
            noise = np.nanstd(p_col) * 0.5

        if noise < 1e-6:
            noise = 1e-6

        snr[t] = max(0.1, contrast / noise)

    return snr


def _compute_two_signal_weights(
    snr_intensity: np.ndarray,
    snr_variance: np.ndarray,
    min_weight: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute adaptive weights for two signals."""
    snr_min = min_weight * 2

    s_int = np.maximum(snr_intensity, snr_min)
    s_var = np.maximum(snr_variance, snr_min)

    total = s_int + s_var
    w_int = s_int / total
    w_var = s_var / total

    return w_int, w_var


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

    if I_xt is None:
        print(f"  ERROR: No intensity data - variance+intensity method requires intensity")
        ds.close()
        return None

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
        I_sub = I_xt[:, s:e]
        time_sub = time_sec[s:e]

        # Run variance+intensity detection
        try:
            result = compute_runup_varint(
                Z_sub, I_sub, x1d, time_sub,
                ig_length=100.0,
            )
        except Exception as ex:
            if verbose:
                print(f"  Burst {i+1}: detection failed - {ex}")
            continue

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
            n_valid = np.sum(~np.isnan(result.X_runup))
            valid_frac = n_valid / len(time_sub) * 100
            print(f"  Burst {i+1}: {burst_time_str} ({valid_frac:.0f}% detection)")

    # Also save summary statistics plot
    if all_results:
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
    valid = ~np.isnan(result.X_runup)
    if valid.any():
        runup_min = np.nanmin(result.X_runup)
        runup_max = np.nanmax(result.X_runup)
        y_zoom_min = runup_min - 20
        y_zoom_max = runup_max + 20
    else:
        y_zoom_min = x1d[len(x1d)//4]
        y_zoom_max = x1d[3*len(x1d)//4]

    extent = [time_sub[0], time_sub[-1], x1d[-1], x1d[0]]

    # Top-left: Variance probability with runup
    ax = axes[0, 0]
    im = ax.imshow(result.p_variance, aspect='auto', extent=extent, cmap='hot', vmin=0, vmax=1)
    if valid.any():
        ax.plot(time_sub[valid], result.X_runup[valid], 'c-', lw=1.5, label='Runup')
        ax.legend(loc='upper right')
    ax.set_ylim([y_zoom_max, y_zoom_min])
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title('Variance Probability')
    plt.colorbar(im, ax=ax, label='P(water)')

    # Top-right: Intensity probability with runup
    ax = axes[0, 1]
    im = ax.imshow(result.p_intensity, aspect='auto', extent=extent, cmap='hot', vmin=0, vmax=1)
    if valid.any():
        ax.plot(time_sub[valid], result.X_runup[valid], 'c-', lw=1.5)
    ax.set_ylim([y_zoom_max, y_zoom_min])
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title('Intensity Probability')
    plt.colorbar(im, ax=ax, label='P(water)')

    # Bottom-left: Runup position time series
    ax = axes[1, 0]
    if valid.any():
        ax.plot(time_sub, result.X_runup, 'b-', lw=1)
        ax.set_ylim([y_zoom_max, y_zoom_min])
    ax.set_ylabel('Runup X (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Runup Position')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Adaptive weights (only 2 signals)
    ax = axes[1, 1]
    ax.fill_between(time_sub, 0, result.w_intensity, alpha=0.7, label='Intensity', color='C1')
    ax.fill_between(time_sub, result.w_intensity, 1, alpha=0.7, label='Variance', color='C2')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Time (s)')
    ax.set_title('Adaptive Weights (Var+Int only)')
    ax.legend(loc='upper right', fontsize=8)

    # Add overall title
    burst_time = region['start_time'] / 60
    n_valid = np.sum(valid)
    detect_rate = n_valid / len(time_sub) * 100
    fig.suptitle(f"Burst at {burst_time:.0f} min | Variance+Intensity | {detect_rate:.0f}% detection",
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
    detection_rates = [np.sum(~np.isnan(r['result'].X_runup)) / len(r['time_sub']) * 100
                       for r in all_results]
    w_int = [np.mean(r['result'].w_intensity) for r in all_results]
    w_var = [np.mean(r['result'].w_variance) for r in all_results]

    # Mean confidence
    mean_conf = [np.nanmean(r['result'].confidence) for r in all_results]

    # Plot 1: Detection rate
    ax = axes[0, 0]
    ax.bar(times, detection_rates, width=5, color='C0', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Runup Detection Rate')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean confidence
    ax = axes[0, 1]
    ax.plot(times, mean_conf, 'o-', color='C3')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Mean Confidence')
    ax.set_title('Detection Confidence')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 3: Adaptive weights over time
    ax = axes[1, 0]
    width = 5
    ax.bar(times, w_int, width, label='Intensity', color='C1', alpha=0.8)
    ax.bar(times, w_var, width, bottom=w_int, label='Variance', color='C2', alpha=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Adaptive Weights')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 4: Runup position range
    ax = axes[1, 1]
    runup_mins = [np.nanmin(r['result'].X_runup) for r in all_results]
    runup_maxs = [np.nanmax(r['result'].X_runup) for r in all_results]
    runup_means = [np.nanmean(r['result'].X_runup) for r in all_results]

    ax.fill_between(times, runup_mins, runup_maxs, alpha=0.3, color='C0', label='Range')
    ax.plot(times, runup_means, 'o-', color='C0', label='Mean')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Runup X (m)')
    ax.set_title('Runup Position Range')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Variance+Intensity Runup Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot runup using variance+intensity only (no elevation).'
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
    l2_output_dir = plot_folder / 'level2' / 'pngs_varint'

    if args.verbose:
        print(f"Config: {config_path.name}")
        print(f"L2 input: {l2_input_dir}")
        print(f"L2 output: {l2_output_dir}")

    # Find L2 files to process
    if args.input:
        if Path(args.input).is_absolute():
            l2_files = [Path(args.input)]
        else:
            l2_files = [l2_input_dir / args.input]
    else:
        l2_files = sorted(l2_input_dir.glob('L2_*.nc'))

    if not l2_files:
        print(f"ERROR: No L2 files found in {l2_input_dir}")
        sys.exit(1)

    print(f"Processing {len(l2_files)} L2 file(s) with variance+intensity method...")

    for l2_path in l2_files:
        if not l2_path.exists():
            print(f"WARNING: File not found: {l2_path}")
            continue

        process_l2_file(l2_path, l2_output_dir, verbose=args.verbose)

    print(f"\nDone! Plots saved to: {l2_output_dir}")


if __name__ == "__main__":
    main()
