#!/usr/bin/env python3
"""
Test multi-signal runup detection on L2 data.

Compares elevation-only vs multi-signal (elevation + intensity + variance) detection.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add code directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import runup


def main():
    # Load L2 data
    l2_path = Path("/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/github/Laserdanger/python/data/TOWR-lidar/surfside-test_20260120/lvx_processed/level2/L2_20260120.nc")

    if not l2_path.exists():
        print(f"L2 file not found: {l2_path}")
        return

    print(f"Loading {l2_path}...")
    ds = xr.open_dataset(l2_path)

    # Extract arrays - check for variable name variations
    if "Z" in ds.data_vars:
        Z_xt = ds["Z"].values
    elif "elevation" in ds.data_vars:
        Z_xt = ds["elevation"].values
    else:
        raise KeyError(f"No elevation variable found. Variables: {list(ds.data_vars)}")

    x1d = ds["x"].values

    # Handle time - could be datetime or seconds
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

    # Check for intensity
    has_intensity = "intensity" in ds.data_vars
    I_xt = ds["intensity"].values if has_intensity else None

    print(f"Data shape: Z_xt={Z_xt.shape}, x1d={len(x1d)}, time={len(time_sec)}")
    print(f"Intensity available: {has_intensity}")
    print(f"Time range: {time_sec[0]:.1f}s to {time_sec[-1]:.1f}s ({(time_sec[-1]-time_sec[0])/60:.1f} min)")

    # Find the valid data range (where not all NaN)
    valid_cols = ~np.all(np.isnan(Z_xt), axis=0)
    valid_indices = np.where(valid_cols)[0]

    if len(valid_indices) == 0:
        print("ERROR: No valid data found!")
        return

    # Use only the valid portion
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1

    Z_sub = Z_xt[:, start_idx:end_idx]
    I_sub = I_xt[:, start_idx:end_idx] if I_xt is not None else None
    time_sub = time_sec[start_idx:end_idx]

    print(f"\nValid data range: {time_sub[0]:.1f}s to {time_sub[-1]:.1f}s ({(time_sub[-1]-time_sub[0])/60:.1f} min)")
    print(f"Using {Z_sub.shape[1]} timesteps")

    # Run elevation-only detection
    print("\n--- Elevation-only detection ---")
    result_elev = runup.compute_runup_stats(
        Z_sub, x1d, time_sub,
        I_xt=None,
        threshold=0.1,
        ig_length=100.0,
    )
    print(f"Valid detections: {result_elev.info['n_valid']}/{len(time_sub)}")
    print(f"Sig (IG): {result_elev.bulk.Sig:.3f} m")
    print(f"Sinc: {result_elev.bulk.Sinc:.3f} m")
    print(f"Beta: {result_elev.bulk.beta:.4f}")

    # Run multi-signal detection
    if has_intensity:
        print("\n--- Multi-signal detection ---")
        result_multi = runup.compute_runup_stats(
            Z_sub, x1d, time_sub,
            I_xt=I_sub,
            threshold=0.1,
            ig_length=100.0,
        )
        print(f"Valid detections: {result_multi.info['n_valid']}/{len(time_sub)}")
        print(f"Sig (IG): {result_multi.bulk.Sig:.3f} m")
        print(f"Sinc: {result_multi.bulk.Sinc:.3f} m")
        print(f"Beta: {result_multi.bulk.beta:.4f}")
        print(f"Mean weights: elev={result_multi.info['mean_weight_elevation']:.2f}, "
              f"int={result_multi.info['mean_weight_intensity']:.2f}, "
              f"var={result_multi.info['mean_weight_variance']:.2f}")
        print(f"Mean confidence: {result_multi.info['mean_confidence']:.2f}")
    else:
        result_multi = None

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Elevation timestack with runup lines
    ax = axes[0]
    extent = [time_sub[0], time_sub[-1], x1d[-1], x1d[0]]
    im = ax.imshow(Z_sub, aspect='auto', extent=extent, cmap='terrain', vmin=-1, vmax=3)
    ax.plot(time_sub, result_elev.timeseries.X_runup, 'r-', lw=1, alpha=0.8, label='Elev-only')
    if result_multi:
        ax.plot(time_sub, result_multi.timeseries.X_runup, 'b-', lw=1, alpha=0.8, label='Multi-signal')
    ax.set_ylabel('Cross-shore (m)')
    ax.set_title('Elevation Timestack with Runup Detection')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Z (m)')

    # Plot 2: Intensity timestack (if available)
    ax = axes[1]
    if has_intensity:
        im = ax.imshow(I_sub, aspect='auto', extent=extent, cmap='gray', vmin=0, vmax=100)
        ax.plot(time_sub, result_elev.timeseries.X_runup, 'r-', lw=1, alpha=0.8, label='Elev-only')
        if result_multi:
            ax.plot(time_sub, result_multi.timeseries.X_runup, 'b-', lw=1, alpha=0.8, label='Multi-signal')
        ax.set_ylabel('Cross-shore (m)')
        ax.set_title('Intensity Timestack')
        plt.colorbar(im, ax=ax, label='Intensity')
    else:
        ax.text(0.5, 0.5, 'No intensity data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Intensity (not available)')

    # Plot 3: Runup position comparison
    ax = axes[2]
    ax.plot(time_sub, result_elev.timeseries.X_runup, 'r-', lw=1, alpha=0.8, label='Elev-only')
    if result_multi:
        ax.plot(time_sub, result_multi.timeseries.X_runup, 'b-', lw=1, alpha=0.8, label='Multi-signal')
    ax.set_ylabel('Runup X (m)')
    ax.set_title('Runup Position Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Adaptive weights over time (if multi-signal)
    ax = axes[3]
    if result_multi and result_multi.timeseries.weights_used.size > 0:
        w = result_multi.timeseries.weights_used
        ax.fill_between(time_sub, 0, w[0, :], alpha=0.7, label='Elevation', color='C0')
        ax.fill_between(time_sub, w[0, :], w[0, :]+w[1, :], alpha=0.7, label='Intensity', color='C1')
        ax.fill_between(time_sub, w[0, :]+w[1, :], 1, alpha=0.7, label='Variance', color='C2')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Signal Weights')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No multi-signal data', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Time (s)')

    plt.tight_layout()

    # Save figure
    out_path = PROJECT_ROOT / "data" / "multisignal_runup_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")

    plt.show()

    ds.close()


if __name__ == "__main__":
    main()
