#!/usr/bin/env python3
"""
Test multi-signal runup detection on all valid data bursts.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import runup


def find_valid_regions(Z_xt, time_sec):
    """Find contiguous regions with valid data."""
    valid_cols = ~np.all(np.isnan(Z_xt), axis=0)

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


def main():
    # Load L2 data
    l2_path = Path("/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/github/Laserdanger/python/data/TOWR-lidar/surfside-test_20260120/lvx_processed/level2/L2_20260120.nc")

    print(f"Loading {l2_path}...")
    ds = xr.open_dataset(l2_path)

    # Extract arrays
    if "elevation" in ds.data_vars:
        Z_xt = ds["elevation"].values
    else:
        Z_xt = ds["Z"].values

    x1d = ds["x"].values

    if "time_seconds" in ds.coords:
        time_sec = ds["time_seconds"].values
    else:
        time_vec = ds["time"].values
        time_sec = (time_vec - time_vec[0]) / np.timedelta64(1, 's')

    I_xt = ds["intensity"].values if "intensity" in ds.data_vars else None

    print(f"Data shape: {Z_xt.shape}")

    # Find valid regions
    regions = find_valid_regions(Z_xt, time_sec)
    print(f"\nFound {len(regions)} valid data bursts")

    # Process each region
    all_results = []

    for i, region in enumerate(regions):
        s, e = region['start_idx'], region['end_idx']

        Z_sub = Z_xt[:, s:e]
        I_sub = I_xt[:, s:e] if I_xt is not None else None
        time_sub = time_sec[s:e]

        print(f"\n--- Burst {i+1}: {region['start_time']/60:.1f}-{region['end_time']/60:.1f} min ({region['n_timesteps']} timesteps) ---")

        # Run multi-signal detection
        result = runup.compute_runup_stats(
            Z_sub, x1d, time_sub,
            I_xt=I_sub,
            threshold=0.1,
            ig_length=100.0,
        )

        valid_frac = result.info['n_valid'] / len(time_sub) * 100
        print(f"  Valid detections: {result.info['n_valid']}/{len(time_sub)} ({valid_frac:.0f}%)")
        print(f"  Sig (IG): {result.bulk.Sig:.3f} m")
        print(f"  Sinc: {result.bulk.Sinc:.3f} m")

        if result.info.get('multisignal_enabled'):
            print(f"  Weights: elev={result.info['mean_weight_elevation']:.2f}, "
                  f"int={result.info['mean_weight_intensity']:.2f}, "
                  f"var={result.info['mean_weight_variance']:.2f}")

        all_results.append({
            'region': region,
            'result': result,
            'Z_sub': Z_sub,
            'I_sub': I_sub,
            'time_sub': time_sub,
        })

    # Create multi-panel figure showing all bursts
    n_bursts = len(all_results)
    fig, axes = plt.subplots(n_bursts, 2, figsize=(14, 3*n_bursts))

    if n_bursts == 1:
        axes = axes.reshape(1, 2)

    for i, data in enumerate(all_results):
        region = data['region']
        result = data['result']
        Z_sub = data['Z_sub']
        I_sub = data['I_sub']
        time_sub = data['time_sub']

        # Left: Elevation timestack with runup
        ax = axes[i, 0]
        extent = [time_sub[0], time_sub[-1], x1d[-1], x1d[0]]
        im = ax.imshow(Z_sub, aspect='auto', extent=extent, cmap='terrain', vmin=-1, vmax=3)

        # Plot runup line
        valid = ~np.isnan(result.timeseries.X_runup)
        if valid.any():
            ax.plot(time_sub[valid], result.timeseries.X_runup[valid], 'b-', lw=1.5, label='Runup')

        ax.set_ylabel('Cross-shore (m)')
        ax.set_title(f'Burst {i+1}: {region["start_time"]/60:.1f}-{region["end_time"]/60:.1f} min')
        if i == 0:
            ax.legend(loc='upper right')

        # Right: Runup position over time
        ax = axes[i, 1]
        if valid.any():
            ax.plot(time_sub, result.timeseries.X_runup, 'b-', lw=1)
            ax.set_ylim([np.nanmin(result.timeseries.X_runup) - 5,
                        np.nanmax(result.timeseries.X_runup) + 5])
        ax.set_ylabel('Runup X (m)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Sig={result.bulk.Sig:.2f}m, Sinc={result.bulk.Sinc:.2f}m')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = PROJECT_ROOT / "data" / "multisignal_all_bursts.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")

    plt.show()
    ds.close()


if __name__ == "__main__":
    main()
