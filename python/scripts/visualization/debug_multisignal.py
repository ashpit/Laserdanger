#!/usr/bin/env python3
"""Debug the multi-signal runup detection - visualize individual signals."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import runup


def main():
    # Load L2 data
    l2_path = Path("/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/github/Laserdanger/python/data/TOWR-lidar/surfside-test_20260120/lvx_processed/level2/L2_20260120.nc")

    ds = xr.open_dataset(l2_path)
    Z_xt = ds["elevation"].values
    I_xt = ds["intensity"].values
    x1d = ds["x"].values
    time_sec = ds["time_seconds"].values

    # Use small subset for debugging
    t_slice = slice(0, 200)  # First 200 timesteps
    Z_sub = Z_xt[:, t_slice]
    I_sub = I_xt[:, t_slice]
    time_sub = time_sec[t_slice]

    dt = np.median(np.diff(time_sub))

    print(f"x1d range: {x1d.min():.1f} to {x1d.max():.1f}")
    print(f"x1d[0] = {x1d[0]:.1f}, x1d[-1] = {x1d[-1]:.1f}")
    print(f"Which end has waves? (looking at elevation variance)")

    # Check which end has more variance (waves)
    var_first_50 = np.nanvar(Z_sub[:50, :])
    var_last_50 = np.nanvar(Z_sub[-50:, :])
    print(f"  Variance in first 50 x positions: {var_first_50:.4f}")
    print(f"  Variance in last 50 x positions: {var_last_50:.4f}")

    # Compute references
    dry_beach = runup.compute_dry_beach_reference(Z_sub, dt, ig_length=100.0)
    I_dry_ref = runup.compute_dry_intensity_reference(I_sub, dt, window_seconds=100.0)

    # Compute signals
    p_elev = runup.compute_elevation_signal(Z_sub, dry_beach, threshold=0.1)
    p_int = runup.compute_intensity_signal(I_sub, I_dry_ref, scale=20.0)
    p_var = runup.compute_variance_signal(Z_sub, dt, window_seconds=2.5)

    # Pick a single time slice to visualize
    t_idx = 50

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # Left column: spatial profiles at one time
    ax = axes[0, 0]
    ax.plot(x1d, Z_sub[:, t_idx], 'b-', label='Z')
    ax.plot(x1d, dry_beach[:, t_idx], 'r--', label='dry_beach')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Elevation at t={time_sub[t_idx]:.1f}s')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x1d, I_sub[:, t_idx], 'b-', label='I')
    ax.plot(x1d, I_dry_ref[:, t_idx], 'r--', label='I_dry_ref')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Intensity')
    ax.set_title('Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(x1d, p_elev[:, t_idx], 'C0-', label='P(elev)')
    ax.plot(x1d, p_int[:, t_idx], 'C1-', label='P(int)')
    ax.plot(x1d, p_var[:, t_idx], 'C2-', label='P(var)')
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Probability')
    ax.set_title('Individual Signal Probabilities')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Simple average fusion for visualization
    P_fused = (p_elev + p_int + p_var) / 3
    ax = axes[3, 0]
    ax.plot(x1d, P_fused[:, t_idx], 'k-', lw=2)
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(x1d, 0, P_fused[:, t_idx], where=P_fused[:, t_idx] > 0.5, alpha=0.3, color='blue', label='P > 0.5 (water)')
    ax.fill_between(x1d, 0, P_fused[:, t_idx], where=P_fused[:, t_idx] <= 0.5, alpha=0.3, color='tan', label='P <= 0.5 (dry)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Fused Probability')
    ax.set_title('Fused Probability (simple average)')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Right column: timestacks
    extent = [time_sub[0], time_sub[-1], x1d[-1], x1d[0]]

    ax = axes[0, 1]
    im = ax.imshow(Z_sub, aspect='auto', extent=extent, cmap='terrain', vmin=-1, vmax=3)
    ax.set_ylabel('x (m)')
    ax.set_title('Elevation Timestack')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(I_sub, aspect='auto', extent=extent, cmap='gray', vmin=0, vmax=100)
    ax.set_ylabel('x (m)')
    ax.set_title('Intensity Timestack')
    plt.colorbar(im, ax=ax)

    ax = axes[2, 1]
    im = ax.imshow(p_elev, aspect='auto', extent=extent, cmap='RdYlBu', vmin=0, vmax=1)
    ax.set_ylabel('x (m)')
    ax.set_title('P(elevation) - blue=water')
    plt.colorbar(im, ax=ax)

    ax = axes[3, 1]
    im = ax.imshow(P_fused, aspect='auto', extent=extent, cmap='RdYlBu', vmin=0, vmax=1)
    ax.set_ylabel('x (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Fused Probability - blue=water')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    out_path = PROJECT_ROOT / "data" / "debug_signals.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")

    # Print some diagnostics
    print(f"\nAt t={time_sub[t_idx]:.1f}s:")
    print(f"  Mean P(elev): {np.nanmean(p_elev[:, t_idx]):.3f}")
    print(f"  Mean P(int): {np.nanmean(p_int[:, t_idx]):.3f}")
    print(f"  Mean P(var): {np.nanmean(p_var[:, t_idx]):.3f}")

    # Where is P > 0.5?
    water_mask = P_fused[:, t_idx] > 0.5
    if water_mask.any():
        water_x = x1d[water_mask]
        print(f"  'Water' detected at x: {water_x.min():.1f} to {water_x.max():.1f}")
    else:
        print(f"  No 'water' detected (P > 0.5)")

    plt.show()
    ds.close()


if __name__ == "__main__":
    main()
