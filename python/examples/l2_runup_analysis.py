#!/usr/bin/env python3
"""
Example: L2 Wave-Resolving Analysis and Runup Detection

This example demonstrates how to:
1. Process lidar data at 2Hz temporal resolution
2. Extract time-resolved Z(x,t) matrices
3. Detect wave runup and compute statistics
4. Generate spectral analysis

Usage:
    python l2_runup_analysis.py /path/to/livox_config.json
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from phase1 import load_config
from phase4 import process_l2, configure_logging
from runup import compute_runup_stats


def main():
    # Configure logging
    configure_logging(verbose=True, log_file=None, quiet=False)

    # Load configuration
    if len(sys.argv) < 2:
        print("Usage: python l2_runup_analysis.py /path/to/livox_config.json")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    print("Processing L2 data (wave-resolving)...")

    # Process L2 data for one file
    result = process_l2(
        config,
        start_date="2025-05-03",
        end_date="2025-05-04",
        time_bin_size=0.5,        # 2Hz sampling
        apply_outlier_detection=True,
        multi_transect=False,      # Single central transect
        show_progress=True,
    )

    if result is None:
        print("No data processed. Check date range and data folder.")
        return

    print(f"\nL2 Dataset:")
    print(f"  Cross-shore positions: {result.nx}")
    print(f"  Time steps: {result.nt}")
    print(f"  Time resolution: {result.dt:.2f} s")
    print(f"  Duration: {result.nt * result.dt / 60:.1f} minutes")

    # Get the Z(x,t) and I(x,t) matrices
    Z_xt = result.Z_xt  # Uses filtered data if outlier detection was applied
    I_xt = result.I_xt
    x1d = result.x1d
    time_vec = result.time_vec

    # Compute runup statistics
    print("\nComputing runup statistics...")
    Spec, Info, Bulk, Tseries = compute_runup_stats(
        Z_xt, I_xt, x1d, time_vec,
        threshold=0.1,      # 10cm water depth threshold
        windowlength=5,     # 5-minute spectral windows
        plot=False,         # Set True for diagnostic plots
    )

    # Print results
    print("\n" + "=" * 50)
    print("RUNUP STATISTICS")
    print("=" * 50)
    print(f"  Significant IG height (Sig):  {Bulk.swashparams[0]:.3f} m")
    print(f"  Significant Inc height (Sinc): {Bulk.swashparams[1]:.3f} m")
    print(f"  Mean water level (eta):        {Bulk.swashparams[2]:.3f} m")
    print(f"  Beach slope (beta):            {Bulk.beta:.4f}")
    print(f"  2% exceedance (R2):            {Bulk.R2:.3f} m")

    # Save results
    output_path = Path("l2_output.nc")
    result.to_netcdf(output_path)
    print(f"\nSaved L2 data to: {output_path}")

    # Optional: Create visualization
    create_visualization(Z_xt, x1d, time_vec, Tseries)


def create_visualization(Z_xt, x1d, time_vec, Tseries):
    """Create visualization of Z(x,t) and runup time series."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Z(x,t) heatmap
    ax = axes[0]
    # Subsample for visualization if needed
    skip_t = max(1, Z_xt.shape[0] // 500)
    skip_x = max(1, Z_xt.shape[1] // 200)
    Z_sub = Z_xt[::skip_t, ::skip_x]
    t_sub = time_vec[::skip_t]
    x_sub = x1d[::skip_x]

    im = ax.pcolormesh(
        x_sub, t_sub / 60,  # Convert to minutes
        Z_sub,
        cmap='viridis',
        shading='auto',
    )
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Z(x,t) - Elevation Time Stack')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Overlay runup position
    if Tseries is not None and hasattr(Tseries, 'Xrunup'):
        Xrunup = Tseries.Xrunup
        t_runup = np.arange(len(Xrunup)) * (time_vec[1] - time_vec[0]) if len(time_vec) > 1 else np.arange(len(Xrunup))
        valid = ~np.isnan(Xrunup)
        ax.plot(Xrunup[valid], t_runup[valid] / 60, 'r-', linewidth=0.5, label='Runup line')
        ax.legend()

    # Runup time series
    ax = axes[1]
    if Tseries is not None and hasattr(Tseries, 'Zrunup'):
        Zrunup = Tseries.Zrunup
        t_runup = np.arange(len(Zrunup)) * (time_vec[1] - time_vec[0]) if len(time_vec) > 1 else np.arange(len(Zrunup))
        ax.plot(t_runup / 60, Zrunup, 'b-', linewidth=0.5)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Runup elevation (m)')
        ax.set_title('Runup Time Series')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('l2_visualization.png', dpi=150)
    print("Saved visualization to: l2_visualization.png")
    plt.close()


if __name__ == "__main__":
    main()
