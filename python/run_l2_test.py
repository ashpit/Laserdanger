#!/usr/bin/env python
"""Run L2 processing with proper transect configuration from MATLAB."""

import sys
from pathlib import Path
from datetime import datetime

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

import phase4
import profiles

# Transect configuration from MATLAB Get3_1Dprofiles.m
TRANSECT_CONFIG = profiles.TransectConfig(
    x1=476190.0618275550,   # Back beach x
    y1=3636333.442333425,   # Back beach y
    x2=475620.6132784432,   # Offshore x
    y2=3636465.645345889,   # Offshore y
    alongshore_spacings=(-8, -6, -4, -2, 0, 2, 4, 6, 8, 10),
    resolution=0.25,
    tolerance=1.0,
    extend_line=(0, -300),
)

def main():
    config_path = Path("livox_config.json")
    output_path = Path("tests/data/test_l2.nc")

    print("Running L2 processing with MATLAB transect configuration...")
    print(f"  Transect: ({TRANSECT_CONFIG.x1:.2f}, {TRANSECT_CONFIG.y1:.2f}) -> ({TRANSECT_CONFIG.x2:.2f}, {TRANSECT_CONFIG.y2:.2f})")
    print(f"  Alongshore offsets: {TRANSECT_CONFIG.alongshore_spacings}")

    result = phase4.process_l2(
        config_path,
        max_files=10,  # Process more files to find valid ones
        profile_config=TRANSECT_CONFIG,
        apply_outlier_detection=True,
        show_progress=True,
    )

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(output_path)

    print(f"\nSaved L2 dataset to {output_path}")
    print(f"  Shape: Z_xt = {result.Z_xt.shape} (n_x, n_t)")
    print(f"  Time range: {result.time_vec[0]:.1f}s to {result.time_vec[-1]:.1f}s")
    print(f"  X range: {result.x1d[0]:.1f}m to {result.x1d[-1]:.1f}m")
    print(f"  dt = {result.dt:.2f}s, dx = {result.dx:.2f}m")

if __name__ == "__main__":
    main()
