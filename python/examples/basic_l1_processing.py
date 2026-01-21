#!/usr/bin/env python3
"""
Example: Basic L1 Processing

This example demonstrates how to process lidar data through the L1 pipeline
to generate beach surface DEMs.

Usage:
    python basic_l1_processing.py /path/to/livox_config.json
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from phase1 import load_config
from phase4 import process_l1, configure_logging


def main():
    # Configure logging for verbose output
    configure_logging(verbose=True, log_file=None, quiet=False)

    # Load configuration
    if len(sys.argv) < 2:
        print("Usage: python basic_l1_processing.py /path/to/livox_config.json")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    print(f"Data folder: {config['dataFolder']}")
    print(f"Process folder: {config['processFolder']}")

    # Process one day of data
    # Adjust dates to match your available data
    result = process_l1(
        config,
        start_date="2025-05-03",
        end_date="2025-05-04",
        bin_size=0.10,           # 10cm spatial resolution
        mode_bin=0.05,           # 5cm mode histogram bins
        apply_residual_filter=True,
        show_progress=True,
    )

    # Check results
    if result.dataset is not None:
        print("\nL1 Processing Complete!")
        print(f"Dataset dimensions: {dict(result.dataset.dims)}")
        print(f"Variables: {list(result.dataset.data_vars)}")
        print(f"Number of profiles: {len(result.profiles)}")

        # Save to NetCDF
        output_path = Path("l1_output.nc")
        result.dataset.to_netcdf(output_path)
        print(f"\nSaved to: {output_path}")

        # Print some statistics
        z_mean = result.dataset["z_mean"].values
        valid = ~np.isnan(z_mean)
        print(f"\nElevation statistics:")
        print(f"  Min: {np.nanmin(z_mean):.2f} m")
        print(f"  Max: {np.nanmax(z_mean):.2f} m")
        print(f"  Valid cells: {np.sum(valid)} / {z_mean.size}")
    else:
        print("No data processed. Check date range and data folder.")


if __name__ == "__main__":
    import numpy as np
    main()
