#!/usr/bin/env python3
"""
Example: Batch Processing with Checkpointing

This example demonstrates how to:
1. Process multiple days of data in batch mode
2. Use checkpointing for resume capability
3. Handle errors gracefully
4. Use parallel processing

Usage:
    python batch_processing.py /path/to/livox_config.json
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from phase1 import load_config
from phase4 import (
    process_l1_batch,
    process_l2_batch,
    configure_logging,
    Checkpoint,
)


def main():
    # Configure logging - write to both console and file
    log_file = Path("batch_processing.log")
    configure_logging(verbose=False, log_file=log_file, quiet=False)

    # Load configuration
    if len(sys.argv) < 2:
        print("Usage: python batch_processing.py /path/to/livox_config.json")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    # Output directory
    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    # Example 1: Basic batch processing
    print("\n--- Example 1: Basic L1 Batch Processing ---")
    run_basic_batch(config, output_dir)

    # Example 2: Resume from checkpoint
    print("\n--- Example 2: Resume from Checkpoint ---")
    demonstrate_checkpoint(config, output_dir)

    # Example 3: Parallel processing
    print("\n--- Example 3: Parallel Processing ---")
    run_parallel_batch(config, output_dir)

    # Example 4: L2 batch processing
    print("\n--- Example 4: L2 Batch Processing ---")
    run_l2_batch(config, output_dir)


def run_basic_batch(config, output_dir):
    """Basic batch processing example."""
    print("Processing 7 days of L1 data...")

    results = process_l1_batch(
        config,
        output_dir=output_dir / "l1_basic",
        start_date="2025-05-01",
        end_date="2025-05-07",
        bin_size=0.10,
        skip_corrupt=True,      # Continue past corrupt files
        show_progress=True,
    )

    print(f"Processed {len(results)} days")
    for date_str, result in results.items():
        if result is not None:
            print(f"  {date_str}: {result.dataset.dims if result.dataset else 'No data'}")


def demonstrate_checkpoint(config, output_dir):
    """Demonstrate checkpoint and resume functionality."""
    checkpoint_dir = output_dir / "l1_checkpoint"
    checkpoint_file = checkpoint_dir / "checkpoint.json"

    # Simulate an interrupted run by processing first 3 days
    print("Simulating interrupted run (first 3 days)...")

    # In a real scenario, this might be interrupted
    results1 = process_l1_batch(
        config,
        output_dir=checkpoint_dir,
        start_date="2025-05-01",
        end_date="2025-05-03",
        bin_size=0.10,
        skip_corrupt=True,
        show_progress=True,
    )

    print(f"First run processed {len(results1)} days")

    # Now resume and process remaining days
    print("\nResuming to complete remaining days...")

    # The resume functionality uses checkpoint file
    results2 = process_l1_batch(
        config,
        output_dir=checkpoint_dir,
        start_date="2025-05-01",
        end_date="2025-05-07",
        bin_size=0.10,
        skip_corrupt=True,
        resume=True,  # Resume from checkpoint
        show_progress=True,
    )

    print(f"Resume run processed {len(results2)} additional days")


def run_parallel_batch(config, output_dir):
    """Parallel batch processing example."""
    print("Processing with parallel workers...")

    results = process_l1_batch(
        config,
        output_dir=output_dir / "l1_parallel",
        start_date="2025-05-01",
        end_date="2025-05-07",
        bin_size=0.10,
        skip_corrupt=True,
        max_workers=4,          # Use 4 parallel workers
        show_progress=True,
    )

    print(f"Processed {len(results)} days with parallel processing")


def run_l2_batch(config, output_dir):
    """L2 batch processing example."""
    print("Processing L2 data (wave-resolving)...")

    results = process_l2_batch(
        config,
        output_dir=output_dir / "l2_batch",
        start_date="2025-05-01",
        end_date="2025-05-03",
        time_bin_size=0.5,      # 2Hz
        apply_outlier_detection=True,
        skip_corrupt=True,
        show_progress=True,
    )

    print(f"Processed {len(results)} L2 datasets")
    for date_str, result in results.items():
        if result is not None:
            print(f"  {date_str}: {result.nt} time steps, {result.nx} positions")


if __name__ == "__main__":
    main()
