#!/usr/bin/env python3
"""
Run L1 processing for each day of LAZ data, producing one NetCDF per day.

This script automatically discovers all LAZ files in the data folder,
determines the date range, and processes each day separately.

Usage:
    python scripts/run_daily_l1.py [options]

Options:
    --config PATH     Path to config file (default: configs/do.json)
    --output-dir DIR  Output directory (default: from config processFolder)
    --resume          Resume from checkpoint if available
    --verbose         Enable debug logging
    --dry-run         Show what would be processed without running
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import phase1
import phase4


def discover_date_range(data_folder: Path) -> tuple[datetime, datetime]:
    """
    Discover the full date range of LAZ files in the data folder.

    Returns (start_date, end_date) where end_date is the day after the last file.
    """
    laz_files = phase1.discover_laz_files(data_folder)
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found in {data_folder}")

    # Get timestamps from all files
    timestamps = [ts for _, ts in laz_files]
    min_ts = min(timestamps)
    max_ts = max(timestamps)

    # Round to day boundaries
    start_date = min_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    # End date is the day after the last file (exclusive)
    end_date = (max_ts + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    return start_date, end_date


def main():
    parser = argparse.ArgumentParser(
        description="Run L1 processing for each day of LAZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/do.json"),
        help="Path to config file (default: configs/do.json). See configs/ for available site configs."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: from config processFolder)"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Override start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Override end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--bin-size", type=float, default=0.1,
        help="Spatial bin size in meters (default: 0.1)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress non-error output"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bars"
    )

    args = parser.parse_args()

    # Configure logging
    phase4.configure_logging(verbose=args.verbose, quiet=args.quiet)
    logger = logging.getLogger(__name__)

    # Load config
    config_path = args.config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        cfg = phase1.load_config(config_path)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        sys.exit(1)

    # Determine output directory (default: python/data/level1/)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "level1"
    output_dir = output_dir.resolve()

    # Discover date range from LAZ files
    logger.info("Scanning for LAZ files in %s", cfg.data_folder)

    try:
        auto_start, auto_end = discover_date_range(cfg.data_folder)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Allow command-line override of date range
    if args.start:
        start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        start_date = auto_start

    if args.end:
        end_date = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    else:
        end_date = auto_end

    # Count days and files
    n_days = (end_date - start_date).days
    laz_files = phase1.discover_laz_files(cfg.data_folder, start=start_date, end=end_date)

    logger.info("Found %d LAZ files spanning %d days", len(laz_files), n_days)
    logger.info("Date range: %s to %s", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    logger.info("Output directory: %s", output_dir)

    if args.dry_run:
        print(f"\nDry run - would process {n_days} days:")
        current = start_date
        while current < end_date:
            day_end = current + timedelta(days=1)
            day_files = phase1.discover_laz_files(cfg.data_folder, start=current, end=day_end)
            output_file = output_dir / f"L1_{current.strftime('%Y%m%d')}.nc"
            status = "exists" if output_file.exists() else "pending"
            print(f"  {current.strftime('%Y-%m-%d')}: {len(day_files)} files -> {output_file.name} [{status}]")
            current = day_end
        return

    # Run batch processing
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress = phase4.process_l1_batch(
            config_path,
            start=start_date,
            end=end_date,
            output_dir=output_dir,
            resume=args.resume,
            show_progress=not args.no_progress,
            bin_size=args.bin_size,
            skip_corrupt=True,
        )

        print(f"\nProcessing complete:")
        print(f"  Succeeded: {progress.completed}")
        print(f"  Failed: {progress.failed}")
        print(f"  Success rate: {progress.success_rate*100:.1f}%")

        if progress.errors:
            print(f"\nFailed days:")
            for date_str, error in progress.errors[:10]:
                print(f"  {date_str}: {error}")
            if len(progress.errors) > 10:
                print(f"  ... and {len(progress.errors) - 10} more")

        # List output files
        nc_files = sorted(output_dir.glob("L1_*.nc"))
        if nc_files:
            print(f"\nOutput files ({len(nc_files)} total):")
            for f in nc_files[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")
            if len(nc_files) > 5:
                print(f"  ... and {len(nc_files) - 5} more")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Use --resume to continue.")
        sys.exit(130)
    except Exception as e:
        logger.error("Processing failed: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
