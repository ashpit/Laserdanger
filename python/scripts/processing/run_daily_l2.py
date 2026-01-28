#!/usr/bin/env python3
"""
Run L2 processing for each day of LAZ data, producing one NetCDF per day.

L2 produces wave-resolving timestack data: Z(x,t) and I(x,t) matrices
at ~2Hz temporal resolution for wave runup analysis.

This script automatically discovers all LAZ files in the data folder,
determines the date range, and processes each day separately.

Usage:
    python scripts/run_daily_l2.py [options]

Options:
    --config PATH       Path to config file (required, e.g., configs/towr_livox_config_20260120.json)
    --output-dir DIR    Output directory (default: python/data/level2)
    --resume            Resume from checkpoint if available
    --verbose           Enable debug logging
    --dry-run           Show what would be processed without running
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

import mop
import phase1
import phase4

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _make_l2_filename(
    date_str: str,
    expansion_rate: float = None,
    mop_num: float = None,
) -> str:
    """
    Generate L2 output filename, optionally with MOP and/or expansion rate suffix.

    Examples:
        _make_l2_filename("20260120") -> "L2_20260120.nc"
        _make_l2_filename("20260120", mop_num=456) -> "L2_20260120_MOP456.nc"
        _make_l2_filename("20260120", mop_num=456.3) -> "L2_20260120_MOP456.3.nc"
        _make_l2_filename("20260120", 0.02) -> "L2_20260120_exp02.nc"
        _make_l2_filename("20260120", 0.02, 456) -> "L2_20260120_MOP456_exp02.nc"
    """
    base = f"L2_{date_str}"

    # Add MOP suffix if specified
    if mop_num is not None:
        base += mop.format_mop_filename_suffix(mop_num)

    # Add expansion rate suffix if specified
    if expansion_rate is not None and expansion_rate > 0:
        base += f"_exp{int(expansion_rate * 100):02d}"

    return base + ".nc"


def _process_l2_batch_chunked(
    config_path: Path,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    chunk_size: int,
    chunk_dir: Path = None,
    cleanup_chunks: bool = True,
    resume: bool = False,
    show_progress: bool = True,
    expansion_rate: float = None,
    mop_num: float = None,
    **kwargs,
) -> phase4.BatchProgress:
    """
    Process L2 data day-by-day using chunked processing for each day.

    This wrapper iterates over days and uses process_l2_chunked() for each,
    which handles memory-efficient chunk-based processing within each day.
    """
    logger = logging.getLogger(__name__)

    # Generate list of days to process
    days_to_process = []
    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end_date:
        days_to_process.append(current)
        current += timedelta(days=1)

    # Check for existing outputs (for resume)
    completed_days = set()
    if resume:
        for day in days_to_process:
            output_file = output_dir / _make_l2_filename(
                day.strftime('%Y%m%d'), expansion_rate, mop_num
            )
            if output_file.exists():
                completed_days.add(day.strftime("%Y-%m-%d"))
                logger.info("Skipping completed day: %s", day.strftime("%Y-%m-%d"))

    remaining_days = [d for d in days_to_process if d.strftime("%Y-%m-%d") not in completed_days]

    progress = phase4.BatchProgress(total_items=len(days_to_process))
    progress.completed = len(completed_days)

    if not remaining_days:
        logger.info("All days already processed")
        return progress

    logger.info(
        "Processing %d days with chunked processing (%d already completed)",
        len(remaining_days), len(completed_days)
    )

    # Process each day
    day_iter = remaining_days
    if show_progress and TQDM_AVAILABLE:
        day_iter = tqdm(remaining_days, desc="Processing days (chunked)", unit="day")

    for day in day_iter:
        day_str = day.strftime("%Y-%m-%d")
        day_end = day + timedelta(days=1)
        progress.current_item = day_str

        # Set up chunk directory for this day
        if chunk_dir is not None:
            day_chunk_dir = chunk_dir / day.strftime("%Y%m%d")
        else:
            day_chunk_dir = None  # Will use temp dir

        output_file = output_dir / _make_l2_filename(
            day.strftime('%Y%m%d'), expansion_rate, mop_num
        )

        try:
            result = phase4.process_l2_chunked(
                config_path,
                start=day,
                end=day_end,
                chunk_size=chunk_size,
                output_path=output_file,
                chunk_dir=day_chunk_dir,
                resume=resume,
                cleanup_chunks=cleanup_chunks,
                show_progress=False,  # Don't show nested progress
                expansion_rate=expansion_rate,
                mop_num=mop_num,
                **kwargs,
            )

            progress.completed += 1
            completed_days.add(day_str)
            logger.info("Completed %s", day_str)

            # Clear memory between days
            del result
            phase4.clear_memory()

        except (FileNotFoundError, phase4.NoDataError) as e:
            progress.failed += 1
            progress.errors.append((day_str, f"No data: {e}"))
            logger.warning("No data for %s: %s", day_str, e)

        except Exception as e:
            progress.failed += 1
            progress.errors.append((day_str, f"Error: {e}"))
            logger.error("Error processing %s: %s", day_str, e)

    logger.info(
        "Chunked batch complete: %d succeeded, %d failed, %.1f%% success rate",
        progress.completed, progress.failed, progress.success_rate * 100
    )

    return progress


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
        description="Run L2 processing for each day of LAZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_daily_l2.py --config configs/do_livox_config_20260112.json
    python scripts/run_daily_l2.py --config configs/do_livox_config_20260112.json --resume
    python scripts/run_daily_l2.py --config configs/towr_livox_config_20260120.json --start 2026-01-20
    python scripts/run_daily_l2.py --config configs/do_livox_config_20260112.json --time-bin 0.25
        """
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to config file (e.g., configs/towr_livox_config_20260120.json)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: {processFolder}/level2 from config)"
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
        "--time-bin", type=float, default=0.5,
        help="Temporal bin size in seconds (default: 0.5 = 2Hz)"
    )
    parser.add_argument(
        "--x-bin", type=float, default=0.1,
        help="Spatial bin size along transect in meters (default: 0.1)"
    )
    parser.add_argument(
        "--multi-transect", action="store_true",
        help="Extract multiple alongshore transects"
    )
    parser.add_argument(
        "--outlier-detection", action="store_true",
        help="Enable outlier detection (disabled by default to preserve wave signals)"
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
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Process LAZ files in chunks of this size to reduce memory usage. "
             "Recommended: 8-10 for large datasets. If not set, loads all files at once."
    )
    parser.add_argument(
        "--chunk-dir", type=Path, default=None,
        help="Directory for intermediate chunk files (default: temp directory)"
    )
    parser.add_argument(
        "--keep-chunks", action="store_true",
        help="Keep intermediate chunk files after processing (for debugging)"
    )
    parser.add_argument(
        "--expansion-rate", type=float, default=None,
        help="Tolerance expansion rate (m/m) for adaptive transect width. "
             "E.g., 0.02 = tolerance grows by 2cm per meter from scanner. "
             "Output files will be named L2_YYYYMMDD_expNN.nc"
    )
    parser.add_argument(
        "--tolerance", type=float, default=None,
        help="Base transect tolerance in meters (default: 1.0). "
             "Points within this perpendicular distance from the transect line are included. "
             "Increase to capture more data points (e.g., 2.0 or 3.0 for sparse data)."
    )

    # MOP transect options
    parser.add_argument(
        "--mop", type=float, default=None,
        help="Use MOP (Monitoring and Prediction) transect number. "
             "Supports fractional MOPs (e.g., 456.3 interpolates between MOP 456 and 457). "
             "Output files will be named L2_YYYYMMDD_MOP###.nc"
    )
    parser.add_argument(
        "--mop-table", type=Path, default=None,
        help="Path to MOP CSV file (default: python/mop_data/MopTable.csv)"
    )
    parser.add_argument(
        "--auto-mop", action="store_true",
        help="Automatically select the best MOP for the data using the method "
             "specified by --mop-method. Prints selected MOP during dry-run."
    )
    parser.add_argument(
        "--mop-method", type=str, default="centroid",
        choices=["centroid", "coverage", "nearest_scanner"],
        help="Method for auto-MOP selection: 'centroid' (default) selects MOP closest "
             "to data centroid, 'coverage' selects MOP with most points within tolerance, "
             "'nearest_scanner' selects MOP closest to scanner position."
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

    # Determine output directory (default: from config processFolder/level2)
    if args.output_dir is None:
        output_dir = cfg.process_folder / "level2"
    else:
        output_dir = args.output_dir

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
    logger.info("Temporal resolution: %.2f Hz (%.3f s bins)", 1.0/args.time_bin, args.time_bin)
    if args.tolerance is not None:
        logger.info("Base transect tolerance: %.2f m", args.tolerance)
    if args.expansion_rate is not None:
        logger.info("Adaptive tolerance: expansion_rate=%.3f (tolerance grows by %.1fcm per meter from scanner)",
                    args.expansion_rate, args.expansion_rate * 100)

    # Determine MOP number to use (explicit or auto-selected)
    mop_num = args.mop
    if args.auto_mop and mop_num is None:
        # For auto-mop, we'll show the selection in dry-run or let phase4 handle it
        logger.info("Auto-MOP selection enabled (method: %s)", args.mop_method)
    elif mop_num is not None:
        logger.info("Using MOP transect: %s", mop_num)

    if args.dry_run:
        print(f"\nDry run - would process {n_days} days:")
        if args.tolerance is not None:
            print(f"  Using base tolerance: {args.tolerance}m")
        if args.expansion_rate is not None:
            print(f"  Using adaptive tolerance with expansion_rate={args.expansion_rate}")

        # Handle MOP selection in dry run
        if mop_num is not None:
            mop_table = mop.MopTable.load(args.mop_table)
            if mop_num != int(mop_num):
                mop_transect = mop_table.get_fractional_mop(mop_num)
            else:
                mop_transect = mop_table.get_mop(int(mop_num))
            print(f"  Using MOP {mop_num} transect (length={mop_transect.length:.1f}m, azimuth={mop_transect.azimuth:.1f}°)")
        elif args.auto_mop:
            # Load some data to demonstrate auto-selection
            print(f"  Auto-MOP selection enabled (method: {args.mop_method})")
            # Try to select MOP from first day's data
            try:
                first_day = start_date
                first_day_end = first_day + timedelta(days=1)
                first_day_files = phase1.discover_laz_files(cfg.data_folder, start=first_day, end=first_day_end)
                if first_day_files:
                    # Load a sample of points for MOP selection demo
                    sample_points = []
                    for path, _ in first_day_files[:5]:  # Use first 5 files
                        from phase4 import load_laz_points_safe
                        result = load_laz_points_safe(path)
                        if result is not None:
                            pts, _, _ = result
                            sample_points.append(pts)
                    if sample_points:
                        import numpy as np
                        all_pts = np.vstack(sample_points)
                        # Transform points
                        transformed = phase1.transform_points(all_pts, cfg.transform_matrix)
                        X, Y = transformed[:, 0], transformed[:, 1]
                        mop_table = mop.MopTable.load(args.mop_table)
                        scanner_pos = (cfg.transform_matrix[0, 3], cfg.transform_matrix[1, 3])
                        selected = mop.select_best_mop(
                            X, Y, mop_table,
                            scanner_position=scanner_pos,
                            method=args.mop_method
                        )
                        print(f"  Would auto-select MOP {selected} (based on sample data)")
                        mop_num = selected  # Use for filename preview
            except Exception as e:
                print(f"  (Could not preview auto-MOP selection: {e})")

        current = start_date
        while current < end_date:
            day_end = current + timedelta(days=1)
            day_files = phase1.discover_laz_files(cfg.data_folder, start=current, end=day_end)
            output_file = output_dir / _make_l2_filename(
                current.strftime('%Y%m%d'), args.expansion_rate, mop_num
            )
            status = "exists" if output_file.exists() else "pending"
            print(f"  {current.strftime('%Y-%m-%d')}: {len(day_files)} files -> {output_file.name} [{status}]")
            current = day_end
        return

    # Run batch processing
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.chunk_size is not None:
            # Use chunked processing for memory efficiency
            logger.info("Using chunked processing with chunk_size=%d", args.chunk_size)
            progress = _process_l2_batch_chunked(
                config_path=config_path,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                chunk_size=args.chunk_size,
                chunk_dir=args.chunk_dir,
                cleanup_chunks=not args.keep_chunks,
                resume=args.resume,
                show_progress=not args.no_progress,
                expansion_rate=args.expansion_rate,
                tolerance=args.tolerance,
                time_bin_size=args.time_bin,
                x_bin_size=args.x_bin,
                multi_transect=args.multi_transect,
                apply_outlier_detection=args.outlier_detection,
                mop_num=args.mop,
                mop_table_path=args.mop_table,
                auto_mop=args.auto_mop,
                mop_method=args.mop_method,
            )
        else:
            # Standard batch processing (loads all files per day at once)
            progress = phase4.process_l2_batch(
                config_path,
                start=start_date,
                end=end_date,
                output_dir=output_dir,
                file_duration=timedelta(days=1),  # One file per day
                resume=args.resume,
                show_progress=not args.no_progress,
                time_bin_size=args.time_bin,
                x_bin_size=args.x_bin,
                multi_transect=args.multi_transect,
                apply_outlier_detection=args.outlier_detection,
                expansion_rate=args.expansion_rate,
                tolerance=args.tolerance,
                skip_corrupt=True,
                mop_num=args.mop,
                mop_table_path=args.mop_table,
                auto_mop=args.auto_mop,
                mop_method=args.mop_method,
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
        nc_files = sorted(output_dir.glob("L2_*.nc"))
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
