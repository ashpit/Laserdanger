"""
Phase 4: orchestration/CLI-style drivers for L1/L2 processing.
This keeps I/O thin and composes the pure functions from phases 1–3.

Features:
- L1 processing (beach surface generation)
- L2 processing (wave-resolving analysis)
- Batch processing with checkpointing and resume
- Progress bars and configurable logging
- Graceful error handling for corrupt files
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

import phase1
import phase2
import phase3
import profiles
import utils

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars, fall back to simple iteration
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        """Fallback when tqdm is not installed."""
        return iterable

# Check for lazrs (fast Rust-based LAZ decompression)
LAZRS_AVAILABLE = False
try:
    import lazrs
    LAZRS_AVAILABLE = True
except ImportError:
    pass

# Loader type: given a path, return (points[N,3], intensities[N], gps_times[N])
LoaderFn = Callable[[Path], Tuple[np.ndarray, np.ndarray, np.ndarray]]


# =============================================================================
# Custom Exceptions
# =============================================================================

class LidarProcessingError(Exception):
    """Base exception for lidar processing errors."""
    pass


class CorruptFileError(LidarProcessingError):
    """Raised when a LAZ file is corrupt or unreadable."""
    def __init__(self, path: Path, reason: str = ""):
        self.path = path
        self.reason = reason
        msg = f"Corrupt or unreadable LAZ file: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class NoDataError(LidarProcessingError):
    """Raised when no valid data is available after filtering."""
    def __init__(self, context: str = ""):
        msg = "No valid data available"
        if context:
            msg += f": {context}"
        super().__init__(msg)


class ConfigurationError(LidarProcessingError):
    """Raised when configuration is invalid or missing."""
    pass


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class L1Result:
    """Result from L1 processing including profiles."""
    dataset: phase3.xr.Dataset
    profiles: Optional[profiles.TransectResult] = None
    profile_config: Optional[profiles.TransectConfig] = None


@dataclass
class BatchProgress:
    """Progress information for batch processing."""
    total_items: int
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    current_item: str = ""
    errors: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.completed + self.failed == 0:
            return 0.0
        return self.completed / (self.completed + self.failed)


@dataclass
class Checkpoint:
    """Checkpoint for resumable batch processing."""
    config_path: str
    output_dir: str
    start_date: str
    end_date: str
    completed_dates: List[str]
    failed_dates: List[str]
    kwargs: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "config_path": self.config_path,
            "output_dir": self.output_dir,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "completed_dates": self.completed_dates,
            "failed_dates": self.failed_dates,
            "kwargs": self.kwargs,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Checkpoint":
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    """
    Configure logging for the pipeline.

    Parameters
    ----------
    verbose : bool
        Enable debug-level logging
    log_file : Path, optional
        Write logs to file in addition to console
    quiet : bool
        Suppress all console output except errors
    """
    level = logging.DEBUG if verbose else logging.INFO
    if quiet:
        level = logging.ERROR

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)


# =============================================================================
# LAZ File Loading with Error Handling
# =============================================================================

_lazrs_warning_shown = False

def load_laz_points(
    path: Path,
    validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a .laz file using laspy and return points, intensities, and GPS times.

    Uses lazrs backend for faster decompression if available (1.5-2x speedup).
    Install with: pip install lazrs

    Parameters
    ----------
    path : Path
        Path to LAZ file
    validate : bool
        Perform basic validation on loaded data

    Returns
    -------
    xyz : ndarray (N, 3)
        Point coordinates
    intensity : ndarray (N,)
        Intensity values
    gps_time : ndarray (N,)
        GPS timestamps

    Raises
    ------
    CorruptFileError
        If file cannot be read or contains invalid data
    """
    global _lazrs_warning_shown

    try:
        import laspy
    except ImportError:
        raise LidarProcessingError(
            "laspy is required for LAZ file reading. "
            "Install with: pip install laspy"
        )

    # Warn once if lazrs is not available
    if not LAZRS_AVAILABLE and not _lazrs_warning_shown:
        logger.warning(
            "lazrs not installed - LAZ decompression may be slow. "
            "Install with: pip install lazrs"
        )
        _lazrs_warning_shown = True

    try:
        with laspy.open(path) as f:
            hdr = f.header
            pts = f.read()
    except Exception as e:
        raise CorruptFileError(path, str(e))

    try:
        xyz = np.vstack((
            pts.X * hdr.scale[0] + hdr.offset[0],
            pts.Y * hdr.scale[1] + hdr.offset[1],
            pts.Z * hdr.scale[2] + hdr.offset[2]
        )).T.astype(float)

        intensity = np.asarray(pts.intensity, dtype=float)

        if hasattr(pts, "gps_time"):
            gps_time = np.asarray(pts.gps_time, dtype=float)
        else:
            gps_time = np.arange(len(xyz), dtype=float)

    except Exception as e:
        raise CorruptFileError(path, f"Failed to extract point data: {e}")

    # Validation
    if validate:
        if len(xyz) == 0:
            raise CorruptFileError(path, "File contains no points")

        if np.any(np.isnan(xyz)) or np.any(np.isinf(xyz)):
            n_invalid = np.sum(np.isnan(xyz) | np.isinf(xyz))
            logger.warning(
                "File %s contains %d invalid coordinates (NaN/Inf)",
                path.name, n_invalid
            )
            # Filter out invalid points
            valid_mask = ~(np.any(np.isnan(xyz), axis=1) | np.any(np.isinf(xyz), axis=1))
            xyz = xyz[valid_mask]
            intensity = intensity[valid_mask]
            gps_time = gps_time[valid_mask]

            if len(xyz) == 0:
                raise CorruptFileError(path, "All points have invalid coordinates")

    return xyz, intensity, gps_time


def load_laz_points_safe(
    path: Path,
    validate: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Safely load LAZ file, returning None on failure.

    Use this for batch processing where individual file failures
    should not stop the entire process.
    """
    try:
        return load_laz_points(path, validate=validate)
    except CorruptFileError as e:
        logger.error("Skipping corrupt file: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading %s: %s", path, e)
        return None


def load_laz_files_parallel(
    laz_files: List[Tuple[Path, datetime]],
    max_workers: int = 4,
    validate: bool = True,
    show_progress: bool = False,
) -> List[Tuple[Path, datetime, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    """
    Load multiple LAZ files in parallel using ThreadPoolExecutor.

    Uses threads (not processes) because LAZ decompression releases the GIL
    and I/O is the bottleneck. Typically 3-4x faster than sequential loading.

    Parameters
    ----------
    laz_files : list
        List of (path, timestamp) tuples
    max_workers : int
        Maximum parallel workers (default 4)
    validate : bool
        Validate loaded data (default True)
    show_progress : bool
        Show progress bar (default False)

    Returns
    -------
    list
        List of (path, timestamp, data) tuples where data is
        (points, intensities, gps_times) or None if loading failed
    """
    results = []

    def load_single(item: Tuple[Path, datetime]) -> Tuple[Path, datetime, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        path, ts = item
        try:
            data = load_laz_points(path, validate=validate)
            return (path, ts, data)
        except CorruptFileError as e:
            logger.warning("Skipping corrupt file %s: %s", path.name, e.reason or str(e))
            return (path, ts, None)
        except Exception as e:
            logger.warning("Error loading %s: %s", path.name, e)
            return (path, ts, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single, item): item for item in laz_files}

        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Loading files", unit="file")
        else:
            iterator = as_completed(futures)

        for future in iterator:
            results.append(future.result())

    # Sort results to maintain original order (as_completed returns in completion order)
    path_order = {item[0]: i for i, item in enumerate(laz_files)}
    results.sort(key=lambda x: path_order.get(x[0], 0))

    return results


def process_l1(
    config_path: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    bin_size: float = 0.1,
    mode_bin: float = 0.05,
    intensity_threshold: float = 100.0,
    max_seconds: Optional[float] = 300.0,
    loader: Optional[LoaderFn] = None,
    data_folder_override: Optional[Path] = None,
    edge_percentile: Optional[Tuple[float, float]] = (1.0, 99.0),
    edge_padding_bins: int = 1,
    apply_residual_filter: bool = True,
    residual_filter_passes: Optional[List[Tuple[float, float]]] = None,
    extract_profiles: bool = False,
    profile_config: Optional[profiles.TransectConfig] = None,
    show_progress: bool = False,
    skip_corrupt: bool = True,
    max_files: Optional[int] = None,
) -> L1Result:
    """
    Orchestrate L1 processing: discover files -> load -> transform/filter ->
    residual filter -> bin -> stack -> extract profiles.

    Parameters
    ----------
    config_path : Path
        Path to livox_config.json
    start, end : datetime, optional
        Date range to process
    bin_size : float
        Spatial bin size in meters (default 0.1)
    mode_bin : float
        Mode quantization bin size (default 0.05)
    intensity_threshold : float
        Maximum intensity to keep (default 100)
    max_seconds : float, optional
        Time window per file in seconds (default 300 = 5 min)
    loader : callable, optional
        Custom LAZ loader function
    data_folder_override : Path, optional
        Override data folder from config
    edge_percentile : tuple, optional
        Percentile range for edge clipping (default 1-99%)
    edge_padding_bins : int
        Extra bins to add at edges (default 1)
    apply_residual_filter : bool
        Apply two-stage residual kernel filtering (default True)
    residual_filter_passes : list, optional
        Filter passes as [(cell_size, threshold), ...]. Default: [(10, 0.2), (3, 0.1)]
    extract_profiles : bool
        Extract 1D cross-shore profiles (default False)
    profile_config : TransectConfig, optional
        Configuration for profile extraction
    show_progress : bool
        Show progress bar (default False)
    skip_corrupt : bool
        Skip corrupt files instead of raising error (default True)

    Returns
    -------
    L1Result
        Contains xarray.Dataset and optional profile results

    Raises
    ------
    FileNotFoundError
        If no matching LAZ files found
    NoDataError
        If all files are filtered out
    CorruptFileError
        If skip_corrupt=False and a corrupt file is encountered
    """
    try:
        cfg = phase1.load_config(config_path)
    except Exception as e:
        raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    if data_folder_override is not None:
        cfg = replace(cfg, data_folder=Path(data_folder_override))

    logger.info("Using data folder: %s", cfg.data_folder)

    if not cfg.data_folder.exists():
        raise ConfigurationError(f"Data folder does not exist: {cfg.data_folder}")

    laz_files = phase1.discover_laz_files(cfg.data_folder, start=start, end=end)
    if not laz_files:
        date_range = ""
        if start or end:
            date_range = f" in range {start} to {end}"
        raise FileNotFoundError(f"No matching .laz files found in {cfg.data_folder}{date_range}")

    # Limit number of files for testing
    if max_files is not None and max_files > 0:
        laz_files = laz_files[:max_files]
        logger.info("Limited to %d files (--n flag)", max_files)

    logger.info("Found %d LAZ files to process", len(laz_files))

    load_fn = loader or load_laz_points
    batches = []
    corrupt_files = []

    # Process files with optional progress bar
    file_iter = laz_files
    if show_progress and TQDM_AVAILABLE:
        file_iter = tqdm(laz_files, desc="Loading files", unit="file")

    for path, ts in file_iter:
        logger.debug("Loading %s", path.name)

        # Load with error handling
        try:
            if loader is not None:
                points, intensities, gps_times = loader(path)
            else:
                points, intensities, gps_times = load_laz_points(path, validate=True)
        except CorruptFileError as e:
            if skip_corrupt:
                logger.warning("Skipping file %s: %s", path.name, e.reason or str(e))
                corrupt_files.append((path, str(e)))
                continue
            else:
                raise

        try:
            f_points, f_intensity, f_times = phase1.prepare_batch(
                points,
                intensities,
                gps_times,
                cfg,
                intensity_threshold=intensity_threshold,
                max_seconds=max_seconds,
            )
        except Exception as e:
            logger.warning("Failed to prepare batch for %s: %s", path.name, e)
            continue

        if len(f_points) == 0:
            logger.debug("No points after filtering for file %s", path)
            continue

        # Apply two-stage residual kernel filtering
        if apply_residual_filter and len(f_points) > 100:
            logger.debug("Applying residual filter to %d points", len(f_points))
            try:
                f_points = phase2.residual_kernel_filter_two_stage(
                    f_points,
                    passes=residual_filter_passes,
                )
            except Exception as e:
                logger.warning("Residual filtering failed for %s: %s", path.name, e)
                # Continue with unfiltered points
            logger.debug("After filtering: %d points", len(f_points))

        if len(f_points) == 0:
            logger.debug("No points after residual filtering for file %s", path)
            continue

        batches.append((f_points, ts))

    if corrupt_files:
        logger.warning("Skipped %d corrupt files", len(corrupt_files))

    if not batches:
        raise NoDataError("All files filtered out; no grids produced")

    logger.info("Processing %d valid batches", len(batches))

    # Use common bin edges across all batches so datasets align
    x_min = min(np.min(b[0][:, 0]) for b in batches)
    x_max = max(np.max(b[0][:, 0]) for b in batches)
    y_min = min(np.min(b[0][:, 1]) for b in batches)
    y_max = max(np.max(b[0][:, 1]) for b in batches)
    if edge_percentile:
        lo, hi = edge_percentile
        x_samples = np.hstack([np.percentile(b[0][:, 0], [lo, hi]) for b in batches])
        y_samples = np.hstack([np.percentile(b[0][:, 1], [lo, hi]) for b in batches])
        x_min = max(x_min, x_samples.min() - edge_padding_bins * bin_size)
        x_max = min(x_max, x_samples.max() + edge_padding_bins * bin_size)
        y_min = max(y_min, y_samples.min() - edge_padding_bins * bin_size)
        y_max = min(y_max, y_samples.max() + edge_padding_bins * bin_size)

    x_edges = phase2.bin_edges(np.array([x_min, x_max]), bin_size)
    y_edges = phase2.bin_edges(np.array([y_min, y_max]), bin_size)

    # Bin with optional progress bar
    grids = []
    batch_iter = batches
    if show_progress and TQDM_AVAILABLE:
        batch_iter = tqdm(batches, desc="Binning", unit="batch")

    for f_points, ts in batch_iter:
        grid = phase2.bin_point_cloud(
            f_points, bin_size=bin_size, mode_bin=mode_bin, x_edges=x_edges, y_edges=y_edges
        )
        grids.append(phase3.GridWithTime(grid=grid, timestamp=ts))

    dataset = phase3.build_dataset(grids)

    # Extract profiles if requested
    profile_result = None
    if extract_profiles:
        logger.info("Extracting profiles")
        # Combine all points for profile extraction
        all_points = np.vstack([b[0] for b in batches])
        X, Y, Z = all_points[:, 0], all_points[:, 1], all_points[:, 2]

        if profile_config is not None:
            profile_result = profiles.extract_transects(X, Y, Z, config=profile_config)
        else:
            logger.warning("Profile extraction requested but no profile_config provided")

    return L1Result(
        dataset=dataset,
        profiles=profile_result,
        profile_config=profile_config,
    )


def process_l2(
    config_path: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    x_bin_size: float = 0.1,
    time_bin_size: float = 0.5,
    intensity_threshold: float = 100.0,
    max_seconds: Optional[float] = 120.0,
    loader: Optional[LoaderFn] = None,
    data_folder_override: Optional[Path] = None,
    apply_residual_filter: bool = True,
    residual_filter_passes: Optional[List[Tuple[float, float]]] = None,
    profile_config: Optional[profiles.TransectConfig] = None,
    apply_outlier_detection: bool = True,
    outlier_params: Optional[Dict] = None,
    extract_intensity_contours: bool = False,
    intensity_contour_thresholds: Optional[List[float]] = None,
    multi_transect: bool = False,
    show_progress: bool = False,
    skip_corrupt: bool = True,
    max_files: Optional[int] = None,
    parallel_load: bool = True,
    load_workers: int = 4,
) -> phase3.TimeResolvedDataset:
    """
    Orchestrate L2 processing: produces time-resolved Z(x,t) matrices for wave analysis.

    L2 differs from L1:
    - Temporal binning at ~2Hz (time_bin_size=0.5s default)
    - Shorter time window (2 min default vs 5 min for L1)
    - Less aggressive filtering
    - Outputs along cross-shore transect(s)
    - Optional 2D outlier detection for spike removal
    - Optional intensity contour extraction for water edge detection

    Parameters
    ----------
    config_path : Path
        Path to livox_config.json
    start, end : datetime, optional
        Date range to process
    x_bin_size : float
        Spatial bin size along transect (default 0.1m)
    time_bin_size : float
        Temporal bin size in seconds (default 0.5 = 2Hz)
    intensity_threshold : float
        Maximum intensity to keep (default 100)
    max_seconds : float
        Time window per file (default 120 = 2 min)
    loader : callable, optional
        Custom LAZ loader function
    data_folder_override : Path, optional
        Override data folder from config
    apply_residual_filter : bool
        Apply residual kernel filtering (default True, but less aggressive)
    residual_filter_passes : list, optional
        Filter passes. Default for L2: [(2, 0.5)] - less aggressive than L1
    profile_config : TransectConfig, optional
        Configuration for transect extraction. Required for multi_transect=True.
    apply_outlier_detection : bool
        Apply 2D gradient/Laplacian outlier detection (default True)
    outlier_params : dict, optional
        Parameters for detect_outliers_conv2d(). Keys:
        - ig_length: IG window in seconds (default 60)
        - gradient_threshold_std: Gradient threshold in std (default 2.5)
        - laplacian_threshold_std: Laplacian threshold in std (default 2.5)
    extract_intensity_contours : bool
        Extract intensity contours for water edge detection (default False)
    intensity_contour_thresholds : list, optional
        Intensity thresholds for contour extraction (default [20, 30, 40, 50])
    multi_transect : bool
        Extract multiple transects at all alongshore offsets in profile_config
        (default False = single central transect only)
    show_progress : bool
        Show progress bar (default False)
    skip_corrupt : bool
        Skip corrupt files instead of raising error (default True)
    parallel_load : bool
        Use parallel file loading with ThreadPoolExecutor (default True).
        Typically 3-4x faster for I/O-bound LAZ decompression.
    load_workers : int
        Number of parallel workers for file loading (default 4)

    Returns
    -------
    TimeResolvedDataset
        Contains Z(x,t), I(x,t), outlier mask, intensity contours, and metadata
    """
    if residual_filter_passes is None:
        # L2 uses less aggressive filtering
        residual_filter_passes = [(2.0, 0.5)]

    try:
        cfg = phase1.load_config(config_path)
    except Exception as e:
        raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    if data_folder_override is not None:
        cfg = replace(cfg, data_folder=Path(data_folder_override))

    laz_files = phase1.discover_laz_files(cfg.data_folder, start=start, end=end)
    if not laz_files:
        raise FileNotFoundError("No matching .laz files found")

    # Limit number of files for testing
    if max_files is not None and max_files > 0:
        laz_files = laz_files[:max_files]
        logger.info("Limited to %d files (--n flag)", max_files)

    logger.info("Found %d LAZ files for L2 processing", len(laz_files))

    all_points = []
    all_intensities = []
    all_times = []
    base_time = None

    # Load files - parallel or sequential
    if parallel_load and loader is None and len(laz_files) > 1:
        # Use parallel loading (3-4x faster for I/O-bound decompression)
        logger.debug("Loading %d files in parallel with %d workers", len(laz_files), load_workers)
        loaded_files = load_laz_files_parallel(
            laz_files,
            max_workers=load_workers,
            validate=True,
            show_progress=show_progress,
        )

        # Process loaded files with optional progress bar
        process_iter = loaded_files
        if show_progress and TQDM_AVAILABLE:
            process_iter = tqdm(loaded_files, desc="Processing L2 files", unit="file")

        for path, ts, data in process_iter:
            if data is None:
                if not skip_corrupt:
                    raise CorruptFileError(path, "Failed to load")
                continue

            points, intensities, gps_times = data

            f_points, f_intensity, f_times = phase1.prepare_batch(
                points,
                intensities,
                gps_times,
                cfg,
                intensity_threshold=intensity_threshold,
                max_seconds=max_seconds,
            )
            if len(f_points) == 0:
                logger.warning("No points after filtering for file %s", path)
                continue

            # Apply residual filtering (less aggressive for L2)
            if apply_residual_filter and len(f_points) > 100:
                f_points = phase2.residual_kernel_filter_two_stage(
                    f_points,
                    passes=residual_filter_passes,
                )

            if len(f_points) == 0:
                continue

            # Track absolute time
            if base_time is None:
                base_time = ts

            # Convert file timestamp offset to seconds
            file_offset = (ts - base_time).total_seconds()
            absolute_times = f_times + file_offset

            all_points.append(f_points)
            all_intensities.append(f_intensity[:len(f_points)] if len(f_intensity) >= len(f_points)
                                   else np.full(len(f_points), np.nan))
            all_times.append(absolute_times[:len(f_points)] if len(absolute_times) >= len(f_points)
                            else np.arange(len(f_points)) + file_offset)
    else:
        # Sequential loading (for custom loaders or single files)
        file_iter = laz_files
        if show_progress and TQDM_AVAILABLE:
            file_iter = tqdm(laz_files, desc="Loading L2 files", unit="file")

        for path, ts in file_iter:
            logger.debug("Loading %s for L2", path.name)

            # Load with error handling
            try:
                if loader is not None:
                    points, intensities, gps_times = loader(path)
                else:
                    points, intensities, gps_times = load_laz_points(path, validate=True)
            except CorruptFileError as e:
                if skip_corrupt:
                    logger.warning("Skipping file %s: %s", path.name, e.reason or str(e))
                    continue
                else:
                    raise

            f_points, f_intensity, f_times = phase1.prepare_batch(
                points,
                intensities,
                gps_times,
                cfg,
                intensity_threshold=intensity_threshold,
                max_seconds=max_seconds,
            )
            if len(f_points) == 0:
                logger.warning("No points after filtering for file %s", path)
                continue

            # Apply residual filtering (less aggressive for L2)
            if apply_residual_filter and len(f_points) > 100:
                f_points = phase2.residual_kernel_filter_two_stage(
                    f_points,
                    passes=residual_filter_passes,
                )

            if len(f_points) == 0:
                continue

            # Track absolute time
            if base_time is None:
                base_time = ts

            # Convert file timestamp offset to seconds
            file_offset = (ts - base_time).total_seconds()
            absolute_times = f_times + file_offset

            all_points.append(f_points)
            all_intensities.append(f_intensity[:len(f_points)] if len(f_intensity) >= len(f_points)
                                   else np.full(len(f_points), np.nan))
            all_times.append(absolute_times[:len(f_points)] if len(absolute_times) >= len(f_points)
                            else np.arange(len(f_points)) + file_offset)

    if not all_points:
        raise RuntimeError("All files filtered out; no data for L2")

    points = np.vstack(all_points)
    intensities = np.hstack(all_intensities)
    times = np.hstack(all_times)

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Determine transects to process
    transect_grids: Optional[Dict[float, phase2.TimeResolvedGrid]] = None
    primary_grid = None

    if profile_config is not None:
        # Calculate transect geometry once
        dx_line = profile_config.x2 - profile_config.x1
        dy_line = profile_config.y2 - profile_config.y1
        length = np.sqrt(dx_line**2 + dy_line**2)
        line_vec = np.array([dx_line, dy_line]) / length

        # Alongshore direction (perpendicular)
        alongshore_angle = np.arctan2(dy_line, dx_line) + np.pi / 2
        dx_along = np.cos(alongshore_angle)
        dy_along = np.sin(alongshore_angle)

        # Determine which transects to process
        if multi_transect:
            offsets_to_process = list(profile_config.alongshore_spacings)
            transect_grids = {}
        else:
            offsets_to_process = [0.0]

        for offset in offsets_to_process:
            # Offset transect endpoints
            xs = profile_config.x1 + offset * dx_along
            ys = profile_config.y1 + offset * dy_along
            line_start = (xs, ys)

            # Project points onto this transect
            proj_dist, dist_to_line, mask = profiles._project_points_to_line(
                np.column_stack([X, Y]), line_start, line_vec, profile_config.tolerance
            )

            if mask.sum() == 0:
                logger.warning("No points near transect at offset %.1f", offset)
                continue

            # Filter to points near transect
            points_1d = proj_dist[mask]
            z_1d = Z[mask]
            intensities_1d = intensities[mask]
            times_1d = times[mask]

            # Create pseudo-3D points for temporal binning
            points_for_binning = np.column_stack([points_1d, np.zeros_like(points_1d), z_1d])

            # Temporal binning
            grid = phase2.bin_point_cloud_temporal(
                points_for_binning,
                intensities_1d,
                times_1d,
                x_bin_size=x_bin_size,
                time_bin_size=time_bin_size,
            )

            if multi_transect:
                transect_grids[offset] = grid
                if offset == 0.0 or primary_grid is None:
                    primary_grid = grid
            else:
                primary_grid = grid
    else:
        # Use X coordinate directly as cross-shore
        points_for_binning = points
        intensities_1d = intensities
        times_1d = times

        # Temporal binning
        primary_grid = phase2.bin_point_cloud_temporal(
            points_for_binning,
            intensities_1d,
            times_1d,
            x_bin_size=x_bin_size,
            time_bin_size=time_bin_size,
        )

    if primary_grid is None:
        raise RuntimeError("No valid transect data produced")

    # Apply 2D outlier detection
    outlier_mask = None
    Z_filtered = None

    if apply_outlier_detection:
        logger.info("Applying 2D outlier detection")
        Z_xt_raw = primary_grid.z_mean.T  # Shape (n_x, n_t)
        dt = float(np.median(np.diff(primary_grid.t_edges)))

        # Get outlier detection parameters
        params = outlier_params or {}
        outlier_result = utils.detect_outliers_conv2d(
            Z_xt_raw,
            dt=dt,
            ig_length=params.get('ig_length', 60.0),
            gradient_threshold_std=params.get('gradient_threshold_std', 2.5),
            laplacian_threshold_std=params.get('laplacian_threshold_std', 2.5),
        )
        outlier_mask = outlier_result.is_outlier
        Z_filtered = outlier_result.Z_filtered

        n_outliers = outlier_mask.sum()
        total = outlier_mask.size
        logger.info("Detected %d outliers (%.1f%%)", n_outliers, 100 * n_outliers / total)

    # Extract intensity contours
    intensity_contours = None

    if extract_intensity_contours:
        logger.info("Extracting intensity contours")
        I_xt = primary_grid.intensity_mean.T
        x1d = (primary_grid.x_edges[:-1] + primary_grid.x_edges[1:]) / 2

        thresholds = intensity_contour_thresholds or [20.0, 30.0, 40.0, 50.0]
        intensity_contours = utils.get_intensity_contours(
            I_xt, x1d, thresholds=thresholds
        )
        logger.info("Extracted %d valid intensity contours", len(intensity_contours))

    return phase3.TimeResolvedDataset(
        grid=primary_grid,
        base_time=base_time,
        profile_config=profile_config,
        transect_grids=transect_grids,
        outlier_mask=outlier_mask,
        Z_filtered=Z_filtered,
        intensity_contours=intensity_contours,
    )


def process_l1_batch(
    config_path: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    checkpoint_file: Optional[Path] = None,
    resume: bool = False,
    show_progress: bool = True,
    parallel: bool = False,
    max_workers: int = 4,
    **kwargs,
) -> BatchProgress:
    """
    Process L1 data in daily batches, saving each day to a separate file.

    Supports resumable processing via checkpoints.

    Parameters
    ----------
    config_path : Path
        Path to config JSON
    start, end : datetime
        Date range to process
    output_dir : Path
        Directory for output NetCDF files
    checkpoint_file : Path, optional
        Path to save/load checkpoint file. If None, uses output_dir/checkpoint.json
    resume : bool
        Resume from checkpoint if it exists (default False)
    show_progress : bool
        Show progress bar (default True)
    parallel : bool
        Use parallel processing (default False, experimental)
    max_workers : int
        Maximum parallel workers (default 4)
    **kwargs
        Additional arguments passed to process_l1()

    Returns
    -------
    BatchProgress
        Progress information including success/failure counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_file is None:
        checkpoint_file = output_dir / "checkpoint.json"

    # Generate list of dates to process
    dates_to_process = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        dates_to_process.append(current)
        current += timedelta(days=1)

    # Handle resume from checkpoint
    completed_dates = set()
    failed_dates = set()

    if resume and checkpoint_file.exists():
        try:
            checkpoint = Checkpoint.load(checkpoint_file)
            completed_dates = set(checkpoint.completed_dates)
            failed_dates = set(checkpoint.failed_dates)
            logger.info(
                "Resuming from checkpoint: %d completed, %d failed",
                len(completed_dates), len(failed_dates)
            )
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)

    # Filter out already completed dates
    remaining_dates = [
        d for d in dates_to_process
        if d.strftime("%Y-%m-%d") not in completed_dates
    ]

    progress = BatchProgress(total_items=len(dates_to_process))
    progress.completed = len(completed_dates)
    progress.skipped = len(failed_dates)

    if not remaining_dates:
        logger.info("All dates already processed")
        return progress

    logger.info("Processing %d days (%d already completed)", len(remaining_dates), len(completed_dates))

    # Process dates
    date_iter = remaining_dates
    if show_progress and TQDM_AVAILABLE:
        date_iter = tqdm(remaining_dates, desc="Processing days", unit="day")

    def process_single_date(date: datetime) -> Tuple[datetime, Optional[Path], Optional[str]]:
        """Process a single date, returning (date, output_path, error_msg)."""
        day_end = date + timedelta(days=1)
        date_str = date.strftime("%Y-%m-%d")

        try:
            result = process_l1(
                config_path,
                start=date,
                end=min(day_end, end),
                **kwargs,
            )

            output_path = output_dir / f"L1_{date.strftime('%Y%m%d')}.nc"
            save_dataset(result.dataset, output_path)
            return (date, output_path, None)

        except (FileNotFoundError, NoDataError) as e:
            return (date, None, f"No data: {e}")
        except Exception as e:
            return (date, None, f"Error: {e}")

    if parallel and len(remaining_dates) > 1:
        # Parallel processing (experimental)
        logger.info("Using parallel processing with %d workers", max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_date, d): d for d in remaining_dates}
            for future in as_completed(futures):
                date, output_path, error = future.result()
                date_str = date.strftime("%Y-%m-%d")

                if output_path:
                    progress.completed += 1
                    completed_dates.add(date_str)
                    logger.info("Completed %s", date_str)
                else:
                    progress.failed += 1
                    failed_dates.add(date_str)
                    progress.errors.append((date_str, error or "Unknown error"))
                    logger.warning("Failed %s: %s", date_str, error)
    else:
        # Sequential processing
        for date in date_iter:
            date_str = date.strftime("%Y-%m-%d")
            progress.current_item = date_str

            date, output_path, error = process_single_date(date)

            if output_path:
                progress.completed += 1
                completed_dates.add(date_str)
                logger.debug("Completed %s", date_str)
            else:
                progress.failed += 1
                failed_dates.add(date_str)
                progress.errors.append((date_str, error or "Unknown error"))
                logger.debug("Failed %s: %s", date_str, error)

            # Save checkpoint periodically
            if (progress.completed + progress.failed) % 5 == 0:
                checkpoint = Checkpoint(
                    config_path=str(config_path),
                    output_dir=str(output_dir),
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    completed_dates=list(completed_dates),
                    failed_dates=list(failed_dates),
                    kwargs={k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
                )
                checkpoint.save(checkpoint_file)

    # Final checkpoint save
    checkpoint = Checkpoint(
        config_path=str(config_path),
        output_dir=str(output_dir),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        completed_dates=list(completed_dates),
        failed_dates=list(failed_dates),
        kwargs={k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
    )
    checkpoint.save(checkpoint_file)

    logger.info(
        "Batch complete: %d succeeded, %d failed, %.1f%% success rate",
        progress.completed, progress.failed, progress.success_rate * 100
    )

    return progress


def process_l2_batch(
    config_path: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    file_duration: timedelta = timedelta(minutes=30),
    checkpoint_file: Optional[Path] = None,
    resume: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> BatchProgress:
    """
    Process L2 data in time-window batches, saving each window to a separate file.

    Parameters
    ----------
    config_path : Path
        Path to config JSON
    start, end : datetime
        Date range to process
    output_dir : Path
        Directory for output NetCDF files
    file_duration : timedelta
        Duration of each output file (default 30 minutes)
    checkpoint_file : Path, optional
        Path to save/load checkpoint file
    resume : bool
        Resume from checkpoint if it exists (default False)
    show_progress : bool
        Show progress bar (default True)
    **kwargs
        Additional arguments passed to process_l2()

    Returns
    -------
    BatchProgress
        Progress information including success/failure counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_file is None:
        checkpoint_file = output_dir / "l2_checkpoint.json"

    # Generate list of time windows to process
    windows_to_process = []
    current = start
    while current < end:
        window_end = min(current + file_duration, end)
        windows_to_process.append((current, window_end))
        current = window_end

    # Handle resume from checkpoint
    completed_windows = set()
    failed_windows = set()

    if resume and checkpoint_file.exists():
        try:
            checkpoint = Checkpoint.load(checkpoint_file)
            completed_windows = set(checkpoint.completed_dates)
            failed_windows = set(checkpoint.failed_dates)
            logger.info(
                "Resuming L2 from checkpoint: %d completed, %d failed",
                len(completed_windows), len(failed_windows)
            )
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)

    # Filter out already completed windows
    remaining_windows = [
        (s, e) for s, e in windows_to_process
        if s.isoformat() not in completed_windows
    ]

    progress = BatchProgress(total_items=len(windows_to_process))
    progress.completed = len(completed_windows)
    progress.skipped = len(failed_windows)

    if not remaining_windows:
        logger.info("All L2 windows already processed")
        return progress

    logger.info(
        "Processing %d L2 windows (%d already completed)",
        len(remaining_windows), len(completed_windows)
    )

    # Process windows
    window_iter = remaining_windows
    if show_progress and TQDM_AVAILABLE:
        window_iter = tqdm(remaining_windows, desc="Processing L2 windows", unit="window")

    for window_start, window_end in window_iter:
        window_key = window_start.isoformat()
        progress.current_item = window_key

        try:
            result = process_l2(
                config_path,
                start=window_start,
                end=window_end,
                **kwargs,
            )

            # Generate filename
            filename = f"L2_{window_start.strftime('%Y%m%d')}.nc"
            output_path = output_dir / filename
            result.to_netcdf(output_path)

            progress.completed += 1
            completed_windows.add(window_key)
            logger.debug("Completed L2 %s", window_key)

        except (FileNotFoundError, NoDataError) as e:
            progress.failed += 1
            failed_windows.add(window_key)
            progress.errors.append((window_key, f"No data: {e}"))
            logger.debug("No data for L2 %s: %s", window_key, e)

        except Exception as e:
            progress.failed += 1
            failed_windows.add(window_key)
            progress.errors.append((window_key, f"Error: {e}"))
            logger.warning("Error processing L2 %s: %s", window_key, e)

        # Save checkpoint periodically
        if (progress.completed + progress.failed) % 10 == 0:
            checkpoint = Checkpoint(
                config_path=str(config_path),
                output_dir=str(output_dir),
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                completed_dates=list(completed_windows),
                failed_dates=list(failed_windows),
                kwargs={k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
            )
            checkpoint.save(checkpoint_file)

    # Final checkpoint save
    checkpoint = Checkpoint(
        config_path=str(config_path),
        output_dir=str(output_dir),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        completed_dates=list(completed_windows),
        failed_dates=list(failed_windows),
        kwargs={k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
    )
    checkpoint.save(checkpoint_file)

    logger.info(
        "L2 batch complete: %d succeeded, %d failed, %.1f%% success rate",
        progress.completed, progress.failed, progress.success_rate * 100
    )

    return progress


# =============================================================================
# Memory Management Utilities
# =============================================================================

def clear_memory() -> None:
    """
    Attempt to free memory by running garbage collection.
    Useful between large batch processing steps.
    """
    import gc
    gc.collect()


def estimate_memory_usage(points: np.ndarray) -> float:
    """
    Estimate memory usage in MB for a point cloud array.

    Parameters
    ----------
    points : ndarray
        Point cloud array

    Returns
    -------
    float
        Estimated memory in MB
    """
    return points.nbytes / (1024 * 1024)


def chunked_file_iterator(
    laz_files: List[Tuple[Path, datetime]],
    max_memory_mb: float = 1000.0,
    loader: Optional[LoaderFn] = None,
) -> Iterator[Tuple[List[Tuple[Path, datetime]], bool]]:
    """
    Iterate over LAZ files in memory-efficient chunks.

    Yields groups of files that should fit within memory limit.
    This is an estimate based on file sizes.

    Parameters
    ----------
    laz_files : list
        List of (path, timestamp) tuples
    max_memory_mb : float
        Maximum memory target per chunk in MB
    loader : callable, optional
        Custom loader to get file info

    Yields
    ------
    chunk : list
        Group of (path, timestamp) tuples
    is_last : bool
        True if this is the last chunk
    """
    chunk = []
    estimated_mb = 0.0

    # Rough estimate: 1MB of LAZ file ≈ 50MB in memory (uncompressed)
    COMPRESSION_RATIO = 50.0

    for i, (path, ts) in enumerate(laz_files):
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            estimated_memory = file_size_mb * COMPRESSION_RATIO
        except OSError:
            estimated_memory = 50.0  # Default estimate

        if chunk and estimated_mb + estimated_memory > max_memory_mb:
            # Yield current chunk and start new one
            yield chunk, False
            chunk = []
            estimated_mb = 0.0
            clear_memory()

        chunk.append((path, ts))
        estimated_mb += estimated_memory

    if chunk:
        yield chunk, True


def save_dataset(ds: phase3.xr.Dataset, output_path: Path) -> None:
    """
    Save dataset to NetCDF using xarray.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)


def export_profiles_json(
    result: L1Result,
    output_path: Path,
    transect_indices: Optional[List[int]] = None,
) -> None:
    """
    Export 1D profiles to JSON for web visualization.
    Matches MATLAB's lidar_plot_data.json format.

    Parameters
    ----------
    result : L1Result
        L1 processing result with profiles
    output_path : Path
        Output JSON file path
    transect_indices : list of int, optional
        Which transects to export (default: all)
    """
    import json

    if result.profiles is None:
        raise ValueError("No profiles in result")

    profiles_data = []
    x1d = result.profiles.x1d.tolist()

    if transect_indices is None:
        transect_indices = list(range(result.profiles.Z3D.shape[0]))

    for i in transect_indices:
        z_profile = result.profiles.Z3D[i, :]
        offset = result.profile_config.alongshore_spacings[i] if result.profile_config else i

        profiles_data.append({
            "alongshore_offset": float(offset),
            "x": x1d,
            "z": [float(v) if not np.isnan(v) else None for v in z_profile],
        })

    output = {
        "profiles": profiles_data,
        "metadata": {
            "resolution": result.profile_config.resolution if result.profile_config else None,
            "n_transects": len(transect_indices),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for lidar processing."""
    parser = argparse.ArgumentParser(
        description="Lidar point cloud processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single L1 dataset
  python phase4.py l1 config.json -o output.nc --start 2024-06-15 --end 2024-06-16

  # Process L2 wave data
  python phase4.py l2 config.json -o l2_output.nc --start 2024-06-15T10:00 --end 2024-06-15T10:30

  # Batch process L1 with resume capability
  python phase4.py batch config.json -o ./output/ --start 2024-06-01 --end 2024-06-30 --resume

  # Batch process L2
  python phase4.py batch-l2 config.json -o ./l2_output/ --start 2024-06-15 --end 2024-06-16
        """,
    )

    # Common arguments shared by all subcommands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    common_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output")
    common_parser.add_argument("--log-file", type=Path, help="Write logs to file")
    common_parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    subparsers = parser.add_subparsers(dest="command", help="Processing command")

    # L1 command
    l1_parser = subparsers.add_parser("l1", help="Process L1 (hourly beach surface)", parents=[common_parser])
    l1_parser.add_argument("config", type=Path, help="Path to config JSON")
    l1_parser.add_argument("-o", "--output", type=Path, required=True, help="Output NetCDF path")
    l1_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD or ISO format)")
    l1_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD or ISO format)")
    l1_parser.add_argument("--bin-size", type=float, default=0.1, help="Spatial bin size in meters (default: 0.1)")
    l1_parser.add_argument("--no-filter", action="store_true", help="Skip residual filtering")
    l1_parser.add_argument("--data-folder", type=Path, help="Override data folder from config")
    l1_parser.add_argument("--skip-corrupt", action="store_true", default=True, help="Skip corrupt files (default: True)")
    l1_parser.add_argument("-n", type=int, help="Limit to first N files (for testing)")

    # L2 command
    l2_parser = subparsers.add_parser("l2", help="Process L2 (wave-resolving)", parents=[common_parser])
    l2_parser.add_argument("config", type=Path, help="Path to config JSON")
    l2_parser.add_argument("-o", "--output", type=Path, required=True, help="Output NetCDF path")
    l2_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD or ISO format)")
    l2_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD or ISO format)")
    l2_parser.add_argument("--time-bin", type=float, default=0.5, help="Time bin size in seconds (default: 0.5)")
    l2_parser.add_argument("--x-bin", type=float, default=0.1, help="Spatial bin size in meters (default: 0.1)")
    l2_parser.add_argument("--data-folder", type=Path, help="Override data folder from config")
    l2_parser.add_argument("--no-outlier-detection", action="store_true", help="Skip 2D outlier detection")
    l2_parser.add_argument("--multi-transect", action="store_true", help="Extract multiple transects")
    l2_parser.add_argument("-n", type=int, help="Limit to first N files (for testing)")

    # Batch L1 command
    batch_parser = subparsers.add_parser("batch", help="Batch process L1 by day", parents=[common_parser])
    batch_parser.add_argument("config", type=Path, help="Path to config JSON")
    batch_parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    batch_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    batch_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    batch_parser.add_argument("--data-folder", type=Path, help="Override data folder from config")
    batch_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    batch_parser.add_argument("--parallel", action="store_true", help="Use parallel processing (experimental)")
    batch_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")

    # Batch L2 command
    batch_l2_parser = subparsers.add_parser("batch-l2", help="Batch process L2 by time window", parents=[common_parser])
    batch_l2_parser.add_argument("config", type=Path, help="Path to config JSON")
    batch_l2_parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    batch_l2_parser.add_argument("--start", type=str, required=True, help="Start datetime (YYYY-MM-DD or ISO)")
    batch_l2_parser.add_argument("--end", type=str, required=True, help="End datetime (YYYY-MM-DD or ISO)")
    batch_l2_parser.add_argument("--window", type=int, default=30, help="Window duration in minutes (default: 30)")
    batch_l2_parser.add_argument("--data-folder", type=Path, help="Override data folder from config")
    batch_l2_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")

    args = parser.parse_args()

    # Configure logging
    configure_logging(
        verbose=args.verbose,
        log_file=args.log_file,
        quiet=args.quiet,
    )

    show_progress = not args.no_progress

    try:
        if args.command == "l1":
            start = datetime.fromisoformat(args.start) if args.start else None
            end = datetime.fromisoformat(args.end) if args.end else None

            result = process_l1(
                args.config,
                start=start,
                end=end,
                bin_size=args.bin_size,
                apply_residual_filter=not args.no_filter,
                data_folder_override=args.data_folder,
                show_progress=show_progress,
                skip_corrupt=args.skip_corrupt,
                max_files=args.n,
            )
            save_dataset(result.dataset, args.output)
            print(f"Saved L1 dataset to {args.output}")

        elif args.command == "l2":
            start = datetime.fromisoformat(args.start) if args.start else None
            end = datetime.fromisoformat(args.end) if args.end else None

            result = process_l2(
                args.config,
                start=start,
                end=end,
                time_bin_size=args.time_bin,
                x_bin_size=args.x_bin,
                data_folder_override=args.data_folder,
                apply_outlier_detection=not args.no_outlier_detection,
                multi_transect=args.multi_transect,
                show_progress=show_progress,
                max_files=args.n,
            )
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_netcdf(output_path)
            print(f"Saved L2 dataset to {args.output}")

        elif args.command == "batch":
            start = datetime.fromisoformat(args.start)
            end = datetime.fromisoformat(args.end)

            progress = process_l1_batch(
                args.config,
                start=start,
                end=end,
                output_dir=args.output_dir,
                data_folder_override=args.data_folder,
                resume=args.resume,
                show_progress=show_progress,
                parallel=args.parallel,
                max_workers=args.workers,
            )
            print(f"Batch complete: {progress.completed} succeeded, {progress.failed} failed")
            if progress.errors:
                print(f"Errors:")
                for date, error in progress.errors[:5]:
                    print(f"  {date}: {error}")
                if len(progress.errors) > 5:
                    print(f"  ... and {len(progress.errors) - 5} more")

        elif args.command == "batch-l2":
            start = datetime.fromisoformat(args.start)
            end = datetime.fromisoformat(args.end)

            progress = process_l2_batch(
                args.config,
                start=start,
                end=end,
                output_dir=args.output_dir,
                file_duration=timedelta(minutes=args.window),
                data_folder_override=args.data_folder,
                resume=args.resume,
                show_progress=show_progress,
            )
            print(f"L2 batch complete: {progress.completed} succeeded, {progress.failed} failed")

        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except LidarProcessingError as e:
        logger.error("Processing error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
