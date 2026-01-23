"""
Phase 1: ingest and preprocessing utilities for LiDAR point clouds.
The functions here are pure/side-effect free to make later stages easy to test.
"""
from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Type aliases
ArrayLike = np.ndarray


@dataclass(frozen=True)
class Config:
    data_folder: Path
    process_folder: Path
    plot_folder: Path
    transform_matrix: ArrayLike
    lidar_boundary: ArrayLike  # shape (N, 2)


def adapt_path_for_os(path_str: str) -> Path:
    """
    Translate paths based on operating system.

    Mac: /Volumes/<group>/... (unchanged)
    Linux: /Volumes/<group>/... -> /project/<group>/...
    """
    system = platform.system()

    if system == "Linux":
        # On Linux, translate /Volumes/ to /project/
        if path_str.startswith("/Volumes/"):
            path_str = "/project/" + path_str[len("/Volumes/"):]
    # On Darwin (macOS) or other systems, keep paths as-is

    return Path(path_str)


# Keep old name for backwards compatibility
_translate_path_for_os = adapt_path_for_os


def load_config(path: Path) -> Config:
    """
    Load configuration from JSON and coerce to strong types.
    Required keys: dataFolder, processFolder, plotFolder, transformMatrix, LidarBoundary.

    Paths are automatically translated based on operating system:
    - macOS: /Volumes/<group>/... (as specified in config)
    - Linux: /Volumes/<group>/... -> /project/<group>/...
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    required = ["dataFolder", "processFolder", "plotFolder", "transformMatrix", "LidarBoundary"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Config missing keys: {', '.join(missing)}")

    tmatrix = np.asarray(raw["transformMatrix"], dtype=float)
    if tmatrix.shape != (4, 4):
        raise ValueError(f"transformMatrix must be 4x4, got {tmatrix.shape}")

    boundary = np.asarray(raw["LidarBoundary"], dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or boundary.shape[0] < 3:
        raise ValueError("LidarBoundary must be Nx2 with at least 3 vertices")

    return Config(
        data_folder=_translate_path_for_os(raw["dataFolder"]),
        process_folder=_translate_path_for_os(raw["processFolder"]),
        plot_folder=_translate_path_for_os(raw["plotFolder"]),
        transform_matrix=tmatrix,
        lidar_boundary=boundary,
    )


def _parse_timestamp_from_filename(stem: str) -> Optional[datetime]:
    """
    Parse timestamp from LAZ filename stem. Supports multiple formats:
    - POSIX timestamp: do-lidar_1735689600 -> timestamp
    - Date format: TOWR-test_20260120_200023_UTC -> 2026-01-20 20:00:23 UTC
    """
    import re

    # Try POSIX timestamp (e.g., do-lidar_1735689600)
    posix_match = re.search(r'(\d{10,})$', stem)
    if posix_match:
        try:
            return datetime.fromtimestamp(int(posix_match.group(1)), tz=timezone.utc)
        except (ValueError, OSError):
            pass

    # Try date format: *_YYYYMMDD_HHMMSS_UTC (e.g., TOWR-test_20260120_200023_UTC)
    date_match = re.search(r'_(\d{8})_(\d{6})_UTC$', stem)
    if date_match:
        try:
            date_str = date_match.group(1)
            time_str = date_match.group(2)
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


def discover_laz_files(
    folder: Path, start: Optional[datetime] = None, end: Optional[datetime] = None
) -> List[Tuple[Path, datetime]]:
    """
    Find all .laz files, parse timestamps from filenames, and sort them.
    Supports multiple filename formats (POSIX timestamps and YYYYMMDD_HHMMSS_UTC).
    Filters by [start, end] if provided (naive datetimes interpreted as UTC).
    """
    paths = list(folder.glob("*.laz"))
    results: List[Tuple[Path, datetime]] = []
    for p in paths:
        ts = _parse_timestamp_from_filename(p.stem)
        if ts is None:
            continue
        results.append((p, ts))

    if start:
        s = _ensure_tz(start)
        results = [(p, t) for p, t in results if t >= s]
    if end:
        e = _ensure_tz(end)
        results = [(p, t) for p, t in results if t <= e]

    return sorted(results, key=lambda x: x[1])


def transform_points(points: ArrayLike, tmatrix: ArrayLike) -> ArrayLike:
    """
    Apply homogeneous transform to 3D points.
    points: (N,3), tmatrix: (4,4)
    """
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if tmatrix.shape != (4, 4):
        raise ValueError("tmatrix must be 4x4")

    ones = np.ones((points.shape[0], 1), dtype=float)
    hom = np.hstack([points, ones])
    transformed = hom @ tmatrix.T
    return transformed[:, :3]


def filter_by_polygon(points: ArrayLike, polygon: ArrayLike, eps: float = 1e-9) -> ArrayLike:
    """
    Return boolean mask of points inside a polygon, including edge points.
    points: (N,2); polygon: (M,2) closed or open.

    Uses matplotlib.path.Path for optimized C-level point-in-polygon testing.
    Falls back to pure Python implementation if matplotlib is not available.
    """
    try:
        from matplotlib.path import Path
        # Create path (matplotlib handles closing automatically)
        path = Path(polygon)
        # contains_points is highly optimized C code
        # radius=eps includes points very close to edges
        inside = path.contains_points(points, radius=eps)
        return inside
    except ImportError:
        # Fallback to pure Python implementation
        return _filter_by_polygon_python(points, polygon, eps)


def _filter_by_polygon_python(points: ArrayLike, polygon: ArrayLike, eps: float = 1e-9) -> ArrayLike:
    """
    Pure Python fallback for point-in-polygon testing using ray casting.
    """
    x = points[:, 0]
    y = points[:, 1]
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        intersect = ((yi > y) != (yj > y)) & (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi + eps
        )
        inside ^= intersect
        j = i
    # Include points lying on polygon edges
    on_edge = np.zeros(len(points), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        dx = xj - xi
        dy = yj - yi
        # Project point onto segment and check distance to line
        t = ((x - xi) * dx + (y - yi) * dy) / (dx * dx + dy * dy + 1e-20)
        on_segment = (t >= -eps) & (t <= 1 + eps)
        proj_x = xi + t * dx
        proj_y = yi + t * dy
        dist2 = (x - proj_x) ** 2 + (y - proj_y) ** 2
        on_edge |= on_segment & (dist2 <= eps)
        j = i
    return inside | on_edge


def filter_points(
    points: ArrayLike,
    intensities: ArrayLike,
    times: ArrayLike,
    polygon: ArrayLike,
    intensity_threshold: float = 100.0,
    max_seconds: Optional[float] = 300.0,
) -> ArrayLike:
    """
    Filter by polygon, intensity, and optional time window.
    times: seconds from start; max_seconds=None disables time limit.
    Returns boolean mask of kept points.
    """
    if not (len(points) == len(intensities) == len(times)):
        raise ValueError("points, intensities, and times must have the same length")

    keep = filter_by_polygon(points[:, :2], polygon)
    keep &= intensities < intensity_threshold
    if max_seconds is not None:
        keep &= times <= max_seconds
    return keep


def prepare_batch(
    points: ArrayLike,
    intensities: ArrayLike,
    gps_times: ArrayLike,
    config: Config,
    intensity_threshold: float = 100.0,
    max_seconds: Optional[float] = 300.0,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Convenience helper: transform points, apply filters, and return filtered arrays.
    """
    transformed = transform_points(points, config.transform_matrix)
    # GPS times in seconds relative to first sample
    rel_times = gps_times - gps_times.min()
    mask = filter_points(
        transformed,
        intensities,
        rel_times,
        config.lidar_boundary,
        intensity_threshold=intensity_threshold,
        max_seconds=max_seconds,
    )
    return transformed[mask], intensities[mask], rel_times[mask]


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
