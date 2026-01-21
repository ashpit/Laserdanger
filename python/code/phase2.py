"""
Phase 2: core algorithms (binning, SNR, residual kernel filtering).
These functions stay pure and accept numpy arrays; I/O happens upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import binned_statistic_2d

ArrayLike = np.ndarray

# Default filtering parameters matching MATLAB L1_pipeline.m
DEFAULT_SNR_THRESHOLD = 100.0
DEFAULT_MIN_COUNT = 10
DEFAULT_PERCENTILE = 50


def bin_edges(values: ArrayLike, bin_size: float) -> ArrayLike:
    """
    Public helper to build inclusive bin edges. Ensures at least two bins.
    """
    return _bin_edges(values, bin_size)


@dataclass(frozen=True)
class BinnedGrid:
    x_edges: ArrayLike
    y_edges: ArrayLike
    z_mean: ArrayLike
    z_min: ArrayLike
    z_max: ArrayLike
    z_std: ArrayLike
    z_mode: ArrayLike
    count: ArrayLike
    snr: ArrayLike
    valid_mask: ArrayLike  # True where SNR >= threshold AND count > min_count


@dataclass(frozen=True)
class TimeResolvedGrid:
    x_edges: ArrayLike
    t_edges: ArrayLike
    z_mean: ArrayLike
    z_min: ArrayLike
    z_max: ArrayLike
    z_std: ArrayLike
    count: ArrayLike
    intensity_mean: ArrayLike


def bin_point_cloud(
    points: ArrayLike,
    bin_size: float = 0.1,
    mode_bin: float = 0.05,
    x_edges: ArrayLike | None = None,
    y_edges: ArrayLike | None = None,
    percentile: Optional[float] = DEFAULT_PERCENTILE,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    min_count: int = DEFAULT_MIN_COUNT,
) -> BinnedGrid:
    """
    Rasterize points (N,3) onto a regular grid and compute statistics.

    Matches MATLAB accumpts.m behavior:
    1. Bin points by (x, y) position
    2. Apply percentile filtering within each bin (keep z <= Nth percentile)
    3. Compute statistics (mean, min, max, std, mode, count)
    4. Compute SNR and mark bins as valid/invalid

    Parameters
    ----------
    points : array (N, 3)
        Point cloud with columns [x, y, z]
    bin_size : float
        Grid cell size in same units as points (default 0.1m)
    mode_bin : float
        Quantization resolution for mode calculation (default 0.05m)
    x_edges, y_edges : array, optional
        Pre-computed bin edges; if None, computed from data
    percentile : float or None
        Keep only z values <= this percentile within each bin (default 50).
        Set to None to disable percentile filtering.
    snr_threshold : float
        Minimum SNR for a bin to be considered valid (default 100)
    min_count : int
        Minimum point count for a bin to be valid (default 10)

    Returns
    -------
    BinnedGrid
        Dataclass with grid statistics and valid_mask
    """
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    x, y, z = points.T
    x_edges = _bin_edges(x, bin_size) if x_edges is None else x_edges
    y_edges = _bin_edges(y, bin_size) if y_edges is None else y_edges

    if percentile is not None:
        # Use custom binning with percentile filtering (matches MATLAB accumpts.m)
        z_mean, z_min, z_max, z_std, z_mode, count = _bin_with_percentile_filter(
            x, y, z, x_edges, y_edges, percentile, mode_bin
        )
    else:
        # Standard binning without percentile filter
        z_mean, _, _, _ = binned_statistic_2d(x, y, z, statistic="mean", bins=[x_edges, y_edges])
        z_min, _, _, _ = binned_statistic_2d(x, y, z, statistic="min", bins=[x_edges, y_edges])
        z_max, _, _, _ = binned_statistic_2d(x, y, z, statistic="max", bins=[x_edges, y_edges])
        z_std, _, _, _ = binned_statistic_2d(x, y, z, statistic="std", bins=[x_edges, y_edges])
        count, _, _, _ = binned_statistic_2d(x, y, z, statistic="count", bins=[x_edges, y_edges])

        # Mode via quantization to reduce unique values
        z_quant = np.round(z / mode_bin) * mode_bin
        z_mode, _, _, _ = binned_statistic_2d(
            x, y, z_quant, statistic=_mode_statistic, bins=[x_edges, y_edges]
        )

    snr = compute_snr(z_mean, z_std, count)
    valid_mask = (snr >= snr_threshold) & (count > min_count)

    return BinnedGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        z_mean=z_mean,
        z_min=z_min,
        z_max=z_max,
        z_std=z_std,
        z_mode=z_mode,
        count=count,
        snr=snr,
        valid_mask=valid_mask,
    )


def bin_point_cloud_temporal(
    points: ArrayLike,
    intensities: ArrayLike,
    times: ArrayLike,
    x_bin_size: float = 0.1,
    time_bin_size: float = 0.5,
    x_edges: ArrayLike | None = None,
    t_edges: ArrayLike | None = None,
) -> TimeResolvedGrid:
    """
    Bin points into cross-shore/time cells (Z(x,t)) and compute per-bin statistics.
    Returns arrays shaped (t_bins, x_bins) with time as the first axis.
    Default time_bin_size=0.5 seconds targets ~2 Hz resolution.
    """
    points = np.asarray(points)
    intensities = np.asarray(intensities)
    times = np.asarray(times)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] != intensities.shape[0] or points.shape[0] != times.shape[0]:
        raise ValueError("points, intensities, and times must have the same length")
    if x_bin_size <= 0 or time_bin_size <= 0:
        raise ValueError("x_bin_size and time_bin_size must be positive")

    x_vals = points[:, 0]
    z_vals = points[:, 2]

    x_edges = _bin_edges(x_vals, x_bin_size) if x_edges is None else x_edges
    t_edges = _bin_edges(times, time_bin_size) if t_edges is None else t_edges

    z_mean, _, _, _ = binned_statistic_2d(times, x_vals, z_vals, statistic="mean", bins=[t_edges, x_edges])
    z_min, _, _, _ = binned_statistic_2d(times, x_vals, z_vals, statistic="min", bins=[t_edges, x_edges])
    z_max, _, _, _ = binned_statistic_2d(times, x_vals, z_vals, statistic="max", bins=[t_edges, x_edges])
    z_std, _, _, _ = binned_statistic_2d(times, x_vals, z_vals, statistic="std", bins=[t_edges, x_edges])
    count, _, _, _ = binned_statistic_2d(times, x_vals, z_vals, statistic="count", bins=[t_edges, x_edges])
    intensity_mean, _, _, _ = binned_statistic_2d(
        times, x_vals, intensities, statistic="mean", bins=[t_edges, x_edges]
    )

    # Normalize empty bins to NaN for consistency across statistics
    empty_mask = count == 0
    for arr in (z_mean, z_min, z_max, z_std, intensity_mean):
        arr[empty_mask] = np.nan

    return TimeResolvedGrid(
        x_edges=x_edges,
        t_edges=t_edges,
        z_mean=z_mean,
        z_min=z_min,
        z_max=z_max,
        z_std=z_std,
        count=count,
        intensity_mean=intensity_mean,
    )


def compute_snr(z_mean: ArrayLike, z_std: ArrayLike, count: ArrayLike) -> ArrayLike:
    """
    Signal-to-noise ratio per bin: mean / (std / sqrt(count)).
    Returns NaN where count == 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = z_std / np.sqrt(count)
        snr = z_mean / standard_error
        snr[count == 0] = np.nan
    return snr


def apply_snr_filter(
    grid: BinnedGrid,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    min_count: int = DEFAULT_MIN_COUNT,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Apply SNR filtering to a BinnedGrid and return valid data as 1D arrays.
    Matches MATLAB accumpts.m output format: vectors of valid points only.

    Parameters
    ----------
    grid : BinnedGrid
        Binned point cloud grid
    snr_threshold : float
        Minimum SNR for validity (default 100)
    min_count : int
        Minimum point count for validity (default 10)

    Returns
    -------
    Xutm, Yutm, Zmean, Zmax, Zmin, Zstd : arrays
        1D arrays of valid bin centers and statistics
    """
    # Compute valid mask
    valid = (grid.snr >= snr_threshold) & (grid.count > min_count)

    # Get bin centers
    x_centers = (grid.x_edges[:-1] + grid.x_edges[1:]) / 2
    y_centers = (grid.y_edges[:-1] + grid.y_edges[1:]) / 2

    # Get 2D indices of valid bins
    valid_i, valid_j = np.where(valid)

    # Extract coordinates and values
    Xutm = x_centers[valid_i]
    Yutm = y_centers[valid_j]
    Zmean = grid.z_mean[valid]
    Zmax = grid.z_max[valid]
    Zmin = grid.z_min[valid]
    Zstd = grid.z_std[valid]

    return Xutm, Yutm, Zmean, Zmax, Zmin, Zstd


def grid_to_points(grid: BinnedGrid, use_valid_only: bool = True) -> ArrayLike:
    """
    Convert a BinnedGrid to point cloud format (N, 3) using z_mean.

    Parameters
    ----------
    grid : BinnedGrid
    use_valid_only : bool
        If True, only return points where valid_mask is True

    Returns
    -------
    array (N, 3)
        Points with columns [x, y, z_mean]
    """
    x_centers = (grid.x_edges[:-1] + grid.x_edges[1:]) / 2
    y_centers = (grid.y_edges[:-1] + grid.y_edges[1:]) / 2
    Xg, Yg = np.meshgrid(x_centers, y_centers, indexing='ij')

    if use_valid_only:
        mask = grid.valid_mask
    else:
        mask = ~np.isnan(grid.z_mean)

    return np.column_stack([Xg[mask], Yg[mask], grid.z_mean[mask]])


def residual_kernel_filter(
    points: ArrayLike,
    window_radius: float = 1.0,
    max_residual: float = 0.5,
    min_neighbors: int = 6,
) -> ArrayLike:
    """
    Simple residual-based ground filter using KD-tree neighbor search.
    For each point, fit a plane to neighbors within window_radius and keep the point
    if its residual is <= max_residual. Returns boolean mask of ground points.

    This is a simpler alternative to residual_kernel_filter_delaunay.
    """
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    tree = cKDTree(points[:, :2])
    keep = np.zeros(len(points), dtype=bool)
    neighbor_lists = tree.query_ball_point(points[:, :2], r=window_radius)
    for idx, (x, y, z) in enumerate(points):
        neighbor_idx = neighbor_lists[idx]
        if len(neighbor_idx) < min_neighbors:
            continue
        neighbor_pts = points[neighbor_idx]
        coeffs = _fit_plane(neighbor_pts)
        z_fit = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        residual = abs(z - z_fit)
        if residual <= max_residual:
            keep[idx] = True
    return keep


def residual_kernel_filter_delaunay(
    points: ArrayLike,
    cell_size: float = 2.0,
    max_residual: float = 0.2,
    min_points_per_cell: int = 10,
    overlap: float = 0.1,
    keep_unprocessed: bool = True,
) -> ArrayLike:
    """
    Delaunay triangulation-based residual filter matching MATLAB ResidualKernelFilter.m.

    Creates a grid, triangulates it, and for each triangle:
    1. Finds points inside the triangle
    2. Fits a plane using SVD
    3. Keeps points with residual < threshold

    Parameters
    ----------
    points : array (N, 3)
        Point cloud with columns [x, y, z]
    cell_size : float
        Base grid cell size (l in MATLAB). Height = 3 * cell_size.
    max_residual : float
        Maximum allowed residual from fitted plane (default 0.2m)
    min_points_per_cell : int
        Minimum points in a triangle to attempt plane fit (default 10)
    overlap : float
        Grid overlap fraction (default 0.1 = 10%)
    keep_unprocessed : bool
        If True, keep points that couldn't be processed (in sparse regions).
        If False, exclude them. Default True for conservative filtering.

    Returns
    -------
    array of int
        Indices of points that pass the filter (ground points)
    """
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Grid parameters (matching MATLAB)
    l = cell_size
    h = 3 * l
    step_x = l * (1 - overlap)
    step_y = h * (1 - overlap)

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Generate grid vertices with extra padding to cover edges
    xv = np.arange(x_min - l, x_max + 2 * l + step_x, step_x)
    yv = np.arange(y_min - h, y_max + 2 * h + step_y, step_y)
    Xgrid, Ygrid = np.meshgrid(xv, yv)
    grid_points = np.column_stack([Xgrid.ravel(), Ygrid.ravel()])

    # Delaunay triangulation of the grid
    tri = Delaunay(grid_points)

    # Use find_simplex to efficiently locate which triangle each point belongs to
    # This is O(n log m) instead of O(n * m) where n=points, m=triangles
    simplex_indices = tri.find_simplex(points[:, :2])

    # Track fitted Z values
    z_interp = np.full(len(points), np.nan)

    # Group points by their simplex for efficient processing
    unique_simplices = np.unique(simplex_indices)

    for simplex_idx in unique_simplices:
        if simplex_idx == -1:
            # Points outside the triangulation - skip (will be kept if keep_unprocessed=True)
            continue

        # Find all points in this triangle
        in_tri = simplex_indices == simplex_idx

        if in_tri.sum() < min_points_per_cell:
            # Not enough points for reliable plane fit - skip
            continue

        section = points[in_tri]

        # Fit plane using SVD (matches MATLAB fitPlane.m)
        z_fit = _fit_plane_svd(section)
        z_interp[in_tri] = z_fit

    # Compute residuals
    residuals = np.abs(Z - z_interp)

    # Determine which points to keep
    if keep_unprocessed:
        # Keep points with small residuals OR points that couldn't be processed
        has_valid_fit = ~np.isnan(z_interp)
        passes_residual = residuals < max_residual
        keep = (has_valid_fit & passes_residual) | (~has_valid_fit)
    else:
        # Only keep points with small residuals
        keep = residuals < max_residual

    ground_indices = np.where(keep)[0]

    return ground_indices


def residual_kernel_filter_two_stage(
    points: ArrayLike,
    passes: Optional[List[Tuple[float, float]]] = None,
    min_points_per_cell: int = 10,
    overlap: float = 0.1,
) -> ArrayLike:
    """
    Two-stage residual kernel filtering matching MATLAB L1_pipeline.m.

    Default passes:
    1. cell_size=10m, threshold=0.2m (coarse pass)
    2. cell_size=3m, threshold=0.1m (fine pass)

    Parameters
    ----------
    points : array (N, 3)
        Point cloud with columns [x, y, z]
    passes : list of (cell_size, threshold) tuples, optional
        Filter passes to apply sequentially. Default: [(10, 0.2), (3, 0.1)]
    min_points_per_cell : int
        Minimum points per triangle cell (default 10)
    overlap : float
        Grid overlap fraction (default 0.1)

    Returns
    -------
    array (M, 3)
        Filtered point cloud (ground points only)
    """
    if passes is None:
        passes = [(10.0, 0.2), (3.0, 0.1)]

    filtered_points = points.copy()

    for cell_size, threshold in passes:
        if len(filtered_points) < min_points_per_cell:
            break

        ground_idx = residual_kernel_filter_delaunay(
            filtered_points,
            cell_size=cell_size,
            max_residual=threshold,
            min_points_per_cell=min_points_per_cell,
            overlap=overlap,
        )
        filtered_points = filtered_points[ground_idx]

    return filtered_points


def _points_in_triangle(points_2d: ArrayLike, triangle: ArrayLike) -> ArrayLike:
    """
    Check which 2D points lie inside a triangle using barycentric coordinates.

    Parameters
    ----------
    points_2d : array (N, 2)
    triangle : array (3, 2)

    Returns
    -------
    array of bool (N,)
    """
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = points_2d - triangle[0]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)

    dot02 = v2 @ v0
    dot12 = v2 @ v1

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.zeros(len(points_2d), dtype=bool)

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) & (v >= 0) & (u + v <= 1)


def _fit_plane_svd(points: ArrayLike, robust: bool = True, trim_percentile: float = 75) -> ArrayLike:
    """
    Fit a plane to 3D points using SVD (matches MATLAB fitPlane.m).

    Parameters
    ----------
    points : array (N, 3)
    robust : bool
        If True, use iterative trimmed fitting to reduce outlier influence
    trim_percentile : float
        Percentile threshold for trimming (default 75)

    Returns the fitted Z values for each input point.
    """
    if len(points) < 3:
        return np.full(len(points), np.nan)

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    def fit_once(pts):
        mean_point = pts.mean(axis=0)
        centered = pts - mean_point
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]
        a, b, c = normal
        d = -np.dot(normal, mean_point)
        return a, b, c, d

    # Initial fit
    a, b, c, d = fit_once(points)

    if abs(c) < 1e-12:
        return np.full(len(points), np.nan)

    if robust and len(points) >= 6:
        # Compute residuals and trim outliers
        z_fit_initial = (-d - a * X - b * Y) / c
        residuals = np.abs(Z - z_fit_initial)

        # Keep points within trim_percentile of residuals
        threshold = np.percentile(residuals, trim_percentile)
        keep = residuals <= threshold

        if keep.sum() >= 3:
            # Re-fit with trimmed points
            a, b, c, d = fit_once(points[keep])

            if abs(c) < 1e-12:
                return np.full(len(points), np.nan)

    # Compute final z values for ALL input points
    z_fit = (-d - a * X - b * Y) / c

    return z_fit


def _bin_edges(values: ArrayLike, bin_size: float) -> ArrayLike:
    vmin = np.min(values)
    vmax = np.max(values)
    min_edge = bin_size * np.floor(vmin / bin_size)
    max_edge = bin_size * np.ceil(vmax / bin_size)
    if max_edge <= min_edge:
        max_edge = min_edge + bin_size
    # include the rightmost edge
    edges = np.arange(min_edge, max_edge + bin_size, bin_size)
    if len(edges) < 3:  # ensure at least two bins
        edges = np.append(edges, edges[-1] + bin_size)
    return edges


def _bin_with_percentile_filter(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    percentile: float,
    mode_bin: float,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Bin points and apply percentile filtering within each bin.
    Matches MATLAB accumpts.m: keeps only z values <= Nth percentile before computing stats.

    Returns (z_mean, z_min, z_max, z_std, z_mode, count) arrays.
    """
    # Digitize to find bin indices
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    # Clip to valid range
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)

    # Initialize output arrays
    z_mean = np.full((nx, ny), np.nan)
    z_min = np.full((nx, ny), np.nan)
    z_max = np.full((nx, ny), np.nan)
    z_std = np.full((nx, ny), np.nan)
    z_mode = np.full((nx, ny), np.nan)
    count = np.zeros((nx, ny), dtype=float)

    # Create flat bin index for grouping
    flat_idx = x_idx * ny + y_idx

    # Group z values by bin
    order = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    z_sorted = z[order]

    # Find unique bins and their boundaries
    unique_bins, split_indices = np.unique(flat_sorted, return_index=True)
    split_indices = np.append(split_indices, len(z_sorted))

    for i, bin_id in enumerate(unique_bins):
        start = split_indices[i]
        end = split_indices[i + 1]
        z_vals = z_sorted[start:end]

        # Apply percentile filter
        threshold = np.percentile(z_vals, percentile)
        z_filtered = z_vals[z_vals <= threshold]

        if len(z_filtered) == 0:
            continue

        # Compute bin indices
        bx = bin_id // ny
        by = bin_id % ny

        z_mean[bx, by] = np.mean(z_filtered)
        z_min[bx, by] = np.min(z_filtered)
        z_max[bx, by] = np.max(z_filtered)
        z_std[bx, by] = np.std(z_filtered, ddof=0)
        count[bx, by] = len(z_filtered)

        # Mode via quantization
        z_quant = np.round(z_filtered / mode_bin) * mode_bin
        uniq, counts = np.unique(z_quant, return_counts=True)
        z_mode[bx, by] = uniq[np.argmax(counts)]

    return z_mean, z_min, z_max, z_std, z_mode, count


def _mode_statistic(vals: ArrayLike) -> float:
    if len(vals) == 0:
        return np.nan
    uniq, counts = np.unique(vals, return_counts=True)
    return float(uniq[np.argmax(counts)])


def _fit_plane(neighbor_pts: ArrayLike) -> Tuple[float, float, float]:
    """
    Fit a plane z = ax + by + c to neighbor points.
    Uses a trimmed re-fit to reduce outlier influence.
    """
    A = np.c_[neighbor_pts[:, 0], neighbor_pts[:, 1], np.ones(len(neighbor_pts))]
    coeffs, *_ = np.linalg.lstsq(A, neighbor_pts[:, 2], rcond=None)
    # Compute residuals for trimming
    z_fit_neighbors = A @ coeffs
    residuals = np.abs(neighbor_pts[:, 2] - z_fit_neighbors)
    keep = residuals <= np.percentile(residuals, 75)  # trim top quartile
    if keep.sum() >= 3:
        A_trim = A[keep]
        z_trim = neighbor_pts[keep, 2]
        coeffs, *_ = np.linalg.lstsq(A_trim, z_trim, rcond=None)
    return coeffs
