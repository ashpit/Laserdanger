"""
Phase 2: core algorithms (binning, SNR, residual kernel filtering).
These functions stay pure and accept numpy arrays; I/O happens upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d

ArrayLike = np.ndarray


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


def bin_point_cloud(
    points: ArrayLike,
    bin_size: float = 0.1,
    mode_bin: float = 0.05,
    x_edges: ArrayLike | None = None,
    y_edges: ArrayLike | None = None,
) -> BinnedGrid:
    """
    Rasterize points (N,3) onto a regular grid and compute statistics.
    Uses scipy.stats.binned_statistic_2d for mean/min/max/std/count;
    mode is computed from quantized elevations (mode_bin resolution).
    """
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    x, y, z = points.T
    x_edges = _bin_edges(x, bin_size) if x_edges is None else x_edges
    y_edges = _bin_edges(y, bin_size) if y_edges is None else y_edges

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


def residual_kernel_filter(
    points: ArrayLike,
    window_radius: float = 1.0,
    max_residual: float = 0.5,
    min_neighbors: int = 6,
) -> ArrayLike:
    """
    Simple residual-based ground filter.
    For each point, fit a plane to neighbors within window_radius and keep the point
    if its residual is <= max_residual. Returns boolean mask of ground points.
    """
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    tree = cKDTree(points[:, :2])
    keep = np.zeros(len(points), dtype=bool)
    for idx, (x, y, z) in enumerate(points):
        neighbor_idx = tree.query_ball_point([x, y], r=window_radius)
        if len(neighbor_idx) < min_neighbors:
            continue
        neighbor_pts = points[neighbor_idx]
        coeffs = _fit_plane(neighbor_pts)
        z_fit = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        residual = abs(z - z_fit)
        if residual <= max_residual:
            keep[idx] = True
    return keep


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
