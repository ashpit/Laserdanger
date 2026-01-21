"""
Utility functions for lidar processing pipeline.
Implements algorithms from various MATLAB utility files.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage, signal


# =============================================================================
# 2D Outlier Detection (detect_outliers_conv2D.m)
# =============================================================================

@dataclass
class OutlierResult:
    """Result from 2D outlier detection."""
    is_outlier: np.ndarray  # Boolean mask (n_x, n_t)
    Z_filtered: np.ndarray  # Cleaned matrix with outliers as NaN
    gradient_magnitude: np.ndarray  # Gradient magnitude field
    laplacian: np.ndarray  # Laplacian (curvature) field


def detect_outliers_conv2d(
    Z_xt: np.ndarray,
    dt: float,
    ig_length: float = 60.0,
    gradient_threshold_std: float = 2.5,
    laplacian_threshold_std: float = 2.5,
    median_kernel: Tuple[int, int] = (3, 3),
    dilate_kernel: Tuple[int, int] = (6, 2),
) -> OutlierResult:
    """
    Detect spike outliers in Z(x,t) matrix using gradient and Laplacian.

    Matches MATLAB detect_outliers_conv2D.m algorithm:
    1. Remove minimum surface (IG timescale)
    2. Apply median filter
    3. Compute Sobel gradients and Laplacian
    4. Identify outliers with both high gradient AND high curvature
    5. Dilate outlier regions

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    dt : float
        Time step in seconds
    ig_length : float
        Window for moving minimum (default 60s)
    gradient_threshold_std : float
        Threshold in standard deviations for gradient (default 2.5)
    laplacian_threshold_std : float
        Threshold in standard deviations for Laplacian (default 2.5)
    median_kernel : tuple
        Median filter kernel size (x, t) (default 3x3)
    dilate_kernel : tuple
        Dilation kernel size (x, t) (default 6x2)

    Returns
    -------
    OutlierResult
        Detection result with mask and filtered data
    """
    n_x, n_t = Z_xt.shape

    # Moving minimum along time axis to get baseline
    ig_filt = int(ig_length / dt)
    if ig_filt < 1:
        ig_filt = 1

    Z_min = ndimage.minimum_filter1d(
        np.nan_to_num(Z_xt, nan=np.nanmax(Z_xt)),
        size=ig_filt,
        axis=1,
        mode='nearest',
    )

    # Difference from baseline
    Z_diff = Z_xt - Z_min

    # Median filter for smoothing
    Z_smooth = ndimage.median_filter(
        np.nan_to_num(Z_diff, nan=0),
        size=median_kernel,
        mode='nearest',
    )

    # Sobel gradient kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobel_t = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    grad_x = ndimage.convolve(Z_smooth, sobel_x, mode='nearest')
    grad_t = ndimage.convolve(Z_smooth, sobel_t, mode='nearest')
    grad_mag = np.sqrt(grad_x**2 + grad_t**2)

    # Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian = ndimage.convolve(Z_smooth, laplacian_kernel, mode='nearest')

    # Compute thresholds
    valid_grad = grad_mag[~np.isnan(Z_xt)]
    valid_lap = np.abs(laplacian[~np.isnan(Z_xt)])

    if len(valid_grad) > 0:
        grad_thresh = np.mean(valid_grad) + gradient_threshold_std * np.std(valid_grad)
        lap_thresh = np.mean(valid_lap) + laplacian_threshold_std * np.std(valid_lap)
    else:
        grad_thresh = np.inf
        lap_thresh = np.inf

    # Identify outliers: both high gradient AND high curvature
    is_outlier = (grad_mag > grad_thresh) & (np.abs(laplacian) > lap_thresh)

    # Also mark existing NaN as outliers
    is_outlier |= np.isnan(Z_xt)

    # Dilate outlier regions
    struct = np.ones(dilate_kernel, dtype=bool)
    is_outlier = ndimage.binary_dilation(is_outlier, structure=struct)

    # Create filtered output
    Z_filtered = Z_xt.copy()
    Z_filtered[is_outlier] = np.nan

    return OutlierResult(
        is_outlier=is_outlier,
        Z_filtered=Z_filtered,
        gradient_magnitude=grad_mag,
        laplacian=laplacian,
    )


# =============================================================================
# Intensity Contours (Get_intensity_contours.m)
# =============================================================================

@dataclass
class IntensityContour:
    """Single intensity contour result."""
    threshold: float
    x_positions: np.ndarray  # Cross-shore position at each time
    time_indices: np.ndarray  # Valid time indices
    mean_position: float
    std_position: float
    valid_fraction: float


def get_intensity_contours(
    I_xt: np.ndarray,
    x1d: np.ndarray,
    thresholds: Union[float, List[float]],
    smooth_window: int = 5,
    max_gap: int = 5,
    min_points: int = 10,
) -> List[IntensityContour]:
    """
    Extract contour lines where intensity crosses thresholds.

    Used for water edge detection based on foam intensity.

    Parameters
    ----------
    I_xt : array (n_x, n_t)
        Intensity matrix
    x1d : array (n_x,)
        Cross-shore positions
    thresholds : float or list
        Intensity threshold(s) to extract
    smooth_window : int
        Moving average window for smoothing (default 5)
    max_gap : int
        Maximum gap to interpolate (default 5)
    min_points : int
        Minimum valid points for a contour (default 10)

    Returns
    -------
    list of IntensityContour
        Contour for each threshold
    """
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    n_x, n_t = I_xt.shape
    results = []

    for thresh in thresholds:
        x_contour = np.full(n_t, np.nan)

        for j in range(n_t):
            I_profile = I_xt[:, j]
            valid = ~np.isnan(I_profile)

            if valid.sum() < 3:
                continue

            x_valid = x1d[valid]
            I_valid = I_profile[valid]

            # Find threshold crossings
            I_diff = I_valid - thresh
            sign_changes = np.where(np.diff(np.sign(I_diff)) != 0)[0]

            if len(sign_changes) == 0:
                continue

            # Take the seaward-most (last) crossing
            idx = sign_changes[-1]

            # Linear interpolation for exact crossing position
            if idx < len(x_valid) - 1:
                x0, x1 = x_valid[idx], x_valid[idx + 1]
                I0, I1 = I_valid[idx], I_valid[idx + 1]
                if abs(I1 - I0) > 1e-10:
                    frac = (thresh - I0) / (I1 - I0)
                    x_contour[j] = x0 + frac * (x1 - x0)
                else:
                    x_contour[j] = x0
            else:
                x_contour[j] = x_valid[idx]

        # Smooth with moving average
        valid_mask = ~np.isnan(x_contour)
        if valid_mask.sum() > smooth_window:
            x_smooth = x_contour.copy()
            x_smooth[valid_mask] = ndimage.uniform_filter1d(
                x_contour[valid_mask], smooth_window, mode='nearest'
            )
            x_contour = x_smooth

        # Fill small gaps
        x_contour = _interp_small_gaps_1d(x_contour, max_gap)

        # Compute statistics
        valid_x = x_contour[~np.isnan(x_contour)]
        valid_indices = np.where(~np.isnan(x_contour))[0]

        if len(valid_x) >= min_points:
            results.append(IntensityContour(
                threshold=thresh,
                x_positions=x_contour,
                time_indices=valid_indices,
                mean_position=np.mean(valid_x),
                std_position=np.std(valid_x),
                valid_fraction=len(valid_x) / n_t,
            ))

    return results


# =============================================================================
# Time Utilities (roundToHalfHour.m, split_L1_days.m)
# =============================================================================

def round_to_half_hour(dt: datetime) -> datetime:
    """
    Round datetime to nearest 30-minute interval.

    Matches MATLAB roundToHalfHour.m:
        14:23:45 → 14:30:00
        14:37:30 → 14:30:00
        14:45:30 → 15:00:00

    Parameters
    ----------
    dt : datetime
        Input datetime

    Returns
    -------
    datetime
        Rounded datetime
    """
    # Round minutes to 0 or 30
    minutes = dt.minute
    if minutes < 15:
        rounded_minutes = 0
        hour_offset = 0
    elif minutes < 45:
        rounded_minutes = 30
        hour_offset = 0
    else:
        rounded_minutes = 0
        hour_offset = 1

    result = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    if hour_offset:
        result = result + timedelta(hours=1)

    return result


def round_to_interval(dt: datetime, interval_minutes: int = 30) -> datetime:
    """
    Round datetime to nearest interval.

    Parameters
    ----------
    dt : datetime
        Input datetime
    interval_minutes : int
        Interval in minutes (default 30)

    Returns
    -------
    datetime
        Rounded datetime
    """
    total_minutes = dt.hour * 60 + dt.minute + dt.second / 60
    rounded_minutes = round(total_minutes / interval_minutes) * interval_minutes

    hours = int(rounded_minutes // 60) % 24
    minutes = int(rounded_minutes % 60)

    # Handle day overflow
    day_offset = int(rounded_minutes // (24 * 60))
    result = dt.replace(hour=hours, minute=minutes, second=0, microsecond=0)
    if day_offset:
        result = result + timedelta(days=day_offset)

    return result


def split_by_day(
    timestamps: List[datetime],
    data: Optional[List] = None,
) -> Dict[datetime, List]:
    """
    Split data by calendar day.

    Parameters
    ----------
    timestamps : list of datetime
        Timestamps to split
    data : list, optional
        Corresponding data items (if None, returns indices)

    Returns
    -------
    dict
        Mapping from day (datetime with time=00:00) to list of items/indices
    """
    result: Dict[datetime, List] = {}

    for i, ts in enumerate(timestamps):
        day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)

        if day_start not in result:
            result[day_start] = []

        if data is not None:
            result[day_start].append(data[i])
        else:
            result[day_start].append(i)

    return result


def deduplicate_by_interval(
    timestamps: List[datetime],
    data: List,
    interval_minutes: int = 30,
) -> Tuple[List[datetime], List]:
    """
    Remove duplicates based on time interval rounding.

    Keeps first occurrence of each rounded time.
    Matches MATLAB split_L1_days.m uniqueness logic.

    Parameters
    ----------
    timestamps : list of datetime
    data : list
        Corresponding data items
    interval_minutes : int
        Rounding interval (default 30)

    Returns
    -------
    unique_timestamps, unique_data : lists
    """
    seen = set()
    unique_ts = []
    unique_data = []

    for ts, item in zip(timestamps, data):
        rounded = round_to_interval(ts, interval_minutes)
        key = rounded.isoformat()

        if key not in seen:
            seen.add(key)
            unique_ts.append(ts)
            unique_data.append(item)

    return unique_ts, unique_data


# =============================================================================
# Gap/Interpolation Utilities
# =============================================================================

def gapsize(x: np.ndarray) -> np.ndarray:
    """
    Compute gap sizes for NaN values in 1D array.

    Each NaN element gets the size of its contiguous NaN gap.
    Non-NaN elements get 0.

    Matches MATLAB gapsize.m

    Parameters
    ----------
    x : array
        1D array with potential NaN values

    Returns
    -------
    array
        Gap sizes (same shape as x)
    """
    x = np.asarray(x)
    is_nan = np.isnan(x)
    sz = np.zeros_like(x, dtype=int)

    if not is_nan.any():
        return sz

    # Label connected NaN regions
    labels, n_labels = ndimage.label(is_nan)

    for label in range(1, n_labels + 1):
        mask = labels == label
        gap_size = mask.sum()
        sz[mask] = gap_size

    return sz


def inpaint_nans_1d(
    x: np.ndarray,
    z: np.ndarray,
    max_gap: float = 4.0,
) -> np.ndarray:
    """
    Fill NaN gaps in 1D array using linear interpolation.

    Only fills gaps smaller than max_gap (in x units).

    Parameters
    ----------
    x : array
        X coordinates
    z : array
        Z values (may contain NaN)
    max_gap : float
        Maximum gap to fill (in x units)

    Returns
    -------
    array
        Interpolated z values
    """
    z_out = z.copy()
    valid = ~np.isnan(z)

    if valid.sum() < 2:
        return z_out

    # Find gap boundaries
    valid_idx = np.where(valid)[0]

    for i in range(len(valid_idx) - 1):
        start = valid_idx[i]
        end = valid_idx[i + 1]

        if end - start <= 1:
            continue

        # Gap size in x units
        gap_x = x[end] - x[start]

        if gap_x <= max_gap:
            # Linear interpolation
            idx_range = np.arange(start, end + 1)
            z_out[idx_range] = np.interp(
                x[idx_range],
                [x[start], x[end]],
                [z[start], z[end]],
            )

    return z_out


def inpaint_nans_2d(
    Z: np.ndarray,
    max_gap: int = 4,
    axis: int = 0,
) -> np.ndarray:
    """
    Fill NaN gaps in 2D array using linear interpolation along axis.

    Parameters
    ----------
    Z : array (n_rows, n_cols)
        2D array with NaN values
    max_gap : int
        Maximum gap size in grid points
    axis : int
        Axis along which to interpolate (0=rows, 1=cols)

    Returns
    -------
    array
        Interpolated array
    """
    Z_out = Z.copy()

    if axis == 0:
        for j in range(Z.shape[1]):
            col = Z[:, j]
            x = np.arange(len(col))
            Z_out[:, j] = inpaint_nans_1d(x, col, max_gap)
    else:
        for i in range(Z.shape[0]):
            row = Z[i, :]
            x = np.arange(len(row))
            Z_out[i, :] = inpaint_nans_1d(x, row, max_gap)

    return Z_out


# =============================================================================
# Helper functions
# =============================================================================

def _interp_small_gaps_1d(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate small gaps in 1D array (by index count)."""
    result = arr.copy()
    valid = ~np.isnan(arr)

    if valid.sum() < 2:
        return result

    valid_idx = np.where(valid)[0]

    for j in range(len(valid_idx) - 1):
        start = valid_idx[j]
        end = valid_idx[j + 1]
        gap_size = end - start - 1

        if 0 < gap_size <= max_gap:
            x_interp = np.arange(start, end + 1)
            result[start:end + 1] = np.interp(
                x_interp,
                [start, end],
                [result[start], result[end]],
            )

    return result
