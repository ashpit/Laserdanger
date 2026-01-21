"""
Profile extraction module - extracts 1D cross-shore transects from point clouds.
Matches MATLAB Get3_1Dprofiles.m functionality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

ArrayLike = np.ndarray


@dataclass(frozen=True)
class TransectConfig:
    """Configuration for transect extraction."""
    # Transect line endpoints (UTM coordinates)
    x1: float  # Backshore x
    y1: float  # Backshore y
    x2: float  # Offshore x
    y2: float  # Offshore y

    # Alongshore offsets for multiple transects (meters)
    alongshore_spacings: Tuple[float, ...] = (-8, -6, -4, -2, 0, 2, 4, 6, 8, 10)

    # Cross-shore resolution (meters)
    resolution: float = 0.25

    # Tolerance for points near transect line (meters)
    tolerance: float = 1.0

    # Extension beyond endpoints (meters) [backshore, offshore]
    extend_line: Tuple[float, float] = (0, -300)

    # Outlier removal threshold (meters from quadratic fit)
    outlier_threshold: float = 0.4

    # Maximum gap size to interpolate (meters)
    max_gap: float = 4.0


@dataclass
class TransectResult:
    """Result of transect extraction."""
    x1d: ArrayLike  # Cross-shore distance array (1D)
    Z3D: ArrayLike  # Elevation matrix (n_transects x n_positions)
    transect_coords: List[Tuple[ArrayLike, ArrayLike]]  # (x, y) coords for each transect


def gapsize(x: ArrayLike) -> ArrayLike:
    """
    Calculate the gap size at each position in an array.

    A gap is defined as consecutive NaN values. Returns an array where
    each NaN position contains the length of its gap, and non-NaN positions
    contain 0.

    Matches MATLAB gapsize.m by Aslak Grinsted.

    Parameters
    ----------
    x : array
        Input array (1D)

    Returns
    -------
    array
        Gap sizes, same length as input
    """
    x = np.asarray(x).ravel()
    has_data = ~np.isnan(x)

    # Find indices where we have data, with sentinels at start and end
    data_indices = np.concatenate([[-1], np.where(has_data)[0], [len(x)]])

    sz = np.zeros(len(x), dtype=int)

    for i in range(len(data_indices) - 1):
        start = data_indices[i] + 1
        end = data_indices[i + 1]
        gap_length = end - start
        if gap_length > 0:
            sz[start:end] = gap_length

    return sz


def inpaint_nans(x: ArrayLike, z: ArrayLike, max_gap: float = 4.0) -> ArrayLike:
    """
    Interpolate NaN values in z, but only for gaps smaller than max_gap.

    Parameters
    ----------
    x : array
        X coordinates (1D)
    z : array
        Z values with NaN gaps (1D)
    max_gap : float
        Maximum gap size (in x units) to interpolate

    Returns
    -------
    array
        Z values with small gaps filled
    """
    x = np.asarray(x).ravel()
    z = np.asarray(z).ravel().copy()

    if len(x) != len(z):
        raise ValueError("x and z must have the same length")

    # Calculate gap sizes
    gaps = gapsize(z)

    # Determine resolution from x spacing
    dx = np.median(np.diff(x))
    max_gap_points = int(np.ceil(max_gap / dx))

    # Find valid (non-NaN) points
    valid = ~np.isnan(z)

    if valid.sum() < 2:
        return z  # Can't interpolate with fewer than 2 points

    # Create interpolator from valid points
    try:
        interp = interp1d(
            x[valid], z[valid],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
    except ValueError:
        return z

    # Fill only small gaps
    small_gap_mask = np.isnan(z) & (gaps <= max_gap_points) & (gaps > 0)
    z[small_gap_mask] = interp(x[small_gap_mask])

    return z


def _project_points_to_line(
    points: ArrayLike,
    line_start: Tuple[float, float],
    line_vec: ArrayLike,
    tolerance: float
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Project 2D points onto a line and filter by distance.

    Parameters
    ----------
    points : array (N, 2)
        Points to project
    line_start : tuple
        (x, y) of line start point
    line_vec : array (2,)
        Unit vector along line
    tolerance : float
        Maximum distance from line to keep points

    Returns
    -------
    proj_dist : array
        Distance along line from start for each kept point
    dist_to_line : array
        Perpendicular distance from line for each kept point
    mask : array
        Boolean mask of points within tolerance
    """
    line_start = np.array(line_start)
    line_vec = np.array(line_vec)

    # Vector from line start to each point
    to_points = points - line_start

    # Project onto line (dot product with unit vector)
    proj_length = to_points @ line_vec

    # Get projected points on the line
    proj_points = line_start + proj_length[:, np.newaxis] * line_vec

    # Distance from each point to its projection
    dist_to_line = np.sqrt(np.sum((points - proj_points) ** 2, axis=1))

    # Filter by tolerance
    mask = dist_to_line <= tolerance

    return proj_length, dist_to_line, mask


def _fit_quadratic_and_remove_outliers(
    x: ArrayLike,
    z: ArrayLike,
    threshold: float = 0.4,
    max_iterations: int = 3
) -> ArrayLike:
    """
    Fit quadratic to data and set outliers to NaN.

    Uses iterative robust fitting: fits quadratic, removes worst outliers,
    refits until convergence or max iterations.

    Parameters
    ----------
    x : array
        X coordinates
    z : array
        Z values
    threshold : float
        Remove points with |residual| >= threshold
    max_iterations : int
        Maximum fitting iterations

    Returns
    -------
    array
        Z values with outliers set to NaN
    """
    if len(x) < 3:
        return z.copy()

    z_clean = z.copy()
    valid = ~np.isnan(z_clean)

    if valid.sum() < 3:
        return z_clean

    # Iterative robust fitting
    for iteration in range(max_iterations):
        current_valid = ~np.isnan(z_clean)
        if current_valid.sum() < 3:
            break

        # Fit quadratic: z = a*x^2 + b*x + c
        try:
            x_valid = x[current_valid]
            z_valid = z_clean[current_valid]
            A = np.column_stack([x_valid ** 2, x_valid, np.ones(len(x_valid))])
            coeffs, *_ = np.linalg.lstsq(A, z_valid, rcond=None)

            # Compute fitted values and residuals for ALL points
            z_fit = coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2]
            residuals = np.abs(z_clean - z_fit)

            # Find outliers (only among currently valid points)
            outlier_mask = (residuals >= threshold) & current_valid

            if not outlier_mask.any():
                break  # No more outliers found

            z_clean[outlier_mask] = np.nan
        except (np.linalg.LinAlgError, ValueError):
            break

    return z_clean


def extract_transects(
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    config: Optional[TransectConfig] = None,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    x2: Optional[float] = None,
    y2: Optional[float] = None,
) -> TransectResult:
    """
    Extract 1D cross-shore transects from a point cloud.

    Projects points onto shore-normal transect lines at multiple alongshore
    positions, removes outliers using quadratic fitting, bins to regular
    spacing, and interpolates small gaps.

    Matches MATLAB Get3_1Dprofiles.m functionality.

    Parameters
    ----------
    X, Y, Z : array
        Point cloud coordinates (1D arrays, same length)
    config : TransectConfig, optional
        Full configuration object. If None, uses x1/y1/x2/y2 with defaults.
    x1, y1 : float, optional
        Backshore endpoint of central transect (UTM)
    x2, y2 : float, optional
        Offshore endpoint of central transect (UTM)

    Returns
    -------
    TransectResult
        Contains x1d (cross-shore distance), Z3D (elevation matrix),
        and transect_coords (UTM coordinates of each transect)

    Examples
    --------
    >>> result = extract_transects(X, Y, Z, x1=476190, y1=3636333, x2=475620, y2=3636465)
    >>> x1d = result.x1d  # Cross-shore distance array
    >>> Z3D = result.Z3D  # Shape: (n_transects, n_positions)
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    Z = np.asarray(Z).ravel()

    if not (len(X) == len(Y) == len(Z)):
        raise ValueError("X, Y, Z must have the same length")

    # Build config
    if config is None:
        if x1 is None or y1 is None or x2 is None or y2 is None:
            raise ValueError("Must provide either config or x1/y1/x2/y2")
        config = TransectConfig(x1=x1, y1=y1, x2=x2, y2=y2)

    # Compute transect geometry
    ang = np.arctan2(config.y2 - config.y1, config.x2 - config.x1)

    # Alongshore direction (perpendicular to transect)
    alongshore_angle = ang + np.pi / 2
    dx_alongshore = np.cos(alongshore_angle)
    dy_alongshore = np.sin(alongshore_angle)

    # Distance along transect
    transect_length = np.sqrt(
        (config.x2 - config.x1) ** 2 + (config.y2 - config.y1) ** 2
    )

    # Cross-shore distance array
    extend_back, extend_off = config.extend_line
    end_dist = transect_length + extend_off
    if end_dist <= extend_back:
        # Offshore extension too large, use full transect length
        end_dist = transect_length
    x1d = np.arange(extend_back, end_dist, config.resolution)

    if len(x1d) == 0:
        x1d = np.arange(0, transect_length, config.resolution)

    # Points array for projection
    points = np.column_stack([X, Y])

    # Process each transect
    n_transects = len(config.alongshore_spacings)
    Z3D = np.full((n_transects, len(x1d)), np.nan)
    transect_coords = []

    for i, offset in enumerate(config.alongshore_spacings):
        # Offset transect endpoints alongshore
        xs = config.x1 + offset * dx_alongshore
        ys = config.y1 + offset * dy_alongshore
        xe = config.x2 + offset * dx_alongshore
        ye = config.y2 + offset * dy_alongshore

        # Unit vector along transect
        line_vec = np.array([xe - xs, ye - ys])
        line_vec = line_vec / np.linalg.norm(line_vec)

        # Store transect coordinates
        # Generate points along the transect line
        n_pts = int(np.ceil(transect_length / config.resolution)) + 1
        t = np.linspace(0, 1, n_pts)
        xt = xs + t * (xe - xs)
        yt = ys + t * (ye - ys)
        transect_coords.append((xt, yt))

        # Project points onto this transect
        proj_dist, dist_to_line, mask = _project_points_to_line(
            points, (xs, ys), line_vec, config.tolerance
        )

        if mask.sum() == 0:
            continue

        # Get cross-shore distance and z values for close points
        x_proj = proj_dist[mask]
        z_values = Z[mask].copy()

        # Quadratic outlier removal
        z_values = _fit_quadratic_and_remove_outliers(
            x_proj, z_values, config.outlier_threshold
        )

        # Bin to regular x1d spacing
        # Find nearest x1d bin for each projected point
        bin_indices = np.searchsorted(x1d, x_proj)
        bin_indices = np.clip(bin_indices, 0, len(x1d) - 1)

        # Also check the bin to the left
        bin_indices_left = np.maximum(bin_indices - 1, 0)

        # Choose closer bin
        dist_right = np.abs(x_proj - x1d[bin_indices])
        dist_left = np.abs(x_proj - x1d[bin_indices_left])
        use_left = dist_left < dist_right
        bin_indices[use_left] = bin_indices_left[use_left]

        # Average z values in each bin
        z1d = np.full(len(x1d), np.nan)
        for j in range(len(x1d)):
            in_bin = bin_indices == j
            if in_bin.sum() > 0:
                z_in_bin = z_values[in_bin]
                valid_z = z_in_bin[~np.isnan(z_in_bin)]
                if len(valid_z) > 0:
                    z1d[j] = np.mean(valid_z)

        # Interpolate small gaps
        z1d = inpaint_nans(x1d, z1d, config.max_gap)

        Z3D[i, :] = z1d

    return TransectResult(x1d=x1d, Z3D=Z3D, transect_coords=transect_coords)


def extract_single_transect(
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    resolution: float = 0.25,
    tolerance: float = 1.0,
    outlier_threshold: float = 0.4,
    max_gap: float = 4.0,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Extract a single transect profile.

    Convenience function for extracting one transect without the full
    TransectConfig machinery.

    Parameters
    ----------
    X, Y, Z : array
        Point cloud coordinates
    x1, y1 : float
        Backshore endpoint (UTM)
    x2, y2 : float
        Offshore endpoint (UTM)
    resolution : float
        Cross-shore spacing (default 0.25m)
    tolerance : float
        Maximum distance from line (default 1m)
    outlier_threshold : float
        Quadratic fit outlier threshold (default 0.4m)
    max_gap : float
        Maximum gap to interpolate (default 4m)

    Returns
    -------
    x1d : array
        Cross-shore distance
    z1d : array
        Elevation profile
    """
    config = TransectConfig(
        x1=x1, y1=y1, x2=x2, y2=y2,
        alongshore_spacings=(0,),  # Single transect at center
        resolution=resolution,
        tolerance=tolerance,
        outlier_threshold=outlier_threshold,
        max_gap=max_gap,
    )

    result = extract_transects(X, Y, Z, config)
    return result.x1d, result.Z3D[0, :]


def transect_to_utm(
    x1d: ArrayLike,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    extend_line: Tuple[float, float] = (0, -300),
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert cross-shore distances back to UTM coordinates.

    Parameters
    ----------
    x1d : array
        Cross-shore distances from backshore point
    x1, y1 : float
        Backshore endpoint (UTM)
    x2, y2 : float
        Offshore endpoint (UTM)
    extend_line : tuple
        Extension beyond endpoints used in extraction

    Returns
    -------
    X_utm, Y_utm : array
        UTM coordinates along transect
    """
    ang = np.arctan2(y2 - y1, x2 - x1)

    X_utm = x1 + x1d * np.cos(ang)
    Y_utm = y1 + x1d * np.sin(ang)

    return X_utm, Y_utm
