"""
Profile extraction module - extracts 1D cross-shore transects from point clouds.
Matches MATLAB Get3_1Dprofiles.m functionality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

ArrayLike = np.ndarray
logger = logging.getLogger(__name__)


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

    # Tolerance expansion rate (meters per meter from scanner)
    # 0.0 = fixed tolerance (default), >0 = adaptive tolerance that grows with distance
    expansion_rate: float = 0.0


@dataclass
class TransectResult:
    """Result of transect extraction."""
    x1d: ArrayLike  # Cross-shore distance array (1D)
    Z3D: ArrayLike  # Elevation matrix (n_transects x n_positions)
    transect_coords: List[Tuple[ArrayLike, ArrayLike]]  # (x, y) coords for each transect


def get_scanner_position(transform_matrix: ArrayLike) -> Tuple[float, float]:
    """
    Extract scanner position from 4x4 homogeneous transform matrix.

    The scanner position is the translation component (last column, rows 0-2).

    Parameters
    ----------
    transform_matrix : array (4, 4)
        Homogeneous transformation matrix from lidar to UTM coordinates

    Returns
    -------
    x, y : float
        Scanner position in UTM coordinates
    """
    transform_matrix = np.asarray(transform_matrix)
    if transform_matrix.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {transform_matrix.shape}")

    x = transform_matrix[0, 3]
    y = transform_matrix[1, 3]
    return x, y


def compute_transect_from_swath(
    X: ArrayLike,
    Y: ArrayLike,
    transform_matrix: Optional[ArrayLike] = None,
    scanner_position: Optional[Tuple[float, float]] = None,
    padding: float = 5.0,
    expansion_rate: float = 0.0,
) -> TransectConfig:
    """
    Auto-compute transect configuration from the lidar swath geometry.

    Places the transect along the centerline of the swath, from the scanner
    position (or data edge nearest to scanner) through the centroid of the data.

    Parameters
    ----------
    X, Y : array
        Point cloud coordinates (1D arrays)
    transform_matrix : array (4, 4), optional
        Homogeneous transformation matrix. Used to extract scanner position.
    scanner_position : tuple (x, y), optional
        Explicit scanner position in UTM. Overrides transform_matrix.
    padding : float
        Padding to add beyond data extent (meters)
    expansion_rate : float
        Tolerance expansion rate for adaptive tolerance (default 0.0 = fixed)

    Returns
    -------
    TransectConfig
        Configuration with auto-computed transect endpoints

    Notes
    -----
    The algorithm:
    1. Get scanner position from transform matrix or explicit coordinates
    2. Compute centroid of point cloud
    3. Create vector from scanner toward centroid (swath centerline direction)
    4. Find extent of data along this direction
    5. Set transect endpoints to cover full data extent with padding
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()

    # Remove NaN values
    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) < 10:
        raise ValueError("Not enough valid points to compute transect")

    # Get scanner position
    if scanner_position is not None:
        scanner_x, scanner_y = scanner_position
    elif transform_matrix is not None:
        scanner_x, scanner_y = get_scanner_position(transform_matrix)
    else:
        # Fallback: estimate scanner at corner of data with highest density
        # This is a heuristic - the scanner is usually at one vertex of the swath
        logger.warning("No scanner position provided, estimating from data geometry")
        scanner_x, scanner_y = _estimate_scanner_position(X, Y)

    # Compute data centroid
    centroid_x = np.mean(X)
    centroid_y = np.mean(Y)

    # Direction from scanner to centroid (swath centerline)
    dx = centroid_x - scanner_x
    dy = centroid_y - scanner_y
    dist_to_centroid = np.sqrt(dx**2 + dy**2)

    if dist_to_centroid < 1.0:
        # Scanner is at centroid - use PCA to find principal direction
        logger.warning("Scanner at data centroid, using PCA for direction")
        direction = _compute_principal_direction(X, Y)
        dx, dy = direction
    else:
        # Normalize direction
        dx /= dist_to_centroid
        dy /= dist_to_centroid

    # Project all points onto the transect line
    # proj = (point - scanner) · direction
    proj_dist = (X - scanner_x) * dx + (Y - scanner_y) * dy

    # Find extent of data along transect
    min_dist = np.min(proj_dist)
    max_dist = np.max(proj_dist)

    # Transect endpoints with padding
    # x1, y1 = backshore (closest to scanner)
    # x2, y2 = offshore (farthest from scanner)
    x1 = scanner_x + (min_dist - padding) * dx
    y1 = scanner_y + (min_dist - padding) * dy
    x2 = scanner_x + (max_dist + padding) * dx
    y2 = scanner_y + (max_dist + padding) * dy

    logger.info(
        f"Auto-computed transect: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f}), "
        f"length={np.sqrt((x2-x1)**2 + (y2-y1)**2):.1f}m"
    )

    return TransectConfig(
        x1=x1, y1=y1,
        x2=x2, y2=y2,
        alongshore_spacings=(0,),  # Single central transect by default
        extend_line=(0, 0),  # No extension needed, endpoints already cover data
        expansion_rate=expansion_rate,
    )


def _estimate_scanner_position(X: ArrayLike, Y: ArrayLike) -> Tuple[float, float]:
    """
    Estimate scanner position from data geometry.

    The scanner is typically at the apex of the fan-shaped swath.
    We find this by looking for the point where data density converges.
    """
    # Use convex hull to find vertices
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack([X, Y])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Find centroid
        centroid = np.array([np.mean(X), np.mean(Y)])

        # The scanner is likely the hull vertex with smallest spread angle
        # (i.e., where the fan originates)
        best_vertex = None
        min_spread = np.inf

        for i, vertex in enumerate(hull_points):
            # Compute vectors to adjacent hull vertices
            prev_vertex = hull_points[i - 1]
            next_vertex = hull_points[(i + 1) % len(hull_points)]

            v1 = prev_vertex - vertex
            v2 = next_vertex - vertex

            # Angle between adjacent edges
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # We want the vertex with smallest interior angle (sharp corner)
            if angle < min_spread:
                min_spread = angle
                best_vertex = vertex

        if best_vertex is not None:
            return float(best_vertex[0]), float(best_vertex[1])

    except Exception as e:
        logger.warning(f"ConvexHull method failed: {e}")

    # Fallback: use corner of bounding box closest to centroid direction
    centroid = np.array([np.mean(X), np.mean(Y)])
    corners = [
        (np.min(X), np.min(Y)),
        (np.min(X), np.max(Y)),
        (np.max(X), np.min(Y)),
        (np.max(X), np.max(Y)),
    ]

    # Find corner with most points nearby (scanner origin has high density)
    best_corner = corners[0]
    max_density = 0
    for corner in corners:
        dist = np.sqrt((X - corner[0])**2 + (Y - corner[1])**2)
        # Count points within 10% of max distance
        threshold = np.percentile(dist, 10)
        density = np.sum(dist < threshold)
        if density > max_density:
            max_density = density
            best_corner = corner

    return best_corner


def _compute_principal_direction(X: ArrayLike, Y: ArrayLike) -> Tuple[float, float]:
    """
    Compute principal direction of point cloud using PCA.
    """
    # Center the data
    X_centered = X - np.mean(X)
    Y_centered = Y - np.mean(Y)

    # Covariance matrix
    cov = np.cov(X_centered, Y_centered)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal direction is eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvalues)
    direction = eigenvectors[:, principal_idx]

    return float(direction[0]), float(direction[1])


def transect_config_from_dict(config_dict: Dict[str, Any]) -> TransectConfig:
    """
    Create TransectConfig from a dictionary (e.g., from JSON config).

    Parameters
    ----------
    config_dict : dict
        Dictionary with transect parameters. Supports two formats:

        Format 1 (endpoints):
            {"x1": float, "y1": float, "x2": float, "y2": float, ...}

        Format 2 (origin + azimuth):
            {"origin_x": float, "origin_y": float, "azimuth": float, "length": float, ...}

        Optional parameters:
            "alongshore_spacings": list of floats
            "resolution": float
            "tolerance": float

    Returns
    -------
    TransectConfig
    """
    # Check which format
    if "x1" in config_dict and "x2" in config_dict:
        x1 = float(config_dict["x1"])
        y1 = float(config_dict["y1"])
        x2 = float(config_dict["x2"])
        y2 = float(config_dict["y2"])
    elif "origin_x" in config_dict and "azimuth" in config_dict:
        origin_x = float(config_dict["origin_x"])
        origin_y = float(config_dict["origin_y"])
        azimuth_deg = float(config_dict["azimuth"])  # degrees from north, clockwise
        length = float(config_dict.get("length", 100.0))

        # Convert azimuth to math angle (from east, counterclockwise)
        azimuth_rad = np.radians(90 - azimuth_deg)

        x1 = origin_x
        y1 = origin_y
        x2 = origin_x + length * np.cos(azimuth_rad)
        y2 = origin_y + length * np.sin(azimuth_rad)
    else:
        raise ValueError(
            "Transect config must have either (x1, y1, x2, y2) or "
            "(origin_x, origin_y, azimuth, length)"
        )

    # Optional parameters
    kwargs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    if "alongshore_spacings" in config_dict:
        kwargs["alongshore_spacings"] = tuple(config_dict["alongshore_spacings"])
    if "resolution" in config_dict:
        kwargs["resolution"] = float(config_dict["resolution"])
    if "tolerance" in config_dict:
        kwargs["tolerance"] = float(config_dict["tolerance"])
    if "extend_line" in config_dict:
        kwargs["extend_line"] = tuple(config_dict["extend_line"])
    if "expansion_rate" in config_dict:
        kwargs["expansion_rate"] = float(config_dict["expansion_rate"])

    return TransectConfig(**kwargs)


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
    tolerance: float,
    scanner_position: Optional[Tuple[float, float]] = None,
    expansion_rate: float = 0.0,
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
        Base maximum distance from line to keep points
    scanner_position : tuple (x, y), optional
        Scanner position for adaptive tolerance. If provided with expansion_rate > 0,
        tolerance expands with distance from scanner.
    expansion_rate : float
        Tolerance expansion rate (meters per meter from scanner).
        0.0 = fixed tolerance (default), >0 = adaptive tolerance.
        Example: 0.02 means tolerance grows by 2cm per meter from scanner.

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

    # Filter by tolerance (adaptive or fixed)
    if expansion_rate > 0 and scanner_position is not None:
        scanner_pos = np.array(scanner_position)
        dist_from_scanner = np.sqrt(np.sum((points - scanner_pos) ** 2, axis=1))
        adaptive_tolerance = tolerance + dist_from_scanner * expansion_rate
        mask = dist_to_line <= adaptive_tolerance
    else:
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
    scanner_position: Optional[Tuple[float, float]] = None,
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
    scanner_position : tuple (x, y), optional
        Scanner position for adaptive tolerance. Used with config.expansion_rate.

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
            points, (xs, ys), line_vec, config.tolerance,
            scanner_position=scanner_position,
            expansion_rate=config.expansion_rate,
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
