"""Tests for profile extraction module."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import profiles  # noqa: E402


# =============================================================================
# gapsize() Tests
# =============================================================================

def test_gapsize_no_gaps():
    """Test gapsize with no NaN values."""
    x = np.array([1, 2, 3, 4, 5])
    sz = profiles.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 0, 0, 0, 0])


def test_gapsize_single_gap():
    """Test gapsize with a single gap."""
    x = np.array([1, np.nan, np.nan, np.nan, 5])
    sz = profiles.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 3, 3, 3, 0])


def test_gapsize_multiple_gaps():
    """Test gapsize with multiple gaps of different sizes."""
    x = np.array([1, np.nan, 3, np.nan, np.nan, 6, np.nan, 8])
    sz = profiles.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 1, 0, 2, 2, 0, 1, 0])


def test_gapsize_gap_at_start():
    """Test gapsize with gap at the start."""
    x = np.array([np.nan, np.nan, 3, 4, 5])
    sz = profiles.gapsize(x)
    np.testing.assert_array_equal(sz, [2, 2, 0, 0, 0])


def test_gapsize_gap_at_end():
    """Test gapsize with gap at the end."""
    x = np.array([1, 2, 3, np.nan, np.nan])
    sz = profiles.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 0, 0, 2, 2])


def test_gapsize_all_nan():
    """Test gapsize with all NaN values."""
    x = np.array([np.nan, np.nan, np.nan])
    sz = profiles.gapsize(x)
    np.testing.assert_array_equal(sz, [3, 3, 3])


# =============================================================================
# inpaint_nans() Tests
# =============================================================================

def test_inpaint_nans_small_gap():
    """Test that small gaps are filled."""
    x = np.array([0, 1, 2, 3, 4, 5])
    z = np.array([0, np.nan, np.nan, 3, 4, 5], dtype=float)

    z_filled = profiles.inpaint_nans(x, z, max_gap=3.0)

    # Gap of 2 points (2m) should be filled
    assert not np.isnan(z_filled[1])
    assert not np.isnan(z_filled[2])
    # Linear interpolation: 0 -> 3 over indices 0-3
    np.testing.assert_allclose(z_filled[1], 1.0)
    np.testing.assert_allclose(z_filled[2], 2.0)


def test_inpaint_nans_large_gap_not_filled():
    """Test that large gaps are not filled."""
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10], dtype=float)

    z_filled = profiles.inpaint_nans(x, z, max_gap=4.0)

    # Gap of 9 points (9m) is > max_gap, should remain NaN
    assert np.isnan(z_filled[5])


def test_inpaint_nans_preserves_valid():
    """Test that valid values are preserved."""
    x = np.array([0, 1, 2, 3, 4])
    z = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    z_filled = profiles.inpaint_nans(x, z, max_gap=4.0)

    np.testing.assert_array_equal(z, z_filled)


def test_inpaint_nans_edge_gaps():
    """Test handling of gaps at array edges."""
    x = np.array([0, 1, 2, 3, 4])
    z = np.array([np.nan, 1, 2, 3, np.nan], dtype=float)

    z_filled = profiles.inpaint_nans(x, z, max_gap=2.0)

    # Edge gaps can't be interpolated (no data outside)
    assert np.isnan(z_filled[0])
    assert np.isnan(z_filled[4])
    # Interior values preserved
    np.testing.assert_array_equal(z_filled[1:4], z[1:4])


# =============================================================================
# _fit_quadratic_and_remove_outliers() Tests
# =============================================================================

def test_quadratic_outlier_removal():
    """Test that outliers from quadratic fit are removed."""
    # Create dense data so a single outlier doesn't skew the fit too much
    x = np.linspace(0, 10, 50)
    # Quadratic: z = 0.1*x^2
    z = 0.1 * x ** 2
    z_with_outlier = z.copy()

    # Add a moderate outlier (0.8m above surface) at index 25 (x=5)
    outlier_idx = 25
    z_with_outlier[outlier_idx] = z[outlier_idx] + 0.8

    z_clean = profiles._fit_quadratic_and_remove_outliers(
        x, z_with_outlier, threshold=0.4
    )

    # Outlier should be NaN
    assert np.isnan(z_clean[outlier_idx]), "Outlier should be removed"

    # Most other points should be preserved (allow some edge effects)
    valid_clean = ~np.isnan(z_clean)
    assert valid_clean.sum() >= 45, f"Should keep most points, got {valid_clean.sum()}"


def test_quadratic_outlier_removal_multiple():
    """Test removal of multiple outliers."""
    x = np.linspace(0, 10, 100)
    z = 0.1 * x ** 2
    z_with_outliers = z.copy()

    # Add several outliers
    outlier_indices = [20, 50, 80]
    for idx in outlier_indices:
        z_with_outliers[idx] = z[idx] + 1.0  # 1m above surface

    z_clean = profiles._fit_quadratic_and_remove_outliers(
        x, z_with_outliers, threshold=0.4
    )

    # All outliers should be NaN
    for idx in outlier_indices:
        assert np.isnan(z_clean[idx]), f"Outlier at index {idx} should be removed"

    # Most other points preserved
    valid_clean = ~np.isnan(z_clean)
    assert valid_clean.sum() >= 90


def test_quadratic_no_outliers():
    """Test that clean data passes through unchanged."""
    x = np.array([0, 1, 2, 3, 4])
    z = x ** 2 * 0.1  # Clean quadratic

    z_clean = profiles._fit_quadratic_and_remove_outliers(x, z, threshold=0.4)

    np.testing.assert_array_almost_equal(z_clean, z)


# =============================================================================
# _project_points_to_line() Tests
# =============================================================================

def test_project_points_to_line():
    """Test point projection onto a line."""
    # Line from (0,0) to (10,0) - horizontal
    line_start = (0, 0)
    line_vec = np.array([1, 0])  # Unit vector along x

    # Points at various positions
    points = np.array([
        [5, 0],    # On line
        [5, 0.5],  # 0.5m above line
        [5, 2],    # 2m above line (outside tolerance)
        [0, 0],    # At start
        [10, 0],   # At "end"
    ])

    proj_dist, dist_to_line, mask = profiles._project_points_to_line(
        points, line_start, line_vec, tolerance=1.0
    )

    # Check projection distances
    np.testing.assert_allclose(proj_dist, [5, 5, 5, 0, 10])

    # Check distances to line
    np.testing.assert_allclose(dist_to_line, [0, 0.5, 2, 0, 0])

    # Check mask (tolerance=1.0)
    np.testing.assert_array_equal(mask, [True, True, False, True, True])


def test_project_points_diagonal_line():
    """Test projection onto a diagonal line."""
    # Line from (0,0) to (10,10) - 45 degree angle
    line_start = (0, 0)
    line_vec = np.array([1, 1]) / np.sqrt(2)  # Unit vector

    points = np.array([
        [5, 5],    # On line
        [4, 6],    # Off line but close
    ])

    proj_dist, dist_to_line, mask = profiles._project_points_to_line(
        points, line_start, line_vec, tolerance=2.0
    )

    # Point (5,5) is at distance sqrt(50) ~ 7.07 along the line
    expected_dist = np.sqrt(50)
    np.testing.assert_allclose(proj_dist[0], expected_dist, rtol=1e-5)

    # First point is on line
    np.testing.assert_allclose(dist_to_line[0], 0, atol=1e-10)


def test_project_points_adaptive_tolerance():
    """Test adaptive tolerance that expands with distance from scanner."""
    # Scanner at origin, line along x-axis
    scanner_position = (0, 0)
    line_start = (0, 0)
    line_vec = np.array([1, 0])

    # Points at different distances from scanner
    # All 1.5m perpendicular to line
    points = np.array([
        [10, 1.5],   # 10m from scanner, 1.5m perpendicular
        [50, 1.5],   # 50m from scanner, 1.5m perpendicular
        [100, 1.5],  # 100m from scanner, 1.5m perpendicular
    ])

    # Fixed tolerance = 1.0m -> all points outside tolerance
    _, _, mask_fixed = profiles._project_points_to_line(
        points, line_start, line_vec, tolerance=1.0,
        scanner_position=None, expansion_rate=0.0
    )
    np.testing.assert_array_equal(mask_fixed, [False, False, False])

    # Adaptive tolerance with expansion_rate=0.02
    # At 10m: tolerance = 1.0 + 10*0.02 = 1.2m -> point outside (1.5m > 1.2m)
    # At 50m: tolerance = 1.0 + 50*0.02 = 2.0m -> point inside (1.5m < 2.0m)
    # At 100m: tolerance = 1.0 + 100*0.02 = 3.0m -> point inside (1.5m < 3.0m)
    _, _, mask_adaptive = profiles._project_points_to_line(
        points, line_start, line_vec, tolerance=1.0,
        scanner_position=scanner_position, expansion_rate=0.02
    )
    np.testing.assert_array_equal(mask_adaptive, [False, True, True])


# =============================================================================
# extract_transects() Tests
# =============================================================================

def test_extract_transects_synthetic_beach():
    """Test transect extraction on synthetic beach data."""
    np.random.seed(42)

    # Create synthetic sloped beach
    # Transect line from (100, 100) to (200, 100) - horizontal
    n_points = 1000
    X = np.random.uniform(90, 210, n_points)
    Y = np.random.uniform(95, 105, n_points)  # Close to transect line
    # Beach slope: z = -0.1 * (x - 100) = elevation drops seaward
    Z = -0.1 * (X - 100) + np.random.normal(0, 0.05, n_points)

    config = profiles.TransectConfig(
        x1=100, y1=100,
        x2=200, y2=100,
        alongshore_spacings=(0,),  # Single transect
        resolution=1.0,
        tolerance=5.0,
        outlier_threshold=0.4,
        max_gap=4.0,
        extend_line=(0, 0),
    )

    result = profiles.extract_transects(X, Y, Z, config)

    # Check output shapes
    assert result.x1d.ndim == 1
    assert result.Z3D.shape[0] == 1  # One transect
    assert result.Z3D.shape[1] == len(result.x1d)

    # Check that profile follows expected slope
    valid = ~np.isnan(result.Z3D[0, :])
    if valid.sum() > 10:
        # Fit line to extracted profile
        x_valid = result.x1d[valid]
        z_valid = result.Z3D[0, valid]
        slope, intercept = np.polyfit(x_valid, z_valid, 1)

        # Should be close to -0.1 slope
        assert abs(slope - (-0.1)) < 0.02, f"Expected slope ~-0.1, got {slope}"


def test_extract_transects_multiple():
    """Test extraction of multiple transects."""
    np.random.seed(42)

    # Create beach with alongshore variation
    n_points = 2000
    X = np.random.uniform(0, 100, n_points)
    Y = np.random.uniform(-15, 15, n_points)  # Spans alongshore
    Z = -0.1 * X + 0.01 * Y + np.random.normal(0, 0.05, n_points)

    config = profiles.TransectConfig(
        x1=0, y1=0,
        x2=100, y2=0,
        alongshore_spacings=(-10, -5, 0, 5, 10),
        resolution=1.0,
        tolerance=3.0,
        extend_line=(0, 0),
    )

    result = profiles.extract_transects(X, Y, Z, config)

    # Should have 5 transects
    assert result.Z3D.shape[0] == 5

    # Each transect should have some valid data
    for i in range(5):
        valid = ~np.isnan(result.Z3D[i, :])
        assert valid.sum() > 10, f"Transect {i} has too few valid points"


def test_extract_single_transect():
    """Test convenience function for single transect."""
    np.random.seed(42)

    X = np.random.uniform(0, 50, 500)
    Y = np.random.uniform(-2, 2, 500)
    Z = -0.1 * X + np.random.normal(0, 0.05, 500)

    x1d, z1d = profiles.extract_single_transect(
        X, Y, Z,
        x1=0, y1=0, x2=50, y2=0,
        resolution=0.5,
        tolerance=3.0,
    )

    assert x1d.ndim == 1
    assert z1d.ndim == 1
    assert len(x1d) == len(z1d)


def test_extract_transects_empty_region():
    """Test handling of transect with no nearby points."""
    # Points only in a small region
    X = np.array([10, 11, 12, 13, 14])
    Y = np.array([0, 0, 0, 0, 0])
    Z = np.array([1, 1, 1, 1, 1])

    config = profiles.TransectConfig(
        x1=0, y1=0,
        x2=100, y2=0,
        alongshore_spacings=(0, 50),  # Second transect at y=50 - no points there
        resolution=1.0,
        tolerance=1.0,
        extend_line=(0, 0),
    )

    result = profiles.extract_transects(X, Y, Z, config)

    # First transect should have some data
    assert not np.all(np.isnan(result.Z3D[0, :]))

    # Second transect (at y=50) should be all NaN - no points nearby
    assert np.all(np.isnan(result.Z3D[1, :]))


def test_transect_config_defaults():
    """Test TransectConfig default values."""
    config = profiles.TransectConfig(x1=0, y1=0, x2=100, y2=0)

    assert config.resolution == 0.25
    assert config.tolerance == 1.0
    assert config.outlier_threshold == 0.4
    assert config.max_gap == 4.0
    assert len(config.alongshore_spacings) == 10


# =============================================================================
# transect_to_utm() Tests
# =============================================================================

def test_transect_to_utm_horizontal():
    """Test UTM conversion for horizontal transect."""
    x1d = np.array([0, 10, 20, 30])

    X_utm, Y_utm = profiles.transect_to_utm(
        x1d, x1=100, y1=50, x2=200, y2=50
    )

    # Horizontal line: Y stays constant, X increases
    np.testing.assert_allclose(Y_utm, 50)
    np.testing.assert_allclose(X_utm, [100, 110, 120, 130])


def test_transect_to_utm_diagonal():
    """Test UTM conversion for diagonal transect."""
    x1d = np.array([0, np.sqrt(2)])  # Distance along 45-degree line

    X_utm, Y_utm = profiles.transect_to_utm(
        x1d, x1=0, y1=0, x2=1, y2=1
    )

    np.testing.assert_allclose(X_utm, [0, 1], atol=1e-10)
    np.testing.assert_allclose(Y_utm, [0, 1], atol=1e-10)


# =============================================================================
# Input Validation Tests
# =============================================================================

def test_extract_transects_mismatched_lengths():
    """Test error on mismatched input lengths."""
    X = np.array([1, 2, 3])
    Y = np.array([1, 2])
    Z = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="same length"):
        profiles.extract_transects(X, Y, Z, x1=0, y1=0, x2=10, y2=0)


def test_extract_transects_no_config_no_coords():
    """Test error when neither config nor coordinates provided."""
    X = np.array([1, 2, 3])
    Y = np.array([1, 2, 3])
    Z = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="config or x1/y1/x2/y2"):
        profiles.extract_transects(X, Y, Z)


def test_inpaint_nans_mismatched_lengths():
    """Test error on mismatched x and z lengths."""
    x = np.array([1, 2, 3])
    z = np.array([1, 2])

    with pytest.raises(ValueError, match="same length"):
        profiles.inpaint_nans(x, z)


# =============================================================================
# Auto-Transect Computation Tests
# =============================================================================

def test_get_scanner_position():
    """Test extracting scanner position from transform matrix."""
    transform = np.array([
        [1, 0, 0, 476191.0],
        [0, 1, 0, 3636211.0],
        [0, 0, 1, 19.5],
        [0, 0, 0, 1],
    ])

    x, y = profiles.get_scanner_position(transform)
    assert x == 476191.0
    assert y == 3636211.0


def test_get_scanner_position_invalid_shape():
    """Test error on invalid transform matrix shape."""
    transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    with pytest.raises(ValueError, match="4x4"):
        profiles.get_scanner_position(transform)


def test_compute_transect_from_swath_with_scanner():
    """Test auto-computing transect from swath with known scanner position."""
    np.random.seed(42)

    # Create fan-shaped swath emanating from scanner at (0, 0)
    # Points spread out at 45-degree angle from scanner
    n_points = 500
    distances = np.random.uniform(10, 100, n_points)
    angles = np.random.uniform(-0.3, 0.3, n_points)  # Small spread around 45 degrees
    base_angle = np.pi / 4  # 45 degrees

    X = distances * np.cos(base_angle + angles)
    Y = distances * np.sin(base_angle + angles)

    config = profiles.compute_transect_from_swath(
        X, Y,
        scanner_position=(0, 0),
        padding=5.0,
    )

    # Transect should start near scanner and extend through data
    # Direction should be approximately 45 degrees
    dx = config.x2 - config.x1
    dy = config.y2 - config.y1
    angle = np.arctan2(dy, dx)

    assert abs(angle - base_angle) < 0.2, f"Expected angle ~45°, got {np.degrees(angle):.1f}°"
    assert config.x1 < config.x2  # Should extend outward from scanner


def test_compute_transect_from_swath_with_transform():
    """Test auto-computing transect using transform matrix."""
    np.random.seed(42)

    scanner_x, scanner_y = 500.0, 1000.0
    n_points = 300
    distances = np.random.uniform(20, 80, n_points)
    angles = np.random.uniform(-0.2, 0.2, n_points)

    # Points extending west from scanner
    X = scanner_x - distances * np.cos(angles)
    Y = scanner_y + distances * np.sin(angles)

    transform = np.array([
        [1, 0, 0, scanner_x],
        [0, 1, 0, scanner_y],
        [0, 0, 1, 10.0],
        [0, 0, 0, 1],
    ])

    config = profiles.compute_transect_from_swath(
        X, Y,
        transform_matrix=transform,
        padding=2.0,
    )

    # Should produce a valid transect
    assert config.x1 != config.x2 or config.y1 != config.y2
    length = np.sqrt((config.x2 - config.x1)**2 + (config.y2 - config.y1)**2)
    assert length > 50  # Should cover most of the data


def test_compute_transect_from_swath_not_enough_points():
    """Test error with too few points."""
    X = np.array([1, 2, 3])
    Y = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="Not enough"):
        profiles.compute_transect_from_swath(X, Y, scanner_position=(0, 0))


def test_transect_config_from_dict_endpoints():
    """Test creating TransectConfig from dict with endpoints."""
    config_dict = {
        "x1": 100.0,
        "y1": 200.0,
        "x2": 150.0,
        "y2": 250.0,
        "resolution": 0.5,
        "tolerance": 2.0,
    }

    config = profiles.transect_config_from_dict(config_dict)

    assert config.x1 == 100.0
    assert config.y1 == 200.0
    assert config.x2 == 150.0
    assert config.y2 == 250.0
    assert config.resolution == 0.5
    assert config.tolerance == 2.0


def test_transect_config_from_dict_azimuth():
    """Test creating TransectConfig from dict with origin and azimuth."""
    config_dict = {
        "origin_x": 100.0,
        "origin_y": 200.0,
        "azimuth": 90.0,  # Due east
        "length": 50.0,
    }

    config = profiles.transect_config_from_dict(config_dict)

    assert config.x1 == 100.0
    assert config.y1 == 200.0
    # Azimuth 90 (east) means x2 should be x1 + length
    assert abs(config.x2 - 150.0) < 0.1
    assert abs(config.y2 - 200.0) < 0.1


def test_transect_config_from_dict_missing_fields():
    """Test error when config dict is missing required fields."""
    config_dict = {"x1": 100.0}  # Missing other coordinates

    with pytest.raises(ValueError, match="must have either"):
        profiles.transect_config_from_dict(config_dict)
