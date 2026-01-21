import sys
import numpy as np
import pytest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase2  # noqa: E402


# =============================================================================
# Basic Binning Tests
# =============================================================================

def test_bin_point_cloud_basic_stats() -> None:
    """Test basic binning produces correct grid shape and statistics."""
    pts = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 1.0],
            [1.1, 0.1, 2.0],
            [1.2, 0.2, 2.0],
        ]
    )
    grid = phase2.bin_point_cloud(pts, bin_size=1.0, mode_bin=0.1, percentile=None)
    assert grid.z_mean.shape == (2, 2)
    # First bin (x<1, y<1) should have mean 1.0 and count 2
    np.testing.assert_allclose(grid.z_mean[0, 0], 1.0)
    np.testing.assert_allclose(grid.count[0, 0], 2)
    # Second bin (x>=1, y<1) should have mean 2.0
    np.testing.assert_allclose(grid.z_mean[1, 0], 2.0)
    assert not np.isnan(grid.z_mode[0, 0])
    # Check valid_mask exists
    assert hasattr(grid, 'valid_mask')
    assert grid.valid_mask.shape == grid.z_mean.shape


def test_compute_snr_handles_zero_count() -> None:
    """Test SNR computation handles zero count bins correctly."""
    z_mean = np.array([[1.0, 0.5], [0.0, 0.0]])
    z_std = np.array([[0.1, 0.2], [0.0, 0.0]])
    count = np.array([[10, 4], [0, 0]])
    snr = phase2.compute_snr(z_mean, z_std, count)
    assert np.isnan(snr[1, 0]) and np.isnan(snr[1, 1])
    assert snr[0, 0] > snr[0, 1]


def test_residual_kernel_filter_removes_outlier() -> None:
    """Test that KD-tree based filter removes outliers above the plane."""
    # Points on plane z = 0 plus one outlier at z=5
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    outlier = np.array([[0.25, 0.25, 5.0]])
    pts = np.vstack([base, outlier])
    mask = phase2.residual_kernel_filter(pts, window_radius=0.8, max_residual=0.2, min_neighbors=4)
    assert mask.sum() == len(base)  # all plane points kept, outlier dropped
    assert not mask[-1]


def test_bin_point_cloud_temporal() -> None:
    """Test temporal binning produces correct shape and statistics."""
    pts = np.array(
        [
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.5],
            [0.1, 0.0, 2.0],
            [1.2, 0.0, 3.0],
        ]
    )
    intensities = np.array([10.0, 14.0, 20.0, 30.0])
    times = np.array([0.1, 0.2, 0.6, 1.1])

    grid = phase2.bin_point_cloud_temporal(
        pts, intensities, times, x_bin_size=1.0, time_bin_size=0.5
    )

    assert grid.z_mean.shape == (3, 2)  # 3 time bins, 2 x bins
    np.testing.assert_allclose(np.diff(grid.t_edges), 0.5)
    np.testing.assert_allclose(grid.z_mean[0, 0], 1.25)  # first time bin average z
    np.testing.assert_allclose(grid.z_mean[1, 0], 2.0)  # middle time bin single point
    np.testing.assert_allclose(grid.z_mean[2, 1], 3.0)  # last bin second x column
    assert np.isnan(grid.z_mean[2, 0]) and grid.count[2, 0] == 0  # empty cell normalized to NaN
    np.testing.assert_allclose(grid.intensity_mean[0, 0], 12.0)


# =============================================================================
# SNR Filtering Tests (Phase 1.1)
# =============================================================================

def test_snr_valid_mask_thresholds() -> None:
    """Test that valid_mask correctly identifies bins based on SNR and count."""
    # Create points with varying densities and spread
    np.random.seed(42)

    # High density, low spread (high SNR) in first quadrant
    high_snr = np.column_stack([
        np.random.uniform(0, 0.9, 100),
        np.random.uniform(0, 0.9, 100),
        np.random.normal(5.0, 0.01, 100)  # Very low std -> high SNR
    ])

    # Low density in second quadrant (count < min_count)
    low_count = np.column_stack([
        np.random.uniform(1.1, 1.9, 5),
        np.random.uniform(0, 0.9, 5),
        np.random.normal(5.0, 0.01, 5)
    ])

    # High spread in third quadrant (low SNR)
    low_snr = np.column_stack([
        np.random.uniform(0, 0.9, 50),
        np.random.uniform(1.1, 1.9, 50),
        np.random.normal(5.0, 2.0, 50)  # High std -> low SNR
    ])

    pts = np.vstack([high_snr, low_count, low_snr])

    grid = phase2.bin_point_cloud(
        pts, bin_size=1.0, percentile=None,
        snr_threshold=100.0, min_count=10
    )

    # High SNR bin should be valid
    assert grid.valid_mask[0, 0] == True, "High SNR bin should be valid"

    # Low count bin should be invalid
    assert grid.valid_mask[1, 0] == False, "Low count bin should be invalid"

    # Low SNR bin should be invalid (high variance)
    assert grid.valid_mask[0, 1] == False, "Low SNR bin should be invalid"


def test_apply_snr_filter_returns_vectors() -> None:
    """Test that apply_snr_filter returns 1D arrays of valid data."""
    np.random.seed(42)

    # Create a simple grid with some valid and invalid bins
    pts = np.column_stack([
        np.random.uniform(0, 2, 200),
        np.random.uniform(0, 2, 200),
        np.random.normal(5.0, 0.01, 200)  # Low std for high SNR
    ])

    grid = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=None)

    Xutm, Yutm, Zmean, Zmax, Zmin, Zstd = phase2.apply_snr_filter(grid)

    # All returned arrays should be 1D
    assert Xutm.ndim == 1
    assert Yutm.ndim == 1
    assert Zmean.ndim == 1

    # All arrays should have the same length
    assert len(Xutm) == len(Yutm) == len(Zmean) == len(Zmax) == len(Zmin) == len(Zstd)

    # Length should match number of valid bins
    assert len(Xutm) == grid.valid_mask.sum()


# =============================================================================
# Percentile Filtering Tests (Phase 1.3)
# =============================================================================

def test_percentile_filter_reduces_z_values() -> None:
    """Test that 50th percentile filtering keeps lower z values."""
    np.random.seed(42)

    # Create points with bimodal z distribution (ground + vegetation)
    n_ground = 50
    n_veg = 50

    # Ground points around z=0
    ground = np.column_stack([
        np.random.uniform(0, 0.9, n_ground),
        np.random.uniform(0, 0.9, n_ground),
        np.random.normal(0.0, 0.05, n_ground)
    ])

    # Vegetation points around z=2 (above ground)
    vegetation = np.column_stack([
        np.random.uniform(0, 0.9, n_veg),
        np.random.uniform(0, 0.9, n_veg),
        np.random.normal(2.0, 0.1, n_veg)
    ])

    pts = np.vstack([ground, vegetation])

    # With percentile=50, should get ground surface (lower z values)
    grid_filtered = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=50)

    # Without percentile filter, mean should be higher
    grid_unfiltered = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=None)

    # Filtered mean should be significantly lower (closer to ground)
    assert grid_filtered.z_mean[0, 0] < grid_unfiltered.z_mean[0, 0]

    # Filtered mean should be close to ground level (0)
    assert abs(grid_filtered.z_mean[0, 0]) < 0.5


def test_percentile_filter_with_different_values() -> None:
    """Test different percentile values produce expected results."""
    np.random.seed(42)

    # Create uniform z distribution from 0 to 10
    pts = np.column_stack([
        np.random.uniform(0, 0.9, 1000),
        np.random.uniform(0, 0.9, 1000),
        np.random.uniform(0, 10, 1000)
    ])

    grid_25 = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=25)
    grid_50 = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=50)
    grid_75 = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=75)

    # Lower percentile should give lower mean
    assert grid_25.z_mean[0, 0] < grid_50.z_mean[0, 0] < grid_75.z_mean[0, 0]


def test_percentile_filter_empty_after_threshold() -> None:
    """Test handling when all points are above percentile threshold."""
    # All points have exactly the same z value
    pts = np.array([
        [0.1, 0.1, 5.0],
        [0.2, 0.2, 5.0],
        [0.3, 0.3, 5.0],
    ])

    # Should not crash, should return valid statistics
    grid = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=50)

    # All points equal, so 50th percentile includes all of them
    assert not np.isnan(grid.z_mean[0, 0])
    np.testing.assert_allclose(grid.z_mean[0, 0], 5.0)


# =============================================================================
# Delaunay Residual Filter Tests (Phase 1.2)
# =============================================================================

def test_delaunay_filter_removes_outliers() -> None:
    """Test that Delaunay-based filter removes points above ground plane."""
    np.random.seed(42)

    # Create ground plane z = 0.01*x + 0.01*y
    n_ground = 200
    x_ground = np.random.uniform(0, 20, n_ground)
    y_ground = np.random.uniform(0, 20, n_ground)
    z_ground = 0.01 * x_ground + 0.01 * y_ground + np.random.normal(0, 0.02, n_ground)
    ground = np.column_stack([x_ground, y_ground, z_ground])

    # Add some outliers above the ground
    n_outliers = 20
    x_out = np.random.uniform(0, 20, n_outliers)
    y_out = np.random.uniform(0, 20, n_outliers)
    z_out = 0.01 * x_out + 0.01 * y_out + 2.0  # 2m above ground
    outliers = np.column_stack([x_out, y_out, z_out])

    pts = np.vstack([ground, outliers])

    # Apply Delaunay filter
    ground_idx = phase2.residual_kernel_filter_delaunay(
        pts, cell_size=5.0, max_residual=0.2, min_points_per_cell=10
    )

    # Most ground points should be kept, most outliers removed
    n_ground_kept = np.sum(ground_idx < n_ground)
    n_outliers_kept = np.sum(ground_idx >= n_ground)

    assert n_ground_kept > n_ground * 0.7, f"Should keep most ground points, kept {n_ground_kept}/{n_ground}"
    assert n_outliers_kept < n_outliers * 0.3, f"Should remove most outliers, kept {n_outliers_kept}/{n_outliers}"


def test_two_stage_filter_progressive_refinement() -> None:
    """Test that two-stage filtering progressively removes noise."""
    np.random.seed(42)

    # Create ground plane with noise at different scales
    n_points = 500
    x = np.random.uniform(0, 50, n_points)
    y = np.random.uniform(0, 50, n_points)
    z_ground = 0.01 * x + 0.01 * y

    # Add small noise (should pass both filters)
    small_noise = np.random.normal(0, 0.03, n_points)

    # Add some medium outliers (should fail fine filter)
    medium_outliers_idx = np.random.choice(n_points, 30, replace=False)

    # Add some large outliers (should fail coarse filter)
    large_outliers_idx = np.random.choice(n_points, 20, replace=False)

    z = z_ground + small_noise
    z[medium_outliers_idx] += 0.15  # Above fine threshold (0.1) but below coarse (0.2)
    z[large_outliers_idx] += 0.5   # Above coarse threshold (0.2)

    pts = np.column_stack([x, y, z])

    # Apply two-stage filter with default passes [(10, 0.2), (3, 0.1)]
    filtered = phase2.residual_kernel_filter_two_stage(pts)

    # Should have removed most outliers
    assert len(filtered) < len(pts)
    assert len(filtered) > n_points * 0.7  # But kept most ground points


def test_two_stage_filter_custom_passes() -> None:
    """Test two-stage filter with custom pass configuration."""
    np.random.seed(42)

    # Create simple ground plane
    n = 100
    pts = np.column_stack([
        np.random.uniform(0, 20, n),
        np.random.uniform(0, 20, n),
        np.random.normal(0, 0.02, n)
    ])

    # Single pass
    filtered_1pass = phase2.residual_kernel_filter_two_stage(
        pts, passes=[(5.0, 0.1)]
    )

    # Two passes
    filtered_2pass = phase2.residual_kernel_filter_two_stage(
        pts, passes=[(10.0, 0.2), (5.0, 0.1)]
    )

    # Both should return reasonable results
    assert len(filtered_1pass) > 0
    assert len(filtered_2pass) > 0


def test_fit_plane_svd_flat_plane() -> None:
    """Test SVD plane fitting on a known flat plane."""
    # Points exactly on z = 2
    pts = np.array([
        [0, 0, 2],
        [1, 0, 2],
        [0, 1, 2],
        [1, 1, 2],
        [0.5, 0.5, 2],
    ], dtype=float)

    z_fit = phase2._fit_plane_svd(pts)

    np.testing.assert_allclose(z_fit, 2.0, atol=1e-10)


def test_fit_plane_svd_tilted_plane() -> None:
    """Test SVD plane fitting on a tilted plane z = x + y."""
    pts = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 2],
        [0.5, 0.5, 1],
    ], dtype=float)

    z_fit = phase2._fit_plane_svd(pts)
    expected = pts[:, 0] + pts[:, 1]  # z = x + y

    np.testing.assert_allclose(z_fit, expected, atol=1e-10)


def test_points_in_triangle() -> None:
    """Test point-in-triangle detection."""
    triangle = np.array([
        [0, 0],
        [2, 0],
        [1, 2],
    ], dtype=float)

    points = np.array([
        [1, 0.5],    # Inside
        [0.5, 0.5],  # Inside
        [1, 1],      # Inside
        [0, 0],      # On vertex
        [3, 0],      # Outside
        [-1, 0],     # Outside
        [1, 3],      # Outside
    ], dtype=float)

    inside = phase2._points_in_triangle(points, triangle)

    assert inside[0] == True   # Inside
    assert inside[1] == True   # Inside
    assert inside[2] == True   # Inside
    assert inside[3] == True   # On vertex (should be inside)
    assert inside[4] == False  # Outside
    assert inside[5] == False  # Outside
    assert inside[6] == False  # Outside


# =============================================================================
# Grid Utility Tests
# =============================================================================

def test_grid_to_points_valid_only() -> None:
    """Test converting grid to points with valid_only=True."""
    np.random.seed(42)

    pts = np.column_stack([
        np.random.uniform(0, 2, 200),
        np.random.uniform(0, 2, 200),
        np.random.normal(5.0, 0.01, 200)
    ])

    grid = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=None)
    points = phase2.grid_to_points(grid, use_valid_only=True)

    assert points.shape[1] == 3
    assert len(points) == grid.valid_mask.sum()


def test_grid_to_points_all() -> None:
    """Test converting grid to points with valid_only=False."""
    np.random.seed(42)

    pts = np.column_stack([
        np.random.uniform(0, 2, 200),
        np.random.uniform(0, 2, 200),
        np.random.normal(5.0, 0.01, 200)
    ])

    grid = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=None)
    points = phase2.grid_to_points(grid, use_valid_only=False)

    assert points.shape[1] == 3
    # Should include all non-NaN bins
    assert len(points) == np.sum(~np.isnan(grid.z_mean))


def test_bin_edges_helper() -> None:
    """Test bin edge computation."""
    values = np.array([0.15, 0.25, 0.95, 1.05])
    edges = phase2.bin_edges(values, 0.5)

    # Should cover all values with 0.5 spacing
    assert edges[0] <= values.min()
    assert edges[-1] >= values.max()
    np.testing.assert_allclose(np.diff(edges), 0.5)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_bin_point_cloud_wrong_shape() -> None:
    """Test that wrong input shape raises ValueError."""
    pts = np.array([[1, 2], [3, 4]])  # 2D instead of 3D

    with pytest.raises(ValueError, match="shape"):
        phase2.bin_point_cloud(pts)


def test_residual_filter_wrong_shape() -> None:
    """Test that wrong input shape raises ValueError."""
    pts = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="shape"):
        phase2.residual_kernel_filter(pts)

    with pytest.raises(ValueError, match="shape"):
        phase2.residual_kernel_filter_delaunay(pts)


def test_temporal_binning_mismatched_lengths() -> None:
    """Test that mismatched array lengths raise ValueError."""
    pts = np.array([[0, 0, 1], [1, 1, 2]])
    intensities = np.array([10.0])  # Wrong length
    times = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="same length"):
        phase2.bin_point_cloud_temporal(pts, intensities, times)


def test_empty_after_filtering() -> None:
    """Test handling when all bins are invalid after SNR filtering."""
    # Very few points with high variance -> all invalid
    pts = np.array([
        [0.1, 0.1, 0.0],
        [0.2, 0.2, 10.0],  # High variance
        [0.3, 0.3, 5.0],
    ])

    grid = phase2.bin_point_cloud(
        pts, bin_size=1.0, percentile=None,
        snr_threshold=100.0, min_count=10
    )

    # All bins should be invalid (count < 10)
    assert grid.valid_mask.sum() == 0

    # apply_snr_filter should return empty arrays
    Xutm, Yutm, Zmean, _, _, _ = phase2.apply_snr_filter(grid)
    assert len(Xutm) == 0
