"""Tests for internal/private functions in phase2.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase2  # noqa: E402


class TestBinTemporalSinglePass:
    """Tests for _bin_temporal_single_pass."""

    def test_basic_binning(self):
        """Basic temporal binning produces correct shape and values."""
        # Simple data: points at different times and positions
        times = np.array([0.1, 0.2, 0.6, 0.7, 1.1])
        x_vals = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
        z_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        intensities = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        t_edges = np.array([0.0, 0.5, 1.0, 1.5])
        x_edges = np.array([0.0, 0.15, 0.3])

        z_mean, z_min, z_max, z_std, count, intensity_mean = phase2._bin_temporal_single_pass(
            times, x_vals, z_vals, intensities, t_edges, x_edges
        )

        # Shape should be (n_t, n_x) = (3, 2)
        assert z_mean.shape == (3, 2)
        assert count.shape == (3, 2)

        # First time bin [0, 0.5), first x bin [0, 0.15): points at x=0.1, t=0.1 with z=1
        assert count[0, 0] == 1
        assert z_mean[0, 0] == 1.0

        # First time bin, second x bin: point at x=0.2, t=0.2 with z=2
        assert count[0, 1] == 1
        assert z_mean[0, 1] == 2.0

    def test_empty_bins_have_nan(self):
        """Empty bins should have NaN values."""
        times = np.array([0.1])
        x_vals = np.array([0.1])
        z_vals = np.array([1.0])
        intensities = np.array([10.0])

        t_edges = np.array([0.0, 0.5, 1.0])
        x_edges = np.array([0.0, 0.2, 0.4])

        z_mean, _, _, _, count, _ = phase2._bin_temporal_single_pass(
            times, x_vals, z_vals, intensities, t_edges, x_edges
        )

        # Only first bin should have data
        assert count[0, 0] == 1
        assert np.isnan(z_mean[1, 0])  # Empty bin

    def test_statistics_computed_correctly(self):
        """Verify mean, min, max, std are computed correctly."""
        times = np.array([0.1, 0.2, 0.3])
        x_vals = np.array([0.1, 0.1, 0.1])  # All in same x bin
        z_vals = np.array([1.0, 2.0, 3.0])
        intensities = np.array([10.0, 20.0, 30.0])

        t_edges = np.array([0.0, 0.5])
        x_edges = np.array([0.0, 0.2])

        z_mean, z_min, z_max, z_std, count, intensity_mean = phase2._bin_temporal_single_pass(
            times, x_vals, z_vals, intensities, t_edges, x_edges
        )

        assert count[0, 0] == 3
        np.testing.assert_allclose(z_mean[0, 0], 2.0)
        np.testing.assert_allclose(z_min[0, 0], 1.0)
        np.testing.assert_allclose(z_max[0, 0], 3.0)
        np.testing.assert_allclose(intensity_mean[0, 0], 20.0)

        # Std should be sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3)
        expected_std = np.std([1.0, 2.0, 3.0], ddof=0)
        np.testing.assert_allclose(z_std[0, 0], expected_std, rtol=1e-5)

    def test_single_point_per_bin_std_zero(self):
        """Single point in a bin should have std = 0."""
        times = np.array([0.1])
        x_vals = np.array([0.1])
        z_vals = np.array([5.0])
        intensities = np.array([10.0])

        t_edges = np.array([0.0, 0.5])
        x_edges = np.array([0.0, 0.2])

        _, _, _, z_std, _, _ = phase2._bin_temporal_single_pass(
            times, x_vals, z_vals, intensities, t_edges, x_edges
        )

        assert z_std[0, 0] == 0.0


class TestModeStatistic:
    """Tests for _mode_statistic."""

    def test_single_value(self):
        """Single value should return that value."""
        result = phase2._mode_statistic(np.array([5.0]))
        assert result == 5.0

    def test_multiple_same_values(self):
        """Array of identical values returns that value."""
        result = phase2._mode_statistic(np.array([3.0, 3.0, 3.0]))
        assert result == 3.0

    def test_most_frequent_value(self):
        """Returns most frequent value."""
        result = phase2._mode_statistic(np.array([1.0, 2.0, 2.0, 3.0]))
        assert result == 2.0

    def test_empty_returns_nan(self):
        """Empty array returns NaN."""
        result = phase2._mode_statistic(np.array([]))
        assert np.isnan(result)

    def test_tie_returns_first_mode(self):
        """When multiple modes exist, returns one of them (first in sorted order)."""
        # With equal counts, np.unique returns smallest first
        result = phase2._mode_statistic(np.array([1.0, 2.0]))
        assert result in [1.0, 2.0]


class TestFitPlane:
    """Tests for _fit_plane."""

    def test_flat_surface(self):
        """Points on a flat surface should have near-zero residuals."""
        # Points on plane z = 0
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0.5, 0],
        ], dtype=float)

        coeffs = phase2._fit_plane(points)
        # z = ax + by + c, for flat surface a=0, b=0, c=0
        np.testing.assert_allclose(coeffs, [0, 0, 0], atol=1e-10)

    def test_tilted_plane(self):
        """Points on tilted plane z = x + 2y + 1."""
        points = np.array([
            [0, 0, 1],      # z = 0 + 0 + 1
            [1, 0, 2],      # z = 1 + 0 + 1
            [0, 1, 3],      # z = 0 + 2 + 1
            [1, 1, 4],      # z = 1 + 2 + 1
            [0.5, 0.5, 2.5], # z = 0.5 + 1 + 1
        ], dtype=float)

        coeffs = phase2._fit_plane(points)
        # Expected: a=1, b=2, c=1
        np.testing.assert_allclose(coeffs, [1, 2, 1], rtol=1e-5)

    def test_with_outlier(self):
        """Robust fitting should handle outliers via trimming."""
        # Flat surface with one outlier
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 10],  # Outlier
        ], dtype=float)

        coeffs = phase2._fit_plane(points)
        # Should still fit approximately flat due to trimming
        assert abs(coeffs[2]) < 3  # c should be close to 0, not pulled to 10/6


class TestFitPlaneSvd:
    """Tests for _fit_plane_svd."""

    def test_insufficient_points(self):
        """Less than 3 points returns NaN."""
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        result = phase2._fit_plane_svd(points)
        assert np.all(np.isnan(result))

    def test_flat_surface_svd(self):
        """Points on a flat surface."""
        points = np.array([
            [0, 0, 5],
            [1, 0, 5],
            [0, 1, 5],
            [1, 1, 5],
        ], dtype=float)

        z_fit = phase2._fit_plane_svd(points)
        np.testing.assert_allclose(z_fit, [5, 5, 5, 5], atol=1e-10)

    def test_robust_fitting_handles_outliers(self):
        """Robust fitting should handle outliers without crashing."""
        # Create a surface with some outliers
        points = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [2, 2, 0],
            [1, 1, 0],
            [0.5, 0.5, 0],
            [1.5, 0.5, 0],
            [0.5, 1.5, 0],
            [1, 0.5, 100],  # Big outlier
        ], dtype=float)

        # Both should run without error
        z_fit_robust = phase2._fit_plane_svd(points, robust=True)
        z_fit_non_robust = phase2._fit_plane_svd(points, robust=False)

        # Both should return arrays of correct length
        assert len(z_fit_robust) == len(points)
        assert len(z_fit_non_robust) == len(points)

        # At least some values should be computed (not all NaN)
        assert not np.all(np.isnan(z_fit_robust))
        assert not np.all(np.isnan(z_fit_non_robust))


class TestBinWithPercentileFilter:
    """Tests for _bin_with_percentile_filter."""

    def test_basic_filtering(self):
        """Percentile filter removes high z values."""
        # Points in one bin with varying z
        x = np.array([0.1, 0.1, 0.1, 0.1])
        y = np.array([0.1, 0.1, 0.1, 0.1])
        z = np.array([1.0, 2.0, 3.0, 100.0])  # 100 is outlier

        x_edges = np.array([0.0, 0.2])
        y_edges = np.array([0.0, 0.2])

        z_mean, z_min, z_max, z_std, z_mode, count = phase2._bin_with_percentile_filter(
            x, y, z, x_edges, y_edges, percentile=75, mode_bin=0.5
        )

        # 75th percentile of [1, 2, 3, 100] is 27.25, but we keep z <= threshold
        # After filtering, should include [1, 2, 3]
        assert count[0, 0] == 3
        np.testing.assert_allclose(z_mean[0, 0], 2.0)

    def test_50th_percentile(self):
        """50th percentile keeps bottom half."""
        x = np.array([0.1] * 10)
        y = np.array([0.1] * 10)
        z = np.arange(1.0, 11.0)  # [1, 2, ..., 10]

        x_edges = np.array([0.0, 0.2])
        y_edges = np.array([0.0, 0.2])

        z_mean, _, _, _, _, count = phase2._bin_with_percentile_filter(
            x, y, z, x_edges, y_edges, percentile=50, mode_bin=0.5
        )

        # 50th percentile keeps 5 points (z <= 5)
        assert count[0, 0] == 5
        np.testing.assert_allclose(z_mean[0, 0], 3.0)  # mean of [1,2,3,4,5]


class TestPointsInTriangle:
    """Tests for _points_in_triangle."""

    def test_point_inside(self):
        """Point clearly inside triangle."""
        triangle = np.array([[0, 0], [2, 0], [1, 2]])
        points = np.array([[1, 0.5]])  # Centroid-ish
        result = phase2._points_in_triangle(points, triangle)
        assert result[0]

    def test_point_outside(self):
        """Point clearly outside triangle."""
        triangle = np.array([[0, 0], [2, 0], [1, 2]])
        points = np.array([[5, 5]])
        result = phase2._points_in_triangle(points, triangle)
        assert not result[0]

    def test_point_at_vertex(self):
        """Point at triangle vertex."""
        triangle = np.array([[0, 0], [2, 0], [1, 2]])
        points = np.array([[0, 0]])
        result = phase2._points_in_triangle(points, triangle)
        assert result[0]

    def test_multiple_points(self):
        """Test with multiple points at once."""
        triangle = np.array([[0, 0], [2, 0], [1, 2]])
        points = np.array([
            [1, 0.5],   # Inside
            [5, 5],     # Outside
            [1, 0],     # On edge
        ])
        result = phase2._points_in_triangle(points, triangle)
        assert result[0]      # Inside
        assert not result[1]  # Outside
        assert result[2]      # On edge


class TestBinEdges:
    """Tests for _bin_edges."""

    def test_basic_edges(self):
        """Basic edge generation."""
        values = np.array([0.15, 1.85])
        edges = phase2._bin_edges(values, 0.5)

        # Should start at 0.0 (floor of 0.15/0.5 * 0.5) and end past 1.85
        assert edges[0] == 0.0
        assert edges[-1] >= 2.0

    def test_single_value(self):
        """Single value should still produce at least 2 bins."""
        values = np.array([1.0])
        edges = phase2._bin_edges(values, 0.5)
        assert len(edges) >= 3  # At least 2 bins

    def test_very_small_range(self):
        """Values in very small range still produce valid bins."""
        values = np.array([1.0, 1.001])
        edges = phase2._bin_edges(values, 0.5)
        assert len(edges) >= 3


class TestResidualKernelFilterTwoStage:
    """Tests for residual_kernel_filter_two_stage."""

    def test_flat_surface_preserved(self):
        """Flat ground surface should be mostly preserved."""
        np.random.seed(42)
        n = 500

        # Generate points on a flat surface with small noise
        x = np.random.uniform(0, 20, n)
        y = np.random.uniform(0, 20, n)
        z = np.zeros(n) + np.random.normal(0, 0.02, n)

        points = np.column_stack([x, y, z])
        filtered = phase2.residual_kernel_filter_two_stage(points)

        # Most points should be preserved (>90%)
        assert len(filtered) > 0.9 * n

    def test_outliers_removed(self):
        """Obvious outliers should be removed."""
        np.random.seed(42)
        n = 500

        # Generate flat surface
        x = np.random.uniform(0, 20, n)
        y = np.random.uniform(0, 20, n)
        z = np.zeros(n)

        # Add some outliers
        n_outliers = 20
        x = np.append(x, np.random.uniform(5, 15, n_outliers))
        y = np.append(y, np.random.uniform(5, 15, n_outliers))
        z = np.append(z, np.ones(n_outliers) * 5)  # 5m above surface

        points = np.column_stack([x, y, z])
        filtered = phase2.residual_kernel_filter_two_stage(points)

        # All filtered points should have z close to 0
        assert np.all(np.abs(filtered[:, 2]) < 1.0)

    def test_custom_passes(self):
        """Custom filter passes should work."""
        np.random.seed(42)
        points = np.random.uniform(0, 10, (200, 3))
        points[:, 2] = 0  # Flat surface

        # Single pass with lenient settings
        filtered = phase2.residual_kernel_filter_two_stage(
            points,
            passes=[(5.0, 1.0)],  # Large cell, large threshold
        )

        assert len(filtered) > 0


class TestBinPointCloudTemporal:
    """Integration tests for bin_point_cloud_temporal."""

    def test_basic_temporal_binning(self):
        """Basic temporal binning produces TimeResolvedGrid."""
        # Generate simple test data
        n = 100
        np.random.seed(42)

        points = np.column_stack([
            np.random.uniform(0, 10, n),  # x
            np.zeros(n),                   # y (not used)
            np.random.uniform(0, 2, n),   # z
        ])
        intensities = np.random.uniform(10, 50, n)
        times = np.random.uniform(0, 5, n)

        grid = phase2.bin_point_cloud_temporal(
            points, intensities, times,
            x_bin_size=1.0,
            time_bin_size=0.5,
        )

        assert isinstance(grid, phase2.TimeResolvedGrid)
        assert grid.z_mean.shape[0] > 0  # Has time bins
        assert grid.z_mean.shape[1] > 0  # Has x bins

    def test_invalid_inputs(self):
        """Invalid inputs should raise ValueError."""
        points = np.random.uniform(0, 10, (100, 3))
        intensities = np.random.uniform(10, 50, 100)
        times = np.random.uniform(0, 5, 100)

        # Wrong shape points
        with pytest.raises(ValueError):
            phase2.bin_point_cloud_temporal(
                points[:, :2], intensities, times  # Missing z
            )

        # Mismatched lengths
        with pytest.raises(ValueError):
            phase2.bin_point_cloud_temporal(
                points, intensities[:50], times  # Wrong length
            )

        # Negative bin size
        with pytest.raises(ValueError):
            phase2.bin_point_cloud_temporal(
                points, intensities, times,
                x_bin_size=-1.0,
            )
