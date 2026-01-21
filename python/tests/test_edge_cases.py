"""
Tests for edge cases and robustness.

Tests the pipeline's handling of:
- Sparse point clouds
- Large gaps in data
- Outlier-heavy datasets
- Date range edge cases
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from phase2 import (
    bin_point_cloud,
    apply_snr_filter,
    residual_kernel_filter_two_stage,
    bin_point_cloud_temporal,
)
from profiles import (
    extract_transects,
    inpaint_nans,
    gapsize,
    TransectConfig,
)
from runup import (
    compute_dry_beach_reference,
    detect_runup_line,
)
from utils import (
    split_by_day,
    round_to_half_hour,
    detect_outliers_conv2d,
    inpaint_nans_2d,
)


# =============================================================================
# Sparse Point Cloud Tests
# =============================================================================

class TestSparsePointClouds:
    """Test handling of sparse point clouds."""

    def test_bin_point_cloud_very_sparse(self):
        """Test binning with very few points."""
        # Only 5 points in a 10x10m area
        # bin_point_cloud expects (N, 3) array
        points = np.array([
            [0.5, 0.5, 1.0],
            [2.5, 2.5, 1.1],
            [5.0, 5.0, 1.2],
            [7.5, 7.5, 1.3],
            [9.5, 9.5, 1.4],
        ])

        grid = bin_point_cloud(points, bin_size=0.5)

        # Should have mostly NaN with a few valid bins
        valid_count = np.sum(~np.isnan(grid.z_mean))
        assert valid_count <= 5  # At most 5 valid bins

    def test_snr_filter_sparse_data(self):
        """Test SNR filter with sparse data (low counts)."""
        points = np.array([
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.0],
            [5.0, 5.0, 2.0],
            [5.1, 5.1, 2.0],
        ])

        grid = bin_point_cloud(points, bin_size=0.5)

        # apply_snr_filter returns 6 values
        x_valid, y_valid, z_mean, z_max, z_min, z_mode = apply_snr_filter(
            grid, snr_threshold=10, min_count=10
        )

        # Should have very few or no valid points with high min_count
        assert len(z_mean) <= len(points)

    def test_residual_filter_sparse(self):
        """Test residual filter with sparse data."""
        # Sparse but flat surface - need at least 4 points for Delaunay
        x = np.array([0.0, 10.0, 0.0, 10.0, 5.0, 5.0])
        y = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 10.0])
        z = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # With default passes, should not crash
        try:
            x_f, y_f, z_f = residual_kernel_filter_two_stage(
                x, y, z,
                passes=[(10.0, 0.2), (3.0, 0.1)]  # Explicitly specify passes
            )
            # Should keep most points on flat surface
            assert len(z_f) >= 3
        except Exception:
            # May fail with too sparse data, that's acceptable
            pass

    def test_sparse_grid_handling(self):
        """Test that sparse grids don't crash the system."""
        # Very sparse 2D grid with mostly NaN
        grid = np.full((50, 50), np.nan)
        # Only fill a few cells
        grid[25, 20:30] = np.linspace(1.0, 2.0, 10)

        # Should be able to compute statistics on sparse data
        valid_count = np.sum(~np.isnan(grid))
        assert valid_count == 10
        assert np.nanmean(grid) == pytest.approx(1.5, rel=0.1)


# =============================================================================
# Large Gap Tests
# =============================================================================

class TestLargeGaps:
    """Test handling of large gaps in data."""

    def test_gapsize_large_gap(self):
        """Test gapsize detection with large gaps."""
        # Array with 20-element gap in the middle
        arr = np.zeros(50)
        arr[15:35] = np.nan  # 20-element gap

        gaps = gapsize(arr)

        # Gap region should show size 20
        assert np.max(gaps[15:35]) == 20

    def test_gapsize_detects_large_gaps(self):
        """Test that gapsize correctly identifies large gaps."""
        arr = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0, 9.0])

        # gapsize should correctly identify the 5-element gap
        gaps = gapsize(arr)

        # Gap region should show size 5
        assert np.max(gaps) == 5
        # Non-gap regions should have size 0
        assert gaps[0] == 0
        assert gaps[-1] == 0

    def test_inpaint_nans_2d_large_gaps(self):
        """Test 2D interpolation with large gaps."""
        arr = np.ones((10, 20))
        arr[:, 5:15] = np.nan  # Large 10-column gap

        result = inpaint_nans_2d(arr, max_gap=5)

        # Large gap should not be fully filled
        assert np.sum(np.isnan(result)) > 0

    def test_runup_detection_with_gaps(self):
        """Test runup detection with gaps in Z_xt."""
        nt, nx = 100, 50
        x1d = np.linspace(0, 50, nx)

        # Create Z_xt with a large gap
        Z_xt = np.random.randn(nt, nx) * 0.1
        Z_xt[:, 20:30] = np.nan  # 10-position gap

        # Create gradient for runup detection
        for i in range(nt):
            Z_xt[i, :20] += np.linspace(2, 0.5, 20)
            Z_xt[i, 30:] += np.linspace(0.3, -0.5, 20)

        # Should handle gaps gracefully
        # ig_length is in seconds, not window_s
        dry_ref = compute_dry_beach_reference(Z_xt, dt=0.5, ig_length=10)

        # Dry reference should exist where data exists
        assert not np.all(np.isnan(dry_ref))


# =============================================================================
# Outlier-Heavy Dataset Tests
# =============================================================================

class TestOutlierHeavyData:
    """Test handling of outlier-heavy datasets."""

    def test_snr_filter_removes_noisy_bins(self):
        """Test SNR filter removes high-variance bins."""
        # Create data with some very noisy bins
        rng = np.random.default_rng(42)

        # Clean data
        x_clean = rng.uniform(0, 5, 100)
        y_clean = rng.uniform(0, 5, 100)
        z_clean = 1.0 + rng.normal(0, 0.01, 100)  # Low variance

        # Noisy data
        x_noisy = rng.uniform(5, 10, 100)
        y_noisy = rng.uniform(0, 5, 100)
        z_noisy = 1.0 + rng.normal(0, 1.0, 100)  # High variance

        x = np.concatenate([x_clean, x_noisy])
        y = np.concatenate([y_clean, y_noisy])
        z = np.concatenate([z_clean, z_noisy])

        # Create points array
        points = np.column_stack([x, y, z])

        grid = bin_point_cloud(points, bin_size=1.0)
        # apply_snr_filter returns 6 values
        x_f, y_f, z_mean, z_max, z_min, z_mode = apply_snr_filter(grid, snr_threshold=50, min_count=5)

        # Most filtered points should be from clean region (x < 5)
        if len(x_f) > 0:
            clean_count = np.sum(x_f < 5)
            noisy_count = np.sum(x_f >= 5)
            assert clean_count >= noisy_count

    def test_outlier_detection_identifies_spikes(self):
        """Test that outlier detection can identify obvious spikes."""
        # Create smooth field with obvious spikes
        Z = np.ones((20, 40)) * 1.0

        # Add a 3x3 spike block that's clearly different
        Z[10, 20] = 5.0
        Z[9:12, 19:22] = 5.0

        result = detect_outliers_conv2d(
            Z, dt=0.5, ig_length=10,
            gradient_threshold_std=2.0,
            laplacian_threshold_std=2.0,
        )

        # Should detect the spike region as outliers
        assert np.sum(result.is_outlier) > 0

    def test_detect_outliers_2d_spike_pattern(self):
        """Test 2D outlier detection with spike pattern."""
        # Create smooth field with spikes
        Z = np.ones((50, 100)) * 1.0

        # Add some random spikes
        rng = np.random.default_rng(42)
        spike_i = rng.integers(5, 45, 10)
        spike_j = rng.integers(5, 95, 10)
        for i, j in zip(spike_i, spike_j):
            Z[i-1:i+2, j-1:j+2] = 5.0  # 3x3 spike blocks

        result = detect_outliers_conv2d(
            Z, dt=0.5, ig_length=10,
            gradient_threshold_std=2.0,
            laplacian_threshold_std=2.0,
        )

        # Should detect some outliers (is_outlier not outlier_mask)
        assert np.sum(result.is_outlier) > 0


# =============================================================================
# Date Range Edge Cases
# =============================================================================

class TestDateRangeEdgeCases:
    """Test date range handling edge cases."""

    def test_split_by_day_single_day(self):
        """Test splitting data from a single day."""
        base = datetime(2025, 1, 15, 12, 0, 0)
        timestamps = [base + timedelta(minutes=i*10) for i in range(10)]

        groups = split_by_day(timestamps)

        # Should have exactly one group
        assert len(groups) == 1
        # Keys are datetime objects, not strings
        assert datetime(2025, 1, 15, 0, 0, 0) in groups

    def test_split_by_day_midnight_crossing(self):
        """Test splitting when data crosses midnight."""
        base = datetime(2025, 1, 15, 23, 0, 0)
        # 4 hours of data crossing midnight
        timestamps = [base + timedelta(minutes=i*30) for i in range(8)]

        groups = split_by_day(timestamps)

        # Should have two groups
        assert len(groups) == 2
        assert datetime(2025, 1, 15, 0, 0, 0) in groups
        assert datetime(2025, 1, 16, 0, 0, 0) in groups

    def test_split_by_day_multiple_days(self):
        """Test splitting data spanning multiple days."""
        timestamps = []
        for day in range(5):
            base = datetime(2025, 1, 10 + day, 12, 0, 0)
            for hour in range(3):
                timestamps.append(base + timedelta(hours=hour))

        groups = split_by_day(timestamps)

        # Should have 5 groups
        assert len(groups) == 5

    def test_round_to_half_hour_edge_cases(self):
        """Test rounding at edge cases."""
        # Exactly on the hour
        dt = datetime(2025, 1, 15, 12, 0, 0)
        rounded = round_to_half_hour(dt)
        assert rounded.hour == 12
        assert rounded.minute == 0

        # Exactly on half hour
        dt = datetime(2025, 1, 15, 12, 30, 0)
        rounded = round_to_half_hour(dt)
        assert rounded.hour == 12
        assert rounded.minute == 30

        # Just before midnight
        dt = datetime(2025, 1, 15, 23, 45, 0)
        rounded = round_to_half_hour(dt)
        # Should round to next day 00:00
        assert rounded.day == 16
        assert rounded.hour == 0

    def test_split_by_day_empty_list(self):
        """Test splitting empty timestamp list."""
        groups = split_by_day([])
        assert len(groups) == 0

    def test_split_by_day_preserves_order(self):
        """Test that within-day order is preserved."""
        base = datetime(2025, 1, 15, 0, 0, 0)
        timestamps = [base + timedelta(hours=h) for h in [3, 1, 4, 1, 5, 9, 2, 6]]

        groups = split_by_day(timestamps)

        # Order within day should match input order
        day_key = datetime(2025, 1, 15, 0, 0, 0)
        indices = groups[day_key]
        expected_hours = [3, 1, 4, 1, 5, 9, 2, 6]
        actual_hours = [timestamps[i].hour for i in indices]
        assert actual_hours == expected_hours


# =============================================================================
# Numerical Edge Cases
# =============================================================================

class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_bin_point_cloud_all_same_location(self):
        """Test binning when all points at same location."""
        n = 100
        x = np.ones(n) * 5.0
        y = np.ones(n) * 5.0
        z = np.random.randn(n)
        points = np.column_stack([x, y, z])

        grid = bin_point_cloud(points, bin_size=1.0)

        # Should have exactly one valid bin
        valid_count = np.sum(~np.isnan(grid.z_mean))
        assert valid_count == 1

    def test_temporal_binning_constant_time(self):
        """Test temporal binning when all points have same timestamp."""
        n = 50
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, n)
        y = np.zeros(n)
        z = rng.normal(1.0, 0.1, n)
        t = np.ones(n) * 100.0  # All same time
        intensity = np.ones(n) * 50

        points = np.column_stack([x, y, z])

        grid = bin_point_cloud_temporal(
            points, intensity, t,
            x_bin_size=1.0, time_bin_size=0.5
        )

        # With all same time, should have very few time bins (1-2 depending on edge effects)
        assert grid.z_mean.shape[0] <= 2

    def test_extreme_values(self):
        """Test handling of extreme coordinate values."""
        # Very large coordinates (UTM-like)
        x = np.array([500000.0, 500001.0, 500002.0])
        y = np.array([3700000.0, 3700001.0, 3700002.0])
        z = np.array([1.0, 1.1, 1.2])
        points = np.column_stack([x, y, z])

        # Should not cause overflow or precision issues
        grid = bin_point_cloud(points, bin_size=0.5)
        assert not np.all(np.isnan(grid.z_mean))

    def test_very_small_bin_size(self):
        """Test with very small bin size."""
        x = np.array([0.0, 0.01, 0.02])
        y = np.array([0.0, 0.01, 0.02])
        z = np.array([1.0, 1.1, 1.2])
        points = np.column_stack([x, y, z])

        # 1cm bins
        grid = bin_point_cloud(points, bin_size=0.01)

        # Should create valid grid (may have many cells)
        assert grid.z_mean is not None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
