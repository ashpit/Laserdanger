"""Tests for utility functions module."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import utils  # noqa: E402


# =============================================================================
# 2D Outlier Detection Tests
# =============================================================================

def test_detect_outliers_returns_result():
    """Test that outlier detection returns OutlierResult."""
    n_x, n_t = 50, 100
    Z_xt = np.random.randn(n_x, n_t) * 0.1 + 1.0
    dt = 0.5

    result = utils.detect_outliers_conv2d(Z_xt, dt)

    assert isinstance(result, utils.OutlierResult)
    assert result.is_outlier.shape == Z_xt.shape
    assert result.Z_filtered.shape == Z_xt.shape


def test_detect_outliers_finds_spikes():
    """Test that outlier detection finds obvious spikes."""
    n_x, n_t = 50, 100
    np.random.seed(42)
    Z_xt = np.random.randn(n_x, n_t) * 0.05 + 1.0

    # Add obvious spike outliers
    Z_xt[25, 50] = 5.0  # Big spike
    Z_xt[30, 60] = -3.0  # Big dip

    dt = 0.5
    result = utils.detect_outliers_conv2d(Z_xt, dt, gradient_threshold_std=2.0)

    # Spikes should be detected (possibly with dilation)
    # At minimum, Z_filtered should have those positions as NaN or different
    assert result.is_outlier.any()


def test_detect_outliers_preserves_good_data():
    """Test that smooth data is mostly preserved."""
    n_x, n_t = 50, 100
    # Create smooth data
    x = np.linspace(0, 10, n_x)
    t = np.linspace(0, 50, n_t)
    X, T = np.meshgrid(x, t, indexing='ij')
    Z_xt = np.sin(0.5 * X) + np.cos(0.1 * T)

    dt = 0.5
    result = utils.detect_outliers_conv2d(Z_xt, dt)

    # Most data should not be marked as outlier
    outlier_fraction = result.is_outlier.sum() / result.is_outlier.size
    assert outlier_fraction < 0.1  # Less than 10% outliers


def test_detect_outliers_handles_nans():
    """Test that existing NaNs are handled."""
    n_x, n_t = 50, 100
    Z_xt = np.random.randn(n_x, n_t) * 0.1 + 1.0
    Z_xt[10:15, 20:25] = np.nan

    dt = 0.5
    result = utils.detect_outliers_conv2d(Z_xt, dt)

    # NaN region should be marked as outlier
    assert np.all(result.is_outlier[10:15, 20:25])


# =============================================================================
# Intensity Contour Tests
# =============================================================================

def test_get_intensity_contours_single_threshold():
    """Test intensity contour extraction with single threshold."""
    n_x, n_t = 50, 100
    x1d = np.arange(n_x) * 0.1

    # Create intensity that increases seaward (higher x)
    I_xt = np.tile(x1d, (n_t, 1)).T * 10  # Shape (n_x, n_t)

    contours = utils.get_intensity_contours(I_xt, x1d, thresholds=25.0)

    assert len(contours) == 1
    assert contours[0].threshold == 25.0

    # Contour should be around x=2.5 (where intensity=25)
    valid_x = contours[0].x_positions[~np.isnan(contours[0].x_positions)]
    if len(valid_x) > 0:
        assert np.mean(valid_x) == pytest.approx(2.5, abs=0.5)


def test_get_intensity_contours_multiple_thresholds():
    """Test intensity contour extraction with multiple thresholds."""
    n_x, n_t = 50, 100
    x1d = np.arange(n_x) * 0.1

    I_xt = np.tile(x1d, (n_t, 1)).T * 10

    contours = utils.get_intensity_contours(I_xt, x1d, thresholds=[20.0, 30.0, 40.0])

    # Should get contours for thresholds that have valid crossings
    assert len(contours) > 0
    # Each contour should have correct threshold
    for c in contours:
        assert c.threshold in [20.0, 30.0, 40.0]


def test_get_intensity_contours_stats():
    """Test that contour statistics are computed."""
    n_x, n_t = 50, 200
    x1d = np.arange(n_x) * 0.1

    # Add some variation to the contour position
    np.random.seed(42)
    base_intensity = np.tile(x1d, (n_t, 1)).T * 10
    I_xt = base_intensity + np.random.randn(n_x, n_t) * 2

    contours = utils.get_intensity_contours(I_xt, x1d, thresholds=25.0)

    if len(contours) > 0:
        c = contours[0]
        assert hasattr(c, 'mean_position')
        assert hasattr(c, 'std_position')
        assert hasattr(c, 'valid_fraction')
        assert 0 <= c.valid_fraction <= 1


# =============================================================================
# Time Rounding Tests
# =============================================================================

def test_round_to_half_hour_exact():
    """Test rounding when already at half hour."""
    dt = datetime(2024, 6, 15, 14, 30, 0)
    result = utils.round_to_half_hour(dt)
    assert result == datetime(2024, 6, 15, 14, 30, 0)


def test_round_to_half_hour_round_down():
    """Test rounding down to :00."""
    dt = datetime(2024, 6, 15, 14, 10, 30)
    result = utils.round_to_half_hour(dt)
    assert result == datetime(2024, 6, 15, 14, 0, 0)


def test_round_to_half_hour_round_to_30():
    """Test rounding to :30."""
    dt = datetime(2024, 6, 15, 14, 23, 45)
    result = utils.round_to_half_hour(dt)
    assert result == datetime(2024, 6, 15, 14, 30, 0)


def test_round_to_half_hour_round_up():
    """Test rounding up to next hour."""
    dt = datetime(2024, 6, 15, 14, 50, 0)
    result = utils.round_to_half_hour(dt)
    assert result == datetime(2024, 6, 15, 15, 0, 0)


def test_round_to_half_hour_midnight_edge():
    """Test rounding near midnight."""
    dt = datetime(2024, 6, 15, 23, 50, 0)
    result = utils.round_to_half_hour(dt)
    assert result == datetime(2024, 6, 16, 0, 0, 0)


def test_round_to_interval_15min():
    """Test rounding to 15-minute interval."""
    dt = datetime(2024, 6, 15, 14, 23, 0)
    result = utils.round_to_interval(dt, 15)
    # 23 minutes rounds to 30 (2*15)
    assert result.minute in [15, 30]


def test_round_to_interval_60min():
    """Test rounding to 60-minute interval."""
    dt = datetime(2024, 6, 15, 14, 40, 0)
    result = utils.round_to_interval(dt, 60)
    assert result == datetime(2024, 6, 15, 15, 0, 0)


# =============================================================================
# Day Splitting Tests
# =============================================================================

def test_split_by_day_single_day():
    """Test splitting timestamps from single day."""
    timestamps = [
        datetime(2024, 6, 15, 10, 0, 0),
        datetime(2024, 6, 15, 12, 0, 0),
        datetime(2024, 6, 15, 14, 0, 0),
    ]
    data = ['a', 'b', 'c']

    result = utils.split_by_day(timestamps, data)

    assert len(result) == 1
    day_key = datetime(2024, 6, 15, 0, 0, 0)
    assert day_key in result
    assert result[day_key] == ['a', 'b', 'c']


def test_split_by_day_multiple_days():
    """Test splitting timestamps across multiple days."""
    timestamps = [
        datetime(2024, 6, 15, 10, 0, 0),
        datetime(2024, 6, 15, 14, 0, 0),
        datetime(2024, 6, 16, 8, 0, 0),
        datetime(2024, 6, 16, 12, 0, 0),
        datetime(2024, 6, 17, 9, 0, 0),
    ]
    data = [1, 2, 3, 4, 5]

    result = utils.split_by_day(timestamps, data)

    assert len(result) == 3
    assert result[datetime(2024, 6, 15)] == [1, 2]
    assert result[datetime(2024, 6, 16)] == [3, 4]
    assert result[datetime(2024, 6, 17)] == [5]


def test_split_by_day_returns_indices():
    """Test splitting returns indices when data is None."""
    timestamps = [
        datetime(2024, 6, 15, 10, 0, 0),
        datetime(2024, 6, 15, 14, 0, 0),
        datetime(2024, 6, 16, 8, 0, 0),
    ]

    result = utils.split_by_day(timestamps)

    assert result[datetime(2024, 6, 15)] == [0, 1]
    assert result[datetime(2024, 6, 16)] == [2]


def test_deduplicate_by_interval():
    """Test deduplication by time interval."""
    timestamps = [
        datetime(2024, 6, 15, 10, 5, 0),
        datetime(2024, 6, 15, 10, 10, 0),  # Same 30-min bucket as above
        datetime(2024, 6, 15, 10, 40, 0),  # Different bucket
    ]
    data = ['a', 'b', 'c']

    ts_unique, data_unique = utils.deduplicate_by_interval(timestamps, data, 30)

    # First two should merge, third stays
    assert len(ts_unique) == 2
    assert data_unique[0] == 'a'  # First occurrence kept
    assert data_unique[1] == 'c'


# =============================================================================
# Gap Size Tests
# =============================================================================

def test_gapsize_no_gaps():
    """Test gapsize with no NaN values."""
    x = np.array([1, 2, 3, 4, 5])
    sz = utils.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 0, 0, 0, 0])


def test_gapsize_single_gap():
    """Test gapsize with single gap."""
    x = np.array([1, np.nan, np.nan, np.nan, 5])
    sz = utils.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 3, 3, 3, 0])


def test_gapsize_multiple_gaps():
    """Test gapsize with multiple gaps."""
    x = np.array([1, np.nan, 3, np.nan, np.nan, 6])
    sz = utils.gapsize(x)
    np.testing.assert_array_equal(sz, [0, 1, 0, 2, 2, 0])


def test_gapsize_all_nan():
    """Test gapsize with all NaN."""
    x = np.array([np.nan, np.nan, np.nan])
    sz = utils.gapsize(x)
    np.testing.assert_array_equal(sz, [3, 3, 3])


# =============================================================================
# Inpaint NaN Tests
# =============================================================================

def test_inpaint_nans_1d_small_gap():
    """Test 1D interpolation of small gap."""
    x = np.array([0, 1, 2, 3, 4, 5])
    z = np.array([0, np.nan, np.nan, 3, 4, 5], dtype=float)

    z_filled = utils.inpaint_nans_1d(x, z, max_gap=3.0)

    # Gap of 2 units should be filled
    assert not np.isnan(z_filled[1])
    assert not np.isnan(z_filled[2])
    np.testing.assert_allclose(z_filled[1], 1.0)
    np.testing.assert_allclose(z_filled[2], 2.0)


def test_inpaint_nans_1d_large_gap_not_filled():
    """Test that large gaps are not filled."""
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10], dtype=float)

    z_filled = utils.inpaint_nans_1d(x, z, max_gap=4.0)

    # Gap of 9 units should NOT be filled
    assert np.isnan(z_filled[5])


def test_inpaint_nans_2d_along_rows():
    """Test 2D interpolation along rows."""
    Z = np.array([
        [1, 2, np.nan, 4, 5],
        [10, 20, 30, 40, 50],
    ], dtype=float)

    Z_filled = utils.inpaint_nans_2d(Z, max_gap=2, axis=1)

    # Row 0, column 2 should be interpolated
    assert not np.isnan(Z_filled[0, 2])
    np.testing.assert_allclose(Z_filled[0, 2], 3.0)


def test_inpaint_nans_2d_preserves_valid():
    """Test that valid values are preserved."""
    Z = np.array([
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
    ], dtype=float)

    Z_filled = utils.inpaint_nans_2d(Z, max_gap=2)

    np.testing.assert_array_equal(Z, Z_filled)


# =============================================================================
# Helper Function Tests
# =============================================================================

def test_interp_small_gaps_1d():
    """Test internal small gap interpolation."""
    arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0])

    result = utils._interp_small_gaps_1d(arr, max_gap=3)

    assert not np.isnan(result[1])
    assert not np.isnan(result[2])
    np.testing.assert_allclose(result[1], 2.0)
    np.testing.assert_allclose(result[2], 3.0)


def test_interp_small_gaps_1d_large_gap():
    """Test that large gaps are not interpolated."""
    arr = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])

    result = utils._interp_small_gaps_1d(arr, max_gap=2)

    # Gap of 4 should not be filled
    assert np.isnan(result[2])
    assert np.isnan(result[3])
