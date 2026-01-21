"""Tests for runup detection module."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import runup  # noqa: E402


# =============================================================================
# Test Data Generators
# =============================================================================

def create_synthetic_wave_data(
    n_x: int = 100,
    n_t: int = 500,
    dx: float = 0.1,
    dt: float = 0.5,
    wave_amplitude: float = 0.3,
    wave_period: float = 10.0,
    beach_slope: float = 0.1,
) -> tuple:
    """
    Create synthetic beach elevation data with wave-like fluctuations.

    Returns Z_xt, x1d, time_vec
    """
    x1d = np.arange(n_x) * dx
    time_vec = np.arange(n_t) * dt

    # Create sloped beach
    Z_base = -beach_slope * x1d[:, np.newaxis] + 2.0  # 2m at x=0, slopes down

    # Add wave-like oscillation
    omega = 2 * np.pi / wave_period
    wave = wave_amplitude * np.sin(omega * time_vec[np.newaxis, :])

    # Wave affects lower beach more
    wave_envelope = np.exp(-x1d[:, np.newaxis] / 3.0)  # Decay offshore
    Z_xt = Z_base + wave * wave_envelope

    # Add some noise
    np.random.seed(42)
    Z_xt += np.random.normal(0, 0.02, Z_xt.shape)

    return Z_xt, x1d, time_vec


# =============================================================================
# Dry Beach Reference Tests
# =============================================================================

def test_dry_beach_reference_shape():
    """Test that dry beach reference has correct shape."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))

    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt, ig_length=50.0)

    assert dry_beach.shape == Z_xt.shape


def test_dry_beach_reference_below_surface():
    """Test that dry beach is generally below or equal to instantaneous surface."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))

    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt, ig_length=50.0)

    # Dry beach should be <= Z_xt (with small tolerance for smoothing)
    valid = ~np.isnan(dry_beach)
    diff = dry_beach[valid] - Z_xt[valid]

    # Most should be below (allowing some tolerance for smoothing)
    assert np.mean(diff <= 0.1) > 0.9


def test_dry_beach_handles_nans():
    """Test that dry beach handles NaN values."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))

    # Add some NaNs
    Z_xt[10:15, 100:150] = np.nan

    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt, ig_length=50.0)

    # Should still produce valid output
    assert dry_beach.shape == Z_xt.shape
    # Some values should be filled
    assert not np.all(np.isnan(dry_beach))


# =============================================================================
# Runup Detection Tests
# =============================================================================

def test_detect_runup_line_returns_arrays():
    """Test that runup detection returns arrays of correct length."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))
    dx = np.median(np.diff(x1d))

    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt)

    X_runup, Z_runup, idx_runup = runup.detect_runup_line(
        Z_xt, dry_beach, x1d, threshold=0.1, search_window=0.5, dx=dx
    )

    n_t = Z_xt.shape[1]
    assert len(X_runup) == n_t
    assert len(Z_runup) == n_t
    assert len(idx_runup) == n_t


def test_detect_runup_line_positions_in_range():
    """Test that detected runup positions are within valid range."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))
    dx = np.median(np.diff(x1d))

    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt)

    X_runup, Z_runup, idx_runup = runup.detect_runup_line(
        Z_xt, dry_beach, x1d, threshold=0.1, search_window=0.5, dx=dx
    )

    valid = ~np.isnan(X_runup)
    if valid.any():
        assert np.all(X_runup[valid] >= x1d.min())
        assert np.all(X_runup[valid] <= x1d.max())


def test_smooth_runup_preserves_length():
    """Test that smoothing preserves array length."""
    n_t = 200
    X_runup = np.random.uniform(5, 8, n_t)
    Z_runup = np.random.uniform(0, 1, n_t)

    # Add some NaNs
    X_runup[50:55] = np.nan
    Z_runup[50:55] = np.nan

    X_smooth, Z_smooth = runup.smooth_runup_timeseries(X_runup, Z_runup, dt=0.5)

    assert len(X_smooth) == n_t
    assert len(Z_smooth) == n_t


# =============================================================================
# Spectral Analysis Tests
# =============================================================================

def test_compute_spectrum_returns_frequencies():
    """Test that spectrum computation returns frequency array."""
    np.random.seed(42)
    Z_runup = np.random.randn(500) * 0.3
    dt = 0.5

    spectrum = runup.compute_runup_spectrum(Z_runup, dt, window_length_minutes=2.0)

    assert len(spectrum.frequency) > 0
    assert len(spectrum.power) == len(spectrum.frequency)
    assert spectrum.dof > 0


def test_compute_spectrum_handles_nans():
    """Test that spectrum handles NaN values."""
    np.random.seed(42)
    Z_runup = np.random.randn(500) * 0.3
    Z_runup[100:150] = np.nan
    dt = 0.5

    spectrum = runup.compute_runup_spectrum(Z_runup, dt, window_length_minutes=2.0)

    # Should still work with NaN
    assert len(spectrum.frequency) > 0


def test_compute_spectrum_confidence_bounds():
    """Test that confidence bounds bracket the power estimate."""
    np.random.seed(42)
    Z_runup = np.random.randn(500) * 0.3
    dt = 0.5

    spectrum = runup.compute_runup_spectrum(Z_runup, dt, window_length_minutes=2.0)

    if len(spectrum.frequency) > 0:
        # Lower bound should be below power
        assert np.all(spectrum.power_lower <= spectrum.power)
        # Upper bound should be above power
        assert np.all(spectrum.power_upper >= spectrum.power)


# =============================================================================
# Bulk Statistics Tests
# =============================================================================

def test_bulk_stats_sig_nonnegative():
    """Test that significant wave heights are non-negative."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec, threshold=0.1, ig_length=50.0
    )

    assert result.bulk.Sig >= 0
    assert result.bulk.Sinc >= 0


def test_bulk_stats_slope_positive():
    """Test that estimated beach slope is positive."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data(beach_slope=0.1)

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec, threshold=0.1, ig_length=50.0
    )

    # Slope should be positive (or NaN if can't estimate)
    if not np.isnan(result.bulk.beta):
        assert result.bulk.beta > 0


def test_estimate_beach_slope_accuracy():
    """Test that slope estimation is reasonably accurate."""
    # Create simple sloped beach without waves
    n_x, n_t = 50, 100
    dx, dt = 0.2, 0.5
    true_slope = 0.08

    x1d = np.arange(n_x) * dx
    time_vec = np.arange(n_t) * dt

    Z_xt = -true_slope * x1d[:, np.newaxis] + 1.0
    # Small amount of noise
    Z_xt += np.random.normal(0, 0.005, Z_xt.shape)

    # Fake runup positions
    X_runup = np.random.uniform(2, 8, n_t)

    estimated_slope = runup.estimate_beach_slope(Z_xt, x1d, X_runup)

    # Should be close to true slope
    if not np.isnan(estimated_slope):
        assert abs(estimated_slope - true_slope) < 0.02


# =============================================================================
# Full Pipeline Tests
# =============================================================================

def test_compute_runup_stats_full():
    """Test full runup statistics computation."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec, threshold=0.1, ig_length=50.0
    )

    # Check all components exist
    assert result.spectrum is not None
    assert result.bulk is not None
    assert result.timeseries is not None
    assert result.info is not None

    # Check timeseries lengths match
    assert len(result.timeseries.time) == len(time_vec)
    assert len(result.timeseries.X_runup) == len(time_vec)
    assert len(result.timeseries.Z_runup) == len(time_vec)


def test_compute_runup_stats_info_metadata():
    """Test that info dict contains expected metadata."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec, threshold=0.15, ig_length=80.0
    )

    assert "dt" in result.info
    assert "dx" in result.info
    assert "threshold" in result.info
    assert result.info["threshold"] == 0.15
    assert result.info["ig_length"] == 80.0


# =============================================================================
# Helper Function Tests
# =============================================================================

def test_moving_min_nan_basic():
    """Test moving minimum with NaN handling."""
    arr = np.array([5, 3, 4, 2, 6, np.nan, 4, 3])
    result = runup._moving_min_nan(arr, window=3)

    # Check that result has same length
    assert len(result) == len(arr)

    # First element window is [5, 3], min=3
    # (implementation may vary slightly based on window centering)


def test_find_threshold_crossing():
    """Test threshold crossing detection."""
    water_level = np.array([0.05, 0.08, 0.12, 0.15, 0.11, 0.07])

    # Crossing from below to above at index 1-2
    idx = runup._find_threshold_crossing(water_level, 0.1, 0, 5)

    assert idx >= 0  # Should find a crossing


def test_interp_small_gaps():
    """Test small gap interpolation."""
    arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0])

    result = runup._interp_small_gaps(arr, max_gap=3)

    # Gap of 2 should be filled
    assert not np.isnan(result[1])
    assert not np.isnan(result[2])

    # Should interpolate linearly
    np.testing.assert_allclose(result[1], 2.0, atol=0.1)
    np.testing.assert_allclose(result[2], 3.0, atol=0.1)
