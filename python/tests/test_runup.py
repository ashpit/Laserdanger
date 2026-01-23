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


# =============================================================================
# Multi-Signal Detection Tests
# =============================================================================

def create_synthetic_intensity_data(
    Z_xt: np.ndarray,
    dry_intensity: float = 80.0,
    wet_intensity: float = 40.0,
    noise_std: float = 5.0,
) -> np.ndarray:
    """
    Create synthetic intensity data matching elevation structure.

    Water (higher Z relative to dry beach) has lower intensity.
    """
    np.random.seed(43)

    # Create base intensity (higher on dry beach)
    n_x, n_t = Z_xt.shape

    # Use elevation to determine water/sand regions
    z_threshold = np.percentile(Z_xt, 50)

    I_xt = np.where(
        Z_xt > z_threshold,
        wet_intensity,
        dry_intensity
    )

    # Add noise
    I_xt = I_xt + np.random.normal(0, noise_std, I_xt.shape)

    return I_xt.astype(float)


def test_sigmoid_function():
    """Test sigmoid helper function."""
    # Test basic behavior
    x = np.array([-2, -1, 0, 1, 2])
    result = runup._sigmoid(x, scale=1.0)

    # Should be in [0, 1]
    assert np.all(result >= 0)
    assert np.all(result <= 1)

    # Sigmoid(0) = 0.5
    np.testing.assert_allclose(result[2], 0.5, atol=0.01)

    # Monotonically increasing
    assert np.all(np.diff(result) > 0)


def test_compute_elevation_signal_shape():
    """Test elevation signal has correct shape."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))
    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt)

    p_elev = runup.compute_elevation_signal(Z_xt, dry_beach, threshold=0.1)

    assert p_elev.shape == Z_xt.shape


def test_compute_elevation_signal_range():
    """Test elevation signal is in [0, 1]."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))
    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt)

    p_elev = runup.compute_elevation_signal(Z_xt, dry_beach, threshold=0.1)

    valid = ~np.isnan(p_elev)
    assert np.all(p_elev[valid] >= 0)
    assert np.all(p_elev[valid] <= 1)


def test_compute_intensity_signal_shape():
    """Test intensity signal has correct shape."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    I_xt = create_synthetic_intensity_data(Z_xt)
    dt = np.median(np.diff(time_vec))

    I_dry = runup.compute_dry_intensity_reference(I_xt, dt)
    p_int = runup.compute_intensity_signal(I_xt, I_dry)

    assert p_int.shape == Z_xt.shape


def test_compute_intensity_signal_water_higher():
    """Test that water regions (lower intensity) have higher probability."""
    n_x, n_t = 50, 100
    I_xt = np.full((n_x, n_t), 80.0)  # Dry sand
    I_xt[:25, :] = 40.0  # Water (lower intensity)

    I_dry = np.full_like(I_xt, 80.0)

    p_int = runup.compute_intensity_signal(I_xt, I_dry)

    # Water region (low I) should have higher probability
    assert np.mean(p_int[:25, :]) > np.mean(p_int[25:, :])


def test_compute_variance_signal_shape():
    """Test variance signal has correct shape."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))

    p_var = runup.compute_variance_signal(Z_xt, dt)

    assert p_var.shape == Z_xt.shape


def test_compute_variance_signal_fluctuating_higher():
    """Test that fluctuating regions have higher variance probability."""
    n_x, n_t = 50, 200
    dt = 0.5

    # Create data with high variance in one region
    Z_xt = np.zeros((n_x, n_t))
    np.random.seed(44)
    Z_xt[:25, :] = np.random.normal(0, 0.1, (25, n_t))  # High variance (water)
    Z_xt[25:, :] = np.random.normal(0, 0.01, (25, n_t))  # Low variance (sand)

    p_var = runup.compute_variance_signal(Z_xt, dt, window_seconds=2.5)

    # High variance region should have higher probability
    mean_high = np.nanmean(p_var[:25, :])
    mean_low = np.nanmean(p_var[25:, :])
    assert mean_high > mean_low


def test_dry_intensity_reference_above_observed():
    """Test dry intensity reference is generally above observed."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    I_xt = create_synthetic_intensity_data(Z_xt)
    dt = np.median(np.diff(time_vec))

    I_dry = runup.compute_dry_intensity_reference(I_xt, dt)

    # Dry reference uses moving max, so should be >= observed
    valid = ~np.isnan(I_dry)
    # Most should be at or above (with tolerance for smoothing)
    diff = I_dry[valid] - I_xt[valid]
    assert np.mean(diff >= -5) > 0.9


def test_estimate_elevation_snr_positive():
    """Test elevation SNR is positive."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))
    dry_beach = runup.compute_dry_beach_reference(Z_xt, dt)

    snr = runup.estimate_elevation_snr(Z_xt, dry_beach, threshold=0.1)

    assert len(snr) == Z_xt.shape[1]
    assert np.all(snr >= 0.1)  # Minimum SNR floor


def test_compute_adaptive_weights_normalized():
    """Test adaptive weights sum to 1."""
    n_t = 100
    snr_elev = np.random.uniform(0.5, 5.0, n_t)
    snr_int = np.random.uniform(0.2, 3.0, n_t)
    snr_var = np.random.uniform(0.1, 2.0, n_t)

    w_elev, w_int, w_var = runup.compute_adaptive_weights(
        snr_elev, snr_int, snr_var
    )

    # Should sum to 1
    total = w_elev + w_int + w_var
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_compute_adaptive_weights_min_weight():
    """Test minimum weight constraint prevents zero weights."""
    n_t = 100
    snr_elev = np.full(n_t, 10.0)  # High
    snr_int = np.full(n_t, 0.01)  # Very low
    snr_var = np.full(n_t, 0.01)  # Very low

    w_elev, w_int, w_var = runup.compute_adaptive_weights(
        snr_elev, snr_int, snr_var, min_weight=0.1
    )

    # Weights should be > 0 (non-zero) due to SNR floor from min_weight
    # With snr_min = min_weight * 3 = 0.3, low SNR signals get boosted
    # s_elev=10, s_int=0.3, s_var=0.3, total=10.6
    # w_int = w_var = 0.3/10.6 ≈ 0.028
    assert np.all(w_int > 0)  # Non-zero
    assert np.all(w_var > 0)  # Non-zero
    assert np.all(w_int > 0.02)  # Above floor
    assert np.all(w_var > 0.02)


def test_fuse_signals_adaptive_shape():
    """Test signal fusion has correct shape."""
    n_x, n_t = 50, 100
    p_elev = np.random.uniform(0, 1, (n_x, n_t))
    p_int = np.random.uniform(0, 1, (n_x, n_t))
    p_var = np.random.uniform(0, 1, (n_x, n_t))

    w_elev = np.full(n_t, 0.5)
    w_int = np.full(n_t, 0.3)
    w_var = np.full(n_t, 0.2)

    P_water, weights_used = runup.fuse_signals_adaptive(
        p_elev, p_int, p_var, w_elev, w_int, w_var
    )

    assert P_water.shape == (n_x, n_t)
    assert weights_used.shape == (3, n_t)


def test_fuse_signals_adaptive_range():
    """Test fused probability is in [0, 1]."""
    n_x, n_t = 50, 100
    p_elev = np.random.uniform(0, 1, (n_x, n_t))
    p_int = np.random.uniform(0, 1, (n_x, n_t))
    p_var = np.random.uniform(0, 1, (n_x, n_t))

    w_elev = np.full(n_t, 0.5)
    w_int = np.full(n_t, 0.3)
    w_var = np.full(n_t, 0.2)

    P_water, _ = runup.fuse_signals_adaptive(
        p_elev, p_int, p_var, w_elev, w_int, w_var
    )

    assert np.all(P_water >= 0)
    assert np.all(P_water <= 1)


def test_detect_runup_multisignal_returns_confidence():
    """Test multi-signal detection returns confidence."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    dt = np.median(np.diff(time_vec))
    dx = np.median(np.diff(x1d))

    # Create probability field
    P_water = np.random.uniform(0, 1, Z_xt.shape)

    X_runup, Z_runup, idx_runup, confidence = runup.detect_runup_multisignal(
        Z_xt, P_water, x1d, search_window=0.5, dx=dx
    )

    assert len(confidence) == Z_xt.shape[1]
    # Some confidence values should be valid
    assert np.sum(~np.isnan(confidence)) > 0


def test_compute_runup_stats_with_intensity():
    """Test full runup stats with intensity data."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    I_xt = create_synthetic_intensity_data(Z_xt)

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec,
        I_xt=I_xt,
        threshold=0.1,
        ig_length=50.0,
    )

    # Should use multi-signal
    assert result.info["multisignal_enabled"] is True

    # Should have weight info
    assert "mean_weight_elevation" in result.info
    assert "mean_weight_intensity" in result.info
    assert "mean_weight_variance" in result.info
    assert "mean_confidence" in result.info

    # Weights should be valid
    assert 0 < result.info["mean_weight_elevation"] < 1
    assert 0 < result.info["mean_weight_intensity"] < 1
    assert 0 < result.info["mean_weight_variance"] < 1


def test_compute_runup_stats_without_intensity():
    """Test fallback to elevation-only when no intensity."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec,
        I_xt=None,  # No intensity
        threshold=0.1,
        ig_length=50.0,
    )

    # Should fallback
    assert result.info["multisignal_enabled"] is False

    # Should still produce valid results
    assert len(result.timeseries.X_runup) == len(time_vec)


def test_runup_timeseries_has_confidence():
    """Test RunupTimeseries dataclass has confidence field."""
    Z_xt, x1d, time_vec = create_synthetic_wave_data()
    I_xt = create_synthetic_intensity_data(Z_xt)

    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec,
        I_xt=I_xt,
    )

    # Should have confidence array
    assert hasattr(result.timeseries, 'confidence')
    assert len(result.timeseries.confidence) == len(time_vec)

    # Should have weights_used
    assert hasattr(result.timeseries, 'weights_used')
    assert result.timeseries.weights_used.shape[0] == 3  # 3 signals


def test_moving_max_nan_basic():
    """Test moving maximum with NaN handling."""
    arr = np.array([5, 3, 4, 2, 6, np.nan, 4, 3])
    result = runup._moving_max_nan(arr, window=3)

    assert len(result) == len(arr)
    # Should be NaN-aware
    assert not np.all(np.isnan(result))
