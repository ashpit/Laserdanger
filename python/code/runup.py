"""
Runup detection and spectral analysis module.

Multi-signal approach combining:
1. Elevation anomaly (Z above dry beach reference)
2. Intensity drop (water has lower lidar intensity than dry sand)
3. Temporal variance (water surface fluctuates, sand is stable)

Uses adaptive SNR-based weighting to fuse signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d, minimum_filter1d, maximum_filter1d

logger = logging.getLogger(__name__)


@dataclass
class RunupSpectrum:
    """Frequency spectrum of runup time series."""
    frequency: np.ndarray  # Hz
    power: np.ndarray  # m^2/Hz
    power_lower: np.ndarray  # Lower confidence bound
    power_upper: np.ndarray  # Upper confidence bound
    dof: int  # Degrees of freedom


@dataclass
class RunupBulk:
    """Bulk runup statistics."""
    Sig: float  # Significant wave height, IG band (m)
    Sinc: float  # Significant wave height, incident band (m)
    eta: float  # Mean water level (m)
    beta: float  # Beach slope (dimensionless)
    R2: float  # 2% exceedance runup (m)


@dataclass
class RunupTimeseries:
    """Runup time series with detection confidence."""
    time: np.ndarray  # Time in seconds
    X_runup: np.ndarray  # Cross-shore runup position (m)
    Z_runup: np.ndarray  # Runup elevation (m)
    idx_runup: np.ndarray  # Index of runup position
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))  # Fused probability at detected position
    weights_used: np.ndarray = field(default_factory=lambda: np.array([]))  # (3, n_t) adaptive weights [elev, int, var]


@dataclass
class RunupResult:
    """Complete runup analysis result."""
    spectrum: RunupSpectrum
    bulk: RunupBulk
    timeseries: RunupTimeseries
    info: dict  # Processing metadata


# =============================================================================
# Sigmoid and utility functions for signal conversion
# =============================================================================

def _sigmoid(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Sigmoid function for converting signals to [0, 1] probability.

    sigmoid(x) = 1 / (1 + exp(-x/scale))

    Parameters
    ----------
    x : array
        Input signal
    scale : float
        Scaling factor (controls sharpness of transition)

    Returns
    -------
    array
        Values in [0, 1]
    """
    # Clip to prevent overflow
    x_scaled = np.clip(x / scale, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_scaled))


# =============================================================================
# Signal computation functions
# =============================================================================

def compute_elevation_signal(
    Z_xt: np.ndarray,
    dry_beach: np.ndarray,
    threshold: float = 0.1,
    scale: float = 0.05,
) -> np.ndarray:
    """
    Compute water probability from elevation anomaly.

    P = sigmoid((Z - dry_beach - threshold) / scale)
    Higher elevation above dry beach = higher water probability.

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    dry_beach : array (n_x, n_t)
        Dry beach reference surface
    threshold : float
        Minimum water depth to consider (default 0.1m)
    scale : float
        Sigmoid scale parameter (default 0.05m)

    Returns
    -------
    array (n_x, n_t)
        Water probability from elevation signal [0, 1]
    """
    water_depth = Z_xt - dry_beach - threshold
    return _sigmoid(water_depth, scale)


def compute_intensity_signal(
    I_xt: np.ndarray,
    I_dry_ref: np.ndarray,
    scale: float = 20.0,
) -> np.ndarray:
    """
    Compute water probability from intensity drop.

    Water typically has lower lidar intensity than dry sand.
    P = sigmoid((I_dry_ref - I) / scale)
    Larger intensity drop = higher water probability.

    Parameters
    ----------
    I_xt : array (n_x, n_t)
        Intensity matrix
    I_dry_ref : array (n_x, n_t)
        Dry sand intensity reference
    scale : float
        Sigmoid scale parameter (default 20.0 intensity units)

    Returns
    -------
    array (n_x, n_t)
        Water probability from intensity signal [0, 1]
    """
    intensity_drop = I_dry_ref - I_xt
    return _sigmoid(intensity_drop, scale)


def compute_variance_signal(
    Z_xt: np.ndarray,
    dt: float,
    window_seconds: float = 2.5,
    var_threshold: float = 0.001,
    scale: float = 0.002,
) -> np.ndarray:
    """
    Compute water probability from local temporal variance.

    Water surface fluctuates, sand is stable.
    P = sigmoid((variance - var_threshold) / scale)
    Higher variance = higher water probability.

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    dt : float
        Time step in seconds
    window_seconds : float
        Variance window (default 2.5s = ~5 samples at 2Hz)
    var_threshold : float
        Baseline variance threshold (default 0.001 m^2)
    scale : float
        Sigmoid scale parameter (default 0.002 m^2)

    Returns
    -------
    array (n_x, n_t)
        Water probability from variance signal [0, 1]
    """
    n_x, n_t = Z_xt.shape
    window_samples = max(3, int(window_seconds / dt))
    if window_samples % 2 == 0:
        window_samples += 1

    # Compute local variance along time axis for each position
    variance = np.full_like(Z_xt, np.nan)

    for i in range(n_x):
        row = Z_xt[i, :]
        valid = ~np.isnan(row)
        if valid.sum() > window_samples:
            # Rolling variance using uniform filter trick:
            # var(x) = E[x^2] - E[x]^2
            row_filled = np.where(valid, row, 0.0)
            mean_x = uniform_filter1d(row_filled, window_samples, mode='nearest')
            mean_x2 = uniform_filter1d(row_filled ** 2, window_samples, mode='nearest')
            local_var = mean_x2 - mean_x ** 2
            local_var = np.maximum(local_var, 0)  # Numerical stability
            variance[i, valid] = local_var[valid]

    return _sigmoid(variance - var_threshold, scale)


def compute_dry_intensity_reference(
    I_xt: np.ndarray,
    dt: float,
    window_seconds: float = 100.0,
) -> np.ndarray:
    """
    Compute dry sand intensity reference using moving maximum filter.

    Analogous to dry beach reference for elevation, but uses maximum
    since dry sand has higher intensity than water.

    Parameters
    ----------
    I_xt : array (n_x, n_t)
        Intensity matrix
    dt : float
        Time step in seconds
    window_seconds : float
        Window length for moving maximum (default 100s)

    Returns
    -------
    array (n_x, n_t)
        Dry sand intensity reference
    """
    n_x, n_t = I_xt.shape

    # Window size in samples
    window = int(window_seconds / dt)
    if window < 1:
        window = 1
    if window % 2 == 0:
        window += 1

    # Moving maximum along time axis
    I_ref = np.full_like(I_xt, np.nan)
    for i in range(n_x):
        row = I_xt[i, :]
        if np.all(np.isnan(row)):
            continue
        I_ref[i, :] = _moving_max_nan(row, window)

    # Fill small gaps
    I_ref = _fill_gaps_linear(I_ref, max_gap=6)

    # Smooth
    smooth_samples = int(50 / dt)
    if smooth_samples < 1:
        smooth_samples = 1
    if smooth_samples % 2 == 0:
        smooth_samples += 1

    for i in range(n_x):
        valid = ~np.isnan(I_ref[i, :])
        if valid.sum() > smooth_samples:
            I_ref[i, valid] = uniform_filter1d(I_ref[i, valid], smooth_samples, mode='nearest')

    return I_ref


# =============================================================================
# SNR estimation and adaptive weighting
# =============================================================================

def estimate_elevation_snr(
    Z_xt: np.ndarray,
    dry_beach: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Estimate SNR for elevation signal.

    SNR = contrast / noise, where:
    - contrast = max(Z - dry_beach) - threshold
    - noise = std of Z in dry region

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
    dry_beach : array (n_x, n_t)
    threshold : float

    Returns
    -------
    array (n_t,)
        Per-timestep SNR estimate
    """
    n_x, n_t = Z_xt.shape
    water_depth = Z_xt - dry_beach

    snr = np.zeros(n_t)
    for t in range(n_t):
        col = water_depth[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            snr[t] = 0.1  # Low SNR for insufficient data
            continue

        # Contrast: max water depth - threshold
        contrast = np.nanmax(col) - threshold

        # Noise: std of values below threshold (dry region)
        dry_mask = col < threshold
        if dry_mask.sum() > 2:
            noise = np.nanstd(col[dry_mask])
        else:
            noise = np.nanstd(col) * 0.5  # Fallback estimate

        if noise < 1e-6:
            noise = 1e-6

        snr[t] = max(0.1, abs(contrast) / noise)

    return snr


def estimate_intensity_snr(
    I_xt: np.ndarray,
    I_dry_ref: np.ndarray,
    p_elevation: np.ndarray,
) -> np.ndarray:
    """
    Estimate SNR for intensity signal.

    Uses elevation probability to identify likely water vs sand regions.

    Parameters
    ----------
    I_xt : array (n_x, n_t)
    I_dry_ref : array (n_x, n_t)
    p_elevation : array (n_x, n_t)
        Elevation probability (used to identify water regions)

    Returns
    -------
    array (n_t,)
        Per-timestep SNR estimate
    """
    n_x, n_t = I_xt.shape
    intensity_drop = I_dry_ref - I_xt

    snr = np.zeros(n_t)
    for t in range(n_t):
        col = intensity_drop[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            snr[t] = 0.1
            continue

        # Use elevation probability to identify regions
        p_col = p_elevation[:, t]
        water_mask = p_col > 0.5
        sand_mask = p_col < 0.3

        if water_mask.sum() > 2 and sand_mask.sum() > 2:
            water_drop = np.nanmean(col[water_mask])
            sand_drop = np.nanmean(col[sand_mask])
            contrast = water_drop - sand_drop
            noise = np.nanstd(col[sand_mask])
        else:
            # Fallback: use overall statistics
            contrast = np.nanmax(col) - np.nanmedian(col)
            noise = np.nanstd(col) * 0.5

        if noise < 1e-6:
            noise = 1e-6

        snr[t] = max(0.1, abs(contrast) / noise)

    return snr


def estimate_variance_snr(
    p_variance: np.ndarray,
    p_elevation: np.ndarray,
) -> np.ndarray:
    """
    Estimate SNR for variance signal.

    Ratio of variance in water region vs sand region.

    Parameters
    ----------
    p_variance : array (n_x, n_t)
    p_elevation : array (n_x, n_t)

    Returns
    -------
    array (n_t,)
        Per-timestep SNR estimate
    """
    n_x, n_t = p_variance.shape

    snr = np.zeros(n_t)
    for t in range(n_t):
        p_var_col = p_variance[:, t]
        p_elev_col = p_elevation[:, t]

        valid = ~np.isnan(p_var_col)
        if valid.sum() < 5:
            snr[t] = 0.1
            continue

        water_mask = p_elev_col > 0.5
        sand_mask = p_elev_col < 0.3

        if water_mask.sum() > 2 and sand_mask.sum() > 2:
            water_var_prob = np.nanmean(p_var_col[water_mask])
            sand_var_prob = np.nanmean(p_var_col[sand_mask])
            # Contrast in probability space
            contrast = water_var_prob - sand_var_prob
            noise = np.nanstd(p_var_col[sand_mask])
        else:
            contrast = np.nanmax(p_var_col) - np.nanmedian(p_var_col)
            noise = np.nanstd(p_var_col) * 0.5

        if noise < 1e-6:
            noise = 1e-6

        snr[t] = max(0.1, abs(contrast) / noise)

    return snr


def compute_adaptive_weights(
    snr_elevation: np.ndarray,
    snr_intensity: np.ndarray,
    snr_variance: np.ndarray,
    min_weight: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert SNR estimates to normalized adaptive weights.

    w_i(t) = max(snr_i(t), snr_min) / sum(max(snr_j(t), snr_min))

    Parameters
    ----------
    snr_elevation, snr_intensity, snr_variance : arrays (n_t,)
    min_weight : float
        Minimum weight for any signal (default 0.1)

    Returns
    -------
    w_elevation, w_intensity, w_variance : arrays (n_t,)
        Normalized weights summing to 1.0 at each timestep
    """
    n_t = len(snr_elevation)

    # Apply minimum SNR floor
    snr_min = min_weight * 3  # Ensures min weight when normalized
    s_elev = np.maximum(snr_elevation, snr_min)
    s_int = np.maximum(snr_intensity, snr_min)
    s_var = np.maximum(snr_variance, snr_min)

    # Normalize to weights
    total = s_elev + s_int + s_var
    w_elev = s_elev / total
    w_int = s_int / total
    w_var = s_var / total

    return w_elev, w_int, w_var


def fuse_signals_adaptive(
    p_elevation: np.ndarray,
    p_intensity: np.ndarray,
    p_variance: np.ndarray,
    w_elevation: np.ndarray,
    w_intensity: np.ndarray,
    w_variance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse signals with adaptive per-timestep weights.

    P_water(x, t) = w_elev(t) * p_elev(x,t) + w_int(t) * p_int(x,t) + w_var(t) * p_var(x,t)

    Parameters
    ----------
    p_elevation, p_intensity, p_variance : arrays (n_x, n_t)
        Signal probabilities
    w_elevation, w_intensity, w_variance : arrays (n_t,)
        Adaptive weights

    Returns
    -------
    P_water : array (n_x, n_t)
        Fused probability
    weights_used : array (3, n_t)
        Weights for diagnostics [elevation, intensity, variance]
    """
    n_x, n_t = p_elevation.shape

    # Broadcast weights for vectorized computation
    # weights are (n_t,), probabilities are (n_x, n_t)
    P_water = (
        w_elevation[np.newaxis, :] * p_elevation +
        w_intensity[np.newaxis, :] * p_intensity +
        w_variance[np.newaxis, :] * p_variance
    )

    weights_used = np.stack([w_elevation, w_intensity, w_variance], axis=0)

    return P_water, weights_used


def compute_runup_stats(
    Z_xt: np.ndarray,
    x1d: np.ndarray,
    time_vec: np.ndarray,
    I_xt: Optional[np.ndarray] = None,
    threshold: float = 0.1,
    ig_length: float = 100.0,
    search_window: float = 0.5,
    window_length_minutes: float = 5.0,
    # Multi-signal parameters
    variance_window_seconds: float = 2.5,
    intensity_scale: float = 20.0,
    elevation_scale: float = 0.05,
    variance_scale: float = 0.002,
    min_weight: float = 0.1,
) -> RunupResult:
    """
    Compute runup statistics from time-resolved elevation matrix.

    Multi-signal approach combining:
    1. Elevation anomaly (Z above dry beach reference)
    2. Intensity drop (water has lower lidar intensity)
    3. Temporal variance (water surface fluctuates)

    When I_xt is provided, uses adaptive SNR-based weighting to fuse all three
    signals. When I_xt is None, falls back to elevation-only detection.

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix with cross-shore in rows, time in columns
    x1d : array (n_x,)
        Cross-shore positions (m)
    time_vec : array (n_t,)
        Time vector (seconds)
    I_xt : array (n_x, n_t), optional
        Intensity matrix. If provided, enables multi-signal detection.
    threshold : float
        Water depth threshold for runup detection (default 0.1m)
    ig_length : float
        Moving minimum/maximum window for reference surfaces (default 100s)
    search_window : float
        Adaptive search window size (default ±0.5m)
    window_length_minutes : float
        Window length for spectral analysis (default 5 min)
    variance_window_seconds : float
        Window for temporal variance calculation (default 2.5s)
    intensity_scale : float
        Sigmoid scale for intensity signal (default 20.0)
    elevation_scale : float
        Sigmoid scale for elevation signal (default 0.05m)
    variance_scale : float
        Sigmoid scale for variance signal (default 0.002 m^2)
    min_weight : float
        Minimum weight for any signal in adaptive fusion (default 0.1)

    Returns
    -------
    RunupResult
        Complete runup analysis results including confidence and weights
    """
    n_x, n_t = Z_xt.shape
    dt = np.median(np.diff(time_vec))
    dx = np.median(np.diff(x1d))

    # Compute dry beach reference surface using moving minimum
    dry_beach = compute_dry_beach_reference(Z_xt, dt, ig_length)

    # Multi-signal detection if intensity available
    use_multisignal = I_xt is not None and I_xt.shape == Z_xt.shape

    if use_multisignal:
        logger.info("Using multi-signal runup detection (elevation + intensity + variance)")

        # Compute all three signals
        p_elevation = compute_elevation_signal(Z_xt, dry_beach, threshold, elevation_scale)

        I_dry_ref = compute_dry_intensity_reference(I_xt, dt, ig_length)
        p_intensity = compute_intensity_signal(I_xt, I_dry_ref, intensity_scale)

        p_variance = compute_variance_signal(Z_xt, dt, variance_window_seconds,
                                              var_threshold=0.001, scale=variance_scale)

        # Estimate SNR for each signal
        snr_elevation = estimate_elevation_snr(Z_xt, dry_beach, threshold)
        snr_intensity = estimate_intensity_snr(I_xt, I_dry_ref, p_elevation)
        snr_variance = estimate_variance_snr(p_variance, p_elevation)

        # Compute adaptive weights
        w_elev, w_int, w_var = compute_adaptive_weights(
            snr_elevation, snr_intensity, snr_variance, min_weight
        )

        # Fuse signals
        P_water, weights_used = fuse_signals_adaptive(
            p_elevation, p_intensity, p_variance,
            w_elev, w_int, w_var
        )

        # Detect runup using fused probability
        X_runup, Z_runup, idx_runup, confidence = detect_runup_multisignal(
            Z_xt, P_water, x1d, search_window, dx
        )

        # Log mean weights for diagnostics
        logger.debug(f"Mean weights: elev={np.mean(w_elev):.2f}, "
                     f"int={np.mean(w_int):.2f}, var={np.mean(w_var):.2f}")

    else:
        if I_xt is None:
            logger.info("No intensity data provided - using elevation-only detection")
        else:
            logger.warning("Intensity shape mismatch - falling back to elevation-only")

        # Fallback to elevation-only detection
        X_runup, Z_runup, idx_runup = detect_runup_line(
            Z_xt, dry_beach, x1d, threshold, search_window, dx
        )
        confidence = np.full(n_t, np.nan)
        weights_used = np.full((3, n_t), np.nan)

    # Clean up runup time series
    X_runup, Z_runup = smooth_runup_timeseries(X_runup, Z_runup, dt)

    # Compute spectral statistics
    spectrum = compute_runup_spectrum(Z_runup, dt, window_length_minutes)

    # Compute bulk statistics
    bulk = compute_bulk_stats(Z_runup, spectrum, Z_xt, x1d, X_runup)

    # Create timeseries output
    timeseries = RunupTimeseries(
        time=time_vec,
        X_runup=X_runup,
        Z_runup=Z_runup,
        idx_runup=idx_runup,
        confidence=confidence,
        weights_used=weights_used,
    )

    info = {
        "dt": dt,
        "dx": dx,
        "threshold": threshold,
        "ig_length": ig_length,
        "n_valid": np.sum(~np.isnan(Z_runup)),
        "duration_seconds": time_vec[-1] - time_vec[0],
        "multisignal_enabled": use_multisignal,
        "variance_window_seconds": variance_window_seconds,
        "intensity_scale": intensity_scale,
        "elevation_scale": elevation_scale,
        "min_weight": min_weight,
    }

    if use_multisignal:
        info["mean_weight_elevation"] = float(np.mean(w_elev))
        info["mean_weight_intensity"] = float(np.mean(w_int))
        info["mean_weight_variance"] = float(np.mean(w_var))
        info["mean_confidence"] = float(np.nanmean(confidence))

    return RunupResult(
        spectrum=spectrum,
        bulk=bulk,
        timeseries=timeseries,
        info=info,
    )


def compute_dry_beach_reference(
    Z_xt: np.ndarray,
    dt: float,
    ig_length: float = 100.0,
) -> np.ndarray:
    """
    Compute dry beach reference surface using moving minimum filter.

    Matches MATLAB:
        M = movmin(Z_xt', IGfilt, 1, 'omitnan')'
        M = fillmissing(M, 'linear', 1, 'MaxGap', 6)
        M = movmean(M, smooth_samples, 2)

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    dt : float
        Time step in seconds
    ig_length : float
        Window length for moving minimum (default 100s)

    Returns
    -------
    array (n_x, n_t)
        Dry beach reference surface
    """
    n_x, n_t = Z_xt.shape

    # Window size in samples
    ig_filt = int(ig_length / dt)
    if ig_filt < 1:
        ig_filt = 1
    if ig_filt % 2 == 0:
        ig_filt += 1  # Make odd for symmetric window

    # Moving minimum along time axis (axis=1)
    M = np.full_like(Z_xt, np.nan)
    for i in range(n_x):
        row = Z_xt[i, :]
        if np.all(np.isnan(row)):
            continue
        # Use generic_filter for moving minimum with NaN handling
        M[i, :] = _moving_min_nan(row, ig_filt)

    # Fill small gaps with linear interpolation (max 6 samples)
    M = _fill_gaps_linear(M, max_gap=6)

    # Smooth with moving mean (50 samples / dt smoothing)
    smooth_samples = int(50 / dt)
    if smooth_samples < 1:
        smooth_samples = 1
    if smooth_samples % 2 == 0:
        smooth_samples += 1

    for i in range(n_x):
        valid = ~np.isnan(M[i, :])
        if valid.sum() > smooth_samples:
            M[i, valid] = uniform_filter1d(M[i, valid], smooth_samples, mode='nearest')

    return M


def detect_runup_line(
    Z_xt: np.ndarray,
    dry_beach: np.ndarray,
    x1d: np.ndarray,
    threshold: float,
    search_window: float,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect runup line by finding threshold crossing.

    For each time step:
    1. Compute water level = Z - dry_beach
    2. Apply median filter for stability
    3. Find landward-most crossing of threshold (maximum inland water extent)
    4. Use adaptive search window around previous position

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
    dry_beach : array (n_x, n_t)
    x1d : array (n_x,)
    threshold : float
    search_window : float (meters)
    dx : float

    Returns
    -------
    X_runup, Z_runup, idx_runup : arrays (n_t,)
    """
    n_x, n_t = Z_xt.shape
    msize = int(search_window / dx)  # Search window in grid points
    if msize < 1:
        msize = 1

    X_runup = np.full(n_t, np.nan)
    Z_runup = np.full(n_t, np.nan)
    idx_runup = np.full(n_t, -1, dtype=int)

    prev_idx = n_x // 2  # Start in middle

    for ii in range(n_t):
        # Get smoothed dry beach surface
        sand = signal.medfilt(dry_beach[:, ii], kernel_size=5)

        # Compute water level with smoothing
        # Use median over 3 time steps if available
        if ii >= 1 and ii < n_t - 1:
            z_stack = np.column_stack([
                Z_xt[:, ii-1] if ii > 0 else Z_xt[:, ii],
                Z_xt[:, ii],
                Z_xt[:, ii+1] if ii < n_t - 1 else Z_xt[:, ii],
            ])
            z_median = np.nanmedian(z_stack, axis=1)
        else:
            z_median = Z_xt[:, ii]

        water_level = uniform_filter1d(z_median - sand, 5, mode='nearest')

        # Define search window
        if ii > 4:
            # Adaptive: search around previous position
            search_lo = max(0, prev_idx - msize)
            search_hi = min(n_x - 1, prev_idx + msize)
        else:
            # Initial: search full range
            search_lo = 0
            search_hi = n_x - 1

        # Find threshold crossing in search window
        crossing_idx = _find_threshold_crossing(
            water_level, threshold, search_lo, search_hi
        )

        if crossing_idx >= 0:
            # Interpolate exact crossing position
            if crossing_idx > 0 and crossing_idx < n_x - 1:
                # Linear interpolation between grid points
                wl0 = water_level[crossing_idx]
                wl1 = water_level[crossing_idx + 1]
                if abs(wl1 - wl0) > 1e-10:
                    frac = (threshold - wl0) / (wl1 - wl0)
                    frac = np.clip(frac, 0, 1)
                    X_runup[ii] = x1d[crossing_idx] + frac * dx
                else:
                    X_runup[ii] = x1d[crossing_idx]
            else:
                X_runup[ii] = x1d[crossing_idx]

            Z_runup[ii] = Z_xt[crossing_idx, ii]
            idx_runup[ii] = crossing_idx
            prev_idx = crossing_idx
        else:
            # Expand search if no crossing found
            crossing_idx = _find_threshold_crossing(water_level, threshold, 0, n_x - 1)
            if crossing_idx >= 0:
                X_runup[ii] = x1d[crossing_idx]
                Z_runup[ii] = Z_xt[crossing_idx, ii]
                idx_runup[ii] = crossing_idx
                prev_idx = crossing_idx

    return X_runup, Z_runup, idx_runup


def detect_runup_multisignal(
    Z_xt: np.ndarray,
    P_water: np.ndarray,
    x1d: np.ndarray,
    search_window: float,
    dx: float,
    prob_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect runup line using fused water probability.

    The runup line is the landward-most position where water probability
    exceeds the threshold - i.e., where water meets dry beach.

    For each time step:
    1. Search from seaward to landward (following x gradient)
    2. Find where P_water drops below threshold (water → dry transition)
    3. Return confidence = P_water at detected position

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    P_water : array (n_x, n_t)
        Fused water probability [0, 1]
    x1d : array (n_x,)
        Cross-shore positions
    search_window : float
        Adaptive search window (meters)
    dx : float
        Grid spacing
    prob_threshold : float
        Probability threshold for water detection (default 0.5)

    Returns
    -------
    X_runup, Z_runup, idx_runup, confidence : arrays (n_t,)
        Runup position, elevation, index, and confidence
    """
    n_x, n_t = Z_xt.shape
    msize = max(1, int(search_window / dx))

    X_runup = np.full(n_t, np.nan)
    Z_runup = np.full(n_t, np.nan)
    idx_runup = np.full(n_t, -1, dtype=int)
    confidence = np.full(n_t, np.nan)

    # Determine search direction based on coordinate system
    # Find which end has higher probability (water) on average
    start_slice = P_water[:n_x//4, :]
    end_slice = P_water[-n_x//4:, :]

    # Handle all-NaN slices gracefully
    start_valid = ~np.isnan(start_slice)
    end_valid = ~np.isnan(end_slice)

    if start_valid.any():
        mean_prob_start = np.nanmean(start_slice)
    else:
        mean_prob_start = 0.0

    if end_valid.any():
        mean_prob_end = np.nanmean(end_slice)
    else:
        mean_prob_end = 0.0

    # Search from water (high prob) toward land (low prob)
    search_seaward_to_landward = mean_prob_start > mean_prob_end

    prev_idx = n_x // 2  # Start in middle

    for ii in range(n_t):
        p_col = P_water[:, ii]

        # Apply small smoothing to probability
        valid = ~np.isnan(p_col)
        if valid.sum() < 5:
            continue

        p_smooth = p_col.copy()
        p_smooth[valid] = uniform_filter1d(p_col[valid], 3, mode='nearest')

        # Define search window
        if ii > 4:
            search_lo = max(0, prev_idx - msize)
            search_hi = min(n_x - 1, prev_idx + msize)
        else:
            search_lo = 0
            search_hi = n_x - 1

        crossing_idx = -1

        if search_seaward_to_landward:
            # Seaward is at low indices, landward at high indices
            # Search from low to high, find where P drops below threshold
            for i in range(search_lo, search_hi):
                if i < n_x - 1 and not np.isnan(p_smooth[i]) and not np.isnan(p_smooth[i + 1]):
                    # Transition from water (high P) to dry (low P)
                    if p_smooth[i] >= prob_threshold > p_smooth[i + 1]:
                        crossing_idx = i
                        break
        else:
            # Seaward is at high indices, landward at low indices
            # Search from high to low, find where P drops below threshold
            for i in range(search_hi, search_lo, -1):
                if i > 0 and not np.isnan(p_smooth[i]) and not np.isnan(p_smooth[i - 1]):
                    # Transition from water (high P) to dry (low P)
                    if p_smooth[i] >= prob_threshold > p_smooth[i - 1]:
                        crossing_idx = i
                        break

        # Expand search if no crossing found in window
        if crossing_idx < 0:
            if search_seaward_to_landward:
                for i in range(0, n_x - 1):
                    if not np.isnan(p_smooth[i]) and not np.isnan(p_smooth[i + 1]):
                        if p_smooth[i] >= prob_threshold > p_smooth[i + 1]:
                            crossing_idx = i
                            break
            else:
                for i in range(n_x - 1, 0, -1):
                    if not np.isnan(p_smooth[i]) and not np.isnan(p_smooth[i - 1]):
                        if p_smooth[i] >= prob_threshold > p_smooth[i - 1]:
                            crossing_idx = i
                            break

        if crossing_idx >= 0:
            # Interpolate exact crossing position
            if search_seaward_to_landward:
                i0, i1 = crossing_idx, crossing_idx + 1
            else:
                i0, i1 = crossing_idx, crossing_idx - 1

            if 0 <= i0 < n_x and 0 <= i1 < n_x:
                p0 = p_smooth[i0]
                p1 = p_smooth[i1]
                if abs(p1 - p0) > 1e-10:
                    frac = (prob_threshold - p0) / (p1 - p0)
                    frac = np.clip(frac, 0, 1)
                    X_runup[ii] = x1d[i0] + frac * (x1d[i1] - x1d[i0])
                else:
                    X_runup[ii] = x1d[crossing_idx]
            else:
                X_runup[ii] = x1d[crossing_idx]

            Z_runup[ii] = Z_xt[crossing_idx, ii]
            idx_runup[ii] = crossing_idx
            confidence[ii] = p_smooth[crossing_idx]
            prev_idx = crossing_idx

    return X_runup, Z_runup, idx_runup, confidence


def smooth_runup_timeseries(
    X_runup: np.ndarray,
    Z_runup: np.ndarray,
    dt: float,
    median_window_seconds: float = 1.0,
    outlier_std_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean up runup time series with outlier removal, median filtering, and gap interpolation.

    Parameters
    ----------
    X_runup, Z_runup : arrays
        Raw runup time series
    dt : float
        Time step
    median_window_seconds : float
        Median filter window (default 1s)
    outlier_std_threshold : float
        Remove points more than this many std from rolling median (default 3.0)

    Returns
    -------
    X_runup, Z_runup : cleaned arrays
    """
    n_t = len(X_runup)
    kernel_size = int(median_window_seconds / dt)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    X_clean = X_runup.copy()
    Z_clean = Z_runup.copy()

    valid = ~np.isnan(X_clean)
    if valid.sum() < kernel_size:
        return X_runup, Z_runup

    # Step 1: Remove outliers based on deviation from rolling median
    # Use a larger window for outlier detection
    outlier_kernel = min(kernel_size * 5, valid.sum() // 2)
    if outlier_kernel < 3:
        outlier_kernel = 3
    if outlier_kernel % 2 == 0:
        outlier_kernel += 1

    if valid.sum() > outlier_kernel:
        # Compute rolling median
        X_valid = X_clean[valid]
        rolling_med = signal.medfilt(X_valid, outlier_kernel)

        # Compute deviation from rolling median
        deviation = np.abs(X_valid - rolling_med)
        dev_std = np.nanstd(deviation)

        if dev_std > 0:
            # Mark outliers
            outlier_mask = deviation > outlier_std_threshold * dev_std
            if outlier_mask.any():
                # Set outliers to NaN
                valid_indices = np.where(valid)[0]
                for i, is_outlier in enumerate(outlier_mask):
                    if is_outlier:
                        X_clean[valid_indices[i]] = np.nan
                        Z_clean[valid_indices[i]] = np.nan

    # Step 2: Apply median filter to remaining valid values
    valid = ~np.isnan(X_clean)
    if valid.sum() > kernel_size:
        X_clean[valid] = signal.medfilt(X_clean[valid], kernel_size)
        Z_clean[valid] = signal.medfilt(Z_clean[valid], kernel_size)

    # Step 3: Fill small gaps (up to 5 samples)
    X_clean = _interp_small_gaps(X_clean, max_gap=5)
    Z_clean = _interp_small_gaps(Z_clean, max_gap=5)

    return X_clean, Z_clean


def compute_runup_spectrum(
    Z_runup: np.ndarray,
    dt: float,
    window_length_minutes: float = 5.0,
    alpha: float = 0.05,
) -> RunupSpectrum:
    """
    Compute power spectrum of runup elevation time series.

    Parameters
    ----------
    Z_runup : array
        Runup elevation time series
    dt : float
        Time step in seconds
    window_length_minutes : float
        FFT window length (default 5 min)
    alpha : float
        Confidence level (default 0.05 = 95%)

    Returns
    -------
    RunupSpectrum
    """
    # Remove NaN and detrend
    valid = ~np.isnan(Z_runup)
    if valid.sum() < 10:
        # Not enough data
        return RunupSpectrum(
            frequency=np.array([]),
            power=np.array([]),
            power_lower=np.array([]),
            power_upper=np.array([]),
            dof=0,
        )

    z_valid = Z_runup[valid]
    z_detrend = signal.detrend(z_valid)

    # FFT parameters
    fs = 1.0 / dt  # Sampling frequency
    nfft = int(window_length_minutes * 60 / dt)
    if nfft > len(z_detrend):
        nfft = len(z_detrend)

    # Compute power spectrum using Welch's method
    freq, power = signal.welch(
        z_detrend,
        fs=fs,
        nperseg=nfft,
        noverlap=nfft // 2,
        scaling='density',
    )

    # Degrees of freedom (approximate)
    n_segments = max(1, 2 * len(z_detrend) // nfft - 1)
    dof = 2 * n_segments

    # Chi-squared confidence intervals
    from scipy.stats import chi2
    chi2_lo = chi2.ppf(alpha / 2, dof)
    chi2_hi = chi2.ppf(1 - alpha / 2, dof)

    power_lower = dof * power / chi2_hi
    power_upper = dof * power / chi2_lo

    return RunupSpectrum(
        frequency=freq,
        power=power,
        power_lower=power_lower,
        power_upper=power_upper,
        dof=dof,
    )


def compute_bulk_stats(
    Z_runup: np.ndarray,
    spectrum: RunupSpectrum,
    Z_xt: np.ndarray,
    x1d: np.ndarray,
    X_runup: np.ndarray,
) -> RunupBulk:
    """
    Compute bulk runup statistics.

    Parameters
    ----------
    Z_runup : array
        Runup elevation time series
    spectrum : RunupSpectrum
        Computed spectrum
    Z_xt : array (n_x, n_t)
        Full elevation matrix (for slope estimation)
    x1d : array
        Cross-shore positions
    X_runup : array
        Runup position time series

    Returns
    -------
    RunupBulk
    """
    # Frequency bands
    f_ig_lo, f_ig_hi = 0.004, 0.04  # Infragravity
    f_inc_lo, f_inc_hi = 0.04, 0.25  # Incident (sea-swell)

    # Compute significant wave heights from spectrum
    if len(spectrum.frequency) > 0 and len(spectrum.power) > 0:
        df = np.median(np.diff(spectrum.frequency)) if len(spectrum.frequency) > 1 else 1.0

        # IG band
        ig_mask = (spectrum.frequency >= f_ig_lo) & (spectrum.frequency < f_ig_hi)
        Sig = 4 * np.sqrt(np.sum(spectrum.power[ig_mask]) * df) if ig_mask.any() else 0.0

        # Incident band
        inc_mask = (spectrum.frequency >= f_inc_lo) & (spectrum.frequency < f_inc_hi)
        Sinc = 4 * np.sqrt(np.sum(spectrum.power[inc_mask]) * df) if inc_mask.any() else 0.0
    else:
        Sig = 0.0
        Sinc = 0.0

    # Mean water level
    eta = np.nanmean(Z_runup)

    # 2% exceedance
    valid_z = Z_runup[~np.isnan(Z_runup)]
    R2 = np.percentile(valid_z, 98) if len(valid_z) > 0 else np.nan

    # Beach slope estimation
    beta = estimate_beach_slope(Z_xt, x1d, X_runup)

    return RunupBulk(
        Sig=Sig,
        Sinc=Sinc,
        eta=eta,
        beta=beta,
        R2=R2,
    )


def estimate_beach_slope(
    Z_xt: np.ndarray,
    x1d: np.ndarray,
    X_runup: np.ndarray,
    min_fraction: float = 0.1,
    max_fraction: float = 0.9,
) -> float:
    """
    Estimate foreshore beach slope from stable beach regions.

    Uses the active runup range and fits a linear slope to low-variance regions.

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    x1d : array (n_x,)
        Cross-shore positions
    X_runup : array (n_t,)
        Runup position time series
    min_fraction, max_fraction : float
        Use positions between these percentiles of runup range

    Returns
    -------
    float
        Beach slope (dz/dx, dimensionless)
    """
    valid_runup = X_runup[~np.isnan(X_runup)]
    if len(valid_runup) < 10:
        return np.nan

    # Get runup range
    x_min = np.percentile(valid_runup, min_fraction * 100)
    x_max = np.percentile(valid_runup, max_fraction * 100)

    # Find indices in this range
    idx_range = (x1d >= x_min) & (x1d <= x_max)
    if idx_range.sum() < 3:
        return np.nan

    # Use temporal mean of elevation
    z_mean = np.nanmean(Z_xt[idx_range, :], axis=1)
    x_subset = x1d[idx_range]

    # Remove NaN
    valid = ~np.isnan(z_mean)
    if valid.sum() < 3:
        return np.nan

    x_fit = x_subset[valid]
    z_fit = z_mean[valid]

    # Linear fit
    try:
        slope, _ = np.polyfit(x_fit, z_fit, 1)
        return abs(slope)  # Return positive slope
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# =============================================================================
# Helper functions
# =============================================================================

def _moving_max_nan(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Moving maximum with NaN handling.

    Analogous to _moving_min_nan but for maximum.
    Used for dry intensity reference computation.
    """
    n = len(arr)

    if n == 0:
        return np.array([])

    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.full(n, np.nan)

    # Replace NaN with small value so it doesn't affect maximum
    valid_min = np.nanmin(arr)
    fill_value = valid_min - 1e6 if np.isfinite(valid_min) else -1e38
    arr_filled = np.where(nan_mask, fill_value, arr)

    # Apply scipy's fast maximum filter
    result = maximum_filter1d(arr_filled, size=window, mode='nearest')

    # Restore NaN where there were no valid points
    valid_count = uniform_filter1d((~nan_mask).astype(float), size=window, mode='nearest')
    result[valid_count < 0.5 / window] = np.nan

    return result


def _moving_min_nan(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Moving minimum with NaN handling.

    Uses scipy.ndimage.minimum_filter1d for ~50x speedup over pure Python loop.
    NaN values are temporarily replaced with a large value so they don't affect
    the minimum, then restored as NaN in the output where no valid data exists.
    """
    n = len(arr)

    # Handle edge cases
    if n == 0:
        return np.array([])

    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.full(n, np.nan)

    # Replace NaN with large value so it doesn't affect minimum
    # Use a value larger than any valid data point
    valid_max = np.nanmax(arr)
    fill_value = valid_max + 1e6 if np.isfinite(valid_max) else 1e38
    arr_filled = np.where(nan_mask, fill_value, arr)

    # Apply scipy's fast minimum filter (C-optimized)
    result = minimum_filter1d(arr_filled, size=window, mode='nearest')

    # Restore NaN where there were no valid points in the window
    # We need to check if any window was entirely NaN
    # Use a count of valid points in each window
    valid_count = uniform_filter1d((~nan_mask).astype(float), size=window, mode='nearest')

    # Where valid_count is ~0, there were no valid points -> result should be NaN
    # Due to floating point, use small threshold
    result[valid_count < 0.5 / window] = np.nan

    return result


def _fill_gaps_linear(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill small gaps with linear interpolation along rows."""
    result = arr.copy()
    n_rows, n_cols = arr.shape

    for i in range(n_rows):
        row = result[i, :]
        valid = ~np.isnan(row)

        if valid.sum() < 2:
            continue

        # Find gaps
        valid_idx = np.where(valid)[0]
        for j in range(len(valid_idx) - 1):
            start = valid_idx[j]
            end = valid_idx[j + 1]
            gap_size = end - start - 1

            if 0 < gap_size <= max_gap:
                # Linear interpolation
                x_interp = np.arange(start, end + 1)
                row[start:end + 1] = np.interp(
                    x_interp,
                    [start, end],
                    [row[start], row[end]],
                )

        result[i, :] = row

    return result


def _find_threshold_crossing(
    water_level: np.ndarray,
    threshold: float,
    search_lo: int,
    search_hi: int,
) -> int:
    """
    Find the landward-most (smallest index) threshold crossing.

    For runup detection, we want the maximum inland extent of water,
    which is the first dry-to-wet transition when searching from
    landward to seaward.

    Returns -1 if no crossing found.
    """
    # Search from landward to seaward (low to high index)
    # First crossing found = landward-most = runup position
    for i in range(search_lo, search_hi):
        if i < len(water_level) - 1:
            # Check for crossing from below threshold to above (dry to wet)
            if water_level[i] <= threshold < water_level[i + 1]:
                return i

    return -1


def _interp_small_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate small gaps in 1D array."""
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
