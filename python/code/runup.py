"""
Runup detection and spectral analysis module.
Implements algorithms from MATLAB get_runupStats_L2.m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


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
    """Runup time series."""
    time: np.ndarray  # Time in seconds
    X_runup: np.ndarray  # Cross-shore runup position (m)
    Z_runup: np.ndarray  # Runup elevation (m)
    idx_runup: np.ndarray  # Index of runup position


@dataclass
class RunupResult:
    """Complete runup analysis result."""
    spectrum: RunupSpectrum
    bulk: RunupBulk
    timeseries: RunupTimeseries
    info: dict  # Processing metadata


def compute_runup_stats(
    Z_xt: np.ndarray,
    x1d: np.ndarray,
    time_vec: np.ndarray,
    I_xt: Optional[np.ndarray] = None,
    threshold: float = 0.1,
    ig_length: float = 100.0,
    search_window: float = 0.5,
    window_length_minutes: float = 5.0,
) -> RunupResult:
    """
    Compute runup statistics from time-resolved elevation matrix.

    Matches MATLAB get_runupStats_L2.m algorithm:
    1. Create dry beach reference using moving minimum filter
    2. Detect runup line by threshold crossing
    3. Track runup position with adaptive search window
    4. Compute spectral statistics

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix with cross-shore in rows, time in columns
    x1d : array (n_x,)
        Cross-shore positions (m)
    time_vec : array (n_t,)
        Time vector (seconds)
    I_xt : array (n_x, n_t), optional
        Intensity matrix (not currently used but kept for compatibility)
    threshold : float
        Water depth threshold for runup detection (default 0.1m)
    ig_length : float
        Moving minimum window for dry beach reference (default 100s)
    search_window : float
        Adaptive search window size (default ±0.5m)
    window_length_minutes : float
        Window length for spectral analysis (default 5 min)

    Returns
    -------
    RunupResult
        Complete runup analysis results
    """
    n_x, n_t = Z_xt.shape
    dt = np.median(np.diff(time_vec))
    dx = np.median(np.diff(x1d))

    # Compute dry beach reference surface using moving minimum
    dry_beach = compute_dry_beach_reference(Z_xt, dt, ig_length)

    # Detect runup line
    X_runup, Z_runup, idx_runup = detect_runup_line(
        Z_xt, dry_beach, x1d, threshold, search_window, dx
    )

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
    )

    info = {
        "dt": dt,
        "dx": dx,
        "threshold": threshold,
        "ig_length": ig_length,
        "n_valid": np.sum(~np.isnan(Z_runup)),
        "duration_seconds": time_vec[-1] - time_vec[0],
    }

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
    3. Find seaward-most crossing of threshold
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


def smooth_runup_timeseries(
    X_runup: np.ndarray,
    Z_runup: np.ndarray,
    dt: float,
    median_window_seconds: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean up runup time series with median filtering and gap interpolation.

    Parameters
    ----------
    X_runup, Z_runup : arrays
        Raw runup time series
    dt : float
        Time step
    median_window_seconds : float
        Median filter window (default 1s)

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

    # Apply median filter to valid values
    valid = ~np.isnan(X_runup)
    if valid.sum() > kernel_size:
        X_clean = X_runup.copy()
        Z_clean = Z_runup.copy()

        # Median filter
        X_clean[valid] = signal.medfilt(X_runup[valid], kernel_size)
        Z_clean[valid] = signal.medfilt(Z_runup[valid], kernel_size)

        # Fill small gaps (up to 5 samples)
        X_clean = _interp_small_gaps(X_clean, max_gap=5)
        Z_clean = _interp_small_gaps(Z_clean, max_gap=5)

        return X_clean, Z_clean

    return X_runup, Z_runup


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

def _moving_min_nan(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving minimum with NaN handling."""
    n = len(arr)
    result = np.full(n, np.nan)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            result[i] = np.min(valid)

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
    Find the seaward-most (largest index) threshold crossing.

    Returns -1 if no crossing found.
    """
    # Search from seaward to landward (high to low index)
    for i in range(search_hi, search_lo, -1):
        if i < len(water_level) - 1:
            # Check for crossing from below threshold to above
            if water_level[i] <= threshold < water_level[i + 1]:
                return i
            # Or from above to below
            if water_level[i] >= threshold > water_level[i + 1]:
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
