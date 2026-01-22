#!/usr/bin/env python3
"""
Generate runup analysis figures from L2 NetCDF files.

This script:
1. Loads L2 timestack data (Z(x,t), I(x,t))
2. Computes runup statistics using the runup module
3. Generates visualization figures

Usage:
    python scripts/plot_runup.py /path/to/L2_file.nc
    python scripts/plot_runup.py /path/to/level2/  # Process all NC files in directory
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

import runup


def process_runup(nc_path: Path, output_dir: Path, verbose: bool = True):
    """
    Process runup from L2 file and generate figures.

    Parameters
    ----------
    nc_path : Path
        Path to the L2 NetCDF file
    output_dir : Path
        Directory for output figures
    verbose : bool
        Print progress info
    """
    print(f"\n{'='*60}")
    print(f"Processing runup: {nc_path.name}")
    print(f"{'='*60}")

    ds = xr.open_dataset(nc_path)

    # Extract date from filename
    if 'time' in ds.coords:
        data_date = str(ds.time.values[0])[:10].replace('-', '')
    else:
        data_date = nc_path.stem.replace('L2_', '')[:8]

    # Create output directory for this date
    date_dir = output_dir / data_date
    date_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    x1d = ds.x.values
    time_vals = ds.time.values
    Z_xt = ds.elevation.values  # Shape: (n_x, n_t)
    I_xt = ds.intensity.values if 'intensity' in ds else None

    # Get time in seconds from start
    time_seconds = ds.time_seconds.values if 'time_seconds' in ds.coords else None
    if time_seconds is None:
        time_start = time_vals[0]
        time_seconds = (time_vals - time_start).astype('timedelta64[ms]').astype(float) / 1000.0

    dt = float(ds.attrs.get('dt', np.median(np.diff(time_seconds))))
    dx = float(ds.attrs.get('dx', np.median(np.diff(x1d))))

    print(f"  Data shape: Z_xt = {Z_xt.shape}")
    print(f"  X range: {x1d[0]:.1f} to {x1d[-1]:.1f} m")
    print(f"  Time range: {time_vals[0]} to {time_vals[-1]}")
    print(f"  dt = {dt:.3f} s, dx = {dx:.3f} m")

    # Check for sufficient valid data
    valid_pct = 100 * np.sum(~np.isnan(Z_xt)) / Z_xt.size
    print(f"  Valid data: {valid_pct:.1f}%")

    if valid_pct < 10:
        print("  WARNING: Less than 10% valid data, runup detection may be unreliable")

    # Compute runup statistics
    print("\n  Computing runup statistics...")
    try:
        result = runup.compute_runup_stats(
            Z_xt=Z_xt,
            x1d=x1d,
            time_vec=time_seconds,
            I_xt=I_xt,
            threshold=0.1,
            ig_length=100.0,
            window_length_minutes=5.0,
        )

        print(f"  Runup detection complete:")
        print(f"    Valid detections: {result.info['n_valid']} / {len(time_seconds)}")
        print(f"    Sig (IG): {result.bulk.Sig:.3f} m")
        print(f"    Sinc: {result.bulk.Sinc:.3f} m")
        print(f"    Mean eta: {result.bulk.eta:.3f} m")
        print(f"    Beach slope: {result.bulk.beta:.4f}")
        print(f"    R2%: {result.bulk.R2:.3f} m")

        # Generate figures
        print("\n  Generating figures...")

        # Figure 1: Runup timeseries overview
        fig1_path = date_dir / "runup_timeseries.png"
        create_runup_timeseries_figure(
            Z_xt, x1d, time_vals, time_seconds, result, fig1_path
        )
        print(f"    Saved: {fig1_path}")

        # Figure 2: Runup spectrum
        fig2_path = date_dir / "runup_spectrum.png"
        create_spectrum_figure(result, fig2_path)
        print(f"    Saved: {fig2_path}")

        # Figure 3: Detailed timestack with runup line
        fig3_path = date_dir / "timestack_with_runup.png"
        create_timestack_runup_figure(Z_xt, x1d, time_vals, result, fig3_path)
        print(f"    Saved: {fig3_path}")

        # Figure 4: Statistics summary
        fig4_path = date_dir / "runup_statistics.png"
        create_statistics_summary_figure(result, fig4_path)
        print(f"    Saved: {fig4_path}")

        # Figure 5: Hourly runup segments
        fig5_path = date_dir / "runup_hourly.png"
        create_hourly_runup_figure(Z_xt, x1d, time_vals, time_seconds, result, fig5_path)
        print(f"    Saved: {fig5_path}")

        # Save runup data to CSV
        csv_path = date_dir / "runup_timeseries.csv"
        save_runup_csv(result, time_vals, csv_path)
        print(f"    Saved: {csv_path}")

        # Save bulk stats to text file
        stats_path = date_dir / "runup_stats.txt"
        save_runup_stats(result, nc_path, stats_path)
        print(f"    Saved: {stats_path}")

    except Exception as e:
        print(f"  ERROR in runup computation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    ds.close()
    return date_dir


def create_runup_timeseries_figure(Z_xt, x1d, time_vals, time_seconds, result, output_path):
    """Create figure showing runup position and elevation over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ts = result.timeseries

    # Panel 1: Cross-shore runup position
    ax = axes[0]
    ax.plot(time_vals, ts.X_runup, 'b-', lw=0.5, alpha=0.7)
    ax.axhline(np.nanmean(ts.X_runup), color='r', ls='--', label=f'Mean = {np.nanmean(ts.X_runup):.1f} m')
    ax.set_ylabel('X runup (m)')
    ax.set_title('Runup Position')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Runup elevation
    ax = axes[1]
    ax.plot(time_vals, ts.Z_runup, 'g-', lw=0.5, alpha=0.7)
    ax.axhline(result.bulk.eta, color='r', ls='--', label=f'Mean = {result.bulk.eta:.3f} m')
    ax.axhline(result.bulk.R2, color='orange', ls=':', label=f'R2% = {result.bulk.R2:.3f} m')
    ax.set_ylabel('Z runup (m)')
    ax.set_title('Runup Elevation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Beach profile context
    ax = axes[2]
    z_mean = np.nanmean(Z_xt, axis=1)
    z_std = np.nanstd(Z_xt, axis=1)
    ax.fill_between(x1d, z_mean - z_std, z_mean + z_std, alpha=0.3, label='Std dev')
    ax.plot(x1d, z_mean, 'b-', lw=1.5, label='Mean profile')
    # Add runup range
    x_runup_valid = ts.X_runup[~np.isnan(ts.X_runup)]
    if len(x_runup_valid) > 0:
        ax.axvspan(np.percentile(x_runup_valid, 2), np.percentile(x_runup_valid, 98),
                   alpha=0.2, color='red', label='Runup range (2-98%)')
    ax.set_xlabel('Cross-shore position (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Mean Beach Profile with Runup Zone')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_spectrum_figure(result, output_path):
    """Create figure showing runup spectrum."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    spec = result.spectrum

    if len(spec.frequency) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'Insufficient data for spectrum', transform=ax.transAxes, ha='center')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    freq = spec.frequency
    power = spec.power

    # Panel 1: Linear scale
    ax = axes[0]
    ax.semilogy(freq, power, 'b-', lw=1)
    ax.fill_between(freq, spec.power_lower, spec.power_upper, alpha=0.3)

    # Mark frequency bands
    ax.axvspan(0.004, 0.04, alpha=0.2, color='red', label='IG band (0.004-0.04 Hz)')
    ax.axvspan(0.04, 0.25, alpha=0.2, color='blue', label='Incident band (0.04-0.25 Hz)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (m²/Hz)')
    ax.set_title('Runup Spectrum (log scale)')
    ax.set_xlim(0, 0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Variance-preserving form
    ax = axes[1]
    variance_preserving = freq * power
    ax.plot(freq, variance_preserving, 'b-', lw=1)

    # Mark bands
    ax.axvspan(0.004, 0.04, alpha=0.2, color='red', label='IG')
    ax.axvspan(0.04, 0.25, alpha=0.2, color='blue', label='Incident')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('f × S(f) (m²)')
    ax.set_title('Variance-Preserving Spectrum')
    ax.set_xlim(0, 0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add text with band energies
    fig.text(0.15, 0.95, f'Sig (IG) = {result.bulk.Sig:.3f} m, Sinc = {result.bulk.Sinc:.3f} m',
             fontsize=12, ha='left')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_timestack_runup_figure(Z_xt, x1d, time_vals, result, output_path):
    """Create detailed timestack with runup line overlay."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Subsample for visualization
    n_t = len(time_vals)
    max_points = 5000
    step = max(1, n_t // max_points)

    Z_sub = Z_xt[:, ::step]
    time_sub = time_vals[::step]
    X_runup_sub = result.timeseries.X_runup[::step]

    # Plot elevation timestack
    vmin, vmax = np.nanpercentile(Z_sub, [2, 98])
    im = ax.pcolormesh(time_sub, x1d, Z_sub,
                       vmin=vmin, vmax=vmax,
                       cmap='terrain', shading='auto')

    # Overlay runup line
    ax.plot(time_sub, X_runup_sub, 'r-', lw=1, alpha=0.8, label='Runup position')

    ax.set_xlabel('Time')
    ax.set_ylabel('Cross-shore position (m)')
    ax.set_title('Elevation Timestack with Runup Line')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_statistics_summary_figure(result, output_path):
    """Create a summary figure with key runup statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ts = result.timeseries
    bulk = result.bulk

    # Panel 1: Histogram of runup elevation
    ax = axes[0, 0]
    z_valid = ts.Z_runup[~np.isnan(ts.Z_runup)]
    if len(z_valid) > 0:
        ax.hist(z_valid, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(bulk.eta, color='r', ls='--', lw=2, label=f'Mean = {bulk.eta:.3f} m')
        ax.axvline(bulk.R2, color='orange', ls=':', lw=2, label=f'R2% = {bulk.R2:.3f} m')
    ax.set_xlabel('Runup Elevation (m)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Runup Elevation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Histogram of runup position
    ax = axes[0, 1]
    x_valid = ts.X_runup[~np.isnan(ts.X_runup)]
    if len(x_valid) > 0:
        ax.hist(x_valid, bins=50, density=True, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(np.mean(x_valid), color='r', ls='--', lw=2, label=f'Mean = {np.mean(x_valid):.1f} m')
    ax.set_xlabel('Cross-shore Position (m)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Runup Position Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Autocorrelation
    ax = axes[1, 0]
    if len(z_valid) > 100:
        # Compute autocorrelation
        z_detrend = signal.detrend(z_valid)
        acf = np.correlate(z_detrend, z_detrend, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]  # Normalize

        dt = result.info['dt']
        lags = np.arange(len(acf)) * dt
        max_lag = 300  # 5 minutes
        idx = lags <= max_lag

        ax.plot(lags[idx], acf[idx], 'b-', lw=1)
        ax.axhline(0, color='k', ls='-', lw=0.5)
        ax.axhline(1.96/np.sqrt(len(z_valid)), color='r', ls='--', alpha=0.5)
        ax.axhline(-1.96/np.sqrt(len(z_valid)), color='r', ls='--', alpha=0.5)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Runup Elevation Autocorrelation')
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary statistics text
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    RUNUP STATISTICS SUMMARY
    {'='*40}

    Wave Heights:
      Sig (IG band):     {bulk.Sig:.4f} m
      Sinc (incident):   {bulk.Sinc:.4f} m

    Elevation:
      Mean (eta):        {bulk.eta:.4f} m
      R2% exceedance:    {bulk.R2:.4f} m

    Beach Parameters:
      Foreshore slope:   {bulk.beta:.5f}

    Processing Info:
      Valid detections:  {result.info['n_valid']}
      Duration:          {result.info['duration_seconds']:.0f} s
      dt:                {result.info['dt']:.3f} s
      dx:                {result.info['dx']:.3f} m
      Threshold:         {result.info['threshold']:.3f} m
    """

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=11,
            verticalalignment='top')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_hourly_runup_figure(Z_xt, x1d, time_vals, time_seconds, result, output_path):
    """Create figure showing runup in 1-hour segments."""
    # Determine number of hours
    total_seconds = time_seconds[-1] - time_seconds[0]
    n_hours = int(np.ceil(total_seconds / 3600))

    if n_hours > 24:
        n_hours = 24  # Cap at 24 hours

    # Create grid of subplots
    n_cols = min(4, n_hours)
    n_rows = int(np.ceil(n_hours / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    ts = result.timeseries

    for h in range(n_hours):
        row = h // n_cols
        col = h % n_cols
        ax = axes[row, col]

        # Time indices for this hour
        t_start = time_seconds[0] + h * 3600
        t_end = t_start + 3600
        idx = (time_seconds >= t_start) & (time_seconds < t_end)

        if idx.sum() == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        # Plot runup elevation for this hour
        t_hour = (time_seconds[idx] - t_start) / 60  # Minutes within hour
        z_hour = ts.Z_runup[idx]

        ax.plot(t_hour, z_hour, 'b-', lw=0.5)
        ax.axhline(np.nanmean(z_hour), color='r', ls='--', lw=1, alpha=0.7)

        # Title with hour
        hour_label = str(time_vals[idx][0])[-8:-3] if idx.sum() > 0 else f"Hour {h}"
        ax.set_title(f'{hour_label}', fontsize=10)

        if col == 0:
            ax.set_ylabel('Z (m)')
        if row == n_rows - 1:
            ax.set_xlabel('Time (min)')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)

    # Hide empty subplots
    for h in range(n_hours, n_rows * n_cols):
        row = h // n_cols
        col = h % n_cols
        axes[row, col].axis('off')

    fig.suptitle('Hourly Runup Segments', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_runup_csv(result, time_vals, output_path):
    """Save runup timeseries to CSV."""
    import pandas as pd

    ts = result.timeseries

    df = pd.DataFrame({
        'time': time_vals,
        'time_seconds': ts.time,
        'X_runup_m': ts.X_runup,
        'Z_runup_m': ts.Z_runup,
        'idx_runup': ts.idx_runup,
    })

    df.to_csv(output_path, index=False)


def save_runup_stats(result, nc_path, output_path):
    """Save runup statistics to text file."""
    bulk = result.bulk
    info = result.info

    with open(output_path, 'w') as f:
        f.write(f"Runup Statistics Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Source: {nc_path}\n\n")

        f.write(f"Wave Heights\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Sig (IG band, 0.004-0.04 Hz):  {bulk.Sig:.6f} m\n")
        f.write(f"  Sinc (incident, 0.04-0.25 Hz): {bulk.Sinc:.6f} m\n\n")

        f.write(f"Elevation Statistics\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Mean (eta):     {bulk.eta:.6f} m\n")
        f.write(f"  R2% exceedance: {bulk.R2:.6f} m\n\n")

        f.write(f"Beach Parameters\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Foreshore slope (beta): {bulk.beta:.6f}\n\n")

        f.write(f"Processing Parameters\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  dt: {info['dt']:.4f} s\n")
        f.write(f"  dx: {info['dx']:.4f} m\n")
        f.write(f"  Threshold: {info['threshold']:.4f} m\n")
        f.write(f"  IG window: {info['ig_length']:.1f} s\n\n")

        f.write(f"Data Coverage\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Valid detections: {info['n_valid']}\n")
        f.write(f"  Duration: {info['duration_seconds']:.1f} s ({info['duration_seconds']/3600:.2f} hr)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate runup figures from L2 NetCDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to NC file or directory containing NC files"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Config file to determine output directory (uses plot_folder/level2/)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for figures (default: config plot_folder/level2/ or figures/l2)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed info"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.config is not None:
        # Use config's plot_folder with level2/ subfolder
        from phase1 import load_config
        config = load_config(args.config)
        output_dir = config.plot_folder / "level2"
    else:
        # Fallback: figures/l2 relative to script
        output_dir = Path(__file__).parent.parent / "figures" / "l2"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find NC files
    input_path = args.input
    if input_path.is_file():
        nc_files = [input_path]
    elif input_path.is_dir():
        nc_files = sorted(input_path.glob("*.nc"))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not nc_files:
        print(f"No NC files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(nc_files)} NC file(s)")
    print(f"Output directory: {output_dir}")

    for nc_path in nc_files:
        try:
            process_runup(nc_path, output_dir, verbose=args.verbose)
        except Exception as e:
            print(f"Error processing {nc_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
