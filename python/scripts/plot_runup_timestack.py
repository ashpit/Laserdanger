#!/usr/bin/env python3
"""
Plot runup timestack figures similar to published wave runup visualizations.

Creates two-panel figures:
  (a) Intensity timestack with runup line overlay
  (b) Water level "score" with runup line overlay

Usage:
    python scripts/plot_runup_timestack.py L2_output.nc
    python scripts/plot_runup_timestack.py L2_output.nc -o figures/
    python scripts/plot_runup_timestack.py L2_output.nc --t-start 200 --t-end 360
"""
import argparse
from pathlib import Path
import sys

# Add code directory to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import minimum_filter1d, uniform_filter1d, median_filter
from scipy import signal

import runup


def load_l2_data(path: Path) -> xr.Dataset:
    """Load L2 NetCDF dataset."""
    return xr.open_dataset(path)


def compute_water_level_score(
    Z_xt: np.ndarray,
    dt: float,
    ig_length: float = 100.0,
) -> np.ndarray:
    """
    Compute water level score (similar to softmax score visualization).

    This represents how far above/below the dry beach reference each point is,
    normalized to create a score that highlights wet vs dry regions.

    Parameters
    ----------
    Z_xt : array (n_x, n_t)
        Elevation matrix
    dt : float
        Time step in seconds
    ig_length : float
        Window for dry beach reference (default 100s)

    Returns
    -------
    score : array (n_x, n_t)
        Water level score (positive = wet, negative = dry)
    """
    # Use runup module's dry beach computation
    dry_ref = runup.compute_dry_beach_reference(Z_xt, dt, ig_length)

    # Water level = elevation - dry reference
    water_level = Z_xt - dry_ref

    # Normalize to create a "score" - scale by local variability
    # This enhances the contrast similar to softmax output
    std_spatial = np.nanstd(water_level, axis=1, keepdims=True)
    std_spatial = np.where(std_spatial < 0.01, 0.01, std_spatial)

    score = water_level / std_spatial

    # Clip to reasonable range
    score = np.clip(score, -10, 10)

    return score


def auto_threshold(Z_xt: np.ndarray, dt: float, ig_length: float = 100.0) -> float:
    """
    Automatically determine a reasonable runup threshold based on water level variability.

    Returns threshold that should capture ~50-70% of the water level excursions.
    """
    dry_ref = runup.compute_dry_beach_reference(Z_xt, dt, ig_length)
    water_level = Z_xt - dry_ref

    # Use the 50th percentile of positive water levels as threshold
    positive_wl = water_level[water_level > 0]
    if len(positive_wl) > 100:
        threshold = np.percentile(positive_wl, 50)
        # Clamp to reasonable range
        threshold = np.clip(threshold, 0.02, 0.5)
    else:
        threshold = 0.1  # fallback

    return threshold


def compute_runup_with_bounds(
    Z_xt: np.ndarray,
    x1d: np.ndarray,
    time_vec: np.ndarray,
    threshold: float = None,
    ig_length: float = 100.0,
) -> dict:
    """
    Compute runup line with upper/lower bounds.

    Returns dict with:
        - X_runup: main runup position (m)
        - X_upper: upper bound (seaward)
        - X_lower: lower bound (landward)
        - Z_runup: runup elevation (m)
    """
    dt = np.median(np.diff(time_vec))

    # Auto-determine threshold if not specified
    if threshold is None:
        threshold = auto_threshold(Z_xt, dt, ig_length)
        print(f"  Auto-detected threshold: {threshold:.3f} m")

    # Main runup detection
    result = runup.compute_runup_stats(
        Z_xt, x1d, time_vec,
        threshold=threshold,
        ig_length=ig_length,
    )

    X_runup = result.timeseries.X_runup
    Z_runup = result.timeseries.Z_runup

    n_valid = np.sum(~np.isnan(X_runup))
    print(f"  Runup detected at {n_valid}/{len(time_vec)} time steps ({100*n_valid/len(time_vec):.1f}%)")

    # Compute bounds using different thresholds
    # Upper bound (seaward) - lower threshold
    result_upper = runup.compute_runup_stats(
        Z_xt, x1d, time_vec,
        threshold=threshold * 0.5,
        ig_length=ig_length,
    )
    X_upper = result_upper.timeseries.X_runup

    # Lower bound (landward) - higher threshold
    result_lower = runup.compute_runup_stats(
        Z_xt, x1d, time_vec,
        threshold=threshold * 1.5,
        ig_length=ig_length,
    )
    X_lower = result_lower.timeseries.X_runup

    return {
        'X_runup': X_runup,
        'X_upper': X_upper,
        'X_lower': X_lower,
        'Z_runup': Z_runup,
        'bulk': result.bulk,
        'spectrum': result.spectrum,
        'threshold': threshold,
    }


def plot_runup_timestack(
    ds: xr.Dataset,
    output_path: Path,
    t_start: float = None,
    t_end: float = None,
    threshold: float = None,
    title: str = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Create two-panel runup timestack figure.

    Panel (a): Intensity timestack with runup lines
    Panel (b): Water level score with runup lines

    Parameters
    ----------
    threshold : float, optional
        Runup detection threshold in meters. If None, auto-detected from data.
    """
    # Extract data
    Z = ds.elevation.values
    I = ds.intensity.values
    x = ds.x.values
    t = ds.time_seconds.values
    dt = float(ds.attrs.get('dt', np.median(np.diff(t))))

    # Time subset
    if t_start is not None or t_end is not None:
        t_start = t_start if t_start is not None else t[0]
        t_end = t_end if t_end is not None else t[-1]
        t_mask = (t >= t_start) & (t <= t_end)
        t = t[t_mask]
        Z = Z[:, t_mask]
        I = I[:, t_mask]

    # Compute runup (threshold=None triggers auto-detection)
    runup_data = compute_runup_with_bounds(Z, x, t, threshold=threshold)

    # Compute water level score
    score = compute_water_level_score(Z, dt)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # === Panel (a): Intensity timestack ===
    ax = axes[0]

    # Plot intensity background
    im1 = ax.pcolormesh(t, x, I, cmap='gray', shading='auto',
                        vmin=np.nanpercentile(I, 5),
                        vmax=np.nanpercentile(I, 95))

    # Overlay runup lines
    ax.plot(t, runup_data['X_runup'], 'k-', linewidth=1.5, label='runup')
    ax.plot(t, runup_data['X_upper'], 'r:', linewidth=1.0, label='upper/lower bound')
    ax.plot(t, runup_data['X_lower'], 'r:', linewidth=1.0)

    ax.set_ylabel('x-shore distance (m)')
    ax.legend(loc='upper right', fontsize=9)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top')

    cb1 = plt.colorbar(im1, ax=ax, label='image intensity', pad=0.02)

    # === Panel (b): Water level score ===
    ax = axes[1]

    # Plot score with diverging colormap
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    im2 = ax.pcolormesh(t, x, score, cmap='RdBu_r', norm=norm, shading='auto')

    # Overlay same runup lines
    ax.plot(t, runup_data['X_runup'], 'k-', linewidth=1.5)
    ax.plot(t, runup_data['X_upper'], 'r:', linewidth=1.0)
    ax.plot(t, runup_data['X_lower'], 'r:', linewidth=1.0)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('x-shore distance (m)')
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top')

    cb2 = plt.colorbar(im2, ax=ax, label='water level score', pad=0.02)

    # Title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()

    return runup_data


def plot_intensity_only(
    ds: xr.Dataset,
    output_path: Path,
    t_start: float = None,
    t_end: float = None,
    threshold: float = None,
    title: str = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Create single-panel intensity timestack with runup overlay.
    """
    # Extract data
    Z = ds.elevation.values
    I = ds.intensity.values
    x = ds.x.values
    t = ds.time_seconds.values

    # Time subset
    if t_start is not None or t_end is not None:
        t_start = t_start if t_start is not None else t[0]
        t_end = t_end if t_end is not None else t[-1]
        t_mask = (t >= t_start) & (t <= t_end)
        t = t[t_mask]
        Z = Z[:, t_mask]
        I = I[:, t_mask]

    # Compute runup (threshold=None triggers auto-detection)
    runup_data = compute_runup_with_bounds(Z, x, t, threshold=threshold)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot intensity background
    im = ax.pcolormesh(t, x, I, cmap='gray', shading='auto',
                       vmin=np.nanpercentile(I, 5),
                       vmax=np.nanpercentile(I, 95))

    # Overlay runup lines
    ax.plot(t, runup_data['X_runup'], 'k-', linewidth=1.5, label='runup')
    ax.plot(t, runup_data['X_upper'], 'r:', linewidth=1.0, label='upper/lower bound')
    ax.plot(t, runup_data['X_lower'], 'r:', linewidth=1.0)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('x-shore distance (m)')
    ax.legend(loc='upper right')

    plt.colorbar(im, ax=ax, label='intensity')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def print_runup_stats(runup_data: dict, title: str = None):
    """Print runup statistics summary."""
    print("=" * 50)
    if title:
        print(f"Runup Statistics: {title}")
    else:
        print("Runup Statistics")
    print("=" * 50)

    # Check if we have valid runup data
    n_valid = np.sum(~np.isnan(runup_data['X_runup']))
    print(f"  Threshold used:    {runup_data.get('threshold', 'N/A'):.3f} m")
    print(f"  Valid detections:  {n_valid}/{len(runup_data['X_runup'])}")

    if n_valid < 10:
        print("\n  WARNING: Insufficient runup detections for reliable statistics.")
        print("  This may indicate calm conditions or data quality issues.")
        print("=" * 50)
        return

    bulk = runup_data['bulk']
    print(f"  Sig (IG band):     {bulk.Sig:.3f} m")
    print(f"  Sinc (incident):   {bulk.Sinc:.3f} m")
    print(f"  Mean elevation:    {bulk.eta:.3f} m" if not np.isnan(bulk.eta) else "  Mean elevation:    N/A")
    print(f"  R2% exceedance:    {bulk.R2:.3f} m" if not np.isnan(bulk.R2) else "  R2% exceedance:    N/A")
    print(f"  Beach slope:       {bulk.beta:.4f}" if not np.isnan(bulk.beta) else "  Beach slope:       N/A")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Plot runup timestack figures with runup line overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/plot_runup_timestack.py L2_output.nc
    python scripts/plot_runup_timestack.py L2_output.nc --t-start 200 --t-end 360
    python scripts/plot_runup_timestack.py L2_output.nc -o figures/ --title "Del Mar 2025-05-03"
        """
    )
    parser.add_argument("input", type=Path, help="Input L2 NetCDF file")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: figures/ in input dir)")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Start time in seconds")
    parser.add_argument("--t-end", type=float, default=None,
                        help="End time in seconds")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Runup detection threshold in meters (default: auto-detect)")
    parser.add_argument("--title", type=str, default=None,
                        help="Figure title")
    parser.add_argument("--single-panel", action="store_true",
                        help="Create single-panel intensity plot only")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output resolution (default: 150)")
    parser.add_argument("--no-stats", action="store_true",
                        help="Don't print runup statistics")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set output directory
    if args.output_dir is None:
        output_dir = args.input.parent / "figures"
    else:
        output_dir = args.output_dir

    # Load data
    print(f"Loading: {args.input}")
    ds = load_l2_data(args.input)

    # Generate filename
    stem = args.input.stem
    if args.t_start is not None or args.t_end is not None:
        t_str = f"_t{args.t_start or 0:.0f}-{args.t_end or 'end'}"
    else:
        t_str = ""

    # Plot
    if args.single_panel:
        output_path = output_dir / f"{stem}_runup_intensity{t_str}.png"
        plot_intensity_only(
            ds, output_path,
            t_start=args.t_start, t_end=args.t_end,
            threshold=args.threshold,
            title=args.title,
            show=args.show,
            dpi=args.dpi,
        )
    else:
        output_path = output_dir / f"{stem}_runup_timestack{t_str}.png"
        runup_data = plot_runup_timestack(
            ds, output_path,
            t_start=args.t_start, t_end=args.t_end,
            threshold=args.threshold,
            title=args.title,
            show=args.show,
            dpi=args.dpi,
        )

        if not args.no_stats:
            print_runup_stats(runup_data, title=args.title)

    ds.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
