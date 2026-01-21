"""
Phase 4: orchestration/CLI-style drivers for L1/L2 processing.
This keeps I/O thin and composes the pure functions from phases 1–3.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import phase1
import phase2
import phase3
import profiles

logger = logging.getLogger(__name__)

# Loader type: given a path, return (points[N,3], intensities[N], gps_times[N])
LoaderFn = Callable[[Path], Tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class L1Result:
    """Result from L1 processing including profiles."""
    dataset: phase3.xr.Dataset
    profiles: Optional[profiles.TransectResult] = None
    profile_config: Optional[profiles.TransectConfig] = None


def load_laz_points(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a .laz file using laspy and return points, intensities, and GPS times.
    """
    import laspy

    with laspy.open(path) as f:
        hdr = f.header
        pts = f.read()
    xyz = np.vstack((pts.X * hdr.scale[0] + hdr.offset[0],
                     pts.Y * hdr.scale[1] + hdr.offset[1],
                     pts.Z * hdr.scale[2] + hdr.offset[2])).T.astype(float)
    intensity = np.asarray(pts.intensity, dtype=float)
    if hasattr(pts, "gps_time"):
        gps_time = np.asarray(pts.gps_time, dtype=float)
    else:
        gps_time = np.arange(len(xyz), dtype=float)
    return xyz, intensity, gps_time


def process_l1(
    config_path: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    bin_size: float = 0.1,
    mode_bin: float = 0.05,
    intensity_threshold: float = 100.0,
    max_seconds: Optional[float] = 300.0,
    loader: Optional[LoaderFn] = None,
    data_folder_override: Optional[Path] = None,
    edge_percentile: Optional[Tuple[float, float]] = (1.0, 99.0),
    edge_padding_bins: int = 1,
    apply_residual_filter: bool = True,
    residual_filter_passes: Optional[List[Tuple[float, float]]] = None,
    extract_profiles: bool = False,
    profile_config: Optional[profiles.TransectConfig] = None,
) -> L1Result:
    """
    Orchestrate L1 processing: discover files -> load -> transform/filter ->
    residual filter -> bin -> stack -> extract profiles.

    Parameters
    ----------
    config_path : Path
        Path to livox_config.json
    start, end : datetime, optional
        Date range to process
    bin_size : float
        Spatial bin size in meters (default 0.1)
    mode_bin : float
        Mode quantization bin size (default 0.05)
    intensity_threshold : float
        Maximum intensity to keep (default 100)
    max_seconds : float, optional
        Time window per file in seconds (default 300 = 5 min)
    loader : callable, optional
        Custom LAZ loader function
    data_folder_override : Path, optional
        Override data folder from config
    edge_percentile : tuple, optional
        Percentile range for edge clipping (default 1-99%)
    edge_padding_bins : int
        Extra bins to add at edges (default 1)
    apply_residual_filter : bool
        Apply two-stage residual kernel filtering (default True)
    residual_filter_passes : list, optional
        Filter passes as [(cell_size, threshold), ...]. Default: [(10, 0.2), (3, 0.1)]
    extract_profiles : bool
        Extract 1D cross-shore profiles (default False)
    profile_config : TransectConfig, optional
        Configuration for profile extraction

    Returns
    -------
    L1Result
        Contains xarray.Dataset and optional profile results
    """
    cfg = phase1.load_config(config_path)
    if data_folder_override is not None:
        cfg = replace(cfg, data_folder=Path(data_folder_override))
    laz_files = phase1.discover_laz_files(cfg.data_folder, start=start, end=end)
    if not laz_files:
        raise FileNotFoundError("No matching .laz files found")

    load_fn = loader or load_laz_points
    batches = []
    for path, ts in laz_files:
        logger.info("Loading %s", path.name)
        points, intensities, gps_times = load_fn(path)
        f_points, f_intensity, f_times = phase1.prepare_batch(
            points,
            intensities,
            gps_times,
            cfg,
            intensity_threshold=intensity_threshold,
            max_seconds=max_seconds,
        )
        if len(f_points) == 0:
            logger.warning("No points after filtering for file %s", path)
            continue

        # Apply two-stage residual kernel filtering
        if apply_residual_filter and len(f_points) > 100:
            logger.info("Applying residual filter to %d points", len(f_points))
            f_points = phase2.residual_kernel_filter_two_stage(
                f_points,
                passes=residual_filter_passes,
            )
            logger.info("After filtering: %d points", len(f_points))

        if len(f_points) == 0:
            logger.warning("No points after residual filtering for file %s", path)
            continue

        batches.append((f_points, ts))

    if not batches:
        raise RuntimeError("All files filtered out; no grids produced")

    # Use common bin edges across all batches so datasets align
    x_min = min(np.min(b[0][:, 0]) for b in batches)
    x_max = max(np.max(b[0][:, 0]) for b in batches)
    y_min = min(np.min(b[0][:, 1]) for b in batches)
    y_max = max(np.max(b[0][:, 1]) for b in batches)
    if edge_percentile:
        lo, hi = edge_percentile
        x_samples = np.hstack([np.percentile(b[0][:, 0], [lo, hi]) for b in batches])
        y_samples = np.hstack([np.percentile(b[0][:, 1], [lo, hi]) for b in batches])
        x_min = max(x_min, x_samples.min() - edge_padding_bins * bin_size)
        x_max = min(x_max, x_samples.max() + edge_padding_bins * bin_size)
        y_min = max(y_min, y_samples.min() - edge_padding_bins * bin_size)
        y_max = min(y_max, y_samples.max() + edge_padding_bins * bin_size)

    x_edges = phase2.bin_edges(np.array([x_min, x_max]), bin_size)
    y_edges = phase2.bin_edges(np.array([y_min, y_max]), bin_size)

    grids = []
    for f_points, ts in batches:
        grid = phase2.bin_point_cloud(
            f_points, bin_size=bin_size, mode_bin=mode_bin, x_edges=x_edges, y_edges=y_edges
        )
        grids.append(phase3.GridWithTime(grid=grid, timestamp=ts))

    dataset = phase3.build_dataset(grids)

    # Extract profiles if requested
    profile_result = None
    if extract_profiles:
        # Combine all points for profile extraction
        all_points = np.vstack([b[0] for b in batches])
        X, Y, Z = all_points[:, 0], all_points[:, 1], all_points[:, 2]

        if profile_config is not None:
            profile_result = profiles.extract_transects(X, Y, Z, config=profile_config)
        else:
            logger.warning("Profile extraction requested but no profile_config provided")

    return L1Result(
        dataset=dataset,
        profiles=profile_result,
        profile_config=profile_config,
    )


def process_l2(
    config_path: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    x_bin_size: float = 0.1,
    time_bin_size: float = 0.5,
    intensity_threshold: float = 100.0,
    max_seconds: Optional[float] = 120.0,
    loader: Optional[LoaderFn] = None,
    data_folder_override: Optional[Path] = None,
    apply_residual_filter: bool = True,
    residual_filter_passes: Optional[List[Tuple[float, float]]] = None,
    profile_config: Optional[profiles.TransectConfig] = None,
) -> phase3.TimeResolvedDataset:
    """
    Orchestrate L2 processing: produces time-resolved Z(x,t) matrices for wave analysis.

    L2 differs from L1:
    - Temporal binning at ~2Hz (time_bin_size=0.5s default)
    - Shorter time window (2 min default vs 5 min for L1)
    - Less aggressive filtering
    - Outputs along a single cross-shore transect

    Parameters
    ----------
    config_path : Path
        Path to livox_config.json
    start, end : datetime, optional
        Date range to process
    x_bin_size : float
        Spatial bin size along transect (default 0.1m)
    time_bin_size : float
        Temporal bin size in seconds (default 0.5 = 2Hz)
    intensity_threshold : float
        Maximum intensity to keep (default 100)
    max_seconds : float
        Time window per file (default 120 = 2 min)
    loader : callable, optional
        Custom LAZ loader function
    data_folder_override : Path, optional
        Override data folder from config
    apply_residual_filter : bool
        Apply residual kernel filtering (default True, but less aggressive)
    residual_filter_passes : list, optional
        Filter passes. Default for L2: [(2, 0.5)] - less aggressive than L1
    profile_config : TransectConfig, optional
        Configuration for the central transect

    Returns
    -------
    TimeResolvedDataset
        Contains Z(x,t), I(x,t), and metadata
    """
    if residual_filter_passes is None:
        # L2 uses less aggressive filtering
        residual_filter_passes = [(2.0, 0.5)]

    cfg = phase1.load_config(config_path)
    if data_folder_override is not None:
        cfg = replace(cfg, data_folder=Path(data_folder_override))

    laz_files = phase1.discover_laz_files(cfg.data_folder, start=start, end=end)
    if not laz_files:
        raise FileNotFoundError("No matching .laz files found")

    load_fn = loader or load_laz_points

    all_points = []
    all_intensities = []
    all_times = []
    base_time = None

    for path, ts in laz_files:
        logger.info("Loading %s for L2", path.name)
        points, intensities, gps_times = load_fn(path)
        f_points, f_intensity, f_times = phase1.prepare_batch(
            points,
            intensities,
            gps_times,
            cfg,
            intensity_threshold=intensity_threshold,
            max_seconds=max_seconds,
        )
        if len(f_points) == 0:
            logger.warning("No points after filtering for file %s", path)
            continue

        # Apply residual filtering (less aggressive for L2)
        if apply_residual_filter and len(f_points) > 100:
            f_points = phase2.residual_kernel_filter_two_stage(
                f_points,
                passes=residual_filter_passes,
            )

        if len(f_points) == 0:
            continue

        # Track absolute time
        if base_time is None:
            base_time = ts

        # Convert file timestamp offset to seconds
        file_offset = (ts - base_time).total_seconds()
        absolute_times = f_times + file_offset

        all_points.append(f_points)
        all_intensities.append(f_intensity[:len(f_points)] if len(f_intensity) >= len(f_points)
                               else np.full(len(f_points), np.nan))
        all_times.append(absolute_times[:len(f_points)] if len(absolute_times) >= len(f_points)
                        else np.arange(len(f_points)) + file_offset)

    if not all_points:
        raise RuntimeError("All files filtered out; no data for L2")

    points = np.vstack(all_points)
    intensities = np.hstack(all_intensities)
    times = np.hstack(all_times)

    # Extract along a transect if config provided
    if profile_config is not None:
        # Project points onto the transect
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

        # Use the center transect (offset=0)
        line_start = (profile_config.x1, profile_config.y1)
        dx = profile_config.x2 - profile_config.x1
        dy = profile_config.y2 - profile_config.y1
        length = np.sqrt(dx**2 + dy**2)
        line_vec = np.array([dx, dy]) / length

        # Project points
        proj_dist, dist_to_line, mask = profiles._project_points_to_line(
            np.column_stack([X, Y]), line_start, line_vec, profile_config.tolerance
        )

        # Filter to points near transect
        points_1d = proj_dist[mask]
        z_1d = Z[mask]
        intensities_1d = intensities[mask]
        times_1d = times[mask]

        # Create pseudo-3D points for temporal binning (x=cross-shore, y=0, z=elevation)
        points_for_binning = np.column_stack([points_1d, np.zeros_like(points_1d), z_1d])
    else:
        # Use X coordinate directly as cross-shore
        points_for_binning = points
        intensities_1d = intensities
        times_1d = times

    # Temporal binning
    grid = phase2.bin_point_cloud_temporal(
        points_for_binning,
        intensities_1d,
        times_1d,
        x_bin_size=x_bin_size,
        time_bin_size=time_bin_size,
    )

    return phase3.TimeResolvedDataset(
        grid=grid,
        base_time=base_time,
        profile_config=profile_config,
    )


def process_l1_batch(
    config_path: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    **kwargs,
) -> List[Path]:
    """
    Process L1 data in daily batches, saving each day to a separate file.

    Parameters
    ----------
    config_path : Path
        Path to config JSON
    start, end : datetime
        Date range to process
    output_dir : Path
        Directory for output NetCDF files
    **kwargs
        Additional arguments passed to process_l1()

    Returns
    -------
    list of Path
        Paths to saved NetCDF files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)

    while current < end:
        day_end = current + timedelta(days=1)
        logger.info("Processing %s", current.strftime("%Y-%m-%d"))

        try:
            result = process_l1(
                config_path,
                start=current,
                end=min(day_end, end),
                **kwargs,
            )

            output_path = output_dir / f"L1_{current.strftime('%Y%m%d')}.nc"
            save_dataset(result.dataset, output_path)
            saved_files.append(output_path)
            logger.info("Saved %s", output_path)

        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("No data for %s: %s", current.strftime("%Y-%m-%d"), e)

        current = day_end

    return saved_files


def save_dataset(ds: phase3.xr.Dataset, output_path: Path) -> None:
    """
    Save dataset to NetCDF using xarray.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)


def export_profiles_json(
    result: L1Result,
    output_path: Path,
    transect_indices: Optional[List[int]] = None,
) -> None:
    """
    Export 1D profiles to JSON for web visualization.
    Matches MATLAB's lidar_plot_data.json format.

    Parameters
    ----------
    result : L1Result
        L1 processing result with profiles
    output_path : Path
        Output JSON file path
    transect_indices : list of int, optional
        Which transects to export (default: all)
    """
    import json

    if result.profiles is None:
        raise ValueError("No profiles in result")

    profiles_data = []
    x1d = result.profiles.x1d.tolist()

    if transect_indices is None:
        transect_indices = list(range(result.profiles.Z3D.shape[0]))

    for i in transect_indices:
        z_profile = result.profiles.Z3D[i, :]
        offset = result.profile_config.alongshore_spacings[i] if result.profile_config else i

        profiles_data.append({
            "alongshore_offset": float(offset),
            "x": x1d,
            "z": [float(v) if not np.isnan(v) else None for v in z_profile],
        })

    output = {
        "profiles": profiles_data,
        "metadata": {
            "resolution": result.profile_config.resolution if result.profile_config else None,
            "n_transects": len(transect_indices),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for lidar processing."""
    parser = argparse.ArgumentParser(
        description="Lidar point cloud processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Processing command")

    # L1 command
    l1_parser = subparsers.add_parser("l1", help="Process L1 (hourly beach surface)")
    l1_parser.add_argument("config", type=Path, help="Path to config JSON")
    l1_parser.add_argument("-o", "--output", type=Path, required=True, help="Output NetCDF path")
    l1_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    l1_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    l1_parser.add_argument("--bin-size", type=float, default=0.1, help="Spatial bin size (m)")
    l1_parser.add_argument("--no-filter", action="store_true", help="Skip residual filtering")
    l1_parser.add_argument("--data-folder", type=Path, help="Override data folder")

    # L2 command
    l2_parser = subparsers.add_parser("l2", help="Process L2 (wave-resolving)")
    l2_parser.add_argument("config", type=Path, help="Path to config JSON")
    l2_parser.add_argument("-o", "--output", type=Path, required=True, help="Output NetCDF path")
    l2_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    l2_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    l2_parser.add_argument("--time-bin", type=float, default=0.5, help="Time bin size (s)")
    l2_parser.add_argument("--data-folder", type=Path, help="Override data folder")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process L1 by day")
    batch_parser.add_argument("config", type=Path, help="Path to config JSON")
    batch_parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    batch_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    batch_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    batch_parser.add_argument("--data-folder", type=Path, help="Override data folder")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.command == "l1":
        start = datetime.fromisoformat(args.start) if args.start else None
        end = datetime.fromisoformat(args.end) if args.end else None

        result = process_l1(
            args.config,
            start=start,
            end=end,
            bin_size=args.bin_size,
            apply_residual_filter=not args.no_filter,
            data_folder_override=args.data_folder,
        )
        save_dataset(result.dataset, args.output)
        print(f"Saved L1 dataset to {args.output}")

    elif args.command == "l2":
        start = datetime.fromisoformat(args.start) if args.start else None
        end = datetime.fromisoformat(args.end) if args.end else None

        result = process_l2(
            args.config,
            start=start,
            end=end,
            time_bin_size=args.time_bin,
            data_folder_override=args.data_folder,
        )
        # Save L2 result
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(output_path)
        print(f"Saved L2 dataset to {args.output}")

    elif args.command == "batch":
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)

        saved = process_l1_batch(
            args.config,
            start=start,
            end=end,
            output_dir=args.output_dir,
            data_folder_override=args.data_folder,
        )
        print(f"Processed {len(saved)} days")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
