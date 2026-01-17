"""
Phase 4: orchestration/CLI-style drivers for L1/L2 processing.
This keeps I/O thin and composes the pure functions from phases 1–3.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

import phase1
import phase2
import phase3

logger = logging.getLogger(__name__)

# Loader type: given a path, return (points[N,3], intensities[N], gps_times[N])
LoaderFn = Callable[[Path], Tuple[np.ndarray, np.ndarray, np.ndarray]]


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
    edge_percentile: Optional[Tuple[float, float]] = (1.0, 99.0),
    edge_padding_bins: int = 1,
) -> phase3.xr.Dataset:
    """
    Orchestrate L1 processing: discover files -> load -> transform/filter -> bin -> stack.
    Edges are shared across batches; optional percentile clipping reduces empty-grid blowup.
    Returns an xarray.Dataset; caller may persist to NetCDF/Parquet as needed.
    """
    cfg = phase1.load_config(config_path)
    laz_files = phase1.discover_laz_files(cfg.data_folder, start=start, end=end)
    if not laz_files:
        raise FileNotFoundError("No matching .laz files found")

    load_fn = loader or load_laz_points
    batches = []
    for path, ts in laz_files:
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
        batches.append((f_points, ts))

    if not batches:
        raise RuntimeError("All files filtered out; no grids produced")

    # Use common bin edges across all batches so datasets align; avoid vstack to save memory.
    x_min = min(np.min(b[0][:, 0]) for b in batches)
    x_max = max(np.max(b[0][:, 0]) for b in batches)
    y_min = min(np.min(b[0][:, 1]) for b in batches)
    y_max = max(np.max(b[0][:, 1]) for b in batches)
    if edge_percentile:
        lo, hi = edge_percentile
        x_samples = np.hstack([np.percentile(b[0][:, 0], [lo, hi]) for b in batches])
        y_samples = np.hstack([np.percentile(b[0][:, 1], [lo, hi]) for b in batches])
        # Clip extrema to reduce massive empty grids, but keep a small padding.
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
    return phase3.build_dataset(grids)


def save_dataset(ds: phase3.xr.Dataset, output_path: Path) -> None:
    """
    Save dataset to NetCDF using xarray.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
