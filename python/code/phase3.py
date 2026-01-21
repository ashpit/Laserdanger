"""
Phase 3: data model using xarray.
Builds time-stacked datasets from binned grids for easy slicing and export.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import numpy as np
import xarray as xr
import pandas as pd

from phase2 import BinnedGrid, TimeResolvedGrid

if TYPE_CHECKING:
    import profiles


@dataclass(frozen=True)
class GridWithTime:
    grid: BinnedGrid
    timestamp: datetime


@dataclass
class TimeResolvedDataset:
    """
    L2 time-resolved dataset containing Z(x,t) and I(x,t) matrices.
    Used for wave-resolving analysis and runup detection.
    """
    grid: TimeResolvedGrid
    base_time: datetime
    profile_config: Optional["profiles.TransectConfig"] = None

    @property
    def x1d(self) -> np.ndarray:
        """Cross-shore positions (bin centers)."""
        return (self.grid.x_edges[:-1] + self.grid.x_edges[1:]) / 2

    @property
    def time_vec(self) -> np.ndarray:
        """Time vector in seconds from base_time."""
        return (self.grid.t_edges[:-1] + self.grid.t_edges[1:]) / 2

    @property
    def Z_xt(self) -> np.ndarray:
        """Elevation matrix Z(x,t) with shape (n_x, n_t)."""
        # grid.z_mean has shape (n_t, n_x), transpose to (n_x, n_t)
        return self.grid.z_mean.T

    @property
    def I_xt(self) -> np.ndarray:
        """Intensity matrix I(x,t) with shape (n_x, n_t)."""
        return self.grid.intensity_mean.T

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        return float(np.median(np.diff(self.grid.t_edges)))

    @property
    def dx(self) -> float:
        """Spatial step in meters."""
        return float(np.median(np.diff(self.grid.x_edges)))

    def to_xarray(self) -> xr.Dataset:
        """Convert to xarray Dataset for export and analysis."""
        x_centers = self.x1d
        t_centers = self.time_vec

        # Create time index
        time_index = pd.to_datetime(self.base_time) + pd.to_timedelta(t_centers, unit='s')

        ds = xr.Dataset(
            data_vars={
                "elevation": (("x", "time"), self.Z_xt),
                "elevation_min": (("x", "time"), self.grid.z_min.T),
                "elevation_max": (("x", "time"), self.grid.z_max.T),
                "elevation_std": (("x", "time"), self.grid.z_std.T),
                "intensity": (("x", "time"), self.I_xt),
                "count": (("x", "time"), self.grid.count.T),
            },
            coords={
                "x": x_centers,
                "time": time_index,
                "time_seconds": (("time",), t_centers),
            },
            attrs={
                "dx": self.dx,
                "dt": self.dt,
                "base_time": str(self.base_time),
                "n_x": len(x_centers),
                "n_t": len(t_centers),
            },
        )
        return ds

    def to_netcdf(self, path: Path) -> None:
        """Save to NetCDF file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_xarray().to_netcdf(path)


def grid_to_dataset(grid: BinnedGrid, timestamp: datetime) -> xr.Dataset:
    """
    Convert a single BinnedGrid to an xarray Dataset with dims (time, y, x).
    """
    x_centers = _bin_centers(grid.x_edges)
    y_centers = _bin_centers(grid.y_edges)
    coords = {"y": y_centers, "x": x_centers}
    ds = xr.Dataset(
        data_vars={
            "elevation": (("y", "x"), grid.z_mean.T),
            "elevation_min": (("y", "x"), grid.z_min.T),
            "elevation_max": (("y", "x"), grid.z_max.T),
            "elevation_std": (("y", "x"), grid.z_std.T),
            "elevation_mode": (("y", "x"), grid.z_mode.T),
            "count": (("y", "x"), grid.count.T),
            "snr": (("y", "x"), grid.snr.T),
        },
        coords=coords,
        attrs={
            "bin_size_x": float(np.diff(grid.x_edges[:2])[0]),
            "bin_size_y": float(np.diff(grid.y_edges[:2])[0]),
            "x_edges": grid.x_edges,
            "y_edges": grid.y_edges,
        },
    )
    # add time as a timezone-aware dimension (always normalized to UTC)
    idx = pd.DatetimeIndex([timestamp])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    ds.attrs["time_timezone"] = "UTC"
    return ds.expand_dims(time=idx)


def build_dataset(grids: Sequence[GridWithTime]) -> xr.Dataset:
    """
    Stack multiple GridWithTime entries into a single Dataset along time.
    Expects all grids to share the same bin edges.
    """
    if not grids:
        raise ValueError("No grids provided")
    _assert_same_edges(grids)
    datasets = [grid_to_dataset(g.grid, g.timestamp) for g in grids]
    return xr.concat(datasets, dim="time")


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def _assert_same_edges(grids: Sequence[GridWithTime]) -> None:
    x0 = grids[0].grid.x_edges
    y0 = grids[0].grid.y_edges
    for g in grids[1:]:
        if not (np.array_equal(g.grid.x_edges, x0) and np.array_equal(g.grid.y_edges, y0)):
            raise ValueError("All grids must share identical x_edges and y_edges")
