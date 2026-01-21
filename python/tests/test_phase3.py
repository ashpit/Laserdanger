import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase2  # noqa: E402
import phase3  # noqa: E402


def make_grid(z_value: float) -> phase2.BinnedGrid:
    x_edges = np.array([0.0, 1.0, 2.0])
    y_edges = np.array([0.0, 1.0, 2.0])
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    z = np.full(shape, z_value)
    return phase2.BinnedGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        z_mean=z,
        z_min=z - 0.1,
        z_max=z + 0.1,
        z_std=np.zeros(shape),
        z_mode=z,
        count=np.ones(shape),
        snr=np.full(shape, 10.0),
        valid_mask=np.ones(shape, dtype=bool),
    )


def test_grid_to_dataset_shapes_and_coords() -> None:
    grid = make_grid(1.0)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ds = phase3.grid_to_dataset(grid, ts)
    assert ds.sizes["time"] == 1
    assert ds.sizes["x"] == 2 and ds.sizes["y"] == 2
    np.testing.assert_allclose(ds.coords["x"], [0.5, 1.5])
    np.testing.assert_allclose(ds.coords["y"], [0.5, 1.5])
    np.testing.assert_allclose(ds["elevation"].sel(time=ts, method="nearest"), grid.z_mean.T)


def test_build_dataset_stacks_time() -> None:
    g1 = phase3.GridWithTime(grid=make_grid(1.0), timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
    g2 = phase3.GridWithTime(
        grid=make_grid(2.0), timestamp=datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc)
    )
    ds = phase3.build_dataset([g1, g2])
    assert ds.sizes["time"] == 2
    np.testing.assert_allclose(ds["elevation"].isel(time=0), g1.grid.z_mean.T)
    np.testing.assert_allclose(ds["elevation"].isel(time=1), g2.grid.z_mean.T)


def test_build_dataset_requires_matching_edges() -> None:
    g1 = phase3.GridWithTime(grid=make_grid(1.0), timestamp=datetime.now(timezone.utc))
    g2_grid = make_grid(2.0)
    g2_grid = g2_grid.__class__(
        x_edges=np.array([0.0, 1.0, 3.0]),  # different edges
        y_edges=g2_grid.y_edges,
        z_mean=g2_grid.z_mean,
        z_min=g2_grid.z_min,
        z_max=g2_grid.z_max,
        z_std=g2_grid.z_std,
        z_mode=g2_grid.z_mode,
        count=g2_grid.count,
        snr=g2_grid.snr,
        valid_mask=g2_grid.valid_mask,
    )
    g2 = phase3.GridWithTime(grid=g2_grid, timestamp=datetime.now(timezone.utc) + timedelta(minutes=5))
    with pytest.raises(ValueError):
        phase3.build_dataset([g1, g2])
