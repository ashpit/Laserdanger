"""Tests for internal functions and edge cases in phase3.py."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase2  # noqa: E402
import phase3  # noqa: E402


class TestBinCenters:
    """Tests for _bin_centers."""

    def test_basic_centers(self):
        """Compute centers from edges."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        centers = phase3._bin_centers(edges)
        np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])

    def test_uneven_edges(self):
        """Works with uneven bin spacing."""
        edges = np.array([0.0, 1.0, 4.0])
        centers = phase3._bin_centers(edges)
        np.testing.assert_allclose(centers, [0.5, 2.5])

    def test_two_edges(self):
        """Minimum case: two edges, one center."""
        edges = np.array([0.0, 1.0])
        centers = phase3._bin_centers(edges)
        np.testing.assert_allclose(centers, [0.5])


class TestAssertSameEdges:
    """Tests for _assert_same_edges validation."""

    def _make_grid(self, x_edges, y_edges):
        """Helper to create a minimal BinnedGrid."""
        nx, ny = len(x_edges) - 1, len(y_edges) - 1
        return phase2.BinnedGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            z_mean=np.zeros((nx, ny)),
            z_min=np.zeros((nx, ny)),
            z_max=np.zeros((nx, ny)),
            z_std=np.zeros((nx, ny)),
            z_mode=np.zeros((nx, ny)),
            count=np.zeros((nx, ny)),
            snr=np.zeros((nx, ny)),
            valid_mask=np.ones((nx, ny), dtype=bool),
        )

    def test_same_edges_passes(self):
        """Grids with same edges should pass validation."""
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges = np.array([0.0, 0.5, 1.0])

        grid1 = self._make_grid(x_edges, y_edges)
        grid2 = self._make_grid(x_edges.copy(), y_edges.copy())

        grids = [
            phase3.GridWithTime(grid=grid1, timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc)),
            phase3.GridWithTime(grid=grid2, timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]

        # Should not raise
        phase3._assert_same_edges(grids)

    def test_different_x_edges_raises(self):
        """Different x_edges should raise ValueError."""
        x_edges1 = np.array([0.0, 1.0, 2.0])
        x_edges2 = np.array([0.0, 1.0, 3.0])  # Different
        y_edges = np.array([0.0, 0.5, 1.0])

        grid1 = self._make_grid(x_edges1, y_edges)
        grid2 = self._make_grid(x_edges2, y_edges)

        grids = [
            phase3.GridWithTime(grid=grid1, timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc)),
            phase3.GridWithTime(grid=grid2, timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]

        with pytest.raises(ValueError, match="identical"):
            phase3._assert_same_edges(grids)

    def test_different_y_edges_raises(self):
        """Different y_edges should raise ValueError."""
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges1 = np.array([0.0, 0.5, 1.0])
        y_edges2 = np.array([0.0, 0.5, 2.0])  # Different

        grid1 = self._make_grid(x_edges, y_edges1)
        grid2 = self._make_grid(x_edges, y_edges2)

        grids = [
            phase3.GridWithTime(grid=grid1, timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc)),
            phase3.GridWithTime(grid=grid2, timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]

        with pytest.raises(ValueError, match="identical"):
            phase3._assert_same_edges(grids)


class TestGridToDataset:
    """Tests for grid_to_dataset conversion."""

    def _make_grid(self):
        """Create a minimal test BinnedGrid."""
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges = np.array([0.0, 1.0])
        nx, ny = 2, 1

        return phase2.BinnedGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            z_mean=np.array([[1.0], [2.0]]),
            z_min=np.array([[0.5], [1.5]]),
            z_max=np.array([[1.5], [2.5]]),
            z_std=np.array([[0.1], [0.2]]),
            z_mode=np.array([[1.0], [2.0]]),
            count=np.array([[100], [200]]),
            snr=np.array([[1000], [500]]),
            valid_mask=np.array([[True], [True]]),
        )

    def test_creates_dataset(self):
        """grid_to_dataset creates xarray Dataset."""
        grid = self._make_grid()
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ds = phase3.grid_to_dataset(grid, ts)

        assert "elevation" in ds
        assert "time" in ds.dims
        assert ds.sizes["time"] == 1

    def test_coordinates_are_centers(self):
        """Coordinates should be bin centers, not edges."""
        grid = self._make_grid()
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ds = phase3.grid_to_dataset(grid, ts)

        np.testing.assert_allclose(ds.x.values, [0.5, 1.5])
        np.testing.assert_allclose(ds.y.values, [0.5])

    def test_naive_timestamp_assumed_utc(self):
        """Naive timestamp should be treated as UTC."""
        grid = self._make_grid()
        naive_ts = datetime(2025, 1, 15, 12, 0, 0)  # No tzinfo
        ds = phase3.grid_to_dataset(grid, naive_ts)

        # Should not raise, and time should be set
        assert ds.sizes["time"] == 1


class TestBuildDataset:
    """Tests for build_dataset function."""

    def _make_grid(self, x_edges, y_edges, value):
        """Create a BinnedGrid with given edges and fill value."""
        nx, ny = len(x_edges) - 1, len(y_edges) - 1
        return phase2.BinnedGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            z_mean=np.full((nx, ny), value),
            z_min=np.full((nx, ny), value - 0.1),
            z_max=np.full((nx, ny), value + 0.1),
            z_std=np.full((nx, ny), 0.05),
            z_mode=np.full((nx, ny), value),
            count=np.full((nx, ny), 100),
            snr=np.full((nx, ny), 500),
            valid_mask=np.ones((nx, ny), dtype=bool),
        )

    def test_empty_grids_raises(self):
        """Empty grids list should raise ValueError."""
        with pytest.raises(ValueError, match="No grids"):
            phase3.build_dataset([])

    def test_single_grid(self):
        """Single grid should work."""
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges = np.array([0.0, 1.0])
        grid = self._make_grid(x_edges, y_edges, 1.0)

        grids = [phase3.GridWithTime(
            grid=grid,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc)
        )]

        ds = phase3.build_dataset(grids)
        assert ds.sizes["time"] == 1

    def test_multiple_grids_stacked(self):
        """Multiple grids should be stacked along time."""
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges = np.array([0.0, 1.0])

        grids = [
            phase3.GridWithTime(
                grid=self._make_grid(x_edges, y_edges, 1.0),
                timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            ),
            phase3.GridWithTime(
                grid=self._make_grid(x_edges, y_edges, 2.0),
                timestamp=datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
            ),
        ]

        ds = phase3.build_dataset(grids)
        assert ds.sizes["time"] == 2

        # Values should be different for each time
        np.testing.assert_allclose(ds.elevation.isel(time=0).values, 1.0)
        np.testing.assert_allclose(ds.elevation.isel(time=1).values, 2.0)


class TestTimeResolvedDataset:
    """Tests for TimeResolvedDataset class."""

    def _make_time_resolved_grid(self):
        """Create a minimal TimeResolvedGrid."""
        x_edges = np.array([0.0, 1.0, 2.0, 3.0])  # 3 x bins
        t_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # 4 time bins

        n_t, n_x = 4, 3
        return phase2.TimeResolvedGrid(
            x_edges=x_edges,
            t_edges=t_edges,
            z_mean=np.random.uniform(0, 2, (n_t, n_x)),
            z_min=np.random.uniform(-0.5, 1, (n_t, n_x)),
            z_max=np.random.uniform(1, 3, (n_t, n_x)),
            z_std=np.random.uniform(0, 0.5, (n_t, n_x)),
            count=np.random.randint(10, 100, (n_t, n_x)).astype(float),
            intensity_mean=np.random.uniform(10, 50, (n_t, n_x)),
        )

    def test_properties(self):
        """Test computed properties."""
        grid = self._make_time_resolved_grid()
        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        assert len(ds.x1d) == 3
        assert len(ds.time_vec) == 4
        assert ds.dt == pytest.approx(0.5)
        assert ds.dx == pytest.approx(1.0)
        assert ds.n_transects == 1

    def test_Z_xt_shape(self):
        """Z_xt should have shape (n_x, n_t)."""
        grid = self._make_time_resolved_grid()
        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        # grid.z_mean is (n_t, n_x), Z_xt should be transposed to (n_x, n_t)
        assert ds.Z_xt.shape == (3, 4)

    def test_Z_filtered_used_when_present(self):
        """When Z_filtered is present, Z_xt should return it."""
        grid = self._make_time_resolved_grid()
        Z_filtered = np.ones((3, 4)) * 99  # Distinctive value

        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            Z_filtered=Z_filtered,
        )

        np.testing.assert_array_equal(ds.Z_xt, Z_filtered)
        # Z_xt_raw should still be original
        assert not np.allclose(ds.Z_xt_raw, 99)

    def test_to_xarray(self):
        """to_xarray should produce valid Dataset."""
        grid = self._make_time_resolved_grid()
        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        xr_ds = ds.to_xarray()
        assert "elevation" in xr_ds
        assert "intensity" in xr_ds
        assert "x" in xr_ds.coords
        assert "time" in xr_ds.coords

    def test_to_xarray_with_outlier_mask(self):
        """Outlier mask should be included in xarray output."""
        grid = self._make_time_resolved_grid()
        outlier_mask = np.zeros((3, 4), dtype=bool)
        outlier_mask[0, 0] = True  # Mark one outlier

        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            outlier_mask=outlier_mask,
        )

        xr_ds = ds.to_xarray()
        assert "outlier_mask" in xr_ds
        assert xr_ds.outlier_mask.values[0, 0] == True

    def test_to_netcdf(self, tmp_path):
        """to_netcdf should save file successfully."""
        grid = self._make_time_resolved_grid()
        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        output_path = tmp_path / "test_l2.nc"
        ds.to_netcdf(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_multi_transect_properties(self):
        """Multi-transect dataset should report correct properties."""
        grid = self._make_time_resolved_grid()
        grid2 = self._make_time_resolved_grid()

        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            transect_grids={-2.0: grid, 0.0: grid, 2.0: grid2},
        )

        assert ds.n_transects == 3
        assert ds.alongshore_offsets == [-2.0, 0.0, 2.0]

    def test_get_transect(self):
        """get_transect should return correct grid."""
        grid = self._make_time_resolved_grid()
        grid2 = self._make_time_resolved_grid()

        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            transect_grids={0.0: grid, 2.0: grid2},
        )

        assert ds.get_transect(0.0) is grid
        assert ds.get_transect(2.0) is grid2
        assert ds.get_transect(99.0) is None
