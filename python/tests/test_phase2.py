import sys
import numpy as np
import pytest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase2  # noqa: E402


def test_bin_point_cloud_basic_stats() -> None:
    # Create a simple 2x2 grid of points with known z
    pts = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 1.0],
            [1.1, 0.1, 2.0],
            [1.2, 0.2, 2.0],
        ]
    )
    grid = phase2.bin_point_cloud(pts, bin_size=1.0, mode_bin=0.1)
    assert grid.z_mean.shape == (2, 2)
    # First bin (x<1, y<1) should have mean 1.0 and count 2
    np.testing.assert_allclose(grid.z_mean[0, 0], 1.0)
    np.testing.assert_allclose(grid.count[0, 0], 2)
    # Second bin (x>=1, y<1) should have mean 2.0
    np.testing.assert_allclose(grid.z_mean[1, 0], 2.0)
    assert not np.isnan(grid.z_mode[0, 0])


def test_compute_snr_handles_zero_count() -> None:
    z_mean = np.array([[1.0, 0.5], [0.0, 0.0]])
    z_std = np.array([[0.1, 0.2], [0.0, 0.0]])
    count = np.array([[10, 4], [0, 0]])
    snr = phase2.compute_snr(z_mean, z_std, count)
    assert np.isnan(snr[1, 0]) and np.isnan(snr[1, 1])
    assert snr[0, 0] > snr[0, 1]


def test_residual_kernel_filter_removes_outlier() -> None:
    # Points on plane z = 0 plus one outlier at z=5
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    outlier = np.array([[0.25, 0.25, 5.0]])
    pts = np.vstack([base, outlier])
    mask = phase2.residual_kernel_filter(pts, window_radius=0.8, max_residual=0.2, min_neighbors=4)
    assert mask.sum() == len(base)  # all plane points kept, outlier dropped
    assert not mask[-1]


def test_bin_point_cloud_temporal() -> None:
    pts = np.array(
        [
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.5],
            [0.1, 0.0, 2.0],
            [1.2, 0.0, 3.0],
        ]
    )
    intensities = np.array([10.0, 14.0, 20.0, 30.0])
    times = np.array([0.1, 0.2, 0.6, 1.1])

    grid = phase2.bin_point_cloud_temporal(
        pts, intensities, times, x_bin_size=1.0, time_bin_size=0.5
    )

    assert grid.z_mean.shape == (3, 2)  # 3 time bins, 2 x bins
    np.testing.assert_allclose(np.diff(grid.t_edges), 0.5)
    np.testing.assert_allclose(grid.z_mean[0, 0], 1.25)  # first time bin average z
    np.testing.assert_allclose(grid.z_mean[1, 0], 2.0)  # middle time bin single point
    np.testing.assert_allclose(grid.z_mean[2, 1], 3.0)  # last bin second x column
    assert np.isnan(grid.z_mean[2, 0]) and grid.count[2, 0] == 0  # empty cell normalized to NaN
    np.testing.assert_allclose(grid.intensity_mean[0, 0], 12.0)
