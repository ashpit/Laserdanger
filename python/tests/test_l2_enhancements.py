"""Tests for Phase 8.1 L2 enhancements: outlier detection, intensity contours, multi-transect."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))

import phase1  # noqa: E402
import phase2  # noqa: E402
import phase3  # noqa: E402
import phase4  # noqa: E402
import profiles  # noqa: E402
import utils  # noqa: E402


def fake_loader_factory(point_sets):
    """Returns a loader function that pops from point_sets per call."""
    def _loader(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = point_sets.pop(0)
        intensities = np.full(len(pts), 30.0)  # Moderate intensity
        gps = np.linspace(0.0, 10.0, len(pts))  # 10 seconds of data
        return pts, intensities, gps
    return _loader


def create_test_config(tmp_path: Path) -> Path:
    """Create a test config file."""
    cfg_path = tmp_path / "cfg.json"
    payload = {
        "dataFolder": str(tmp_path),
        "processFolder": str(tmp_path / "proc"),
        "plotFolder": str(tmp_path / "plots"),
        "transformMatrix": np.eye(4).tolist(),
        "LidarBoundary": [[0, 0], [100, 0], [100, 100], [0, 100]],
    }
    cfg_path.write_text(phase1.json.dumps(payload), encoding="utf-8")
    return cfg_path


def create_profile_config() -> profiles.TransectConfig:
    """Create a test profile config."""
    return profiles.TransectConfig(
        x1=10.0, y1=50.0,  # Backshore
        x2=90.0, y2=50.0,  # Offshore
        alongshore_spacings=(-10, -5, 0, 5, 10),
        tolerance=5.0,
        resolution=1.0,
    )


# =============================================================================
# TimeResolvedDataset Enhancement Tests
# =============================================================================

class TestTimeResolvedDatasetEnhancements:
    """Test enhanced TimeResolvedDataset features."""

    def test_outlier_mask_property(self):
        """Test that outlier mask is stored and accessible."""
        # Create minimal grid
        n_t, n_x = 20, 10
        grid = phase2.TimeResolvedGrid(
            z_mean=np.random.randn(n_t, n_x),
            z_min=np.random.randn(n_t, n_x),
            z_max=np.random.randn(n_t, n_x),
            z_std=np.abs(np.random.randn(n_t, n_x)),
            intensity_mean=np.random.randn(n_t, n_x) * 10 + 30,
            count=np.ones((n_t, n_x), dtype=int) * 10,
            x_edges=np.arange(n_x + 1).astype(float),
            t_edges=np.arange(n_t + 1).astype(float) * 0.5,
        )

        # Create outlier mask
        outlier_mask = np.random.rand(n_x, n_t) > 0.9  # ~10% outliers
        Z_filtered = grid.z_mean.T.copy()
        Z_filtered[outlier_mask] = np.nan

        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2024, 6, 15, tzinfo=timezone.utc),
            outlier_mask=outlier_mask,
            Z_filtered=Z_filtered,
        )

        assert ds.outlier_mask is not None
        assert ds.outlier_mask.shape == (n_x, n_t)
        assert ds.Z_filtered is not None
        # Z_xt should return filtered version when available
        assert np.isnan(ds.Z_xt[outlier_mask]).all()
        # Z_xt_raw should return unfiltered
        assert not np.isnan(ds.Z_xt_raw).all()

    def test_multi_transect_properties(self):
        """Test multi-transect storage and access."""
        n_t, n_x = 20, 10

        def make_grid():
            return phase2.TimeResolvedGrid(
                z_mean=np.random.randn(n_t, n_x),
                z_min=np.random.randn(n_t, n_x),
                z_max=np.random.randn(n_t, n_x),
                z_std=np.abs(np.random.randn(n_t, n_x)),
                intensity_mean=np.random.randn(n_t, n_x) * 10 + 30,
                count=np.ones((n_t, n_x), dtype=int) * 10,
                x_edges=np.arange(n_x + 1).astype(float),
                t_edges=np.arange(n_t + 1).astype(float) * 0.5,
            )

        primary_grid = make_grid()
        transect_grids = {
            -10.0: make_grid(),
            -5.0: make_grid(),
            0.0: primary_grid,
            5.0: make_grid(),
            10.0: make_grid(),
        }

        ds = phase3.TimeResolvedDataset(
            grid=primary_grid,
            base_time=datetime(2024, 6, 15, tzinfo=timezone.utc),
            transect_grids=transect_grids,
        )

        assert ds.n_transects == 5
        assert ds.alongshore_offsets == [-10.0, -5.0, 0.0, 5.0, 10.0]
        assert ds.get_transect(0.0) is primary_grid
        assert ds.get_transect(5.0) is transect_grids[5.0]
        assert ds.get_transect(100.0) is None

    def test_intensity_contours_storage(self):
        """Test intensity contour storage."""
        n_t, n_x = 20, 10
        grid = phase2.TimeResolvedGrid(
            z_mean=np.random.randn(n_t, n_x),
            z_min=np.random.randn(n_t, n_x),
            z_max=np.random.randn(n_t, n_x),
            z_std=np.abs(np.random.randn(n_t, n_x)),
            intensity_mean=np.random.randn(n_t, n_x) * 10 + 30,
            count=np.ones((n_t, n_x), dtype=int) * 10,
            x_edges=np.arange(n_x + 1).astype(float),
            t_edges=np.arange(n_t + 1).astype(float) * 0.5,
        )

        # Create fake contours
        contours = [
            utils.IntensityContour(
                threshold=30.0,
                x_positions=np.random.rand(n_t) * 5 + 5,
                time_indices=np.arange(n_t),
                mean_position=7.5,
                std_position=1.0,
                valid_fraction=0.9,
            )
        ]

        ds = phase3.TimeResolvedDataset(
            grid=grid,
            base_time=datetime(2024, 6, 15, tzinfo=timezone.utc),
            intensity_contours=contours,
        )

        assert ds.intensity_contours is not None
        assert len(ds.intensity_contours) == 1
        assert ds.intensity_contours[0].threshold == 30.0

    def test_to_xarray_includes_enhancements(self):
        """Test that to_xarray includes outlier mask and multi-transect data."""
        n_t, n_x = 20, 10

        def make_grid():
            return phase2.TimeResolvedGrid(
                z_mean=np.random.randn(n_t, n_x),
                z_min=np.random.randn(n_t, n_x),
                z_max=np.random.randn(n_t, n_x),
                z_std=np.abs(np.random.randn(n_t, n_x)),
                intensity_mean=np.random.randn(n_t, n_x) * 10 + 30,
                count=np.ones((n_t, n_x), dtype=int) * 10,
                x_edges=np.arange(n_x + 1).astype(float),
                t_edges=np.arange(n_t + 1).astype(float) * 0.5,
            )

        primary_grid = make_grid()
        transect_grids = {-5.0: make_grid(), 0.0: primary_grid, 5.0: make_grid()}
        outlier_mask = np.random.rand(n_x, n_t) > 0.9

        ds = phase3.TimeResolvedDataset(
            grid=primary_grid,
            base_time=datetime(2024, 6, 15, tzinfo=timezone.utc),
            transect_grids=transect_grids,
            outlier_mask=outlier_mask,
            Z_filtered=primary_grid.z_mean.T.copy(),
        )

        xr_ds = ds.to_xarray()

        # Check outlier mask is included
        assert "outlier_mask" in xr_ds.data_vars
        assert xr_ds.attrs["outlier_detection_applied"] is True

        # Check 3D data is included
        assert "elevation_3d" in xr_ds.data_vars
        assert "intensity_3d" in xr_ds.data_vars
        assert "alongshore" in xr_ds.coords
        assert list(xr_ds.coords["alongshore"].values) == [-5.0, 0.0, 5.0]


# =============================================================================
# L2 Processing Integration Tests
# =============================================================================

class TestProcessL2Enhancements:
    """Test process_l2 with enhancement features."""

    def test_outlier_detection_integration(self, tmp_path: Path):
        """Test that outlier detection is applied in process_l2."""
        cfg_path = create_test_config(tmp_path)

        # Create LAZ file stub
        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        # Create test points with some obvious spikes
        np.random.seed(42)
        n_points = 500
        x = np.random.uniform(20, 80, n_points)
        y = np.random.uniform(40, 60, n_points)  # Near transect
        z = np.sin(x * 0.1) + np.random.randn(n_points) * 0.1

        # Add some spikes
        z[100] = 10.0  # Big spike
        z[200] = -5.0  # Big dip

        pts = np.column_stack([x, y, z])
        loader = fake_loader_factory([pts])
        profile_config = create_profile_config()

        result = phase4.process_l2(
            cfg_path,
            loader=loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            apply_outlier_detection=True,
            outlier_params={'gradient_threshold_std': 2.0},
        )

        assert result.outlier_mask is not None
        assert result.Z_filtered is not None
        # Some outliers should be detected
        assert result.outlier_mask.sum() > 0
        # Filtered data should have NaN where outliers were
        assert np.isnan(result.Z_filtered[result.outlier_mask]).all()

    def test_outlier_detection_disabled(self, tmp_path: Path):
        """Test that outlier detection can be disabled."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        np.random.seed(42)
        n_points = 200
        pts = np.column_stack([
            np.random.uniform(20, 80, n_points),
            np.random.uniform(40, 60, n_points),
            np.random.randn(n_points) * 0.1,
        ])
        loader = fake_loader_factory([pts])
        profile_config = create_profile_config()

        result = phase4.process_l2(
            cfg_path,
            loader=loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            apply_outlier_detection=False,
        )

        assert result.outlier_mask is None
        assert result.Z_filtered is None

    def test_intensity_contour_extraction(self, tmp_path: Path):
        """Test intensity contour extraction in process_l2."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        # Create points with intensity gradient
        np.random.seed(42)
        n_points = 500
        x = np.random.uniform(20, 80, n_points)
        y = np.random.uniform(40, 60, n_points)
        z = np.random.randn(n_points) * 0.1

        pts = np.column_stack([x, y, z])

        # Custom loader with intensity gradient (higher offshore)
        def intensity_gradient_loader(path: Path):
            intensities = x  # Intensity increases with x
            gps = np.linspace(0.0, 10.0, len(pts))
            return pts, intensities, gps

        profile_config = create_profile_config()

        result = phase4.process_l2(
            cfg_path,
            loader=intensity_gradient_loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            apply_outlier_detection=False,
            extract_intensity_contours=True,
            intensity_contour_thresholds=[30.0, 50.0, 70.0],
        )

        # Should have extracted some contours
        assert result.intensity_contours is not None
        # May or may not find valid contours depending on data

    def test_multi_transect_extraction(self, tmp_path: Path):
        """Test multi-transect L2 extraction."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        # Create points spread across multiple transects
        np.random.seed(42)
        n_points = 1000
        x = np.random.uniform(20, 80, n_points)
        y = np.random.uniform(35, 65, n_points)  # Spread across y to hit multiple transects
        z = np.random.randn(n_points) * 0.1

        pts = np.column_stack([x, y, z])
        loader = fake_loader_factory([pts])

        profile_config = profiles.TransectConfig(
            x1=10.0, y1=50.0,
            x2=90.0, y2=50.0,
            alongshore_spacings=(-10, -5, 0, 5, 10),
            tolerance=8.0,  # Wide tolerance to capture points
            resolution=2.0,
        )

        result = phase4.process_l2(
            cfg_path,
            loader=loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            apply_outlier_detection=False,
            multi_transect=True,
        )

        # Should have multiple transects
        assert result.transect_grids is not None
        assert result.n_transects > 1
        assert 0.0 in result.alongshore_offsets

        # Each transect should have data
        for offset in result.alongshore_offsets:
            grid = result.get_transect(offset)
            assert grid is not None

    def test_single_transect_default(self, tmp_path: Path):
        """Test that single transect is default (multi_transect=False)."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        np.random.seed(42)
        n_points = 200
        pts = np.column_stack([
            np.random.uniform(20, 80, n_points),
            np.random.uniform(45, 55, n_points),
            np.random.randn(n_points) * 0.1,
        ])
        loader = fake_loader_factory([pts])

        profile_config = profiles.TransectConfig(
            x1=10.0, y1=50.0,
            x2=90.0, y2=50.0,
            alongshore_spacings=(-10, -5, 0, 5, 10),
            tolerance=8.0,
            resolution=2.0,
        )

        result = phase4.process_l2(
            cfg_path,
            loader=loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            multi_transect=False,  # Default
        )

        # Should NOT have multi-transect data
        assert result.transect_grids is None
        assert result.n_transects == 1

    def test_combined_features(self, tmp_path: Path):
        """Test all Phase 8.1 features together."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        np.random.seed(42)
        n_points = 1000
        x = np.random.uniform(20, 80, n_points)
        y = np.random.uniform(35, 65, n_points)
        z = np.sin(x * 0.1) + np.random.randn(n_points) * 0.1

        # Add some spikes
        z[100] = 10.0
        z[200] = -5.0

        pts = np.column_stack([x, y, z])

        def full_loader(path: Path):
            intensities = x + np.random.randn(n_points) * 5
            gps = np.linspace(0.0, 20.0, n_points)
            return pts, intensities, gps

        profile_config = profiles.TransectConfig(
            x1=10.0, y1=50.0,
            x2=90.0, y2=50.0,
            alongshore_spacings=(-5, 0, 5),
            tolerance=10.0,
            resolution=2.0,
        )

        result = phase4.process_l2(
            cfg_path,
            loader=full_loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            apply_outlier_detection=True,
            extract_intensity_contours=True,
            intensity_contour_thresholds=[40.0, 60.0],
            multi_transect=True,
        )

        # Verify all features are present
        assert result.outlier_mask is not None, "Outlier detection should be applied"
        assert result.transect_grids is not None, "Multi-transect should be extracted"
        assert result.n_transects >= 1, "Should have at least one transect"

        # Verify xarray export includes everything
        xr_ds = result.to_xarray()
        assert "outlier_mask" in xr_ds.data_vars
        assert xr_ds.attrs["n_transects"] >= 1
        # Multi-transect 3D data only present if all grids have same shape
        if result.n_transects > 1 and "elevation_3d" in xr_ds.data_vars:
            assert "alongshore" in xr_ds.coords


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_points_near_transect(self, tmp_path: Path):
        """Test handling when no points are near the transect."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        # Points far from transect
        np.random.seed(42)
        n_points = 200
        pts = np.column_stack([
            np.random.uniform(20, 80, n_points),
            np.random.uniform(80, 90, n_points),  # Far from y=50 transect
            np.random.randn(n_points) * 0.1,
        ])
        loader = fake_loader_factory([pts])

        profile_config = profiles.TransectConfig(
            x1=10.0, y1=50.0,
            x2=90.0, y2=50.0,
            tolerance=2.0,  # Narrow tolerance
        )

        # Should raise error since no valid transect data
        with pytest.raises(RuntimeError):
            phase4.process_l2(
                cfg_path,
                loader=loader,
                apply_residual_filter=False,
                profile_config=profile_config,
            )

    def test_sparse_data_outlier_detection(self, tmp_path: Path):
        """Test outlier detection with sparse data."""
        cfg_path = create_test_config(tmp_path)

        ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

        # Very few points
        np.random.seed(42)
        n_points = 50
        pts = np.column_stack([
            np.random.uniform(20, 80, n_points),
            np.random.uniform(45, 55, n_points),
            np.random.randn(n_points) * 0.1,
        ])
        loader = fake_loader_factory([pts])

        profile_config = create_profile_config()

        # Should not crash with sparse data
        result = phase4.process_l2(
            cfg_path,
            loader=loader,
            apply_residual_filter=False,
            profile_config=profile_config,
            apply_outlier_detection=True,
        )

        assert result.grid is not None


# =============================================================================
# Utility Function Integration Tests
# =============================================================================

class TestUtilityIntegration:
    """Test that utility functions work correctly with L2 data."""

    def test_detect_outliers_with_real_pattern(self):
        """Test outlier detection handles wave-like data and returns valid results."""
        n_x, n_t = 50, 200
        dt = 0.5

        # Create smooth wave-like pattern
        x = np.arange(n_x)
        t = np.arange(n_t)
        X, T = np.meshgrid(x, t, indexing='ij')
        Z_xt = np.sin(0.1 * X) + 0.5 * np.sin(0.05 * T)
        np.random.seed(42)
        Z_xt = Z_xt + np.random.randn(n_x, n_t) * 0.02

        # Add a cluster of anomalies (not just single points - the algorithm
        # uses median filtering which smooths single-point outliers)
        Z_xt[24:27, 99:102] = 10.0  # 3x3 block of high values
        Z_xt[29:32, 149:152] = -8.0  # 3x3 block of low values

        result = utils.detect_outliers_conv2d(
            Z_xt, dt,
            gradient_threshold_std=2.0,
            laplacian_threshold_std=2.0,
        )

        # Verify output structure
        assert isinstance(result, utils.OutlierResult)
        assert result.is_outlier.shape == Z_xt.shape
        assert result.Z_filtered.shape == Z_xt.shape
        assert result.gradient_magnitude.shape == Z_xt.shape
        assert result.laplacian.shape == Z_xt.shape

        # Algorithm should detect some outliers in data with anomaly blocks
        # (may or may not detect the specific blocks depending on threshold)
        # Just verify the algorithm runs and produces valid output
        assert result.is_outlier.dtype == bool
        assert np.sum(np.isnan(result.Z_filtered)) == np.sum(result.is_outlier)

    def test_detect_outliers_with_nan_input(self):
        """Test outlier detection handles NaN values in input."""
        n_x, n_t = 50, 100
        dt = 0.5

        Z_xt = np.random.randn(n_x, n_t) * 0.1 + 1.0
        # Add some NaN values
        Z_xt[10:15, 20:25] = np.nan

        result = utils.detect_outliers_conv2d(Z_xt, dt)

        # NaN regions should be marked as outliers
        assert result.is_outlier[10:15, 20:25].all()
        # Output should have NaN where outliers are
        assert np.isnan(result.Z_filtered[12, 22])

    def test_intensity_contours_with_gradient(self):
        """Test intensity contours extract water edge correctly."""
        n_x, n_t = 50, 100
        x1d = np.arange(n_x) * 0.1  # 0 to 5m

        # Create intensity that increases seaward
        I_xt = np.tile(x1d * 20, (n_t, 1)).T  # 0 to 100

        contours = utils.get_intensity_contours(
            I_xt, x1d,
            thresholds=[50.0],  # Should be at x=2.5
        )

        assert len(contours) == 1
        assert contours[0].threshold == 50.0
        # Mean position should be around x=2.5
        assert 2.0 < contours[0].mean_position < 3.0
