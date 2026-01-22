"""Comprehensive tests for phase4.py - L2 processing, batch processing, and utilities."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase1  # noqa: E402
import phase2  # noqa: E402
import phase3  # noqa: E402
import phase4  # noqa: E402
import profiles  # noqa: E402


def fake_loader_factory(point_sets):
    """
    Returns a loader function that pops from point_sets per call.
    Each point_set should be a tuple of (points, intensities, gps_times) or just points array.
    """
    def _loader(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = point_sets.pop(0)
        if isinstance(data, tuple) and len(data) == 3:
            pts, intensities, gps = data
        else:
            pts = data
            intensities = np.full(len(pts), 10.0)
            gps = np.linspace(0.0, 100.0, len(pts))
        return pts, intensities, gps
    return _loader


def create_config_file(tmp_path: Path, boundary=None) -> Path:
    """Create a minimal config file for testing."""
    cfg_path = tmp_path / "cfg.json"
    if boundary is None:
        boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    payload = {
        "dataFolder": str(tmp_path),
        "processFolder": str(tmp_path / "proc"),
        "plotFolder": str(tmp_path / "plots"),
        "transformMatrix": np.eye(4).tolist(),
        "LidarBoundary": boundary,
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    return cfg_path


def create_laz_files(tmp_path: Path, timestamps: list) -> list:
    """Create stub LAZ files with given timestamps."""
    files = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            ts_posix = int(ts.timestamp())
        else:
            ts_posix = ts
        path = tmp_path / f"do-lidar_{ts_posix}.laz"
        path.write_text("stub", encoding="utf-8")
        files.append(path)
    return files


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_lidar_processing_error(self):
        """Base exception works."""
        with pytest.raises(phase4.LidarProcessingError):
            raise phase4.LidarProcessingError("test error")

    def test_corrupt_file_error(self):
        """CorruptFileError stores path and reason."""
        path = Path("/test/file.laz")
        err = phase4.CorruptFileError(path, "invalid header")

        assert err.path == path
        assert err.reason == "invalid header"
        assert "Corrupt" in str(err)
        assert "file.laz" in str(err)

    def test_no_data_error(self):
        """NoDataError with context."""
        err = phase4.NoDataError("all points filtered")
        assert "No valid data" in str(err)
        assert "filtered" in str(err)

    def test_configuration_error(self):
        """ConfigurationError is a LidarProcessingError."""
        err = phase4.ConfigurationError("missing key")
        assert isinstance(err, phase4.LidarProcessingError)


class TestCheckpoint:
    """Tests for Checkpoint class."""

    def test_checkpoint_creation(self):
        """Checkpoint creates with default timestamp."""
        cp = phase4.Checkpoint(
            config_path="/path/config.json",
            output_dir="/output",
            start_date="2025-01-01",
            end_date="2025-01-31",
            completed_dates=["2025-01-01", "2025-01-02"],
            failed_dates=["2025-01-03"],
            kwargs={"bin_size": 0.1},
        )

        assert cp.config_path == "/path/config.json"
        assert len(cp.completed_dates) == 2
        assert cp.timestamp  # Should be set

    def test_checkpoint_to_dict_roundtrip(self):
        """to_dict and from_dict should roundtrip."""
        cp = phase4.Checkpoint(
            config_path="/path/config.json",
            output_dir="/output",
            start_date="2025-01-01",
            end_date="2025-01-31",
            completed_dates=["2025-01-01"],
            failed_dates=[],
            kwargs={},
        )

        data = cp.to_dict()
        cp2 = phase4.Checkpoint.from_dict(data)

        assert cp2.config_path == cp.config_path
        assert cp2.completed_dates == cp.completed_dates

    def test_checkpoint_save_load(self, tmp_path):
        """Save and load checkpoint file."""
        cp = phase4.Checkpoint(
            config_path="/path/config.json",
            output_dir="/output",
            start_date="2025-01-01",
            end_date="2025-01-31",
            completed_dates=["2025-01-01"],
            failed_dates=[],
            kwargs={"test": True},
        )

        checkpoint_path = tmp_path / "checkpoint.json"
        cp.save(checkpoint_path)

        assert checkpoint_path.exists()

        cp2 = phase4.Checkpoint.load(checkpoint_path)
        assert cp2.config_path == cp.config_path
        assert cp2.kwargs["test"] == True


class TestBatchProgress:
    """Tests for BatchProgress class."""

    def test_success_rate_calculation(self):
        """success_rate computes correctly."""
        progress = phase4.BatchProgress(total_items=10)
        progress.completed = 7
        progress.failed = 3

        assert progress.success_rate == pytest.approx(0.7)

    def test_success_rate_zero_when_empty(self):
        """success_rate is 0 when no items processed."""
        progress = phase4.BatchProgress(total_items=10)
        assert progress.success_rate == 0.0

    def test_errors_list(self):
        """Errors list accumulates."""
        progress = phase4.BatchProgress(total_items=5)
        progress.errors.append(("2025-01-01", "File not found"))
        progress.errors.append(("2025-01-02", "Corrupt file"))

        assert len(progress.errors) == 2


class TestProcessL2:
    """Tests for process_l2 function."""

    def test_basic_l2_processing(self, tmp_path):
        """Basic L2 processing produces TimeResolvedDataset."""
        cfg_path = create_config_file(tmp_path)

        # Create LAZ file timestamps
        ts1 = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        create_laz_files(tmp_path, [ts1])

        # Create synthetic point data along x-axis
        n = 200
        pts = np.column_stack([
            np.random.uniform(1, 9, n),  # x
            np.random.uniform(1, 9, n),  # y
            np.random.uniform(0, 2, n),  # z
        ])
        intensities = np.random.uniform(10, 50, n)
        gps_times = np.linspace(0, 60, n)  # 60 seconds of data

        loader = fake_loader_factory([(pts, intensities, gps_times)])

        result = phase4.process_l2(
            cfg_path,
            time_bin_size=1.0,
            x_bin_size=0.5,
            loader=loader,
            apply_residual_filter=False,
            apply_outlier_detection=False,
            parallel_load=False,
        )

        assert isinstance(result, phase3.TimeResolvedDataset)
        assert result.Z_xt.shape[0] > 0  # Has x bins
        assert result.Z_xt.shape[1] > 0  # Has time bins

    def test_l2_with_outlier_detection(self, tmp_path):
        """L2 processing with outlier detection enabled."""
        cfg_path = create_config_file(tmp_path)
        ts1 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        create_laz_files(tmp_path, [ts1])

        n = 300
        pts = np.column_stack([
            np.random.uniform(1, 9, n),
            np.random.uniform(1, 9, n),
            np.random.uniform(0, 2, n),
        ])
        intensities = np.random.uniform(10, 50, n)
        gps_times = np.linspace(0, 120, n)

        loader = fake_loader_factory([(pts, intensities, gps_times)])

        result = phase4.process_l2(
            cfg_path,
            time_bin_size=2.0,
            x_bin_size=1.0,
            loader=loader,
            apply_residual_filter=False,
            apply_outlier_detection=True,
            parallel_load=False,
        )

        assert result.outlier_mask is not None
        assert result.Z_filtered is not None

    def test_l2_with_profile_config(self, tmp_path):
        """L2 processing with transect configuration."""
        cfg_path = create_config_file(tmp_path)
        ts1 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        create_laz_files(tmp_path, [ts1])

        n = 500
        pts = np.column_stack([
            np.random.uniform(1, 9, n),
            np.random.uniform(1, 9, n),
            np.random.uniform(0, 2, n),
        ])
        intensities = np.random.uniform(10, 50, n)
        gps_times = np.linspace(0, 120, n)

        loader = fake_loader_factory([(pts, intensities, gps_times)])

        profile_config = profiles.TransectConfig(
            x1=1.0, y1=5.0,
            x2=9.0, y2=5.0,
            tolerance=2.0,
        )

        result = phase4.process_l2(
            cfg_path,
            time_bin_size=2.0,
            x_bin_size=1.0,
            loader=loader,
            apply_residual_filter=False,
            apply_outlier_detection=False,
            profile_config=profile_config,
            parallel_load=False,
        )

        assert isinstance(result, phase3.TimeResolvedDataset)
        assert result.profile_config is profile_config

    def test_l2_raises_on_no_files(self, tmp_path):
        """L2 raises FileNotFoundError when no files found."""
        cfg_path = create_config_file(tmp_path)
        # No LAZ files created

        with pytest.raises(FileNotFoundError):
            phase4.process_l2(cfg_path, parallel_load=False)

    def test_l2_multiple_files(self, tmp_path):
        """L2 processing combines multiple files."""
        cfg_path = create_config_file(tmp_path)

        # Create multiple LAZ files
        ts1 = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        ts2 = int(datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc).timestamp())
        create_laz_files(tmp_path, [ts1, ts2])

        n = 100
        pts1 = np.column_stack([np.random.uniform(1, 9, n), np.random.uniform(1, 9, n), np.ones(n)])
        pts2 = np.column_stack([np.random.uniform(1, 9, n), np.random.uniform(1, 9, n), np.ones(n) * 2])

        loader = fake_loader_factory([pts1, pts2])

        result = phase4.process_l2(
            cfg_path,
            time_bin_size=1.0,
            x_bin_size=1.0,
            loader=loader,
            apply_residual_filter=False,
            apply_outlier_detection=False,
            parallel_load=False,
        )

        # Should have data from both files
        assert result.Z_xt.shape[1] > 0


class TestProcessL1Batch:
    """Tests for process_l1_batch function."""

    def test_batch_processes_multiple_days(self, tmp_path, monkeypatch):
        """Batch processing handles multiple days."""
        cfg_path = create_config_file(tmp_path)

        # Create files for 2 days
        day1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        day2 = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        create_laz_files(tmp_path, [day1, day2])

        n = 50
        pts1 = np.column_stack([np.random.uniform(1, 9, n), np.random.uniform(1, 9, n), np.ones(n)])
        pts2 = np.column_stack([np.random.uniform(1, 9, n), np.random.uniform(1, 9, n), np.ones(n) * 2])

        call_count = [0]
        def mock_load_laz_points(path, validate=True):
            idx = call_count[0]
            call_count[0] += 1
            pts = pts1 if idx == 0 else pts2
            return pts, np.full(len(pts), 10.0), np.linspace(0, 100, len(pts))

        monkeypatch.setattr(phase4, "load_laz_points", mock_load_laz_points)

        output_dir = tmp_path / "output"

        progress = phase4.process_l1_batch(
            cfg_path,
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 3),
            output_dir=output_dir,
            bin_size=1.0,
            apply_residual_filter=False,
            show_progress=False,
        )

        assert progress.completed >= 1  # At least some processed
        assert output_dir.exists()

    def test_batch_creates_checkpoint(self, tmp_path, monkeypatch):
        """Batch processing creates checkpoint file."""
        cfg_path = create_config_file(tmp_path)
        day1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        create_laz_files(tmp_path, [day1])

        n = 50
        pts = np.column_stack([np.random.uniform(1, 9, n), np.random.uniform(1, 9, n), np.ones(n)])

        def mock_load_laz_points(path, validate=True):
            return pts, np.full(len(pts), 10.0), np.linspace(0, 100, len(pts))

        monkeypatch.setattr(phase4, "load_laz_points", mock_load_laz_points)

        output_dir = tmp_path / "output"
        checkpoint_file = output_dir / "checkpoint.json"

        phase4.process_l1_batch(
            cfg_path,
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 2),
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
            apply_residual_filter=False,
            show_progress=False,
        )

        assert checkpoint_file.exists()

    def test_batch_resume_skips_completed(self, tmp_path, monkeypatch):
        """Resume skips already completed dates."""
        cfg_path = create_config_file(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create checkpoint with one date already completed
        checkpoint = phase4.Checkpoint(
            config_path=str(cfg_path),
            output_dir=str(output_dir),
            start_date="2025-01-01",
            end_date="2025-01-03",
            completed_dates=["2025-01-01"],
            failed_dates=[],
            kwargs={},
        )
        checkpoint_file = output_dir / "checkpoint.json"
        checkpoint.save(checkpoint_file)

        # Create LAZ files for day 2
        day2 = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        create_laz_files(tmp_path, [day2])

        n = 50
        pts = np.column_stack([np.random.uniform(1, 9, n), np.random.uniform(1, 9, n), np.ones(n)])

        def mock_load_laz_points(path, validate=True):
            return pts, np.full(len(pts), 10.0), np.linspace(0, 100, len(pts))

        monkeypatch.setattr(phase4, "load_laz_points", mock_load_laz_points)

        progress = phase4.process_l1_batch(
            cfg_path,
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 3),
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
            resume=True,
            apply_residual_filter=False,
            show_progress=False,
        )

        # Should count 2025-01-01 as already completed
        assert progress.completed >= 1


class TestProcessL2Batch:
    """Tests for process_l2_batch function."""

    def test_l2_batch_creates_output(self, tmp_path, monkeypatch):
        """L2 batch processing creates output files."""
        cfg_path = create_config_file(tmp_path)

        # Create LAZ file
        ts = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        create_laz_files(tmp_path, [ts])

        n = 200
        pts = np.column_stack([
            np.random.uniform(1, 9, n),
            np.random.uniform(1, 9, n),
            np.random.uniform(0, 2, n),
        ])
        intensities = np.random.uniform(10, 50, n)
        gps_times = np.linspace(0, 120, n)

        def mock_load_laz_points(path, validate=True):
            return pts, intensities, gps_times

        monkeypatch.setattr(phase4, "load_laz_points", mock_load_laz_points)

        output_dir = tmp_path / "l2_output"

        progress = phase4.process_l2_batch(
            cfg_path,
            start=datetime(2025, 1, 1, 12, 0, 0),
            end=datetime(2025, 1, 1, 12, 30, 0),
            output_dir=output_dir,
            file_duration=timedelta(minutes=30),
            apply_residual_filter=False,
            apply_outlier_detection=False,
            show_progress=False,
            parallel_load=False,
        )

        assert output_dir.exists()


class TestMemoryUtilities:
    """Tests for memory management utilities."""

    def test_estimate_memory_usage(self):
        """estimate_memory_usage returns reasonable value."""
        points = np.random.random((10000, 3))
        mb = phase4.estimate_memory_usage(points)

        # 10000 * 3 * 8 bytes = 240000 bytes ≈ 0.23 MB
        assert 0.2 < mb < 0.3

    def test_clear_memory_runs(self):
        """clear_memory should not raise."""
        # Just ensure it runs without error
        phase4.clear_memory()

    def test_chunked_file_iterator(self, tmp_path):
        """chunked_file_iterator groups files by memory."""
        # Create multiple small files
        files = []
        for i in range(5):
            path = tmp_path / f"file_{i}.laz"
            path.write_bytes(b"x" * 1000)  # ~1KB each
            ts = datetime(2025, 1, 1, i, 0, 0, tzinfo=timezone.utc)
            files.append((path, ts))

        # With high memory limit, should get one chunk
        chunks = list(phase4.chunked_file_iterator(files, max_memory_mb=1000.0))
        assert len(chunks) == 1
        assert chunks[0][1] == True  # is_last


class TestSaveDataset:
    """Tests for save_dataset function."""

    def test_save_creates_file(self, tmp_path):
        """save_dataset creates NetCDF file."""
        import xarray as xr

        ds = xr.Dataset({
            "temperature": (["x", "y"], np.random.random((5, 5))),
        })

        output_path = tmp_path / "test.nc"
        phase4.save_dataset(ds, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_dataset creates parent directories."""
        import xarray as xr

        ds = xr.Dataset({
            "temperature": (["x", "y"], np.random.random((5, 5))),
        })

        output_path = tmp_path / "nested" / "dirs" / "test.nc"
        phase4.save_dataset(ds, output_path)

        assert output_path.exists()


class TestExportProfilesJson:
    """Tests for export_profiles_json function."""

    def test_export_creates_json(self, tmp_path):
        """export_profiles_json creates JSON file."""
        # Create minimal L1Result with profiles
        x_edges = np.array([0, 1, 2, 3, 4])
        y_edges = np.array([0, 1])
        n_x, n_y = 4, 1

        grid = phase2.BinnedGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            z_mean=np.zeros((n_x, n_y)),
            z_min=np.zeros((n_x, n_y)),
            z_max=np.zeros((n_x, n_y)),
            z_std=np.zeros((n_x, n_y)),
            z_mode=np.zeros((n_x, n_y)),
            count=np.ones((n_x, n_y)) * 100,
            snr=np.ones((n_x, n_y)) * 500,
            valid_mask=np.ones((n_x, n_y), dtype=bool),
        )

        profile_config = profiles.TransectConfig(
            x1=0, y1=0, x2=4, y2=0,
            alongshore_spacings=[-1, 0, 1],
        )

        # TransectResult requires transect_coords
        x1d = np.array([0.5, 1.5, 2.5, 3.5])
        transect_coords = [
            (x1d, np.zeros_like(x1d) - 1),  # Transect at y=-1
            (x1d, np.zeros_like(x1d)),       # Transect at y=0
            (x1d, np.zeros_like(x1d) + 1),   # Transect at y=1
        ]

        profile_result = profiles.TransectResult(
            x1d=x1d,
            Z3D=np.random.random((3, 4)),
            transect_coords=transect_coords,
        )

        ds = phase3.grid_to_dataset(grid, datetime(2025, 1, 1, tzinfo=timezone.utc))

        result = phase4.L1Result(
            dataset=ds,
            profiles=profile_result,
            profile_config=profile_config,
        )

        output_path = tmp_path / "profiles.json"
        phase4.export_profiles_json(result, output_path)

        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            data = json.load(f)

        assert "profiles" in data
        assert len(data["profiles"]) == 3

    def test_export_without_profiles_raises(self, tmp_path):
        """export_profiles_json raises when no profiles."""
        import xarray as xr

        ds = xr.Dataset({"elevation": (["x", "y"], np.zeros((5, 5)))})
        result = phase4.L1Result(dataset=ds, profiles=None)

        with pytest.raises(ValueError, match="No profiles"):
            phase4.export_profiles_json(result, tmp_path / "profiles.json")


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_verbose_sets_debug_level(self):
        """Verbose mode sets DEBUG level."""
        import logging

        phase4.configure_logging(verbose=True)

        # Check handler level (not logger level, which is always DEBUG)
        root = logging.getLogger()
        assert any(h.level == logging.DEBUG for h in root.handlers)

    def test_quiet_sets_error_level(self):
        """Quiet mode sets ERROR level."""
        import logging

        phase4.configure_logging(quiet=True)

        root = logging.getLogger()
        assert any(h.level == logging.ERROR for h in root.handlers)

    def test_log_file_creation(self, tmp_path):
        """Log file is created when specified."""
        log_file = tmp_path / "test.log"
        phase4.configure_logging(log_file=log_file)

        # Trigger a log message
        import logging
        logging.getLogger().info("test message")

        assert log_file.exists()


class TestL1ResultClass:
    """Tests for L1Result dataclass."""

    def test_l1_result_creation(self):
        """L1Result can be created with minimal data."""
        import xarray as xr

        ds = xr.Dataset({"elevation": (["x", "y"], np.zeros((5, 5)))})
        result = phase4.L1Result(dataset=ds)

        assert result.dataset is ds
        assert result.profiles is None
        assert result.profile_config is None
