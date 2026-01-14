import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase1  # noqa: E402


def test_load_config_valid(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    payload = {
        "dataFolder": "/data",
        "processFolder": "/proc",
        "plotFolder": "/plots",
        "transformMatrix": np.eye(4).tolist(),
        "LidarBoundary": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = phase1.load_config(cfg_path)
    assert cfg.data_folder == Path("/data")
    assert cfg.transform_matrix.shape == (4, 4)
    assert cfg.lidar_boundary.shape == (4, 2)


def test_load_config_missing_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"dataFolder": "/data"}), encoding="utf-8")
    with pytest.raises(ValueError):
        phase1.load_config(cfg_path)


def test_discover_laz_files_filters_and_sorts(tmp_path: Path) -> None:
    times = [1_000_000_100, 1_000_000_000, 1_000_000_050]
    for ts in times:
        (tmp_path / f"do-lidar_{ts}.laz").write_text("x")
    start = datetime.fromtimestamp(1_000_000_050, tz=timezone.utc)
    files = phase1.discover_laz_files(tmp_path, start=start)
    assert [p.name for p, _ in files] == ["do-lidar_1000000050.laz", "do-lidar_1000000100.laz"]
    assert files[0][1].tzinfo == timezone.utc


def test_transform_points_applies_homogeneous_matrix() -> None:
    points = np.array([[1, 2, 3]])
    tmat = np.eye(4)
    tmat[:3, 3] = [10, 0, -5]
    out = phase1.transform_points(points, tmat)
    np.testing.assert_allclose(out, [[11, 2, -2]])


def test_filter_by_polygon_and_filters() -> None:
    points = np.array(
        [
            [0.1, 0.1, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
        ]
    )
    intensities = np.array([10, 200, 5])
    times = np.array([0.0, 1.0, 2.0])
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mask = phase1.filter_points(points, intensities, times, polygon, intensity_threshold=100.0)
    assert mask.tolist() == [True, False, False]


def test_prepare_batch_transforms_and_filters() -> None:
    raw_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.5, 0.0]])
    intensities = np.array([5.0, 10.0, 20.0])
    gps_times = np.array([1_000.0, 1_100.0, 1_400.0])
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    tmat = np.eye(4)
    cfg = phase1.Config(
        data_folder=Path("/d"),
        process_folder=Path("/p"),
        plot_folder=Path("/o"),
        transform_matrix=tmat,
        lidar_boundary=polygon,
    )
    filtered_points, filtered_intensity, rel_times = phase1.prepare_batch(
        raw_points, intensities, gps_times, cfg, max_seconds=150.0
    )
    # First two points are inside polygon; third excluded by polygon + time window
    assert filtered_points.shape[0] == 2
    np.testing.assert_allclose(rel_times, [0.0, 100.0])
