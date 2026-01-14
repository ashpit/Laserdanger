import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase1  # noqa: E402
import phase4  # noqa: E402


def fake_loader_factory(point_sets):
    """
    Returns a loader function that pops from point_sets per call.
    """
    def _loader(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = point_sets.pop(0)
        intensities = np.full(len(pts), 10.0)
        gps = np.linspace(1000.0, 1100.0, len(pts))
        return pts, intensities, gps
    return _loader


def test_process_l1_builds_dataset(tmp_path: Path) -> None:
    # Create config JSON
    cfg_path = tmp_path / "cfg.json"
    payload = {
        "dataFolder": str(tmp_path),
        "processFolder": str(tmp_path / "proc"),
        "plotFolder": str(tmp_path / "plots"),
        "transformMatrix": np.eye(4).tolist(),
        "LidarBoundary": [[0, 0], [2, 0], [2, 2], [0, 2]],
    }
    cfg_path.write_text(phase1.json.dumps(payload), encoding="utf-8")

    # Create synthetic laz filenames
    ts1 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    ts2 = int(datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc).timestamp())
    for ts in [ts1, ts2]:
        (tmp_path / f"do-lidar_{ts}.laz").write_text("stub", encoding="utf-8")

    # Two batches of points: centered in different bins
    pts1 = np.array([[0.1, 0.1, 1.0], [0.2, 0.2, 1.0]])
    pts2 = np.array([[1.1, 0.1, 2.0], [1.2, 0.2, 2.0]])
    loader = fake_loader_factory([pts1, pts2])

    ds = phase4.process_l1(cfg_path, bin_size=1.0, mode_bin=0.1, loader=loader)
    assert ds.sizes["time"] == 2
    # First time slice should reflect z=1 in first bin
    np.testing.assert_allclose(ds["elevation"].isel(time=0)[0, 0], 1.0)
    # Second slice should reflect z=2 in second bin
    np.testing.assert_allclose(ds["elevation"].isel(time=1)[0, 1], 2.0)


def test_process_l1_raises_on_no_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    payload = {
        "dataFolder": str(tmp_path),
        "processFolder": str(tmp_path / "proc"),
        "plotFolder": str(tmp_path / "plots"),
        "transformMatrix": np.eye(4).tolist(),
        "LidarBoundary": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }
    cfg_path.write_text(phase1.json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        phase4.process_l1(cfg_path, loader=fake_loader_factory([]))
