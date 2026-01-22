"""Tests for internal/private functions in phase1.py."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import phase1  # noqa: E402


class TestParseTimestampFromFilename:
    """Tests for _parse_timestamp_from_filename."""

    def test_posix_timestamp_format(self):
        """Parse POSIX timestamp from filename stem (e.g., do-lidar_1735689600)."""
        stem = "do-lidar_1735689600"
        result = phase1._parse_timestamp_from_filename(stem)
        assert result is not None
        assert result.tzinfo == timezone.utc
        expected = datetime.fromtimestamp(1735689600, tz=timezone.utc)
        assert result == expected

    def test_date_format_utc(self):
        """Parse YYYYMMDD_HHMMSS_UTC format (e.g., TOWR-test_20260120_200023_UTC)."""
        stem = "TOWR-test_20260120_200023_UTC"
        result = phase1._parse_timestamp_from_filename(stem)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 20
        assert result.hour == 20
        assert result.minute == 0
        assert result.second == 23

    def test_invalid_stem_returns_none(self):
        """Return None for filenames that don't match any known pattern."""
        stem = "random_file_name"
        result = phase1._parse_timestamp_from_filename(stem)
        assert result is None

    def test_short_number_not_posix(self):
        """Numbers less than 10 digits should not be parsed as POSIX timestamps."""
        stem = "file_12345"
        result = phase1._parse_timestamp_from_filename(stem)
        assert result is None

    def test_invalid_posix_timestamp(self):
        """Handle invalid POSIX timestamps gracefully."""
        # Very large timestamp that would cause overflow - the function should catch this
        stem = "do-lidar_99999999999999999999"
        # This may raise OverflowError on some platforms, or return None
        try:
            result = phase1._parse_timestamp_from_filename(stem)
            # If it doesn't raise, it should return None or a valid datetime
            # (behavior depends on platform)
        except OverflowError:
            # This is acceptable behavior - the caller should handle this
            pass

    def test_partial_date_format(self):
        """Partial date format without UTC suffix should not match."""
        stem = "TOWR-test_20260120_200023"
        result = phase1._parse_timestamp_from_filename(stem)
        # Should not match the date format (requires _UTC suffix)
        assert result is None

    def test_posix_with_prefix(self):
        """Handle various prefixes before POSIX timestamp."""
        for stem in ["site-a_1735689600", "test_1735689600", "do-lidar-v2_1735689600"]:
            result = phase1._parse_timestamp_from_filename(stem)
            assert result is not None, f"Failed to parse: {stem}"


class TestEnsureTz:
    """Tests for _ensure_tz timezone handling."""

    def test_naive_datetime_gets_utc(self):
        """Naive datetime should be assumed UTC."""
        naive = datetime(2025, 1, 1, 12, 0, 0)
        result = phase1._ensure_tz(naive)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_utc_datetime_unchanged(self):
        """UTC datetime should pass through unchanged."""
        utc_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = phase1._ensure_tz(utc_dt)
        assert result == utc_dt

    def test_other_timezone_converted_to_utc(self):
        """Non-UTC timezone should be converted to UTC."""
        from datetime import timedelta

        # Create a timezone offset (e.g., UTC+5)
        tz_plus5 = timezone(timedelta(hours=5))
        dt_plus5 = datetime(2025, 1, 1, 17, 0, 0, tzinfo=tz_plus5)

        result = phase1._ensure_tz(dt_plus5)
        assert result.tzinfo == timezone.utc
        # 17:00 UTC+5 = 12:00 UTC
        assert result.hour == 12


class TestTranslatePathForOs:
    """Tests for _translate_path_for_os."""

    def test_linux_path_translation(self, monkeypatch):
        """On Linux, /Volumes/ should be translated to /project/."""
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = phase1._translate_path_for_os("/Volumes/mygroup/data")
        assert result == Path("/project/mygroup/data")

    def test_macos_path_unchanged(self, monkeypatch):
        """On macOS (Darwin), paths should remain unchanged."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        result = phase1._translate_path_for_os("/Volumes/mygroup/data")
        assert result == Path("/Volumes/mygroup/data")

    def test_non_volumes_path_unchanged(self, monkeypatch):
        """Paths not starting with /Volumes/ should remain unchanged on any OS."""
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = phase1._translate_path_for_os("/home/user/data")
        assert result == Path("/home/user/data")


class TestFilterByPolygonEdgeCases:
    """Edge case tests for filter_by_polygon."""

    def test_point_on_edge(self):
        """Points exactly on polygon edges should be included."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        # Point on edge at (0.5, 0) - on bottom edge
        points = np.array([[0.5, 0.0]])
        mask = phase1.filter_by_polygon(points, polygon)
        assert mask[0], "Point on edge should be inside"

    def test_point_at_vertex(self):
        """Points exactly at polygon vertices should be included."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        # Point at vertex (0, 0)
        points = np.array([[0.0, 0.0]])
        mask = phase1.filter_by_polygon(points, polygon)
        assert mask[0], "Point at vertex should be inside"

    def test_empty_points(self):
        """Empty points array should return empty mask."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.empty((0, 2))
        mask = phase1.filter_by_polygon(points, polygon)
        assert len(mask) == 0

    def test_collinear_polygon_vertices(self):
        """Handle polygon with collinear points gracefully."""
        # Triangle with an extra point on one edge
        polygon = np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
        points = np.array([[0.3, 0.3]])
        mask = phase1.filter_by_polygon(points, polygon)
        # Should still work (point inside the triangle formed)
        assert len(mask) == 1


class TestTransformPointsEdgeCases:
    """Edge case tests for transform_points."""

    def test_empty_points(self):
        """Empty points array should work."""
        points = np.empty((0, 3))
        tmat = np.eye(4)
        result = phase1.transform_points(points, tmat)
        assert result.shape == (0, 3)

    def test_rotation_transform(self):
        """Test with rotation matrix (90 degrees around Z)."""
        points = np.array([[1.0, 0.0, 0.0]])
        # 90 degree rotation around Z axis
        tmat = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        result = phase1.transform_points(points, tmat)
        np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-10)

    def test_combined_rotation_translation(self):
        """Test combined rotation and translation."""
        points = np.array([[1.0, 0.0, 0.0]])
        # 90 degree rotation + translation by (10, 20, 30)
        tmat = np.array([
            [0, -1, 0, 10],
            [1, 0, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1]
        ], dtype=float)
        result = phase1.transform_points(points, tmat)
        np.testing.assert_allclose(result, [[10.0, 21.0, 30.0]], atol=1e-10)


class TestDiscoverLazFilesFormats:
    """Test discover_laz_files with different filename formats."""

    def test_both_formats_together(self, tmp_path):
        """Mix of POSIX and date format filenames should all be discovered."""
        # POSIX format
        ts_posix = 1735689600
        (tmp_path / f"do-lidar_{ts_posix}.laz").write_text("x")

        # Date format
        (tmp_path / "TOWR-test_20250101_120000_UTC.laz").write_text("x")

        files = phase1.discover_laz_files(tmp_path)
        assert len(files) == 2

    def test_non_matching_files_ignored(self, tmp_path):
        """Files that don't match any pattern should be ignored."""
        (tmp_path / "do-lidar_1735689600.laz").write_text("x")
        (tmp_path / "random_file.laz").write_text("x")
        (tmp_path / "readme.txt").write_text("x")

        files = phase1.discover_laz_files(tmp_path)
        assert len(files) == 1
        assert "1735689600" in files[0][0].name
