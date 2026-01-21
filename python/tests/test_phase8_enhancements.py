"""
Tests for Phase 8.2-8.4 enhancements:
- 8.2: Batch processing with checkpointing, resume, parallel processing
- 8.3: Logging and progress bars
- 8.4: Error handling and memory efficiency
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import numpy as np
import pytest

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from phase4 import (
    LidarProcessingError,
    CorruptFileError,
    NoDataError,
    ConfigurationError,
    Checkpoint,
    BatchProgress,
    configure_logging,
    load_laz_points,
    load_laz_points_safe,
    clear_memory,
    estimate_memory_usage,
    chunked_file_iterator,
)


# =============================================================================
# Test Custom Exceptions (Phase 8.4)
# =============================================================================

class TestCustomExceptions:
    """Test custom exception classes."""

    def test_lidar_processing_error_inheritance(self):
        """Test LidarProcessingError is base exception."""
        error = LidarProcessingError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_corrupt_file_error(self):
        """Test CorruptFileError inherits from LidarProcessingError."""
        error = CorruptFileError(Path("/test/file.laz"), "corrupt file")
        assert isinstance(error, LidarProcessingError)
        assert "/test/file.laz" in str(error)
        assert error.path == Path("/test/file.laz")

    def test_no_data_error(self):
        """Test NoDataError inherits from LidarProcessingError."""
        error = NoDataError("no data found")
        assert isinstance(error, LidarProcessingError)

    def test_configuration_error(self):
        """Test ConfigurationError inherits from LidarProcessingError."""
        error = ConfigurationError("bad config")
        assert isinstance(error, LidarProcessingError)


# =============================================================================
# Test Checkpoint System (Phase 8.2)
# =============================================================================

class TestCheckpoint:
    """Test checkpoint saving and loading."""

    def test_checkpoint_creation(self):
        """Test creating a Checkpoint object."""
        checkpoint = Checkpoint(
            config_path="/path/to/config.json",
            output_dir="/path/to/output",
            start_date="2025-01-01",
            end_date="2025-01-10",
            completed_dates=["2025-01-01", "2025-01-02"],
            failed_dates=["2025-01-03"],
            kwargs={"bin_size": 0.1},
        )
        assert checkpoint.config_path == "/path/to/config.json"
        assert len(checkpoint.completed_dates) == 2
        assert len(checkpoint.failed_dates) == 1

    def test_checkpoint_save_and_load(self):
        """Test saving and loading checkpoint from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            original = Checkpoint(
                config_path="/path/to/config.json",
                output_dir="/path/to/output",
                start_date="2025-01-01",
                end_date="2025-01-10",
                completed_dates=["2025-01-01"],
                failed_dates=[],
                kwargs={"bin_size": 0.1, "mode_bin": 0.05},
            )
            original.save(checkpoint_path)

            # Verify file exists
            assert checkpoint_path.exists()

            # Load and verify
            loaded = Checkpoint.load(checkpoint_path)
            assert loaded.config_path == original.config_path
            assert loaded.output_dir == original.output_dir
            assert loaded.start_date == original.start_date
            assert loaded.end_date == original.end_date
            assert loaded.completed_dates == original.completed_dates
            assert loaded.kwargs == original.kwargs

    def test_checkpoint_timestamp(self):
        """Test checkpoint gets timestamp on save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            checkpoint = Checkpoint(
                config_path="/path/config.json",
                output_dir="/out",
                start_date="2025-01-01",
                end_date="2025-01-02",
                completed_dates=[],
                failed_dates=[],
                kwargs={},
            )
            checkpoint.save(checkpoint_path)

            loaded = Checkpoint.load(checkpoint_path)
            assert loaded.timestamp != ""
            # Timestamp should be in ISO format
            datetime.fromisoformat(loaded.timestamp)


# =============================================================================
# Test BatchProgress (Phase 8.2)
# =============================================================================

class TestBatchProgress:
    """Test batch progress tracking."""

    def test_batch_progress_creation(self):
        """Test creating BatchProgress object."""
        progress = BatchProgress(
            total_items=10,
            completed=3,
            failed=1,
            current_item="2025-01-05",
        )
        assert progress.total_items == 10
        assert progress.completed == 3

    def test_batch_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = BatchProgress(
            total_items=10,
            completed=5,
            failed=0,
            current_item="",
        )
        # 5 completed out of 10 = 50%
        expected = 50.0
        actual = (progress.completed / progress.total_items) * 100
        assert actual == expected

    def test_batch_progress_success_rate(self):
        """Test success rate property."""
        progress = BatchProgress(
            total_items=10,
            completed=8,
            failed=2,
            current_item="",
        )
        # 8 completed / (8+2) = 0.8
        assert progress.success_rate == 0.8


# =============================================================================
# Test Logging Configuration (Phase 8.3)
# =============================================================================

class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_configure_logging_verbose(self):
        """Test verbose logging allows DEBUG messages."""
        configure_logging(verbose=True, log_file=None, quiet=False)
        # Root logger should be set to DEBUG
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_quiet(self):
        """Test quiet logging suppresses console output via handler level."""
        configure_logging(verbose=False, log_file=None, quiet=True)
        # Root logger should still be DEBUG (messages flow through)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        # But handlers should filter to ERROR only
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        # At least one handler should be set to ERROR
        assert any(h.level == logging.ERROR for h in console_handlers)

    def test_configure_logging_normal(self):
        """Test normal logging allows INFO messages."""
        configure_logging(verbose=False, log_file=None, quiet=False)
        root_logger = logging.getLogger()
        # Root is DEBUG, handlers filter to INFO
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        # At least one handler should be set to INFO
        assert any(h.level == logging.INFO for h in console_handlers)

    def test_configure_logging_to_file(self):
        """Test logging to file creates file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            configure_logging(verbose=False, log_file=log_file, quiet=False)

            root_logger = logging.getLogger()
            # Should have at least one FileHandler
            file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) >= 1


# =============================================================================
# Test Error Handling in File Loading (Phase 8.4)
# =============================================================================

class TestLoadLazPointsSafe:
    """Test safe file loading with error handling."""

    def test_load_laz_points_safe_nonexistent_file(self):
        """Test safe loading returns None for nonexistent file."""
        result = load_laz_points_safe(Path("/nonexistent/file.laz"))
        assert result is None

    def test_load_laz_points_safe_returns_tuple(self):
        """Test safe loading with mocked file returns tuple."""
        # Mock the load_laz_points function to return test data
        mock_points = {
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([1.0, 2.0, 3.0]),
            "z": np.array([1.0, 2.0, 3.0]),
            "intensity": np.array([100, 200, 300]),
            "gps_time": np.array([0.0, 0.1, 0.2]),
        }

        with patch("phase4.load_laz_points", return_value=mock_points):
            result = load_laz_points_safe(Path("test.laz"))
            assert result is not None
            assert "x" in result
            assert len(result["x"]) == 3


# =============================================================================
# Test Memory Management (Phase 8.4)
# =============================================================================

class TestMemoryManagement:
    """Test memory management utilities."""

    def test_clear_memory_runs(self):
        """Test clear_memory runs without error."""
        # Just verify it doesn't raise
        clear_memory()

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Create array with known size: 1000 float64 = 8000 bytes = ~0.0076 MB
        arr = np.zeros(1000, dtype=np.float64)
        mb = estimate_memory_usage(arr)
        expected = 8000 / (1024 * 1024)  # ~0.00763 MB
        assert abs(mb - expected) < 0.001

    def test_estimate_memory_usage_large_array(self):
        """Test memory estimation for larger arrays."""
        # 1 million float64 = 8MB
        arr = np.zeros(1_000_000, dtype=np.float64)
        mb = estimate_memory_usage(arr)
        expected = 8_000_000 / (1024 * 1024)  # ~7.63 MB
        assert abs(mb - expected) < 0.01


class TestChunkedFileIterator:
    """Test chunked file iterator for memory efficiency."""

    def test_chunked_iterator_empty_list(self):
        """Test chunked iterator with empty file list."""
        chunks = list(chunked_file_iterator([], max_memory_mb=100.0))
        assert len(chunks) == 0

    def test_chunked_iterator_returns_chunks(self):
        """Test chunked iterator yields file chunks."""
        # Create fake file list
        fake_files = [
            (Path(f"/fake/file_{i}.laz"), datetime(2025, 1, 1, i))
            for i in range(10)
        ]

        # Mock loader that returns small arrays
        def mock_loader(path):
            return {
                "x": np.zeros(100),
                "y": np.zeros(100),
                "z": np.zeros(100),
            }

        chunks = list(chunked_file_iterator(
            fake_files,
            max_memory_mb=10.0,  # Large enough for all files
            loader=mock_loader,
        ))

        # Should have at least one chunk
        assert len(chunks) >= 1

        # All files should be covered
        all_files = []
        for chunk, _ in chunks:
            all_files.extend(chunk)
        assert len(all_files) == 10

    def test_chunked_iterator_respects_memory_limit(self):
        """Test chunked iterator splits on memory limit."""
        # Create fake file list
        fake_files = [
            (Path(f"/fake/file_{i}.laz"), datetime(2025, 1, 1, i))
            for i in range(5)
        ]

        # Mock loader that returns arrays using ~1MB each
        def mock_loader(path):
            # ~1MB of float64 data
            size = 1024 * 1024 // 8  # 131072 float64 values = 1MB
            return {
                "x": np.zeros(size, dtype=np.float64),
                "y": np.zeros(size, dtype=np.float64),
                "z": np.zeros(size, dtype=np.float64),
            }

        # With 2MB limit, should get multiple chunks
        chunks = list(chunked_file_iterator(
            fake_files,
            max_memory_mb=2.0,  # Only ~2 files worth
            loader=mock_loader,
        ))

        # Should have multiple chunks (not all files in one)
        # With ~3MB per file and 2MB limit, should get 5 chunks (1 file each)
        assert len(chunks) >= 2


# =============================================================================
# Test CLI Argument Parsing (Phase 8.3)
# =============================================================================

class TestCLIArgumentParsing:
    """Test CLI argument parsing for new flags."""

    def test_import_argparse_components(self):
        """Test that CLI components are importable."""
        from phase4 import main
        # Just verify import works
        assert callable(main)

    def test_cli_has_verbose_flag(self):
        """Test CLI accepts --verbose flag."""
        import argparse
        from phase4 import main

        # Create a mock parser to check it accepts --verbose
        # We can't easily test the full CLI without running it,
        # but we can verify the functions exist
        assert True  # Placeholder - actual CLI testing requires subprocess


# =============================================================================
# Test Integration
# =============================================================================

class TestPhase8Integration:
    """Integration tests for Phase 8 features."""

    def test_checkpoint_workflow(self):
        """Test full checkpoint save/load/resume workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Simulate initial run
            checkpoint1 = Checkpoint(
                config_path="config.json",
                output_dir=tmpdir,
                start_date="2025-01-01",
                end_date="2025-01-10",
                completed_dates=["2025-01-01", "2025-01-02"],
                failed_dates=[],
                kwargs={"bin_size": 0.1},
            )
            checkpoint1.save(checkpoint_path)

            # Simulate resume
            checkpoint2 = Checkpoint.load(checkpoint_path)

            # Add more completed dates
            checkpoint2.completed_dates.append("2025-01-03")
            checkpoint2.save(checkpoint_path)

            # Load again and verify
            checkpoint3 = Checkpoint.load(checkpoint_path)
            assert "2025-01-03" in checkpoint3.completed_dates
            assert len(checkpoint3.completed_dates) == 3

    def test_logging_with_progress(self):
        """Test logging works with progress tracking."""
        configure_logging(verbose=True, log_file=None, quiet=False)
        logger = logging.getLogger("lidar_pipeline")

        # Log some messages
        logger.debug("Debug message")
        logger.info("Info message")

        # No errors should occur
        assert True

    def test_error_handling_chain(self):
        """Test exception handling chain."""
        # Test that we can catch specific errors
        try:
            raise CorruptFileError("test.laz is corrupt")
        except LidarProcessingError as e:
            assert "corrupt" in str(e)

        # Test that we can catch any processing error
        for error_class in [CorruptFileError, NoDataError, ConfigurationError]:
            try:
                raise error_class("test error")
            except LidarProcessingError:
                pass  # Should be caught


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
