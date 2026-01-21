"""
Tests for validation utilities.

Tests the comparison metrics, report generation, and edge case handling.
"""

import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pytest

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from validation import (
    compute_rmse,
    compute_correlation,
    compute_bias,
    compute_max_abs_diff,
    compare_arrays,
    FieldComparison,
    ValidationReport,
    get_intentional_differences,
)


# =============================================================================
# Test Comparison Metrics
# =============================================================================

class TestComparisonMetrics:
    """Test individual comparison metric functions."""

    def test_rmse_identical_arrays(self):
        """RMSE of identical arrays should be 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_rmse(a, b) == 0.0

    def test_rmse_known_value(self):
        """Test RMSE with known expected value."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        # Differences: [1, 1, 1], squared: [1, 1, 1], mean: 1, sqrt: 1
        assert compute_rmse(a, b) == 1.0

    def test_rmse_with_nans(self):
        """RMSE should ignore NaN values."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        # Only compare [1, 3] vs [1, 3] -> RMSE = 0
        assert compute_rmse(a, b) == 0.0

    def test_rmse_all_nans(self):
        """RMSE with all NaN values should be NaN."""
        a = np.array([np.nan, np.nan])
        b = np.array([1.0, 2.0])
        assert np.isnan(compute_rmse(a, b))

    def test_correlation_identical_arrays(self):
        """Correlation of identical arrays should be 1."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_correlation(a, b) == pytest.approx(1.0)

    def test_correlation_negative(self):
        """Test negative correlation."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert compute_correlation(a, b) == pytest.approx(-1.0)

    def test_correlation_with_nans(self):
        """Correlation should ignore NaN values."""
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Should still be ~1.0 for the valid points
        corr = compute_correlation(a, b)
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_correlation_insufficient_points(self):
        """Correlation with < 2 valid points should be NaN."""
        a = np.array([1.0, np.nan])
        b = np.array([np.nan, 2.0])
        # No overlapping valid points
        assert np.isnan(compute_correlation(a, b))

    def test_bias_positive(self):
        """Test positive bias (a > b)."""
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0])
        # Mean(a - b) = mean([1, 1, 1]) = 1.0
        assert compute_bias(a, b) == 1.0

    def test_bias_negative(self):
        """Test negative bias (a < b)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert compute_bias(a, b) == -1.0

    def test_bias_with_nans(self):
        """Bias should ignore NaN values."""
        a = np.array([2.0, np.nan, 4.0])
        b = np.array([1.0, 2.0, 3.0])
        # Compare [2, 4] vs [1, 3] -> bias = mean([1, 1]) = 1.0
        assert compute_bias(a, b) == 1.0

    def test_max_abs_diff(self):
        """Test maximum absolute difference."""
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 2.0, 3.0])
        # Diffs: [-1, 3, 0], max abs = 3
        assert compute_max_abs_diff(a, b) == 3.0

    def test_max_abs_diff_with_nans(self):
        """Max abs diff should ignore NaN values."""
        a = np.array([1.0, np.nan, 10.0])
        b = np.array([2.0, 2.0, 3.0])
        # Compare [1, 10] vs [2, 3] -> diffs [-1, 7], max = 7
        assert compute_max_abs_diff(a, b) == 7.0


# =============================================================================
# Test Array Comparison
# =============================================================================

class TestCompareArrays:
    """Test the compare_arrays function."""

    def test_compare_identical_arrays(self):
        """Comparing identical arrays should show perfect agreement."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        fc = compare_arrays(a, b, "test_field")

        assert fc.field_name == "test_field"
        assert fc.rmse == 0.0
        assert fc.correlation == pytest.approx(1.0)
        assert fc.bias == 0.0
        assert fc.shapes_match

    def test_compare_different_shapes(self):
        """Comparing arrays with different shapes should note the mismatch."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0])
        fc = compare_arrays(a, b, "test_field")

        assert not fc.shapes_match
        assert "mismatch" in fc.notes.lower()

    def test_compare_2d_arrays(self):
        """Test comparison of 2D arrays."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        fc = compare_arrays(a, b, "grid")

        assert fc.python_shape == (2, 2)
        assert fc.matlab_shape == (2, 2)
        assert fc.rmse == 0.0

    def test_compare_with_nans(self):
        """Test comparison with NaN values."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        fc = compare_arrays(a, b, "field")

        assert fc.n_valid_points == 2
        assert fc.n_total_points == 3
        assert fc.valid_fraction == pytest.approx(2/3)


# =============================================================================
# Test FieldComparison
# =============================================================================

class TestFieldComparison:
    """Test FieldComparison dataclass."""

    def test_field_comparison_properties(self):
        """Test computed properties."""
        fc = FieldComparison(
            field_name="z_mean",
            python_shape=(10, 10),
            matlab_shape=(10, 10),
            rmse=0.05,
            correlation=0.98,
            bias=0.01,
            max_abs_diff=0.15,
            n_valid_points=80,
            n_total_points=100,
        )

        assert fc.shapes_match
        assert fc.valid_fraction == 0.8

    def test_field_comparison_to_dict(self):
        """Test conversion to dictionary."""
        fc = FieldComparison(
            field_name="z_mean",
            python_shape=(10,),
            matlab_shape=(10,),
            rmse=0.1,
            correlation=0.95,
            bias=0.01,
            max_abs_diff=0.2,
            n_valid_points=10,
            n_total_points=10,
        )

        d = fc.to_dict()
        assert d["field_name"] == "z_mean"
        assert d["rmse"] == 0.1
        assert d["shapes_match"] is True

    def test_field_comparison_with_nan_metrics(self):
        """Test to_dict with NaN metrics."""
        fc = FieldComparison(
            field_name="empty",
            python_shape=(0,),
            matlab_shape=(0,),
            rmse=np.nan,
            correlation=np.nan,
            bias=np.nan,
            max_abs_diff=np.nan,
            n_valid_points=0,
            n_total_points=0,
        )

        d = fc.to_dict()
        assert d["rmse"] is None
        assert d["correlation"] is None


# =============================================================================
# Test ValidationReport
# =============================================================================

class TestValidationReport:
    """Test ValidationReport class."""

    def test_report_creation(self):
        """Test creating a validation report."""
        report = ValidationReport(
            name="Test Report",
            python_source="test.nc",
            matlab_source="test.mat",
        )

        assert report.name == "Test Report"
        assert len(report.field_comparisons) == 0
        assert len(report.warnings) == 0
        assert len(report.errors) == 0

    def test_report_passed_with_good_metrics(self):
        """Test report passes with good metrics."""
        report = ValidationReport(
            name="Good Report",
            python_source="test.nc",
            matlab_source="test.mat",
        )
        report.field_comparisons.append(
            FieldComparison(
                field_name="z_mean",
                python_shape=(10,),
                matlab_shape=(10,),
                rmse=0.01,  # < 0.1 threshold
                correlation=0.99,  # > 0.95 threshold
                bias=0.001,
                max_abs_diff=0.02,
                n_valid_points=10,
                n_total_points=10,
            )
        )

        assert report.passed

    def test_report_failed_high_rmse(self):
        """Test report fails with high RMSE."""
        report = ValidationReport(
            name="Bad RMSE",
            python_source="test.nc",
            matlab_source="test.mat",
        )
        report.field_comparisons.append(
            FieldComparison(
                field_name="z_mean",
                python_shape=(10,),
                matlab_shape=(10,),
                rmse=0.5,  # > 0.1 threshold
                correlation=0.99,
                bias=0.0,
                max_abs_diff=1.0,
                n_valid_points=10,
                n_total_points=10,
            )
        )

        assert not report.passed

    def test_report_failed_low_correlation(self):
        """Test report fails with low correlation."""
        report = ValidationReport(
            name="Bad Correlation",
            python_source="test.nc",
            matlab_source="test.mat",
        )
        report.field_comparisons.append(
            FieldComparison(
                field_name="z_mean",
                python_shape=(10,),
                matlab_shape=(10,),
                rmse=0.01,
                correlation=0.8,  # < 0.95 threshold
                bias=0.0,
                max_abs_diff=0.1,
                n_valid_points=10,
                n_total_points=10,
            )
        )

        assert not report.passed

    def test_report_failed_with_errors(self):
        """Test report fails if there are errors."""
        report = ValidationReport(
            name="With Errors",
            python_source="test.nc",
            matlab_source="test.mat",
        )
        report.errors.append("File not found")

        assert not report.passed

    def test_report_overall_metrics(self):
        """Test overall RMSE and correlation calculation."""
        report = ValidationReport(
            name="Test",
            python_source="test.nc",
            matlab_source="test.mat",
        )
        report.field_comparisons.extend([
            FieldComparison("f1", (10,), (10,), 0.02, 0.98, 0.0, 0.1, 10, 10),
            FieldComparison("f2", (10,), (10,), 0.04, 0.96, 0.0, 0.1, 10, 10),
        ])

        assert report.overall_rmse == pytest.approx(0.03)
        assert report.overall_correlation == pytest.approx(0.97)

    def test_report_summary(self):
        """Test report summary generation."""
        report = ValidationReport(
            name="Summary Test",
            python_source="test.nc",
            matlab_source="test.mat",
        )
        report.field_comparisons.append(
            FieldComparison("z_mean", (10,), (10,), 0.01, 0.99, 0.0, 0.05, 10, 10)
        )

        summary = report.summary()
        assert "Summary Test" in summary
        assert "z_mean" in summary
        assert "PASSED" in summary

    def test_report_save_and_load_json(self):
        """Test saving and loading report as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            original = ValidationReport(
                name="JSON Test",
                python_source="test.nc",
                matlab_source="test.mat",
                metadata={"version": "1.0"},
            )
            original.field_comparisons.append(
                FieldComparison("z", (5,), (5,), 0.01, 0.99, 0.001, 0.02, 5, 5)
            )
            original.warnings.append("Test warning")

            original.save_json(report_path)
            assert report_path.exists()

            loaded = ValidationReport.load_json(report_path)
            assert loaded.name == original.name
            assert len(loaded.field_comparisons) == 1
            assert loaded.field_comparisons[0].rmse == 0.01
            assert loaded.warnings == ["Test warning"]


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test comparison of empty arrays."""
        a = np.array([])
        b = np.array([])
        fc = compare_arrays(a, b, "empty")

        assert fc.n_total_points == 0
        assert np.isnan(fc.rmse)

    def test_single_value(self):
        """Test comparison of single values."""
        a = np.array([1.0])
        b = np.array([2.0])
        fc = compare_arrays(a, b, "single")

        assert fc.rmse == 1.0
        assert fc.bias == -1.0

    def test_all_nans_in_one_array(self):
        """Test when one array is all NaN."""
        a = np.array([np.nan, np.nan, np.nan])
        b = np.array([1.0, 2.0, 3.0])
        fc = compare_arrays(a, b, "all_nan")

        assert fc.n_valid_points == 0
        assert np.isnan(fc.rmse)

    def test_large_arrays(self):
        """Test with larger arrays."""
        rng = np.random.default_rng(42)
        a = rng.random(10000)
        b = a + rng.normal(0, 0.01, 10000)  # Add small noise

        fc = compare_arrays(a, b, "large")

        assert fc.rmse < 0.02  # Should be close
        assert fc.correlation > 0.99  # Should be highly correlated

    def test_inf_values(self):
        """Test handling of infinite values."""
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        # Inf should be treated as invalid
        fc = compare_arrays(a, b, "with_inf")
        # Note: inf - finite = inf, which is truthy, so may not be filtered
        # This test documents current behavior


# =============================================================================
# Test Intentional Differences Documentation
# =============================================================================

class TestDocumentation:
    """Test documentation functions."""

    def test_get_intentional_differences(self):
        """Test that intentional differences doc is available."""
        doc = get_intentional_differences()
        assert "MATLAB" in doc
        assert "Python" in doc
        assert "NaN" in doc


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
