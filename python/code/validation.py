"""
Validation utilities for comparing Python pipeline outputs to MATLAB reference data.

This module provides functions for:
- Loading MATLAB .mat files for comparison
- Computing comparison metrics (RMSE, correlation, bias)
- Generating comparison reports
- Visualizing differences

Usage
-----
# Compare L1 outputs
from validation import compare_l1_outputs, ValidationReport
report = compare_l1_outputs(python_nc_path, matlab_mat_path)
print(report.summary())

# Compare L2 runup statistics
from validation import compare_runup_stats
report = compare_runup_stats(python_result, matlab_result)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import xarray as xr

try:
    from scipy.io import loadmat
    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Validation Results
# =============================================================================

@dataclass
class FieldComparison:
    """Comparison result for a single field/variable."""
    field_name: str
    python_shape: Tuple[int, ...]
    matlab_shape: Tuple[int, ...]
    rmse: float
    correlation: float
    bias: float  # mean(python - matlab)
    max_abs_diff: float
    n_valid_points: int
    n_total_points: int
    notes: str = ""

    @property
    def valid_fraction(self) -> float:
        """Fraction of points that are valid in both datasets."""
        if self.n_total_points == 0:
            return 0.0
        return self.n_valid_points / self.n_total_points

    @property
    def shapes_match(self) -> bool:
        """Check if shapes match."""
        return self.python_shape == self.matlab_shape

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "field_name": self.field_name,
            "python_shape": list(self.python_shape),
            "matlab_shape": list(self.matlab_shape),
            "rmse": float(self.rmse) if not np.isnan(self.rmse) else None,
            "correlation": float(self.correlation) if not np.isnan(self.correlation) else None,
            "bias": float(self.bias) if not np.isnan(self.bias) else None,
            "max_abs_diff": float(self.max_abs_diff) if not np.isnan(self.max_abs_diff) else None,
            "n_valid_points": self.n_valid_points,
            "n_total_points": self.n_total_points,
            "valid_fraction": self.valid_fraction,
            "shapes_match": self.shapes_match,
            "notes": self.notes,
        }


@dataclass
class ValidationReport:
    """Complete validation report comparing Python and MATLAB outputs."""
    name: str
    python_source: str
    matlab_source: str
    field_comparisons: List[FieldComparison] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all comparisons pass thresholds."""
        if self.errors:
            return False
        # Default thresholds - can be customized
        for fc in self.field_comparisons:
            if fc.rmse > 0.1:  # 10cm RMSE threshold for elevation
                return False
            if fc.correlation < 0.95:  # 95% correlation threshold
                return False
        return True

    @property
    def overall_rmse(self) -> float:
        """Average RMSE across all fields."""
        rmses = [fc.rmse for fc in self.field_comparisons if not np.isnan(fc.rmse)]
        return np.mean(rmses) if rmses else np.nan

    @property
    def overall_correlation(self) -> float:
        """Average correlation across all fields."""
        corrs = [fc.correlation for fc in self.field_comparisons if not np.isnan(fc.correlation)]
        return np.mean(corrs) if corrs else np.nan

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Validation Report: {self.name}",
            "=" * 60,
            f"Python source: {self.python_source}",
            f"MATLAB source: {self.matlab_source}",
            "",
            f"Overall Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Overall RMSE: {self.overall_rmse:.6f}",
            f"Overall Correlation: {self.overall_correlation:.6f}",
            "",
            "Field Comparisons:",
            "-" * 60,
        ]

        for fc in self.field_comparisons:
            lines.append(
                f"  {fc.field_name}: RMSE={fc.rmse:.6f}, "
                f"r={fc.correlation:.4f}, bias={fc.bias:.6f}"
            )
            if not fc.shapes_match:
                lines.append(f"    WARNING: Shape mismatch {fc.python_shape} vs {fc.matlab_shape}")
            if fc.notes:
                lines.append(f"    Note: {fc.notes}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "name": self.name,
            "python_source": self.python_source,
            "matlab_source": self.matlab_source,
            "passed": self.passed,
            "overall_rmse": float(self.overall_rmse) if not np.isnan(self.overall_rmse) else None,
            "overall_correlation": float(self.overall_correlation) if not np.isnan(self.overall_correlation) else None,
            "field_comparisons": [fc.to_dict() for fc in self.field_comparisons],
            "metadata": self.metadata,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def save_json(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "ValidationReport":
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)

        report = cls(
            name=data["name"],
            python_source=data["python_source"],
            matlab_source=data["matlab_source"],
            metadata=data.get("metadata", {}),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
        )

        for fc_data in data.get("field_comparisons", []):
            fc = FieldComparison(
                field_name=fc_data["field_name"],
                python_shape=tuple(fc_data["python_shape"]),
                matlab_shape=tuple(fc_data["matlab_shape"]),
                rmse=fc_data["rmse"] if fc_data["rmse"] is not None else np.nan,
                correlation=fc_data["correlation"] if fc_data["correlation"] is not None else np.nan,
                bias=fc_data["bias"] if fc_data["bias"] is not None else np.nan,
                max_abs_diff=fc_data["max_abs_diff"] if fc_data["max_abs_diff"] is not None else np.nan,
                n_valid_points=fc_data["n_valid_points"],
                n_total_points=fc_data["n_total_points"],
                notes=fc_data.get("notes", ""),
            )
            report.field_comparisons.append(fc)

        return report


# =============================================================================
# Comparison Metrics
# =============================================================================

def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Root Mean Square Error between two arrays.

    NaN values are ignored (pairwise complete).

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare (must have same shape)

    Returns
    -------
    float
        RMSE value, or NaN if no valid points
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    valid = ~(np.isnan(a) | np.isnan(b))
    if not np.any(valid):
        return np.nan

    diff = a[valid] - b[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two arrays.

    NaN values are ignored (pairwise complete).

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare (must have same shape)

    Returns
    -------
    float
        Correlation coefficient, or NaN if insufficient valid points
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    valid = ~(np.isnan(a) | np.isnan(b))
    if np.sum(valid) < 2:
        return np.nan

    a_valid = a[valid]
    b_valid = b[valid]

    # Detrend
    a_centered = a_valid - np.mean(a_valid)
    b_centered = b_valid - np.mean(b_valid)

    # Correlation
    denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    if denom == 0:
        return np.nan

    return float(np.sum(a_centered * b_centered) / denom)


def compute_bias(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean bias (a - b).

    NaN values are ignored.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare (must have same shape)

    Returns
    -------
    float
        Mean bias, or NaN if no valid points
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    valid = ~(np.isnan(a) | np.isnan(b))
    if not np.any(valid):
        return np.nan

    return float(np.mean(a[valid] - b[valid]))


def compute_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute maximum absolute difference.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare

    Returns
    -------
    float
        Maximum absolute difference, or NaN if no valid points
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    valid = ~(np.isnan(a) | np.isnan(b))
    if not np.any(valid):
        return np.nan

    return float(np.max(np.abs(a[valid] - b[valid])))


def compare_arrays(
    python_arr: np.ndarray,
    matlab_arr: np.ndarray,
    field_name: str = "unknown",
) -> FieldComparison:
    """
    Compare two arrays and compute all metrics.

    Parameters
    ----------
    python_arr : np.ndarray
        Array from Python pipeline
    matlab_arr : np.ndarray
        Array from MATLAB pipeline
    field_name : str
        Name of the field being compared

    Returns
    -------
    FieldComparison
        Comparison results
    """
    python_arr = np.asarray(python_arr)
    matlab_arr = np.asarray(matlab_arr)

    python_shape = python_arr.shape
    matlab_shape = matlab_arr.shape

    # Handle shape mismatches
    notes = ""
    if python_shape != matlab_shape:
        notes = f"Shape mismatch: Python {python_shape} vs MATLAB {matlab_shape}"
        # Try to compare overlapping region
        if len(python_shape) == len(matlab_shape):
            # Take minimum along each dimension
            slices = tuple(slice(0, min(p, m)) for p, m in zip(python_shape, matlab_shape))
            python_arr = python_arr[slices]
            matlab_arr = matlab_arr[slices]
            notes += f"; comparing overlap region {python_arr.shape}"

    # Count valid points
    valid_mask = ~(np.isnan(python_arr) | np.isnan(matlab_arr))
    n_valid = int(np.sum(valid_mask))
    n_total = int(python_arr.size)

    return FieldComparison(
        field_name=field_name,
        python_shape=python_shape,
        matlab_shape=matlab_shape,
        rmse=compute_rmse(python_arr, matlab_arr),
        correlation=compute_correlation(python_arr, matlab_arr),
        bias=compute_bias(python_arr, matlab_arr),
        max_abs_diff=compute_max_abs_diff(python_arr, matlab_arr),
        n_valid_points=n_valid,
        n_total_points=n_total,
        notes=notes,
    )


# =============================================================================
# MATLAB File Loading
# =============================================================================

def load_matlab_struct(mat_path: Path, struct_name: str = "L1") -> Dict[str, np.ndarray]:
    """
    Load a MATLAB struct from a .mat file.

    Parameters
    ----------
    mat_path : Path
        Path to .mat file
    struct_name : str
        Name of the struct variable to load (e.g., "L1", "L2")

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping field names to arrays
    """
    if not HAS_SCIPY_IO:
        raise ImportError("scipy.io is required to load MATLAB files")

    mat_data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    if struct_name not in mat_data:
        available = [k for k in mat_data.keys() if not k.startswith("_")]
        raise KeyError(f"Struct '{struct_name}' not found in {mat_path}. Available: {available}")

    struct = mat_data[struct_name]

    # Handle struct arrays (multiple time steps)
    if hasattr(struct, '__len__') and not isinstance(struct, str):
        # It's an array of structs - get field names from first element
        if len(struct) > 0:
            field_names = struct[0]._fieldnames if hasattr(struct[0], '_fieldnames') else dir(struct[0])
        else:
            return {}
    else:
        # Single struct
        field_names = struct._fieldnames if hasattr(struct, '_fieldnames') else dir(struct)

    result = {}
    for name in field_names:
        if name.startswith('_'):
            continue
        try:
            if hasattr(struct, '__len__') and not isinstance(struct, str):
                # Stack across struct array
                values = [getattr(s, name) for s in struct]
                result[name] = np.array(values)
            else:
                result[name] = np.asarray(getattr(struct, name))
        except Exception as e:
            logger.warning(f"Could not load field '{name}': {e}")

    return result


def load_matlab_l1(mat_path: Path) -> Dict[str, np.ndarray]:
    """
    Load MATLAB L1 output struct.

    Expected fields: Dates, X, Y, Zmean, Zmax, Zmin, Zstdv, Zmode

    Parameters
    ----------
    mat_path : Path
        Path to .mat file containing L1 struct

    Returns
    -------
    Dict[str, np.ndarray]
        L1 data fields
    """
    return load_matlab_struct(mat_path, "L1")


def load_matlab_l2(mat_path: Path) -> Dict[str, np.ndarray]:
    """
    Load MATLAB L2 output.

    Expected fields: Z_xt, I_xt, x1d, time_vec, etc.

    Parameters
    ----------
    mat_path : Path
        Path to .mat file containing L2 data

    Returns
    -------
    Dict[str, np.ndarray]
        L2 data fields
    """
    if not HAS_SCIPY_IO:
        raise ImportError("scipy.io is required to load MATLAB files")

    mat_data = loadmat(str(mat_path), squeeze_me=True)

    # Filter out MATLAB metadata
    return {k: v for k, v in mat_data.items() if not k.startswith("_")}


# =============================================================================
# L1 Comparison
# =============================================================================

def compare_l1_outputs(
    python_path: Path,
    matlab_path: Path,
    time_index: Optional[int] = None,
) -> ValidationReport:
    """
    Compare Python L1 output (NetCDF) to MATLAB L1 output (.mat).

    Parameters
    ----------
    python_path : Path
        Path to Python NetCDF output
    matlab_path : Path
        Path to MATLAB .mat output
    time_index : int, optional
        If specified, compare only this time index

    Returns
    -------
    ValidationReport
        Comparison results
    """
    report = ValidationReport(
        name="L1 Comparison",
        python_source=str(python_path),
        matlab_source=str(matlab_path),
    )

    try:
        # Load Python data
        python_ds = xr.open_dataset(python_path)
    except Exception as e:
        report.errors.append(f"Failed to load Python NetCDF: {e}")
        return report

    try:
        # Load MATLAB data
        matlab_data = load_matlab_l1(matlab_path)
    except Exception as e:
        report.errors.append(f"Failed to load MATLAB .mat: {e}")
        return report

    # Field mappings: Python name -> MATLAB name
    field_map = {
        "z_mean": "Zmean",
        "z_max": "Zmax",
        "z_min": "Zmin",
        "z_std": "Zstdv",
        "z_mode": "Zmode",
    }

    for py_name, mat_name in field_map.items():
        if py_name not in python_ds:
            report.warnings.append(f"Python field '{py_name}' not found")
            continue
        if mat_name not in matlab_data:
            report.warnings.append(f"MATLAB field '{mat_name}' not found")
            continue

        py_arr = python_ds[py_name].values
        mat_arr = matlab_data[mat_name]

        if time_index is not None:
            if py_arr.ndim >= 3:
                py_arr = py_arr[time_index]
            if mat_arr.ndim >= 3:
                mat_arr = mat_arr[time_index]

        fc = compare_arrays(py_arr, mat_arr, py_name)
        report.field_comparisons.append(fc)

    python_ds.close()
    return report


# =============================================================================
# L2 / Runup Comparison
# =============================================================================

def compare_l2_outputs(
    python_path: Path,
    matlab_path: Path,
) -> ValidationReport:
    """
    Compare Python L2 output to MATLAB L2 output.

    Parameters
    ----------
    python_path : Path
        Path to Python NetCDF output
    matlab_path : Path
        Path to MATLAB .mat output

    Returns
    -------
    ValidationReport
        Comparison results
    """
    report = ValidationReport(
        name="L2 Comparison",
        python_source=str(python_path),
        matlab_source=str(matlab_path),
    )

    try:
        python_ds = xr.open_dataset(python_path)
    except Exception as e:
        report.errors.append(f"Failed to load Python NetCDF: {e}")
        return report

    try:
        matlab_data = load_matlab_l2(matlab_path)
    except Exception as e:
        report.errors.append(f"Failed to load MATLAB .mat: {e}")
        return report

    # Compare Z_xt matrices
    field_map = {
        "Z": "Z_xt",
        "I": "I_xt",
    }

    for py_name, mat_name in field_map.items():
        if py_name not in python_ds:
            report.warnings.append(f"Python field '{py_name}' not found")
            continue
        if mat_name not in matlab_data:
            report.warnings.append(f"MATLAB field '{mat_name}' not found")
            continue

        fc = compare_arrays(
            python_ds[py_name].values,
            matlab_data[mat_name],
            py_name,
        )
        report.field_comparisons.append(fc)

    python_ds.close()
    return report


def compare_runup_stats(
    python_stats: Dict[str, Any],
    matlab_stats: Dict[str, Any],
) -> ValidationReport:
    """
    Compare runup statistics between Python and MATLAB.

    Parameters
    ----------
    python_stats : dict
        Python runup statistics (from compute_runup_stats)
    matlab_stats : dict
        MATLAB runup statistics (from get_runupStats_L2)

    Returns
    -------
    ValidationReport
        Comparison results
    """
    report = ValidationReport(
        name="Runup Statistics Comparison",
        python_source="Python compute_runup_stats()",
        matlab_source="MATLAB get_runupStats_L2()",
    )

    # Bulk statistics mapping
    bulk_map = {
        "Sig": "Sig",      # IG significant wave height
        "Sinc": "Sinc",    # Incident significant wave height
        "eta": "eta",      # Mean water level
        "beta": "beta",    # Beach slope
        "R2": "R2",        # 2% exceedance
    }

    for py_name, mat_name in bulk_map.items():
        py_val = python_stats.get(py_name)
        mat_val = matlab_stats.get(mat_name)

        if py_val is None:
            report.warnings.append(f"Python stat '{py_name}' not found")
            continue
        if mat_val is None:
            report.warnings.append(f"MATLAB stat '{mat_name}' not found")
            continue

        fc = compare_arrays(
            np.array([py_val]),
            np.array([mat_val]),
            py_name,
        )
        report.field_comparisons.append(fc)

    return report


# =============================================================================
# Coordinate Comparison
# =============================================================================

def compare_coordinates(
    python_x: np.ndarray,
    python_y: np.ndarray,
    matlab_x: np.ndarray,
    matlab_y: np.ndarray,
) -> ValidationReport:
    """
    Compare coordinate grids between Python and MATLAB.

    This is important because small coordinate differences can
    lead to large apparent differences in gridded data.

    Parameters
    ----------
    python_x, python_y : np.ndarray
        Python coordinate arrays
    matlab_x, matlab_y : np.ndarray
        MATLAB coordinate arrays

    Returns
    -------
    ValidationReport
        Comparison results
    """
    report = ValidationReport(
        name="Coordinate Comparison",
        python_source="Python coordinates",
        matlab_source="MATLAB coordinates",
    )

    report.field_comparisons.append(
        compare_arrays(python_x, matlab_x, "X coordinates")
    )
    report.field_comparisons.append(
        compare_arrays(python_y, matlab_y, "Y coordinates")
    )

    return report


# =============================================================================
# Batch Validation
# =============================================================================

def validate_batch(
    python_dir: Path,
    matlab_dir: Path,
    output_dir: Path,
    level: str = "L1",
) -> List[ValidationReport]:
    """
    Validate multiple files in batch.

    Parameters
    ----------
    python_dir : Path
        Directory containing Python outputs
    matlab_dir : Path
        Directory containing MATLAB outputs
    output_dir : Path
        Directory to save validation reports
    level : str
        "L1" or "L2"

    Returns
    -------
    List[ValidationReport]
        List of validation reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    # Find Python files
    python_files = sorted(python_dir.glob("*.nc"))

    for py_file in python_files:
        # Try to find matching MATLAB file
        # Assume naming convention: YYYYMMDD.nc -> YYYYMMDD.mat
        stem = py_file.stem
        mat_file = matlab_dir / f"{stem}.mat"

        if not mat_file.exists():
            logger.warning(f"No MATLAB match for {py_file.name}")
            continue

        logger.info(f"Validating {py_file.name}")

        if level == "L1":
            report = compare_l1_outputs(py_file, mat_file)
        else:
            report = compare_l2_outputs(py_file, mat_file)

        reports.append(report)

        # Save individual report
        report_path = output_dir / f"{stem}_validation.json"
        report.save_json(report_path)

    return reports


def summarize_batch_validation(reports: List[ValidationReport]) -> str:
    """
    Generate summary of batch validation results.

    Parameters
    ----------
    reports : List[ValidationReport]
        List of validation reports

    Returns
    -------
    str
        Human-readable summary
    """
    if not reports:
        return "No validation reports to summarize."

    n_passed = sum(1 for r in reports if r.passed)
    n_total = len(reports)

    all_rmses = []
    all_corrs = []
    for r in reports:
        for fc in r.field_comparisons:
            if not np.isnan(fc.rmse):
                all_rmses.append(fc.rmse)
            if not np.isnan(fc.correlation):
                all_corrs.append(fc.correlation)

    lines = [
        "Batch Validation Summary",
        "=" * 60,
        f"Total files validated: {n_total}",
        f"Passed: {n_passed} ({100*n_passed/n_total:.1f}%)",
        f"Failed: {n_total - n_passed}",
        "",
        "Aggregate Statistics:",
        f"  Mean RMSE: {np.mean(all_rmses):.6f}" if all_rmses else "  Mean RMSE: N/A",
        f"  Max RMSE: {np.max(all_rmses):.6f}" if all_rmses else "  Max RMSE: N/A",
        f"  Mean Correlation: {np.mean(all_corrs):.6f}" if all_corrs else "  Mean Correlation: N/A",
        f"  Min Correlation: {np.min(all_corrs):.6f}" if all_corrs else "  Min Correlation: N/A",
    ]

    # List failed files
    failed = [r for r in reports if not r.passed]
    if failed:
        lines.append("")
        lines.append("Failed Files:")
        for r in failed:
            lines.append(f"  - {Path(r.python_source).name}: RMSE={r.overall_rmse:.6f}")

    return "\n".join(lines)


# =============================================================================
# Intentional Differences Documentation
# =============================================================================

INTENTIONAL_DIFFERENCES = """
# Intentional Differences Between Python and MATLAB Implementations

This document describes known differences between the Python and MATLAB
implementations of the lidar processing pipeline.

## 1. Data Formats

- **MATLAB**: Uses .mat files with struct arrays
- **Python**: Uses NetCDF4 with xarray for better interoperability

## 2. Coordinate Handling

- **MATLAB**: Uses 1-based indexing
- **Python**: Uses 0-based indexing
- Grid edges may differ by half a bin width depending on convention

## 3. NaN Handling

- Both implementations use NaN for missing data
- Python uses numpy.nan, MATLAB uses NaN
- Comparison functions ignore NaN values (pairwise complete)

## 4. Floating Point Precision

- Minor differences expected due to floating point arithmetic
- RMSE < 1e-6 should be considered equivalent

## 5. Algorithm Differences

### Percentile Filtering
- Both use 50th percentile within bins
- Numerical differences may arise from different quantile algorithms

### SNR Calculation
- Formula: SNR = mean / (std / sqrt(count))
- Same in both implementations

### Residual Kernel Filter
- Both use Delaunay triangulation
- Minor differences in triangulation may exist for edge cases

### Runup Detection
- Threshold crossing algorithm is identical
- Moving minimum filter uses same window sizes

## 6. Known Acceptable Differences

| Metric | Acceptable Threshold |
|--------|---------------------|
| Elevation RMSE | < 0.01 m (1 cm) |
| Correlation | > 0.99 |
| Coordinate RMSE | < 0.001 m |
| Runup position RMSE | < 0.1 m |
"""


def get_intentional_differences() -> str:
    """Return documentation of intentional differences."""
    return INTENTIONAL_DIFFERENCES
