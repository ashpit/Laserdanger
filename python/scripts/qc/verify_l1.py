#!/usr/bin/env python3
"""
Verification script for L1 pipeline outputs.

Compares Python L1 outputs to MATLAB L1 outputs and generates
a validation report.

Usage
-----
# Compare single files
python verify_l1.py python_output.nc matlab_output.mat

# Compare directories (batch mode)
python verify_l1.py python_dir/ matlab_dir/ --batch

# Save report to file
python verify_l1.py python.nc matlab.mat -o report.json
"""

import argparse
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from validation import (
    compare_l1_outputs,
    compare_coordinates,
    validate_batch,
    summarize_batch_validation,
    ValidationReport,
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Python L1 outputs against MATLAB reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "python_path",
        type=Path,
        help="Python output file (.nc) or directory for batch mode",
    )
    parser.add_argument(
        "matlab_path",
        type=Path,
        help="MATLAB output file (.mat) or directory for batch mode",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for validation report (JSON)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: compare all files in directories",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        help="Compare only this time index (for multi-time files)",
    )
    parser.add_argument(
        "--threshold-rmse",
        type=float,
        default=0.1,
        help="RMSE threshold for pass/fail (default: 0.1m)",
    )
    parser.add_argument(
        "--threshold-correlation",
        type=float,
        default=0.95,
        help="Correlation threshold for pass/fail (default: 0.95)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.batch:
        # Batch mode
        if not args.python_path.is_dir():
            print(f"Error: {args.python_path} is not a directory")
            sys.exit(1)
        if not args.matlab_path.is_dir():
            print(f"Error: {args.matlab_path} is not a directory")
            sys.exit(1)

        output_dir = args.output or Path("validation_reports")
        reports = validate_batch(
            args.python_path,
            args.matlab_path,
            output_dir,
            level="L1",
        )

        print(summarize_batch_validation(reports))

    else:
        # Single file mode
        if not args.python_path.exists():
            print(f"Error: Python file not found: {args.python_path}")
            sys.exit(1)
        if not args.matlab_path.exists():
            print(f"Error: MATLAB file not found: {args.matlab_path}")
            sys.exit(1)

        report = compare_l1_outputs(
            args.python_path,
            args.matlab_path,
            time_index=args.time_index,
        )

        # Print summary
        print(report.summary())

        # Save report if requested
        if args.output:
            report.save_json(args.output)
            print(f"\nReport saved to: {args.output}")

        # Exit with appropriate code
        if report.passed:
            print("\n✓ Validation PASSED")
            sys.exit(0)
        else:
            print("\n✗ Validation FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
