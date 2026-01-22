#!/usr/bin/env python3
"""
Verification script for L2 pipeline outputs.

Compares Python L2 outputs (time-resolved Z(x,t) matrices and runup
statistics) to MATLAB L2 outputs.

Usage
-----
# Compare L2 outputs
python verify_l2.py python_output.nc matlab_output.mat

# Compare directories (batch mode)
python verify_l2.py python_dir/ matlab_dir/ --batch

# Compare runup statistics only
python verify_l2.py --runup-only python_stats.json matlab_stats.mat
"""

import argparse
import json
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from validation import (
    compare_l2_outputs,
    compare_runup_stats,
    validate_batch,
    summarize_batch_validation,
    load_matlab_l2,
)


def load_python_runup_stats(path: Path) -> dict:
    """Load Python runup statistics from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_matlab_runup_stats(path: Path) -> dict:
    """Load MATLAB runup statistics from .mat file."""
    try:
        data = load_matlab_l2(path)
        # Extract bulk statistics
        stats = {}
        if "Bulk" in data:
            bulk = data["Bulk"]
            if hasattr(bulk, "swashparams"):
                sp = bulk.swashparams
                stats["Sig"] = sp[0] if len(sp) > 0 else None
                stats["Sinc"] = sp[1] if len(sp) > 1 else None
                stats["eta"] = sp[2] if len(sp) > 2 else None
            if hasattr(bulk, "beta"):
                stats["beta"] = bulk.beta
        return stats
    except Exception as e:
        print(f"Warning: Could not load MATLAB runup stats: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Verify Python L2 outputs against MATLAB reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "python_path",
        type=Path,
        help="Python output file (.nc/.json) or directory for batch mode",
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
        "--runup-only",
        action="store_true",
        help="Compare runup statistics only (not Z_xt matrices)",
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

        output_dir = args.output or Path("validation_reports_l2")
        reports = validate_batch(
            args.python_path,
            args.matlab_path,
            output_dir,
            level="L2",
        )

        print(summarize_batch_validation(reports))

    elif args.runup_only:
        # Runup statistics comparison
        if not args.python_path.exists():
            print(f"Error: Python file not found: {args.python_path}")
            sys.exit(1)
        if not args.matlab_path.exists():
            print(f"Error: MATLAB file not found: {args.matlab_path}")
            sys.exit(1)

        python_stats = load_python_runup_stats(args.python_path)
        matlab_stats = load_matlab_runup_stats(args.matlab_path)

        report = compare_runup_stats(python_stats, matlab_stats)

        print(report.summary())

        if args.output:
            report.save_json(args.output)
            print(f"\nReport saved to: {args.output}")

    else:
        # Single file mode
        if not args.python_path.exists():
            print(f"Error: Python file not found: {args.python_path}")
            sys.exit(1)
        if not args.matlab_path.exists():
            print(f"Error: MATLAB file not found: {args.matlab_path}")
            sys.exit(1)

        report = compare_l2_outputs(args.python_path, args.matlab_path)

        print(report.summary())

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
