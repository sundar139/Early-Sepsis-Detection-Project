from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.data.synthetic import generate_synthetic_icu_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic ICU dataset for local smoke runs")
    parser.add_argument(
        "--output-path",
        default="tests/fixtures/generated_synthetic.csv",
        help="Output file path for csv mode or output directory for physionet mode",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["csv", "physionet"],
        default="csv",
        help="Synthetic output format",
    )
    parser.add_argument("--patient-count", type=int, default=16)
    parser.add_argument("--min-hours", type=int, default=10)
    parser.add_argument("--max-hours", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = generate_synthetic_icu_dataset(
        output_path=args.output_path,
        dataset_format=args.dataset_format,
        patient_count=args.patient_count,
        min_hours=args.min_hours,
        max_hours=args.max_hours,
        random_seed=args.seed,
    )

    report = {
        "output_path": str(result.output_path),
        "dataset_format": result.dataset_format,
        "patient_count": result.patient_count,
        "row_count": result.row_count,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
