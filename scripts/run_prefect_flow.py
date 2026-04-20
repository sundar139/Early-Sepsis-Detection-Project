from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.data.ingestion import DatasetFormat
from early_sepsis.orchestration.flow import run_training_flow
from early_sepsis.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Prefect training flow")
    parser.add_argument(
        "--data-path",
        default="tests/fixtures/synthetic_tabular.csv",
        help="Dataset file or directory path",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "physionet"],
        default="auto",
        help="Input dataset format",
    )
    parser.add_argument("--target-column", default=None, help="Target column name")
    parser.add_argument("--output-path", default=None, help="Model output path")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = get_settings()
    target_column = args.target_column or settings.default_target_column
    dataset_format = cast(DatasetFormat, args.dataset_format)

    result = run_training_flow(
        data_path=args.data_path,
        dataset_format=dataset_format,
        target_column=target_column,
        model_output_path=args.output_path,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
