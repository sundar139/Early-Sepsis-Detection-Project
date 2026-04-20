from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.data.ingestion import DatasetFormat, ingest_raw_dataset, validate_schema
from early_sepsis.logging_utils import configure_logging
from early_sepsis.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate raw sepsis source files")
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Directory or file path containing raw PhysioNet PSV or CSV mirror files",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "physionet"],
        default="auto",
        help="Input dataset format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code when any file validation issue is found",
    )
    return parser


def _print_layout_guidance(raw_path: Path) -> None:
    guidance = {
        "message": "Raw data path was not found. Place files and rerun validation.",
        "raw_path": str(raw_path),
        "physionet_layout": [
            "<raw_path>/patient_0001.psv",
            "<raw_path>/patient_0002.psv",
            "...",
        ],
        "csv_layout": [
            "<raw_path>/sepsis_data.csv",
            "CSV should include SepsisLabel and either patient_id or one patient per file",
        ],
    }
    print(json.dumps(guidance, indent=2))


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    parser = _build_parser()
    args = parser.parse_args()

    raw_path = Path(args.raw_path) if args.raw_path else settings.raw_data_dir
    if not raw_path.exists():
        _print_layout_guidance(raw_path)
        raise SystemExit(1)

    dataset_format = args.dataset_format  # argparse constrains values

    validation_result = validate_schema(raw_path, dataset_format=dataset_format)
    report = {
        "raw_path": str(raw_path),
        "dataset_format": validation_result.dataset_format,
        "file_count": validation_result.file_count,
        "skipped_file_count": validation_result.skipped_file_count,
        "row_count": int(len(validation_result.dataframe)),
        "patient_count": int(validation_result.dataframe["patient_id"].nunique()),
        "feature_count": len(validation_result.feature_columns),
        "issues": [
            {"file_path": issue.file_path, "reason": issue.reason}
            for issue in validation_result.issues
        ],
    }
    print(json.dumps(report, indent=2))

    if args.strict and validation_result.skipped_file_count > 0:
        raise SystemExit(2)

    # If strict mode is enabled, run strict ingestion to fail-fast for malformed rows.
    if args.strict:
        ingest_raw_dataset(
            data_path=raw_path,
            dataset_format=dataset_format,
            strict_validation=True,
        )


if __name__ == "__main__":
    main()
