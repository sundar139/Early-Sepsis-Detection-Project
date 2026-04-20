from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.data.pipeline import create_window_pipeline
from early_sepsis.logging_utils import configure_logging
from early_sepsis.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sliding windows for sepsis modeling")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory produced by preprocess_data.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store window parquet files",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Number of hourly timesteps in each input window",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=None,
        help="Positive label horizon in hours",
    )
    parser.add_argument(
        "--padding-mode",
        action="store_true",
        help="Enable left-padding for incomplete early-history windows",
    )
    parser.add_argument(
        "--without-masks",
        action="store_true",
        help="Exclude missingness mask arrays from output windows",
    )
    parser.add_argument(
        "--without-static",
        action="store_true",
        help="Exclude static feature vectors from output windows",
    )
    parser.add_argument(
        "--static-columns",
        default=None,
        help="Comma-separated static feature columns (overrides metadata)",
    )
    return parser


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    parser = _build_parser()
    args = parser.parse_args()

    processed_dir = args.processed_dir or str(settings.processed_data_dir)
    output_dir = args.output_dir or str(settings.window_data_dir)
    window_length = args.window_length if args.window_length is not None else settings.window_length
    prediction_horizon = (
        args.prediction_horizon
        if args.prediction_horizon is not None
        else settings.prediction_horizon
    )
    static_columns = None
    if args.static_columns:
        static_columns = [column.strip() for column in args.static_columns.split(",") if column.strip()]

    result = create_window_pipeline(
        processed_dir=processed_dir,
        output_dir=output_dir,
        window_length=window_length,
        prediction_horizon=prediction_horizon,
        padding_mode=args.padding_mode,
        include_masks=not args.without_masks,
        include_static=not args.without_static,
        static_feature_columns=static_columns,
    )

    summary = {
        "output_dir": str(result.output_dir),
        "metadata_path": str(result.metadata_path),
        "feature_schema_path": str(result.feature_schema_path),
        "split_paths": {name: str(path) for name, path in result.split_paths.items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
