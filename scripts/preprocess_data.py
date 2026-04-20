from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.data.ingestion import DatasetFormat
from early_sepsis.data.pipeline import run_preprocessing_pipeline
from early_sepsis.logging_utils import configure_logging
from early_sepsis.settings import get_settings, load_config_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic preprocessing pipeline")
    parser.add_argument("--config", default="configs/data_pipeline.yaml", help="Pipeline config YAML")
    parser.add_argument("--raw-path", default=None, help="Raw input path")
    parser.add_argument("--output-dir", default=None, help="Output directory for processed files")
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "physionet"],
        default=None,
        help="Input dataset format",
    )
    parser.add_argument("--train-ratio", type=float, default=None, help="Train patient ratio")
    parser.add_argument("--validation-ratio", type=float, default=None, help="Validation patient ratio")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test patient ratio")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
    parser.add_argument("--strict", action="store_true", help="Strict file validation mode")
    return parser


def _load_pipeline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    try:
        loaded = load_config_file(config_path)
    except ModuleNotFoundError:
        # Keep CLI runnable when optional YAML loader dependencies are absent.
        return {}

    data_config = loaded.get("data_pipeline", {})
    if not isinstance(data_config, dict):
        msg = f"data_pipeline section must be a mapping in {config_path}"
        raise ValueError(msg)
    return data_config


def _coerce_dataset_format(value: str) -> DatasetFormat:
    if value not in {"auto", "csv", "physionet"}:
        msg = f"Invalid dataset format: {value}"
        raise ValueError(msg)
    return cast(DatasetFormat, value)


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    parser = _build_parser()
    args = parser.parse_args()

    config = _load_pipeline_config(args.config)

    raw_path = args.raw_path or str(config.get("raw_data_path", settings.raw_data_dir))
    output_dir = args.output_dir or str(config.get("output_dir", settings.processed_data_dir))
    dataset_format_raw = args.dataset_format or str(config.get("dataset_format", "auto"))
    train_ratio = (
        args.train_ratio
        if args.train_ratio is not None
        else float(config.get("train_ratio", settings.train_ratio))
    )
    validation_ratio = (
        args.validation_ratio
        if args.validation_ratio is not None
        else float(config.get("validation_ratio", settings.validation_ratio))
    )
    test_ratio = (
        args.test_ratio
        if args.test_ratio is not None
        else float(config.get("test_ratio", settings.test_ratio))
    )
    random_seed = (
        args.random_seed
        if args.random_seed is not None
        else int(config.get("random_seed", settings.random_seed))
    )

    result = run_preprocessing_pipeline(
        raw_data_path=raw_path,
        output_dir=output_dir,
        dataset_format=_coerce_dataset_format(dataset_format_raw),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        strict_validation=args.strict,
    )

    summary = {
        "output_dir": str(result.output_dir),
        "metadata_path": str(result.metadata_path),
        "feature_schema_path": str(result.feature_schema_path),
        "split_paths": {name: str(path) for name, path in result.split_paths.items()},
        "manifest_paths": {name: str(path) for name, path in result.manifest_paths.items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
