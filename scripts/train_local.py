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
from early_sepsis.logging_utils import configure_logging, get_logger
from early_sepsis.modeling.train import train_and_save_model
from early_sepsis.settings import get_settings, load_config_file

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline early sepsis model")
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--data-path", default=None, help="Dataset file or directory path")
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "physionet"],
        default=None,
        help="Input dataset format",
    )
    parser.add_argument("--target-column", default=None, help="Target column name")
    parser.add_argument("--output-path", default=None, help="Model output path")
    parser.add_argument("--test-size", type=float, default=None, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed")
    return parser


def _coerce_dataset_format(value: str) -> DatasetFormat:
    if value not in {"auto", "csv", "physionet"}:
        msg = f"Invalid dataset format: {value}"
        raise ValueError(msg)
    return cast(DatasetFormat, value)


def _load_training_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        loaded = load_config_file(path)
    except ModuleNotFoundError:
        # Keep script runnable when optional YAML loader dependencies are absent.
        return {}

    training = loaded.get("training", {})
    if not isinstance(training, dict):
        msg = f"training key must map to a dictionary in {path}"
        raise ValueError(msg)

    return training


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    parser = _build_parser()
    args = parser.parse_args()

    training_config = _load_training_config(args.config)

    data_path = args.data_path or str(training_config.get("data_path", "tests/fixtures/synthetic_tabular.csv"))
    dataset_format_raw = args.dataset_format or str(training_config.get("dataset_format", "auto"))
    target_column = args.target_column or str(
        training_config.get("target_column", settings.default_target_column)
    )
    output_path = args.output_path or str(
        training_config.get("model_output_path", settings.model_artifact_path)
    )
    test_size = (
        args.test_size
        if args.test_size is not None
        else float(training_config.get("test_size", settings.train_test_split_ratio))
    )
    random_state = (
        args.random_state
        if args.random_state is not None
        else int(training_config.get("random_state", settings.random_seed))
    )

    result = train_and_save_model(
        data_path=data_path,
        model_output_path=output_path,
        dataset_format=_coerce_dataset_format(dataset_format_raw),
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        settings=settings,
    )

    print(
        json.dumps(
            {
                "model_path": str(result.model_path),
                "metrics": result.metrics,
                "feature_count": result.feature_count,
                "row_count": result.row_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
