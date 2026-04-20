from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.logging_utils import configure_logging
from early_sepsis.modeling.sequence_models import SequenceModelConfig
from early_sepsis.modeling.sequence_pipeline import (
    SequenceTrainingConfig,
    merge_model_overrides,
    merge_training_overrides,
    sequence_model_config_from_dict,
    sequence_model_family_name,
    sequence_training_config_from_dict,
    train_sequence_model,
)
from early_sepsis.settings import get_settings, load_config_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sequence model on windowed sepsis data")
    parser.add_argument("--config", default="configs/model_training.yaml", help="YAML config path")
    parser.add_argument("--windows-dir", default=None, help="Window dataset directory")
    parser.add_argument("--output-dir", default=None, help="Model artifact output directory")
    parser.add_argument(
        "--model-type",
        choices=["gru", "lstm", "patchtst"],
        default=None,
        help="Sequence model architecture",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda")
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "pos_weight", "weighted_sampler", "both"],
        default=None,
        help="Class imbalance handling strategy",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--without-mask", action="store_true", help="Do not use missingness mask")
    parser.add_argument("--without-static", action="store_true", help="Do not use static features")
    return parser


def _load_training_payload(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        loaded = load_config_file(path)
    except ModuleNotFoundError:
        return {}

    if not isinstance(loaded, dict):
        msg = f"Invalid config payload in {path}"
        raise ValueError(msg)
    return loaded


def _build_configs(payload: dict[str, Any]) -> tuple[SequenceTrainingConfig, SequenceModelConfig]:
    training_section = payload.get("training", {})
    model_section = payload.get("model", {})
    mlflow_section = payload.get("mlflow", {})

    if not isinstance(training_section, dict):
        msg = "training section must be a mapping"
        raise ValueError(msg)
    if not isinstance(model_section, dict):
        msg = "model section must be a mapping"
        raise ValueError(msg)
    if not isinstance(mlflow_section, dict):
        msg = "mlflow section must be a mapping"
        raise ValueError(msg)

    training_payload = {
        **training_section,
        "mlflow_enabled": mlflow_section.get(
            "enabled",
            training_section.get("mlflow_enabled", True),
        ),
        "mlflow_tracking_uri": mlflow_section.get(
            "tracking_uri", training_section.get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        ),
        "mlflow_experiment_name": mlflow_section.get(
            "experiment_name",
            training_section.get("mlflow_experiment_name", "early-sepsis-sequence"),
        ),
        "mlflow_run_name": mlflow_section.get(
            "run_name", training_section.get("mlflow_run_name")
        ),
    }

    training_config = sequence_training_config_from_dict(training_payload)
    model_config = sequence_model_config_from_dict(model_section)
    return training_config, model_config


def main() -> None:
    app_settings = get_settings()
    configure_logging(level=app_settings.log_level, json_logs=app_settings.json_logs)

    parser = _build_parser()
    args = parser.parse_args()

    payload = _load_training_payload(args.config)
    training_config, model_config = _build_configs(payload)

    model_overrides: dict[str, Any] = {
        "model_type": args.model_type,
    }
    if args.without_mask:
        model_overrides["include_mask"] = False
    if args.without_static:
        model_overrides["include_static"] = False

    training_config = merge_training_overrides(
        training_config,
        windows_dir=Path(args.windows_dir) if args.windows_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        imbalance_strategy=args.imbalance_strategy,
        mlflow_enabled=False if args.disable_mlflow else None,
    )
    model_config = merge_model_overrides(model_config, **model_overrides)
    training_config = merge_training_overrides(
        training_config,
        model_name=sequence_model_family_name(model_config.model_type),
    )

    result = train_sequence_model(training_config=training_config, model_config=model_config)

    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "best_checkpoint": str(result.best_checkpoint_path),
                "last_checkpoint": str(result.last_checkpoint_path),
                "selected_threshold": result.selected_threshold,
                "best_validation_metrics": result.best_validation_metrics,
                "test_metrics": result.test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
