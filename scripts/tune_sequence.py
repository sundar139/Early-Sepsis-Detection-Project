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
    sequence_model_family_name,
)
from early_sepsis.modeling.sequence_tuning import tune_sequence_model
from early_sepsis.settings import get_settings, load_config_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune sequence models with Optuna")
    parser.add_argument("--config", default="configs/model_tuning.yaml", help="YAML config path")
    parser.add_argument("--windows-dir", default=None, help="Window dataset directory")
    parser.add_argument("--output-dir", default=None, help="Tuning artifact directory")
    parser.add_argument(
        "--model-type",
        choices=["gru", "lstm", "patchtst"],
        default=None,
        help="Model family to tune",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    return parser


def _load_payload(config_path: str | Path) -> dict[str, Any]:
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


def main() -> None:
    app_settings = get_settings()
    configure_logging(level=app_settings.log_level, json_logs=app_settings.json_logs)

    args = _build_parser().parse_args()
    payload = _load_payload(args.config)

    training_section = payload.get("training", {})
    model_section = payload.get("model", {})
    tuning_section = payload.get("tuning", {})

    if not isinstance(training_section, dict):
        msg = "training section must be a mapping"
        raise ValueError(msg)
    if not isinstance(model_section, dict):
        msg = "model section must be a mapping"
        raise ValueError(msg)
    if not isinstance(tuning_section, dict):
        msg = "tuning section must be a mapping"
        raise ValueError(msg)

    training_payload = {
        **training_section,
        "windows_dir": args.windows_dir
        or training_section.get("windows_dir", "artifacts/windows"),
        "output_dir": args.output_dir
        or training_section.get("output_dir", "artifacts/models/sequence_tuning"),
        "epochs": args.epochs if args.epochs is not None else training_section.get("epochs", 8),
        "batch_size": args.batch_size
        if args.batch_size is not None
        else training_section.get("batch_size", 192),
        "mlflow_enabled": False,
    }

    model_payload = {
        **model_section,
        "model_type": args.model_type or model_section.get("model_type", "patchtst"),
    }
    training_payload["model_name"] = sequence_model_family_name(model_payload["model_type"])

    training_config = SequenceTrainingConfig(
        **{
            key: Path(value) if key in {"windows_dir", "output_dir"} else value
            for key, value in training_payload.items()
            if key in SequenceTrainingConfig.__dataclass_fields__
        }
    )
    model_config = SequenceModelConfig(
        **{
            key: value
            for key, value in model_payload.items()
            if key in SequenceModelConfig.__dataclass_fields__
        }
    )

    n_trials = (
        args.n_trials if args.n_trials is not None else int(tuning_section.get("n_trials", 20))
    )
    timeout = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else tuning_section.get("timeout_seconds")
    )
    study_name = args.study_name or str(
        tuning_section.get("study_name", "early-sepsis-sequence-tuning")
    )

    result = tune_sequence_model(
        training_config=training_config,
        model_config=model_config,
        n_trials=n_trials,
        timeout_seconds=int(timeout) if timeout is not None else None,
        study_name=study_name,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
