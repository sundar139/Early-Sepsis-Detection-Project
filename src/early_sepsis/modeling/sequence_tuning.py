from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from early_sepsis.logging_utils import get_logger
from early_sepsis.modeling.sequence_models import SequenceModelConfig
from early_sepsis.modeling.sequence_pipeline import (
    SequenceTrainingConfig,
    sequence_model_config_from_dict,
    train_sequence_model,
)

logger = get_logger(__name__)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sample_model_config(trial: optuna.Trial, base_model: SequenceModelConfig) -> SequenceModelConfig:
    if base_model.model_type in {"gru", "lstm"}:
        return replace(
            base_model,
            recurrent_hidden_dim=trial.suggest_categorical("recurrent_hidden_dim", [64, 96, 128, 192]),
            recurrent_num_layers=trial.suggest_int("recurrent_num_layers", 1, 3),
            recurrent_dropout=trial.suggest_float("recurrent_dropout", 0.1, 0.5),
            recurrent_bidirectional=trial.suggest_categorical("recurrent_bidirectional", [True, False]),
        )

    return replace(
        base_model,
        patch_len=trial.suggest_categorical("patch_len", [2, 4, 8]),
        patch_stride=trial.suggest_categorical("patch_stride", [1, 2, 4]),
        patch_d_model=trial.suggest_categorical("patch_d_model", [64, 96, 128, 160]),
        patch_num_heads=trial.suggest_categorical("patch_num_heads", [2, 4, 8]),
        patch_num_layers=trial.suggest_int("patch_num_layers", 2, 4),
        patch_ff_dim=trial.suggest_categorical("patch_ff_dim", [128, 256, 384]),
        patch_dropout=trial.suggest_float("patch_dropout", 0.1, 0.5),
    )


def _sample_training_config(
    trial: optuna.Trial,
    base_training: SequenceTrainingConfig,
    trial_output_dir: Path,
) -> SequenceTrainingConfig:
    return replace(
        base_training,
        output_dir=trial_output_dir,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 192, 256]),
        early_stopping_patience=max(2, min(base_training.early_stopping_patience, base_training.epochs - 1)),
        mlflow_enabled=False,
        mlflow_run_name=None,
    )


def tune_sequence_model(
    training_config: SequenceTrainingConfig,
    model_config: SequenceModelConfig,
    n_trials: int,
    timeout_seconds: int | None = None,
    study_name: str = "early-sepsis-sequence-tuning",
) -> dict[str, Any]:
    """Runs Optuna hyperparameter tuning optimizing validation AUPRC."""

    tuning_dir = training_config.output_dir / "optuna"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_dir = tuning_dir / f"trial_{trial.number:04d}"
        sampled_model = _sample_model_config(trial=trial, base_model=model_config)
        sampled_training = _sample_training_config(
            trial=trial,
            base_training=training_config,
            trial_output_dir=trial_dir,
        )

        result = train_sequence_model(
            training_config=sampled_training,
            model_config=sampled_model,
            trial=trial,
        )

        val_auprc = float(result.best_validation_metrics.get("auprc", 0.0))
        val_auroc = float(result.best_validation_metrics.get("auroc", 0.0))

        trial.set_user_attr("validation_auroc", val_auroc)
        trial.set_user_attr("selected_threshold", result.selected_threshold)
        trial.set_user_attr("run_dir", str(result.run_dir))
        return val_auprc

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    best_payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_validation_auroc": float(study.best_trial.user_attrs.get("validation_auroc", 0.0)),
        "best_trial_selected_threshold": float(
            study.best_trial.user_attrs.get("selected_threshold", training_config.threshold)
        ),
        "best_trial_run_dir": str(study.best_trial.user_attrs.get("run_dir", "")),
        "n_trials": len(study.trials),
    }

    _json_dump(tuning_dir / "best_result.json", best_payload)
    _json_dump(
        tuning_dir / "study_trials.json",
        {
            "trials": [
                {
                    "number": int(trial.number),
                    "state": str(trial.state),
                    "value": float(trial.value) if trial.value is not None else None,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
                for trial in study.trials
            ]
        },
    )

    logger.info(
        "Optuna tuning completed",
        extra={
            "study_name": study_name,
            "best_validation_auprc": best_payload["best_value"],
            "best_params": best_payload["best_params"],
            "trial_count": best_payload["n_trials"],
        },
    )

    return best_payload


def sequence_tuning_from_dict(payload: dict[str, Any]) -> tuple[SequenceTrainingConfig, SequenceModelConfig]:
    """Builds training and model configs from YAML-style payload."""

    training_section = payload.get("training", {})
    model_section = payload.get("model", {})

    if not isinstance(training_section, dict):
        msg = "training section must be a mapping"
        raise ValueError(msg)
    if not isinstance(model_section, dict):
        msg = "model section must be a mapping"
        raise ValueError(msg)

    training_overrides = {
        key: value
        for key, value in training_section.items()
        if key in SequenceTrainingConfig.__dataclass_fields__
    }
    model_overrides = {
        key: value
        for key, value in model_section.items()
        if key in SequenceModelConfig.__dataclass_fields__
    }

    training_config = SequenceTrainingConfig(**training_overrides)
    model_config = sequence_model_config_from_dict(model_overrides)
    return training_config, model_config
