from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from early_sepsis.logging_utils import get_logger
from early_sepsis.settings import AppSettings

logger = get_logger(__name__)


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in parameters.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = str(value)
    return normalized


def log_training_run(
    settings: AppSettings,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
    model_path: Path,
) -> None:
    """Logs a training run to MLflow when enabled in settings."""

    if not settings.enable_mlflow:
        return

    if not settings.mlflow_tracking_uri:
        logger.warning("MLflow logging enabled but tracking URI is not configured.")
        return

    try:
        import mlflow
    except ModuleNotFoundError:
        logger.warning("MLflow is not installed. Skipping experiment tracking.")
        return

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run():
            mlflow.log_params(_normalize_parameters(parameters))
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            mlflow.log_artifact(str(model_path))
    except Exception:
        logger.exception("MLflow logging failed.")
