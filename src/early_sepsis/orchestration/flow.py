from __future__ import annotations

from typing import Any

from prefect import flow, task

from early_sepsis.data.ingestion import DatasetFormat
from early_sepsis.modeling.train import train_and_save_model
from early_sepsis.settings import get_settings


@task(name="train-sepsis-model")
def _train_task(
    data_path: str,
    dataset_format: DatasetFormat,
    target_column: str,
    model_output_path: str,
) -> dict[str, Any]:
    result = train_and_save_model(
        data_path=data_path,
        dataset_format=dataset_format,
        target_column=target_column,
        model_output_path=model_output_path,
    )
    return {
        "model_path": str(result.model_path),
        "metrics": result.metrics,
        "feature_count": result.feature_count,
        "row_count": result.row_count,
    }


@flow(name="early-sepsis-training-flow")
def run_training_flow(
    data_path: str,
    dataset_format: DatasetFormat = "auto",
    target_column: str | None = None,
    model_output_path: str | None = None,
) -> dict[str, Any]:
    """Runs training via Prefect flow orchestration."""

    settings = get_settings()
    resolved_target = target_column or settings.default_target_column
    resolved_model_path = model_output_path or str(settings.model_artifact_path)

    return _train_task(
        data_path=data_path,
        dataset_format=dataset_format,
        target_column=resolved_target,
        model_output_path=resolved_model_path,
    )
