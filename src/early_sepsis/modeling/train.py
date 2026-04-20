from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from early_sepsis.data.ingestion import DatasetFormat, load_dataset
from early_sepsis.data.preprocessing import build_preprocessor, prepare_training_data
from early_sepsis.logging_utils import get_logger
from early_sepsis.modeling.evaluate import evaluate_binary_classifier
from early_sepsis.settings import AppSettings, get_settings
from early_sepsis.tracking.experiment import log_training_run

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingResult:
    """Outcome metadata from a training run."""

    model_path: Path
    metrics: dict[str, float]
    feature_count: int
    row_count: int


def train_and_save_model(
    data_path: str | Path,
    model_output_path: str | Path | None = None,
    dataset_format: DatasetFormat = "auto",
    target_column: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    settings: AppSettings | None = None,
) -> TrainingResult:
    """Trains a baseline logistic regression model and stores the artifact."""

    resolved_settings = settings or get_settings()
    resolved_target_column = target_column or resolved_settings.default_target_column
    resolved_test_size = (
        test_size if test_size is not None else resolved_settings.train_test_split_ratio
    )
    resolved_random_state = random_state if random_state is not None else resolved_settings.random_seed

    dataframe = load_dataset(data_path=data_path, dataset_format=dataset_format)
    prepared = prepare_training_data(
        dataframe=dataframe,
        target_column=resolved_target_column,
        test_size=resolved_test_size,
        random_state=resolved_random_state,
    )

    preprocessor = build_preprocessor(prepared.X_train)
    classifier = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", classifier)])

    pipeline.fit(prepared.X_train, prepared.y_train)
    predictions = pipeline.predict(prepared.X_test)
    probabilities = pipeline.predict_proba(prepared.X_test)[:, 1]
    metrics = evaluate_binary_classifier(
        y_true=prepared.y_test,
        y_pred=predictions,
        y_prob=probabilities,
    )

    output_path = (
        Path(model_output_path) if model_output_path is not None else resolved_settings.model_artifact_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_payload = {
        "pipeline": pipeline,
        "feature_columns": prepared.feature_columns,
        "target_column": resolved_target_column,
    }
    joblib.dump(artifact_payload, output_path)

    training_params = {
        "dataset_format": dataset_format,
        "target_column": resolved_target_column,
        "test_size": resolved_test_size,
        "random_state": resolved_random_state,
        "feature_count": len(prepared.feature_columns),
        "row_count": len(dataframe),
    }
    log_training_run(
        settings=resolved_settings,
        parameters=training_params,
        metrics=metrics,
        model_path=output_path,
    )

    logger.info(
        "Training completed",
        extra={"model_path": str(output_path), "metrics": metrics},
    )

    return TrainingResult(
        model_path=output_path,
        metrics=metrics,
        feature_count=len(prepared.feature_columns),
        row_count=len(dataframe),
    )
