from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class ModelBundle:
    """Container for model artifact payload."""

    pipeline: Pipeline
    feature_columns: list[str]
    target_column: str


def load_model_bundle(model_path: str | Path) -> ModelBundle:
    """Loads and validates a serialized model bundle."""

    path = Path(model_path)
    if not path.exists():
        msg = f"Model artifact not found: {path}"
        raise FileNotFoundError(msg)

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        msg = f"Invalid model artifact payload at: {path}"
        raise ValueError(msg)

    pipeline = payload.get("pipeline")
    feature_columns = payload.get("feature_columns")
    target_column = payload.get("target_column", "SepsisLabel")

    if not isinstance(pipeline, Pipeline):
        msg = "Model artifact is missing a valid sklearn pipeline."
        raise ValueError(msg)
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        msg = "Model artifact is missing valid feature columns."
        raise ValueError(msg)
    if not isinstance(target_column, str):
        msg = "Model artifact is missing a valid target column."
        raise ValueError(msg)

    return ModelBundle(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_column=target_column,
    )


def _records_to_frame(records: Sequence[Mapping[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    if not records:
        msg = "Prediction records cannot be empty."
        raise ValueError(msg)

    frame = pd.DataFrame(records)
    if frame.empty:
        msg = "Prediction records cannot be converted to a dataframe."
        raise ValueError(msg)

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    return frame.loc[:, feature_columns]


def predict_records(
    records: Sequence[Mapping[str, Any]],
    model_path: str | Path,
) -> list[dict[str, float | int]]:
    """Generates risk scores and binary predictions for input records."""

    bundle = load_model_bundle(model_path=model_path)
    feature_frame = _records_to_frame(records=records, feature_columns=bundle.feature_columns)

    if hasattr(bundle.pipeline, "predict_proba"):
        risk_scores = bundle.pipeline.predict_proba(feature_frame)[:, 1]
    else:
        risk_scores = bundle.pipeline.predict(feature_frame).astype(float)

    labels = (risk_scores >= 0.5).astype(int)
    return [
        {
            "sepsis_risk": float(risk_score),
            "predicted_label": int(label),
        }
        for risk_score, label in zip(risk_scores, labels, strict=True)
    ]
