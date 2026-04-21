from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from early_sepsis.explain.local_llm import explain_predictions
from early_sepsis.logging_utils import configure_logging, get_logger
from early_sepsis.modeling.predict import predict_records
from early_sepsis.runtime_paths import resolve_runtime_path, sanitize_public_path
from early_sepsis.serving.sequence_service import (
    THRESHOLD_MODES,
    OperatingMode,
    SequenceServingError,
    get_selected_model_info,
    predict_sequence_samples,
)
from early_sepsis.settings import get_settings

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
logger = get_logger(__name__)


def _allow_raw_paths() -> bool:
    return settings.environment.strip().lower() == "development"


def _sanitize_path_value(path_value: str) -> str:
    return sanitize_public_path(path_value, allow_raw_paths=_allow_raw_paths())


def _sanitize_path_fields(payload: Any, key_name: str | None = None) -> Any:
    if isinstance(payload, dict):
        return {key: _sanitize_path_fields(value, key) for key, value in payload.items()}

    if isinstance(payload, list):
        return [_sanitize_path_fields(item, key_name) for item in payload]

    if isinstance(payload, str) and key_name is not None:
        normalized_key = key_name.lower()
        if "path" in normalized_key or "dir" in normalized_key:
            return _sanitize_path_value(payload)

    return payload


class PredictionRequest(BaseModel):
    """Inference request body."""

    records: list[dict[str, Any]] = Field(min_length=1)
    include_explanation: bool = False


class PredictionItem(BaseModel):
    """Single prediction payload."""

    sepsis_risk: float
    predicted_label: int


class PredictionResponse(BaseModel):
    """Inference response body."""

    predictions: list[PredictionItem]
    explanations: list[str] | None = None


class SequenceSampleRequest(BaseModel):
    """One sequence sample for manifest-backed sequence inference."""

    patient_id: str | int | None = None
    end_hour: int | None = None
    features: list[list[float]]
    missing_mask: list[list[float]] | None = None
    static_features: list[float] | None = None


class SequencePredictionRequest(BaseModel):
    """Batch request for selected sequence model inference."""

    dataset_tag: str = Field(min_length=1)
    operating_mode: str = Field(default="default")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    samples: list[SequenceSampleRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_threshold_selection(self) -> SequencePredictionRequest:
        if self.threshold is None:
            return self

        if self.operating_mode.strip().lower() != "default":
            msg = (
                "Provide either operating_mode or explicit threshold. "
                "When threshold is provided, operating_mode must be 'default'."
            )
            raise ValueError(msg)

        return self


class SequencePredictionItem(BaseModel):
    """Single sequence prediction output item."""

    patient_id: str | int | None = None
    end_hour: int | None = None
    predicted_probability: float
    predicted_label: int
    threshold_used: float
    operating_mode: OperatingMode


class SequencePredictionResponse(BaseModel):
    """Response for selected sequence model inference."""

    predictions: list[SequencePredictionItem]
    selected_model_manifest_path: str


def create_app() -> FastAPI:
    """Creates a FastAPI app for model inference."""

    api = FastAPI(
        title="Early Sepsis Detection API",
        version="0.1.0",
        description="Inference API for early sepsis risk scoring.",
    )

    @api.get("/health")
    def health() -> dict[str, Any]:
        selected_manifest_path = resolve_runtime_path(settings.selected_sequence_manifest_path)
        tabular_model_path = resolve_runtime_path(settings.model_artifact_path)
        selected_sequence_payload: dict[str, Any] = {
            "manifest_path": _sanitize_path_value(str(selected_manifest_path)),
            "manifest_exists": selected_manifest_path.exists(),
            "available": False,
            "checkpoint_path": None,
            "checkpoint_exists": False,
        }

        if selected_manifest_path.exists():
            try:
                manifest_payload = get_selected_model_info(selected_manifest_path)
                selected_run_payload = manifest_payload.get("selected_run", {})
                checkpoint_path = None
                if isinstance(selected_run_payload, dict):
                    checkpoint_path = selected_run_payload.get("checkpoint_path")

                checkpoint_exists = bool(manifest_payload.get("checkpoint_exists", False))
                selected_sequence_payload.update(
                    {
                        "available": checkpoint_exists,
                        "checkpoint_path": (
                            _sanitize_path_value(checkpoint_path)
                            if isinstance(checkpoint_path, str)
                            else checkpoint_path
                        ),
                        "checkpoint_exists": checkpoint_exists,
                    }
                )
            except (ValueError, KeyError) as exc:
                selected_sequence_payload["error"] = str(exc)

        return {
            "status": "ok",
            "model_available": bool(selected_sequence_payload["available"]),
            "selected_sequence_model": selected_sequence_payload,
            "tabular_model": {
                "artifact_path": _sanitize_path_value(str(tabular_model_path)),
                "available": tabular_model_path.exists(),
            },
            "default_operating_mode": settings.serving_default_operating_mode,
            "environment": settings.environment,
        }

    @api.get("/model-info")
    def model_info() -> dict[str, Any]:
        selected_manifest_path = resolve_runtime_path(settings.selected_sequence_manifest_path)
        tabular_model_path = resolve_runtime_path(settings.model_artifact_path)
        selected_model_payload: dict[str, Any] = {
            "manifest_path": _sanitize_path_value(str(selected_manifest_path)),
            "manifest_exists": selected_manifest_path.exists(),
            "available": False,
        }

        if selected_manifest_path.exists():
            try:
                manifest_payload = get_selected_model_info(selected_manifest_path)
                selected_model_payload["manifest"] = _sanitize_path_fields(manifest_payload)
                selected_model_payload["available"] = bool(
                    manifest_payload.get("checkpoint_exists", False)
                )

                manifest_thresholds = manifest_payload.get("thresholds", {})
                if isinstance(manifest_thresholds, dict):
                    selected_model_payload["threshold_modes"] = {
                        "available": list(THRESHOLD_MODES),
                        "configured_default_mode": settings.serving_default_operating_mode,
                        "configured_default_threshold": manifest_thresholds.get(
                            settings.serving_default_operating_mode,
                            manifest_thresholds.get("default"),
                        ),
                    }
            except (ValueError, KeyError) as exc:
                selected_model_payload["error"] = str(exc)

        return {
            "tabular_model": {
                "artifact_path": _sanitize_path_value(str(tabular_model_path)),
                "available": tabular_model_path.exists(),
            },
            "selected_sequence_model": selected_model_payload,
            "default_operating_mode": settings.serving_default_operating_mode,
            "environment": settings.environment,
        }

    @api.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        model_artifact_path = resolve_runtime_path(settings.model_artifact_path)
        if not model_artifact_path.exists():
            msg = (
                f"Model artifact not found at {model_artifact_path}. "
                "Train a model before requesting predictions."
            )
            raise HTTPException(status_code=503, detail=msg)

        try:
            raw_predictions = predict_records(
                records=payload.records,
                model_path=model_artifact_path,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected prediction error")
            raise HTTPException(status_code=500, detail="Prediction request failed.") from exc

        risk_scores = [float(item["sepsis_risk"]) for item in raw_predictions]
        explanations: list[str] | None = None
        if payload.include_explanation:
            explanations = explain_predictions(
                records=payload.records,
                risk_scores=risk_scores,
                settings=settings,
            )

        return PredictionResponse(
            predictions=[PredictionItem(**item) for item in raw_predictions],
            explanations=explanations,
        )

    @api.post("/predict-sequence", response_model=SequencePredictionResponse)
    def predict_sequence(payload: SequencePredictionRequest) -> SequencePredictionResponse:
        selected_manifest_path = resolve_runtime_path(settings.selected_sequence_manifest_path)
        if not selected_manifest_path.exists():
            msg = (
                "Selected sequence model manifest is missing at "
                f"{selected_manifest_path}. "
                "Run model selection before serving sequence requests."
            )
            raise HTTPException(status_code=503, detail=msg)

        operating_mode = payload.operating_mode
        if payload.threshold is None and payload.operating_mode.strip().lower() == "default":
            operating_mode = settings.serving_default_operating_mode

        try:
            raw_predictions = predict_sequence_samples(
                manifest_path=selected_manifest_path,
                dataset_tag=payload.dataset_tag,
                samples=[sample.model_dump(mode="python") for sample in payload.samples],
                operating_mode=operating_mode,
                threshold_override=payload.threshold,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (SequenceServingError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected sequence prediction error")
            raise HTTPException(
                status_code=500, detail="Sequence prediction request failed."
            ) from exc

        return SequencePredictionResponse(
            predictions=[SequencePredictionItem(**item) for item in raw_predictions],
            selected_model_manifest_path=_sanitize_path_value(str(selected_manifest_path)),
        )

    return api


app = create_app()
