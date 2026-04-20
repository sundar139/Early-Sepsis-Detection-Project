from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from math import isclose
from pathlib import Path
from typing import Any

SUPPORTED_MODEL_TYPES = frozenset({"gru", "lstm", "patchtst"})
REQUIRED_THRESHOLD_KEYS = ("default", "balanced", "high_recall")


class ModelManifestValidationError(ValueError):
    """Raised when a selected-model manifest is malformed."""


def build_feature_signature(feature_columns: list[str]) -> str:
    """Builds a deterministic hash signature for a feature-column ordering."""

    payload = "||".join(feature_columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{name} must be an object"
        raise ModelManifestValidationError(msg)
    return value


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        msg = f"{name} is missing required keys: {missing}"
        raise ModelManifestValidationError(msg)


def _canonical_model_family(model_type: str) -> str:
    return f"{model_type}_classifier"


def _load_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        msg = f"{name} file not found: {resolved_path}"
        raise FileNotFoundError(msg)

    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        msg = f"{name} file must contain a JSON object: {resolved_path}"
        raise ModelManifestValidationError(msg)
    return payload


def _normalize_threshold_mapping(mapping: dict[str, Any], *, name: str) -> dict[str, float]:
    _require_keys(mapping, REQUIRED_THRESHOLD_KEYS, name)

    normalized: dict[str, float] = {}
    for threshold_key in REQUIRED_THRESHOLD_KEYS:
        threshold_value = mapping[threshold_key]
        if not isinstance(threshold_value, (float, int)):
            msg = f"{name}.{threshold_key} must be numeric"
            raise ModelManifestValidationError(msg)

        resolved_value = float(threshold_value)
        if not 0.0 <= resolved_value <= 1.0:
            msg = f"{name}.{threshold_key} must be in [0, 1]"
            raise ModelManifestValidationError(msg)

        normalized[threshold_key] = resolved_value

    return normalized


def load_threshold_recommendations(path: str | Path) -> dict[str, float]:
    """Loads calibration threshold recommendations and validates supported keys."""

    payload = _load_json_object(path, name="threshold recommendations")
    normalized = _normalize_threshold_mapping(payload, name="threshold recommendations")

    high_recall_target = payload.get("high_recall_target")
    if high_recall_target is not None:
        if not isinstance(high_recall_target, (float, int)):
            msg = "threshold recommendations.high_recall_target must be numeric"
            raise ModelManifestValidationError(msg)
        resolved_target = float(high_recall_target)
        if not 0.0 <= resolved_target <= 1.0:
            msg = "threshold recommendations.high_recall_target must be in [0, 1]"
            raise ModelManifestValidationError(msg)
        normalized["high_recall_target"] = resolved_target

    return normalized


def _validate_calibration_summary_against_manifest(
    *,
    manifest: dict[str, Any],
    recommendations: dict[str, float],
    calibration_summary: dict[str, Any],
) -> None:
    summary_checkpoint_value = calibration_summary.get("checkpoint_path")
    if not isinstance(summary_checkpoint_value, str):
        msg = "calibration summary checkpoint_path must be a string"
        raise ModelManifestValidationError(msg)

    manifest_checkpoint_path = Path(str(manifest["selected_run"]["checkpoint_path"])).resolve()
    summary_checkpoint_path = Path(summary_checkpoint_value).resolve()
    if summary_checkpoint_path != manifest_checkpoint_path:
        msg = (
            "Calibration summary checkpoint_path does not match selected model checkpoint: "
            f"{summary_checkpoint_path} != {manifest_checkpoint_path}"
        )
        raise ModelManifestValidationError(msg)

    summary_recommendations = calibration_summary.get("recommended_thresholds")
    if not isinstance(summary_recommendations, dict):
        msg = "calibration summary recommended_thresholds must be an object"
        raise ModelManifestValidationError(msg)

    normalized_summary_thresholds = _normalize_threshold_mapping(
        summary_recommendations,
        name="calibration summary recommended_thresholds",
    )

    for threshold_key in REQUIRED_THRESHOLD_KEYS:
        if not isclose(
            normalized_summary_thresholds[threshold_key],
            recommendations[threshold_key],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            msg = (
                "Calibration summary and threshold recommendation values differ for "
                f"{threshold_key!r}"
            )
            raise ModelManifestValidationError(msg)

    expected_high_recall_target = recommendations.get("high_recall_target")
    if expected_high_recall_target is None:
        return

    summary_high_recall_target = summary_recommendations.get("high_recall_target")
    if not isinstance(summary_high_recall_target, (float, int)):
        msg = "calibration summary recommended_thresholds.high_recall_target must be numeric"
        raise ModelManifestValidationError(msg)

    if not isclose(
        float(summary_high_recall_target),
        expected_high_recall_target,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        msg = "Calibration summary high_recall_target does not match recommendations"
        raise ModelManifestValidationError(msg)


def validate_model_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validates selected-model manifest structure and invariants."""

    _require_keys(
        manifest,
        (
            "schema_version",
            "selected_at",
            "selection_metric",
            "selected_run",
            "dataset",
            "model",
            "thresholds",
            "metrics",
        ),
        "manifest",
    )

    selected_run = _require_mapping(manifest["selected_run"], "selected_run")
    _require_keys(
        selected_run,
        (
            "run_name",
            "run_dir",
            "checkpoint_path",
            "run_config_path",
            "model_type",
        ),
        "selected_run",
    )

    dataset = _require_mapping(manifest["dataset"], "dataset")
    _require_keys(
        dataset,
        (
            "dataset_tag",
            "dataset_format",
            "raw_data_path",
            "windows_dir",
            "processed_dir",
            "feature_columns",
            "mask_columns",
            "static_feature_columns",
            "feature_signature",
        ),
        "dataset",
    )

    model = _require_mapping(manifest["model"], "model")
    _require_keys(
        model,
        (
            "model_type",
            "input_dim",
            "static_dim",
            "window_length",
            "include_mask",
            "include_static",
        ),
        "model",
    )

    thresholds = _require_mapping(manifest["thresholds"], "thresholds")
    manifest["thresholds"] = _normalize_threshold_mapping(thresholds, name="thresholds")
    thresholds = manifest["thresholds"]

    metrics = _require_mapping(manifest["metrics"], "metrics")
    _require_keys(metrics, ("validation", "test"), "metrics")

    threshold_metadata = manifest.get("threshold_metadata")
    if threshold_metadata is not None:
        metadata = _require_mapping(threshold_metadata, "threshold_metadata")
        high_recall_target = metadata.get("high_recall_target")
        if high_recall_target is not None:
            if not isinstance(high_recall_target, (float, int)):
                msg = "threshold_metadata.high_recall_target must be numeric"
                raise ModelManifestValidationError(msg)
            if not 0.0 <= float(high_recall_target) <= 1.0:
                msg = "threshold_metadata.high_recall_target must be in [0, 1]"
                raise ModelManifestValidationError(msg)

    if not isinstance(dataset["feature_columns"], list) or not all(
        isinstance(column, str) for column in dataset["feature_columns"]
    ):
        msg = "dataset.feature_columns must be a list of strings"
        raise ModelManifestValidationError(msg)

    if not isinstance(dataset["static_feature_columns"], list) or not all(
        isinstance(column, str) for column in dataset["static_feature_columns"]
    ):
        msg = "dataset.static_feature_columns must be a list of strings"
        raise ModelManifestValidationError(msg)

    expected_signature = build_feature_signature(dataset["feature_columns"])
    if dataset["feature_signature"] != expected_signature:
        msg = "dataset.feature_signature does not match dataset.feature_columns"
        raise ModelManifestValidationError(msg)

    input_dim = model["input_dim"]
    static_dim = model["static_dim"]
    window_length = model["window_length"]

    if not isinstance(input_dim, int) or input_dim <= 0:
        msg = "model.input_dim must be a positive integer"
        raise ModelManifestValidationError(msg)
    if not isinstance(static_dim, int) or static_dim < 0:
        msg = "model.static_dim must be a non-negative integer"
        raise ModelManifestValidationError(msg)
    if not isinstance(window_length, int) or window_length <= 0:
        msg = "model.window_length must be a positive integer"
        raise ModelManifestValidationError(msg)

    if len(dataset["feature_columns"]) != input_dim:
        msg = (
            "dataset.feature_columns length does not match model.input_dim: "
            f"{len(dataset['feature_columns'])} != {input_dim}"
        )
        raise ModelManifestValidationError(msg)

    include_static = model["include_static"]
    if not isinstance(include_static, bool):
        msg = "model.include_static must be a boolean"
        raise ModelManifestValidationError(msg)

    if include_static and len(dataset["static_feature_columns"]) != static_dim:
        msg = (
            "dataset.static_feature_columns length does not match model.static_dim: "
            f"{len(dataset['static_feature_columns'])} != {static_dim}"
        )
        raise ModelManifestValidationError(msg)

    if not include_static and static_dim != 0:
        msg = "model.static_dim must be 0 when model.include_static is false"
        raise ModelManifestValidationError(msg)

    include_mask = model["include_mask"]
    if not isinstance(include_mask, bool):
        msg = "model.include_mask must be a boolean"
        raise ModelManifestValidationError(msg)

    selected_model_type = str(selected_run["model_type"]).strip().lower()
    model_model_type = str(model["model_type"]).strip().lower()
    if selected_model_type not in SUPPORTED_MODEL_TYPES:
        msg = f"selected_run.model_type is unsupported: {selected_model_type!r}"
        raise ModelManifestValidationError(msg)
    if model_model_type not in SUPPORTED_MODEL_TYPES:
        msg = f"model.model_type is unsupported: {model_model_type!r}"
        raise ModelManifestValidationError(msg)
    if selected_model_type != model_model_type:
        msg = (
            "selected_run.model_type must match model.model_type: "
            f"{selected_model_type!r} != {model_model_type!r}"
        )
        raise ModelManifestValidationError(msg)

    expected_model_family = _canonical_model_family(selected_model_type)
    selected_model_family = selected_run.get("model_family")
    if (
        selected_model_family is not None
        and str(selected_model_family).strip() != expected_model_family
    ):
        msg = (
            "selected_run.model_family must match canonical family derived from model_type: "
            f"{expected_model_family!r}"
        )
        raise ModelManifestValidationError(msg)

    model_family = model.get("model_family")
    if model_family is not None and str(model_family).strip() != expected_model_family:
        msg = (
            "model.model_family must match canonical family derived from model_type: "
            f"{expected_model_family!r}"
        )
        raise ModelManifestValidationError(msg)

    return manifest


def load_model_manifest(path: str | Path) -> dict[str, Any]:
    """Loads and validates a selected-model manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        msg = f"Model manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    payload = _load_json_object(manifest_path, name="manifest")

    return validate_model_manifest(payload)


def save_model_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Validates and persists a selected-model manifest."""

    validated_manifest = validate_model_manifest(manifest)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(validated_manifest, handle, indent=2)

    return manifest_path


def update_manifest_thresholds(
    path: str | Path,
    *,
    default_threshold: float,
    balanced_threshold: float,
    high_recall_threshold: float,
) -> dict[str, Any]:
    """Updates operating thresholds in an existing selected-model manifest."""

    manifest = load_model_manifest(path)
    manifest["thresholds"] = {
        "default": float(default_threshold),
        "balanced": float(balanced_threshold),
        "high_recall": float(high_recall_threshold),
    }

    save_model_manifest(path, manifest)
    return manifest


def sync_manifest_thresholds_from_calibration(
    manifest_path: str | Path,
    recommendations_path: str | Path,
    *,
    calibration_summary_path: str | Path | None = None,
    write_changes: bool = True,
) -> dict[str, Any]:
    """Synchronizes manifest thresholds from calibration recommendation artifacts safely."""

    manifest = load_model_manifest(manifest_path)
    recommendations = load_threshold_recommendations(recommendations_path)

    resolved_summary_path: Path | None = None
    if calibration_summary_path is not None:
        resolved_summary_path = Path(calibration_summary_path)
        summary_payload = _load_json_object(resolved_summary_path, name="calibration summary")
        _validate_calibration_summary_against_manifest(
            manifest=manifest,
            recommendations=recommendations,
            calibration_summary=summary_payload,
        )

    manifest["thresholds"] = {
        "default": recommendations["default"],
        "balanced": recommendations["balanced"],
        "high_recall": recommendations["high_recall"],
    }

    threshold_metadata: dict[str, Any] = {
        "source": "calibration_recommendations",
        "recommendations_path": str(Path(recommendations_path).resolve()),
        "synchronized_at": datetime.now(UTC).isoformat(),
    }
    if resolved_summary_path is not None:
        threshold_metadata["calibration_summary_path"] = str(resolved_summary_path.resolve())
    if "high_recall_target" in recommendations:
        threshold_metadata["high_recall_target"] = recommendations["high_recall_target"]

    manifest["threshold_metadata"] = threshold_metadata

    if write_changes:
        save_model_manifest(manifest_path, manifest)

    return manifest
