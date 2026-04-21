from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from early_sepsis.modeling.model_manifest import load_model_manifest
from early_sepsis.modeling.sequence_pipeline import load_model_from_checkpoint
from early_sepsis.runtime_paths import resolve_runtime_path

OperatingMode = Literal["default", "balanced", "high_recall"]
THRESHOLD_MODES: tuple[OperatingMode, ...] = ("default", "balanced", "high_recall")


class SequenceServingError(ValueError):
    """Raised for invalid sequence-serving requests or incompatible artifacts."""


@dataclass(slots=True)
class SelectedSequenceRuntime:
    manifest: dict[str, Any]
    model: nn.Module
    device: torch.device


def _as_float_matrix(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        msg = f"{field_name} must be a 2D array"
        raise SequenceServingError(msg)
    return array


def _as_float_vector(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        msg = f"{field_name} must be a 1D array"
        raise SequenceServingError(msg)
    return array


def resolve_operating_mode(operating_mode: str) -> OperatingMode:
    """Validates and normalizes operating mode values for serving requests."""

    normalized_mode = operating_mode.strip().lower()
    if normalized_mode not in THRESHOLD_MODES:
        msg = "operating_mode must be one of: default, balanced, high_recall"
        raise SequenceServingError(msg)
    return cast(OperatingMode, normalized_mode)


def _threshold_for_mode(manifest: dict[str, Any], operating_mode: OperatingMode) -> float:
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        msg = "Manifest thresholds payload is invalid"
        raise SequenceServingError(msg)

    if operating_mode not in thresholds:
        msg = f"Unsupported operating mode: {operating_mode}"
        raise SequenceServingError(msg)

    threshold_value = thresholds[operating_mode]
    if not isinstance(threshold_value, (float, int)):
        msg = f"Manifest threshold value for mode {operating_mode!r} must be numeric"
        raise SequenceServingError(msg)

    resolved_threshold = float(threshold_value)
    if not 0.0 <= resolved_threshold <= 1.0:
        msg = f"Manifest threshold value for mode {operating_mode!r} must be in [0, 1]"
        raise SequenceServingError(msg)

    return resolved_threshold


def resolve_operating_threshold(
    manifest: dict[str, Any],
    *,
    operating_mode: str = "default",
    threshold_override: float | None = None,
) -> tuple[float, OperatingMode]:
    """Resolves inference threshold from operating mode or explicit request override."""

    resolved_mode = resolve_operating_mode(operating_mode)

    if threshold_override is None:
        return _threshold_for_mode(manifest, resolved_mode), resolved_mode

    resolved_threshold = float(threshold_override)
    if not 0.0 <= resolved_threshold <= 1.0:
        msg = "threshold_override must be in [0, 1]"
        raise SequenceServingError(msg)

    return resolved_threshold, resolved_mode


def validate_sequence_samples(
    *,
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    dataset_tag: str,
) -> list[dict[str, Any]]:
    """Validates dataset tag and sample dimensions against selected manifest."""

    if not samples:
        msg = "samples cannot be empty"
        raise SequenceServingError(msg)

    expected_dataset_tag = str(manifest["dataset"]["dataset_tag"])
    if dataset_tag != expected_dataset_tag:
        msg = (
            f"Dataset tag mismatch: request={dataset_tag!r} manifest={expected_dataset_tag!r}. "
            "Use a checkpoint selected for the same dataset tag."
        )
        raise SequenceServingError(msg)

    model_metadata = manifest["model"]
    include_mask = bool(model_metadata["include_mask"])
    include_static = bool(model_metadata["include_static"])
    expected_window_length = int(model_metadata["window_length"])
    expected_input_dim = int(model_metadata["input_dim"])
    expected_static_dim = int(model_metadata["static_dim"])

    normalized_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        features = _as_float_matrix(sample.get("features"), field_name=f"samples[{index}].features")
        if features.shape[0] != expected_window_length:
            msg = (
                f"samples[{index}].features has window length "
                f"{features.shape[0]} but model expects "
                f"{expected_window_length}"
            )
            raise SequenceServingError(msg)
        if features.shape[1] != expected_input_dim:
            msg = (
                f"samples[{index}].features has feature dimension "
                f"{features.shape[1]} but model expects "
                f"{expected_input_dim}"
            )
            raise SequenceServingError(msg)

        missing_mask_value = sample.get("missing_mask")
        if include_mask:
            if missing_mask_value is None:
                msg = f"samples[{index}].missing_mask is required for this model"
                raise SequenceServingError(msg)
            missing_mask = _as_float_matrix(
                missing_mask_value, field_name=f"samples[{index}].missing_mask"
            )
            if missing_mask.shape != features.shape:
                msg = (
                    f"samples[{index}].missing_mask shape "
                    f"{tuple(missing_mask.shape)} does not match "
                    f"features shape {tuple(features.shape)}"
                )
                raise SequenceServingError(msg)
        else:
            missing_mask = None

        static_value = sample.get("static_features")
        if include_static and expected_static_dim > 0:
            if static_value is None:
                msg = f"samples[{index}].static_features is required for this model"
                raise SequenceServingError(msg)
            static_features = _as_float_vector(
                static_value,
                field_name=f"samples[{index}].static_features",
            )
            if static_features.shape[0] != expected_static_dim:
                msg = (
                    f"samples[{index}].static_features length "
                    f"{static_features.shape[0]} does not match "
                    f"expected static_dim {expected_static_dim}"
                )
                raise SequenceServingError(msg)
        else:
            static_features = None

        normalized_samples.append(
            {
                "patient_id": sample.get("patient_id"),
                "end_hour": sample.get("end_hour"),
                "features": features,
                "missing_mask": missing_mask,
                "static_features": static_features,
            }
        )

    return normalized_samples


@lru_cache(maxsize=4)
def _load_selected_sequence_runtime(manifest_path: str) -> SelectedSequenceRuntime:
    resolved_manifest_path = resolve_runtime_path(manifest_path)
    manifest = load_model_manifest(resolved_manifest_path)
    checkpoint_path = resolve_runtime_path(
        manifest["selected_run"]["checkpoint_path"],
        anchor=resolved_manifest_path.parent,
    )

    if not checkpoint_path.exists():
        msg = f"Selected checkpoint is missing: {checkpoint_path}"
        raise FileNotFoundError(msg)

    model, checkpoint, device = load_model_from_checkpoint(checkpoint_path=checkpoint_path)

    manifest_model = manifest["model"]
    expected_window_length = int(manifest_model["window_length"])
    expected_input_dim = int(manifest_model["input_dim"])
    expected_static_dim = int(manifest_model["static_dim"])

    actual_window_length = int(checkpoint.get("sequence_length", -1))
    actual_input_dim = int(checkpoint.get("input_dim", -1))
    actual_static_dim = int(checkpoint.get("static_dim", -1))

    if (
        expected_window_length != actual_window_length
        or expected_input_dim != actual_input_dim
        or expected_static_dim != actual_static_dim
    ):
        msg = (
            "Selected model manifest does not match checkpoint dimensions: "
            "manifest("
            f"window={expected_window_length}, input={expected_input_dim}, "
            f"static={expected_static_dim}) "
            "checkpoint("
            f"window={actual_window_length}, input={actual_input_dim}, "
            f"static={actual_static_dim})"
        )
        raise SequenceServingError(msg)

    return SelectedSequenceRuntime(manifest=manifest, model=model, device=device)


def clear_sequence_runtime_cache() -> None:
    """Clears cached selected-model runtime state."""

    _load_selected_sequence_runtime.cache_clear()


def get_selected_model_info(manifest_path: str | Path) -> dict[str, Any]:
    """Returns model metadata for serving and observability endpoints."""

    resolved_manifest_path = resolve_runtime_path(manifest_path)
    manifest = load_model_manifest(resolved_manifest_path)
    checkpoint_path = resolve_runtime_path(
        manifest["selected_run"]["checkpoint_path"],
        anchor=resolved_manifest_path.parent,
    )

    return {
        "schema_version": manifest["schema_version"],
        "selected_at": manifest["selected_at"],
        "selection_metric": manifest["selection_metric"],
        "selected_run": manifest["selected_run"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "thresholds": manifest["thresholds"],
        "threshold_modes": list(THRESHOLD_MODES),
        "threshold_metadata": manifest.get("threshold_metadata", {}),
        "checkpoint_exists": checkpoint_path.exists(),
    }


def predict_sequence_samples(
    *,
    manifest_path: str | Path,
    dataset_tag: str,
    samples: list[dict[str, Any]],
    operating_mode: str = "default",
    threshold_override: float | None = None,
) -> list[dict[str, Any]]:
    """Runs batch sequence inference against the selected manifest checkpoint."""

    resolved_manifest_path = resolve_runtime_path(manifest_path)
    runtime = _load_selected_sequence_runtime(str(resolved_manifest_path))
    manifest = runtime.manifest

    threshold, resolved_mode = resolve_operating_threshold(
        manifest,
        operating_mode=operating_mode,
        threshold_override=threshold_override,
    )
    validated_samples = validate_sequence_samples(
        samples=samples,
        manifest=manifest,
        dataset_tag=dataset_tag,
    )

    feature_batch = np.stack([sample["features"] for sample in validated_samples], axis=0).astype(
        np.float32
    )
    features_tensor = torch.from_numpy(feature_batch).to(device=runtime.device, dtype=torch.float32)

    mask_tensor: Tensor | None = None
    if bool(manifest["model"]["include_mask"]):
        mask_batch = np.stack(
            [sample["missing_mask"] for sample in validated_samples], axis=0
        ).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_batch).to(device=runtime.device, dtype=torch.float32)

    static_tensor: Tensor | None = None
    if bool(manifest["model"]["include_static"]) and int(manifest["model"]["static_dim"]) > 0:
        static_batch = np.stack(
            [sample["static_features"] for sample in validated_samples], axis=0
        ).astype(np.float32)
        static_tensor = torch.from_numpy(static_batch).to(
            device=runtime.device, dtype=torch.float32
        )

    with torch.no_grad():
        logits = runtime.model(
            features=features_tensor, missing_mask=mask_tensor, static_features=static_tensor
        )
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)

    outputs: list[dict[str, Any]] = []
    for sample, probability in zip(validated_samples, probabilities, strict=True):
        probability_value = float(probability)
        outputs.append(
            {
                "patient_id": sample["patient_id"],
                "end_hour": sample["end_hour"],
                "predicted_probability": probability_value,
                "predicted_label": int(probability_value >= threshold),
                "threshold_used": float(threshold),
                "operating_mode": resolved_mode,
            }
        )

    return outputs
