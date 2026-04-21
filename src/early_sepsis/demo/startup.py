from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from early_sepsis.modeling.model_manifest import load_model_manifest
from early_sepsis.runtime_paths import make_portable_path, resolve_runtime_path


class DemoStartupError(RuntimeError):
    """Raised when the Streamlit demo cannot start with the selected sequence artifacts."""


@dataclass(slots=True)
class DemoStartupStatus:
    """Resolved startup artifact metadata for the Streamlit demo."""

    manifest_path: Path
    checkpoint_path: Path
    dataset_tag: str
    model_type: str
    manifest: dict[str, Any]


@dataclass(slots=True)
class DemoInferenceSource:
    """Resolved source for inference input windows in the public demo."""

    source_kind: Literal["parquet", "walkthrough", "unavailable"]
    source_label: str
    parquet_path: Path | None = None
    walkthrough_payload_path: Path | None = None
    reason: str | None = None


def _resolve_public_manifest_path(
    *,
    base: str | Path | None,
    public_artifacts_dir: str | Path | None,
) -> Path:
    public_root = resolve_runtime_path(
        public_artifacts_dir or Path("public_artifacts"),
        project_root=base,
    )
    return (public_root / "models" / "registry" / "selected_model.json").resolve()


def _resolve_standard_windows_split_path(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    split: str,
) -> Path | None:
    dataset_section = manifest.get("dataset")
    if not isinstance(dataset_section, dict):
        return None

    windows_dir_value = dataset_section.get("windows_dir")
    if not isinstance(windows_dir_value, str) or not windows_dir_value.strip():
        return None

    windows_dir = resolve_runtime_path(windows_dir_value, anchor=manifest_path.parent)
    return (windows_dir / f"{split}.parquet").resolve()


def resolve_demo_inference_source(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    split: str,
    public_mode: bool,
    bundled_demo_path: str | Path = Path("assets/demo/sequence_demo_samples.parquet"),
    walkthrough_payload_path: str | Path = Path("assets/demo/saved_example_payload.json"),
) -> DemoInferenceSource:
    """Resolves inference source order for deployment-safe public demo rendering."""

    bundled_path = resolve_runtime_path(bundled_demo_path, anchor=manifest_path.parent)
    standard_windows_path = _resolve_standard_windows_split_path(
        manifest,
        manifest_path=manifest_path,
        split=split,
    )

    parquet_candidates: list[tuple[str, Path]] = []
    if public_mode:
        parquet_candidates.append(("Bundled demo artifact", bundled_path))
        if standard_windows_path is not None:
            parquet_candidates.append((f"Evaluation {split} windows", standard_windows_path))
    else:
        if standard_windows_path is not None:
            parquet_candidates.append((f"Evaluation {split} windows", standard_windows_path))
        parquet_candidates.append(("Bundled demo artifact", bundled_path))

    for source_label, source_path in parquet_candidates:
        if source_path.exists() and source_path.is_file():
            return DemoInferenceSource(
                source_kind="parquet",
                source_label=source_label,
                parquet_path=source_path,
            )

    resolved_walkthrough_payload = resolve_runtime_path(
        walkthrough_payload_path,
        anchor=manifest_path.parent,
    )
    if resolved_walkthrough_payload.exists() and resolved_walkthrough_payload.is_file():
        return DemoInferenceSource(
            source_kind="walkthrough",
            source_label="Saved Example Walkthrough",
            walkthrough_payload_path=resolved_walkthrough_payload,
            reason=(
                "Live parquet-backed windows are unavailable in this deployment package."
            ),
        )

    return DemoInferenceSource(
        source_kind="unavailable",
        source_label="Inference unavailable",
        reason=(
            "No bundled demo parquet, evaluation windows, or saved walkthrough payload "
            "is available."
        ),
    )


def build_saved_example_walkthrough_sample(
    manifest: dict[str, Any],
    payload_path: str | Path,
) -> dict[str, Any]:
    """Builds one synthetic-safe walkthrough sample from a saved deployment payload."""

    resolved_payload_path = resolve_runtime_path(payload_path)
    payload: dict[str, Any] = {}
    try:
        payload_candidate = json.loads(resolved_payload_path.read_text(encoding="utf-8"))
        if isinstance(payload_candidate, dict):
            payload = payload_candidate
    except (OSError, ValueError):
        payload = {}

    model_section = manifest.get("model", {})
    expected_window = max(int(model_section.get("window_length", 8)), 1)
    expected_input_dim = max(int(model_section.get("input_dim", 1)), 1)
    expected_static_dim = max(int(model_section.get("static_dim", 0)), 0)

    sample_id_value = str(payload.get("sample_id", "DS-EX-001")).strip()
    sample_id = sample_id_value if sample_id_value else "DS-EX-001"
    if not sample_id.startswith("DS-"):
        sample_id = f"DS-{sample_id}"

    end_hour = int(payload.get("end_hour", expected_window))
    label = 1 if int(payload.get("label", 0)) == 1 else 0

    base_value = float(payload.get("base_value", 0.16))
    trend_delta = float(payload.get("trend_delta", 0.22))
    trend_feature_index = int(payload.get("trend_feature_index", 0))
    trend_feature_index = min(max(trend_feature_index, 0), expected_input_dim - 1)

    feature_matrix = np.full((expected_window, expected_input_dim), base_value, dtype=np.float32)
    feature_matrix[:, trend_feature_index] += np.linspace(
        0.0,
        trend_delta,
        expected_window,
        dtype=np.float32,
    )
    missing_mask = np.zeros((expected_window, expected_input_dim), dtype=np.float32)

    static_template = payload.get("static_template")
    static_features: list[float]
    if expected_static_dim > 0:
        if isinstance(static_template, list) and len(static_template) == expected_static_dim:
            static_features = np.asarray(static_template, dtype=np.float32).tolist()
        else:
            static_features = np.zeros((expected_static_dim,), dtype=np.float32).tolist()
    else:
        static_features = []

    walkthrough_note = str(
        payload.get(
            "walkthrough_note",
            "Saved synthetic walkthrough sample bundled for deployment-safe demonstration.",
        )
    )

    return {
        "sample_id": sample_id,
        "label": label,
        "walkthrough_note": walkthrough_note,
        "request_sample": {
            "patient_id": sample_id,
            "end_hour": end_hour,
            "features": feature_matrix.tolist(),
            "missing_mask": missing_mask.tolist(),
            "static_features": static_features,
        },
    }


def resolve_manifest_path(
    path_value: str | Path,
    *,
    base: str | Path | None = None,
    public_artifacts_dir: str | Path | None = None,
) -> Path:
    """Resolves manifest paths for local and deployment runtimes."""

    resolved_path = resolve_runtime_path(path_value, project_root=base)
    if resolved_path.exists():
        return resolved_path

    public_manifest_path = _resolve_public_manifest_path(
        base=base,
        public_artifacts_dir=public_artifacts_dir,
    )
    if public_manifest_path.exists():
        return public_manifest_path

    return resolved_path


def _coerce_float_matrix(value: Any) -> np.ndarray:
    try:
        candidate = np.asarray(value, dtype=np.float32)
        if candidate.ndim == 2:
            return candidate
    except (TypeError, ValueError):
        pass

    object_candidate = np.asarray(value, dtype=object)
    if object_candidate.ndim == 1 and object_candidate.size > 0:
        try:
            return np.stack(
                [np.asarray(row, dtype=np.float32) for row in object_candidate],
                axis=0,
            )
        except (TypeError, ValueError):
            pass

    msg = "Unable to coerce value into a 2D float matrix"
    raise ValueError(msg)


def _is_demo_sample_compatible(frame: pd.DataFrame, manifest: dict[str, Any]) -> bool:
    required_columns = {
        "patient_id",
        "end_hour",
        "label",
        "features",
        "missing_mask",
        "static_features",
    }
    if not required_columns.issubset(frame.columns):
        return False
    if frame.empty:
        return False

    model_section = manifest.get("model", {})
    if not isinstance(model_section, dict):
        return False

    expected_window = int(model_section.get("window_length", 0))
    expected_input_dim = int(model_section.get("input_dim", 0))
    expected_static_dim = int(model_section.get("static_dim", 0))

    first_row = frame.iloc[0]
    try:
        features = _coerce_float_matrix(first_row["features"])
    except ValueError:
        return False
    if features.ndim != 2:
        return False
    if features.shape[0] != expected_window or features.shape[1] != expected_input_dim:
        return False

    static_features = np.asarray(first_row["static_features"], dtype=np.float32)
    if expected_static_dim > 0 and static_features.shape[0] != expected_static_dim:
        return False

    try:
        missing_mask = _coerce_float_matrix(first_row["missing_mask"])
    except ValueError:
        return False
    return missing_mask.shape == features.shape


def ensure_demo_sample_parquet(
    manifest: dict[str, Any],
    sample_path: str | Path,
    *,
    max_rows: int = 16,
    public_fallback_path: str | Path | None = None,
) -> tuple[Path, bool]:
    """Ensures a small safe demo parquet exists and matches manifest dimensions."""

    resolved_sample_path = resolve_runtime_path(sample_path)
    if resolved_sample_path.exists():
        try:
            existing_frame = pd.read_parquet(resolved_sample_path)
            if _is_demo_sample_compatible(existing_frame, manifest):
                return resolved_sample_path, False
        except Exception:
            pass

    if public_fallback_path is not None:
        resolved_public_fallback_path = resolve_runtime_path(public_fallback_path)
        if resolved_public_fallback_path.exists():
            try:
                fallback_frame = pd.read_parquet(resolved_public_fallback_path)
                if _is_demo_sample_compatible(fallback_frame, manifest):
                    return resolved_public_fallback_path, False
            except Exception:
                pass

    model_section = manifest["model"]
    include_static = bool(model_section["include_static"])
    static_dim = int(model_section["static_dim"])
    window_length = int(model_section["window_length"])
    input_dim = int(model_section["input_dim"])

    rows: list[dict[str, Any]] = []
    for index in range(max_rows):
        base_value = 0.05 * (index + 1)
        feature_matrix = np.full((window_length, input_dim), base_value, dtype=np.float32)
        feature_matrix[:, 0] += np.linspace(0.0, 0.2, window_length, dtype=np.float32)

        sample: dict[str, Any] = {
            "patient_id": f"demo_patient_{index + 1:03d}",
            "end_hour": int(window_length + index),
            "label": int(index % 2),
            "features": feature_matrix.tolist(),
            "missing_mask": np.zeros((window_length, input_dim), dtype=np.float32).tolist(),
            "static_features": (
                np.zeros((static_dim,), dtype=np.float32).tolist()
                if include_static and static_dim > 0
                else []
            ),
        }
        rows.append(sample)

    frame = pd.DataFrame(rows)
    try:
        resolved_sample_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(resolved_sample_path, index=False)
        return resolved_sample_path, True
    except OSError:
        fallback_path = (
            Path(tempfile.gettempdir())
            / "early_sepsis_demo"
            / "sequence_demo_samples.parquet"
        )
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(fallback_path, index=False)
        return fallback_path.resolve(), True


def validate_demo_startup(manifest_path: str | Path) -> DemoStartupStatus:
    """Validates selected manifest and checkpoint presence for demo startup."""

    resolved_manifest_path = resolve_manifest_path(manifest_path)
    portable_manifest_path = make_portable_path(resolved_manifest_path)
    if not resolved_manifest_path.exists():
        msg = (
            "Selected sequence manifest was not found. "
            "Set SEPSIS_SELECTED_SEQUENCE_MANIFEST_PATH to a valid selected_model.json. "
            f"Resolved path: {portable_manifest_path}."
        )
        raise DemoStartupError(msg)

    try:
        manifest = load_model_manifest(resolved_manifest_path)
    except Exception as exc:
        msg = (
            "Selected sequence manifest could not be loaded. "
            f"Path: {portable_manifest_path}. {exc}"
        )
        raise DemoStartupError(msg) from exc

    selected_run = manifest.get("selected_run")
    if not isinstance(selected_run, dict):
        msg = "Selected sequence manifest is missing a valid selected_run object."
        raise DemoStartupError(msg)

    checkpoint_value = selected_run.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        msg = "Selected sequence manifest is missing selected_run.checkpoint_path."
        raise DemoStartupError(msg)

    checkpoint_path = resolve_runtime_path(
        checkpoint_value,
        anchor=resolved_manifest_path.parent,
    )
    portable_checkpoint_path = make_portable_path(checkpoint_path)

    if not checkpoint_path.exists():
        msg = (
            "Selected checkpoint file is missing for the selected manifest. "
            "Ensure selected_run.checkpoint_path points to an available checkpoint file. "
            f"Resolved checkpoint path: {portable_checkpoint_path}."
        )
        raise DemoStartupError(msg)

    dataset_section = manifest.get("dataset", {})
    model_section = manifest.get("model", {})
    dataset_tag = str(dataset_section.get("dataset_tag", "unknown"))
    model_type = str(model_section.get("model_type", "unknown"))

    return DemoStartupStatus(
        manifest_path=resolved_manifest_path,
        checkpoint_path=checkpoint_path,
        dataset_tag=dataset_tag,
        model_type=model_type,
        manifest=manifest,
    )
