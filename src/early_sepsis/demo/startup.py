from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
