from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from early_sepsis.demo.startup import (
    DemoStartupError,
    ensure_demo_sample_parquet,
    resolve_manifest_path,
    validate_demo_startup,
)
from early_sepsis.modeling.model_manifest import build_feature_signature, save_model_manifest


def _build_manifest_payload(tmp_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    feature_columns = ["HR", "O2Sat", "SBP"]
    static_feature_columns = ["Age", "Gender"]

    return {
        "schema_version": "1.0",
        "selected_at": "2026-01-01T00:00:00+00:00",
        "selection_metric": "validation_auprc",
        "selected_run": {
            "run_name": "run_demo",
            "run_dir": str(checkpoint_path.parent),
            "checkpoint_path": str(checkpoint_path),
            "run_config_path": str(checkpoint_path.parent / "run_config.json"),
            "model_type": "patchtst",
            "model_family": "patchtst_classifier",
        },
        "dataset": {
            "dataset_tag": "physionet",
            "dataset_format": "physionet",
            "raw_data_path": str(tmp_path / "data" / "raw"),
            "windows_dir": str(tmp_path / "windows"),
            "processed_dir": str(tmp_path / "processed"),
            "feature_columns": feature_columns,
            "mask_columns": [f"{item}_mask" for item in feature_columns],
            "static_feature_columns": static_feature_columns,
            "feature_signature": build_feature_signature(feature_columns),
        },
        "model": {
            "model_type": "patchtst",
            "model_family": "patchtst_classifier",
            "input_dim": 3,
            "static_dim": 2,
            "window_length": 8,
            "include_mask": True,
            "include_static": True,
        },
        "thresholds": {
            "default": 0.5,
            "balanced": 0.6,
            "high_recall": 0.4,
        },
        "metrics": {
            "validation": {"auprc": 0.6, "auroc": 0.7},
            "test": {"auprc": 0.58, "auroc": 0.69},
        },
    }


def test_resolve_manifest_path_resolves_relative_paths(tmp_path: Path) -> None:
    relative_path = Path("artifacts/models/registry/selected_model.json")

    resolved_path = resolve_manifest_path(relative_path, base=tmp_path)

    assert resolved_path == (tmp_path / relative_path).resolve()


def test_resolve_manifest_path_uses_public_artifacts_fallback(tmp_path: Path) -> None:
    public_manifest_path = (
        tmp_path / "public_artifacts" / "models" / "registry" / "selected_model.json"
    )
    public_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    public_manifest_path.write_text("{}", encoding="utf-8")

    resolved_path = resolve_manifest_path(
        Path("artifacts/models/registry/selected_model.json"),
        base=tmp_path,
        public_artifacts_dir=tmp_path / "public_artifacts",
    )

    assert resolved_path == public_manifest_path.resolve()


def test_validate_demo_startup_rejects_missing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_manifest_path = tmp_path / "artifacts" / "models" / "registry" / "selected_model.json"

    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(DemoStartupError, match="Selected sequence manifest was not found"):
        validate_demo_startup(missing_manifest_path)


def test_validate_demo_startup_rejects_missing_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "models" / "sequence" / "run_demo" / "best.pt"
    manifest_path = tmp_path / "artifacts" / "models" / "registry" / "selected_model.json"

    payload = _build_manifest_payload(tmp_path, checkpoint_path)
    save_model_manifest(manifest_path, payload)

    with pytest.raises(DemoStartupError, match="Selected checkpoint file is missing"):
        validate_demo_startup(manifest_path)


def test_validate_demo_startup_returns_status_with_valid_artifacts(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "models" / "sequence" / "run_demo" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    manifest_path = tmp_path / "artifacts" / "models" / "registry" / "selected_model.json"
    payload = _build_manifest_payload(tmp_path, checkpoint_path)
    save_model_manifest(manifest_path, payload)

    startup_status = validate_demo_startup(manifest_path)

    assert startup_status.manifest_path == manifest_path.resolve()
    assert startup_status.checkpoint_path == checkpoint_path.resolve()
    assert startup_status.dataset_tag == "physionet"
    assert startup_status.model_type == "patchtst"


def test_validate_demo_startup_supports_project_root_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "artifacts" / "models" / "sequence" / "run_demo" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    manifest_path = tmp_path / "artifacts" / "models" / "registry" / "selected_model.json"
    payload = _build_manifest_payload(tmp_path, checkpoint_path)
    payload["selected_run"]["checkpoint_path"] = "artifacts/models/sequence/run_demo/best.pt"
    save_model_manifest(manifest_path, payload)

    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    startup_status = validate_demo_startup(Path("artifacts/models/registry/selected_model.json"))

    assert startup_status.manifest_path == manifest_path.resolve()
    assert startup_status.checkpoint_path == checkpoint_path.resolve()


def test_ensure_demo_sample_parquet_creates_and_reuses_compatible_sample(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "models" / "sequence" / "run_demo" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    manifest_path = tmp_path / "artifacts" / "models" / "registry" / "selected_model.json"
    payload = _build_manifest_payload(tmp_path, checkpoint_path)
    save_model_manifest(manifest_path, payload)

    sample_path = tmp_path / "data" / "demo" / "sequence_demo_samples.parquet"
    created_path, created = ensure_demo_sample_parquet(payload, sample_path)

    assert created is True
    assert created_path == sample_path.resolve()
    assert created_path.exists()

    frame = pd.read_parquet(created_path)
    assert len(frame) == 16
    assert {"patient_id", "features", "missing_mask", "static_features"}.issubset(frame.columns)

    reused_path, recreated = ensure_demo_sample_parquet(payload, sample_path)
    assert reused_path == created_path
    assert recreated is False


def test_ensure_demo_sample_parquet_uses_public_fallback_when_available(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "models" / "sequence" / "run_demo" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    payload = _build_manifest_payload(tmp_path, checkpoint_path)

    public_sample_path = tmp_path / "public_artifacts" / "demo" / "sequence_demo_samples.parquet"
    created_path, created = ensure_demo_sample_parquet(payload, public_sample_path)
    assert created is True
    assert created_path == public_sample_path.resolve()

    missing_local_sample = tmp_path / "data" / "demo" / "missing_sample.parquet"
    resolved_path, generated = ensure_demo_sample_parquet(
        payload,
        missing_local_sample,
        public_fallback_path=public_sample_path,
    )

    assert generated is False
    assert resolved_path == public_sample_path.resolve()
