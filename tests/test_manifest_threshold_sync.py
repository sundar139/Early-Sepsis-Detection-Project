from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from early_sepsis.modeling.model_manifest import (
    ModelManifestValidationError,
    build_feature_signature,
    load_model_manifest,
    save_model_manifest,
    sync_manifest_thresholds_from_calibration,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _create_manifest(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint_path = tmp_path / "models" / "sequence" / "run_a" / "best_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    feature_columns = ["HR", "O2Sat", "SBP"]
    static_feature_columns = ["Age", "Gender"]

    manifest = {
        "schema_version": "1.0",
        "selected_at": "2026-01-01T00:00:00+00:00",
        "selection_metric": "validation_auprc",
        "selected_run": {
            "run_name": "run_a",
            "run_dir": str(checkpoint_path.parent),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "run_config_path": str(checkpoint_path.parent / "run_config.json"),
            "model_type": "gru",
            "model_family": "gru_classifier",
        },
        "dataset": {
            "dataset_tag": "physionet",
            "dataset_format": "physionet",
            "raw_data_path": str(tmp_path / "data" / "raw"),
            "windows_dir": str(tmp_path / "windows"),
            "processed_dir": str(tmp_path / "processed"),
            "feature_columns": feature_columns,
            "mask_columns": ["HR_mask", "O2Sat_mask", "SBP_mask"],
            "static_feature_columns": static_feature_columns,
            "feature_signature": build_feature_signature(feature_columns),
        },
        "model": {
            "model_type": "gru",
            "model_family": "gru_classifier",
            "input_dim": 3,
            "static_dim": 2,
            "window_length": 8,
            "include_mask": True,
            "include_static": True,
        },
        "thresholds": {
            "default": 0.5,
            "balanced": 0.5,
            "high_recall": 0.5,
        },
        "metrics": {
            "validation": {"auprc": 0.6, "auroc": 0.7},
            "test": {"auprc": 0.58, "auroc": 0.69},
        },
    }

    manifest_path = tmp_path / "registry" / "selected_model.json"
    save_model_manifest(manifest_path, manifest)
    return manifest_path, checkpoint_path.resolve()


def test_sync_manifest_thresholds_from_calibration_updates_thresholds_and_metadata(
    tmp_path: Path,
) -> None:
    manifest_path, checkpoint_path = _create_manifest(tmp_path)

    recommendations_path = tmp_path / "analysis" / "threshold_recommendations.json"
    summary_path = tmp_path / "analysis" / "calibration_summary.json"

    _write_json(
        recommendations_path,
        {
            "default": 0.5,
            "balanced": 0.8,
            "high_recall": 0.2,
            "high_recall_target": 0.9,
        },
    )
    _write_json(
        summary_path,
        {
            "checkpoint_path": str(checkpoint_path),
            "recommended_thresholds": {
                "default": 0.5,
                "balanced": 0.8,
                "high_recall": 0.2,
                "high_recall_target": 0.9,
            },
        },
    )

    updated_manifest = sync_manifest_thresholds_from_calibration(
        manifest_path=manifest_path,
        recommendations_path=recommendations_path,
        calibration_summary_path=summary_path,
    )

    assert updated_manifest["thresholds"]["default"] == pytest.approx(0.5)
    assert updated_manifest["thresholds"]["balanced"] == pytest.approx(0.8)
    assert updated_manifest["thresholds"]["high_recall"] == pytest.approx(0.2)

    threshold_metadata = updated_manifest["threshold_metadata"]
    assert threshold_metadata["source"] == "calibration_recommendations"
    assert threshold_metadata["high_recall_target"] == pytest.approx(0.9)
    assert "synchronized_at" in threshold_metadata

    persisted_manifest = load_model_manifest(manifest_path)
    assert persisted_manifest["thresholds"]["balanced"] == pytest.approx(0.8)
    assert persisted_manifest["threshold_metadata"]["source"] == "calibration_recommendations"


def test_sync_manifest_thresholds_rejects_mismatched_calibration_checkpoint(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _create_manifest(tmp_path)

    recommendations_path = tmp_path / "analysis" / "threshold_recommendations.json"
    summary_path = tmp_path / "analysis" / "calibration_summary.json"

    _write_json(
        recommendations_path,
        {
            "default": 0.5,
            "balanced": 0.8,
            "high_recall": 0.2,
        },
    )
    _write_json(
        summary_path,
        {
            "checkpoint_path": str(tmp_path / "models" / "other" / "best_checkpoint.pt"),
            "recommended_thresholds": {
                "default": 0.5,
                "balanced": 0.8,
                "high_recall": 0.2,
            },
        },
    )

    with pytest.raises(ModelManifestValidationError, match="checkpoint_path"):
        sync_manifest_thresholds_from_calibration(
            manifest_path=manifest_path,
            recommendations_path=recommendations_path,
            calibration_summary_path=summary_path,
        )
