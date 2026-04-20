from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from early_sepsis.modeling.experiment_analysis import (
    aggregate_sequence_experiments,
    build_model_manifest_from_row,
    export_experiment_comparison,
    select_best_run,
)
from early_sepsis.modeling.model_manifest import (
    ModelManifestValidationError,
    build_feature_signature,
    load_model_manifest,
    save_model_manifest,
    validate_model_manifest,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _create_run_artifacts(
    tmp_path: Path,
    *,
    model_type: str = "gru",
    run_dir_name: str | None = None,
) -> tuple[Path, Path]:
    processed_dir = tmp_path / "processed"
    windows_dir = tmp_path / "windows"
    model_root = tmp_path / "models"
    resolved_run_dir_name = run_dir_name or f"{model_type}_classifier_20260101_010203"
    run_dir = model_root / "sequence" / resolved_run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        processed_dir / "metadata.json",
        {
            "dataset_format": "physionet",
            "raw_data_path": str(tmp_path / "data" / "raw"),
        },
    )

    _write_json(
        windows_dir / "metadata.json",
        {
            "processed_dir": str(processed_dir),
            "feature_columns": ["HR", "O2Sat", "SBP"],
            "mask_columns": ["HR_mask", "O2Sat_mask", "SBP_mask"],
            "static_feature_columns": ["Age", "Gender"],
        },
    )

    _write_json(
        run_dir / "run_config.json",
        {
            "training_config": {
                "windows_dir": str(windows_dir),
                "model_name": f"{model_type}_classifier",
                "batch_size": 16,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "imbalance_strategy": "both",
                "epochs": 2,
                "mlflow_tracking_uri": "file:./artifacts/mlruns",
                "mlflow_experiment_name": "early-sepsis-sequence",
            },
            "model_config": {
                "model_type": model_type,
                "include_mask": True,
                "include_static": True,
            },
            "input_dim": 3,
            "static_dim": 2,
            "sequence_length": 4,
        },
    )

    _write_json(
        run_dir / "validation_metrics.json",
        {
            "threshold": 0.42,
            "metrics": {
                "auprc": 0.81,
                "auroc": 0.85,
                "precision": 0.66,
                "recall": 0.75,
                "f1": 0.70,
            },
        },
    )
    _write_json(
        run_dir / "test_metrics.json",
        {
            "threshold": 0.42,
            "metrics": {
                "auprc": 0.77,
                "auroc": 0.80,
                "precision": 0.61,
                "recall": 0.72,
                "f1": 0.66,
            },
        },
    )
    _write_json(run_dir / "training_history.json", {"history": [{"epoch": 1}, {"epoch": 2}]})

    (run_dir / "best_checkpoint.pt").touch()
    (run_dir / "last_checkpoint.pt").touch()

    return model_root, run_dir


def test_aggregate_export_and_manifest_selection(tmp_path: Path) -> None:
    model_root, _ = _create_run_artifacts(tmp_path)

    frame = aggregate_sequence_experiments(model_root=model_root, project_root=tmp_path)
    assert not frame.empty
    assert len(frame) == 1

    row = frame.iloc[0]
    assert row["dataset_tag"] == "physionet"
    assert row["model_type"] == "gru"
    assert row["model_family"] == "gru_classifier"
    assert row["input_dim"] == 3
    assert row["static_dim"] == 2

    output_dir = tmp_path / "analysis"
    csv_path, md_path = export_experiment_comparison(
        frame,
        csv_path=output_dir / "comparison.csv",
        markdown_path=output_dir / "comparison.md",
    )

    assert csv_path.exists()
    assert md_path.exists()

    report_text = md_path.read_text(encoding="utf-8")
    assert "Experiment Comparison" in report_text
    assert "Best Per Model" in report_text

    reloaded = pd.read_csv(csv_path)
    best_row = select_best_run(
        reloaded, selection_metric="validation_auprc", dataset_tag="physionet"
    )
    manifest = build_model_manifest_from_row(best_row, project_root=tmp_path)

    manifest_path = tmp_path / "registry" / "selected_model.json"
    save_model_manifest(manifest_path, manifest)

    loaded_manifest = load_model_manifest(manifest_path)
    assert loaded_manifest["dataset"]["dataset_tag"] == "physionet"
    assert loaded_manifest["selected_run"]["model_type"] == "gru"
    assert loaded_manifest["selected_run"]["model_family"] == "gru_classifier"
    assert loaded_manifest["model"]["model_family"] == "gru_classifier"
    assert loaded_manifest["dataset"]["feature_signature"] == build_feature_signature(
        loaded_manifest["dataset"]["feature_columns"]
    )


@pytest.mark.parametrize("model_type", ["patchtst", "gru", "lstm"])
def test_manifest_metadata_prefers_explicit_run_config_model_type(
    tmp_path: Path,
    model_type: str,
) -> None:
    misleading_run_name = "patchtst_classifier_20260101_010203"
    model_root, _ = _create_run_artifacts(
        tmp_path,
        model_type=model_type,
        run_dir_name=misleading_run_name,
    )

    frame = aggregate_sequence_experiments(model_root=model_root, project_root=tmp_path)
    assert len(frame) == 1

    row = frame.iloc[0]
    expected_family = f"{model_type}_classifier"
    assert row["model_type"] == model_type
    assert row["model_family"] == expected_family

    manifest = build_model_manifest_from_row(row, project_root=tmp_path)
    assert manifest["selected_run"]["model_type"] == model_type
    assert manifest["selected_run"]["model_family"] == expected_family
    assert manifest["model"]["model_type"] == model_type
    assert manifest["model"]["model_family"] == expected_family


def test_manifest_validation_rejects_signature_mismatch() -> None:
    manifest = {
        "schema_version": "1.0",
        "selected_at": "2026-01-01T00:00:00+00:00",
        "selection_metric": "validation_auprc",
        "selected_run": {
            "run_name": "run_a",
            "run_dir": "artifacts/models/sequence/run_a",
            "checkpoint_path": "artifacts/models/sequence/run_a/best_checkpoint.pt",
            "run_config_path": "artifacts/models/sequence/run_a/run_config.json",
            "model_type": "gru",
        },
        "dataset": {
            "dataset_tag": "physionet",
            "dataset_format": "physionet",
            "raw_data_path": "data/raw",
            "windows_dir": "artifacts/windows",
            "processed_dir": "artifacts/processed",
            "feature_columns": ["HR", "O2Sat", "SBP"],
            "mask_columns": ["HR_mask", "O2Sat_mask", "SBP_mask"],
            "static_feature_columns": ["Age", "Gender"],
            "feature_signature": "bad_signature",
        },
        "model": {
            "model_type": "gru",
            "input_dim": 3,
            "static_dim": 2,
            "window_length": 8,
            "include_mask": True,
            "include_static": True,
        },
        "thresholds": {
            "default": 0.5,
            "balanced": 0.5,
            "high_recall": 0.4,
        },
        "metrics": {
            "validation": {"auprc": 0.8},
            "test": {"auprc": 0.75},
        },
    }

    with pytest.raises(ModelManifestValidationError):
        validate_model_manifest(manifest)


def test_manifest_validation_rejects_model_type_mismatch() -> None:
    feature_columns = ["HR", "O2Sat", "SBP"]
    manifest = {
        "schema_version": "1.0",
        "selected_at": "2026-01-01T00:00:00+00:00",
        "selection_metric": "validation_auprc",
        "selected_run": {
            "run_name": "run_a",
            "run_dir": "artifacts/models/sequence/run_a",
            "checkpoint_path": "artifacts/models/sequence/run_a/best_checkpoint.pt",
            "run_config_path": "artifacts/models/sequence/run_a/run_config.json",
            "model_type": "gru",
            "model_family": "gru_classifier",
        },
        "dataset": {
            "dataset_tag": "physionet",
            "dataset_format": "physionet",
            "raw_data_path": "data/raw",
            "windows_dir": "artifacts/windows",
            "processed_dir": "artifacts/processed",
            "feature_columns": feature_columns,
            "mask_columns": ["HR_mask", "O2Sat_mask", "SBP_mask"],
            "static_feature_columns": ["Age", "Gender"],
            "feature_signature": build_feature_signature(feature_columns),
        },
        "model": {
            "model_type": "lstm",
            "model_family": "lstm_classifier",
            "input_dim": 3,
            "static_dim": 2,
            "window_length": 8,
            "include_mask": True,
            "include_static": True,
        },
        "thresholds": {
            "default": 0.5,
            "balanced": 0.5,
            "high_recall": 0.4,
        },
        "metrics": {
            "validation": {"auprc": 0.8},
            "test": {"auprc": 0.75},
        },
    }

    with pytest.raises(ModelManifestValidationError):
        validate_model_manifest(manifest)
