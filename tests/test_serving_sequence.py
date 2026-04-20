from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from early_sepsis.modeling.model_manifest import build_feature_signature, save_model_manifest
from early_sepsis.modeling.sequence_models import SequenceModelConfig, build_sequence_model
from early_sepsis.serving import api as serving_api
from early_sepsis.serving.sequence_service import (
    SequenceServingError,
    clear_sequence_runtime_cache,
    predict_sequence_samples,
)


def _create_selected_manifest(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "models" / "sequence" / "run_a" / "best_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = SequenceModelConfig(
        model_type="gru",
        include_mask=True,
        include_static=True,
        recurrent_hidden_dim=16,
        recurrent_num_layers=1,
        recurrent_dropout=0.0,
        recurrent_bidirectional=False,
    )
    model = build_sequence_model(
        input_dim=3,
        static_dim=2,
        sequence_length=4,
        config=model_config,
    )

    torch.save(
        {
            "created_at": "2026-01-01T00:00:00+00:00",
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": {
                "windows_dir": str(tmp_path / "windows"),
                "output_dir": str(tmp_path / "models"),
            },
            "input_dim": 3,
            "sequence_length": 4,
            "static_dim": 2,
            "threshold": 0.5,
            "validation_metrics": {"auprc": 0.6},
        },
        checkpoint_path,
    )

    feature_columns = ["HR", "O2Sat", "SBP"]
    static_columns = ["Age", "Gender"]
    manifest = {
        "schema_version": "1.0",
        "selected_at": "2026-01-01T00:00:00+00:00",
        "selection_metric": "validation_auprc",
        "selected_run": {
            "run_name": "run_a",
            "run_dir": str(checkpoint_path.parent),
            "checkpoint_path": str(checkpoint_path),
            "run_config_path": str(checkpoint_path.parent / "run_config.json"),
            "model_type": "gru",
        },
        "dataset": {
            "dataset_tag": "physionet",
            "dataset_format": "physionet",
            "raw_data_path": str(tmp_path / "data" / "raw"),
            "windows_dir": str(tmp_path / "windows"),
            "processed_dir": str(tmp_path / "processed"),
            "feature_columns": feature_columns,
            "mask_columns": [f"{name}_mask" for name in feature_columns],
            "static_feature_columns": static_columns,
            "feature_signature": build_feature_signature(feature_columns),
        },
        "model": {
            "model_type": "gru",
            "input_dim": 3,
            "static_dim": 2,
            "window_length": 4,
            "include_mask": True,
            "include_static": True,
        },
        "thresholds": {
            "default": 0.5,
            "balanced": 0.45,
            "high_recall": 0.3,
        },
        "metrics": {
            "validation": {"auprc": 0.6, "auroc": 0.7},
            "test": {"auprc": 0.55, "auroc": 0.68},
        },
    }

    manifest_path = tmp_path / "registry" / "selected_model.json"
    save_model_manifest(manifest_path, manifest)
    return manifest_path


def _build_valid_sample() -> dict[str, object]:
    return {
        "patient_id": "patient_1",
        "end_hour": 8,
        "features": np.ones((4, 3), dtype=np.float32).tolist(),
        "missing_mask": np.zeros((4, 3), dtype=np.float32).tolist(),
        "static_features": np.array([61.0, 1.0], dtype=np.float32).tolist(),
    }


def test_predict_sequence_samples_dataset_guard(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    manifest_path = _create_selected_manifest(tmp_path)
    sample = _build_valid_sample()

    clear_sequence_runtime_cache()
    predictions = predict_sequence_samples(
        manifest_path=manifest_path,
        dataset_tag="physionet",
        samples=[sample],
        operating_mode="balanced",
    )

    assert len(predictions) == 1
    assert predictions[0]["operating_mode"] == "balanced"
    assert predictions[0]["threshold_used"] == pytest.approx(0.45)

    explicit_threshold_predictions = predict_sequence_samples(
        manifest_path=manifest_path,
        dataset_tag="physionet",
        samples=[sample],
        operating_mode="balanced",
        threshold_override=0.2,
    )
    assert explicit_threshold_predictions[0]["threshold_used"] == pytest.approx(0.2)

    with pytest.raises(SequenceServingError, match="operating_mode must be one of"):
        predict_sequence_samples(
            manifest_path=manifest_path,
            dataset_tag="physionet",
            samples=[sample],
            operating_mode="unsupported_mode",
        )

    with pytest.raises(SequenceServingError, match="Dataset tag mismatch"):
        predict_sequence_samples(
            manifest_path=manifest_path,
            dataset_tag="kaggle_csv",
            samples=[sample],
            operating_mode="balanced",
        )

    wrong_dim_sample = _build_valid_sample()
    wrong_dim_sample["features"] = np.ones((4, 2), dtype=np.float32).tolist()
    with pytest.raises(SequenceServingError, match="feature dimension"):
        predict_sequence_samples(
            manifest_path=manifest_path,
            dataset_tag="physionet",
            samples=[wrong_dim_sample],
            operating_mode="default",
        )


def test_predict_sequence_api_compatibility(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    manifest_path = _create_selected_manifest(tmp_path)
    sample = _build_valid_sample()

    clear_sequence_runtime_cache()
    original_manifest_path = serving_api.settings.selected_sequence_manifest_path
    original_default_mode = serving_api.settings.serving_default_operating_mode

    try:
        serving_api.settings.selected_sequence_manifest_path = manifest_path
        serving_api.settings.serving_default_operating_mode = "balanced"

        client = TestClient(serving_api.create_app())

        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["model_available"] is True
        assert health_payload["selected_sequence_model"]["available"] is True
        assert health_payload["selected_sequence_model"]["checkpoint_exists"] is True
        assert health_payload["selected_sequence_model"]["manifest_path"] == str(manifest_path)

        info_response = client.get("/model-info")
        assert info_response.status_code == 200
        payload = info_response.json()
        assert payload["selected_sequence_model"]["available"] is True
        assert payload["selected_sequence_model"]["threshold_modes"]["available"] == [
            "default",
            "balanced",
            "high_recall",
        ]

        response = client.post(
            "/predict-sequence",
            json={
                "dataset_tag": "physionet",
                "operating_mode": "default",
                "samples": [sample],
            },
        )
        assert response.status_code == 200
        prediction_payload = response.json()
        assert len(prediction_payload["predictions"]) == 1
        assert prediction_payload["predictions"][0]["threshold_used"] == pytest.approx(0.45)

        explicit_threshold_response = client.post(
            "/predict-sequence",
            json={
                "dataset_tag": "physionet",
                "threshold": 0.2,
                "samples": [sample],
            },
        )
        assert explicit_threshold_response.status_code == 200
        explicit_threshold_payload = explicit_threshold_response.json()
        assert explicit_threshold_payload["predictions"][0]["threshold_used"] == pytest.approx(0.2)

        invalid_mode_response = client.post(
            "/predict-sequence",
            json={
                "dataset_tag": "physionet",
                "operating_mode": "unsupported_mode",
                "samples": [sample],
            },
        )
        assert invalid_mode_response.status_code == 400
        assert "operating_mode must be one of" in invalid_mode_response.text

        mismatch_response = client.post(
            "/predict-sequence",
            json={
                "dataset_tag": "kaggle_csv",
                "operating_mode": "default",
                "samples": [sample],
            },
        )
        assert mismatch_response.status_code == 400
    finally:
        serving_api.settings.selected_sequence_manifest_path = original_manifest_path
        serving_api.settings.serving_default_operating_mode = original_default_mode
