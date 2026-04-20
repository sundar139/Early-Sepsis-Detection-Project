from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from early_sepsis.data.pipeline import create_window_pipeline, run_preprocessing_pipeline
from early_sepsis.data.synthetic import generate_synthetic_icu_dataset
from early_sepsis.modeling.sequence_models import SequenceModelConfig
from early_sepsis.modeling.sequence_pipeline import (
    SequenceTrainingConfig,
    _mlflow_run,
    build_sequence_run_name,
    evaluate_checkpoint,
    predict_from_checkpoint,
    train_sequence_model,
)


def test_sequence_training_and_evaluation_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    raw_csv_path = tmp_path / "synthetic_raw.csv"
    generate_synthetic_icu_dataset(
        output_path=raw_csv_path,
        dataset_format="csv",
        patient_count=16,
        min_hours=8,
        max_hours=12,
        random_seed=21,
    )

    processed_dir = tmp_path / "processed"
    windows_dir = tmp_path / "windows"

    run_preprocessing_pipeline(
        raw_data_path=raw_csv_path,
        output_dir=processed_dir,
        dataset_format="csv",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=21,
    )
    create_window_pipeline(
        processed_dir=processed_dir,
        output_dir=windows_dir,
        window_length=6,
        prediction_horizon=4,
        padding_mode=False,
        include_masks=True,
        include_static=True,
    )

    training_config = SequenceTrainingConfig(
        windows_dir=windows_dir,
        output_dir=tmp_path / "models",
        model_name="patchtst_classifier",
        seed=21,
        device="cpu",
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=2,
        scheduler_patience=1,
        scheduler_factor=0.5,
        threshold=0.5,
        optimize_threshold=True,
        imbalance_strategy="both",
        mlflow_enabled=False,
    )
    model_config = SequenceModelConfig(
        model_type="gru",
        include_mask=True,
        include_static=True,
        recurrent_hidden_dim=24,
        recurrent_num_layers=1,
        recurrent_dropout=0.1,
        recurrent_bidirectional=True,
    )

    result = train_sequence_model(training_config=training_config, model_config=model_config)

    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path.exists()
    assert result.run_dir.name.startswith("gru_classifier_")
    assert "auprc" in result.best_validation_metrics
    assert "auroc" in result.test_metrics

    evaluation = evaluate_checkpoint(
        checkpoint_path=result.best_checkpoint_path,
        parquet_path=windows_dir / "test.parquet",
        batch_size=64,
        num_workers=0,
    )
    assert "metrics" in evaluation
    assert "auprc" in evaluation["metrics"]

    prediction_path = tmp_path / "predictions.parquet"
    predictions = predict_from_checkpoint(
        checkpoint_path=result.best_checkpoint_path,
        parquet_path=windows_dir / "test.parquet",
        output_path=prediction_path,
        batch_size=64,
        num_workers=0,
    )
    assert prediction_path.exists()
    assert len(predictions) > 0
    assert {"predicted_probability", "predicted_label", "true_label"}.issubset(predictions.columns)


def test_mlflow_context_falls_back_when_start_run_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenMlflowModule(types.SimpleNamespace):
        def set_tracking_uri(self, uri: str) -> None:
            self.uri = uri

        def set_experiment(self, name: str) -> None:
            self.experiment = name

        @contextmanager
        def start_run(self, run_name: str | None = None):
            del run_name
            raise RuntimeError("simulated mlflow backend failure")
            yield None

    monkeypatch.setitem(sys.modules, "mlflow", BrokenMlflowModule())

    training_config = SequenceTrainingConfig(
        mlflow_enabled=True,
        mlflow_tracking_uri="file:./artifacts/mlruns_test",
        windows_dir=Path("artifacts/windows"),
        output_dir=Path("artifacts/models/sequence"),
    )
    model_config = SequenceModelConfig(model_type="gru")

    with _mlflow_run(training_config=training_config, model_config=model_config) as client:
        assert client is None


@pytest.mark.parametrize(
    ("model_type", "expected_prefix"),
    [
        ("patchtst", "patchtst_classifier"),
        ("gru", "gru_classifier"),
        ("lstm", "lstm_classifier"),
    ],
)
def test_build_sequence_run_name_uses_canonical_family_prefix(
    model_type: str,
    expected_prefix: str,
) -> None:
    run_name = build_sequence_run_name(
        model_type,
        started_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert run_name == f"{expected_prefix}_20260102_030405"
