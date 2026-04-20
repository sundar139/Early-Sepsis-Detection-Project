from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from early_sepsis.modeling.predict import predict_records
from early_sepsis.modeling.train import train_and_save_model
from early_sepsis.serving.api import create_app

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_training_and_prediction_smoke(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    result = train_and_save_model(
        data_path=FIXTURES_DIR / "synthetic_tabular.csv",
        dataset_format="csv",
        model_output_path=model_path,
        target_column="SepsisLabel",
        test_size=0.25,
        random_state=42,
    )

    assert model_path.exists()
    assert result.row_count > 0
    assert "f1" in result.metrics

    records = json.loads((FIXTURES_DIR / "synthetic_records.json").read_text(encoding="utf-8"))
    predictions = predict_records(records=records, model_path=model_path)

    assert len(predictions) == len(records)
    assert all(0.0 <= float(item["sepsis_risk"]) <= 1.0 for item in predictions)


def test_api_health_smoke() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
