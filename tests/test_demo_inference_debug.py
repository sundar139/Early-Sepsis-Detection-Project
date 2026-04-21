from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

import early_sepsis.demo.app as demo_app
from early_sepsis.demo.app import _build_inference_result_frame, _resolve_operational_source
from early_sepsis.demo.inference_debug import build_inference_diagnostics
from early_sepsis.demo.startup import DemoInferenceSource


def _request_sample(seed: float) -> dict[str, Any]:
    features = np.full((8, 3), seed, dtype=np.float32)
    missing_mask = np.zeros((8, 3), dtype=np.float32)
    static_features = np.asarray([seed, seed + 1.0], dtype=np.float32)
    return {
        "patient_id": f"DS-RAW-{seed:.1f}",
        "end_hour": 8,
        "features": features.tolist(),
        "missing_mask": missing_mask.tolist(),
        "static_features": static_features.tolist(),
    }


def test_inference_diagnostics_reports_unique_windows_and_probabilities() -> None:
    samples = [_request_sample(0.1), _request_sample(0.1), _request_sample(0.9)]
    predictions = [
        {"predicted_probability": 0.101},
        {"predicted_probability": 0.101},
        {"predicted_probability": 0.889},
    ]

    report = build_inference_diagnostics(samples=samples, predictions=predictions)

    assert report["sample_count"] == 3
    assert report["unique_window_count"] == 2
    assert report["unique_probability_count"] == 2
    assert report["probability_shapes"] == [[], [], []]


def test_result_table_scores_match_prediction_mapping() -> None:
    preview_rows = [
        {"Sample": "S001", "Sample ID": "DS-001"},
        {"Sample": "S002", "Sample ID": "DS-002"},
    ]
    predictions = [
        {
            "end_hour": 8,
            "predicted_probability": 0.1234567,
            "predicted_label": 0,
            "threshold_used": 0.95,
            "operating_mode": "default",
        },
        {
            "end_hour": 9,
            "predicted_probability": 0.9876543,
            "predicted_label": 1,
            "threshold_used": 0.95,
            "operating_mode": "default",
        },
    ]

    result_frame = _build_inference_result_frame(preview_rows=preview_rows, predictions=predictions)

    displayed_scores = result_frame["Risk Score"].to_list()
    assert displayed_scores == [0.123457, 0.987654]

    report = build_inference_diagnostics(
        samples=[_request_sample(0.2), _request_sample(0.8)],
        predictions=predictions,
        displayed_scores=displayed_scores,
        displayed_round_decimals=6,
    )

    assert report["displayed_scores_match"] is True
    assert report["unique_probability_count"] == 2


def test_resolve_operational_source_prefers_compact_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    subset_path = public_root / "demo" / "operational_windows_subset.parquet"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "patient_id": ["DS-OP-0001"],
            "end_hour": [8],
            "label": [0],
            "features": [np.zeros((8, 3), dtype=np.float32).tolist()],
            "missing_mask": [np.zeros((8, 3), dtype=np.float32).tolist()],
            "static_features": [np.zeros((2,), dtype=np.float32).tolist()],
        }
    ).to_parquet(subset_path, index=False)

    source_label, source_path, unavailable_reason = _resolve_operational_source(
        public_artifacts_root=public_root,
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        dataset_section={"windows_dir": "missing_windows"},
        split="validation",
        inference_source=DemoInferenceSource(
            source_kind="walkthrough",
            source_label="Saved Example Walkthrough",
            parquet_path=None,
            walkthrough_payload_path=None,
        ),
    )

    assert source_label == "Deployment operational subset"
    assert source_path == subset_path.resolve()
    assert unavailable_reason is None


def test_operational_frame_loads_from_compact_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subset_path = tmp_path / "operational_windows_subset.parquet"
    frame = pd.DataFrame(
        {
            "patient_id": ["DS-OP-0001", "DS-OP-0002", "DS-OP-0003"],
            "end_hour": [8, 9, 10],
            "label": [0, 1, 1],
            "features": [
                np.zeros((8, 3), dtype=np.float32).tolist(),
                np.ones((8, 3), dtype=np.float32).tolist(),
                (np.ones((8, 3), dtype=np.float32) * 2.0).tolist(),
            ],
            "missing_mask": [
                np.zeros((8, 3), dtype=np.float32).tolist(),
                np.zeros((8, 3), dtype=np.float32).tolist(),
                np.zeros((8, 3), dtype=np.float32).tolist(),
            ],
            "static_features": [
                np.zeros((2,), dtype=np.float32).tolist(),
                np.ones((2,), dtype=np.float32).tolist(),
                (np.ones((2,), dtype=np.float32) * 2.0).tolist(),
            ],
        }
    )
    frame.to_parquet(subset_path, index=False)

    def _fake_predict_sequence_samples(
        *,
        samples: list[dict[str, Any]],
        **_: Any,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            outputs.append(
                {
                    "patient_id": sample.get("patient_id"),
                    "end_hour": int(sample.get("end_hour", 0)),
                    "predicted_probability": [0.12, 0.47, 0.93][index],
                    "predicted_label": [0, 0, 1][index],
                    "threshold_used": 0.95,
                    "operating_mode": "default",
                }
            )
        return outputs

    monkeypatch.setattr(demo_app, "predict_sequence_samples", _fake_predict_sequence_samples)
    demo_app._load_split_samples.clear()
    demo_app._load_operational_probability_frame.clear()

    operational = demo_app._load_operational_probability_frame(
        manifest_path=str(tmp_path / "selected_model.json"),
        manifest_mtime=0.0,
        dataset_tag="physionet",
        parquet_path=str(subset_path),
        max_rows=10,
    )

    assert not operational.empty
    assert operational["Risk Score"].to_list() == [0.12, 0.47, 0.93]
    assert operational["Observed Label"].to_list() == [0, 1, 1]
