from __future__ import annotations

from pathlib import Path
from typing import Any

from early_sepsis.demo.presentation import (
    collect_metric_snapshot,
    compute_operational_metrics,
    find_duplicate_threshold_modes,
    safe_data_source_label,
    sanitize_public_text,
    serialize_public_ui_metadata,
)


def test_serialize_public_ui_metadata_redacts_sensitive_path_fields() -> None:
    payload: dict[str, Any] = {
        "manifest_path": "C:\\Users\\rohit\\repo\\artifacts\\models\\registry\\selected_model.json",
        "nested": {
            "checkpoint_path": "C:\\Users\\rohit\\repo\\artifacts\\models\\best_checkpoint.pt",
        },
        "samples": [
            {
                "parquet_path": "/Users/rohit/repo/artifacts/windows/validation.parquet",
            }
        ],
    }

    sanitized = serialize_public_ui_metadata(payload)

    assert sanitized["manifest_path"] == "<redacted>"
    assert sanitized["nested"]["checkpoint_path"] == "<redacted>"
    assert sanitized["samples"][0]["parquet_path"] == "<redacted>"


def test_serialize_public_ui_metadata_allows_internal_paths_when_enabled() -> None:
    payload = {
        "manifest_path": "C:\\Users\\rohit\\repo\\artifacts\\models\\registry\\selected_model.json"
    }

    sanitized = serialize_public_ui_metadata(payload, allow_internal_paths=True)

    assert sanitized["manifest_path"].startswith("C:\\Users\\rohit")


def test_sanitize_public_text_redacts_embedded_path_substrings() -> None:
    message = "Unable to open C:\\Users\\rohit\\repo\\data\\windows\\validation.parquet"

    sanitized = sanitize_public_text(message)

    assert "C:\\Users\\rohit" not in sanitized
    assert "<redacted>" in sanitized


def test_collect_metric_snapshot_prefers_calibration_metrics() -> None:
    manifest = {
        "metrics": {
            "validation": {
                "auroc": 0.72,
                "auprc": 0.61,
                "precision": 0.5,
                "recall": 0.6,
                "f1": 0.55,
            },
            "test": {
                "auroc": 0.7,
                "auprc": 0.58,
                "precision": 0.49,
                "recall": 0.57,
                "f1": 0.53,
            },
        }
    }
    calibration_summary = {
        "default_metrics": {
            "auroc": 0.75,
            "auprc": 0.64,
            "precision": 0.54,
            "recall": 0.62,
            "f1": 0.58,
            "accuracy": 0.71,
            "brier_score": 0.21,
            "expected_calibration_error": 0.08,
        }
    }

    snapshot, source_label = collect_metric_snapshot(
        manifest,
        calibration_summary=calibration_summary,
    )

    assert source_label == "Calibration analysis"
    assert snapshot["auroc"] == 0.75
    assert snapshot["brier_score"] == 0.21
    assert snapshot["expected_calibration_error"] == 0.08


def test_safe_data_source_label_uses_public_safe_labels() -> None:
    assert safe_data_source_label(public_mode=True, split="validation") == "Public demo sample"
    assert safe_data_source_label(public_mode=False, split="test") == "Demo Test split"


def test_compute_operational_metrics_changes_with_threshold_mode() -> None:
    probabilities = [0.12, 0.39, 0.51, 0.67, 0.93]
    labels = [0, 1, 0, 1, 1]

    relaxed_metrics = compute_operational_metrics(
        probabilities=probabilities,
        labels=labels,
        threshold=0.4,
    )
    strict_metrics = compute_operational_metrics(
        probabilities=probabilities,
        labels=labels,
        threshold=0.8,
    )

    assert relaxed_metrics["alert_count"] > strict_metrics["alert_count"]
    assert relaxed_metrics["predicted_positive_rate"] > strict_metrics["predicted_positive_rate"]
    assert relaxed_metrics["true_positive"] >= strict_metrics["true_positive"]


def test_collect_metric_snapshot_stays_threshold_invariant() -> None:
    manifest = {
        "metrics": {
            "validation": {
                "auroc": 0.73,
                "auprc": 0.11,
                "brier_score": 0.22,
                "expected_calibration_error": 0.09,
            }
        }
    }

    snapshot, _ = collect_metric_snapshot(manifest, calibration_summary=None)

    threshold_a = compute_operational_metrics(
        probabilities=[0.2, 0.7, 0.8],
        labels=[0, 1, 1],
        threshold=0.5,
    )
    threshold_b = compute_operational_metrics(
        probabilities=[0.2, 0.7, 0.8],
        labels=[0, 1, 1],
        threshold=0.75,
    )

    assert snapshot["auroc"] == 0.73
    assert snapshot["auprc"] == 0.11
    assert snapshot["brier_score"] == 0.22
    assert snapshot["expected_calibration_error"] == 0.09
    assert threshold_a["alert_count"] != threshold_b["alert_count"]


def test_find_duplicate_threshold_modes_reports_shared_thresholds() -> None:
    thresholds = {
        "default": 0.95,
        "balanced": 0.99,
        "high_recall": 0.9500000000000001,
    }

    duplicates = find_duplicate_threshold_modes(
        thresholds,
        modes=["default", "balanced", "high_recall"],
    )

    assert duplicates == [(0.95, ("default", "high_recall"))]


def test_sanitize_public_text_redacts_unix_style_paths() -> None:
    message = "Failed to load /Users/rohit/Documents/project/artifacts/windows/validation.parquet"

    sanitized = sanitize_public_text(message)

    assert "/Users/rohit" not in sanitized
    assert "<redacted>" in sanitized


def test_streamlit_app_uses_public_facing_wording_without_raw_json() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "early_sepsis" / "demo" / "app.py"
    app_source = app_path.read_text(encoding="utf-8")

    assert "st.json(" not in app_source
    assert "Developer debug mode" not in app_source
    assert "Developer Debug Metadata" not in app_source
    assert "Manifest override" not in app_source
    assert 'st.toggle(' not in app_source
    assert "@media print" in app_source
    assert "break-inside: avoid" in app_source
    assert "Patient ID" not in app_source
    assert "Sample ID" in app_source
    assert "DS-" in app_source
    assert "artifact-unavailable-card" in app_source
    assert "Threshold-Invariant Evaluation Summary" in app_source
    assert "Threshold-Dependent Operational Summary" in app_source
    assert "find_duplicate_threshold_modes" in app_source
    assert "operational-grid" in app_source
