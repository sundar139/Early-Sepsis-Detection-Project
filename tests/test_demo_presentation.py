from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from early_sepsis.demo.app import _build_reliability_chart
from early_sepsis.demo.inference_debug import count_unique_demo_windows
from early_sepsis.demo.presentation import (
    build_metric_annotation,
    build_operational_subset_note,
    build_threshold_collapse_explanation,
    collect_metric_snapshot,
    collect_plot_artifacts,
    compute_operational_metrics,
    find_duplicate_threshold_modes,
    load_experiment_comparison,
    load_feature_importance_artifact,
    load_reliability_curve,
    resolve_calibration_summary,
    safe_data_source_label,
    sanitize_reliability_curve,
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


def test_build_metric_annotation_includes_direction_and_context() -> None:
    auroc_annotation = build_metric_annotation("auroc")
    auprc_annotation = build_metric_annotation("auprc")
    brier_annotation = build_metric_annotation("brier_score")
    ece_annotation = build_metric_annotation("expected_calibration_error")

    assert auroc_annotation.startswith("Higher is better")
    assert "threshold-invariant" in auroc_annotation.lower()
    assert auprc_annotation.startswith("Higher is better")
    assert "low prevalence" in auprc_annotation.lower()
    assert "threshold-invariant" in auprc_annotation.lower()
    assert brier_annotation.startswith("Lower is better")
    assert "probabilities" in brier_annotation.lower()
    assert "calibration-quality" in brier_annotation.lower()
    assert ece_annotation.startswith("Lower is better")
    assert "predicted risk" in ece_annotation.lower()
    assert "calibration-quality" in ece_annotation.lower()


def test_build_threshold_collapse_explanation_mentions_artifact_specific_behavior() -> None:
    note = build_threshold_collapse_explanation(
        [(0.95, ("default", "high_recall"))]
    )

    assert "Default (Baseline)" in note
    assert "High Recall Target" in note
    assert "artifact/model-specific behavior" in note
    assert "not a UI bug" in note


def test_build_operational_subset_note_explains_zero_positive_case() -> None:
    note = build_operational_subset_note(
        source_label="Deployment operational subset",
        sample_count=120,
        positive_count=0,
    )

    assert "compact deployment subset" in note.lower()
    assert "zero actual positives" in note.lower()
    assert "precision and sensitivity" in note.lower()


def test_sanitize_reliability_curve_coerces_and_filters_invalid_rows() -> None:
    frame = pd.DataFrame(
        {
            "bin": [0, 1, 2, "x", 4],
            "bin_accuracy": ["0.2", "inf", -0.4, 0.5, 1.4],
            "bin_confidence": [0.1, 0.2, 1.7, "bad", 0.6],
            "sample_count": [10, 8, 0, 5, 3],
        }
    )

    sanitized = sanitize_reliability_curve(frame)

    assert not sanitized.empty
    assert sanitized["bin_accuracy"].between(0.0, 1.0).all()
    assert sanitized["bin_confidence"].between(0.0, 1.0).all()
    assert (sanitized["sample_count"] > 0).all()
    assert sanitized["bin_confidence"].is_monotonic_increasing


def test_reliability_chart_axes_are_locked_to_unit_interval() -> None:
    frame = pd.DataFrame(
        {
            "bin": [0, 1, 2],
            "bin_accuracy": [0.05, 0.2, 0.6],
            "bin_confidence": [0.1, 0.3, 0.7],
            "sample_count": [12, 10, 8],
        }
    )

    chart_spec = _build_reliability_chart(frame).to_dict()
    line_encoding = chart_spec["layer"][0]["encoding"]

    assert line_encoding["x"]["scale"]["domain"] == [0.0, 1.0]
    assert line_encoding["y"]["scale"]["domain"] == [0.0, 1.0]


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


def test_resolve_calibration_summary_uses_public_artifacts_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    summary_path = public_root / "analysis" / "calibration" / "calibration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('{"default_metrics": {"auroc": 0.7}}', encoding="utf-8")

    summary_payload, resolved_path = resolve_calibration_summary(
        manifest={},
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        public_artifacts_root=public_root,
    )

    assert summary_payload is not None
    assert resolved_path == summary_path.resolve()


def test_collect_plot_artifacts_uses_public_artifacts_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    plot_path = public_root / "analysis" / "calibration" / "roc_curve.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"PNG")

    plot_paths = collect_plot_artifacts(
        calibration_summary=None,
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        public_artifacts_root=public_root,
    )

    assert "roc_curve" in plot_paths
    assert plot_paths["roc_curve"] == plot_path.resolve()


def test_load_reliability_curve_uses_public_artifacts_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    reliability_path = public_root / "analysis" / "calibration" / "reliability_curve.csv"
    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    reliability_path.write_text(
        "bin,bin_accuracy,bin_confidence,sample_count\n0,0.2,0.1,10\n1,0.5,0.4,8\n",
        encoding="utf-8",
    )

    frame = load_reliability_curve(
        calibration_summary_path=None,
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        public_artifacts_root=public_root,
    )

    assert frame is not None
    assert list(frame.columns) == ["bin", "bin_accuracy", "bin_confidence", "sample_count"]


def test_load_reliability_curve_normalizes_alternate_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    reliability_path = public_root / "analysis" / "calibration" / "reliability_curve.csv"
    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    reliability_path.write_text(
        (
            "bin_index,bin_lower,bin_upper,sample_count,"
            "mean_predicted_probability,observed_positive_rate\n"
            "0,0.0,0.1,12,0.08,0.04\n"
            "1,0.1,0.2,10,0.16,0.09\n"
        ),
        encoding="utf-8",
    )

    frame = load_reliability_curve(
        calibration_summary_path=None,
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        public_artifacts_root=public_root,
    )

    assert frame is not None
    assert list(frame.columns) == ["bin", "bin_accuracy", "bin_confidence", "sample_count"]
    assert frame.iloc[0]["bin"] == 0
    assert frame.iloc[0]["bin_accuracy"] == pytest.approx(0.04)
    assert frame.iloc[1]["bin_confidence"] == pytest.approx(0.16)


def test_load_reliability_curve_returns_empty_when_all_rows_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    reliability_path = public_root / "analysis" / "calibration" / "reliability_curve.csv"
    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    reliability_path.write_text(
        (
            "bin,bin_accuracy,bin_confidence,sample_count\n"
            "0,inf,0.4,0\n"
            "1,NaN,1.5,-2\n"
        ),
        encoding="utf-8",
    )

    frame = load_reliability_curve(
        calibration_summary_path=None,
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        public_artifacts_root=public_root,
    )

    assert frame is not None
    assert frame.empty


def test_load_experiment_comparison_uses_public_artifacts_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    comparison_path = (
        public_root / "analysis" / "experiments" / "sequence_experiment_comparison.csv"
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "run_name": ["run_a"],
            "model_type": ["patchtst"],
            "model_family": ["patchtst_classifier"],
            "dataset_tag": ["physionet"],
            "validation_auprc": [0.12],
            "validation_auroc": [0.72],
            "test_auprc": [0.09],
            "runtime_seconds": [12.3],
        }
    )
    frame.to_csv(comparison_path, index=False)

    comparison = load_experiment_comparison(limit=5, public_artifacts_root=public_root)

    assert comparison is not None
    assert len(comparison) == 1
    assert comparison.iloc[0]["Run"] == "run_a"


def test_load_feature_importance_artifact_uses_public_artifacts_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEPSIS_PROJECT_ROOT", str(tmp_path))

    public_root = tmp_path / "public_artifacts"
    importance_path = public_root / "analysis" / "explainability" / "feature_importance.json"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.write_text(
        '{"HR_trend": 0.42, "Lactate_slope": 0.33, "Respiration_variability": 0.11}',
        encoding="utf-8",
    )

    frame = load_feature_importance_artifact(
        manifest_path=tmp_path / "artifacts" / "models" / "registry" / "selected_model.json",
        public_artifacts_root=public_root,
        limit=2,
    )

    assert frame is not None
    assert list(frame.columns) == ["Feature", "Importance"]
    assert len(frame) == 2
    assert frame.iloc[0]["Feature"] == "HR_trend"


def test_streamlit_app_uses_public_facing_wording_without_raw_json() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "early_sepsis" / "demo" / "app.py"
    app_source = app_path.read_text(encoding="utf-8")
    presentation_path = (
        Path(__file__).resolve().parents[1] / "src" / "early_sepsis" / "demo" / "presentation.py"
    )
    presentation_source = presentation_path.read_text(encoding="utf-8")

    assert "st.json(" not in app_source
    assert "Developer debug mode" not in app_source
    assert "Developer Debug Metadata" not in app_source
    assert "Manifest override" not in app_source
    assert 'st.toggle(' not in app_source
    assert "@media print" in app_source
    assert "break-inside: avoid" in app_source
    assert "Patient ID" not in app_source
    assert "| Patient " not in app_source
    assert "Sample ID" in app_source
    assert "DS-" in app_source
    assert "artifact-unavailable-card" in app_source
    assert "Generate calibration analysis outputs to include this panel." in app_source
    assert "Unable to load sequence windows for inference" not in app_source
    assert "Saved Example Walkthrough mode" in app_source
    assert "Threshold-Invariant Evaluation Summary" in app_source
    assert "Threshold-Dependent Operational Summary" in app_source
    assert "find_duplicate_threshold_modes" in app_source
    assert "operational-grid" in app_source
    assert "filtered out during" in app_source
    assert "sanitization because the bin statistics were invalid" in app_source
    assert "Axes are fixed to [0, 1]." in app_source
    assert "not a UI bug" in presentation_source
    assert (
        "if feature_importance_frame is not None and not feature_importance_frame.empty"
        in app_source
    )
    assert "Feature-importance artifacts were not bundled for this deployment" not in app_source
    assert "C:\\\\" not in app_source
    assert "/Users/" not in app_source


def test_curated_demo_bundle_uses_synthetic_ids_and_unique_windows() -> None:
    demo_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "demo"
        / "sequence_demo_samples.parquet"
    )
    if not demo_path.exists():
        pytest.skip("Curated demo bundle is not present in this environment.")

    frame = pd.read_parquet(
        demo_path,
        columns=["patient_id", "end_hour", "features", "missing_mask", "static_features"],
    )

    assert not frame.empty
    assert frame["patient_id"].astype(str).str.startswith("DS-").all()

    samples = [
        {
            "patient_id": row.patient_id,
            "end_hour": int(row.end_hour),
            "features": row.features,
            "missing_mask": row.missing_mask,
            "static_features": row.static_features,
        }
        for row in frame.itertuples(index=False)
    ]
    assert count_unique_demo_windows(samples) > 1


def test_operational_subset_bundle_uses_synthetic_ids_when_available() -> None:
    subset_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "demo"
        / "operational_windows_subset.parquet"
    )
    if not subset_path.exists():
        pytest.skip("Operational subset bundle is not present in this environment.")

    frame = pd.read_parquet(subset_path, columns=["patient_id"])
    assert not frame.empty
    assert frame["patient_id"].astype(str).str.startswith("DS-").all()
