from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from early_sepsis.runtime_paths import resolve_runtime_path

METRIC_LABELS: dict[str, str] = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "accuracy": "Accuracy",
    "brier_score": "Brier Score",
    "expected_calibration_error": "Expected Calibration Error",
}

METRIC_DIRECTION_LABELS: dict[str, str] = {
    "auroc": "Higher is better",
    "auprc": "Higher is better",
    "precision": "Higher is better",
    "recall": "Higher is better",
    "f1": "Higher is better",
    "accuracy": "Higher is better",
    "brier_score": "Lower is better",
    "expected_calibration_error": "Lower is better",
}

METRIC_EXPLANATIONS: dict[str, str] = {
    "auroc": "Measures ranking quality across all thresholds.",
    "auprc": (
        "Measures positive-class retrieval quality and is typically more informative under low "
        "prevalence."
    ),
    "precision": "Fraction of alerts that are true positives.",
    "recall": "Fraction of true positives captured by the model.",
    "f1": "Harmonic mean of precision and recall.",
    "accuracy": "Overall fraction of correct predictions.",
    "brier_score": "Mean squared error between predicted probabilities and true outcomes.",
    "expected_calibration_error": (
        "Measures calibration gap between predicted confidence and observed frequency."
    ),
}

METRIC_THRESHOLD_CONTEXT: dict[str, str] = {
    "auroc": "Threshold-invariant.",
    "auprc": "Threshold-invariant.",
    "precision": "Threshold-dependent.",
    "recall": "Threshold-dependent.",
    "f1": "Threshold-dependent.",
    "accuracy": "Threshold-dependent.",
    "brier_score": "Threshold-invariant calibration-quality metric.",
    "expected_calibration_error": "Threshold-invariant calibration-quality metric.",
}

PREVALENCE_ANNOTATION = (
    "Threshold-invariant dataset characteristic. Low prevalence increases class imbalance, "
    "so AUPRC is often more decision-relevant than AUROC for deployment review."
)

CALIBRATION_UNAVAILABLE_GUIDANCE = (
    "Calibration visuals are unavailable in this deployment package. Expected Calibration Error "
    "(ECE) tracks how closely predicted risk matches observed frequency, while Brier "
    "score captures overall probability error. Calibration quality matters in clinical "
    "risk scoring because poorly "
    "calibrated probabilities can mislead downstream triage decisions."
)

OPERATIONAL_UNAVAILABLE_GUIDANCE = (
    "Threshold-dependent operational summary requires packaged evaluation windows with labels and "
    "model-compatible sequence features. This deployment does not include those windows, so the "
    "operational panel is disabled here."
)

SAVED_WALKTHROUGH_UNAVAILABLE_GUIDANCE = (
    "Live inference windows are unavailable for this deployment and no saved walkthrough payload "
    "was bundled."
)

PLOT_TITLES: dict[str, str] = {
    "roc_curve": "ROC Curve",
    "pr_curve": "Precision-Recall Curve",
    "confusion_matrix": "Confusion Matrix",
    "reliability_curve": "Reliability / Calibration Curve",
    "score_distribution": "Score Distribution",
}

PLOT_FILE_NAMES: dict[str, str] = {
    "roc_curve": "roc_curve.png",
    "pr_curve": "pr_curve.png",
    "confusion_matrix": "confusion_matrix.png",
    "reliability_curve": "reliability_curve.png",
    "score_distribution": "score_distribution.png",
}

PLOT_DISPLAY_ORDER: tuple[str, ...] = (
    "roc_curve",
    "pr_curve",
    "confusion_matrix",
    "reliability_curve",
    "score_distribution",
)

THRESHOLD_MODE_LABELS: dict[str, str] = {
    "default": "Default (Baseline)",
    "balanced": "Balanced",
    "high_recall": "High Recall Target",
}

THRESHOLD_MODE_DESCRIPTIONS: dict[str, str] = {
    "default": "Matches the model's baseline calibration threshold.",
    "balanced": "Optimizes precision-recall balance for fewer false alarms and misses.",
    "high_recall": "Prioritizes sensitivity for earlier detection at the cost of more alerts.",
}

_SENSITIVE_KEY_TOKENS: tuple[str, ...] = (
    "path",
    "dir",
    "manifest",
    "checkpoint",
    "parquet",
    "username",
    "home",
)

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/].+")
_WINDOWS_EMBEDDED_RE = re.compile(r"[A-Za-z]:\\\\[^\s,;]+")
_POSIX_EMBEDDED_RE = re.compile(r"/(?:Users|home)/[^\s,;]+")


def format_threshold_mode(mode: str) -> str:
    return THRESHOLD_MODE_LABELS.get(mode, mode.replace("_", " ").title())


def describe_threshold_mode(mode: str) -> str:
    return THRESHOLD_MODE_DESCRIPTIONS.get(mode, "")


def build_metric_annotation(
    metric_key: str,
    *,
    metric_value: float | None = None,
    prevalence_value: float | None = None,
) -> str:
    direction = METRIC_DIRECTION_LABELS.get(metric_key, "")
    explanation = METRIC_EXPLANATIONS.get(metric_key, "")
    threshold_context = METRIC_THRESHOLD_CONTEXT.get(metric_key, "")

    parts = [part for part in (direction, explanation, threshold_context) if part]

    if metric_key == "auprc":
        if isinstance(prevalence_value, (float, int)) and 0.0 < float(prevalence_value) <= 1.0:
            baseline = float(prevalence_value)
            if isinstance(metric_value, (float, int)) and float(metric_value) >= 0.0:
                lift = float(metric_value) / baseline
                parts.append(
                    "Random-baseline AUPRC is approximately prevalence "
                    f"({baseline:.3f}); current AUPRC is {lift:.1f}x baseline."
                )
            else:
                parts.append(
                    "Random-baseline AUPRC is approximately the prevalence "
                    f"({baseline:.3f})."
                )
        else:
            parts.append("Under very low prevalence, random-baseline AUPRC is approximately prevalence.")

    return " ".join(parts)


def build_threshold_collapse_explanation(
    duplicate_thresholds: Sequence[tuple[float, Sequence[str]]],
) -> str:
    if not duplicate_thresholds:
        return ""

    collapse_summary = "; ".join(
        (
            f"{', '.join(format_threshold_mode(mode) for mode in modes)} "
            f"all map to {threshold_value:.3f}"
        )
        for threshold_value, modes in duplicate_thresholds
    )
    return (
        "For this selected artifact set, "
        f"{collapse_summary}. "
        "This is expected artifact/model-specific behavior, not a UI bug."
    )


def build_operational_subset_note(
    *,
    source_label: str,
    sample_count: int,
    positive_count: int,
) -> str:
    note = (
        "This panel uses a compact deployment subset for responsiveness and portability; "
        "it is not the full offline evaluation cohort."
    )
    if positive_count <= 0:
        return (
            f"{note} At low prevalence and small row counts (n={sample_count}), sampled windows "
            "can contain zero actual positives, which makes precision and sensitivity display as 0."
        )

    if positive_count < 3:
        return (
            f"{note} The current subset includes only {positive_count} positive windows out of "
            f"{sample_count}, so precision and sensitivity can vary substantially."
        )

    if "subset" in source_label.lower():
        return note

    return ""


def find_duplicate_threshold_modes(
    thresholds: dict[str, Any],
    *,
    modes: Sequence[str],
    decimals: int = 6,
) -> list[tuple[float, tuple[str, ...]]]:
    grouped: dict[float, list[str]] = {}
    for mode in modes:
        value = thresholds.get(mode)
        if not isinstance(value, (float, int)):
            continue
        normalized_threshold = round(float(value), decimals)
        grouped.setdefault(normalized_threshold, []).append(mode)

    duplicates: list[tuple[float, tuple[str, ...]]] = []
    for threshold_value, threshold_modes in grouped.items():
        if len(threshold_modes) > 1:
            duplicates.append((threshold_value, tuple(threshold_modes)))

    duplicates.sort(key=lambda item: item[0])
    return duplicates


def compute_operational_metrics(
    *,
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, float | int]:
    probability_array = np.asarray(list(probabilities), dtype=np.float64)
    label_array = np.asarray(list(labels), dtype=np.int64)

    if probability_array.ndim != 1 or label_array.ndim != 1:
        msg = "probabilities and labels must be 1D sequences"
        raise ValueError(msg)
    if probability_array.shape[0] != label_array.shape[0]:
        msg = "probabilities and labels must contain the same number of elements"
        raise ValueError(msg)
    if not 0.0 <= float(threshold) <= 1.0:
        msg = "threshold must be within [0, 1]"
        raise ValueError(msg)

    prediction_array = (probability_array >= float(threshold)).astype(np.int64)
    positive_mask = label_array == 1
    negative_mask = label_array == 0

    true_positive = int(np.logical_and(positive_mask, prediction_array == 1).sum())
    false_positive = int(np.logical_and(negative_mask, prediction_array == 1).sum())
    false_negative = int(np.logical_and(positive_mask, prediction_array == 0).sum())
    true_negative = int(np.logical_and(negative_mask, prediction_array == 0).sum())

    sample_count = int(label_array.shape[0])
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    alert_count = int(prediction_array.sum())

    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    sensitivity = recall
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    accuracy = _safe_ratio(true_positive + true_negative, sample_count)
    positive_rate = _safe_ratio(positive_count, sample_count)
    predicted_positive_rate = _safe_ratio(alert_count, sample_count)

    return {
        "threshold": float(threshold),
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "alert_count": alert_count,
        "positive_rate": positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "balanced_accuracy": balanced_accuracy,
    }


def safe_data_source_label(*, public_mode: bool, split: str) -> str:
    if public_mode:
        return "Public demo sample"
    return f"Demo {split.capitalize()} split"


def _looks_like_path_text(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False

    if _WINDOWS_ABSOLUTE_RE.match(stripped) or stripped.startswith("\\\\"):
        return True

    lowered = stripped.lower()
    if "\\users\\" in lowered or "/users/" in lowered or "/home/" in lowered:
        return True

    if "\\" in stripped or "/" in stripped:
        if lowered.endswith((".json", ".pt", ".parquet", ".csv", ".png", ".pkl", ".md")):
            return True
        parts = [part for part in re.split(r"[\\/]+", stripped) if part]
        if len(parts) >= 3 and "." in parts[-1]:
            return True

    return False


def sanitize_public_text(value: str, *, allow_internal_paths: bool = False) -> str:
    if allow_internal_paths:
        return value

    redacted = _WINDOWS_EMBEDDED_RE.sub("<redacted>", value)
    redacted = _POSIX_EMBEDDED_RE.sub("<redacted>", redacted)

    if redacted != value:
        return redacted

    if _looks_like_path_text(value):
        return "<redacted>"

    return value


def serialize_public_ui_metadata(
    payload: Any,
    *,
    allow_internal_paths: bool = False,
    key_name: str | None = None,
) -> Any:
    if isinstance(payload, dict):
        return {
            key: serialize_public_ui_metadata(
                value,
                allow_internal_paths=allow_internal_paths,
                key_name=key,
            )
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        return [
            serialize_public_ui_metadata(
                item,
                allow_internal_paths=allow_internal_paths,
                key_name=key_name,
            )
            for item in payload
        ]

    if isinstance(payload, str):
        normalized_key = (key_name or "").lower()
        if any(token in normalized_key for token in _SENSITIVE_KEY_TOKENS):
            return payload if allow_internal_paths else "<redacted>"
        return sanitize_public_text(payload, allow_internal_paths=allow_internal_paths)

    return payload


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def resolve_calibration_summary(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    public_artifacts_root: Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    candidates: list[str | Path] = []

    threshold_metadata = manifest.get("threshold_metadata")
    if isinstance(threshold_metadata, dict):
        summary_path_value = threshold_metadata.get("calibration_summary_path")
        if isinstance(summary_path_value, str) and summary_path_value.strip():
            candidates.append(summary_path_value)

    candidates.append(Path("artifacts/analysis/calibration/calibration_summary.json"))
    if public_artifacts_root is not None:
        candidates.append(
            public_artifacts_root / "analysis" / "calibration" / "calibration_summary.json"
        )

    for candidate in candidates:
        resolved_path = resolve_runtime_path(candidate, anchor=manifest_path.parent)
        if not resolved_path.exists():
            continue

        payload = _load_json_mapping(resolved_path)
        if payload is not None:
            return payload, resolved_path

    return None, None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def collect_metric_snapshot(
    manifest: dict[str, Any],
    *,
    calibration_summary: dict[str, Any] | None,
) -> tuple[dict[str, float | None], str]:
    manifest_metrics = manifest.get("metrics", {})
    if isinstance(manifest_metrics, dict):
        validation_metrics = manifest_metrics.get("validation", {})
        test_metrics = manifest_metrics.get("test", {})
    else:
        validation_metrics = {}
        test_metrics = {}

    calibration_metrics: dict[str, Any] = {}
    if calibration_summary is not None:
        default_metrics = calibration_summary.get("default_metrics")
        if isinstance(default_metrics, dict):
            calibration_metrics = default_metrics

    sources: list[tuple[str, dict[str, Any]]] = []
    if calibration_metrics:
        sources.append(("Calibration analysis", calibration_metrics))
    if test_metrics:
        sources.append(("Selected model test", test_metrics))
    if validation_metrics:
        sources.append(("Selected model validation", validation_metrics))

    source_label = "No evaluation metrics found"
    primary_metrics: dict[str, Any] = {}
    if sources:
        source_label, primary_metrics = sources[0]

    metric_snapshot: dict[str, float | None] = {}
    for metric_key in METRIC_LABELS:
        value = _as_float(primary_metrics.get(metric_key))
        if value is None:
            for _, fallback_metrics in sources[1:]:
                fallback_value = _as_float(fallback_metrics.get(metric_key))
                if fallback_value is not None:
                    value = fallback_value
                    break
        metric_snapshot[metric_key] = value

    return metric_snapshot, source_label


def collect_plot_artifacts(
    *,
    calibration_summary: dict[str, Any] | None,
    manifest_path: Path,
    public_artifacts_root: Path | None = None,
) -> dict[str, Path]:
    available_plots: dict[str, Path] = {}
    plot_mapping = calibration_summary.get("plot_paths", {}) if calibration_summary else {}

    for plot_key in PLOT_DISPLAY_ORDER:
        resolved_path: Path | None = None
        if isinstance(plot_mapping, dict):
            explicit_path = plot_mapping.get(plot_key)
            if isinstance(explicit_path, str) and explicit_path.strip():
                resolved_path = resolve_runtime_path(explicit_path, anchor=manifest_path.parent)

        if resolved_path is None:
            local_fallback = Path("artifacts/analysis/calibration") / PLOT_FILE_NAMES[plot_key]
            resolved_path = resolve_runtime_path(local_fallback, anchor=manifest_path.parent)
            if not resolved_path.exists() and public_artifacts_root is not None:
                public_fallback = (
                    public_artifacts_root
                    / "analysis"
                    / "calibration"
                    / PLOT_FILE_NAMES[plot_key]
                )
                resolved_path = resolve_runtime_path(public_fallback, anchor=manifest_path.parent)

        if resolved_path.exists():
            available_plots[plot_key] = resolved_path

    return available_plots


def load_reliability_curve(
    *,
    calibration_summary_path: Path | None,
    manifest_path: Path,
    public_artifacts_root: Path | None = None,
) -> pd.DataFrame | None:
    candidate_paths: list[Path] = []

    if calibration_summary_path is not None:
        candidate_paths.append(calibration_summary_path.parent / "reliability_curve.csv")

    candidate_paths.append(
        resolve_runtime_path(
            Path("artifacts/analysis/calibration/reliability_curve.csv"),
            anchor=manifest_path.parent,
        )
    )
    if public_artifacts_root is not None:
        candidate_paths.append(
            resolve_runtime_path(
                public_artifacts_root / "analysis" / "calibration" / "reliability_curve.csv",
                anchor=manifest_path.parent,
            )
        )

    for candidate in candidate_paths:
        if not candidate.exists():
            continue

        try:
            frame = pd.read_csv(candidate)
        except Exception:
            continue

        required_columns = {"bin", "bin_accuracy", "bin_confidence", "sample_count"}
        if required_columns.issubset(frame.columns):
            sanitized = sanitize_reliability_curve(
                frame[["bin", "bin_accuracy", "bin_confidence", "sample_count"]]
            )
            return sanitized

        alternate_columns = {
            "bin_index",
            "observed_positive_rate",
            "mean_predicted_probability",
            "sample_count",
        }
        if alternate_columns.issubset(frame.columns):
            normalized = frame.rename(
                columns={
                    "bin_index": "bin",
                    "observed_positive_rate": "bin_accuracy",
                    "mean_predicted_probability": "bin_confidence",
                }
            )
            sanitized = sanitize_reliability_curve(
                normalized[["bin", "bin_accuracy", "bin_confidence", "sample_count"]]
            )
            return sanitized

    return None


def sanitize_reliability_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes reliability bins for safe public plotting."""

    base_columns = ["bin", "bin_accuracy", "bin_confidence", "sample_count"]
    if frame.empty:
        return pd.DataFrame(columns=base_columns)

    missing_columns = [column for column in base_columns if column not in frame.columns]
    if missing_columns:
        return pd.DataFrame(columns=base_columns)

    sanitized = frame[base_columns].copy()
    for column in base_columns:
        sanitized[column] = pd.to_numeric(sanitized[column], errors="coerce")

    sanitized = sanitized.replace([np.inf, -np.inf], np.nan)
    sanitized = sanitized.dropna(subset=base_columns)
    if sanitized.empty:
        return pd.DataFrame(columns=base_columns)

    sanitized = sanitized[sanitized["sample_count"] > 0]
    if sanitized.empty:
        return pd.DataFrame(columns=base_columns)

    sanitized["sample_count"] = np.floor(sanitized["sample_count"]).astype(np.int64)

    # Probability-like statistics are clipped to valid bounds for robust plotting.
    sanitized["bin_accuracy"] = sanitized["bin_accuracy"].clip(0.0, 1.0)
    sanitized["bin_confidence"] = sanitized["bin_confidence"].clip(0.0, 1.0)

    sanitized = sanitized.sort_values(by=["bin_confidence", "bin"], ascending=True)
    return sanitized.reset_index(drop=True)


def load_experiment_comparison(
    *,
    limit: int = 5,
    public_artifacts_root: Path | None = None,
) -> pd.DataFrame | None:
    candidate_paths = [
        resolve_runtime_path(Path("artifacts/analysis/experiments/sequence_experiment_comparison.csv"))
    ]
    if public_artifacts_root is not None:
        candidate_paths.append(
            resolve_runtime_path(
                public_artifacts_root
                / "analysis"
                / "experiments"
                / "sequence_experiment_comparison.csv"
            )
        )

    frame: pd.DataFrame | None = None
    for comparison_path in candidate_paths:
        if not comparison_path.exists():
            continue
        try:
            frame = pd.read_csv(comparison_path)
            break
        except Exception:
            continue

    if frame is None:
        return None

    if frame.empty:
        return None

    desired_columns = {
        "run_name": "Run",
        "model_type": "Model Type",
        "model_family": "Model Family",
        "dataset_tag": "Dataset",
        "validation_auprc": "Validation AUPRC",
        "validation_auroc": "Validation AUROC",
        "test_auprc": "Test AUPRC",
        "runtime_seconds": "Runtime (s)",
    }

    available_columns = [column for column in desired_columns if column in frame.columns]
    if not available_columns:
        return None

    ordered = frame.copy()
    if "validation_auprc" in ordered.columns:
        ordered = ordered.sort_values(by="validation_auprc", ascending=False)

    selected = ordered[available_columns].head(limit).rename(columns=desired_columns)
    return selected.reset_index(drop=True)


def _standardize_feature_importance_frame(
    frame: pd.DataFrame,
    *,
    limit: int,
) -> pd.DataFrame | None:
    feature_candidates = ("feature", "Feature", "name", "variable")
    importance_candidates = ("importance", "Importance", "weight", "score")

    feature_column = next(
        (column for column in feature_candidates if column in frame.columns),
        None,
    )
    importance_column = next(
        (column for column in importance_candidates if column in frame.columns),
        None,
    )
    if feature_column is None or importance_column is None:
        return None

    normalized = frame[[feature_column, importance_column]].copy()
    normalized.columns = ["Feature", "Importance"]
    normalized["Feature"] = normalized["Feature"].astype(str)
    normalized["Importance"] = pd.to_numeric(normalized["Importance"], errors="coerce")
    normalized = normalized.dropna(subset=["Importance"])
    if normalized.empty:
        return None

    normalized = normalized.sort_values(by="Importance", ascending=False).head(limit)
    return normalized.reset_index(drop=True)


def _load_feature_importance_json(path: Path, *, limit: int) -> pd.DataFrame | None:
    payload = _load_json_mapping(path)
    if payload is None:
        return None

    if all(
        isinstance(key, str) and isinstance(value, (float, int))
        for key, value in payload.items()
    ):
        frame = pd.DataFrame(
            {
                "Feature": list(payload.keys()),
                "Importance": [float(value) for value in payload.values()],
            }
        )
        return (
            frame.sort_values(by="Importance", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )

    rows_payload = payload.get("rows")
    if isinstance(rows_payload, list):
        row_frame = pd.DataFrame(rows_payload)
        return _standardize_feature_importance_frame(row_frame, limit=limit)

    return None


def load_feature_importance_artifact(
    *,
    manifest_path: Path,
    public_artifacts_root: Path | None = None,
    limit: int = 10,
) -> pd.DataFrame | None:
    candidate_paths = [
        resolve_runtime_path(
            Path("artifacts/analysis/explainability/feature_importance.csv"),
            anchor=manifest_path.parent,
        ),
        resolve_runtime_path(
            Path("artifacts/analysis/explainability/feature_importance.json"),
            anchor=manifest_path.parent,
        ),
        resolve_runtime_path(
            Path("artifacts/analysis/experiments/feature_importance.csv"),
            anchor=manifest_path.parent,
        ),
        resolve_runtime_path(
            Path("artifacts/analysis/experiments/feature_importance.json"),
            anchor=manifest_path.parent,
        ),
    ]

    if public_artifacts_root is not None:
        candidate_paths.extend(
            [
                resolve_runtime_path(
                    public_artifacts_root
                    / "analysis"
                    / "explainability"
                    / "feature_importance.csv",
                    anchor=manifest_path.parent,
                ),
                resolve_runtime_path(
                    public_artifacts_root
                    / "analysis"
                    / "explainability"
                    / "feature_importance.json",
                    anchor=manifest_path.parent,
                ),
                resolve_runtime_path(
                    public_artifacts_root / "analysis" / "experiments" / "feature_importance.csv",
                    anchor=manifest_path.parent,
                ),
                resolve_runtime_path(
                    public_artifacts_root / "analysis" / "experiments" / "feature_importance.json",
                    anchor=manifest_path.parent,
                ),
            ]
        )

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue

        if candidate_path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(candidate_path)
            except Exception:
                continue
            standardized = _standardize_feature_importance_frame(frame, limit=limit)
            if standardized is not None:
                return standardized
            continue

        if candidate_path.suffix.lower() == ".json":
            standardized = _load_feature_importance_json(candidate_path, limit=limit)
            if standardized is not None:
                return standardized

    return None


def detect_latest_pytest_status(*, project_root: Path) -> tuple[str, str]:
    cache_path = project_root / ".pytest_cache" / "v" / "cache" / "lastfailed"
    if not cache_path.exists():
        return "Not available", "No cached pytest status artifact was found."

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Not available", "Cached pytest status could not be parsed."

    if not isinstance(payload, dict):
        return "Not available", "Cached pytest status has an unsupported format."

    if not payload:
        return "Passing", "Latest cached pytest run reported no failures."

    return "Attention", f"Latest cached pytest run reported {len(payload)} failing tests."
