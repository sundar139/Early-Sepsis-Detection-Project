from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from early_sepsis.modeling.model_manifest import build_feature_signature
from early_sepsis.modeling.sequence_metrics import compute_binary_metrics
from early_sepsis.modeling.sequence_pipeline import (
    evaluate_checkpoint,
    sequence_model_family_name,
)

SUPPORTED_MODEL_TYPES = frozenset({"gru", "lstm", "patchtst"})


@dataclass(slots=True)
class CalibrationAnalysisArtifacts:
    output_dir: Path
    threshold_sweep_path: Path
    reliability_curve_path: Path
    recommendations_path: Path
    summary_path: Path
    markdown_report_path: Path
    plot_paths: dict[str, Path]
    recommendations: dict[str, float]


def _normalize_model_type(value: Any) -> str:
    return str(value).strip().lower()


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _resolve_path(path_value: str | Path, *, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _parse_run_start_from_name(run_name: str) -> datetime | None:
    suffix = run_name.rsplit("_", maxsplit=2)
    if len(suffix) < 3:
        return None

    date_part = suffix[-2]
    time_part = suffix[-1]
    candidate = f"{date_part}_{time_part}"
    try:
        return datetime.strptime(candidate, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _infer_runtime_seconds(run_dir: Path) -> float | None:
    run_start = _parse_run_start_from_name(run_dir.name)
    file_paths = [path for path in run_dir.rglob("*") if path.is_file()]
    if not file_paths:
        return None

    if run_start is not None:
        end_timestamp = max(path.stat().st_mtime for path in file_paths)
        run_end = datetime.fromtimestamp(end_timestamp, tz=UTC)
        runtime = (run_end - run_start).total_seconds()
        return float(max(runtime, 0.0))

    start_timestamp = min(path.stat().st_mtime for path in file_paths)
    end_timestamp = max(path.stat().st_mtime for path in file_paths)
    return float(max(end_timestamp - start_timestamp, 0.0))


def _derive_dataset_tag(
    dataset_format: str,
    raw_data_path: str,
    feature_columns: list[str],
) -> str:
    lowered_raw_path = raw_data_path.lower()

    if dataset_format == "physionet":
        return "physionet"

    if dataset_format == "csv":
        if "kaggle" in lowered_raw_path:
            return "kaggle_csv"
        if "local_csv" in lowered_raw_path:
            return "kaggle_csv"
        if "Unnamed: 0" in feature_columns or "Hour" in feature_columns:
            return "kaggle_csv"
        return "csv"

    return "unknown"


def _dataset_context_from_windows_dir(windows_dir: Path, *, project_root: Path) -> dict[str, Any]:
    resolved_windows_dir = _resolve_path(windows_dir, base=project_root)

    windows_metadata_path = resolved_windows_dir / "metadata.json"
    if not windows_metadata_path.exists():
        return {
            "dataset_tag": "unknown",
            "dataset_format": "unknown",
            "raw_data_path": "",
            "windows_dir": str(resolved_windows_dir),
            "processed_dir": "",
            "feature_columns": [],
            "mask_columns": [],
            "static_feature_columns": [],
        }

    windows_metadata = _json_load(windows_metadata_path)
    feature_columns = [str(item) for item in windows_metadata.get("feature_columns", [])]
    mask_columns = [str(item) for item in windows_metadata.get("mask_columns", [])]
    static_feature_columns = [
        str(item) for item in windows_metadata.get("static_feature_columns", [])
    ]

    processed_dir_value = windows_metadata.get("processed_dir")
    if not isinstance(processed_dir_value, str):
        processed_dir_value = ""

    processed_dir = (
        _resolve_path(processed_dir_value, base=project_root) if processed_dir_value else Path("")
    )
    dataset_format = "unknown"
    raw_data_path = ""

    processed_metadata_path = processed_dir / "metadata.json" if processed_dir else Path("")
    if processed_metadata_path and processed_metadata_path.exists():
        processed_metadata = _json_load(processed_metadata_path)
        dataset_format = str(processed_metadata.get("dataset_format", "unknown"))
        raw_data_path = str(processed_metadata.get("raw_data_path", ""))

    dataset_tag = _derive_dataset_tag(
        dataset_format=dataset_format,
        raw_data_path=raw_data_path,
        feature_columns=feature_columns,
    )

    return {
        "dataset_tag": dataset_tag,
        "dataset_format": dataset_format,
        "raw_data_path": raw_data_path,
        "windows_dir": str(resolved_windows_dir),
        "processed_dir": str(processed_dir) if processed_dir else "",
        "feature_columns": feature_columns,
        "mask_columns": mask_columns,
        "static_feature_columns": static_feature_columns,
    }


def discover_run_directories(model_root: str | Path) -> list[Path]:
    """Finds sequence run directories that contain a resolved run_config.json."""

    root = Path(model_root)
    if not root.exists():
        return []

    run_dirs: list[Path] = []
    for run_config_path in root.rglob("run_config.json"):
        run_dir = run_config_path.parent
        if (run_dir / "best_checkpoint.pt").exists() and (
            run_dir / "validation_metrics.json"
        ).exists():
            run_dirs.append(run_dir)

    unique = sorted(set(run_dirs))
    return unique


def aggregate_sequence_experiments(
    *,
    model_root: str | Path = "artifacts/models",
    project_root: str | Path = ".",
) -> pd.DataFrame:
    """Aggregates comparable run metadata across GRU, LSTM, and PatchTST experiments."""

    resolved_project_root = Path(project_root).resolve()
    resolved_model_root = Path(model_root).resolve()

    records: list[dict[str, Any]] = []
    for run_dir in discover_run_directories(model_root):
        run_config_path = run_dir / "run_config.json"
        validation_metrics_path = run_dir / "validation_metrics.json"
        test_metrics_path = run_dir / "test_metrics.json"
        training_history_path = run_dir / "training_history.json"

        run_config = _json_load(run_config_path)
        model_config = run_config.get("model_config", {})
        training_config = run_config.get("training_config", {})

        if not isinstance(model_config, dict) or not isinstance(training_config, dict):
            continue

        model_type = _normalize_model_type(model_config.get("model_type", "unknown"))
        if model_type not in SUPPORTED_MODEL_TYPES:
            continue
        model_family = sequence_model_family_name(model_type)

        validation_payload = _json_load(validation_metrics_path)
        test_payload = (
            _json_load(test_metrics_path) if test_metrics_path.exists() else {"metrics": {}}
        )

        validation_metrics = validation_payload.get("metrics", {})
        test_metrics = test_payload.get("metrics", {})
        if not isinstance(validation_metrics, dict) or not isinstance(test_metrics, dict):
            continue

        history_payload: dict[str, Any] = {}
        if training_history_path.exists():
            history_payload = _json_load(training_history_path)

        windows_dir_value = training_config.get("windows_dir", "")
        windows_dir = Path(str(windows_dir_value)) if windows_dir_value else Path(".")
        dataset_context = _dataset_context_from_windows_dir(
            windows_dir, project_root=resolved_project_root
        )

        epochs_completed = 0
        history_rows = (
            history_payload.get("history", []) if isinstance(history_payload, dict) else []
        )
        if isinstance(history_rows, list):
            epochs_completed = len(history_rows)

        config_summary = {
            "model": model_config,
            "training": {
                "batch_size": training_config.get("batch_size"),
                "learning_rate": training_config.get("learning_rate"),
                "weight_decay": training_config.get("weight_decay"),
                "imbalance_strategy": training_config.get("imbalance_strategy"),
                "epochs": training_config.get("epochs"),
            },
        }

        resolved_run_dir = run_dir.resolve()
        if resolved_run_dir.is_relative_to(resolved_model_root):
            run_group = resolved_run_dir.relative_to(resolved_model_root).parts[0]
        else:
            run_group = run_dir.parent.name

        records.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(resolved_run_dir),
                "run_group": run_group,
                "model_type": model_type,
                "model_family": model_family,
                "dataset_tag": dataset_context["dataset_tag"],
                "dataset_format": dataset_context["dataset_format"],
                "raw_data_path": dataset_context["raw_data_path"],
                "windows_dir": dataset_context["windows_dir"],
                "validation_auprc": float(validation_metrics.get("auprc", np.nan)),
                "validation_auroc": float(validation_metrics.get("auroc", np.nan)),
                "validation_precision": float(validation_metrics.get("precision", np.nan)),
                "validation_recall": float(validation_metrics.get("recall", np.nan)),
                "validation_f1": float(validation_metrics.get("f1", np.nan)),
                "validation_threshold": float(validation_payload.get("threshold", np.nan)),
                "test_auprc": float(test_metrics.get("auprc", np.nan)),
                "test_auroc": float(test_metrics.get("auroc", np.nan)),
                "test_precision": float(test_metrics.get("precision", np.nan)),
                "test_recall": float(test_metrics.get("recall", np.nan)),
                "test_f1": float(test_metrics.get("f1", np.nan)),
                "test_threshold": float(test_payload.get("threshold", np.nan))
                if "threshold" in test_payload
                else float(validation_payload.get("threshold", np.nan)),
                "checkpoint_path": str((run_dir / "best_checkpoint.pt").resolve()),
                "runtime_seconds": _infer_runtime_seconds(run_dir),
                "epochs_completed": epochs_completed,
                "sequence_length": int(run_config.get("sequence_length", 0)),
                "input_dim": int(run_config.get("input_dim", 0)),
                "static_dim": int(run_config.get("static_dim", 0)),
                "run_config_path": str(run_config_path.resolve()),
                "validation_metrics_path": str(validation_metrics_path.resolve()),
                "test_metrics_path": str(test_metrics_path.resolve()),
                "training_history_path": str(training_history_path.resolve()),
                "config": json.dumps(config_summary, sort_keys=True),
                "mlflow_tracking_uri": str(training_config.get("mlflow_tracking_uri", "")),
                "mlflow_experiment_name": str(training_config.get("mlflow_experiment_name", "")),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame

    frame = frame.sort_values(
        by=["validation_auprc", "validation_auroc", "test_auprc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return frame


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _to_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in frame[columns].iterrows():
        rendered = [_format_markdown_value(row[column]) for column in columns]
        rows.append("| " + " | ".join(rendered) + " |")

    return "\n".join([header, separator, *rows])


def export_experiment_comparison(
    frame: pd.DataFrame,
    *,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Exports experiment summary to CSV and compact Markdown report."""

    resolved_csv_path = Path(csv_path)
    resolved_markdown_path = Path(markdown_path)
    resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(resolved_csv_path, index=False)

    if frame.empty:
        report = "# Experiment Comparison\n\nNo eligible sequence runs were found."
        resolved_markdown_path.write_text(report, encoding="utf-8")
        return resolved_csv_path, resolved_markdown_path

    summary_columns = [
        "run_name",
        "model_type",
        "model_family",
        "dataset_tag",
        "validation_auprc",
        "validation_auroc",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_auprc",
        "runtime_seconds",
        "checkpoint_path",
    ]

    top_overall = frame.head(12)
    best_per_model = (
        frame.sort_values(["validation_auprc", "validation_auroc"], ascending=[False, False])
        .groupby("model_type", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    report_sections = [
        "# Experiment Comparison",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
        f"Total runs: {len(frame)}",
        "",
        "## Best Per Model",
        "",
        _to_markdown_table(best_per_model, summary_columns),
        "",
        "## Top Runs",
        "",
        _to_markdown_table(top_overall, summary_columns),
        "",
    ]

    resolved_markdown_path.write_text("\n".join(report_sections), encoding="utf-8")
    return resolved_csv_path, resolved_markdown_path


def select_best_run(
    frame: pd.DataFrame,
    *,
    selection_metric: str = "validation_auprc",
    dataset_tag: str | None = None,
) -> pd.Series:
    """Selects the best run row using a deterministic metric-first ranking."""

    if frame.empty:
        msg = "No experiments are available for selection"
        raise ValueError(msg)

    candidates = frame.copy()
    if dataset_tag is not None:
        candidates = candidates.loc[candidates["dataset_tag"] == dataset_tag].copy()

    if candidates.empty:
        msg = f"No experiments found for dataset_tag={dataset_tag!r}"
        raise ValueError(msg)

    if selection_metric not in candidates.columns:
        msg = f"Selection metric column not found: {selection_metric}"
        raise KeyError(msg)

    candidates["runtime_seconds"] = candidates["runtime_seconds"].fillna(np.inf)
    ranked = candidates.sort_values(
        by=[selection_metric, "validation_auroc", "test_auprc", "runtime_seconds"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def build_model_manifest_from_row(
    row: pd.Series,
    *,
    project_root: str | Path = ".",
    selection_metric: str = "validation_auprc",
) -> dict[str, Any]:
    """Builds a deployment-ready manifest for the selected run."""

    resolved_project_root = Path(project_root).resolve()
    run_config_path = Path(str(row["run_config_path"]))
    run_config = _json_load(run_config_path)

    model_config = run_config.get("model_config", {})
    training_config = run_config.get("training_config", {})
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        msg = f"Malformed run configuration at {run_config_path}"
        raise ValueError(msg)

    row_model_type = _normalize_model_type(row.get("model_type", "unknown"))
    config_model_type = _normalize_model_type(model_config.get("model_type", "unknown"))
    if config_model_type in SUPPORTED_MODEL_TYPES:
        selected_model_type = config_model_type
    elif row_model_type in SUPPORTED_MODEL_TYPES:
        selected_model_type = row_model_type
    else:
        msg = f"Unable to resolve supported model_type for run: {run_config_path}"
        raise ValueError(msg)
    selected_model_family = sequence_model_family_name(selected_model_type)

    windows_dir = Path(str(training_config.get("windows_dir", "")))
    dataset_context = _dataset_context_from_windows_dir(
        windows_dir, project_root=resolved_project_root
    )

    validation_payload = _json_load(Path(str(row["validation_metrics_path"])))
    test_payload = _json_load(Path(str(row["test_metrics_path"])))

    default_threshold = float(row.get("validation_threshold", 0.5))

    feature_columns = dataset_context["feature_columns"]
    static_feature_columns = dataset_context["static_feature_columns"]

    manifest = {
        "schema_version": "1.0",
        "selected_at": datetime.now(UTC).isoformat(),
        "selection_metric": selection_metric,
        "selected_run": {
            "run_name": str(row["run_name"]),
            "run_dir": str(Path(str(row["run_dir"])).resolve()),
            "checkpoint_path": str(Path(str(row["checkpoint_path"])).resolve()),
            "run_config_path": str(run_config_path.resolve()),
            "model_type": selected_model_type,
            "model_family": selected_model_family,
        },
        "dataset": {
            "dataset_tag": dataset_context["dataset_tag"],
            "dataset_format": dataset_context["dataset_format"],
            "raw_data_path": dataset_context["raw_data_path"],
            "windows_dir": dataset_context["windows_dir"],
            "processed_dir": dataset_context["processed_dir"],
            "feature_columns": feature_columns,
            "mask_columns": dataset_context["mask_columns"],
            "static_feature_columns": static_feature_columns,
            "feature_signature": build_feature_signature(feature_columns),
        },
        "model": {
            "model_type": selected_model_type,
            "model_family": selected_model_family,
            "input_dim": int(run_config.get("input_dim", row.get("input_dim", 0))),
            "static_dim": int(run_config.get("static_dim", row.get("static_dim", 0))),
            "window_length": int(run_config.get("sequence_length", row.get("sequence_length", 0))),
            "include_mask": bool(model_config.get("include_mask", True)),
            "include_static": bool(model_config.get("include_static", True)),
        },
        "thresholds": {
            "default": default_threshold,
            "balanced": default_threshold,
            "high_recall": default_threshold,
        },
        "metrics": {
            "validation": validation_payload.get("metrics", {}),
            "test": test_payload.get("metrics", {}),
        },
        "mlflow": {
            "tracking_uri": str(training_config.get("mlflow_tracking_uri", "")),
            "experiment_name": str(training_config.get("mlflow_experiment_name", "")),
        },
    }

    return manifest


def build_threshold_sweep(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.01,
) -> pd.DataFrame:
    """Builds a threshold sweep table for operating-point selection."""

    thresholds = np.arange(start, end + 1e-12, step)
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))
        rows.append(metrics.to_dict())

    return pd.DataFrame(rows)


def recommend_operating_thresholds(
    threshold_sweep: pd.DataFrame,
    *,
    high_recall_target: float = 0.9,
) -> dict[str, float]:
    """Returns recommended thresholds for balanced and high-recall operating modes."""

    if threshold_sweep.empty:
        msg = "Threshold sweep cannot be empty"
        raise ValueError(msg)

    balanced_row = threshold_sweep.sort_values(
        by=["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]

    high_recall_candidates = threshold_sweep.loc[threshold_sweep["recall"] >= high_recall_target]
    if high_recall_candidates.empty:
        high_recall_row = threshold_sweep.sort_values(
            by=["recall", "precision", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        high_recall_row = high_recall_candidates.sort_values(
            by=["precision", "f1", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    return {
        "balanced": float(balanced_row["threshold"]),
        "high_recall": float(high_recall_row["threshold"]),
    }


def build_reliability_curve(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Builds reliability-curve statistics with fixed-width bins."""

    if bins <= 0:
        msg = "bins must be greater than zero"
        raise ValueError(msg)

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []

    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        in_bin = (y_prob >= lower) & (y_prob < upper if index < bins - 1 else y_prob <= upper)
        count = int(in_bin.sum())
        if count == 0:
            rows.append(
                {
                    "bin_index": index,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "sample_count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_rate": np.nan,
                }
            )
            continue

        rows.append(
            {
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_count": count,
                "mean_predicted_probability": float(y_prob[in_bin].mean()),
                "observed_positive_rate": float(y_true[in_bin].mean()),
            }
        )

    return pd.DataFrame(rows)


def generate_evaluation_plots(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> dict[str, Path]:
    """Generates ROC/PR/confusion/distribution plots and returns file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)

    roc_path = output_dir / "roc_curve.png"
    pr_path = output_dir / "pr_curve.png"
    confusion_path = output_dir / "confusion_matrix.png"
    distribution_path = output_dir / "score_distribution.png"

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        roc_auc = float("nan")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.4f}" if not np.isnan(roc_auc) else "AUROC=nan")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC={ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # Confusion matrix at selected threshold
    predictions = (y_prob >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    plt.figure(figsize=(5.2, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                int(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=150)
    plt.close()

    # Score distribution by class
    positives = y_prob[y_true == 1]
    negatives = y_prob[y_true == 0]

    plt.figure(figsize=(6.4, 4.8))
    plt.hist(negatives, bins=30, alpha=0.65, label="Negative", color="#1f77b4", density=True)
    plt.hist(positives, bins=30, alpha=0.65, label="Positive", color="#d62728", density=True)
    plt.axvline(
        threshold, color="black", linestyle="--", linewidth=1.2, label=f"Threshold={threshold:.2f}"
    )
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Score Distribution by True Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=150)
    plt.close()

    return {
        "roc_curve": roc_path,
        "pr_curve": pr_path,
        "confusion_matrix": confusion_path,
        "score_distribution": distribution_path,
    }


def analyze_checkpoint_calibration(
    *,
    checkpoint_path: str | Path,
    parquet_path: str | Path,
    output_dir: str | Path,
    default_threshold: float,
    high_recall_target: float = 0.9,
    calibration_bins: int = 10,
    batch_size: int = 256,
    num_workers: int = 0,
) -> CalibrationAnalysisArtifacts:
    """Produces calibration outputs, threshold sweep, and evaluation plots."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        parquet_path=parquet_path,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=default_threshold,
        calibration_bins=calibration_bins,
    )

    y_true = np.asarray(evaluation["targets"], dtype=np.int64)
    y_prob = np.asarray(evaluation["probabilities"], dtype=np.float64)

    threshold_sweep = build_threshold_sweep(y_true=y_true, y_prob=y_prob)
    recommendations = recommend_operating_thresholds(
        threshold_sweep,
        high_recall_target=high_recall_target,
    )

    reliability_curve = build_reliability_curve(
        y_true=y_true,
        y_prob=y_prob,
        bins=calibration_bins,
    )

    balanced_threshold = recommendations["balanced"]
    plot_paths = generate_evaluation_plots(
        y_true=y_true,
        y_prob=y_prob,
        threshold=balanced_threshold,
        output_dir=resolved_output_dir,
    )

    threshold_sweep_path = resolved_output_dir / "threshold_sweep.csv"
    reliability_curve_path = resolved_output_dir / "reliability_curve.csv"
    recommendations_path = resolved_output_dir / "threshold_recommendations.json"
    summary_path = resolved_output_dir / "calibration_summary.json"
    markdown_report_path = resolved_output_dir / "calibration_report.md"

    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    reliability_curve.to_csv(reliability_curve_path, index=False)

    recommendations_payload = {
        "default": float(default_threshold),
        "balanced": float(recommendations["balanced"]),
        "high_recall": float(recommendations["high_recall"]),
        "high_recall_target": float(high_recall_target),
    }
    _json_dump(recommendations_path, recommendations_payload)

    summary_payload = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "parquet_path": str(Path(parquet_path).resolve()),
        "sample_count": len(y_true),
        "positive_rate": float(y_true.mean()) if len(y_true) > 0 else 0.0,
        "default_threshold": float(default_threshold),
        "default_metrics": evaluation["metrics"],
        "recommended_thresholds": recommendations_payload,
        "plot_paths": {name: str(path.resolve()) for name, path in plot_paths.items()},
    }
    _json_dump(summary_path, summary_payload)

    report_lines = [
        "# Calibration Analysis",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
        f"Checkpoint: {summary_payload['checkpoint_path']}",
        f"Evaluation split: {summary_payload['parquet_path']}",
        f"Sample count: {summary_payload['sample_count']}",
        f"Positive rate: {summary_payload['positive_rate']:.6f}",
        "",
        "## Recommended Thresholds",
        "",
        f"- Default: {recommendations_payload['default']:.6f}",
        f"- Balanced (max F1): {recommendations_payload['balanced']:.6f}",
        (
            f"- High Recall (target={recommendations_payload['high_recall_target']:.2f}): "
            f"{recommendations_payload['high_recall']:.6f}"
        ),
        "",
        "## Default Metrics",
        "",
        f"- AUROC: {float(evaluation['metrics'].get('auroc', float('nan'))):.6f}",
        f"- AUPRC: {float(evaluation['metrics'].get('auprc', float('nan'))):.6f}",
        f"- Precision: {float(evaluation['metrics'].get('precision', float('nan'))):.6f}",
        f"- Recall: {float(evaluation['metrics'].get('recall', float('nan'))):.6f}",
        f"- F1: {float(evaluation['metrics'].get('f1', float('nan'))):.6f}",
        "",
        "## Output Files",
        "",
        f"- Threshold sweep CSV: {threshold_sweep_path}",
        f"- Reliability curve CSV: {reliability_curve_path}",
        f"- Recommendations JSON: {recommendations_path}",
        f"- Summary JSON: {summary_path}",
        f"- ROC curve: {plot_paths['roc_curve']}",
        f"- PR curve: {plot_paths['pr_curve']}",
        f"- Confusion matrix: {plot_paths['confusion_matrix']}",
        f"- Score distribution: {plot_paths['score_distribution']}",
        "",
    ]
    markdown_report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return CalibrationAnalysisArtifacts(
        output_dir=resolved_output_dir,
        threshold_sweep_path=threshold_sweep_path,
        reliability_curve_path=reliability_curve_path,
        recommendations_path=recommendations_path,
        summary_path=summary_path,
        markdown_report_path=markdown_report_path,
        plot_paths=plot_paths,
        recommendations={
            "default": recommendations_payload["default"],
            "balanced": recommendations_payload["balanced"],
            "high_recall": recommendations_payload["high_recall"],
        },
    )
