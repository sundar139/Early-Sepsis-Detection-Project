from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from early_sepsis.demo.presentation import (
    CALIBRATION_UNAVAILABLE_GUIDANCE,
    METRIC_LABELS,
    OPERATIONAL_UNAVAILABLE_GUIDANCE,
    PLOT_TITLES,
    SAVED_WALKTHROUGH_UNAVAILABLE_GUIDANCE,
    build_metric_annotation,
    collect_metric_snapshot,
    collect_plot_artifacts,
    compute_operational_metrics,
    describe_threshold_mode,
    detect_latest_pytest_status,
    find_duplicate_threshold_modes,
    format_threshold_mode,
    load_experiment_comparison,
    load_feature_importance_artifact,
    load_reliability_curve,
    resolve_calibration_summary,
    sanitize_public_text,
)
from early_sepsis.demo.startup import (
    DemoInferenceSource,
    DemoStartupError,
    build_saved_example_walkthrough_sample,
    resolve_demo_inference_source,
    resolve_manifest_path,
    validate_demo_startup,
)
from early_sepsis.runtime_paths import resolve_runtime_path
from early_sepsis.serving.sequence_service import (
    SequenceServingError,
    predict_sequence_samples,
    resolve_operating_threshold,
)
from early_sepsis.settings import get_settings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_github_repo_url(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None

    if normalized.startswith("git@github.com:"):
        normalized = normalized.replace("git@github.com:", "https://github.com/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    if normalized.startswith("https://github.com/"):
        return normalized

    return None


def _resolve_repo_url(*, settings_repo_url: str | None, project_root: Path) -> str | None:
    explicit_value = _normalize_github_repo_url(settings_repo_url or "")
    if explicit_value is not None:
        return explicit_value

    config_path = project_root / ".git" / "config"
    if not config_path.exists():
        return None

    parser = configparser.ConfigParser()
    try:
        parser.read(config_path, encoding="utf-8")
    except Exception:
        return None

    remote_section = 'remote "origin"'
    if not parser.has_option(remote_section, "url"):
        return None

    return _normalize_github_repo_url(parser.get(remote_section, "url"))


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(
                160% 140% at 85% -10%,
                #1d3153 0%,
                #0a1020 48%,
                #060913 100%
            );
            color: #e8eefc;
        }
        .main .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1630 0%, #0a1328 100%);
            border-right: 1px solid rgba(125, 158, 214, 0.24);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-bottom: 1.2rem;
        }
        [data-testid="stSidebar"] h3 {
            color: #eef5ff;
            font-size: 1rem;
            letter-spacing: 0.02em;
            margin-bottom: 0.45rem;
        }
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stSelectbox {
            background: rgba(18, 32, 60, 0.72);
            border: 1px solid rgba(109, 143, 203, 0.25);
            border-radius: 10px;
            padding: 0.45rem 0.55rem 0.2rem;
            margin-bottom: 0.55rem;
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(20, 35, 62, 0.95), rgba(12, 22, 38, 0.9));
            border: 1px solid rgba(119, 156, 221, 0.24);
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.32);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            margin-bottom: 0.35rem;
            color: #f0f5ff;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #c0d0ef;
            line-height: 1.55;
            max-width: 820px;
        }
        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }
        .hero-link-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 9px;
            border: 1px solid rgba(150, 180, 232, 0.42);
            background: rgba(41, 66, 105, 0.9);
            color: #f2f7ff !important;
            font-size: 0.83rem;
            font-weight: 620;
            text-decoration: none !important;
            padding: 0.42rem 0.78rem;
        }
        .hero-link-button:hover {
            background: rgba(57, 86, 130, 0.95);
            border-color: rgba(171, 198, 244, 0.7);
        }
        .why-matters-card {
            background: rgba(18, 29, 48, 0.82);
            border: 1px solid rgba(112, 146, 208, 0.3);
            border-radius: 13px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 1rem;
            color: #d2e0f7;
            line-height: 1.5;
            font-size: 0.9rem;
        }
        .disclaimer-banner {
            margin-top: 1rem;
            background: rgba(80, 34, 40, 0.34);
            border: 1px solid rgba(255, 135, 150, 0.32);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            color: #ffd8df;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.95rem;
        }
        .badge-pill {
            border-radius: 999px;
            border: 1px solid rgba(141, 173, 229, 0.3);
            background: rgba(34, 53, 86, 0.7);
            color: #d7e6ff;
            padding: 0.3rem 0.72rem;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.25rem 0 1rem;
        }
        .status-chip {
            background: rgba(22, 35, 61, 0.84);
            border: 1px solid rgba(120, 156, 221, 0.25);
            color: #d6e5ff;
            border-radius: 10px;
            padding: 0.42rem 0.72rem;
            font-size: 0.82rem;
            line-height: 1.2;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }
        .status-chip strong {
            color: #f0f6ff;
            font-weight: 650;
        }
        .section-title {
            font-size: 1.18rem;
            font-weight: 760;
            margin: 0.35rem 0 0.7rem;
            color: #f6fbff;
            letter-spacing: 0.2px;
        }
        .surface-card {
            background: rgba(16, 25, 44, 0.86);
            border: 1px solid rgba(98, 130, 188, 0.25);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.75rem;
            min-height: 126px;
            break-inside: avoid;
            page-break-inside: avoid;
        }
        .visual-card {
            break-inside: avoid;
            page-break-inside: avoid;
            margin-bottom: 0.7rem;
        }
        .visual-card--large {
            break-inside: avoid-page;
            page-break-inside: avoid;
        }
        .artifact-unavailable-card {
            background: rgba(26, 36, 56, 0.86);
            border: 1px dashed rgba(136, 164, 213, 0.4);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-top: 0.4rem;
            color: #c7d5ee;
            font-size: 0.86rem;
            line-height: 1.45;
            break-inside: avoid;
            page-break-inside: avoid;
        }
        .calibration-explainer-card {
            background: rgba(22, 34, 56, 0.9);
            border: 1px solid rgba(126, 160, 221, 0.38);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            margin-top: 0.45rem;
            color: #ccdaf1;
            font-size: 0.86rem;
            line-height: 1.48;
            break-inside: avoid;
            page-break-inside: avoid;
        }
        .card-title {
            font-size: 0.82rem;
            font-weight: 520;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9eb3d8;
            margin-bottom: 0.5rem;
        }
        .card-value {
            font-size: 1.22rem;
            font-weight: 700;
            color: #f4f8ff;
            margin-bottom: 0.35rem;
        }
        .card-subtitle {
            font-size: 0.82rem;
            color: #b5c3df;
            line-height: 1.45;
        }
        .footer-note {
            margin-top: 1rem;
            color: #9eafce;
            font-size: 0.86rem;
            line-height: 1.5;
        }
        .inference-card {
            break-inside: avoid;
            page-break-inside: avoid;
        }
        @media print {
            .stApp {
                background: #ffffff !important;
                color: #111111 !important;
            }
            [data-testid="stSidebar"],
            .stButton,
            .stSlider,
            .stMultiSelect,
            .stSelectbox,
            .stRadio,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"] {
                display: none !important;
                visibility: hidden !important;
            }
            .hero-panel,
            .surface-card,
            .inference-card,
            .visual-card,
            .visual-card--large,
            .artifact-unavailable-card,
            .calibration-explainer-card,
            [data-testid="stImage"],
            [data-testid="stLineChart"],
            [data-testid="stDataFrame"] {
                break-inside: avoid !important;
                page-break-inside: avoid !important;
            }
            .evaluation-grid [data-testid="column"] {
                width: 100% !important;
                max-width: 100% !important;
                flex: 1 0 100% !important;
            }
            .operational-grid [data-testid="column"] {
                width: 100% !important;
                max-width: 100% !important;
                flex: 1 0 100% !important;
            }
            .section-title {
                color: #0a0a0a !important;
                font-weight: 800 !important;
                letter-spacing: 0.03em !important;
            }
            .hero-title,
            .card-value,
            .card-title,
            .card-subtitle,
            .footer-note,
            .status-chip,
            .status-chip strong {
                color: #111111 !important;
            }
            .hero-panel,
            .surface-card,
            .artifact-unavailable-card,
            .calibration-explainer-card,
            .status-chip {
                border-color: #a6a6a6 !important;
                background: #ffffff !important;
                box-shadow: none !important;
            }
            .disclaimer-banner {
                background: #f5f5f5 !important;
                border-color: #b3b3b3 !important;
                color: #111111 !important;
            }
            a,
            a:visited {
                color: #111111 !important;
                text-decoration: none !important;
            }
            .main .block-container {
                padding-bottom: 0.6rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_split_samples(parquet_path: str, max_rows: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        parquet_path,
        columns=["patient_id", "end_hour", "label", "features", "missing_mask", "static_features"],
    )
    return frame.head(max_rows).copy()


@st.cache_data(show_spinner=False)
def _load_operational_probability_frame(
    *,
    manifest_path: str,
    manifest_mtime: float,
    dataset_tag: str,
    parquet_path: str,
    max_rows: int,
) -> pd.DataFrame:
    split_frame = _load_split_samples(parquet_path, max_rows)
    if split_frame.empty:
        return pd.DataFrame(
            columns=["Sample", "End Hour", "Observed Label", "Risk Score"]
        )

    request_samples = [_row_to_request_sample(row) for _, row in split_frame.iterrows()]
    predictions = predict_sequence_samples(
        manifest_path=manifest_path,
        dataset_tag=dataset_tag,
        samples=request_samples,
        operating_mode="default",
    )

    rows: list[dict[str, Any]] = []
    for index, (row, prediction) in enumerate(
        zip(split_frame.to_dict("records"), predictions, strict=True)
    ):
        rows.append(
            {
                "Sample": f"S{index + 1:03d}",
                "End Hour": int(row.get("end_hour", 0)),
                "Observed Label": int(row.get("label", 0)),
                "Risk Score": float(prediction["predicted_probability"]),
            }
        )

    return pd.DataFrame(rows)


def _row_to_request_sample(row: pd.Series) -> dict[str, Any]:
    def _to_matrix_list(value: Any) -> list[list[float]]:
        array = np.asarray(value)
        if array.ndim == 1 and array.dtype == object:
            array = np.asarray(array.tolist(), dtype=np.float32)
        else:
            array = np.asarray(value, dtype=np.float32)

        if array.ndim != 2:
            msg = "Expected a 2D matrix payload for inference input"
            raise ValueError(msg)

        return array.tolist()

    def _to_vector_list(value: Any) -> list[float] | None:
        if value is None:
            return None

        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 1:
            msg = "Expected a 1D static-feature vector"
            raise ValueError(msg)

        return array.tolist()

    return {
        "patient_id": row.get("patient_id"),
        "end_hour": int(row.get("end_hour", 0)),
        "features": _to_matrix_list(row.get("features")),
        "missing_mask": _to_matrix_list(row.get("missing_mask")),
        "static_features": _to_vector_list(row.get("static_features")),
    }


def _build_inference_result_frame(
    *,
    preview_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for preview_row, prediction in zip(preview_rows, predictions, strict=True):
        probability = float(prediction["predicted_probability"])
        threshold_used = float(prediction["threshold_used"])
        predicted_label = int(prediction["predicted_label"])
        rows.append(
            {
                "Sample": preview_row["Sample"],
                "Sample ID": preview_row["Sample ID"],
                "End Hour": prediction["end_hour"],
                "Risk Score": round(probability, 6),
                "Predicted Class": "Alert" if predicted_label == 1 else "No Alert",
                "Threshold Mode": format_threshold_mode(prediction["operating_mode"]),
                "Threshold Used": round(threshold_used, 4),
                "Interpretation": _risk_interpretation(
                    probability=probability,
                    threshold=threshold_used,
                ),
            }
        )

    return pd.DataFrame(rows)


def _deterministic_explanation(
    *,
    sample: dict[str, Any],
    feature_names: list[str],
    predicted_probability: float,
    threshold: float,
) -> str:
    feature_window = np.asarray(sample["features"], dtype=np.float64)
    if feature_window.ndim != 2:
        return "Unable to render deterministic explanation because feature window shape is invalid."

    latest_values = feature_window[-1]
    baseline_values = feature_window[0]
    trend_values = latest_values - baseline_values

    top_magnitude_indices = np.argsort(np.abs(latest_values))[-3:][::-1]
    top_trend_indices = np.argsort(np.abs(trend_values))[-2:][::-1]

    magnitude_parts = []
    for index in top_magnitude_indices:
        feature_name = feature_names[index] if index < len(feature_names) else f"feature_{index}"
        magnitude_parts.append(f"{feature_name}={latest_values[index]:.3f}")

    trend_parts = []
    for index in top_trend_indices:
        feature_name = feature_names[index] if index < len(feature_names) else f"feature_{index}"
        trend_parts.append(f"{feature_name} delta={trend_values[index]:+.3f}")

    risk_side = "above" if predicted_probability >= threshold else "below"
    margin = abs(predicted_probability - threshold)

    return (
        "Heuristic summary: "
        f"latest strongest signals ({', '.join(magnitude_parts)}); "
        f"largest trajectory changes ({', '.join(trend_parts)}). "
        f"Predicted risk {predicted_probability:.3f} is {risk_side} threshold {threshold:.3f} "
        f"by {margin:.3f}."
    )


def _risk_interpretation(*, probability: float, threshold: float) -> str:
    margin = probability - threshold
    if margin >= 0.15:
        return "Score is substantially above threshold, indicating elevated risk for this window."
    if margin >= 0.0:
        return "Score is above threshold, indicating a positive risk alert for this window."
    if margin >= -0.08:
        return "Score is close to the operating threshold; monitor trajectory context carefully."
    return "Score is below threshold and does not trigger a positive alert in this mode."


def _format_metric_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def resolve_demo_threshold_for_mode(manifest: dict[str, Any], operating_mode: str) -> float:
    """Resolves display and inference threshold for the selected demo operating mode."""

    resolved_threshold, _ = resolve_operating_threshold(
        manifest,
        operating_mode=operating_mode,
        threshold_override=None,
    )
    return resolved_threshold


def _render_card(*, title: str, value: str, subtitle: str = "") -> None:
    subtitle_html = f"<div class='card-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        (
            "<div class='surface-card'>"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-value'>{value}</div>"
            f"{subtitle_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_artifact_unavailable_card(*, title: str, guidance: str) -> None:
    st.markdown(
        (
            "<div class='artifact-unavailable-card'>"
            f"<strong>{title} unavailable.</strong><br>{guidance}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_hero_section(
    *,
    model_family: str,
    dataset_tag: str,
    window_length: int,
    feature_count: int,
    status: str,
    repo_url: str | None,
) -> None:
    badges = [
        f"Model Family: {model_family}",
        f"Data Source: {dataset_tag}",
        f"Window Length: {window_length}",
        f"Feature Count: {feature_count}",
        f"Status: {status}",
    ]
    badges_markup = "".join(f"<span class='badge-pill'>{badge}</span>" for badge in badges)
    actions_markup = ""
    if repo_url:
        actions_markup = (
            "<div class='hero-actions'>"
            f"<a class='hero-link-button' href='{repo_url}' target='_blank' "
            "rel='noopener noreferrer'>View GitHub Repository</a>"
            "</div>"
        )

    st.markdown(
        (
            "<section class='hero-panel'>"
            "<div class='hero-title'>Early Sepsis Detection</div>"
            "<div class='hero-subtitle'>"
            "Sequence-model inference for early sepsis risk scoring from ICU windows. "
            "Designed for transparent model demonstration, threshold strategy review, and "
            "artifact-backed evaluation."
            "</div>"
            "<div class='disclaimer-banner'>"
            "Research use only. This demonstration is not a medical device and must not be "
            "used for clinical diagnosis or treatment decisions."
            "</div>"
            f"{actions_markup}"
            f"<div class='badge-row'>{badges_markup}</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _render_why_this_matters() -> None:
    st.markdown("<div class='section-title'>Why This Matters</div>", unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='why-matters-card'>"
            "Sepsis can escalate rapidly, and delayed recognition increases mortality risk and "
            "critical-care burden. Earlier risk visibility can help teams prioritize monitoring "
            "and escalation conversations before deterioration becomes irreversible."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_project_status_strip(
    *,
    calibration_synced: bool,
    tests_status: str,
    threshold_modes: list[str],
    plot_count: int,
) -> None:
    chips: list[tuple[str, str]] = [
        ("Model", "Ready"),
        ("Calibration", "Synced" if calibration_synced else "Pending"),
        ("Threshold Modes", str(len(threshold_modes))),
        ("Visual Artifacts", str(plot_count)),
    ]
    if tests_status != "Not available":
        chips.append(("Test Signal", tests_status))

    chips_markup = "".join(
        f"<span class='status-chip'><strong>{label}:</strong> {value}</span>"
        for label, value in chips
    )
    st.markdown(f"<div class='status-strip'>{chips_markup}</div>", unsafe_allow_html=True)


def _render_system_status(
    *,
    manifest: dict[str, Any],
    threshold_modes: list[str],
    calibration_available: bool,
) -> None:
    dataset = manifest.get("dataset", {})
    model = manifest.get("model", {})
    selected_run = manifest.get("selected_run", {})

    feature_count = len(dataset.get("feature_columns", []))
    window_length = int(model.get("window_length", 0))
    model_type = str(model.get("model_type", "unknown")).upper()
    model_family = str(selected_run.get("model_family", "unknown"))

    st.markdown("<div class='section-title'>System Status Overview</div>", unsafe_allow_html=True)
    columns = st.columns(4)
    cards = [
        ("Model Availability", "Ready", "Selected checkpoint and manifest validated."),
        ("Selected Model", model_type, f"{model_family}"),
        ("Dataset", str(dataset.get("dataset_tag", "unknown")), "Configured selected-model tag."),
        ("Window Length", f"{window_length}", "Timesteps per inference sample."),
        ("Feature Count", f"{feature_count}", "Dynamic sequence feature dimensions."),
        ("Threshold Modes", f"{len(threshold_modes)}", "Default, Balanced, and High Recall."),
        (
            "Last Evaluation",
            "Available" if calibration_available else "Limited",
            "Calibration artifacts detected."
            if calibration_available
            else "Using manifest metrics only.",
        ),
        (
            "Serving Profile",
            "Manifest-backed",
            "Runtime checks enforce dimensions and dataset tag compatibility.",
        ),
    ]

    for index, (title, value, subtitle) in enumerate(cards):
        with columns[index % 4]:
            _render_card(title=title, value=value, subtitle=subtitle)


def _render_performance_summary(
    *,
    metric_snapshot: dict[str, float | None],
    metric_source: str,
    calibration_summary: dict[str, Any] | None,
    comparison_frame: pd.DataFrame | None,
) -> None:
    st.markdown(
        "<div class='section-title'>Threshold-Invariant Evaluation Summary</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "These model-evaluation metrics are threshold-invariant and do not change when the "
        "operating mode changes."
    )
    st.caption(
        "Threshold-dependent alerting behavior is summarized separately in the operational "
        "section below."
    )
    st.caption(f"Metric source: {metric_source}")

    invariant_metric_keys = (
        "auroc",
        "auprc",
        "brier_score",
        "expected_calibration_error",
    )
    columns = st.columns(len(invariant_metric_keys) + 1)
    for index, metric_key in enumerate(invariant_metric_keys):
        with columns[index]:
            _render_card(
                title=METRIC_LABELS[metric_key],
                value=_format_metric_value(metric_snapshot.get(metric_key)),
                subtitle=build_metric_annotation(metric_key),
            )

    prevalence_value: float | None = None
    if calibration_summary is not None:
        positive_rate = calibration_summary.get("positive_rate")
        if isinstance(positive_rate, (float, int)):
            prevalence_value = float(positive_rate)

    with columns[-1]:
        _render_card(
            title="Dataset Prevalence",
            value=_format_percent(prevalence_value),
            subtitle=(
                "Lower prevalence increases class imbalance and typically makes AUPRC more "
                "decision-relevant than AUROC for deployment review."
            ),
        )

    st.info(
        "In low-prevalence sepsis screening, AUPRC usually provides more deployment-relevant "
        "signal than AUROC because it reflects precision-recall tradeoffs under class imbalance."
    )

    if calibration_summary is not None:
        sample_count = calibration_summary.get("sample_count")
        if isinstance(sample_count, int) and prevalence_value is not None:
            st.caption(
                "Calibration context: "
                f"{sample_count} windows analyzed, positive prevalence {prevalence_value:.3f}."
            )

    if comparison_frame is not None:
        st.markdown("#### Compact Top-Run Comparison")
        st.dataframe(comparison_frame, width="stretch", hide_index=True)


def _render_threshold_strategy(
    *,
    manifest: dict[str, Any],
) -> tuple[str, float, list[tuple[float, tuple[str, ...]]]]:
    thresholds = manifest.get("thresholds", {})
    available_modes = [
        mode for mode in ("default", "balanced", "high_recall") if mode in thresholds
    ]
    if not available_modes:
        msg = "Selected manifest is missing configured threshold modes."
        raise SequenceServingError(msg)

    st.markdown("<div class='section-title'>Threshold Strategy</div>", unsafe_allow_html=True)
    selected_mode = st.radio(
        "Inference threshold mode",
        options=available_modes,
        horizontal=True,
        format_func=format_threshold_mode,
        help=(
            "Choose which manifest-configured operating threshold to apply. This does not change "
            "model weights or probability math; it only changes alert decision cutoff."
        ),
    )

    columns = st.columns(len(available_modes))
    for column, mode in zip(columns, available_modes, strict=True):
        threshold_value = float(thresholds[mode])
        with column:
            _render_card(
                title=format_threshold_mode(mode),
                value=f"{threshold_value:.3f}",
                subtitle=describe_threshold_mode(mode),
            )

    duplicate_thresholds = find_duplicate_threshold_modes(
        thresholds,
        modes=available_modes,
    )
    if duplicate_thresholds:
        duplicate_notes = "; ".join(
            (
                f"{', '.join(format_threshold_mode(mode) for mode in modes)}"
                f" all map to {threshold_value:.3f}"
            )
            for threshold_value, modes in duplicate_thresholds
        )
        st.info(
            "Some operating modes currently share the same threshold for this selected model: "
            f"{duplicate_notes}."
        )

    st.caption(
        "Mode selection changes only the decision threshold. Model weights, preprocessing, and "
        "inference logic remain unchanged."
    )
    selected_threshold = float(thresholds[selected_mode])
    return selected_mode, selected_threshold, duplicate_thresholds


def _render_operational_summary(
    *,
    operational_frame: pd.DataFrame,
    operating_mode: str,
    applied_threshold: float,
    source_label: str,
    duplicate_thresholds: list[tuple[float, tuple[str, ...]]],
    unavailable_reason: str | None = None,
) -> None:
    st.markdown(
        "<div class='section-title'>Threshold-Dependent Operational Summary</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "These operational metrics update with the selected operating mode and threshold. "
        f"Source: {source_label}."
    )

    if operational_frame.empty:
        _render_artifact_unavailable_card(
            title="Operational summary",
            guidance=(
                f"{OPERATIONAL_UNAVAILABLE_GUIDANCE} "
                f"{unavailable_reason or ''}"
            ).strip(),
        )
        return

    metrics = compute_operational_metrics(
        probabilities=operational_frame["Risk Score"].to_numpy(),
        labels=operational_frame["Observed Label"].to_numpy(),
        threshold=applied_threshold,
    )

    cards = [
        (
            "Applied Threshold",
            f"{applied_threshold:.3f}",
            f"Current mode: {format_threshold_mode(operating_mode)}",
        ),
        (
            "Alert Windows",
            f"{int(metrics['alert_count'])}",
            f"of {int(metrics['sample_count'])} analyzed windows",
        ),
        (
            "Alert Rate",
            _format_percent(float(metrics["predicted_positive_rate"])),
            "Predicted positive rate under current threshold.",
        ),
        (
            "Observed Prevalence",
            _format_percent(float(metrics["positive_rate"])),
            "Ground-truth positive rate in analyzed windows.",
        ),
        (
            "Precision",
            _format_metric_value(float(metrics["precision"])),
            "Positive predictive value.",
        ),
        (
            "Sensitivity",
            _format_metric_value(float(metrics["sensitivity"])),
            "True positive rate (recall).",
        ),
        (
            "Specificity",
            _format_metric_value(float(metrics["specificity"])),
            "True negative rate.",
        ),
        (
            "Balanced Accuracy",
            _format_metric_value(float(metrics["balanced_accuracy"])),
            "Mean of sensitivity and specificity.",
        ),
    ]

    card_columns = st.columns(4)
    for index, (title, value, subtitle) in enumerate(cards):
        with card_columns[index % 4]:
            _render_card(title=title, value=value, subtitle=subtitle)

    shared_modes = [
        modes
        for threshold_value, modes in duplicate_thresholds
        if abs(threshold_value - applied_threshold) <= 1e-6
    ]
    if shared_modes:
        matching_modes = ", ".join(format_threshold_mode(mode) for mode in shared_modes[0])
        st.info(
            "Operational outputs remain identical across modes sharing this threshold: "
            f"{matching_modes}."
        )

    confusion_table = pd.DataFrame(
        {
            "Predicted No Alert": [
                int(metrics["true_negative"]),
                int(metrics["false_negative"]),
            ],
            "Predicted Alert": [
                int(metrics["false_positive"]),
                int(metrics["true_positive"]),
            ],
        },
        index=["Actual No Alert", "Actual Alert"],
    )

    histogram_bins = np.linspace(0.0, 1.0, num=11)
    histogram_counts, histogram_edges = np.histogram(
        operational_frame["Risk Score"].to_numpy(),
        bins=histogram_bins,
    )
    histogram_labels = [
        f"{histogram_edges[index]:.1f}-{histogram_edges[index + 1]:.1f}"
        for index in range(len(histogram_edges) - 1)
    ]
    distribution_frame = pd.DataFrame(
        {
            "Risk Score Bin": histogram_labels,
            "Window Count": histogram_counts,
        }
    ).set_index("Risk Score Bin")

    st.markdown("<div class='operational-grid'>", unsafe_allow_html=True)
    chart_columns = st.columns(2)
    with chart_columns[0], st.container(border=True):
        st.markdown("<div class='visual-card visual-card--large'>", unsafe_allow_html=True)
        st.markdown("**Confusion Matrix (Current Threshold)**")
        st.dataframe(confusion_table, width="stretch")
        st.caption("Rows are actual labels; columns are predictions under the selected threshold.")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_columns[1], st.container(border=True):
        st.markdown("<div class='visual-card visual-card--large'>", unsafe_allow_html=True)
        st.markdown("**Score Distribution (Current Selection)**")
        st.bar_chart(distribution_frame, height=280)
        st.caption(
            f"Applied threshold {applied_threshold:.3f}. "
            "Scores at or above threshold count as alerts."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _resolve_operational_source(
    *,
    public_artifacts_root: Path,
    manifest_path: Path,
    dataset_section: dict[str, Any],
    split: str,
    inference_source: DemoInferenceSource,
) -> tuple[str, Path | None, str | None]:
    candidates: list[tuple[str, Path]] = [
        (
            "Deployment operational subset",
            public_artifacts_root / "demo" / "operational_windows_subset.parquet",
        ),
        (
            "Bundled operational subset",
            resolve_runtime_path(
                Path("assets/demo/operational_windows_subset.parquet"),
                anchor=manifest_path.parent,
            ),
        ),
    ]

    windows_dir_value = dataset_section.get("windows_dir")
    if isinstance(windows_dir_value, str) and windows_dir_value.strip():
        candidates.append(
            (
                f"{split.capitalize()} evaluation windows",
                resolve_runtime_path(
                    Path(windows_dir_value) / f"{split}.parquet",
                    anchor=manifest_path.parent,
                ),
            )
        )

    if inference_source.parquet_path is not None:
        candidates.append((inference_source.source_label, inference_source.parquet_path))

    for label, candidate_path in candidates:
        if candidate_path.exists() and candidate_path.is_file():
            return label, candidate_path, None

    return (
        "Deployment operational subset",
        None,
        "No deployment-safe operational subset or evaluation windows were bundled.",
    )


def _render_evaluation_visuals(
    *,
    plot_paths: dict[str, Path],
    reliability_curve: pd.DataFrame | None,
) -> None:
    st.markdown(
        "<div class='section-title'>Evaluation Visuals (Threshold-Invariant)</div>",
        unsafe_allow_html=True,
    )
    st.caption("These artifact-backed plots describe model ranking and calibration behavior.")
    st.markdown("<div class='evaluation-grid'>", unsafe_allow_html=True)
    columns = st.columns(2)

    invariant_plot_keys = ("roc_curve", "pr_curve", "reliability_curve")
    for index, plot_key in enumerate(invariant_plot_keys):
        card_class = "visual-card"

        with columns[index % 2], st.container(border=True):
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
            st.markdown(f"**{PLOT_TITLES[plot_key]}**")
            if plot_key in plot_paths:
                st.image(str(plot_paths[plot_key]), width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
                continue

            if plot_key == "reliability_curve" and reliability_curve is not None:
                rendered_curve = reliability_curve.rename(
                    columns={
                        "bin": "Bin",
                        "bin_accuracy": "Observed Frequency",
                        "bin_confidence": "Predicted Confidence",
                    }
                )
                st.line_chart(
                    rendered_curve.set_index("Bin")[[
                        "Observed Frequency",
                        "Predicted Confidence",
                    ]],
                    height=260,
                )
                st.caption("Reliability curve generated from artifact-backed bin statistics.")
            else:
                if plot_key == "reliability_curve":
                    st.markdown(
                        (
                            "<div class='calibration-explainer-card'>"
                            "<strong>Calibration Reliability Overview</strong><br>"
                            f"{CALIBRATION_UNAVAILABLE_GUIDANCE}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    _render_artifact_unavailable_card(
                        title=PLOT_TITLES[plot_key],
                        guidance=(
                            "This artifact was not found for the selected model package. "
                            "Generate calibration analysis outputs to include this panel."
                        ),
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_saved_example_walkthrough(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    operating_mode: str,
    feature_names: list[str],
    walkthrough_payload_path: Path | None,
) -> None:
    if walkthrough_payload_path is None:
        _render_artifact_unavailable_card(
            title="Saved Example Walkthrough",
            guidance=SAVED_WALKTHROUGH_UNAVAILABLE_GUIDANCE,
        )
        return

    walkthrough_payload = build_saved_example_walkthrough_sample(
        manifest,
        walkthrough_payload_path,
    )
    request_sample = walkthrough_payload["request_sample"]

    dataset_section = manifest.get("dataset", {})
    try:
        predictions = predict_sequence_samples(
            manifest_path=manifest_path,
            dataset_tag=str(dataset_section.get("dataset_tag", "")),
            samples=[request_sample],
            operating_mode=operating_mode,
        )
    except Exception:
        _render_artifact_unavailable_card(
            title="Saved Example Walkthrough",
            guidance=SAVED_WALKTHROUGH_UNAVAILABLE_GUIDANCE,
        )
        return

    prediction = predictions[0]
    probability = float(prediction["predicted_probability"])
    threshold_used = float(prediction["threshold_used"])
    decision = "Alert" if int(prediction["predicted_label"]) == 1 else "No Alert"
    sample_id = str(walkthrough_payload["sample_id"])

    st.markdown("#### Saved Example Walkthrough")
    with st.container(border=True):
        st.markdown("<div class='inference-card'>", unsafe_allow_html=True)
        st.markdown(f"**Sample ID:** {sample_id} (synthetic)")
        st.markdown(f"**Risk Score:** {probability:.3f}")
        st.markdown(
            "**Selected Threshold:** "
            f"{threshold_used:.3f} ({format_threshold_mode(prediction['operating_mode'])})"
        )
        st.markdown(f"**Decision:** {decision}")
        st.caption(
            _risk_interpretation(
                probability=probability,
                threshold=threshold_used,
            )
        )
        st.caption(str(walkthrough_payload.get("walkthrough_note", "")))
        st.caption(
            _deterministic_explanation(
                sample=request_sample,
                feature_names=feature_names,
                predicted_probability=probability,
                threshold=threshold_used,
            )
        )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_live_inference_demo(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    public_mode: bool,
    parquet_path: Path,
    source_label: str,
    max_rows: int,
    operating_mode: str,
    selected_threshold: float,
    feature_names: list[str],
    walkthrough_payload_path: Path | None,
) -> None:
    try:
        split_frame = _load_split_samples(str(parquet_path), max_rows)
    except Exception:
        _render_artifact_unavailable_card(
            title="Live inference",
            guidance=(
                "Parquet-backed windows could not be loaded safely in this deployment package."
            ),
        )
        _render_saved_example_walkthrough(
            manifest=manifest,
            manifest_path=manifest_path,
            operating_mode=operating_mode,
            feature_names=feature_names,
            walkthrough_payload_path=walkthrough_payload_path,
        )
        return

    if split_frame.empty:
        _render_artifact_unavailable_card(
            title="Live inference",
            guidance=(
                "No candidate windows were found in the selected parquet source for this "
                "deployment build."
            ),
        )
        _render_saved_example_walkthrough(
            manifest=manifest,
            manifest_path=manifest_path,
            operating_mode=operating_mode,
            feature_names=feature_names,
            walkthrough_payload_path=walkthrough_payload_path,
        )
        return

    st.caption(f"Source: {source_label}. Loaded {len(split_frame)} windows.")
    st.caption(
        f"Current operating mode {format_threshold_mode(operating_mode)} applies threshold "
        f"{selected_threshold:.3f}."
    )

    preview = split_frame[["patient_id", "end_hour", "label"]].copy().reset_index(drop=True)
    preview["Sample"] = [f"S{index + 1:03d}" for index in range(len(preview))]
    preview["Sample ID"] = [
        f"DS-{index + 1:03d}" if public_mode else f"SM-{index + 1:03d}"
        for index in range(len(preview))
    ]
    preview = preview[["Sample", "Sample ID", "end_hour", "label"]].rename(
        columns={
            "end_hour": "End Hour",
            "label": "Observed Label",
        }
    )
    st.dataframe(preview, width="stretch", hide_index=True)

    option_to_index: dict[str, int] = {}
    for row_index, row in preview.iterrows():
        option = (
            f"{row['Sample']} | {row['Sample ID']} | "
            f"End Hour {row['End Hour']} | Label {row['Observed Label']}"
        )
        option_to_index[option] = int(row_index)

    options = list(option_to_index.keys())
    default_count = min(4, len(options))
    selected_options = st.multiselect(
        "Select sample windows for inference",
        options=options,
        default=options[:default_count],
        help=(
            "Choose one or more sample windows to run through the selected checkpoint and "
            "operating threshold mode."
        ),
    )

    if not selected_options:
        st.info("Select at least one sample window to run live inference.")
        return

    selected_indices = [option_to_index[item] for item in selected_options]
    selected_rows = split_frame.iloc[selected_indices].reset_index(drop=True)
    selected_preview = preview.iloc[selected_indices].reset_index(drop=True)
    request_samples = [_row_to_request_sample(row) for _, row in selected_rows.iterrows()]

    if not st.button("Run Inference", type="primary"):
        return

    dataset_section = manifest.get("dataset", {})
    try:
        predictions = predict_sequence_samples(
            manifest_path=manifest_path,
            dataset_tag=str(dataset_section.get("dataset_tag", "")),
            samples=request_samples,
            operating_mode=operating_mode,
        )
    except (SequenceServingError, Exception):
        _render_artifact_unavailable_card(
            title="Live inference",
            guidance=(
                "Live inference is unavailable for this deployment package. Showing a saved "
                "example walkthrough instead."
            ),
        )
        _render_saved_example_walkthrough(
            manifest=manifest,
            manifest_path=manifest_path,
            operating_mode=operating_mode,
            feature_names=feature_names,
            walkthrough_payload_path=walkthrough_payload_path,
        )
        return

    st.success("Inference completed successfully.")

    selected_preview_records = selected_preview.to_dict("records")
    result_frame = _build_inference_result_frame(
        preview_rows=selected_preview_records,
        predictions=predictions,
    )
    st.dataframe(result_frame, width="stretch", hide_index=True)
    alert_count = int((result_frame["Predicted Class"] == "Alert").sum())
    st.caption(
        f"Operational result: {alert_count} of {len(result_frame)} selected windows are "
        f"classified as Alert under threshold {selected_threshold:.3f}."
    )

    st.markdown("#### Score Explanations")
    for sample, preview_row, prediction in zip(
        request_samples,
        selected_preview_records,
        predictions,
        strict=True,
    ):
        with st.container(border=True):
            st.markdown("<div class='inference-card'>", unsafe_allow_html=True)
            st.markdown(
                f"**{preview_row['Sample']} | {preview_row['Sample ID']} | "
                f"End Hour {prediction['end_hour']}**"
            )
            st.write(
                _risk_interpretation(
                    probability=float(prediction["predicted_probability"]),
                    threshold=float(prediction["threshold_used"]),
                )
            )
            st.caption(
                _deterministic_explanation(
                    sample=sample,
                    feature_names=feature_names,
                    predicted_probability=float(prediction["predicted_probability"]),
                    threshold=float(prediction["threshold_used"]),
                )
            )
            st.markdown("</div>", unsafe_allow_html=True)


def _render_inference_demo(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    public_mode: bool,
    inference_source: DemoInferenceSource,
    max_rows: int,
    operating_mode: str,
    selected_threshold: float,
    feature_names: list[str],
) -> None:
    st.markdown("<div class='section-title'>Inference Demo</div>", unsafe_allow_html=True)
    st.caption(
        f"Current operating mode {format_threshold_mode(operating_mode)} applies threshold "
        f"{selected_threshold:.3f}."
    )

    if inference_source.source_kind == "walkthrough":
        st.info(
            "Live sequence-window artifacts are unavailable in this deployment, so this panel "
            "runs in Saved Example Walkthrough mode."
        )
        _render_saved_example_walkthrough(
            manifest=manifest,
            manifest_path=manifest_path,
            operating_mode=operating_mode,
            feature_names=feature_names,
            walkthrough_payload_path=inference_source.walkthrough_payload_path,
        )
        return

    if inference_source.source_kind == "unavailable" or inference_source.parquet_path is None:
        _render_artifact_unavailable_card(
            title="Inference demo",
            guidance=(
                f"{SAVED_WALKTHROUGH_UNAVAILABLE_GUIDANCE} "
                f"{inference_source.reason or ''}"
            ).strip(),
        )
        return

    _render_live_inference_demo(
        manifest=manifest,
        manifest_path=manifest_path,
        public_mode=public_mode,
        parquet_path=inference_source.parquet_path,
        source_label=inference_source.source_label,
        max_rows=max_rows,
        operating_mode=operating_mode,
        selected_threshold=selected_threshold,
        feature_names=feature_names,
        walkthrough_payload_path=inference_source.walkthrough_payload_path,
    )


def _render_explainability_section(
    *,
    feature_importance_frame: pd.DataFrame | None,
) -> None:
    st.markdown("<div class='section-title'>Model Explainability</div>", unsafe_allow_html=True)
    explainability_frame = feature_importance_frame.copy()
    st.caption("Top artifact-backed feature-importance signals from packaged analysis outputs.")
    st.dataframe(explainability_frame, width="stretch", hide_index=True)

    chart_frame = explainability_frame.set_index("Feature")
    st.bar_chart(chart_frame[["Importance"]], height=280)


def _render_credibility_section(
    *,
    manifest: dict[str, Any],
    threshold_modes: list[str],
    plot_paths: dict[str, Path],
    tests_status: str,
    tests_detail: str,
) -> None:
    st.markdown(
        "<div class='section-title'>Project Credibility and Evidence</div>",
        unsafe_allow_html=True,
    )

    threshold_metadata = manifest.get("threshold_metadata", {})
    calibration_synced = bool(
        isinstance(threshold_metadata, dict)
        and threshold_metadata.get("source") == "calibration_recommendations"
    )
    serving_ready = set(threshold_modes) == {"default", "balanced", "high_recall"}

    cards: list[tuple[str, str, str]] = [
        ("Selected Model", "Ready", "Selected manifest and checkpoint validation passed."),
        (
            "Calibration Sync",
            "Synced" if calibration_synced else "Not synced",
            "Threshold metadata source confirms calibration synchronization."
            if calibration_synced
            else "Threshold metadata does not confirm calibration synchronization.",
        ),
        (
            "Serving Compatibility",
            "Ready" if serving_ready else "Check",
            "Threshold modes available for serving endpoints."
            if serving_ready
            else "One or more threshold modes are missing from selected manifest.",
        ),
        (
            "Evaluation Artifacts",
            f"{len(plot_paths)} available",
            "Plot artifacts discovered from calibration outputs.",
        ),
    ]
    if tests_status != "Not available":
        cards.append(("Automated Tests", tests_status, tests_detail))

    columns = st.columns(len(cards))
    for column, (title, value, subtitle) in zip(columns, cards, strict=True):
        with column:
            _render_card(title=title, value=value, subtitle=subtitle)


def _render_footer() -> None:
    st.markdown(
        "<div class='footer-note'>"
        "Research-use demonstration only. Not a medical device. "
        "Inference uses de-identified, artifact-backed sequence windows and "
        "selected model artifacts."
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Early Sepsis Detection", page_icon="ES", layout="wide")
    _apply_theme()
    public_artifacts_root = resolve_runtime_path(settings.public_artifacts_dir)

    deployment_public_mode = settings.environment.strip().lower() != "development"

    with st.sidebar:
        st.markdown("### Demo Controls")
        public_mode = True if deployment_public_mode else bool(settings.demo_public_mode)
        if public_mode:
            st.caption("Public-safe dataset mode is active.")
        split = "validation"
        if not public_mode:
            split = st.selectbox(
                "Evaluation split",
                options=["validation", "test"],
                index=0,
                help=(
                    "Select which packaged evaluation split to inspect for operational summary "
                    "and live inference windows."
                ),
            )

        max_rows = st.slider(
            "Rows to load",
            min_value=20,
            max_value=1000,
            value=200,
            step=20,
            help=(
                "Controls how many candidate windows are loaded from packaged parquet artifacts. "
                "Higher values increase compute time."
            ),
        )

        manifest_path = resolve_manifest_path(
            settings.selected_sequence_manifest_path,
            public_artifacts_dir=public_artifacts_root,
        )

    try:
        startup_status = validate_demo_startup(manifest_path)
    except DemoStartupError as exc:
        st.error("Startup validation failed for selected model artifacts.")
        st.caption(sanitize_public_text(str(exc)))
        st.stop()

    manifest = startup_status.manifest
    dataset_section = manifest["dataset"]
    model_section = manifest["model"]
    feature_names = [str(item) for item in dataset_section.get("feature_columns", [])]

    calibration_summary, calibration_summary_path = resolve_calibration_summary(
        manifest,
        manifest_path=startup_status.manifest_path,
        public_artifacts_root=public_artifacts_root,
    )
    metric_snapshot, metric_source = collect_metric_snapshot(
        manifest,
        calibration_summary=calibration_summary,
    )
    plot_paths = collect_plot_artifacts(
        calibration_summary=calibration_summary,
        manifest_path=startup_status.manifest_path,
        public_artifacts_root=public_artifacts_root,
    )
    reliability_curve = load_reliability_curve(
        calibration_summary_path=calibration_summary_path,
        manifest_path=startup_status.manifest_path,
        public_artifacts_root=public_artifacts_root,
    )
    comparison_frame = load_experiment_comparison(
        limit=5,
        public_artifacts_root=public_artifacts_root,
    )
    feature_importance_frame = load_feature_importance_artifact(
        manifest_path=startup_status.manifest_path,
        public_artifacts_root=public_artifacts_root,
        limit=10,
    )

    inference_source = resolve_demo_inference_source(
        manifest,
        manifest_path=startup_status.manifest_path,
        split=split,
        public_mode=public_mode,
        bundled_demo_path=Path("assets/demo/sequence_demo_samples.parquet"),
        walkthrough_payload_path=Path("assets/demo/saved_example_payload.json"),
    )

    threshold_modes = [
        mode for mode in ("default", "balanced", "high_recall") if mode in manifest["thresholds"]
    ]
    if not threshold_modes:
        st.error("Selected model manifest does not include threshold modes.")
        st.stop()

    threshold_metadata = manifest.get("threshold_metadata", {})
    calibration_synced = bool(
        isinstance(threshold_metadata, dict)
        and threshold_metadata.get("source") == "calibration_recommendations"
    )
    tests_status, tests_detail = detect_latest_pytest_status(project_root=_project_root())

    selected_mode = settings.serving_default_operating_mode
    if selected_mode not in threshold_modes:
        selected_mode = threshold_modes[0]

    repo_url = _resolve_repo_url(
        settings_repo_url=getattr(settings, "public_repo_url", None),
        project_root=_project_root(),
    )

    _render_hero_section(
        model_family=str(model_section.get("model_family", "unknown")),
        dataset_tag=str(dataset_section.get("dataset_tag", "unknown")),
        window_length=int(model_section.get("window_length", 0)),
        feature_count=len(feature_names),
        status="Ready",
        repo_url=repo_url,
    )

    _render_why_this_matters()

    _render_project_status_strip(
        calibration_synced=calibration_synced,
        tests_status=tests_status,
        threshold_modes=threshold_modes,
        plot_count=len(plot_paths),
    )

    _render_system_status(
        manifest=manifest,
        threshold_modes=threshold_modes,
        calibration_available=calibration_summary is not None,
    )

    _render_performance_summary(
        metric_snapshot=metric_snapshot,
        metric_source=metric_source,
        calibration_summary=calibration_summary,
        comparison_frame=comparison_frame,
    )

    try:
        selected_mode, selected_threshold, duplicate_thresholds = _render_threshold_strategy(
            manifest=manifest
        )
    except SequenceServingError as exc:
        st.error(sanitize_public_text(str(exc)))
        st.stop()

    operational_source_label = "Deployment operational subset"
    operational_unavailable_reason: str | None = None
    operational_frame = pd.DataFrame(columns=["Sample", "End Hour", "Observed Label", "Risk Score"])

    (
        operational_source_label,
        operational_source_path,
        operational_unavailable_reason,
    ) = _resolve_operational_source(
        public_artifacts_root=public_artifacts_root,
        manifest_path=startup_status.manifest_path,
        dataset_section=dataset_section,
        split=split,
        inference_source=inference_source,
    )

    if operational_source_path is not None:
        try:
            operational_frame = _load_operational_probability_frame(
                manifest_path=str(startup_status.manifest_path),
                manifest_mtime=startup_status.manifest_path.stat().st_mtime,
                dataset_tag=str(dataset_section.get("dataset_tag", "")),
                parquet_path=str(operational_source_path),
                max_rows=max_rows,
            )
            operational_unavailable_reason = None
        except Exception:
            operational_unavailable_reason = (
                "A packaged operational source was detected but could not be processed in this "
                "deployment environment."
            )

    _render_operational_summary(
        operational_frame=operational_frame,
        operating_mode=selected_mode,
        applied_threshold=selected_threshold,
        source_label=operational_source_label,
        duplicate_thresholds=duplicate_thresholds,
        unavailable_reason=operational_unavailable_reason,
    )

    _render_evaluation_visuals(
        plot_paths=plot_paths,
        reliability_curve=reliability_curve,
    )

    if feature_importance_frame is not None and not feature_importance_frame.empty:
        _render_explainability_section(
            feature_importance_frame=feature_importance_frame,
        )

    _render_inference_demo(
        manifest=manifest,
        manifest_path=startup_status.manifest_path,
        public_mode=public_mode,
        inference_source=inference_source,
        max_rows=max_rows,
        operating_mode=selected_mode,
        selected_threshold=selected_threshold,
        feature_names=feature_names,
    )

    _render_credibility_section(
        manifest=manifest,
        threshold_modes=threshold_modes,
        plot_paths=plot_paths,
        tests_status=tests_status,
        tests_detail=tests_detail,
    )

    _render_footer()


if __name__ == "__main__":
    main()
