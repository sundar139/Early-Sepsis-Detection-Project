from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from early_sepsis.modeling.model_manifest import load_model_manifest
from early_sepsis.serving.sequence_service import (
    SequenceServingError,
    predict_sequence_samples,
    resolve_operating_threshold,
)
from early_sepsis.settings import get_settings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path, *, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@st.cache_data(show_spinner=False)
def _load_split_samples(parquet_path: str, max_rows: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        parquet_path,
        columns=["patient_id", "end_hour", "label", "features", "missing_mask", "static_features"],
    )
    return frame.head(max_rows).copy()


def _row_to_request_sample(row: pd.Series) -> dict[str, Any]:
    return {
        "patient_id": row.get("patient_id"),
        "end_hour": int(row.get("end_hour", 0)),
        "features": np.asarray(row["features"]).tolist(),
        "missing_mask": np.asarray(row["missing_mask"]).tolist()
        if row.get("missing_mask") is not None
        else None,
        "static_features": np.asarray(row["static_features"]).tolist()
        if row.get("static_features") is not None
        else None,
    }


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


def resolve_demo_threshold_for_mode(manifest: dict[str, Any], operating_mode: str) -> float:
    """Resolves display and inference threshold for the selected demo operating mode."""

    resolved_threshold, _ = resolve_operating_threshold(
        manifest,
        operating_mode=operating_mode,
        threshold_override=None,
    )
    return resolved_threshold


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Early Sepsis Demo", page_icon="+", layout="wide")
    st.title("Early Sepsis Selected Sequence Model Demo")
    st.caption("Runs local inference using the selected manifest-backed sequence checkpoint.")

    st.warning(
        (
            "Research use only. This demo is not a medical device and must not be used "
            "for clinical diagnosis or treatment decisions."
        ),
    )

    project_root = _project_root()
    default_manifest = _resolve_path(settings.selected_sequence_manifest_path, base=project_root)

    manifest_path = Path(
        st.text_input(
            "Selected model manifest path",
            value=str(default_manifest),
        )
    )

    if not manifest_path.exists():
        st.error(
            "Selected model manifest is missing. Run model selection first with "
            "scripts/select_best_model.py."
        )
        st.stop()

    try:
        manifest = load_model_manifest(manifest_path)
    except Exception as exc:
        st.error(f"Failed to load selected manifest: {exc}")
        st.stop()

    dataset_section = manifest["dataset"]
    model_section = manifest["model"]
    feature_names = [str(item) for item in dataset_section.get("feature_columns", [])]

    with st.expander("Selected model info", expanded=True):
        st.json(
            {
                "selected_run": manifest["selected_run"],
                "dataset": {
                    "dataset_tag": dataset_section["dataset_tag"],
                    "windows_dir": dataset_section["windows_dir"],
                    "feature_count": len(feature_names),
                },
                "model": model_section,
                "thresholds": manifest["thresholds"],
            }
        )

    split = st.selectbox("Window split", options=["validation", "test"], index=0)
    default_windows_path = Path(dataset_section["windows_dir"]) / f"{split}.parquet"
    parquet_path = st.text_input("Split parquet path", value=str(default_windows_path))
    max_rows = st.slider("Rows to load", min_value=20, max_value=1000, value=200, step=20)

    try:
        split_frame = _load_split_samples(parquet_path, max_rows)
    except Exception as exc:
        st.error(f"Unable to load split windows: {exc}")
        st.stop()

    st.write(f"Loaded {len(split_frame)} candidate windows from {parquet_path}")
    preview_columns = ["patient_id", "end_hour", "label"]
    st.dataframe(split_frame[preview_columns], width="stretch", hide_index=True)

    default_count = min(4, len(split_frame))
    selected_indices = st.multiselect(
        "Select window row indices for inference",
        options=list(range(len(split_frame))),
        default=list(range(default_count)),
    )

    operating_mode = st.selectbox(
        "Operating mode",
        options=["default", "balanced", "high_recall"],
        index=0,
    )
    try:
        preview_threshold = resolve_demo_threshold_for_mode(manifest, operating_mode)
    except SequenceServingError as exc:
        st.error(f"Invalid operating mode configuration: {exc}")
        st.stop()

    st.caption(
        f"Selected mode '{operating_mode}' applies threshold {preview_threshold:.3f} for predictions."
    )

    if st.button("Run Selected Model Inference"):
        if not selected_indices:
            st.error("Select at least one window row index.")
            return

        selected_rows = split_frame.iloc[selected_indices]
        request_samples = [_row_to_request_sample(row) for _, row in selected_rows.iterrows()]

        try:
            predictions = predict_sequence_samples(
                manifest_path=manifest_path,
                dataset_tag=str(dataset_section["dataset_tag"]),
                samples=request_samples,
                operating_mode=operating_mode,
            )
        except SequenceServingError as exc:
            st.error(f"Sequence validation failed: {exc}")
            return
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

        st.success("Inference completed")
        result_frame = pd.DataFrame(predictions)
        st.dataframe(result_frame, width="stretch", hide_index=True)

        st.subheader("Deterministic explanation")
        for sample, prediction in zip(request_samples, predictions, strict=True):
            explanation = _deterministic_explanation(
                sample=sample,
                feature_names=feature_names,
                predicted_probability=float(prediction["predicted_probability"]),
                threshold=float(prediction["threshold_used"]),
            )
            summary = (
                f"- Patient {prediction['patient_id']}, "
                f"end_hour={prediction['end_hour']}: {explanation}"
            )
            st.markdown(summary)

        payload_preview = {
            "dataset_tag": dataset_section["dataset_tag"],
            "operating_mode": operating_mode,
            "samples": request_samples,
        }
        st.subheader("Request payload preview")
        st.code(json.dumps(payload_preview, indent=2), language="json")


if __name__ == "__main__":
    main()
