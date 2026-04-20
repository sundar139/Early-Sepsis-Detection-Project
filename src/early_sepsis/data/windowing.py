from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from early_sepsis.data.schema import PATIENT_ID_COLUMN, TARGET_COLUMN, TIME_COLUMN


@dataclass(slots=True)
class WindowConfig:
    """Configuration for sliding-window label generation."""

    window_length: int
    prediction_horizon: int
    padding_mode: bool = False

    def validate(self) -> None:
        if self.window_length <= 0:
            msg = "window_length must be greater than zero"
            raise ValueError(msg)
        if self.prediction_horizon <= 0:
            msg = "prediction_horizon must be greater than zero"
            raise ValueError(msg)


@dataclass(slots=True)
class WindowGenerationResult:
    """Generated windows and aggregate counts."""

    windows: pd.DataFrame
    positive_labels: int
    total_windows: int


def _to_window_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected_length: int,
    padding_mode: bool,
) -> np.ndarray | None:
    values = frame.loc[:, columns].to_numpy(dtype=np.float32)

    if len(values) == expected_length:
        return values
    if len(values) > expected_length:
        return values[-expected_length:]
    if not padding_mode:
        return None

    pad_length = expected_length - len(values)
    padding = np.zeros((pad_length, len(columns)), dtype=np.float32)
    return np.vstack([padding, values])


def generate_sliding_windows(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    mask_columns: Sequence[str] | None,
    static_columns: Sequence[str] | None,
    config: WindowConfig,
    patient_column: str = PATIENT_ID_COLUMN,
    time_column: str = TIME_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> WindowGenerationResult:
    """Generates per-patient sliding windows with horizon-based binary labels."""

    config.validate()
    missing_columns = [
        column
        for column in [patient_column, time_column, target_column, *feature_columns]
        if column not in dataframe.columns
    ]
    if missing_columns:
        msg = f"Missing required columns for windowing: {missing_columns}"
        raise KeyError(msg)

    resolved_mask_columns = list(mask_columns or [])
    resolved_static_columns = list(static_columns or [])

    for column in resolved_mask_columns + resolved_static_columns:
        if column not in dataframe.columns:
            msg = f"Column '{column}' requested for window output but not present in dataframe"
            raise KeyError(msg)

    records: list[dict[str, Any]] = []

    grouped = dataframe.sort_values([patient_column, time_column]).groupby(patient_column, sort=False)
    for patient_id, patient_frame in grouped:
        patient_frame = patient_frame.reset_index(drop=True)
        labels = patient_frame[target_column].to_numpy(dtype=np.int64)
        hours = patient_frame[time_column].to_numpy(dtype=np.float32)

        positive_indices = np.flatnonzero(labels == 1)
        onset_index = int(positive_indices[0]) if len(positive_indices) > 0 else None
        onset_hour = float(hours[onset_index]) if onset_index is not None else None

        for end_index in range(len(patient_frame)):
            if onset_index is not None and end_index >= onset_index:
                break

            start_index = max(0, end_index - config.window_length + 1)
            history_frame = patient_frame.iloc[start_index : end_index + 1]

            features_matrix = _to_window_matrix(
                frame=history_frame,
                columns=feature_columns,
                expected_length=config.window_length,
                padding_mode=config.padding_mode,
            )
            if features_matrix is None:
                continue

            mask_matrix: np.ndarray | None = None
            if resolved_mask_columns:
                mask_matrix = _to_window_matrix(
                    frame=history_frame,
                    columns=resolved_mask_columns,
                    expected_length=config.window_length,
                    padding_mode=config.padding_mode,
                )
                if mask_matrix is None:
                    continue

            current_hour = float(hours[end_index])
            label = 0
            if onset_hour is not None:
                hours_to_onset = onset_hour - current_hour
                if 0 < hours_to_onset <= float(config.prediction_horizon):
                    label = 1

            static_values: list[float] | None = None
            if resolved_static_columns:
                static_values = (
                    patient_frame.loc[0, resolved_static_columns]
                    .to_numpy(dtype=np.float32)
                    .tolist()
                )

            record = {
                patient_column: str(patient_id),
                "end_hour": current_hour,
                "label": int(label),
                "features": features_matrix.tolist(),
            }
            if mask_matrix is not None:
                record["missing_mask"] = mask_matrix.tolist()
            if static_values is not None:
                record["static_features"] = static_values

            records.append(record)

    windows = pd.DataFrame.from_records(records)
    positive_labels = int(windows["label"].sum()) if not windows.empty else 0
    return WindowGenerationResult(
        windows=windows,
        positive_labels=positive_labels,
        total_windows=len(windows),
    )


def summarize_windows(windows_by_split: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Returns simple per-split window summary statistics."""

    summary: dict[str, dict[str, int]] = {}
    for split_name, frame in windows_by_split.items():
        positive_labels = int(frame["label"].sum()) if "label" in frame.columns else 0
        summary[split_name] = {
            "window_count": int(len(frame)),
            "positive_label_count": positive_labels,
        }
    return summary
