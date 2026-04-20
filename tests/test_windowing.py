from __future__ import annotations

import pandas as pd

from early_sepsis.data.schema import PATIENT_ID_COLUMN, TARGET_COLUMN, TIME_COLUMN
from early_sepsis.data.windowing import WindowConfig, generate_sliding_windows


def _single_patient_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for hour in range(11):
        rows.append(
            {
                PATIENT_ID_COLUMN: "patient_a",
                TIME_COLUMN: hour,
                "HR": float(70 + hour),
                "HR__missing": 0,
                TARGET_COLUMN: int(hour >= 8),
            }
        )
    return pd.DataFrame(rows)


def test_label_generation_near_onset_excludes_post_onset_windows() -> None:
    frame = _single_patient_frame()
    result = generate_sliding_windows(
        dataframe=frame,
        feature_columns=["HR"],
        mask_columns=["HR__missing"],
        static_columns=None,
        config=WindowConfig(window_length=4, prediction_horizon=2, padding_mode=False),
    )

    windows = result.windows
    assert len(windows) == 5
    assert all(hour < 8 for hour in windows["end_hour"].tolist())

    positive_hours = set(windows.loc[windows["label"] == 1, "end_hour"].tolist())
    assert positive_hours == {6.0, 7.0}


def test_padding_mode_allows_incomplete_early_history() -> None:
    frame = _single_patient_frame()
    result = generate_sliding_windows(
        dataframe=frame,
        feature_columns=["HR"],
        mask_columns=None,
        static_columns=None,
        config=WindowConfig(window_length=4, prediction_horizon=2, padding_mode=True),
    )

    windows = result.windows
    assert len(windows) == 8
    first_window = windows.iloc[0]["features"]
    assert len(first_window) == 4
