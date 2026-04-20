from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from early_sepsis.data.schema import PATIENT_ID_COLUMN, TARGET_COLUMN, TIME_COLUMN


@dataclass(slots=True)
class SyntheticDataResult:
    """Result metadata for generated synthetic ICU datasets."""

    output_path: Path
    dataset_format: str
    patient_count: int
    row_count: int


def _inject_missingness(
    dataframe: pd.DataFrame,
    columns: list[str],
    missing_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = dataframe.copy()
    for column in columns:
        mask = rng.random(len(result)) < missing_rate
        result.loc[mask, column] = np.nan
    return result


def generate_synthetic_icu_dataset(
    output_path: str | Path,
    dataset_format: str,
    patient_count: int = 18,
    min_hours: int = 10,
    max_hours: int = 20,
    sepsis_prevalence: float = 0.35,
    missing_rate: float = 0.1,
    random_seed: int = 42,
) -> SyntheticDataResult:
    """Generates deterministic synthetic ICU trajectories for local testing."""

    if dataset_format not in {"csv", "physionet"}:
        msg = "dataset_format must be either 'csv' or 'physionet'"
        raise ValueError(msg)
    if patient_count < 3:
        msg = "patient_count must be at least 3"
        raise ValueError(msg)

    rng = np.random.default_rng(seed=random_seed)
    rows: list[dict[str, float | int | str]] = []

    for index in range(patient_count):
        patient_id = f"patient_{index:04d}"
        hours = int(rng.integers(min_hours, max_hours + 1))

        age = int(rng.integers(18, 90))
        gender = int(rng.integers(0, 2))
        positive_case = bool(rng.random() < sepsis_prevalence)
        onset_hour = int(rng.integers(6, hours)) if positive_case and hours > 7 else None

        for hour in range(hours):
            risk_boost = 0.0
            if onset_hour is not None and hour >= max(0, onset_hour - 3):
                risk_boost = min(1.0, (hour - (onset_hour - 3)) / 3.0)

            row = {
                PATIENT_ID_COLUMN: patient_id,
                TIME_COLUMN: hour,
                "HR": float(rng.normal(82 + (14 * risk_boost), 8)),
                "O2Sat": float(rng.normal(97 - (4 * risk_boost), 1.5)),
                "Temp": float(rng.normal(36.9 + (1.1 * risk_boost), 0.25)),
                "MAP": float(rng.normal(82 - (12 * risk_boost), 7)),
                "Resp": float(rng.normal(17 + (8 * risk_boost), 3)),
                "Age": age,
                "Gender": gender,
                TARGET_COLUMN: int(onset_hour is not None and hour >= onset_hour),
            }
            rows.append(row)

    dataframe = pd.DataFrame(rows)
    dataframe = _inject_missingness(
        dataframe=dataframe,
        columns=["HR", "O2Sat", "Temp", "MAP", "Resp"],
        missing_rate=missing_rate,
        rng=rng,
    )

    output = Path(output_path)
    if dataset_format == "csv":
        output.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output, index=False)
    else:
        output.mkdir(parents=True, exist_ok=True)
        for patient_id, patient_frame in dataframe.groupby(PATIENT_ID_COLUMN, sort=False):
            patient_output = output / f"{patient_id}.psv"
            patient_frame.drop(columns=[PATIENT_ID_COLUMN]).to_csv(
                patient_output,
                index=False,
                sep="|",
            )

    return SyntheticDataResult(
        output_path=output,
        dataset_format=dataset_format,
        patient_count=patient_count,
        row_count=len(dataframe),
    )
