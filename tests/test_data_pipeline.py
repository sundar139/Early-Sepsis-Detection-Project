from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from early_sepsis.data.pipeline import run_preprocessing_pipeline
from early_sepsis.data.preprocessing import preprocess_time_series_splits
from early_sepsis.data.schema import PATIENT_ID_COLUMN, TARGET_COLUMN, TIME_COLUMN
from early_sepsis.data.synthetic import generate_synthetic_icu_dataset


def test_patient_split_has_no_leakage(tmp_path: Path) -> None:
    raw_csv = tmp_path / "synthetic_raw.csv"
    generate_synthetic_icu_dataset(
        output_path=raw_csv,
        dataset_format="csv",
        patient_count=24,
        random_seed=11,
    )

    result = run_preprocessing_pipeline(
        raw_data_path=raw_csv,
        output_dir=tmp_path / "processed",
        dataset_format="csv",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=77,
    )

    train_patients = set(pd.read_csv(result.manifest_paths["train"])[PATIENT_ID_COLUMN].tolist())
    validation_patients = set(
        pd.read_csv(result.manifest_paths["validation"])[PATIENT_ID_COLUMN].tolist()
    )
    test_patients = set(pd.read_csv(result.manifest_paths["test"])[PATIENT_ID_COLUMN].tolist())

    assert train_patients
    assert validation_patients
    assert test_patients

    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)


def test_imputation_and_standardization_use_train_statistics_only() -> None:
    train_frame = pd.DataFrame(
        {
            PATIENT_ID_COLUMN: ["p1", "p1", "p2", "p2"],
            TIME_COLUMN: [0, 1, 0, 1],
            "HR": [1.0, np.nan, 3.0, 5.0],
            TARGET_COLUMN: [0, 0, 0, 0],
        }
    )
    validation_frame = pd.DataFrame(
        {
            PATIENT_ID_COLUMN: ["p3", "p3"],
            TIME_COLUMN: [0, 1],
            "HR": [np.nan, 7.0],
            TARGET_COLUMN: [0, 0],
        }
    )
    test_frame = pd.DataFrame(
        {
            PATIENT_ID_COLUMN: ["p4"],
            TIME_COLUMN: [0],
            "HR": [np.nan],
            TARGET_COLUMN: [0],
        }
    )

    processed = preprocess_time_series_splits(
        split_frames={
            "train": train_frame,
            "validation": validation_frame,
            "test": test_frame,
        },
        feature_columns=["HR"],
    )

    stats = processed.statistics
    assert np.isclose(stats.medians["HR"], 2.0)

    expected_train_mean = 2.5
    expected_train_std = np.std(np.array([1.0, 1.0, 3.0, 5.0], dtype=np.float32), ddof=0)
    assert np.isclose(stats.means["HR"], expected_train_mean)
    assert np.isclose(stats.stds["HR"], expected_train_std)

    test_value = float(processed.split_frames["test"].iloc[0]["HR"])
    expected_test_value = (2.0 - expected_train_mean) / expected_train_std
    assert np.isclose(test_value, expected_test_value)
