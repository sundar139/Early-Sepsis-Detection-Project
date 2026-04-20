from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from early_sepsis.data.schema import PATIENT_ID_COLUMN


@dataclass(slots=True)
class PatientSplitResult:
    """Patient-level split assignments and convenience accessors."""

    assignments: pd.DataFrame

    @property
    def train_patients(self) -> list[str]:
        return self.assignments.loc[self.assignments["split"] == "train", PATIENT_ID_COLUMN].tolist()

    @property
    def validation_patients(self) -> list[str]:
        return self.assignments.loc[
            self.assignments["split"] == "validation", PATIENT_ID_COLUMN
        ].tolist()

    @property
    def test_patients(self) -> list[str]:
        return self.assignments.loc[self.assignments["split"] == "test", PATIENT_ID_COLUMN].tolist()


def _normalize_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        msg = "Split ratios must sum to a positive value."
        raise ValueError(msg)

    return train_ratio / total, validation_ratio / total, test_ratio / total


def _allocate_counts(
    patient_count: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[int, int, int]:
    if patient_count < 3:
        msg = "At least three patients are required for train/validation/test splitting."
        raise ValueError(msg)

    train_count = int(np.floor(patient_count * train_ratio))
    validation_count = int(np.floor(patient_count * validation_ratio))

    train_count = max(1, train_count)
    validation_count = max(1, validation_count)
    test_count = patient_count - train_count - validation_count

    if test_count <= 0:
        test_count = 1
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1

    if train_count + validation_count + test_count != patient_count:
        test_count = patient_count - train_count - validation_count

    if min(train_count, validation_count, test_count) <= 0:
        msg = "Unable to allocate non-empty train/validation/test patient groups."
        raise ValueError(msg)

    return train_count, validation_count, test_count


def split_patients(
    dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
    patient_column: str = PATIENT_ID_COLUMN,
) -> PatientSplitResult:
    """Creates deterministic patient-level split assignments."""

    if patient_column not in dataframe.columns:
        msg = f"Patient column '{patient_column}' is missing from input data."
        raise KeyError(msg)

    patient_ids = sorted(dataframe[patient_column].astype(str).unique().tolist())
    if not patient_ids:
        msg = "No patients were found in the input dataframe."
        raise ValueError(msg)

    normalized_train, normalized_validation, _ = _normalize_ratios(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    train_count, validation_count, _ = _allocate_counts(
        patient_count=len(patient_ids),
        train_ratio=normalized_train,
        validation_ratio=normalized_validation,
    )

    rng = np.random.default_rng(seed=random_seed)
    shuffled_patients = rng.permutation(patient_ids)

    train_patients = shuffled_patients[:train_count]
    validation_patients = shuffled_patients[train_count : train_count + validation_count]
    test_patients = shuffled_patients[train_count + validation_count :]

    assignment_rows: list[dict[str, str]] = []
    assignment_rows.extend(
        {patient_column: str(patient_id), "split": "train"} for patient_id in train_patients
    )
    assignment_rows.extend(
        {patient_column: str(patient_id), "split": "validation"}
        for patient_id in validation_patients
    )
    assignment_rows.extend(
        {patient_column: str(patient_id), "split": "test"} for patient_id in test_patients
    )

    assignments = pd.DataFrame(assignment_rows).sort_values(["split", patient_column])
    assignments = assignments.reset_index(drop=True)
    return PatientSplitResult(assignments=assignments)


def apply_split_assignments(
    dataframe: pd.DataFrame,
    split_result: PatientSplitResult,
    patient_column: str = PATIENT_ID_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Applies patient assignments to produce split-specific dataframes."""

    merged = dataframe.merge(split_result.assignments, on=patient_column, how="inner")
    split_frames = {
        split_name: split_frame.drop(columns=["split"]).reset_index(drop=True)
        for split_name, split_frame in merged.groupby("split", sort=False)
    }

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(split_frames)
    if missing_splits:
        msg = f"Missing split outputs: {sorted(missing_splits)}"
        raise ValueError(msg)

    return split_frames


def save_split_manifests(
    split_result: PatientSplitResult,
    output_dir: str | Path,
    patient_column: str = PATIENT_ID_COLUMN,
) -> dict[str, Path]:
    """Writes split manifests to disk and returns created paths."""

    manifest_dir = Path(output_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    assignments_path = manifest_dir / "patient_split_assignments.csv"
    split_result.assignments.to_csv(assignments_path, index=False)
    paths["assignments"] = assignments_path

    for split_name in ("train", "validation", "test"):
        subset = split_result.assignments.loc[
            split_result.assignments["split"] == split_name, [patient_column]
        ]
        split_path = manifest_dir / f"{split_name}_patients.csv"
        subset.to_csv(split_path, index=False)
        paths[split_name] = split_path

    return paths
