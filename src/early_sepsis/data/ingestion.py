from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from early_sepsis.data.schema import (
    PATIENT_ID_ALIASES,
    PATIENT_ID_COLUMN,
    ROW_SOURCE_PATIENT_ALIASES,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    TARGET_ALIASES,
    TARGET_COLUMN,
    TIME_ALIASES,
    TIME_COLUMN,
    SchemaValidationIssue,
    first_matching_column,
    infer_feature_columns,
)
from early_sepsis.logging_utils import get_logger

logger = get_logger(__name__)

DatasetFormat = Literal["auto", "csv", "physionet"]


@dataclass(slots=True)
class IngestionResult:
    """Result payload for schema-aware data ingestion."""

    dataframe: pd.DataFrame
    dataset_format: Literal["csv", "physionet"]
    feature_columns: list[str]
    issues: list[SchemaValidationIssue]
    file_count: int
    skipped_file_count: int


def _find_files(data_path: Path, pattern: str) -> list[Path]:
    if data_path.is_file():
        return [data_path] if data_path.match(pattern) else []
    return sorted(path for path in data_path.rglob(pattern) if path.is_file())


def detect_dataset_format(data_path: Path) -> Literal["csv", "physionet"]:
    """Detects whether input data is CSV or PhysioNet PSV format."""

    if data_path.is_file():
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix == ".psv":
            return "physionet"

    csv_files = _find_files(data_path, "*.csv")
    physionet_files = _find_files(data_path, "*.psv")

    if physionet_files:
        return "physionet"
    if csv_files:
        return "csv"

    msg = (
        f"Unable to infer dataset format for path: {data_path}. "
        "Expected CSV files or PhysioNet PSV files."
    )
    raise ValueError(msg)


def _read_file(file_path: Path, dataset_format: Literal["csv", "physionet"]) -> pd.DataFrame:
    if dataset_format == "physionet":
        return pd.read_csv(file_path, sep="|")
    return pd.read_csv(file_path)


def _coerce_numeric_features(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    coerced = dataframe.copy()
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(coerced[column]):
            continue
        converted = pd.to_numeric(coerced[column], errors="coerce")
        if converted.notna().any() or coerced[column].isna().all():
            coerced[column] = converted
    return coerced


def _normalize_file_schema(raw_frame: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    if raw_frame.empty:
        msg = "file is empty"
        raise ValueError(msg)

    frame = raw_frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    patient_column = first_matching_column(frame.columns, PATIENT_ID_ALIASES)
    if patient_column is not None and patient_column != PATIENT_ID_COLUMN:
        frame = frame.rename(columns={patient_column: PATIENT_ID_COLUMN})

    time_column = first_matching_column(frame.columns, TIME_ALIASES)
    if time_column is not None and time_column != TIME_COLUMN:
        frame = frame.rename(columns={time_column: TIME_COLUMN})

    target_column = first_matching_column(frame.columns, TARGET_ALIASES)
    if target_column is None:
        msg = f"missing required target column ({TARGET_COLUMN})"
        raise ValueError(msg)
    if target_column != TARGET_COLUMN:
        frame = frame.rename(columns={target_column: TARGET_COLUMN})

    if PATIENT_ID_COLUMN not in frame.columns:
        row_source_column = first_matching_column(frame.columns, ROW_SOURCE_PATIENT_ALIASES)
        if row_source_column is not None:
            frame[PATIENT_ID_COLUMN] = frame[row_source_column]
        else:
            frame[PATIENT_ID_COLUMN] = file_path.stem

    if TIME_COLUMN not in frame.columns:
        logger.warning(
            "No time column found. Falling back to row order as hourly index.",
            extra={"file_path": str(file_path)},
        )
        frame[TIME_COLUMN] = np.arange(len(frame), dtype=np.int64)

    frame[SOURCE_FILE_COLUMN] = file_path.name
    frame[SOURCE_ROW_COLUMN] = np.arange(len(frame), dtype=np.int64)

    frame[PATIENT_ID_COLUMN] = frame[PATIENT_ID_COLUMN].fillna(file_path.stem).astype(str).str.strip()
    frame.loc[frame[PATIENT_ID_COLUMN] == "", PATIENT_ID_COLUMN] = file_path.stem

    frame[TIME_COLUMN] = pd.to_numeric(frame[TIME_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    invalid_row_mask = frame[TIME_COLUMN].isna() | frame[TARGET_COLUMN].isna()
    invalid_row_count = int(invalid_row_mask.sum())
    if invalid_row_count > 0:
        logger.warning(
            "Dropping rows with invalid time or target values",
            extra={"file_path": str(file_path), "dropped_rows": invalid_row_count},
        )
        frame = frame.loc[~invalid_row_mask].copy()

    if frame.empty:
        msg = "all rows became invalid after time/target validation"
        raise ValueError(msg)

    frame[TIME_COLUMN] = frame[TIME_COLUMN].astype(np.int64)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(np.int64)

    unique_targets = set(frame[TARGET_COLUMN].unique().tolist())
    invalid_targets = unique_targets - {0, 1}
    if invalid_targets:
        msg = f"target column must be binary 0/1, found {sorted(invalid_targets)}"
        raise ValueError(msg)

    feature_columns = infer_feature_columns(frame.columns)
    if not feature_columns:
        msg = "no usable feature columns were found after schema normalization"
        raise ValueError(msg)

    return _coerce_numeric_features(frame, feature_columns)


def ingest_raw_dataset(
    data_path: str | Path,
    dataset_format: DatasetFormat = "auto",
    strict_validation: bool = False,
) -> IngestionResult:
    """Ingests PhysioNet PSV or CSV-mirror files with schema validation."""

    path = Path(data_path)
    if not path.exists():
        msg = f"Input path does not exist: {path}"
        raise FileNotFoundError(msg)

    resolved_format = detect_dataset_format(path) if dataset_format == "auto" else dataset_format
    if resolved_format == "physionet":
        files = _find_files(path, "*.psv")
    elif resolved_format == "csv":
        files = _find_files(path, "*.csv")
    else:
        msg = f"Unsupported dataset format: {resolved_format}"
        raise ValueError(msg)

    if not files:
        msg = f"No input files found for format '{resolved_format}' at {path}"
        raise FileNotFoundError(msg)

    issues: list[SchemaValidationIssue] = []
    valid_frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            raw_frame = _read_file(file_path=file_path, dataset_format=resolved_format)
            normalized_frame = _normalize_file_schema(raw_frame=raw_frame, file_path=file_path)
            valid_frames.append(normalized_frame)
        except Exception as exc:
            issue = SchemaValidationIssue(file_path=str(file_path), reason=str(exc))
            issues.append(issue)
            logger.warning(
                "Skipping invalid raw data file",
                extra={"file_path": str(file_path), "reason": str(exc)},
            )
            if strict_validation:
                msg = f"Validation failed for {file_path}: {exc}"
                raise ValueError(msg) from exc

    if not valid_frames:
        issue_preview = "; ".join(f"{issue.file_path}: {issue.reason}" for issue in issues[:3])
        msg = f"No valid files were ingested from {path}. Issues: {issue_preview}"
        raise ValueError(msg)

    dataframe = pd.concat(valid_frames, ignore_index=True)
    dataframe = dataframe.sort_values(
        [PATIENT_ID_COLUMN, TIME_COLUMN, SOURCE_FILE_COLUMN, SOURCE_ROW_COLUMN]
    )
    dataframe = dataframe.reset_index(drop=True)

    feature_columns = infer_feature_columns(dataframe.columns)

    logger.info(
        "Raw ingestion completed",
        extra={
            "dataset_format": resolved_format,
            "files_total": len(files),
            "files_skipped": len(issues),
            "rows": len(dataframe),
            "patients": int(dataframe[PATIENT_ID_COLUMN].nunique()),
        },
    )

    return IngestionResult(
        dataframe=dataframe,
        dataset_format=resolved_format,
        feature_columns=feature_columns,
        issues=issues,
        file_count=len(files),
        skipped_file_count=len(issues),
    )


def validate_schema(data_path: str | Path, dataset_format: DatasetFormat = "auto") -> IngestionResult:
    """Validates raw files and returns ingestion diagnostics."""

    return ingest_raw_dataset(
        data_path=data_path,
        dataset_format=dataset_format,
        strict_validation=False,
    )


def load_csv_dataset(data_path: Path) -> pd.DataFrame:
    """Loads one CSV file or concatenates all CSV files under a directory."""

    return ingest_raw_dataset(data_path=data_path, dataset_format="csv").dataframe


def load_physionet_dataset(data_path: Path) -> pd.DataFrame:
    """Loads official PhysioNet sepsis challenge PSV files."""

    return ingest_raw_dataset(data_path=data_path, dataset_format="physionet").dataframe


def load_dataset(data_path: str | Path, dataset_format: DatasetFormat = "auto") -> pd.DataFrame:
    """Loads tabular data from local CSV or official PhysioNet format."""

    return ingest_raw_dataset(data_path=data_path, dataset_format=dataset_format).dataframe
