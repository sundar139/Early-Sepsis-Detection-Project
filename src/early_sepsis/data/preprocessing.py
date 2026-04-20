from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from early_sepsis.data.schema import PATIENT_ID_COLUMN, SOURCE_ROW_COLUMN, TARGET_COLUMN, TIME_COLUMN
from early_sepsis.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PreparedData:
    """Container for train/test split outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]


def split_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Splits a dataframe into features and binary target."""

    if target_column not in dataframe.columns:
        msg = f"Target column '{target_column}' not found in dataset."
        raise KeyError(msg)

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column].astype(int)
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Creates preprocessing pipelines for numeric and categorical columns."""

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def prepare_training_data(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> PreparedData:
    """Prepares train/test splits with optional class stratification."""

    features, target = split_features_and_target(dataframe=dataframe, target_column=target_column)
    stratify = target if target.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=features.columns.tolist(),
    )


@dataclass(slots=True)
class PreprocessingStatistics:
    """Train-only statistics used for imputation and standardization."""

    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    feature_columns: list[str]
    mask_columns: list[str]


@dataclass(slots=True)
class ProcessedTimeSeriesSplits:
    """Processed split frames and reusable preprocessing metadata."""

    split_frames: dict[str, pd.DataFrame]
    statistics: PreprocessingStatistics
    static_feature_columns: list[str]


def _coerce_features_to_numeric(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    coerced = dataframe.copy()
    for column in feature_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def sort_and_enforce_monotonic_hourly_ordering(
    dataframe: pd.DataFrame,
    patient_column: str = PATIENT_ID_COLUMN,
    time_column: str = TIME_COLUMN,
) -> pd.DataFrame:
    """Sorts rows and removes duplicate hourly rows per patient."""

    if patient_column not in dataframe.columns or time_column not in dataframe.columns:
        msg = f"Expected columns '{patient_column}' and '{time_column}' in dataframe"
        raise KeyError(msg)

    sort_columns = [patient_column, time_column]
    if SOURCE_ROW_COLUMN in dataframe.columns:
        sort_columns.append(SOURCE_ROW_COLUMN)

    ordered = dataframe.sort_values(sort_columns).reset_index(drop=True)

    before = len(ordered)
    ordered = ordered.drop_duplicates(
        subset=[patient_column, time_column],
        keep="first",
    ).reset_index(drop=True)
    removed_duplicates = before - len(ordered)
    if removed_duplicates > 0:
        logger.warning(
            "Removed duplicate patient-hour rows while enforcing monotonic order",
            extra={"removed_rows": removed_duplicates},
        )

    time_delta = ordered.groupby(patient_column, sort=False)[time_column].diff()
    if bool((time_delta < 0).any()):
        msg = "Detected non-monotonic hourly ordering after sorting"
        raise ValueError(msg)

    return ordered


def add_missingness_masks(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Adds one binary missingness mask column per feature."""

    masked = dataframe.copy()
    mask_columns: list[str] = []
    for feature in feature_columns:
        mask_column = f"{feature}__missing"
        masked[mask_column] = masked[feature].isna().astype(np.int8)
        mask_columns.append(mask_column)
    return masked, mask_columns


def forward_fill_within_patient(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    patient_column: str = PATIENT_ID_COLUMN,
) -> pd.DataFrame:
    """Forward-fills feature values inside each patient trajectory."""

    result = dataframe.copy()
    result[feature_columns] = result.groupby(patient_column, sort=False)[feature_columns].ffill()
    return result


def fit_train_statistics(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> PreprocessingStatistics:
    """Fits median and standardization statistics from training data only."""

    if train_frame.empty:
        msg = "Training frame is empty; cannot compute preprocessing statistics"
        raise ValueError(msg)

    medians_series = train_frame[feature_columns].median(skipna=True)
    means_series = train_frame[feature_columns].mean(skipna=True)
    stds_series = train_frame[feature_columns].std(skipna=True, ddof=0)

    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature in feature_columns:
        median_value = float(medians_series.get(feature, np.nan))
        mean_value = float(means_series.get(feature, np.nan))
        std_value = float(stds_series.get(feature, np.nan))

        if np.isnan(median_value):
            median_value = 0.0
        if np.isnan(mean_value):
            mean_value = 0.0
        if np.isnan(std_value) or std_value == 0.0:
            std_value = 1.0

        medians[feature] = median_value
        means[feature] = mean_value
        stds[feature] = std_value

    mask_columns = [f"{feature}__missing" for feature in feature_columns]
    return PreprocessingStatistics(
        medians=medians,
        means=means,
        stds=stds,
        feature_columns=feature_columns,
        mask_columns=mask_columns,
    )


def apply_imputation_and_standardization(
    dataframe: pd.DataFrame,
    statistics: PreprocessingStatistics,
) -> pd.DataFrame:
    """Applies train-fitted median imputation and z-score scaling."""

    processed = dataframe.copy()
    for feature in statistics.feature_columns:
        processed[feature] = pd.to_numeric(processed[feature], errors="coerce")
        processed[feature] = processed[feature].fillna(statistics.medians[feature])
        processed[feature] = (processed[feature] - statistics.means[feature]) / statistics.stds[feature]
    return processed


def detect_static_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    patient_column: str = PATIENT_ID_COLUMN,
) -> list[str]:
    """Finds columns that are constant within each patient trajectory."""

    if dataframe.empty:
        return []

    static_features: list[str] = []
    grouped = dataframe.groupby(patient_column, sort=False)
    for feature in feature_columns:
        max_unique_per_patient = grouped[feature].nunique(dropna=False).max()
        if int(max_unique_per_patient) <= 1:
            static_features.append(feature)

    return static_features


def preprocess_time_series_splits(
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    patient_column: str = PATIENT_ID_COLUMN,
    time_column: str = TIME_COLUMN,
) -> ProcessedTimeSeriesSplits:
    """Processes train/validation/test frames with train-only fitted statistics."""

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(split_frames)
    if missing_splits:
        msg = f"Missing required splits: {sorted(missing_splits)}"
        raise ValueError(msg)

    sorted_and_filled: dict[str, pd.DataFrame] = {}
    resolved_mask_columns: list[str] = []
    for split_name, frame in split_frames.items():
        split_frame = sort_and_enforce_monotonic_hourly_ordering(
            dataframe=frame,
            patient_column=patient_column,
            time_column=time_column,
        )
        split_frame = _coerce_features_to_numeric(split_frame, feature_columns)
        split_frame, mask_columns = add_missingness_masks(split_frame, feature_columns)
        split_frame = forward_fill_within_patient(
            dataframe=split_frame,
            feature_columns=feature_columns,
            patient_column=patient_column,
        )
        sorted_and_filled[split_name] = split_frame
        resolved_mask_columns = mask_columns

    statistics = fit_train_statistics(
        train_frame=sorted_and_filled["train"],
        feature_columns=feature_columns,
    )
    statistics.mask_columns = resolved_mask_columns

    processed_splits = {
        split_name: apply_imputation_and_standardization(frame, statistics)
        for split_name, frame in sorted_and_filled.items()
    }

    static_feature_columns = detect_static_features(
        dataframe=processed_splits["train"],
        feature_columns=feature_columns,
        patient_column=patient_column,
    )

    return ProcessedTimeSeriesSplits(
        split_frames=processed_splits,
        statistics=statistics,
        static_feature_columns=static_feature_columns,
    )


def serialize_preprocessing_statistics(statistics: PreprocessingStatistics) -> dict[str, Any]:
    """Serializes preprocessing statistics to JSON-friendly values."""

    return {
        "feature_columns": statistics.feature_columns,
        "mask_columns": statistics.mask_columns,
        "medians": {key: float(value) for key, value in statistics.medians.items()},
        "means": {key: float(value) for key, value in statistics.means.items()},
        "stds": {key: float(value) for key, value in statistics.stds.items()},
    }
