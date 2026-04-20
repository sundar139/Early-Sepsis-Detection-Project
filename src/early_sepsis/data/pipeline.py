from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from early_sepsis.data.ingestion import DatasetFormat, ingest_raw_dataset
from early_sepsis.data.preprocessing import (
    preprocess_time_series_splits,
    serialize_preprocessing_statistics,
)
from early_sepsis.data.schema import PATIENT_ID_COLUMN, TARGET_COLUMN
from early_sepsis.data.splitting import (
    apply_split_assignments,
    save_split_manifests,
    split_patients,
)
from early_sepsis.data.windowing import WindowConfig, generate_sliding_windows, summarize_windows
from early_sepsis.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PreprocessingPipelineResult:
    """Output paths and summary metadata from preprocessing pipeline execution."""

    output_dir: Path
    metadata_path: Path
    feature_schema_path: Path
    split_paths: dict[str, Path]
    manifest_paths: dict[str, Path]


@dataclass(slots=True)
class WindowPipelineResult:
    """Output paths and summary metadata from window generation execution."""

    output_dir: Path
    metadata_path: Path
    feature_schema_path: Path
    split_paths: dict[str, Path]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _relative_to(base: Path, target: Path) -> str:
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)


def _build_feature_schema(
    feature_columns: list[str],
    mask_columns: list[str],
    static_feature_columns: list[str],
) -> dict[str, Any]:
    mask_lookup = {mask_column.replace("__missing", ""): mask_column for mask_column in mask_columns}
    return {
        "features": [
            {
                "name": feature,
                "dtype": "float32",
                "standardized": True,
                "is_static": feature in static_feature_columns,
                "missing_mask": mask_lookup.get(feature),
            }
            for feature in feature_columns
        ]
    }


def run_preprocessing_pipeline(
    raw_data_path: str | Path,
    output_dir: str | Path,
    dataset_format: DatasetFormat = "auto",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    strict_validation: bool = False,
    selected_feature_columns: Sequence[str] | None = None,
) -> PreprocessingPipelineResult:
    """Runs ingestion, patient split, preprocessing, and artifact persistence."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ingestion_result = ingest_raw_dataset(
        data_path=raw_data_path,
        dataset_format=dataset_format,
        strict_validation=strict_validation,
    )

    split_result = split_patients(
        dataframe=ingestion_result.dataframe,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    split_frames = apply_split_assignments(
        dataframe=ingestion_result.dataframe,
        split_result=split_result,
    )

    if selected_feature_columns is not None:
        feature_columns = [feature for feature in selected_feature_columns if feature in split_frames["train"]]
    else:
        feature_columns = [
            feature
            for feature in ingestion_result.feature_columns
            if feature != TARGET_COLUMN and feature in split_frames["train"].columns
        ]

    if not feature_columns:
        msg = "No usable feature columns were found for preprocessing"
        raise ValueError(msg)

    processed = preprocess_time_series_splits(
        split_frames=split_frames,
        feature_columns=feature_columns,
    )

    split_paths: dict[str, Path] = {}
    split_summary: dict[str, dict[str, int]] = {}
    for split_name, frame in processed.split_frames.items():
        split_file = output_path / f"{split_name}.parquet"
        frame.to_parquet(split_file, index=False)
        split_paths[split_name] = split_file
        split_summary[split_name] = {
            "row_count": int(len(frame)),
            "patient_count": int(frame[PATIENT_ID_COLUMN].nunique()),
        }

    manifest_paths = save_split_manifests(
        split_result=split_result,
        output_dir=output_path / "split_manifests",
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_data_path": str(raw_data_path),
        "dataset_format": ingestion_result.dataset_format,
        "random_seed": random_seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "feature_columns": processed.statistics.feature_columns,
        "mask_columns": processed.statistics.mask_columns,
        "static_feature_columns": processed.static_feature_columns,
        "preprocessing_statistics": serialize_preprocessing_statistics(processed.statistics),
        "ingestion": {
            "file_count": ingestion_result.file_count,
            "skipped_file_count": ingestion_result.skipped_file_count,
            "issues": [
                {"file_path": issue.file_path, "reason": issue.reason}
                for issue in ingestion_result.issues
            ],
        },
        "split_summary": split_summary,
        "split_files": {
            split_name: _relative_to(output_path, split_file)
            for split_name, split_file in split_paths.items()
        },
        "manifests": {
            name: _relative_to(output_path, manifest_path)
            for name, manifest_path in manifest_paths.items()
        },
    }

    metadata_path = output_path / "metadata.json"
    _write_json(metadata_path, metadata)

    feature_schema = _build_feature_schema(
        feature_columns=processed.statistics.feature_columns,
        mask_columns=processed.statistics.mask_columns,
        static_feature_columns=processed.static_feature_columns,
    )
    feature_schema_path = output_path / "feature_schema.json"
    _write_json(feature_schema_path, feature_schema)

    logger.info(
        "Preprocessing pipeline completed",
        extra={
            "output_dir": str(output_path),
            "split_summary": split_summary,
            "feature_count": len(processed.statistics.feature_columns),
        },
    )

    return PreprocessingPipelineResult(
        output_dir=output_path,
        metadata_path=metadata_path,
        feature_schema_path=feature_schema_path,
        split_paths=split_paths,
        manifest_paths=manifest_paths,
    )


def load_pipeline_metadata(processed_dir: str | Path) -> dict[str, Any]:
    """Loads preprocessing metadata from a processed output directory."""

    metadata_path = Path(processed_dir) / "metadata.json"
    if not metadata_path.exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    with metadata_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        msg = f"Invalid metadata format in {metadata_path}"
        raise ValueError(msg)

    return payload


def create_window_pipeline(
    processed_dir: str | Path,
    output_dir: str | Path,
    window_length: int,
    prediction_horizon: int,
    padding_mode: bool = False,
    include_masks: bool = True,
    include_static: bool = True,
    static_feature_columns: Sequence[str] | None = None,
) -> WindowPipelineResult:
    """Generates sliding windows per split and persists parquet outputs."""

    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = load_pipeline_metadata(processed_path)
    feature_columns = metadata.get("feature_columns", [])
    mask_columns = metadata.get("mask_columns", []) if include_masks else []

    if static_feature_columns is not None:
        resolved_static_features = list(static_feature_columns)
    elif include_static:
        resolved_static_features = metadata.get("static_feature_columns", [])
    else:
        resolved_static_features = []

    config = WindowConfig(
        window_length=window_length,
        prediction_horizon=prediction_horizon,
        padding_mode=padding_mode,
    )

    split_paths: dict[str, Path] = {}
    window_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        split_input_path = processed_path / f"{split_name}.parquet"
        if not split_input_path.exists():
            msg = f"Processed split parquet not found: {split_input_path}"
            raise FileNotFoundError(msg)

        split_frame = pd.read_parquet(split_input_path)
        generated = generate_sliding_windows(
            dataframe=split_frame,
            feature_columns=feature_columns,
            mask_columns=mask_columns,
            static_columns=resolved_static_features,
            config=config,
        )

        split_output_path = output_path / f"{split_name}.parquet"
        generated.windows.to_parquet(split_output_path, index=False)

        split_paths[split_name] = split_output_path
        window_frames[split_name] = generated.windows

    summary = summarize_windows(window_frames)
    window_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processed_dir": str(processed_path),
        "window_length": window_length,
        "prediction_horizon": prediction_horizon,
        "padding_mode": padding_mode,
        "include_masks": include_masks,
        "include_static": include_static,
        "feature_columns": feature_columns,
        "mask_columns": mask_columns,
        "static_feature_columns": resolved_static_features,
        "summary": summary,
        "split_files": {
            split_name: _relative_to(output_path, split_file)
            for split_name, split_file in split_paths.items()
        },
    }
    metadata_path = output_path / "metadata.json"
    _write_json(metadata_path, window_metadata)

    feature_schema = _build_feature_schema(
        feature_columns=feature_columns,
        mask_columns=mask_columns,
        static_feature_columns=resolved_static_features,
    )
    feature_schema_path = output_path / "feature_schema.json"
    _write_json(feature_schema_path, feature_schema)

    logger.info(
        "Window generation completed",
        extra={
            "output_dir": str(output_path),
            "window_summary": summary,
            "window_length": window_length,
            "prediction_horizon": prediction_horizon,
        },
    )

    return WindowPipelineResult(
        output_dir=output_path,
        metadata_path=metadata_path,
        feature_schema_path=feature_schema_path,
        split_paths=split_paths,
    )


def build_split_summary(processed_dir: str | Path) -> dict[str, dict[str, int | float]]:
    """Builds split-level patient and label summaries from processed parquet files."""

    processed_path = Path(processed_dir)
    summary: dict[str, dict[str, int | float]] = {}
    for split_name in ("train", "validation", "test"):
        split_path = processed_path / f"{split_name}.parquet"
        if not split_path.exists():
            continue

        frame = pd.read_parquet(split_path)
        positive_rate = float(frame[TARGET_COLUMN].mean()) if len(frame) > 0 else 0.0
        summary[split_name] = {
            "row_count": int(len(frame)),
            "patient_count": int(frame[PATIENT_ID_COLUMN].nunique()),
            "positive_rate": positive_rate,
        }

    return summary
