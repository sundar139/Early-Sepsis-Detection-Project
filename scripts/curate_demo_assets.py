from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.demo.inference_debug import build_inference_diagnostics
from early_sepsis.modeling.model_manifest import load_model_manifest
from early_sepsis.runtime_paths import resolve_runtime_path
from early_sepsis.serving.sequence_service import predict_sequence_samples


def _to_matrix_list(value: Any) -> list[list[float]]:
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == object:
        array = np.asarray(array.tolist(), dtype=np.float32)
    else:
        array = np.asarray(value, dtype=np.float32)

    if array.ndim != 2:
        msg = "Expected 2D matrix payload"
        raise ValueError(msg)

    return array.tolist()


def _to_vector_list(value: Any) -> list[float] | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        msg = "Expected 1D vector payload"
        raise ValueError(msg)

    return array.tolist()


def _row_to_request_sample(row: pd.Series) -> dict[str, Any]:
    return {
        "patient_id": row.get("patient_id"),
        "end_hour": int(row.get("end_hour", 0)),
        "features": _to_matrix_list(row.get("features")),
        "missing_mask": _to_matrix_list(row.get("missing_mask")),
        "static_features": _to_vector_list(row.get("static_features")),
    }


def _canonicalize_frame(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    canonical = frame.copy().reset_index(drop=True)
    canonical["patient_id"] = [f"DS-{prefix}-{index + 1:04d}" for index in range(len(canonical))]
    canonical["end_hour"] = canonical["end_hour"].astype(int)
    canonical["label"] = canonical["label"].astype(int)
    canonical["features"] = [_to_matrix_list(value) for value in canonical["features"]]
    canonical["missing_mask"] = [_to_matrix_list(value) for value in canonical["missing_mask"]]
    canonical["static_features"] = [
        _to_vector_list(value) for value in canonical["static_features"]
    ]
    return canonical


def _select_evenly(indices: np.ndarray, count: int) -> list[int]:
    if count <= 0 or indices.size == 0:
        return []
    if indices.size <= count:
        return [int(item) for item in indices.tolist()]

    positions = np.linspace(0, indices.size - 1, num=count)
    selected = indices[np.round(positions).astype(int)]
    deduplicated: list[int] = []
    seen: set[int] = set()
    for item in selected.tolist():
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _derive_default_sources(manifest_path: Path) -> list[Path]:
    manifest = load_model_manifest(manifest_path)
    windows_dir_value = manifest.get("dataset", {}).get("windows_dir")
    if not isinstance(windows_dir_value, str) or not windows_dir_value.strip():
        return []

    windows_dir = resolve_runtime_path(windows_dir_value, anchor=manifest_path.parent)
    return [windows_dir / "validation.parquet", windows_dir / "test.parquet"]


def _load_candidate_rows(
    *,
    source_paths: list[Path],
    candidate_rows_per_source: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    required_columns = [
        "patient_id",
        "end_hour",
        "label",
        "features",
        "missing_mask",
        "static_features",
    ]

    for source_path in source_paths:
        if not source_path.exists() or not source_path.is_file():
            continue

        try:
            frame = pd.read_parquet(source_path, columns=required_columns).head(
                candidate_rows_per_source
            )
        except Exception:
            continue

        if frame.empty:
            continue

        frame = frame.dropna(subset=["features", "missing_mask", "static_features"]).copy()
        if frame.empty:
            continue

        frames.append(frame)

    if not frames:
        msg = "No usable candidate windows were loaded from configured source parquet files."
        raise FileNotFoundError(msg)

    return pd.concat(frames, ignore_index=True)


def _choose_demo_indices(
    *,
    probabilities: np.ndarray,
    threshold: float,
    demo_count: int,
) -> list[int]:
    if probabilities.size == 0:
        return []

    per_bucket = max(1, demo_count // 3)

    sorted_indices = np.argsort(probabilities)
    low_pool = np.where(probabilities < max(0.2, threshold - 0.25))[0]
    border_pool = np.where(np.abs(probabilities - threshold) <= 0.05)[0]
    high_pool = np.where(probabilities >= min(0.8, threshold))[0]

    if low_pool.size == 0:
        low_pool = sorted_indices[: max(per_bucket * 2, 1)]
    if border_pool.size == 0:
        border_pool = np.argsort(np.abs(probabilities - threshold))[: max(per_bucket * 2, 1)]
    if high_pool.size == 0:
        high_pool = sorted_indices[-max(per_bucket * 2, 1) :]

    low_selected = _select_evenly(low_pool[np.argsort(probabilities[low_pool])], per_bucket)
    border_selected = _select_evenly(
        border_pool[np.argsort(np.abs(probabilities[border_pool] - threshold))],
        per_bucket,
    )
    high_selected = _select_evenly(
        high_pool[np.argsort(probabilities[high_pool])[::-1]],
        per_bucket,
    )

    selected: list[int] = []
    seen: set[int] = set()
    for index in low_selected + border_selected + high_selected:
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)

    fill_candidates = np.concatenate(
        [
            np.argsort(probabilities),
            np.argsort(np.abs(probabilities - threshold)),
            np.argsort(probabilities)[::-1],
        ]
    )
    for index in fill_candidates.tolist():
        item = int(index)
        if item in seen:
            continue
        selected.append(item)
        seen.add(item)
        if len(selected) >= demo_count:
            break

    return selected[: min(demo_count, probabilities.size)]


def _choose_operational_indices(*, probabilities: np.ndarray, count: int) -> list[int]:
    if probabilities.size == 0:
        return []
    sorted_indices = np.argsort(probabilities)
    return _select_evenly(sorted_indices, min(count, probabilities.size))


def _save_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate deployment-safe demo windows with real score diversity and an operational "
            "subset from existing model-compatible windows."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/models/registry/selected_model.json"),
        help="Path to selected manifest.",
    )
    parser.add_argument(
        "--primary-parquet",
        type=Path,
        default=None,
        help=(
            "Primary source parquet. If omitted, validation split from manifest windows_dir "
            "is used."
        ),
    )
    parser.add_argument(
        "--secondary-parquet",
        type=Path,
        default=None,
        help=(
            "Optional secondary source parquet. If omitted, test split from manifest "
            "windows_dir is used."
        ),
    )
    parser.add_argument(
        "--candidate-rows-per-source",
        type=int,
        default=3000,
        help="Rows loaded from each source parquet before curation.",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=36,
        help="Row count for curated live-inference demo parquet.",
    )
    parser.add_argument(
        "--operational-count",
        type=int,
        default=600,
        help="Row count for curated operational deployment subset.",
    )
    parser.add_argument(
        "--operating-mode",
        type=str,
        default="default",
        help="Operating mode used for probability stratification.",
    )
    parser.add_argument(
        "--output-demo-path",
        type=Path,
        default=Path("assets/demo/sequence_demo_samples.parquet"),
        help="Output parquet for deployment live-inference demo samples.",
    )
    parser.add_argument(
        "--output-operational-path",
        type=Path,
        default=Path("assets/demo/operational_windows_subset.parquet"),
        help="Output parquet for deployment operational summary subset.",
    )
    parser.add_argument(
        "--output-walkthrough-path",
        type=Path,
        default=Path("assets/demo/saved_example_payload.json"),
        help="Output path for saved walkthrough payload JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = resolve_runtime_path(args.manifest_path)
    if not manifest_path.exists():
        msg = f"Manifest was not found: {manifest_path}"
        raise FileNotFoundError(msg)

    manifest = load_model_manifest(manifest_path)
    dataset_tag = str(manifest["dataset"]["dataset_tag"])

    source_paths: list[Path] = []
    if args.primary_parquet is not None:
        source_paths.append(
            resolve_runtime_path(args.primary_parquet, anchor=manifest_path.parent)
        )
    if args.secondary_parquet is not None:
        source_paths.append(
            resolve_runtime_path(args.secondary_parquet, anchor=manifest_path.parent)
        )

    if not source_paths:
        source_paths = _derive_default_sources(manifest_path)

    candidate_frame = _load_candidate_rows(
        source_paths=source_paths,
        candidate_rows_per_source=args.candidate_rows_per_source,
    )

    request_samples = [_row_to_request_sample(row) for _, row in candidate_frame.iterrows()]
    predictions = predict_sequence_samples(
        manifest_path=manifest_path,
        dataset_tag=dataset_tag,
        samples=request_samples,
        operating_mode=args.operating_mode,
    )

    probabilities = np.asarray(
        [float(prediction["predicted_probability"]) for prediction in predictions],
        dtype=np.float64,
    )
    threshold_used = float(predictions[0]["threshold_used"]) if predictions else 0.5

    demo_indices = _choose_demo_indices(
        probabilities=probabilities,
        threshold=threshold_used,
        demo_count=args.demo_count,
    )
    operational_indices = _choose_operational_indices(
        probabilities=probabilities,
        count=args.operational_count,
    )

    demo_frame = _canonicalize_frame(candidate_frame.iloc[demo_indices], prefix="EX")
    operational_frame = _canonicalize_frame(candidate_frame.iloc[operational_indices], prefix="OP")

    _save_parquet(resolve_runtime_path(args.output_demo_path), demo_frame)
    _save_parquet(resolve_runtime_path(args.output_operational_path), operational_frame)

    walkthrough_payload = {
        "sample_id": "DS-EX-001",
        "label": int(demo_frame.iloc[0]["label"]) if not demo_frame.empty else 0,
        "end_hour": int(demo_frame.iloc[0]["end_hour"]) if not demo_frame.empty else 8,
        "walkthrough_note": (
            "Saved synthetic walkthrough sample used when live parquet windows are unavailable "
            "in deployment."
        ),
    }
    walkthrough_output_path = resolve_runtime_path(args.output_walkthrough_path)
    walkthrough_output_path.parent.mkdir(parents=True, exist_ok=True)
    walkthrough_output_path.write_text(
        json.dumps(walkthrough_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    demo_request_samples = [_row_to_request_sample(row) for _, row in demo_frame.iterrows()]
    demo_predictions = predict_sequence_samples(
        manifest_path=manifest_path,
        dataset_tag=dataset_tag,
        samples=demo_request_samples,
        operating_mode=args.operating_mode,
    )

    diagnostics = build_inference_diagnostics(
        samples=demo_request_samples,
        predictions=demo_predictions,
        displayed_scores=[
            round(float(item["predicted_probability"]), 6) for item in demo_predictions
        ],
        displayed_round_decimals=6,
    )

    report = {
        "manifest_path": str(manifest_path),
        "source_paths": [str(path) for path in source_paths],
        "candidate_count": len(candidate_frame),
        "threshold_used": threshold_used,
        "demo_output_path": str(resolve_runtime_path(args.output_demo_path)),
        "operational_output_path": str(resolve_runtime_path(args.output_operational_path)),
        "demo_row_count": len(demo_frame),
        "operational_row_count": len(operational_frame),
        "demo_probability_min": float(probabilities[demo_indices].min()) if demo_indices else None,
        "demo_probability_max": float(probabilities[demo_indices].max()) if demo_indices else None,
        "diagnostics": diagnostics,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
