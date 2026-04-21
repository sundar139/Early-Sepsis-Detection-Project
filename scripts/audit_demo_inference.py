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
        msg = "Expected 2D matrix payload for features or missing_mask"
        raise ValueError(msg)

    return array.tolist()


def _to_vector_list(value: Any) -> list[float] | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        msg = "Expected 1D static_features payload"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit demo inference diversity and score-mapping fidelity."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("public_artifacts/models/registry/selected_model.json"),
        help="Path to selected model manifest.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("assets/demo/sequence_demo_samples.parquet"),
        help="Path to demo sample parquet.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Maximum number of rows to audit.",
    )
    parser.add_argument(
        "--operating-mode",
        type=str,
        default="default",
        help="Serving operating mode used for score extraction.",
    )
    parser.add_argument(
        "--display-round-decimals",
        type=int,
        default=6,
        help="Display rounding precision used by the app result table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = resolve_runtime_path(args.manifest_path)
    parquet_path = resolve_runtime_path(args.parquet_path)

    if not manifest_path.exists():
        msg = f"Manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)
    if not parquet_path.exists():
        msg = f"Parquet not found: {parquet_path}"
        raise FileNotFoundError(msg)

    manifest = load_model_manifest(manifest_path)
    dataset_tag = str(manifest["dataset"]["dataset_tag"])

    frame = pd.read_parquet(
        parquet_path,
        columns=["patient_id", "end_hour", "label", "features", "missing_mask", "static_features"],
    ).head(args.max_rows)

    samples = [_row_to_request_sample(row) for _, row in frame.iterrows()]
    predictions = predict_sequence_samples(
        manifest_path=manifest_path,
        dataset_tag=dataset_tag,
        samples=samples,
        operating_mode=args.operating_mode,
    )

    displayed_scores = [
        round(float(prediction["predicted_probability"]), args.display_round_decimals)
        for prediction in predictions
    ]

    report = build_inference_diagnostics(
        samples=samples,
        predictions=predictions,
        displayed_scores=displayed_scores,
        displayed_round_decimals=args.display_round_decimals,
    )
    report["manifest_path"] = str(manifest_path)
    report["parquet_path"] = str(parquet_path)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
