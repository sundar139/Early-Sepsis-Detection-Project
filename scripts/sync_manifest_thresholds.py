from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.modeling.model_manifest import (
    load_model_manifest,
    load_threshold_recommendations,
    sync_manifest_thresholds_from_calibration,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize selected-model manifest thresholds from calibration artifacts"
    )
    parser.add_argument(
        "--manifest-path",
        default="artifacts/models/registry/selected_model.json",
        help="Path to selected model manifest",
    )
    parser.add_argument(
        "--recommendations-path",
        default="artifacts/analysis/calibration/threshold_recommendations.json",
        help="Path to calibration threshold recommendations JSON",
    )
    parser.add_argument(
        "--summary-path",
        default="artifacts/analysis/calibration/calibration_summary.json",
        help="Path to calibration summary JSON used for checkpoint consistency validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview updates without writing the manifest",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    current_manifest = load_model_manifest(args.manifest_path)
    recommended_thresholds = load_threshold_recommendations(args.recommendations_path)

    previous_thresholds = {
        "default": float(current_manifest["thresholds"]["default"]),
        "balanced": float(current_manifest["thresholds"]["balanced"]),
        "high_recall": float(current_manifest["thresholds"]["high_recall"]),
    }

    updated_manifest = sync_manifest_thresholds_from_calibration(
        manifest_path=args.manifest_path,
        recommendations_path=args.recommendations_path,
        calibration_summary_path=args.summary_path,
        write_changes=not args.dry_run,
    )

    payload = {
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "recommendations_path": str(Path(args.recommendations_path).resolve()),
        "summary_path": str(Path(args.summary_path).resolve()),
        "applied": not args.dry_run,
        "previous_thresholds": previous_thresholds,
        "recommended_thresholds": {
            "default": recommended_thresholds["default"],
            "balanced": recommended_thresholds["balanced"],
            "high_recall": recommended_thresholds["high_recall"],
        },
        "updated_thresholds": updated_manifest["thresholds"],
        "threshold_metadata": updated_manifest.get("threshold_metadata", {}),
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
