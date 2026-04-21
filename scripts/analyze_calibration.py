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

from early_sepsis.modeling.experiment_analysis import analyze_checkpoint_calibration
from early_sepsis.modeling.model_manifest import (
    load_model_manifest,
    sync_manifest_thresholds_from_calibration,
)
from early_sepsis.runtime_paths import resolve_runtime_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run calibration and plotting analysis for a selected sequence checkpoint"
    )
    parser.add_argument(
        "--manifest-path",
        default="artifacts/models/registry/selected_model.json",
        help="Path to selected model manifest",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Window split to analyze when parquet-path is omitted",
    )
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Explicit evaluation parquet path (overrides split resolution)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/calibration",
        help="Output directory for analysis artifacts",
    )
    parser.add_argument(
        "--high-recall-target",
        type=float,
        default=0.9,
        help="Recall target for recommending a high-recall threshold",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sync-manifest-thresholds",
        action="store_true",
        help="Synchronize selected manifest thresholds from calibration artifacts",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_parquet_path(
    manifest: dict[str, object], split: str, explicit_path: str | None
) -> Path:
    if explicit_path is not None:
        return resolve_runtime_path(explicit_path, project_root=PROJECT_ROOT)

    dataset_section = manifest.get("dataset", {})
    if not isinstance(dataset_section, dict):
        msg = "Manifest dataset section is invalid"
        raise ValueError(msg)

    windows_dir_value = dataset_section.get("windows_dir")
    if not isinstance(windows_dir_value, str):
        msg = "Manifest dataset.windows_dir must be a string"
        raise ValueError(msg)

    windows_dir = resolve_runtime_path(windows_dir_value, project_root=PROJECT_ROOT)
    return windows_dir / f"{split}.parquet"


def main() -> None:
    args = _build_parser().parse_args()

    manifest = load_model_manifest(args.manifest_path)
    checkpoint_path = resolve_runtime_path(
        manifest["selected_run"]["checkpoint_path"],
        project_root=PROJECT_ROOT,
    )
    parquet_path = _resolve_parquet_path(
        manifest, split=args.split, explicit_path=args.parquet_path
    )

    artifacts = analyze_checkpoint_calibration(
        checkpoint_path=checkpoint_path,
        parquet_path=parquet_path,
        output_dir=args.output_dir,
        default_threshold=float(manifest["thresholds"]["default"]),
        high_recall_target=args.high_recall_target,
        calibration_bins=args.calibration_bins,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    should_sync_manifest = args.sync_manifest_thresholds or args.update_manifest
    if should_sync_manifest:
        manifest = sync_manifest_thresholds_from_calibration(
            manifest_path=args.manifest_path,
            recommendations_path=artifacts.recommendations_path,
            calibration_summary_path=artifacts.summary_path,
        )

    payload = {
        "output_dir": str(artifacts.output_dir),
        "threshold_sweep_path": str(artifacts.threshold_sweep_path),
        "reliability_curve_path": str(artifacts.reliability_curve_path),
        "recommendations_path": str(artifacts.recommendations_path),
        "summary_path": str(artifacts.summary_path),
        "markdown_report_path": str(artifacts.markdown_report_path),
        "plot_paths": {name: str(path) for name, path in artifacts.plot_paths.items()},
        "recommended_thresholds": artifacts.recommendations,
        "manifest_thresholds": manifest["thresholds"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
