from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from early_sepsis.modeling.model_manifest import load_model_manifest, save_model_manifest
from early_sepsis.runtime_paths import make_portable_path, resolve_runtime_path


def _copy_file_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists() or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_manifest_related_path(
    value: Any,
    *,
    manifest_path: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return resolve_runtime_path(value, anchor=manifest_path.parent)


def _sanitize_manifest_dataset_paths(manifest_payload: dict[str, Any]) -> None:
    dataset_payload = manifest_payload.get("dataset")
    if not isinstance(dataset_payload, dict):
        return

    for key in ("raw_data_path", "windows_dir", "processed_dir"):
        path_value = dataset_payload.get(key)
        if isinstance(path_value, str) and path_value.strip():
            dataset_payload[key] = make_portable_path(path_value)


def _write_public_experiment_comparison(source: Path, destination: Path) -> bool:
    if not source.exists() or not source.is_file():
        return False

    desired_columns = [
        "run_name",
        "model_type",
        "model_family",
        "dataset_tag",
        "validation_auprc",
        "validation_auroc",
        "test_auprc",
        "runtime_seconds",
    ]

    try:
        with source.open("r", encoding="utf-8", newline="") as reader_handle:
            reader = csv.DictReader(reader_handle)
            if reader.fieldnames is None:
                return False

            output_columns = [
                column_name for column_name in desired_columns if column_name in reader.fieldnames
            ]
            if not output_columns:
                return False

            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as writer_handle:
                writer = csv.DictWriter(writer_handle, fieldnames=output_columns)
                writer.writeheader()
                for row in reader:
                    writer.writerow(
                        {
                            column_name: row.get(column_name, "")
                            for column_name in output_columns
                        }
                    )
    except OSError:
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Builds a compact public_artifacts bundle for Streamlit deployment from "
            "the selected manifest and related calibration outputs."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/models/registry/selected_model.json"),
        help="Path to selected_model.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("public_artifacts"),
        help="Target directory for compact deployment artifacts.",
    )
    parser.add_argument(
        "--include-experiment-comparison",
        action="store_true",
        help="Include sequence_experiment_comparison.csv when available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = resolve_runtime_path(args.manifest_path)
    if not manifest_path.exists():
        msg = f"Selected manifest was not found: {manifest_path}"
        raise FileNotFoundError(msg)

    output_root = resolve_runtime_path(args.output_dir)
    output_manifest_path = output_root / "models" / "registry" / "selected_model.json"

    manifest = load_model_manifest(manifest_path)
    rewritten_manifest = json.loads(json.dumps(manifest))
    _sanitize_manifest_dataset_paths(rewritten_manifest)

    selected_run = rewritten_manifest.get("selected_run", {})
    if not isinstance(selected_run, dict):
        msg = "Manifest selected_run payload is invalid"
        raise ValueError(msg)

    checkpoint_source = _resolve_manifest_related_path(
        selected_run.get("checkpoint_path"),
        manifest_path=manifest_path,
    )
    if checkpoint_source is None or not checkpoint_source.exists():
        msg = "Selected manifest checkpoint_path is missing or invalid"
        raise FileNotFoundError(msg)

    checkpoint_target = output_root / "models" / "checkpoints" / "best_checkpoint.pt"
    _copy_file_if_exists(checkpoint_source, checkpoint_target)

    selected_run["checkpoint_path"] = make_portable_path(checkpoint_target)
    selected_run["run_dir"] = make_portable_path(checkpoint_target.parent)

    run_config_source = _resolve_manifest_related_path(
        selected_run.get("run_config_path"),
        manifest_path=manifest_path,
    )
    if run_config_source is not None and run_config_source.exists():
        run_config_target = output_root / "models" / "configs" / "run_config.json"
        _copy_file_if_exists(run_config_source, run_config_target)
        selected_run["run_config_path"] = make_portable_path(run_config_target)

    copied_files: list[str] = [make_portable_path(checkpoint_target)]

    threshold_metadata = rewritten_manifest.get("threshold_metadata", {})
    calibration_summary_source: Path | None = None
    recommendations_source: Path | None = None
    if isinstance(threshold_metadata, dict):
        calibration_summary_source = _resolve_manifest_related_path(
            threshold_metadata.get("calibration_summary_path"),
            manifest_path=manifest_path,
        )
        recommendations_source = _resolve_manifest_related_path(
            threshold_metadata.get("recommendations_path"),
            manifest_path=manifest_path,
        )

    default_calibration_dir = resolve_runtime_path(Path("artifacts/analysis/calibration"))
    if calibration_summary_source is None:
        summary_fallback = default_calibration_dir / "calibration_summary.json"
        if summary_fallback.exists():
            calibration_summary_source = summary_fallback
    if recommendations_source is None:
        recommendations_fallback = default_calibration_dir / "threshold_recommendations.json"
        if recommendations_fallback.exists():
            recommendations_source = recommendations_fallback

    calibration_dir = output_root / "analysis" / "calibration"

    if recommendations_source is not None and recommendations_source.exists():
        recommendations_target = calibration_dir / "threshold_recommendations.json"
        _copy_file_if_exists(recommendations_source, recommendations_target)
        copied_files.append(make_portable_path(recommendations_target))
        if isinstance(threshold_metadata, dict):
            threshold_metadata["recommendations_path"] = make_portable_path(recommendations_target)

    if calibration_summary_source is not None and calibration_summary_source.exists():
        summary_payload = _load_json(calibration_summary_source)
        if summary_payload is not None:
            summary_target = calibration_dir / "calibration_summary.json"

            plot_mapping = summary_payload.get("plot_paths")
            if isinstance(plot_mapping, dict):
                rewritten_plot_mapping: dict[str, str] = {}
                for plot_key, plot_value in plot_mapping.items():
                    plot_source = _resolve_manifest_related_path(
                        plot_value,
                        manifest_path=calibration_summary_source,
                    )
                    if plot_source is None:
                        continue
                    plot_target = calibration_dir / f"{plot_key}.png"
                    if _copy_file_if_exists(plot_source, plot_target):
                        rewritten_plot_mapping[str(plot_key)] = make_portable_path(plot_target)
                        copied_files.append(make_portable_path(plot_target))

                # Reliability visuals are frequently stored outside plot_paths; prefer bundling
                # them explicitly when present to keep calibration panel artifact-backed.
                for reliability_name in ("reliability_curve.png", "reliability_curve.csv"):
                    reliability_source = calibration_summary_source.parent / reliability_name
                    reliability_target = calibration_dir / reliability_name
                    if _copy_file_if_exists(reliability_source, reliability_target):
                        copied_files.append(make_portable_path(reliability_target))
                        if reliability_name.endswith(".png"):
                            rewritten_plot_mapping["reliability_curve"] = make_portable_path(
                                reliability_target
                            )

                summary_payload["plot_paths"] = rewritten_plot_mapping

            for reliability_name in ("reliability_curve.csv", "reliability_curve.png"):
                reliability_source = calibration_summary_source.parent / reliability_name
                reliability_target = calibration_dir / reliability_name
                if _copy_file_if_exists(reliability_source, reliability_target):
                    copied_files.append(make_portable_path(reliability_target))

            summary_payload["checkpoint_path"] = make_portable_path(checkpoint_target)
            parquet_path_value = summary_payload.get("parquet_path")
            if isinstance(parquet_path_value, str) and parquet_path_value.strip():
                summary_payload["parquet_path"] = make_portable_path(parquet_path_value)
            _save_json(summary_target, summary_payload)

            copied_files.append(make_portable_path(summary_target))
            if isinstance(threshold_metadata, dict):
                threshold_metadata["calibration_summary_path"] = make_portable_path(summary_target)

    if args.include_experiment_comparison:
        comparison_source = resolve_runtime_path(
            Path("artifacts/analysis/experiments/sequence_experiment_comparison.csv")
        )
        comparison_target = (
            output_root / "analysis" / "experiments" / "sequence_experiment_comparison.csv"
        )
        if _write_public_experiment_comparison(comparison_source, comparison_target):
            copied_files.append(make_portable_path(comparison_target))

    demo_target = output_root / "demo" / "sequence_demo_samples.parquet"
    demo_candidates = [
        resolve_runtime_path(Path("assets/demo/sequence_demo_samples.parquet")),
        resolve_runtime_path(Path("data/demo/sequence_demo_samples.parquet")),
    ]
    for demo_candidate in demo_candidates:
        if _copy_file_if_exists(demo_candidate, demo_target):
            copied_files.append(make_portable_path(demo_target))
            break

    walkthrough_target = output_root / "demo" / "saved_example_payload.json"
    walkthrough_candidates = [
        resolve_runtime_path(Path("assets/demo/saved_example_payload.json")),
    ]
    for walkthrough_candidate in walkthrough_candidates:
        if _copy_file_if_exists(walkthrough_candidate, walkthrough_target):
            copied_files.append(make_portable_path(walkthrough_target))
            break

    operational_subset_target = output_root / "demo" / "operational_windows_subset.parquet"
    operational_subset_candidates = [
        resolve_runtime_path(Path("assets/demo/operational_windows_subset.parquet")),
    ]
    for operational_subset_candidate in operational_subset_candidates:
        if _copy_file_if_exists(operational_subset_candidate, operational_subset_target):
            copied_files.append(make_portable_path(operational_subset_target))
            break

    explainability_candidates = [
        resolve_runtime_path(Path("artifacts/analysis/explainability/feature_importance.csv")),
        resolve_runtime_path(Path("artifacts/analysis/explainability/feature_importance.json")),
        resolve_runtime_path(Path("assets/demo/feature_importance.csv")),
        resolve_runtime_path(Path("assets/demo/feature_importance.json")),
    ]
    for explainability_candidate in explainability_candidates:
        if not explainability_candidate.exists():
            continue

        explainability_target = (
            output_root
            / "analysis"
            / "explainability"
            / explainability_candidate.name
        )
        if _copy_file_if_exists(explainability_candidate, explainability_target):
            copied_files.append(make_portable_path(explainability_target))
            break

    save_model_manifest(output_manifest_path, rewritten_manifest)
    copied_files.append(make_portable_path(output_manifest_path))

    print(
        json.dumps(
            {
                "manifest_source": make_portable_path(manifest_path),
                "output_root": make_portable_path(output_root),
                "copied_file_count": len(copied_files),
                "copied_files": sorted(set(copied_files)),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
