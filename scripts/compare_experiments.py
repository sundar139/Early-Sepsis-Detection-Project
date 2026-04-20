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

from early_sepsis.modeling.experiment_analysis import (
    aggregate_sequence_experiments,
    export_experiment_comparison,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare sequence experiments and export summary artifacts"
    )
    parser.add_argument(
        "--model-root", default="artifacts/models", help="Root directory for model run artifacts"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/experiments",
        help="Directory for exported comparison outputs",
    )
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Optional dataset tag filter (e.g., physionet or kaggle_csv)",
    )
    parser.add_argument(
        "--csv-name",
        default="sequence_experiment_comparison.csv",
        help="CSV filename for the tabular summary",
    )
    parser.add_argument(
        "--report-name",
        default="sequence_experiment_report.md",
        help="Markdown filename for compact report",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    frame = aggregate_sequence_experiments(
        model_root=args.model_root,
        project_root=PROJECT_ROOT,
    )

    if args.dataset_tag:
        frame = frame.loc[frame["dataset_tag"] == args.dataset_tag].reset_index(drop=True)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / args.csv_name
    report_path = output_dir / args.report_name

    resolved_csv_path, resolved_report_path = export_experiment_comparison(
        frame,
        csv_path=csv_path,
        markdown_path=report_path,
    )

    payload: dict[str, object] = {
        "run_count": len(frame),
        "csv_path": str(resolved_csv_path),
        "report_path": str(resolved_report_path),
    }

    if not frame.empty:
        best = frame.iloc[0]
        payload["best_run"] = {
            "run_name": str(best["run_name"]),
            "model_type": str(best["model_type"]),
            "model_family": str(best.get("model_family", "")),
            "dataset_tag": str(best["dataset_tag"]),
            "validation_auprc": float(best["validation_auprc"]),
            "validation_auroc": float(best["validation_auroc"]),
            "checkpoint_path": str(best["checkpoint_path"]),
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
