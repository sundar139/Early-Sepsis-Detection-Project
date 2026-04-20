from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.data.pipeline import build_split_summary, load_pipeline_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print processed split summary")
    parser.add_argument(
        "--processed-dir",
        default="artifacts/processed",
        help="Directory containing processed train/validation/test parquet files",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    metadata = load_pipeline_metadata(args.processed_dir)
    split_summary = build_split_summary(args.processed_dir)

    report = {
        "processed_dir": args.processed_dir,
        "feature_count": len(metadata.get("feature_columns", [])),
        "mask_count": len(metadata.get("mask_columns", [])),
        "static_feature_count": len(metadata.get("static_feature_columns", [])),
        "split_summary": split_summary,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
