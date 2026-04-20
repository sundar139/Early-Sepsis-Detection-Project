from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.modeling.sequence_pipeline import evaluate_checkpoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a sequence-model checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--parquet-path", default=None, help="Specific window parquet split path")
    parser.add_argument("--windows-dir", default="artifacts/windows", help="Window dataset directory")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Split to evaluate when parquet-path is not supplied",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-json", default=None, help="Optional output path for metrics JSON")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    parquet_path = Path(args.parquet_path) if args.parquet_path else Path(args.windows_dir) / f"{args.split}.parquet"
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        parquet_path=parquet_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "parquet_path": str(parquet_path),
        "loss": result["loss"],
        "threshold": result["threshold"],
        "metrics": result["metrics"],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
