from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from early_sepsis.modeling.sequence_pipeline import predict_from_checkpoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sequence-model predictions for window parquet data")
    parser.add_argument("--checkpoint-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--parquet-path", required=True, help="Input window parquet file")
    parser.add_argument(
        "--output-path",
        default="artifacts/predictions/sequence_predictions.parquet",
        help="Output parquet or csv path",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    predictions = predict_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "input_path": str(args.parquet_path),
        "output_path": str(args.output_path),
        "row_count": int(len(predictions)),
        "predicted_positive_rate": float(predictions["predicted_label"].mean()),
        "mean_predicted_probability": float(predictions["predicted_probability"].mean()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
