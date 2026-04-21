# Public Artifacts Bundle

This directory is the compact artifact bundle for public Streamlit deployment.

Required files for model-backed demo inference:

- models/registry/selected_model.json
- models/checkpoints/best_checkpoint.pt

Optional files for richer artifact-backed presentation:

- analysis/calibration/calibration_summary.json
- analysis/calibration/threshold_recommendations.json
- analysis/calibration/roc_curve.png
- analysis/calibration/pr_curve.png
- analysis/calibration/confusion_matrix.png
- analysis/calibration/score_distribution.png
- analysis/calibration/reliability_curve.csv
- analysis/experiments/sequence_experiment_comparison.csv
- demo/sequence_demo_samples.parquet
- demo/saved_example_payload.json

Notes:

- Do not include secrets in this directory.
- Do not include restricted clinical source data.
- Keep only files required for the public demo.
