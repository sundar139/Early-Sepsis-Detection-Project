# Early Sepsis Detection System

A production-style Python repository for early sepsis detection with a deterministic data engineering pipeline for irregular ICU time-series data.

## Project Overview

The repository supports:

- schema-aware ingestion from official PhysioNet-style PSV files or CSV mirror files
- deterministic patient-level train/validation/test splitting
- time-series preprocessing with missingness masks, forward fill, train-only median imputation, and train-only normalization
- sliding-window generation with configurable window length and prediction horizon
- parquet-based processed datasets with metadata and feature schema files
- PyTorch-ready dataset and dataloader utilities
- optional model training, API serving, orchestration, experiment tracking, and local explanation workflows

Real clinical source data is not stored in this repository.

## Expected Raw-Data Layout

Use one of these two layouts under a local directory.

PhysioNet-style directory:

```text
data/raw/
|-- patient_0001.psv
|-- patient_0002.psv
`-- ...
```

Each PSV file is pipe-delimited. `patient_id` is inferred from the file name when not present.

CSV mirror layout:

```text
data/local_csv/
`-- sepsis_data.csv
```

The CSV mirror must include `SepsisLabel` and either:

- a `patient_id` column for multi-patient files, or
- one patient per file when no patient column exists

If a time column equivalent to `ICULOS` is missing, row order is used as an hourly fallback and logged.

## Architecture Summary

- [src/early_sepsis/data/ingestion.py](src/early_sepsis/data/ingestion.py): file discovery, schema normalization, validation diagnostics
- [src/early_sepsis/data/splitting.py](src/early_sepsis/data/splitting.py): deterministic patient-level split manifests
- [src/early_sepsis/data/preprocessing.py](src/early_sepsis/data/preprocessing.py): sorting, monotonic enforcement, masks, forward fill, train-fit imputation and standardization
- [src/early_sepsis/data/windowing.py](src/early_sepsis/data/windowing.py): sliding windows and onset-horizon labels
- [src/early_sepsis/data/pipeline.py](src/early_sepsis/data/pipeline.py): end-to-end preprocessing and window artifact orchestration
- [src/early_sepsis/data/torch_dataset.py](src/early_sepsis/data/torch_dataset.py): PyTorch dataset and dataloader wrappers
- [src/early_sepsis/data/synthetic.py](src/early_sepsis/data/synthetic.py): deterministic synthetic ICU data generator
- [src/early_sepsis/modeling/sequence_models.py](src/early_sepsis/modeling/sequence_models.py): GRU/LSTM baseline and PatchTST-style classifier implementations
- [src/early_sepsis/modeling/sequence_pipeline.py](src/early_sepsis/modeling/sequence_pipeline.py): training, checkpointing, evaluation, prediction, class imbalance handling, and MLflow integration
- [src/early_sepsis/modeling/sequence_metrics.py](src/early_sepsis/modeling/sequence_metrics.py): AUROC, AUPRC, thresholded metrics, confusion matrix, and calibration metrics
- [src/early_sepsis/modeling/sequence_tuning.py](src/early_sepsis/modeling/sequence_tuning.py): Optuna tuning with validation AUPRC objective and pruning
- [src/early_sepsis/modeling/experiment_analysis.py](src/early_sepsis/modeling/experiment_analysis.py): experiment aggregation, report export, threshold sweep, calibration summaries, and evaluation plots
- [src/early_sepsis/modeling/model_manifest.py](src/early_sepsis/modeling/model_manifest.py): selected-model manifest schema, validation, and threshold updates
- [src/early_sepsis/serving/sequence_service.py](src/early_sepsis/serving/sequence_service.py): selected-checkpoint loading, dataset-tag guardrails, dimension checks, and sequence inference runtime

## Setup

1. Install Python 3.12 and uv.
2. Install dependencies.
3. Run checks.

```bash
uv python install 3.12
uv sync --extra dev
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Data Engineering Commands

Generate synthetic input data for local smoke runs:

```bash
uv run python scripts/generate_synthetic_data.py --output-path tests/fixtures/generated_synthetic.csv --dataset-format csv
```

Validate raw data and schema:

```bash
uv run python scripts/validate_raw_data.py --raw-path data/raw --dataset-format auto
```

Strict validation mode fails on any invalid file:

```bash
uv run python scripts/validate_raw_data.py --raw-path data/raw --dataset-format auto --strict
```

Run preprocessing pipeline and save split artifacts:

```bash
uv run python scripts/preprocess_data.py --config configs/data_pipeline.yaml --raw-path data/raw --dataset-format auto
```

Create sliding windows for train/validation/test:

```bash
uv run python scripts/create_windows.py --processed-dir artifacts/processed --output-dir artifacts/windows --window-length 8 --prediction-horizon 6
```

Print split summary:

```bash
uv run python scripts/print_split_summary.py --processed-dir artifacts/processed
```

## Sequence Modeling Stack

Two sequence classifiers are implemented for windowed ICU trajectories:

- Recurrent baseline: GRU or LSTM with optional bidirectionality
- Primary model: PatchTST-style classifier with temporal patch extraction, patch embeddings, positional embeddings, transformer encoder blocks, pooled representation, and binary classification head

Training uses:

- deterministic seeding
- class imbalance handling through `pos_weight`, weighted sampler, or both
- checkpointing for best and last model states
- early stopping on validation AUPRC
- `ReduceLROnPlateau` scheduling
- SQLite-backed MLflow tracking by default (`sqlite:///mlflow.db`)

## Sequence Training Commands

Train a sequence model from window artifacts:

```bash
uv run python scripts/train_sequence.py --config configs/model_training.yaml --windows-dir artifacts/windows --model-type patchtst
```

Evaluate a saved checkpoint on one split:

```powershell
$CHECKPOINT = (Get-ChildItem artifacts/models/sequence -Recurse -Filter best_checkpoint.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
uv run python scripts/evaluate_sequence.py --checkpoint-path $CHECKPOINT --windows-dir artifacts/windows --split test
```

Generate probability predictions from a checkpoint:

```powershell
$CHECKPOINT = (Get-ChildItem artifacts/models/sequence -Recurse -Filter best_checkpoint.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
uv run python scripts/predict_sequence.py --checkpoint-path $CHECKPOINT --parquet-path artifacts/windows/test.parquet --output-path artifacts/predictions/test_predictions.parquet
```

Run Optuna tuning (validation AUPRC objective, pruning enabled):

```bash
uv run python scripts/tune_sequence.py --config configs/model_tuning.yaml --windows-dir artifacts/windows --model-type patchtst --n-trials 20
```

## Experiment Comparison, Selection, And Calibration

Export a compact cross-run comparison table (CSV + Markdown):

```bash
uv run python scripts/compare_experiments.py --model-root artifacts/models --output-dir artifacts/analysis/experiments
```

Select the best checkpoint and persist a registry-style selected-model manifest:

```bash
uv run python scripts/select_best_model.py --model-root artifacts/models --selection-metric validation_auprc --dataset-tag physionet --manifest-path artifacts/models/registry/selected_model.json
```

Selected-manifest path fields are stored as repository-relative paths when possible to keep artifacts portable across machines.

Analyze calibration and threshold sweep for the selected checkpoint, then update manifest thresholds:

```bash
uv run python scripts/analyze_calibration.py --manifest-path artifacts/models/registry/selected_model.json --split validation --output-dir artifacts/analysis/calibration --high-recall-target 0.90 --sync-manifest-thresholds
```

Synchronize manifest thresholds from existing calibration artifacts without rerunning calibration:

```bash
uv run python scripts/sync_manifest_thresholds.py --manifest-path artifacts/models/registry/selected_model.json --recommendations-path artifacts/analysis/calibration/threshold_recommendations.json --summary-path artifacts/analysis/calibration/calibration_summary.json
```

Rewrite existing selected-manifest and calibration metadata paths to portable repository-relative form:

```bash
uv run python scripts/migrate_manifest_paths.py --manifest-path artifacts/models/registry/selected_model.json
```

Threshold operating modes supported across manifest-backed serving:

- `default`: baseline operating threshold stored in selected manifest
- `balanced`: balanced operating threshold stored in selected manifest
- `high_recall`: high-recall operating threshold stored in selected manifest

Calibration-to-manifest synchronization is safety-checked by matching the calibration summary checkpoint
to the selected manifest checkpoint before applying thresholds.

Inspect selected model metadata from the API (after starting server):

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/model-info" | ConvertTo-Json -Depth 8
```

`/model-info` includes available threshold modes and configured default serving mode.
In non-development environments, filesystem paths are sanitized in `/health` and `/model-info` payloads.

Run manifest-backed sequence inference through API with dataset guardrails:

```bash
uv run python -c "import json, httpx, pandas as pd; m=json.load(open('artifacts/models/registry/selected_model.json','r',encoding='utf-8')); tag=m['dataset']['dataset_tag']; split='validation'; df=pd.read_parquet(f\"{m['dataset']['windows_dir']}/{split}.parquet\"); r=df.iloc[0]; payload={'dataset_tag': tag, 'operating_mode': 'balanced', 'samples': [{'patient_id': r['patient_id'], 'end_hour': int(r['end_hour']), 'features': r['features'].tolist(), 'missing_mask': r['missing_mask'].tolist() if r['missing_mask'] is not None else None, 'static_features': r['static_features'].tolist() if r['static_features'] is not None else None}]}; resp=httpx.post('http://127.0.0.1:8000/predict-sequence', json=payload, timeout=60.0); print(resp.status_code); print(resp.text)"
```

Run manifest-backed sequence inference with an explicit request threshold override:

```bash
uv run python -c "import json, httpx, pandas as pd; m=json.load(open('artifacts/models/registry/selected_model.json','r',encoding='utf-8')); tag=m['dataset']['dataset_tag']; split='validation'; df=pd.read_parquet(f\"{m['dataset']['windows_dir']}/{split}.parquet\"); r=df.iloc[0]; payload={'dataset_tag': tag, 'threshold': 0.20, 'samples': [{'patient_id': r['patient_id'], 'end_hour': int(r['end_hour']), 'features': r['features'].tolist(), 'missing_mask': r['missing_mask'].tolist() if r['missing_mask'] is not None else None, 'static_features': r['static_features'].tolist() if r['static_features'] is not None else None}]}; resp=httpx.post('http://127.0.0.1:8000/predict-sequence', json=payload, timeout=60.0); print(resp.status_code); print(resp.text)"
```

## Output Files

Preprocessing outputs in `artifacts/processed/`:

- `train.parquet`, `validation.parquet`, `test.parquet`: row-level processed split files
- `metadata.json`: preprocessing statistics, ingestion diagnostics, split summary, and provenance
- `feature_schema.json`: feature list, static-feature flags, and missing-mask mappings
- `split_manifests/patient_split_assignments.csv`: patient-to-split mapping
- `split_manifests/train_patients.csv`, `validation_patients.csv`, `test_patients.csv`

Window outputs in `artifacts/windows/`:

- `train.parquet`, `validation.parquet`, `test.parquet`: window-level datasets with labels
- `metadata.json`: window generation configuration and counts
- `feature_schema.json`: feature schema used by window artifacts

Sequence-model outputs in `artifacts/models/sequence/<run_name>/`:

- `best_checkpoint.pt`: checkpoint selected by best validation AUPRC
- `last_checkpoint.pt`: final epoch checkpoint
- `run_config.json`: resolved model and training config
- `training_history.json`: epoch-level losses and metrics
- `validation_metrics.json`: best validation metrics and selected threshold
- `test_metrics.json`: held-out test metrics with selected threshold

Selected-model registry output in `artifacts/models/registry/`:

- `selected_model.json`: validated selected-checkpoint manifest including dataset tag, feature signature, dimensions, thresholds, and run metadata

Experiment analysis outputs in `artifacts/analysis/experiments/`:

- `sequence_experiment_comparison.csv`: cross-run comparison table
- `sequence_experiment_report.md`: compact leaderboard report

Calibration analysis outputs in `artifacts/analysis/calibration/`:

- `threshold_sweep.csv`: threshold-level precision/recall/F1 operating table
- `reliability_curve.csv`: bin-wise calibration summary
- `threshold_recommendations.json`: default/balanced/high-recall recommendations
- `calibration_summary.json`: calibration run metadata and key metrics
- `calibration_report.md`: compact human-readable report
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `score_distribution.png`: evaluation visuals

Threshold synchronization output in selected manifest:

- `threshold_metadata` in `artifacts/models/registry/selected_model.json`: threshold source,
  recommendation artifact path, summary artifact path, and synchronization timestamp

Optuna tuning outputs in `artifacts/models/sequence_tuning/optuna/`:

- `best_result.json`: best trial summary and chosen hyperparameters
- `study_trials.json`: all trials with parameters, states, and values

## Notes On Synthetic And Real Data

- Only tiny synthetic data is intended for tests and demo runs.
- Real source datasets must remain outside version control.
- Place licensed or restricted datasets under local paths such as `data/raw` or `data/local_csv`.
- Generated artifacts and local raw data paths are ignored via [.gitignore](.gitignore).

## Existing Model And Service Commands

Train tabular baseline model on synthetic CSV:

```bash
uv run python scripts/train_local.py --data-path tests/fixtures/synthetic_tabular.csv --dataset-format csv
```

Start API server:

```bash
uv run python scripts/serve_api.py
```

Quick health and selected-model checks:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/model-info" | ConvertTo-Json -Depth 8
```

`/health` reports selected sequence manifest and checkpoint availability as the primary serving readiness signal.

Run Streamlit demo:

```bash
uv run streamlit run streamlit_app.py
```

The Streamlit demo uses the selected operating mode (`default`, `balanced`, `high_recall`) and
displays the resolved threshold before running predictions.

Threshold-invariant model metrics (AUROC, AUPRC, Brier score, calibration error) remain static
across operating modes, while threshold-dependent operational outputs (alert rate, confusion matrix,
precision/recall/specificity, and predicted classifications) update with the selected mode.

If two operating modes map to the same numeric threshold in selected artifacts, the UI reports that
those modes are currently equivalent for operational outputs.

The Streamlit demo runs inference directly against the selected checkpoint and does not require the
local FastAPI service to be running.

The deployed UI is presentation-safe by default: no raw manifest dumps, no checkpoint/parquet path
exposure, and no machine-specific absolute paths are rendered.

When `SEPSIS_DEMO_PUBLIC_MODE=true` (or `SEPSIS_ENVIRONMENT` is non-development), inference fallback
order is:

1. `assets/demo/sequence_demo_samples.parquet`
2. `<manifest.dataset.windows_dir>/<split>.parquet`
3. `assets/demo/saved_example_payload.json` (single-sample saved walkthrough)

If none of these sources is available, the inference section remains visible with a polished
artifact-unavailable card instead of a runtime traceback.

## Deploy

Primary target: Streamlit Community Cloud.

Deployment entrypoint and runtime:

- App file: `streamlit_app.py`
- Dependencies: `requirements.txt`
- Python runtime: `runtime.txt`

### Local Validation

```bash
uv run streamlit run streamlit_app.py
```

### Compact Public Artifacts

Use `public_artifacts/` for deployment-safe model artifacts.

Required for model-backed demo inference:

- `public_artifacts/models/registry/selected_model.json`
- `public_artifacts/models/checkpoints/best_checkpoint.pt`

Optional for richer UI:

- `public_artifacts/analysis/calibration/calibration_summary.json`
- `public_artifacts/analysis/calibration/threshold_recommendations.json`
- calibration plot images and reliability CSV under `public_artifacts/analysis/calibration/`
- `public_artifacts/analysis/experiments/sequence_experiment_comparison.csv`
- `public_artifacts/demo/sequence_demo_samples.parquet`
- `public_artifacts/demo/saved_example_payload.json`

Recommended repository assets for public fallback order:

- `assets/demo/sequence_demo_samples.parquet`
- `assets/demo/saved_example_payload.json`

Build a compact bundle from local selected artifacts:

```bash
uv run python scripts/prepare_public_artifacts.py --manifest-path artifacts/models/registry/selected_model.json --output-dir public_artifacts
```

The bundle script copies demo fallback assets in this order when available:

1. `assets/demo/sequence_demo_samples.parquet`
2. `data/demo/sequence_demo_samples.parquet`
3. `assets/demo/saved_example_payload.json`

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app and set main file path to `streamlit_app.py`.
3. In app settings, add secrets in TOML format:

```toml
[sepsis]
environment = "production"
demo_public_mode = true
selected_sequence_manifest_path = "public_artifacts/models/registry/selected_model.json"
public_artifacts_dir = "public_artifacts"
demo_sample_parquet_path = "public_artifacts/demo/sequence_demo_samples.parquet"
public_repo_url = "https://github.com/<owner>/<repo>"
```

The entrypoint maps these values to the existing `SEPSIS_*` settings at runtime. Do not commit
`.streamlit/secrets.toml`.

### Troubleshooting

- `Selected sequence manifest was not found`: confirm `public_artifacts/models/registry/selected_model.json` exists.
- `Selected checkpoint file is missing`: confirm manifest `selected_run.checkpoint_path` points to a file inside repo.
- Missing visuals: include calibration artifacts in `public_artifacts/analysis/calibration/`; the app will show a safe unavailable state otherwise.
- Inference section shows walkthrough-only mode: add `assets/demo/sequence_demo_samples.parquet` for live multi-window inference, or keep `assets/demo/saved_example_payload.json` for single-sample fallback.
- Import errors in cloud: ensure requirements install completed successfully and Python version matches `runtime.txt`.

Print/PDF note:

- The app controls print styling, but browser headers/footers (URL/date/page number) are browser-managed. Disable "Headers and footers" in the print dialog for clean report exports.

### Container Fallback

If full repository artifact volume is too large for cloud hosting workflows, use the included Streamlit Docker path:

```bash
docker build -t early-sepsis-streamlit .
docker run --rm -p 8501:8501 -e PORT=8501 early-sepsis-streamlit
```

The container binds Streamlit to `$PORT` and defaults to public demo mode.

## Limitations And Assumptions

- Sequence training expects pre-generated window parquet files with columns compatible with [src/early_sepsis/data/windowing.py](src/early_sepsis/data/windowing.py).
- PatchTST checkpoints assume consistent window length between training and inference.
- Selected sequence serving enforces dataset tag and dimension compatibility against `artifacts/models/registry/selected_model.json`.
- The Streamlit explanation is deterministic and heuristic; it is not causal feature attribution.
- Outputs are for research and operational validation only, not for clinical decision support.
- Hyperparameter tuning can be compute-intensive on CPU; reduce trial count and epochs for quick local iteration.
- Calibration metrics are computed from probability outputs but do not include post-hoc calibration fitting by default.
