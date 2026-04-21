# Early Sepsis Detection Platform

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Stack](https://img.shields.io/badge/stack-Streamlit%20%7C%20FastAPI%20%7C%20PyTorch-0b7285.svg)
![License](https://img.shields.io/badge/license-MIT-informational.svg)

Production-style machine learning repository for early sepsis risk prediction from ICU time-series data, with deterministic preprocessing, sequence modeling, manifest-based serving, and deployment-safe public demo packaging.

## 1. Executive Summary

This repository implements an end-to-end workflow for early sepsis detection:

- schema-aware ingestion for official PhysioNet PSV files and local/Kaggle-style CSV mirrors
- deterministic patient-level splitting and leakage-resistant preprocessing
- sliding window generation for sequence learning
- GRU, LSTM, and PatchTST-style sequence classifiers
- experiment comparison, calibration analysis, and manifest-backed threshold modes
- FastAPI inference endpoints and a public-safe Streamlit presentation app

Real clinical source data is intentionally excluded from version control.

## 2. Clinical Scope and Safety Boundaries

- Intended use: research, engineering validation, and portfolio demonstration
- Not intended use: direct clinical decision support
- Demo output is explanatory and operationally oriented, not treatment guidance
- Public deployment paths sanitize sensitive filesystem details when environment is non-development

## 3. Implemented Capabilities

- Data ingestion and validation: [src/early_sepsis/data/ingestion.py](src/early_sepsis/data/ingestion.py)
- Deterministic split and preprocessing: [src/early_sepsis/data/pipeline.py](src/early_sepsis/data/pipeline.py)
- Window generation: [src/early_sepsis/data/windowing.py](src/early_sepsis/data/windowing.py)
- Sequence models: [src/early_sepsis/modeling/sequence_models.py](src/early_sepsis/modeling/sequence_models.py)
- Sequence training/evaluation/prediction: [src/early_sepsis/modeling/sequence_pipeline.py](src/early_sepsis/modeling/sequence_pipeline.py)
- Experiment analysis and calibration: [src/early_sepsis/modeling/experiment_analysis.py](src/early_sepsis/modeling/experiment_analysis.py)
- Selected model manifest management: [src/early_sepsis/modeling/model_manifest.py](src/early_sepsis/modeling/model_manifest.py)
- API serving: [src/early_sepsis/serving/api.py](src/early_sepsis/serving/api.py)
- Streamlit demo and startup checks: [src/early_sepsis/demo/app.py](src/early_sepsis/demo/app.py), [src/early_sepsis/demo/startup.py](src/early_sepsis/demo/startup.py)

## 4. Architecture

| Layer | Primary modules | Purpose |
|---|---|---|
| Ingestion | [src/early_sepsis/data/ingestion.py](src/early_sepsis/data/ingestion.py) | Detect format, normalize schema aliases, validate rows/files |
| Feature pipeline | [src/early_sepsis/data/preprocessing.py](src/early_sepsis/data/preprocessing.py) | Train-only stats, imputation, scaling, masks |
| Temporal dataset | [src/early_sepsis/data/windowing.py](src/early_sepsis/data/windowing.py) | Build label-aware windows with configurable horizon |
| Modeling | [src/early_sepsis/modeling/sequence_models.py](src/early_sepsis/modeling/sequence_models.py) | GRU/LSTM/PatchTST sequence classifiers |
| Training + metrics | [src/early_sepsis/modeling/sequence_pipeline.py](src/early_sepsis/modeling/sequence_pipeline.py), [src/early_sepsis/modeling/sequence_metrics.py](src/early_sepsis/modeling/sequence_metrics.py) | Fit, evaluate, threshold selection, checkpointing |
| Registry + selection | [src/early_sepsis/modeling/model_manifest.py](src/early_sepsis/modeling/model_manifest.py) | Portable selected model manifest |
| Serving | [src/early_sepsis/serving/sequence_service.py](src/early_sepsis/serving/sequence_service.py), [src/early_sepsis/serving/api.py](src/early_sepsis/serving/api.py) | Manifest-backed inference with dataset and shape guardrails |
| Presentation | [src/early_sepsis/demo/app.py](src/early_sepsis/demo/app.py) | Public-safe dashboard with artifact-backed visuals |

## 5. End-to-End Workflow

1. Validate and ingest raw data.
2. Split patients into train/validation/test cohorts.
3. Preprocess splits using train-derived statistics only.
4. Build temporal windows and labels.
5. Train sequence model(s) and export checkpoints.
6. Compare runs and select best checkpoint into manifest.
7. Analyze calibration and synchronize threshold modes.
8. Serve through FastAPI and/or present through Streamlit.
9. Build compact public artifact bundle for deployment.

## 6. Supported Data Inputs

### PhysioNet-style PSV directory

```text
data/raw/
├── patient_0001.psv
├── patient_0002.psv
└── ...
```

### Local/Kaggle-style CSV

```text
data/local_csv/
└── sepsis_data.csv
```

Requirements enforced by ingestion:

- target must map to `SepsisLabel` (alias handling is implemented)
- patient identifier aliases are supported (falls back to file stem when absent)
- time aliases are supported (falls back to row order when absent)
- malformed rows are dropped with warning; strict mode can fail fast

Synthetic test/demo assets are included under [tests/fixtures](tests/fixtures), [assets/demo](assets/demo), and [data/demo](data/demo). Restricted source clinical data is not committed.

## 7. Local Environment Setup

Run from repository root.

Recommended (uv):

```powershell
uv python install 3.12
uv sync --extra dev
```

Alternative (venv + pip):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

## 8. Configuration and Runtime Settings

Base settings class: [src/early_sepsis/settings.py](src/early_sepsis/settings.py)

Environment template: [.env.example](.env.example)

Common runtime variables:

- `SEPSIS_ENVIRONMENT`
- `SEPSIS_SELECTED_SEQUENCE_MANIFEST_PATH`
- `SEPSIS_PUBLIC_ARTIFACTS_DIR`
- `SEPSIS_SERVING_DEFAULT_OPERATING_MODE`
- `SEPSIS_DEMO_PUBLIC_MODE`
- `SEPSIS_MODEL_ARTIFACT_PATH`
- `SEPSIS_API_HOST`
- `SEPSIS_API_PORT`

Config files:

- [configs/data_pipeline.yaml](configs/data_pipeline.yaml)
- [configs/model_training.yaml](configs/model_training.yaml)
- [configs/model_tuning.yaml](configs/model_tuning.yaml)
- [configs/api.yaml](configs/api.yaml)
- [configs/orchestration.yaml](configs/orchestration.yaml)

## 9. Quickstart with Synthetic Data

Generate synthetic ICU data, preprocess, window, and run a short training smoke flow:

```powershell
uv run python scripts/generate_synthetic_data.py --output-path tests/fixtures/generated_synthetic.csv --dataset-format csv --patient-count 24 --min-hours 10 --max-hours 20 --seed 42
uv run python scripts/preprocess_data.py --raw-path tests/fixtures/generated_synthetic.csv --dataset-format csv --output-dir artifacts/processed_cli_smoke --strict
uv run python scripts/create_windows.py --processed-dir artifacts/processed_cli_smoke --output-dir artifacts/windows_cli_smoke --window-length 8 --prediction-horizon 6
uv run python scripts/train_sequence.py --windows-dir artifacts/windows_cli_smoke --output-dir artifacts/models/sequence_cli_smoke --model-type gru --epochs 2 --batch-size 64 --disable-mlflow
```

## 10. Raw Data Validation and Preprocessing

Validate source files:

```powershell
uv run python scripts/validate_raw_data.py --raw-path data/raw --dataset-format auto
uv run python scripts/validate_raw_data.py --raw-path data/raw --dataset-format auto --strict
```

Run deterministic preprocessing pipeline:

```powershell
uv run python scripts/preprocess_data.py --config configs/data_pipeline.yaml --raw-path data/raw --dataset-format auto --output-dir artifacts/processed
uv run python scripts/print_split_summary.py --processed-dir artifacts/processed
```

## 11. Window Dataset Generation

Create train/validation/test window parquet datasets:

```powershell
uv run python scripts/create_windows.py --processed-dir artifacts/processed --output-dir artifacts/windows --window-length 8 --prediction-horizon 6
```

Optional toggles are available in CLI for masks/static features and padding mode.

## 12. Sequence Modeling: Train, Evaluate, Predict

Train sequence model:

```powershell
uv run python scripts/train_sequence.py --config configs/model_training.yaml --windows-dir artifacts/windows --output-dir artifacts/models/sequence --model-type patchtst
```

Resolve latest checkpoint and evaluate:

```powershell
$CHECKPOINT = (Get-ChildItem artifacts/models/sequence -Recurse -Filter best_checkpoint.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
uv run python scripts/evaluate_sequence.py --checkpoint-path $CHECKPOINT --windows-dir artifacts/windows --split test
```

Generate sequence prediction parquet:

```powershell
uv run python scripts/predict_sequence.py --checkpoint-path $CHECKPOINT --parquet-path artifacts/windows/test.parquet --output-path artifacts/predictions/sequence_predictions.parquet
```

Tabular baseline path (used by `/predict` endpoint):

```powershell
uv run python scripts/train_local.py --data-path tests/fixtures/synthetic_tabular.csv --dataset-format csv
```

## 13. Hyperparameter Tuning and Experiment Comparison

Run Optuna tuning:

```powershell
uv run python scripts/tune_sequence.py --config configs/model_tuning.yaml --windows-dir artifacts/windows --model-type patchtst --n-trials 20
```

Aggregate experiments:

```powershell
uv run python scripts/compare_experiments.py --model-root artifacts/models --output-dir artifacts/analysis/experiments
```

## 14. Selected Model Manifest and Calibration Workflow

Select best run into manifest:

```powershell
uv run python scripts/select_best_model.py --model-root artifacts/models --selection-metric validation_auprc --dataset-tag physionet --manifest-path artifacts/models/registry/selected_model.json
```

Run calibration analysis and synchronize thresholds:

```powershell
uv run python scripts/analyze_calibration.py --manifest-path artifacts/models/registry/selected_model.json --split validation --output-dir artifacts/analysis/calibration --high-recall-target 0.90 --sync-manifest-thresholds
```

Manual threshold synchronization and path normalization:

```powershell
uv run python scripts/sync_manifest_thresholds.py --manifest-path artifacts/models/registry/selected_model.json --recommendations-path artifacts/analysis/calibration/threshold_recommendations.json --summary-path artifacts/analysis/calibration/calibration_summary.json
uv run python scripts/migrate_manifest_paths.py --manifest-path artifacts/models/registry/selected_model.json
```

Threshold operating modes currently implemented across serving and demo:

- `default`
- `balanced`
- `high_recall`

Evaluation interpretation in this project:

- Threshold-invariant metrics: AUROC, AUPRC, Brier score, Expected Calibration Error, prevalence
- Threshold-dependent outputs: predicted labels, confusion matrix counts, precision, recall, F1, alert rate

## 15. FastAPI Serving and Inference Contracts

Start API:

```powershell
uv run python scripts/serve_api.py
```

Health and model metadata:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/model-info" | ConvertTo-Json -Depth 8
```

Tabular inference request (`/predict`):

```powershell
$records = Get-Content tests/fixtures/synthetic_records.json -Raw | ConvertFrom-Json
$body = @{ records = $records; include_explanation = $false } | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8
```

Sequence inference request (`/predict-sequence`):

```powershell
uv run python -c "import json,httpx,pandas as pd; m=json.load(open('artifacts/models/registry/selected_model.json','r',encoding='utf-8')); tag=m['dataset']['dataset_tag']; df=pd.read_parquet(f\"{m['dataset']['windows_dir']}/validation.parquet\"); r=df.iloc[0]; payload={'dataset_tag':tag,'operating_mode':'balanced','samples':[{'patient_id':r['patient_id'],'end_hour':int(r['end_hour']),'features':r['features'].tolist(),'missing_mask':r['missing_mask'].tolist() if r['missing_mask'] is not None else None,'static_features':r['static_features'].tolist() if r['static_features'] is not None else None}]}; resp=httpx.post('http://127.0.0.1:8000/predict-sequence',json=payload,timeout=60.0); print(resp.status_code); print(resp.text)"
```

## 16. Streamlit Demo Behavior

Start demo:

```powershell
uv run streamlit run streamlit_app.py
```

Equivalent helper script:

```powershell
uv run python scripts/run_demo.py
```

Demo behavior is artifact-backed and public-safe:

- selected model loaded from manifest
- threshold mode selector maps to manifest thresholds
- threshold-invariant and threshold-dependent outputs are separated in presentation
- calibration reliability fallback chart is sanitized and bounded to [0, 1]
- sensitive local paths are not exposed when environment is non-development

Inference source order:

- Public mode: bundled demo parquet, then evaluation split parquet, then saved walkthrough payload
- Non-public mode: evaluation split parquet, then bundled demo parquet, then saved walkthrough payload

Operational summary source order:

- `public_artifacts/demo/operational_windows_subset.parquet`
- `assets/demo/operational_windows_subset.parquet`
- `<manifest.dataset.windows_dir>/<split>.parquet`
- current inference parquet source (if available)

## 17. Deployment-Safe Artifact Bundling

Curate compact demo windows and operational subset:

```powershell
uv run python scripts/curate_demo_assets.py --manifest-path artifacts/models/registry/selected_model.json --candidate-rows-per-source 3000 --demo-count 36 --operational-count 600
```

Audit demo-score diversity and display mapping:

```powershell
uv run python scripts/audit_demo_inference.py --manifest-path artifacts/models/registry/selected_model.json --parquet-path assets/demo/sequence_demo_samples.parquet --display-round-decimals 6
```

Build compact public bundle:

```powershell
uv run python scripts/prepare_public_artifacts.py --manifest-path artifacts/models/registry/selected_model.json --output-dir public_artifacts
```

Required deployment files are documented in [public_artifacts/README.md](public_artifacts/README.md).

## 18. Deployment Options

### Streamlit Community Cloud

Entrypoint and runtime files:

- [streamlit_app.py](streamlit_app.py)
- [requirements.txt](requirements.txt)
- [runtime.txt](runtime.txt)

Recommended secrets block:

```toml
[sepsis]
environment = "production"
demo_public_mode = true
selected_sequence_manifest_path = "public_artifacts/models/registry/selected_model.json"
public_artifacts_dir = "public_artifacts"
demo_sample_parquet_path = "public_artifacts/demo/sequence_demo_samples.parquet"
public_repo_url = "https://github.com/<owner>/<repo>"
```

### Streamlit Docker container

```powershell
docker build -t early-sepsis-streamlit .
docker run --rm -p 8501:8501 -e PORT=8501 early-sepsis-streamlit
```

### API Docker container

```powershell
docker compose -f docker/docker-compose.yml up --build api
```

## 19. Testing and Quality Gates

Run full test suite:

```powershell
uv run pytest -q
```

Run targeted suites used by serving/demo paths:

```powershell
uv run pytest tests/test_serving_sequence.py tests/test_demo_presentation.py tests/test_demo_thresholds.py -q
```

Static checks:

```powershell
uv run ruff check .
uv run mypy src
```

## 20. Limitations, Improvement Backlog, and License

Current limitations:

- Research implementation only; not a clinical decision-support product
- Sequence performance and threshold recommendations are artifact-dependent and dataset-dependent
- Public demo prioritizes portability and safety over full-fidelity offline evaluation scale
- Training and tuning can be compute-intensive on CPU-only systems

Practical improvement backlog:

- add automated drift monitoring jobs for post-training score distribution tracking
- add richer model comparison visualization overlays for operating mode review
- harden deployment CI for artifact integrity checks before release

License note:

- `pyproject.toml` declares MIT license metadata
- a standalone `LICENSE` file is not currently committed
