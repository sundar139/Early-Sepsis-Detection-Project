from __future__ import annotations

import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from early_sepsis.data.torch_dataset import SepsisWindowDataset
from early_sepsis.logging_utils import get_logger
from early_sepsis.modeling.sequence_metrics import compute_binary_metrics, find_optimal_threshold
from early_sepsis.modeling.sequence_models import (
    ModelKind,
    SequenceModelConfig,
    build_sequence_model,
)

logger = get_logger(__name__)

ImbalanceStrategy = Literal["none", "pos_weight", "weighted_sampler", "both"]


@dataclass(slots=True)
class SequenceTrainingConfig:
    """Runtime settings for sequence model training."""

    windows_dir: Path = Path("artifacts/windows")
    output_dir: Path = Path("artifacts/models/sequence")
    model_name: str = "patchtst_classifier"

    seed: int = 42
    device: str = "auto"
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 0

    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    threshold: float = 0.5
    optimize_threshold: bool = True
    calibration_bins: int | None = 10

    imbalance_strategy: ImbalanceStrategy = "both"

    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "early-sepsis-sequence"
    mlflow_run_name: str | None = None


@dataclass(slots=True)
class SequenceTrainingResult:
    """Artifacts and key metrics from one sequence training run."""
    run_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_validation_metrics: dict[str, float | int | None]
    test_metrics: dict[str, float | int | None]
    selected_threshold: float


@dataclass(slots=True)
class _SplitLoaders:
    train_loader: DataLoader[dict[str, Tensor]]
    validation_loader: DataLoader[dict[str, Tensor]]
    test_loader: DataLoader[dict[str, Tensor]]
    input_dim: int
    sequence_length: int
    static_dim: int
    class_counts: dict[str, int]
    pos_weight: float | None


def sequence_model_family_name(model_type: ModelKind | str) -> str:
    """Returns canonical artifact family name for a supported sequence model type."""

    normalized_model_type = str(model_type).strip().lower()
    if normalized_model_type not in {"gru", "lstm", "patchtst"}:
        msg = f"Unsupported sequence model type for artifact naming: {model_type}"
        raise ValueError(msg)
    return f"{normalized_model_type}_classifier"


def build_sequence_run_name(
    model_type: ModelKind | str,
    *,
    started_at: datetime | None = None,
) -> str:
    """Builds canonical run directory names tied to explicit model family metadata."""

    run_started_at = started_at or datetime.now(UTC)
    family_name = sequence_model_family_name(model_type)
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    return f"{family_name}_{timestamp}"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_dataloader(
    dataset: SepsisWindowDataset,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None,
    num_workers: int,
) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )


def _build_split_loaders(
    config: SequenceTrainingConfig,
    model_config: SequenceModelConfig,
) -> _SplitLoaders:
    train_path = config.windows_dir / "train.parquet"
    validation_path = config.windows_dir / "validation.parquet"
    test_path = config.windows_dir / "test.parquet"

    train_dataset = SepsisWindowDataset(
        parquet_path=train_path,
        include_mask=model_config.include_mask,
        include_static=model_config.include_static,
    )
    validation_dataset = SepsisWindowDataset(
        parquet_path=validation_path,
        include_mask=model_config.include_mask,
        include_static=model_config.include_static,
    )
    test_dataset = SepsisWindowDataset(
        parquet_path=test_path,
        include_mask=model_config.include_mask,
        include_static=model_config.include_static,
    )

    train_labels = train_dataset.labels_numpy
    positive_count = int(train_labels.sum())
    negative_count = int(len(train_labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        msg = "Training set must contain both positive and negative samples."
        raise ValueError(msg)

    class_counts = {
        "negative": negative_count,
        "positive": positive_count,
    }

    pos_weight: float | None = None
    if config.imbalance_strategy in {"pos_weight", "both"}:
        pos_weight = float(negative_count / positive_count)

    sampler: WeightedRandomSampler | None = None
    if config.imbalance_strategy in {"weighted_sampler", "both"}:
        class_weight = {
            0: len(train_labels) / (2.0 * max(negative_count, 1)),
            1: len(train_labels) / (2.0 * max(positive_count, 1)),
        }
        sample_weights = np.array(
            [class_weight[int(label)] for label in train_labels],
            dtype=np.float64,
        )
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    first_item = train_dataset[0]
    sequence_length, input_dim = tuple(first_item["features"].shape)
    static_dim = int(first_item["static"].numel()) if model_config.include_static else 0

    logger.info(
        "Prepared sequence dataloaders",
        extra={
            "windows_dir": str(config.windows_dir),
            "sequence_length": sequence_length,
            "input_dim": input_dim,
            "static_dim": static_dim,
            "class_counts": class_counts,
            "imbalance_strategy": config.imbalance_strategy,
            "pos_weight": pos_weight,
        },
    )

    return _SplitLoaders(
        train_loader=_make_dataloader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            sampler=sampler,
            num_workers=config.num_workers,
        ),
        validation_loader=_make_dataloader(
            dataset=validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=config.num_workers,
        ),
        test_loader=_make_dataloader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=config.num_workers,
        ),
        input_dim=input_dim,
        sequence_length=sequence_length,
        static_dim=static_dim,
        class_counts=class_counts,
        pos_weight=pos_weight,
    )


def _to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    features = batch["features"].to(device=device, dtype=torch.float32)
    labels = batch["label"].to(device=device, dtype=torch.float32)
    mask = batch.get("mask")
    static = batch.get("static")

    mask_tensor = mask.to(device=device, dtype=torch.float32) if mask is not None else None
    static_tensor = static.to(device=device, dtype=torch.float32) if static is not None else None
    return features, labels, mask_tensor, static_tensor


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    probabilities: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for batch in loader:
        features, labels, mask, static = _to_device(batch=batch, device=device)

        with torch.set_grad_enabled(is_training):
            logits = model(features=features, missing_mask=mask, static_features=static)
            loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        total_loss += float(loss.item()) * len(labels)

        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

    sample_count = max(sum(len(batch) for batch in labels_list), 1)
    average_loss = total_loss / sample_count
    return average_loss, np.concatenate(probabilities), np.concatenate(labels_list)


def _normalize_mlflow_tracking_uri(tracking_uri: str) -> str:
    """Normalizes MLflow tracking URIs and auto-migrates deprecated file-store URIs."""

    normalized_uri = tracking_uri.strip()

    if normalized_uri.startswith("sqlite:///"):
        raw_path = normalized_uri.removeprefix("sqlite:///")
        if raw_path == ":memory:":
            return normalized_uri

        database_path = Path(raw_path)
        if not database_path.is_absolute():
            database_path = database_path.resolve()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{database_path.as_posix()}"

    if normalized_uri.startswith("file:///"):
        raw_file_path = normalized_uri.removeprefix("file:///")
    elif normalized_uri.startswith("file:"):
        raw_file_path = normalized_uri.removeprefix("file:")
    else:
        raw_file_path = ""

    if raw_file_path:
        file_path = Path(raw_file_path)
        if not file_path.is_absolute():
            file_path = file_path.resolve()

        if file_path.name == "mlruns" or not file_path.suffix:
            database_path = file_path.parent / "mlflow.db"
        else:
            database_path = file_path

        database_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{database_path.as_posix()}"

    return normalized_uri


def _sqlite_mlflow_tracking_uri() -> str:
    database_path = Path("mlflow.db").resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path.as_posix()}"


@contextmanager
def _mlflow_run(
    training_config: SequenceTrainingConfig,
    model_config: SequenceModelConfig,
) -> Iterator[Any | None]:
    if not training_config.mlflow_enabled:
        yield None
        return

    try:
        import mlflow
    except ModuleNotFoundError:
        logger.warning("MLflow is not installed. Continuing without tracking.")
        yield None
        return

    run_name = training_config.mlflow_run_name or (
        f"{model_config.model_type}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )

    tracking_uri = _normalize_mlflow_tracking_uri(training_config.mlflow_tracking_uri)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(training_config.mlflow_experiment_name)
        with mlflow.start_run(run_name=run_name):
            yield mlflow
        return
    except Exception as exc:
        sqlite_uri = _sqlite_mlflow_tracking_uri()

        try:
            mlflow.set_tracking_uri(sqlite_uri)
            mlflow.set_experiment(training_config.mlflow_experiment_name)
            with mlflow.start_run(run_name=run_name):
                logger.warning(
                    "MLflow file tracking failed; switched to SQLite tracking backend.",
                    extra={
                        "original_tracking_uri": tracking_uri,
                        "fallback_tracking_uri": sqlite_uri,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                yield mlflow
            return
        except Exception as fallback_exc:
            logger.warning(
                "MLflow tracking initialization failed. Continuing without tracking.",
                extra={
                    "tracking_uri": tracking_uri,
                    "fallback_tracking_uri": sqlite_uri,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                },
            )
            yield None
            return


def _checkpoint_payload(
    model: nn.Module,
    model_config: SequenceModelConfig,
    training_config: SequenceTrainingConfig,
    epoch: int,
    input_dim: int,
    sequence_length: int,
    static_dim: int,
    threshold: float,
    metrics: dict[str, float | int | None],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": {
            **asdict(training_config),
            "windows_dir": str(training_config.windows_dir),
            "output_dir": str(training_config.output_dir),
        },
        "input_dim": input_dim,
        "sequence_length": sequence_length,
        "static_dim": static_dim,
        "threshold": threshold,
        "validation_metrics": metrics,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_sequence_model(
    training_config: SequenceTrainingConfig,
    model_config: SequenceModelConfig,
    trial: Any | None = None,
) -> SequenceTrainingResult:
    """Trains a sequence model with checkpointing, early stopping, and scheduler."""

    canonical_model_name = sequence_model_family_name(model_config.model_type)
    if training_config.model_name != canonical_model_name:
        logger.info(
            "Aligning training model_name to model_type-derived artifact family name",
            extra={
                "configured_model_name": training_config.model_name,
                "canonical_model_name": canonical_model_name,
                "model_type": model_config.model_type,
            },
        )
        training_config = replace(training_config, model_name=canonical_model_name)

    _set_deterministic_seed(training_config.seed)
    device = _resolve_device(training_config.device)

    split_loaders = _build_split_loaders(config=training_config, model_config=model_config)
    model = build_sequence_model(
        input_dim=split_loaders.input_dim,
        static_dim=split_loaders.static_dim,
        sequence_length=split_loaders.sequence_length,
        config=model_config,
    ).to(device)

    if split_loaders.pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([split_loaders.pos_weight], device=device, dtype=torch.float32)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
    )

    run_name = build_sequence_run_name(model_config.model_type)
    run_dir = training_config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    history_path = run_dir / "training_history.json"
    config_path = run_dir / "run_config.json"
    best_validation_path = run_dir / "validation_metrics.json"
    test_metrics_path = run_dir / "test_metrics.json"

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_threshold = training_config.threshold
    best_metrics: dict[str, float | int | None] = {}
    stale_epochs = 0

    with _mlflow_run(training_config=training_config, model_config=model_config) as mlflow_client:
        if mlflow_client is not None:
            mlflow_client.log_params(
                {
                    "model_type": model_config.model_type,
                    "include_mask": model_config.include_mask,
                    "include_static": model_config.include_static,
                    "batch_size": training_config.batch_size,
                    "epochs": training_config.epochs,
                    "learning_rate": training_config.learning_rate,
                    "weight_decay": training_config.weight_decay,
                    "imbalance_strategy": training_config.imbalance_strategy,
                    "sequence_length": split_loaders.sequence_length,
                    "input_dim": split_loaders.input_dim,
                    "static_dim": split_loaders.static_dim,
                }
            )

        for epoch in range(1, training_config.epochs + 1):
            train_loss, train_probs, train_targets = _run_epoch(
                model=model,
                loader=split_loaders.train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                max_grad_norm=training_config.max_grad_norm,
            )
            validation_loss, validation_probs, validation_targets = _run_epoch(
                model=model,
                loader=split_loaders.validation_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                max_grad_norm=training_config.max_grad_norm,
            )

            if training_config.optimize_threshold:
                selected_threshold, _ = find_optimal_threshold(
                    y_true=validation_targets,
                    y_prob=validation_probs,
                )
            else:
                selected_threshold = training_config.threshold

            train_metrics = compute_binary_metrics(
                y_true=train_targets,
                y_prob=train_probs,
                threshold=selected_threshold,
                calibration_bins=training_config.calibration_bins,
            )
            validation_metrics = compute_binary_metrics(
                y_true=validation_targets,
                y_prob=validation_probs,
                threshold=selected_threshold,
                calibration_bins=training_config.calibration_bins,
            )

            scheduler.step(float(validation_metrics.auprc))

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_metrics": train_metrics.to_dict(),
                "validation_metrics": validation_metrics.to_dict(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)

            if mlflow_client is not None:
                mlflow_client.log_metrics(
                    {
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "train_auprc": float(train_metrics.auprc),
                        "validation_auprc": float(validation_metrics.auprc),
                        "validation_auroc": float(validation_metrics.auroc),
                        "validation_f1": float(validation_metrics.f1),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=epoch,
                )

            current_score = float(validation_metrics.auprc)
            improved = current_score > (best_score + training_config.early_stopping_min_delta)
            if improved:
                best_score = current_score
                best_threshold = float(selected_threshold)
                best_metrics = validation_metrics.to_dict()
                stale_epochs = 0
                _save_checkpoint(
                    best_checkpoint_path,
                    _checkpoint_payload(
                        model=model,
                        model_config=model_config,
                        training_config=training_config,
                        epoch=epoch,
                        input_dim=split_loaders.input_dim,
                        sequence_length=split_loaders.sequence_length,
                        static_dim=split_loaders.static_dim,
                        threshold=best_threshold,
                        metrics=best_metrics,
                    ),
                )
            else:
                stale_epochs += 1

            _save_checkpoint(
                last_checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    model_config=model_config,
                    training_config=training_config,
                    epoch=epoch,
                    input_dim=split_loaders.input_dim,
                    sequence_length=split_loaders.sequence_length,
                    static_dim=split_loaders.static_dim,
                    threshold=float(selected_threshold),
                    metrics=validation_metrics.to_dict(),
                ),
            )

            if trial is not None:
                trial.report(float(validation_metrics.auprc), step=epoch)
                if trial.should_prune():
                    try:
                        import optuna

                        raise optuna.exceptions.TrialPruned("Validation AUPRC did not improve.")
                    except ModuleNotFoundError as exc:
                        msg = "Trial pruning requested but Optuna is unavailable."
                        raise RuntimeError(msg) from exc

            if stale_epochs >= training_config.early_stopping_patience:
                logger.info(
                    "Early stopping activated",
                    extra={"epoch": epoch, "best_auprc": best_score},
                )
                break

        if not best_checkpoint_path.exists():
            msg = "No checkpoint was created during training."
            raise RuntimeError(msg)

        test_evaluation = evaluate_checkpoint(
            checkpoint_path=best_checkpoint_path,
            parquet_path=training_config.windows_dir / "test.parquet",
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            threshold=best_threshold,
            calibration_bins=training_config.calibration_bins,
        )

        if mlflow_client is not None:
            mlflow_client.log_metrics(
                {
                    "test_auprc": float(test_evaluation["metrics"]["auprc"]),
                    "test_auroc": float(test_evaluation["metrics"]["auroc"]),
                    "test_f1": float(test_evaluation["metrics"]["f1"]),
                }
            )
            mlflow_client.log_artifact(str(best_checkpoint_path))
            mlflow_client.log_artifact(str(last_checkpoint_path))

    _json_dump(
        config_path,
        {
            "training_config": {
                **asdict(training_config),
                "windows_dir": str(training_config.windows_dir),
                "output_dir": str(training_config.output_dir),
            },
            "model_config": asdict(model_config),
            "class_distribution": split_loaders.class_counts,
            "input_dim": split_loaders.input_dim,
            "sequence_length": split_loaders.sequence_length,
            "static_dim": split_loaders.static_dim,
        },
    )
    _json_dump(history_path, {"history": history})
    _json_dump(best_validation_path, {"metrics": best_metrics, "threshold": best_threshold})
    _json_dump(
        test_metrics_path,
        {"metrics": test_evaluation["metrics"], "threshold": best_threshold},
    )

    logger.info(
        "Sequence model training completed",
        extra={
            "run_dir": str(run_dir),
            "best_validation_auprc": best_metrics.get("auprc", 0.0),
            "test_auprc": test_evaluation["metrics"].get("auprc", 0.0),
            "selected_threshold": best_threshold,
        },
    )

    return SequenceTrainingResult(
        run_dir=run_dir,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        best_validation_metrics=best_metrics,
        test_metrics=test_evaluation["metrics"],
        selected_threshold=best_threshold,
    )


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)
    return torch.load(checkpoint_path, map_location=device)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    """Loads a trained sequence model and metadata from checkpoint."""

    resolved_device = _resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)

    model_config = SequenceModelConfig(**checkpoint["model_config"])
    model = build_sequence_model(
        input_dim=int(checkpoint["input_dim"]),
        static_dim=int(checkpoint["static_dim"]),
        sequence_length=int(checkpoint["sequence_length"]),
        config=model_config,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, resolved_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    parquet_path: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
    threshold: float | None = None,
    calibration_bins: int | None = 10,
) -> dict[str, Any]:
    """Evaluates a trained checkpoint against one window parquet split."""

    model, checkpoint, device = load_model_from_checkpoint(checkpoint_path=checkpoint_path)
    model_config = SequenceModelConfig(**checkpoint["model_config"])
    resolved_threshold = float(
        threshold if threshold is not None else checkpoint.get("threshold", 0.5)
    )

    dataset = SepsisWindowDataset(
        parquet_path=parquet_path,
        include_mask=model_config.include_mask,
        include_static=model_config.include_static,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()
    loss, probabilities, targets = _run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        max_grad_norm=0.0,
    )

    metrics = compute_binary_metrics(
        y_true=targets,
        y_prob=probabilities,
        threshold=resolved_threshold,
        calibration_bins=calibration_bins,
    ).to_dict()

    return {
        "loss": loss,
        "metrics": metrics,
        "threshold": resolved_threshold,
        "targets": targets.tolist(),
        "probabilities": probabilities.tolist(),
    }


def predict_from_checkpoint(
    checkpoint_path: str | Path,
    parquet_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Generates probability and label predictions for one parquet split."""

    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        parquet_path=parquet_path,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
        calibration_bins=None,
    )

    split_frame = pd.read_parquet(parquet_path, columns=["patient_id", "end_hour", "label"])
    probabilities = np.asarray(evaluation["probabilities"], dtype=np.float64)
    decision_threshold = float(evaluation["threshold"])

    output_frame = split_frame.copy()
    output_frame = output_frame.rename(columns={"label": "true_label"})
    output_frame["predicted_probability"] = probabilities
    output_frame["predicted_label"] = (probabilities >= decision_threshold).astype(np.int64)
    output_frame["threshold_used"] = decision_threshold

    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.suffix.lower() == ".csv":
            output_frame.to_csv(target_path, index=False)
        else:
            output_frame.to_parquet(target_path, index=False)

    return output_frame


def merge_training_overrides(
    config: SequenceTrainingConfig,
    **overrides: Any,
) -> SequenceTrainingConfig:
    """Applies runtime overrides while preserving typed dataclass semantics."""

    payload = asdict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        payload[key] = value

    payload["windows_dir"] = Path(payload["windows_dir"])
    payload["output_dir"] = Path(payload["output_dir"])
    return replace(config, **payload)


def merge_model_overrides(
    config: SequenceModelConfig,
    **overrides: Any,
) -> SequenceModelConfig:
    """Applies runtime overrides to model hyperparameters."""

    payload = asdict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        payload[key] = value
    return SequenceModelConfig(**payload)


def sequence_training_config_from_dict(payload: dict[str, Any]) -> SequenceTrainingConfig:
    """Loads training config from dictionary payload."""

    normalized = {**payload}
    if "windows_dir" in normalized:
        normalized["windows_dir"] = Path(normalized["windows_dir"])
    if "output_dir" in normalized:
        normalized["output_dir"] = Path(normalized["output_dir"])

    if "imbalance_strategy" in normalized and normalized["imbalance_strategy"] not in {
        "none",
        "pos_weight",
        "weighted_sampler",
        "both",
    }:
        msg = f"Unsupported imbalance_strategy: {normalized['imbalance_strategy']}"
        raise ValueError(msg)

    return SequenceTrainingConfig(**normalized)


def sequence_model_config_from_dict(payload: dict[str, Any]) -> SequenceModelConfig:
    """Loads model config from dictionary payload."""

    return SequenceModelConfig(**payload)
