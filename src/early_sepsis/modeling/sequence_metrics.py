from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class ClassificationMetrics:
    """Container for probability and threshold-based binary metrics."""

    auroc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    positive_rate: float
    predicted_positive_rate: float
    brier_score: float | None = None
    expected_calibration_error: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_positive": self.true_positive,
            "positive_rate": self.positive_rate,
            "predicted_positive_rate": self.predicted_positive_rate,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
        }


def _to_numpy(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    return array.astype(np.float64)


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int,
) -> float:
    if bins <= 0:
        msg = "Calibration bins must be greater than zero."
        raise ValueError(msg)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    sample_count = len(y_true)

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        in_bin = (y_prob >= lower) & (y_prob < upper if index < bins - 1 else y_prob <= upper)

        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue

        bin_true = float(y_true[in_bin].mean())
        bin_confidence = float(y_prob[in_bin].mean())
        ece += abs(bin_true - bin_confidence) * (bin_count / sample_count)

    return float(ece)


def compute_binary_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
    calibration_bins: int | None = None,
) -> ClassificationMetrics:
    """Computes robust binary metrics from probabilities and a decision threshold."""

    y_true_array = _to_numpy(y_true).astype(np.int64)
    y_prob_array = np.clip(_to_numpy(y_prob), 0.0, 1.0)

    if y_true_array.shape[0] != y_prob_array.shape[0]:
        msg = "y_true and y_prob must contain the same number of samples."
        raise ValueError(msg)
    if y_true_array.size == 0:
        msg = "Cannot compute metrics with zero samples."
        raise ValueError(msg)

    y_pred_array = (y_prob_array >= threshold).astype(np.int64)

    unique_targets = np.unique(y_true_array)
    if unique_targets.size > 1:
        auroc = float(roc_auc_score(y_true_array, y_prob_array))
        auprc = float(average_precision_score(y_true_array, y_prob_array))
    else:
        auroc = 0.0
        auprc = float(average_precision_score(y_true_array, y_prob_array))

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()

    brier: float | None = None
    if unique_targets.size > 1:
        brier = float(brier_score_loss(y_true_array, y_prob_array))

    ece: float | None = None
    if calibration_bins is not None:
        ece = _expected_calibration_error(y_true=y_true_array, y_prob=y_prob_array, bins=calibration_bins)

    return ClassificationMetrics(
        auroc=auroc,
        auprc=auprc,
        accuracy=float(accuracy_score(y_true_array, y_pred_array)),
        precision=float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        recall=float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        f1=float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        threshold=float(threshold),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        positive_rate=float(y_true_array.mean()),
        predicted_positive_rate=float(y_pred_array.mean()),
        brier_score=brier,
        expected_calibration_error=ece,
    )


def find_optimal_threshold(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.05,
) -> tuple[float, float]:
    """Finds threshold maximizing F1 score over a threshold sweep."""

    y_true_array = _to_numpy(y_true).astype(np.int64)
    y_prob_array = np.clip(_to_numpy(y_prob), 0.0, 1.0)

    if y_true_array.shape[0] != y_prob_array.shape[0]:
        msg = "y_true and y_prob must contain the same number of samples."
        raise ValueError(msg)

    threshold_values = np.arange(min_threshold, max_threshold + 1e-9, step)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in threshold_values:
        predictions = (y_prob_array >= threshold).astype(np.int64)
        current_f1 = float(f1_score(y_true_array, predictions, zero_division=0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1
