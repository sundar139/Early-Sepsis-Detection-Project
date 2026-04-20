from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_binary_classifier(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Computes baseline binary classification metrics."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
    }

    if y_prob is not None and np.unique(y_true_array).size > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, np.asarray(y_prob)))
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics
