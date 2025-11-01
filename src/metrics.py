"""Evaluation metrics for ABSA prompt tuning."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def build_compute_metrics(num_labels: int) -> Callable[[tuple], dict]:
    """Return a compute_metrics callback compatible with `Trainer`."""

    if num_labels <= 0:
        raise ValueError("`num_labels` must be positive")

    def _compute_metrics(eval_pred: tuple) -> dict:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="micro")),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
        }

    return _compute_metrics
