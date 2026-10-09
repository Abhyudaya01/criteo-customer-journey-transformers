"""Evaluation metrics shared across model families."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def conversion_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "binary_cross_entropy": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
    return out


def precision_recall_at_k(y_true, scores, k: int = 10) -> tuple[float, float]:
    """Micro-averaged Precision@K and Recall@K for multi-label campaign prediction."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    topk = np.argpartition(-scores, kth=min(k, scores.shape[1]) - 1, axis=1)[:, :k]

    hits = 0
    possible = 0
    for i, idx in enumerate(topk):
        truth = set(np.flatnonzero(y_true[i]))
        pred = set(idx.tolist())
        hits += len(truth & pred)
        possible += len(truth)

    precision = hits / (len(y_true) * k) if len(y_true) else 0.0
    recall = hits / possible if possible else 0.0
    return float(precision), float(recall)
