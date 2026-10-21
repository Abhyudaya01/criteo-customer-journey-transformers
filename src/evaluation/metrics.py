"""Evaluation metrics shared across model families.

Conversion (binary, heavily imbalanced): PR-AUC is primary; ROC-AUC, balanced accuracy, F1
and BCE as in the proposal; plus the Lu & Kannan targeting metrics (top-decile conversion
rate, lift, gain) and calibration (ECE, Brier), which matter because predicted
probabilities are what a bidder or budget allocator would actually consume.

Campaign (multi-label over ~700 campaigns, most next windows empty): ranking metrics are
computed only on positions whose next window has at least one campaign, otherwise every
metric is dominated by empty rows. Report the share of such positions alongside.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


# ------------------------------------------------------------------ conversion

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
        "brier": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": expected_calibration_error(y_true, y_prob),
        "base_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }
    out.update(top_fraction_metrics(y_true, y_prob, fraction=0.10, prefix="top_decile"))
    return out


def best_f1_threshold(y_true, y_prob) -> tuple[float, float]:
    """Threshold maximizing F1. Choose it on validation, then apply it to test.

    With a ~0.1% positive rate a fixed 0.5 threshold makes F1/balanced accuracy near-meaningless.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(-y_prob)
    tp = np.cumsum(y_true[order])
    k = np.arange(1, len(order) + 1)
    pos = y_true.sum()
    if pos == 0:
        return 0.5, 0.0
    f1 = 2 * tp / (k + pos)
    i = int(np.argmax(f1))
    return float(y_prob[order][i]), float(f1[i])


def top_fraction_metrics(y_true, y_prob, fraction: float = 0.10, prefix: str = "top_decile") -> dict[str, float]:
    """True conversion rate, lift and gain in the top ``fraction`` of scored rows (Lu & Kannan)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return {f"{prefix}_conversion_rate": 0.0, f"{prefix}_lift": 0.0, f"{prefix}_gain": 0.0}
    k = max(1, int(np.ceil(n * fraction)))
    top = np.argsort(-y_prob, kind="stable")[:k]
    tcr = y_true[top].mean()
    return {
        f"{prefix}_conversion_rate": float(tcr),
        f"{prefix}_lift": float(tcr / y_true.mean()),
        f"{prefix}_gain": float(y_true[top].sum() / y_true.sum()),
    }


def gain_curve(y_true, y_prob, n_bins: int = 10) -> np.ndarray:
    """Cumulative share of conversions captured in the top 10%, 20%, ... of scores."""
    return np.array([top_fraction_metrics(y_true, y_prob, f / n_bins, "x")["x_gain"]
                     for f in range(1, n_bins + 1)])


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE with equal-mass bins (equal-width bins are almost all empty at low base rates)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    # Quantile edges, then bin by value so tied probabilities always share a bin.
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.searchsorted(edges, y_prob, side="right")
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        ece += m.mean() * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


# ------------------------------------------------------------------ campaigns

def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    return idx[row, np.argsort(-scores[row, idx], axis=1)]


def precision_recall_at_k(y_true, scores, k: int = 10) -> tuple[float, float]:
    """Micro-averaged Precision@K and Recall@K for multi-label campaign prediction."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(y_true) == 0:
        return 0.0, 0.0
    topk = _topk(scores, k)
    hits = np.take_along_axis(y_true, topk, axis=1).sum()
    possible = y_true.sum()
    precision = hits / (len(y_true) * topk.shape[1])
    recall = hits / possible if possible else 0.0
    return float(precision), float(recall)


def campaign_ranking_metrics(y_true, scores, ks=(1, 5, 10), nonempty_only: bool = True) -> dict[str, float]:
    """Precision/Recall/HitRate/NDCG@K, MAP@K and label-ranking coverage for campaigns."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    out = {"frac_positions_nonempty": float((y_true.sum(1) > 0).mean()) if len(y_true) else 0.0}
    if nonempty_only:
        keep = y_true.sum(1) > 0
        y_true, scores = y_true[keep], scores[keep]
    if len(y_true) == 0:
        return out
    n_rel = y_true.sum(1)
    for k in ks:
        topk = _topk(scores, k)
        rel = np.take_along_axis(y_true, topk, axis=1).astype(float)
        kk = topk.shape[1]
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = (rel * discounts).sum(1)
        ideal = np.array([discounts[: min(int(r), kk)].sum() for r in n_rel])
        prec_at_i = np.cumsum(rel, axis=1) / np.arange(1, kk + 1)
        ap = (prec_at_i * rel).sum(1) / np.minimum(n_rel, kk)
        p, r = precision_recall_at_k(y_true, scores, k)
        out.update({
            f"precision_at_{k}": p,
            f"recall_at_{k}": r,
            f"hit_rate_at_{k}": float((rel.sum(1) > 0).mean()),
            f"ndcg_at_{k}": float(np.mean(dcg / ideal)),
            f"map_at_{k}": float(ap.mean()),
        })
    return out


def campaign_bce(y_true, probs, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy over all positions x campaigns."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
