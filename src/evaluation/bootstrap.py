"""Bootstrap confidence intervals over users, for any model's saved predictions.

Positions from the same user are correlated (one user contributes up to ~60 positions), so we
resample *users* with replacement, not rows. Each replicate gives every row a weight equal to
the number of times its user was drawn, and the metric is computed with those weights.

Design choices that keep this cheap on 20M-row test sets (CPU only, no retraining):

* Scores are sorted once per prediction array; every replicate reuses that order and only
  recomputes weighted cumulative sums.
* All models and seeds in one call share the same user draws (same RNG seed in every worker),
  so differences between models are *paired*.
* With several seeds per model, each replicate averages the metric over seeds, giving a CI for
  the seed-averaged metric. The seed standard deviation is reported next to it.
* ``neg_frac < 1`` runs the replicates on all positives plus a fixed random share of negatives
  weighted by ``1 / neg_frac`` (the same rows for every model, so pairing holds). Point
  estimates always use every row. With ~20M negatives and ``neg_frac = 0.1`` the subsample
  adds far less noise than the user resampling itself, and runs ~10x faster.
* ``n_jobs > 1`` runs the prediction arrays in parallel processes.

Weighted PR-AUC and ROC-AUC match ``sklearn.metrics.average_precision_score`` and
``roc_auc_score`` with ``sample_weight`` (tests/test_bootstrap.py checks this, ties included).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

METRICS = ("pr_auc", "roc_auc", "top_decile_gain")


@dataclass
class SortedPreds:
    """One prediction array, pre-sorted by descending score (stable, like top_fraction_metrics)."""

    order: np.ndarray        # row indices in descending score order
    y: np.ndarray            # labels in that order (float64)
    group_end: np.ndarray    # positions (in sorted order) where a run of tied scores ends
    codes: np.ndarray | None = None      # user code per sorted row (for replicate weights)
    base_w: np.ndarray | None = None     # fixed row weight per sorted row (negative subsampling)

    @classmethod
    def build(cls, y_true, scores, user_codes=None, base_weight=None) -> "SortedPreds":
        y_true = np.asarray(y_true)
        scores = np.asarray(scores, dtype=np.float64)
        if y_true.shape != scores.shape:
            raise ValueError(f"y_true {y_true.shape} and scores {scores.shape} differ")
        order = np.argsort(-scores, kind="mergesort")
        s = scores[order]
        group_end = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
        return cls(order=order, y=y_true[order].astype(np.float64), group_end=group_end,
                   codes=None if user_codes is None else np.asarray(user_codes)[order],
                   base_w=None if base_weight is None else np.asarray(base_weight, np.float64)[order])


def _metrics_from_sorted_weights(sp: SortedPreds, w: np.ndarray, metrics: Sequence[str],
                                 top_fraction: float) -> dict[str, float]:
    cw = np.cumsum(w)
    tps_all = np.cumsum(w * sp.y)
    tps = tps_all[sp.group_end]
    fps = cw[sp.group_end] - tps
    P, N = tps[-1], fps[-1]
    nan = float("nan")
    out: dict[str, float] = {}
    if "pr_auc" in metrics:
        out["pr_auc"] = nan if P <= 0 else float(
            np.sum(np.diff(np.r_[0.0, tps / P]) * (tps / np.maximum(tps + fps, 1e-300))))
    if "roc_auc" in metrics:
        if P <= 0 or N <= 0:
            out["roc_auc"] = nan
        else:
            tpr, fpr = np.r_[0.0, tps / P], np.r_[0.0, fps / N]
            out["roc_auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    if "top_decile_gain" in metrics:
        if P <= 0:
            out["top_decile_gain"] = nan
        else:
            k = np.ceil(cw[-1] * top_fraction)
            i = int(np.searchsorted(cw, k, side="left"))
            out["top_decile_gain"] = float(tps_all[min(i, len(cw) - 1)] / P)
    if "base_rate" in metrics:
        out["base_rate"] = float(P / (P + N)) if (P + N) > 0 else nan
    return out


def sorted_metrics(sp: SortedPreds, weights: np.ndarray | None = None,
                   metrics: Sequence[str] = METRICS, top_fraction: float = 0.10) -> dict[str, float]:
    """PR-AUC, ROC-AUC and top-decile gain; ``weights`` are row weights in the original order."""
    w = np.ones(len(sp.y)) if weights is None else np.asarray(weights, dtype=np.float64)[sp.order]
    return _metrics_from_sorted_weights(sp, w, metrics, top_fraction)


def _draw_counts(rng: np.random.Generator, n_users: int) -> np.ndarray:
    return np.bincount(rng.integers(0, n_users, n_users), minlength=n_users).astype(np.float64)


def _replicates(sp: SortedPreds, n_users: int, n_boot: int, seed: int,
                metrics: Sequence[str]) -> np.ndarray:
    """[n_boot, len(metrics)] metric draws. Same ``seed`` -> same user draws in every process."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(metrics)))
    for b in range(n_boot):
        w = _draw_counts(rng, n_users)[sp.codes]
        if sp.base_w is not None:
            w *= sp.base_w
        r = _metrics_from_sorted_weights(sp, w, metrics, 0.10)
        out[b] = [r[m] for m in metrics]
    return out


def user_weights(user_codes: np.ndarray, n_users: int, rng: np.random.Generator) -> np.ndarray:
    """Row weights for one replicate: how many times each row's user was drawn."""
    return _draw_counts(rng, n_users)[user_codes]


def user_bootstrap(y_true, predictions: Mapping[str, np.ndarray | Sequence[np.ndarray]], users,
                   n_boot: int = 1000, seed: int = 0, metrics: Sequence[str] = METRICS,
                   reference: str | None = None, alpha: float = 0.05, neg_frac: float = 1.0,
                   n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster (user) bootstrap for one or more models on the *same* rows.

    ``predictions`` maps model name -> scores, or -> a list of score arrays (one per seed).
    Every array must be row-aligned with ``y_true`` and ``users``.

    Returns ``(ci, diffs)``:
      ci:    model, metric, point, seed_sd, ci_low, ci_high, n_seeds, n_boot, n_rows, n_users
      diffs: model, reference, metric, diff, ci_low, ci_high, p_not_better
             (share of replicates where model <= reference); empty when ``reference`` is None.
    ``point`` is the seed-averaged metric on all rows, not the bootstrap mean.
    """
    metrics = list(metrics)
    y_true = np.asarray(y_true)
    codes, uniques = pd.factorize(pd.Series(np.asarray(users)), sort=False)
    codes = codes.astype(np.int64)
    n_users = len(uniques)

    arrays: dict[str, list[np.ndarray]] = {}
    for name, p in predictions.items():
        arrays[name] = [p] if isinstance(p, np.ndarray) and p.ndim == 1 else list(p)
    if reference is not None and reference not in arrays:
        raise KeyError(f"reference model {reference!r} not in predictions {list(arrays)}")

    # Rows used by the replicates: all positives + a fixed share of negatives (same for all models).
    if neg_frac < 1.0:
        keep = (y_true == 1) | (np.random.default_rng(seed + 1).random(len(y_true)) < neg_frac)
        base_w = np.where(y_true[keep] == 1, 1.0, 1.0 / neg_frac)
    else:
        keep, base_w = np.ones(len(y_true), dtype=bool), None

    jobs, point_vals = [], {}
    for name, arrs in arrays.items():
        for a in arrs:
            a = np.asarray(a)
            point_vals.setdefault(name, []).append(sorted_metrics(SortedPreds.build(y_true, a), None, metrics))
            jobs.append((name, SortedPreds.build(y_true[keep], a[keep], codes[keep], base_w)))

    if n_jobs > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_replicates)(sp, n_users, n_boot, seed, metrics) for _, sp in jobs)
    else:
        results = [_replicates(sp, n_users, n_boot, seed, metrics) for _, sp in jobs]

    draws: dict[str, list[np.ndarray]] = {}
    for (name, _), r in zip(jobs, results):
        draws.setdefault(name, []).append(r)
    seed_avg = {name: np.mean(np.stack(rs), axis=0) for name, rs in draws.items()}  # [n_boot, M]

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci_rows, diff_rows = [], []
    for name in arrays:
        vals = point_vals[name]
        for j, m in enumerate(metrics):
            v = [x[m] for x in vals]
            d = seed_avg[name][:, j]
            ci_rows.append({
                "model": name, "metric": m, "point": float(np.mean(v)),
                "seed_sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                "ci_low": float(np.nanpercentile(d, lo)), "ci_high": float(np.nanpercentile(d, hi)),
                "n_seeds": len(v), "n_boot": n_boot, "n_rows": len(y_true), "n_users": n_users,
                "neg_frac": neg_frac,
            })
    if reference is not None:
        ref_point = {m: float(np.mean([x[m] for x in point_vals[reference]])) for m in metrics}
        for name in arrays:
            if name == reference:
                continue
            for j, m in enumerate(metrics):
                d = seed_avg[name][:, j] - seed_avg[reference][:, j]
                diff_rows.append({
                    "model": name, "reference": reference, "metric": m,
                    "diff": float(np.mean([x[m] for x in point_vals[name]])) - ref_point[m],
                    "ci_low": float(np.nanpercentile(d, lo)), "ci_high": float(np.nanpercentile(d, hi)),
                    "p_not_better": float(np.nanmean(d <= 0)),
                })
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def format_ci(point: float, low: float, high: float, decimals: int = 4) -> str:
    """'0.0331 [0.0318, 0.0344]' with the plan's precision rule (4 decimals for AUCs)."""
    return f"{point:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"
