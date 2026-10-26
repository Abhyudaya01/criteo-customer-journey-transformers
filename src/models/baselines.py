"""Non-sequential baselines. CPU only, minutes on the full data.

Conversion (will the user convert in window t+1?):
    recency_frequency  heuristic score cum_impressions / (1 + windows_since_active)
    logistic           standardized logistic regression on FEATURE_COLUMNS
    gbdt               sklearn HistGradientBoostingClassifier on FEATURE_COLUMNS

Campaign (which campaigns appear in window t+1?), all returning top-K lists:
    popularity         training-period campaign frequency, same list for everyone
    repeat_last        campaigns of the user's latest active window <= t, then popularity
    user_history       all campaigns the user saw up to t (count, then recency), then popularity

A Transformer has to beat these to be worth its compute; logistic/gbdt are the bar for
conversion, user_history is the bar for campaigns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ conversion

def recency_frequency_score(df: pd.DataFrame) -> np.ndarray:
    return (df["cum_impressions"].to_numpy() + 0.5 * df["cum_conversions"].to_numpy()) / (
        1.0 + df["windows_since_active"].to_numpy())


def make_conversion_model(name: str, seed: int = 42):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              early_stopping=True, validation_fraction=0.1,
                                              random_state=seed)
    raise ValueError(name)


def correct_for_negative_sampling(p: np.ndarray, neg_rate: float) -> np.ndarray:
    """Undo the shift from keeping only a fraction ``neg_rate`` of negatives during training.

    odds_true = odds_sampled * neg_rate. Rankings (PR-AUC, ROC-AUC, lift) are unchanged;
    calibration-dependent metrics (BCE, ECE, Brier, thresholds) need this.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return p * neg_rate / (p * neg_rate + (1 - p))


# ------------------------------------------------------------------ campaigns

def campaign_popularity(window_campaigns: pd.DataFrame, n_campaigns: int) -> np.ndarray:
    """Campaign indices sorted by total impressions in the given (training) rows."""
    counts = np.bincount(window_campaigns["campaign_idx"].to_numpy(),
                         weights=window_campaigns["n_impressions"].to_numpy(),
                         minlength=n_campaigns)
    return np.argsort(-counts, kind="stable")


def _fill(personal: list, popular: np.ndarray, k: int) -> list:
    out = list(personal[:k])
    if len(out) < k:
        seen = set(out)
        for c in popular:
            if c not in seen:
                out.append(int(c))
                if len(out) == k:
                    break
    return out


def campaign_topk(positions: pd.DataFrame, window_campaigns: pd.DataFrame,
                  popular: np.ndarray, k: int = 10) -> dict[str, np.ndarray]:
    """Top-K campaign lists per position for popularity / repeat_last / user_history.

    Only window_campaigns rows with window <= t are used for position (uid, t).
    Returns arrays [N, k] aligned with ``positions`` row order.
    """
    pos = positions[["uid", "window"]].reset_index(drop=True)
    pos["_row"] = np.arange(len(pos))
    wc = window_campaigns[["uid", "window", "campaign_idx", "n_impressions"]].rename(
        columns={"window": "w"})
    m = pos.merge(wc, on="uid", how="inner")
    m = m[m["w"] <= m["window"]]

    hist = (m.groupby(["_row", "campaign_idx"], sort=False)
              .agg(cnt=("n_impressions", "sum"), last=("w", "max")).reset_index()
              .sort_values(["_row", "cnt", "last"], ascending=[True, False, False]))
    last_w = m.groupby("_row")["w"].transform("max")
    last = (m[m["w"] == last_w].sort_values(["_row", "n_impressions"], ascending=[True, False]))

    hist_lists = hist.groupby("_row")["campaign_idx"].apply(list)
    last_lists = last.groupby("_row")["campaign_idx"].apply(list)
    n = len(pos)
    pop_row = list(map(int, popular[:k]))
    out = {"popularity": np.tile(np.array(pop_row, dtype=np.int64), (n, 1))}
    for name, lists in (("repeat_last", last_lists), ("user_history", hist_lists)):
        arr = np.empty((n, k), dtype=np.int64)
        get = lists.to_dict()
        for i in range(n):
            arr[i] = _fill(get.get(i, []), popular, k)
        out[name] = arr
    return out
