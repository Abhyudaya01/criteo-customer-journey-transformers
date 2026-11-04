"""Subgroup flags for prediction positions, so every model is scored on the same slices.

A position is (uid, window t); its label window is t + 1. Flags are computed from the processed
tables written by scripts/prepare_criteo.py, never from model inputs, so any model's saved
predictions can be sliced the same way without retraining.

Subgroups (Final Stretch Plan, Week 1):

``active_5plus``
    The user has been active (>= 1 impression) in at least 5 windows up to and including t.
    History only. Note: the interim report's "10.4% of users" counts active windows over the
    whole log; this flag counts them *as of t*, which is what a deployed model could know.
``recent_click``
    The user clicked in the last 24 hours, i.e. in windows t - k + 1 .. t with
    k = ceil(24 / window_hours) (k = 2 for 12-hour windows, 4 for 6-hour windows). History only.
``next_new_campaign``
    The label window t + 1 contains at least one campaign the user had never been shown in
    windows <= t. This looks at the label window, so it is a *descriptive* slice (where do the
    models do well) and not a segment a bidder could select in advance. Same definition as the
    "new-to-user" statistic in scripts/dataset_stats.py.

For the campaign task, ``split_truth_by_novelty`` separates each position's next-window
campaigns into seen and new-to-user, so Recall@K on new campaigns can go through
``ranking_metrics_from_topk``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SUBGROUPS = ["all", "active_5plus", "recent_click", "next_new_campaign"]
HISTORY_ONLY = {"all": True, "active_5plus": True, "recent_click": True, "next_new_campaign": False}


def recent_click_windows(window_hours: float, hours: float = 24.0) -> int:
    """Number of windows (including t) that cover the last ``hours`` hours."""
    return max(1, int(math.ceil(hours / window_hours)))


def _cum_active_at(positions: pd.DataFrame, user_windows: pd.DataFrame) -> np.ndarray:
    """Count of active windows <= t for each position (same row order as ``positions``)."""
    act = user_windows.loc[user_windows["is_active"] == 1, ["uid", "window"]].astype("int64")
    act = act.sort_values(["uid", "window"], kind="mergesort").reset_index(drop=True)
    act["cum_active"] = act.groupby("uid", sort=False).cumcount() + 1
    left = positions[["uid", "window"]].astype("int64").reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("window", kind="mergesort"),
                      act.sort_values("window", kind="mergesort"),
                      on="window", by="uid", direction="backward")
    return m.sort_values("_row")["cum_active"].fillna(0).to_numpy(np.int32)


def _recent_click(positions: pd.DataFrame, user_windows: pd.DataFrame, k: int) -> np.ndarray:
    clicked = user_windows.loc[user_windows["n_clicks"] > 0, ["uid", "window"]].drop_duplicates()
    keys = positions[["uid", "window"]].reset_index(drop=True)
    hit = np.zeros(len(keys), dtype=bool)
    for lag in range(k):
        probe = keys.assign(window=keys["window"] - lag, _row=np.arange(len(keys)))
        m = probe.merge(clicked, on=["uid", "window"], how="inner")
        hit[m["_row"].to_numpy()] = True
    return hit


def _first_seen(window_campaigns: pd.DataFrame) -> pd.DataFrame:
    return (window_campaigns.groupby(["uid", "campaign_idx"], sort=False)["window"].min()
            .rename("first_seen").reset_index())


def _next_new_campaign(positions: pd.DataFrame, window_campaigns: pd.DataFrame,
                       next_campaign_labels: pd.DataFrame) -> np.ndarray:
    lab = next_campaign_labels[["uid", "window", "label_window", "campaign_idx"]].merge(
        _first_seen(window_campaigns), on=["uid", "campaign_idx"], how="left")
    novel = lab.loc[lab["first_seen"] == lab["label_window"], ["uid", "window"]].drop_duplicates()
    novel["_novel"] = True
    m = positions[["uid", "window"]].merge(novel, on=["uid", "window"], how="left")
    return m["_novel"].fillna(False).to_numpy(bool)


def position_subgroup_flags(positions: pd.DataFrame, user_windows: pd.DataFrame,
                            window_campaigns: pd.DataFrame, next_campaign_labels: pd.DataFrame,
                            window_hours: float, min_active_windows: int = 5) -> pd.DataFrame:
    """Return ``uid, window`` plus one boolean column per subgroup, row-aligned with ``positions``.

    All tables must come from the same user shard (as ``src.data.io.load_shard`` returns them),
    because history is looked up per user.
    """
    pos = positions[["uid", "window"]].reset_index(drop=True)
    out = pos.copy()
    out["cum_active_windows"] = _cum_active_at(pos, user_windows)
    out["active_5plus"] = out["cum_active_windows"] >= min_active_windows
    out["recent_click"] = _recent_click(pos, user_windows, recent_click_windows(window_hours))
    out["next_new_campaign"] = _next_new_campaign(pos, window_campaigns, next_campaign_labels)
    return out


def split_truth_by_novelty(next_campaign_labels: pd.DataFrame,
                           window_campaigns: pd.DataFrame) -> pd.DataFrame:
    """Next-window campaigns per position, split into ``seen`` and ``new`` (lists of indices).

    ``new`` = campaigns never shown to the user before the label window. Feed ``new`` (rows where
    it is non-empty) to ``ranking_metrics_from_topk`` for Recall@K on new-to-user campaigns.
    """
    lab = next_campaign_labels[["uid", "window", "label_window", "campaign_idx"]].merge(
        _first_seen(window_campaigns), on=["uid", "campaign_idx"], how="left")
    lab["is_new"] = lab["first_seen"] == lab["label_window"]
    g = lab.groupby(["uid", "window"], sort=True)
    seen = lab[~lab["is_new"]].groupby(["uid", "window"])["campaign_idx"].agg(list)
    new = lab[lab["is_new"]].groupby(["uid", "window"])["campaign_idx"].agg(list)
    out = pd.DataFrame(index=g.size().index)
    out["seen"] = seen.reindex(out.index)
    out["new"] = new.reindex(out.index)
    for c in ("seen", "new"):
        out[c] = out[c].apply(lambda v: v if isinstance(v, list) else [])
    return out.reset_index()
