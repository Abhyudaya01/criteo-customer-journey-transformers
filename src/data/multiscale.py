"""Fine-grained history, coarse target: 6-hour input tokens predicting the next 12 hours.

Window-length ablation design (Final Stretch Plan, Must item "6-hour vs. 12-hour"):

    The *target* stays the 12-hour next-window label on the *same* prediction positions as every
    other model; only the input tokenization changes. A 12h position (uid, k) predicts 12h window
    k + 1 from history up to the end of 12h window k. With r = 12 / 6 = 2 fine windows per coarse
    window, the end of 12h window k is the end of 6h window r*k + r - 1, so the position is
    re-anchored there. History through 6h window r*k + r - 1 covers exactly the same hours as
    history through 12h window k, so the model sees the same information, cut finer.

Why not 6h targets: a 6h label halves the base rate and changes the task, so PR-AUC would no
longer be comparable, and STAJT (which reads raw events, not window tokens) would need a full
retrain only to change its horizon. With this design STAJT's 12h result is already the
comparison point, and every 6h-input prediction is scored on the same test positions, so the
user bootstrap pairs it with every other model.

The output folder has the fine tables (user_windows, window_campaigns, conversions) unchanged,
plus coarse positions / next-window campaign labels re-anchored to fine window indices.
``JourneySequences`` and ``build_position_features`` read it with no changes (use the fine
window length). ``to_coarse_keys`` maps fine-anchored predictions back to 12h (uid, window) keys
for scripts/evaluate_predictions.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def window_ratio(coarse_hours: float, fine_hours: float) -> int:
    r = coarse_hours / fine_hours
    if abs(r - round(r)) > 1e-9 or r < 1:
        raise ValueError(f"{coarse_hours}h is not a whole multiple of {fine_hours}h")
    return int(round(r))


def fine_anchor(coarse_window, ratio: int):
    """Last fine window inside coarse window ``k``: ratio * k + ratio - 1."""
    return coarse_window * ratio + (ratio - 1)


def to_coarse_keys(fine_window, ratio: int):
    """Inverse of ``fine_anchor`` (raises if a window is not a coarse-window end)."""
    fw = np.asarray(fine_window)
    if ((fw + 1) % ratio).any():
        raise ValueError("some fine windows are not the last fine window of a coarse window")
    return (fw + 1) // ratio - 1


def _fine_recency(keys: pd.DataFrame, fine_user_windows: pd.DataFrame) -> np.ndarray:
    """Fine windows since the last impression at or before each fine anchor (-1 if none)."""
    act = fine_user_windows.loc[fine_user_windows["is_active"] == 1, ["uid", "window"]]
    act = act.astype({"uid": "int64", "window": "int64"})
    act = act.rename(columns={"window": "last_active"}).assign(window=lambda d: d["last_active"])
    left = keys[["uid", "window"]].astype("int64").reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("window", kind="mergesort"),
                      act.sort_values("window", kind="mergesort"),
                      on="window", by="uid", direction="backward").sort_values("_row")
    since = m["window"].to_numpy() - m["last_active"].to_numpy()
    return np.where(np.isnan(since), -1, since).astype(np.int16)


def reanchor_positions(coarse_positions: pd.DataFrame, fine_user_windows: pd.DataFrame,
                       ratio: int) -> pd.DataFrame:
    """Coarse positions placed at fine anchors, keeping coarse labels and split columns.

    ``window`` becomes the fine anchor; ``coarse_window`` / ``coarse_label_window`` keep the
    original keys. ``is_active`` / ``windows_since_active`` are recomputed in fine units (an
    impression in the last fine window), matching what a fine-window model would see.
    ``label_window`` is set to ``window + 1`` (the first fine window of the label period) so
    window-index code that expects it still lines up; the label itself covers the whole
    coarse window ``coarse_label_window``.
    """
    pos = coarse_positions.copy()
    pos["coarse_window"] = pos["window"].astype("int32")
    pos["coarse_label_window"] = pos["label_window"].astype("int32")
    pos["window"] = fine_anchor(pos["coarse_window"].to_numpy(), ratio).astype("int32")
    pos["label_window"] = (pos["window"] + 1).astype("int32")
    since = _fine_recency(pos, fine_user_windows)
    if (since < 0).any():
        raise ValueError(f"{int((since < 0).sum())} positions have no fine-window impression at or "
                         "before their anchor; coarse and fine tables come from different users")
    pos["windows_since_active"] = since
    pos["is_active"] = (since == 0).astype("int8")
    return pos


def reanchor_next_campaign_labels(coarse_next: pd.DataFrame, ratio: int) -> pd.DataFrame:
    out = coarse_next.rename(columns={"window": "coarse_window",
                                      "label_window": "coarse_label_window"}).copy()
    out["window"] = fine_anchor(out["coarse_window"].to_numpy(), ratio).astype("int32")
    out["label_window"] = (out["window"] + 1).astype("int32")
    return out[["uid", "window", "label_window", "campaign_idx", "coarse_window", "coarse_label_window"]]


def build_shard(coarse: dict, fine: dict, ratio: int) -> dict:
    """One user shard of the fine-history / coarse-target dataset (tables as load_shard returns)."""
    return {
        "user_windows": fine["user_windows"],
        "window_campaigns": fine["window_campaigns"],
        "conversions": fine["conversions"],
        "positions": reanchor_positions(coarse["positions"], fine["user_windows"], ratio),
        "next_campaign_labels": reanchor_next_campaign_labels(coarse["next_campaign_labels"], ratio),
    }
