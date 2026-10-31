"""Past-only tabular features for each prediction position (uid, t).

Every feature for position (uid, t) is computed from user_windows rows with window <= t, so
nothing from the label window t+1 (or later) can leak in. ``merge_asof(direction="backward")``
enforces this; tests/test_baselines.py checks it by perturbing future rows.

These features feed the non-sequential baselines (logistic regression, gradient boosting) and
can double as numeric side inputs for the sequence models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "windows_since_active",   # 0 if the user had an impression in window t
    "is_active",
    "cur_impressions", "cur_clicks", "cur_campaigns",           # window t only
    "cum_impressions", "cum_clicks", "cum_conversions",         # windows <= t
    "cum_active_windows", "cum_cost",
    "windows_since_first",    # t - first active window
    "windows_since_conversion",  # t - last conversion window (capped), large if none
    "click_rate",             # cum_clicks / cum_impressions
    "slot_of_day",            # 0..(windows per day - 1)
    "day_index",              # day number in the 31-day log (relative calendar)
]

NO_CONVERSION = 999


def _user_history(user_windows: pd.DataFrame) -> pd.DataFrame:
    uw = user_windows.sort_values(["uid", "window"]).reset_index(drop=True)
    g = uw.groupby("uid", sort=False)
    hist = pd.DataFrame({
        "uid": uw["uid"].to_numpy(),
        "window": uw["window"].to_numpy(),
        "hist_window": uw["window"].to_numpy(),
        "w_impressions": uw["n_impressions"].to_numpy(),
        "w_clicks": uw["n_clicks"].to_numpy(),
        "w_campaigns": uw["n_campaigns"].to_numpy(),
        "cum_impressions": g["n_impressions"].cumsum().to_numpy(),
        "cum_clicks": g["n_clicks"].cumsum().to_numpy(),
        "cum_conversions": g["n_conversions"].cumsum().to_numpy(),
        "cum_active_windows": g["is_active"].cumsum().to_numpy(),
        "cum_cost": g["cost_sum"].cumsum().to_numpy(),
    })
    first_active = uw.loc[uw["is_active"] == 1].groupby("uid")["window"].min()
    hist["first_active"] = hist["uid"].map(first_active)
    conv_w = uw["window"].where(uw["n_conversions"] > 0)
    hist["last_conv_window"] = conv_w.groupby(uw["uid"]).ffill().to_numpy()
    return hist


def build_position_features(positions: pd.DataFrame, user_windows: pd.DataFrame,
                            window_hours: float) -> pd.DataFrame:
    """Return positions (same rows, same order) with FEATURE_COLUMNS added."""
    hist = _user_history(user_windows).sort_values("window", kind="mergesort")
    pos = positions.reset_index(drop=True).copy()
    pos["_row"] = np.arange(len(pos))
    left = pos[["uid", "window", "_row"]].sort_values("window", kind="mergesort")
    right = hist.rename(columns={"hist_window": "window_h"})
    m = pd.merge_asof(left, right, on="window", by="uid", direction="backward")
    m = m.sort_values("_row").reset_index(drop=True)

    cur = (m["window_h"] == m["window"]).to_numpy()
    t = pos["window"].to_numpy()
    per_day = max(1, int(round(24 / window_hours)))
    out = pos.drop(columns="_row")
    out["cur_impressions"] = np.where(cur, m["w_impressions"], 0).astype("float32")
    out["cur_clicks"] = np.where(cur, m["w_clicks"], 0).astype("float32")
    out["cur_campaigns"] = np.where(cur, m["w_campaigns"], 0).astype("float32")
    for c in ["cum_impressions", "cum_clicks", "cum_conversions", "cum_active_windows", "cum_cost"]:
        out[c] = m[c].fillna(0).to_numpy(dtype="float32")
    # Every position has an impression at or before t, so first_active is always defined.
    out["windows_since_first"] = (t - m["first_active"].to_numpy()).astype("float32")
    since_conv = t - m["last_conv_window"].to_numpy()
    out["windows_since_conversion"] = np.where(np.isnan(since_conv), NO_CONVERSION,
                                               since_conv).astype("float32")
    out["click_rate"] = (out["cum_clicks"] / np.maximum(out["cum_impressions"], 1)).astype("float32")
    out["slot_of_day"] = (t % per_day).astype("float32")
    out["day_index"] = (t // per_day).astype("float32")
    out["windows_since_active"] = out["windows_since_active"].astype("float32")
    out["is_active"] = out["is_active"].astype("float32")
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """log1p on count-like columns; returns float32 [N, len(FEATURE_COLUMNS)]."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype="float32").copy()
    log_cols = [i for i, c in enumerate(FEATURE_COLUMNS)
                if c.startswith(("cur_", "cum_", "windows_since"))]
    X[:, log_cols] = np.log1p(np.maximum(X[:, log_cols], 0))
    return X
