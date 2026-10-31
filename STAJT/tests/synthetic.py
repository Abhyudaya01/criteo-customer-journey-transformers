"""Tiny Criteo-shaped data generator for tests and smoke runs (no download needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import CRITEO_COLUMNS

DAY = 86400


def make_synthetic_criteo(n_users: int = 400, n_campaigns: int = 25, days: int = 30,
                          conv_prob: float = 0.08, seed: int = 0) -> pd.DataFrame:
    """Impression rows with the official column names and the same conversion linkage:
    every impression before a user's conversion (within 30 days) carries that conversion."""
    rng = np.random.default_rng(seed)
    horizon = days * DAY
    rows = []
    cid = 0
    for uid in range(n_users):
        n_imp = int(rng.integers(1, 15))
        ts = np.sort(rng.integers(0, horizon, n_imp))
        camps = rng.integers(0, n_campaigns, n_imp) * 1000 + 7  # non-contiguous raw ids
        conv_ts = -1
        if rng.random() < conv_prob * 3:
            conv_ts = int(ts[rng.integers(0, n_imp)] + rng.integers(1, 5 * DAY))
            cid += 1
        for t, c in zip(ts, camps):
            linked = conv_ts >= 0 and t <= conv_ts <= t + 30 * DAY
            click = int(rng.random() < 0.3)
            rows.append({
                "timestamp": int(t), "uid": uid, "campaign": int(c),
                "conversion": int(linked), "conversion_timestamp": conv_ts if linked else -1,
                "conversion_id": cid if linked else -1, "attribution": int(linked and click),
                "click": click, "click_pos": -1, "click_nb": -1,
                "cost": float(rng.random()), "cpo": -1.0, "time_since_last_click": -1,
                **{f"cat{i}": int(rng.integers(0, 50)) for i in range(1, 10)},
            })
    # Anchor the span at [0, 30 days) like the real file, via an extra non-converting user.
    for t in (0, horizon - 1):
        rows.append({**rows[0], "timestamp": t, "uid": n_users, "conversion": 0,
                     "conversion_timestamp": -1, "conversion_id": -1, "attribution": 0})
    df = pd.DataFrame(rows, columns=CRITEO_COLUMNS)
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
