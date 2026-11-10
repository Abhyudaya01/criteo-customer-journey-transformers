"""Event-level histories for DARNN, built on STAJT's event loader.

Final Stretch Plan (Should, Rachel, Nov 22): DARNN under *our* chronological protocol only,
using STAJT's event loader. "Same loader" means DARNN sees exactly what STAJT sees:

* the same event cache: ``STAJT/data/stajt_full_eval_cache/events/part-XXX.parquet`` (every
  user, 64 shards, written by STAJT/scripts/evaluate_stajt_full.py) with training-period
  vocabularies (campaign and cat1..cat9 ids; 0 = unseen after the training cutoff);
* the same prediction anchors: label window * window seconds, so history is every event
  strictly before the label window starts (windows <= t);
* the same history window: the last ``MAX_LEN = 64`` events, left-padded;
* the same training sample: STAJT's cached team-aligned anchors (all positives + 1% of
  negatives), so DARNN trains on identical examples.

What changes for DARNN is the *role of the click*. DARNN (Ren et al., 2018) encodes impression
features x_j with an LSTM and trains a decoder to predict each click z_j from the previous
clicks z_<j; impression-level and click-level attention are then combined for conversion.
So the click flag must not be an input feature here (STAJT feeds it in through
``raw_action_id`` / ``behavior_family``). ``darnn_item`` returns impression features without
it and the click sequence separately. See docs/darnn_protocol.md for the adaptations.

The core is NumPy only; ``make_darnn_dataset`` imports torch lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LEN = 64
CAT_ID_COLS = [f"cat{i}_id" for i in range(1, 10)]
EVENT_COLS = [
    "uid", "timestamp", "entity_id", "raw_action_id", "behavior_family",
    "log_cost", "log_delta_hours", "entity_seen_before", "log_entity_gap_hours", *CAT_ID_COLS,
]
CLICK_ACTION_ID = 2  # STAJT: raw_action_id = 2 if click else 1
# Same order as STAJT's numeric input. None of these depends on the click flag:
#   log_cost, log hours from event to anchor, log hours since the previous event,
#   campaign seen before, log hours since this campaign was last seen.
NUMERIC_FEATURES = ["log_cost", "log_age_hours", "log_delta_hours", "entity_seen_before",
                    "log_entity_gap_hours"]


class EventStore:
    """Per-user event arrays sorted by (uid, timestamp), as in STAJT's EventStore."""

    def __init__(self, events: pd.DataFrame):
        missing = set(EVENT_COLS) - set(events.columns)
        if missing:
            raise ValueError(f"event table is missing {sorted(missing)}")
        ev = events
        du = np.diff(ev["uid"].to_numpy(np.int64))
        dt = np.diff(ev["timestamp"].to_numpy(np.int64))
        if not ((du > 0) | ((du == 0) & (dt >= 0))).all():
            ev = ev.sort_values(["uid", "timestamp"], kind="mergesort").reset_index(drop=True)
        self.uid = ev["uid"].to_numpy(np.int64)
        self.timestamp = ev["timestamp"].to_numpy(np.int64)
        self.entity_id = ev["entity_id"].to_numpy(np.int64)
        self.click = (ev["raw_action_id"].to_numpy() == CLICK_ACTION_ID).astype(np.float32)
        self.log_cost = ev["log_cost"].to_numpy(np.float32)
        self.log_delta_hours = ev["log_delta_hours"].to_numpy(np.float32)
        self.entity_seen_before = ev["entity_seen_before"].to_numpy(np.float32)
        self.log_entity_gap_hours = ev["log_entity_gap_hours"].to_numpy(np.float32)
        self.context = ev[CAT_ID_COLS].to_numpy(np.int64)
        users, starts, counts = np.unique(self.uid, return_index=True, return_counts=True)
        self.ranges = {int(u): (int(s), int(s + c)) for u, s, c in zip(users, starts, counts)}

    @classmethod
    def from_parquet(cls, path: Path | str) -> "EventStore":
        return cls(pd.read_parquet(path, columns=EVENT_COLS))


def history_bounds(store: EventStore, uid: int, anchor_ts: int, max_len: int = MAX_LEN) -> tuple[int, int]:
    """Row range [s, e) of the last ``max_len`` events strictly before ``anchor_ts``."""
    s, e = store.ranges.get(int(uid), (0, 0))
    end = s + int(np.searchsorted(store.timestamp[s:e], anchor_ts, side="left"))
    return max(s, end - max_len), end


def darnn_item(store: EventStore, uid: int, anchor_ts: int, y: float,
               max_len: int = MAX_LEN) -> dict:
    """One prediction position as DARNN inputs (left-padded to ``max_len``, oldest first).

    campaign  int64   [L]     campaign id (0 = padding or unseen campaign)
    context   int64   [L, 9]  cat1..cat9 ids
    numeric   float32 [L, 5]  NUMERIC_FEATURES (no click information)
    click     float32 [L]     z_j: 1 if event j was clicked (decoder target and decoder input)
    mask      bool    [L]     True on real events
    n_events  int             number of real events (0 = no history before the anchor)
    y         float32         conversion in the label window
    """
    s, e = history_bounds(store, uid, anchor_ts, max_len)
    n = e - s
    L = max_len
    out = {
        "campaign": np.zeros(L, np.int64),
        "context": np.zeros((L, len(CAT_ID_COLS)), np.int64),
        "numeric": np.zeros((L, len(NUMERIC_FEATURES)), np.float32),
        "click": np.zeros(L, np.float32),
        "mask": np.zeros(L, bool),
        "n_events": int(n),
        "y": np.float32(y),
    }
    if n > 0:
        off, sl = L - n, slice(s, e)
        ts = store.timestamp[sl]
        out["campaign"][off:] = store.entity_id[sl]
        out["context"][off:] = store.context[sl]
        out["numeric"][off:, 0] = store.log_cost[sl]
        out["numeric"][off:, 1] = np.log1p(np.maximum((anchor_ts - ts) / 3600.0, 0.0))
        out["numeric"][off:, 2] = store.log_delta_hours[sl]
        out["numeric"][off:, 3] = store.entity_seen_before[sl]
        out["numeric"][off:, 4] = store.log_entity_gap_hours[sl]
        out["click"][off:] = store.click[sl]
        out["mask"][off:] = True
    return out


# ------------------------------------------------------------------ anchors

def anchor_timestamps(label_window, window_hours: float = 12) -> np.ndarray:
    """Start of the label window: history = events strictly before it (windows <= t)."""
    return np.asarray(label_window, dtype=np.int64) * int(round(window_hours * 3600))


@dataclass
class Anchors:
    uid: np.ndarray
    anchor_ts: np.ndarray
    y: np.ndarray
    window: np.ndarray  # anchor window t, for saving predictions keyed (uid, window)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, window_hours: float = 12) -> "Anchors":
        lw = df["label_window"].to_numpy(np.int64)
        ts = (df["anchor_timestamp"].to_numpy(np.int64) if "anchor_timestamp" in df.columns
              else anchor_timestamps(lw, window_hours))
        if not np.array_equal(ts, anchor_timestamps(lw, window_hours)):
            raise ValueError("anchor_timestamp != label_window * window seconds")
        return cls(df["uid"].to_numpy(np.int64), ts, df["y_conversion"].to_numpy(np.float32),
                   (lw - 1).astype(np.int64))

    def __len__(self) -> int:
        return len(self.uid)


def split_anchors(positions: pd.DataFrame, split: str, column: str = "time_split",
                  window_hours: float = 12) -> Anchors:
    """Every position of one split (for validation/test scoring)."""
    df = positions.loc[positions[column].astype(str) == split, ["uid", "label_window", "y_conversion"]]
    return Anchors.from_frame(df.reset_index(drop=True), window_hours)


def load_stajt_train_anchors(cache_dir: Path | str, neg_rate: float = 0.01,
                             window_hours: float = 12) -> Anchors:
    """STAJT's team-aligned training anchors (train_stajt_fair_chrono.py cache), unchanged."""
    path = Path(cache_dir) / f"train_anchors_neg_rate_{neg_rate:.4f}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run STAJT/scripts/train_stajt_fair_chrono.py's "
                                "anchor step first, or point at its cache directory")
    return Anchors.from_frame(pd.read_parquet(path), window_hours)


# ------------------------------------------------------------------ torch glue (lazy)

class DarnnDataset:
    """Map-style dataset of ``darnn_item`` dicts as tensors; DataLoader's default collate stacks them.

    Defined at module level (not inside a function) so DataLoader workers can be started with
    any multiprocessing method, not only fork.
    """

    def __init__(self, store: EventStore, anchors: Anchors, max_len: int = MAX_LEN):
        self.store, self.anchors, self.max_len = store, anchors, max_len

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> dict:
        import torch
        a = self.anchors
        it = darnn_item(self.store, a.uid[i], a.anchor_ts[i], a.y[i], self.max_len)
        return {k: (torch.tensor(v) if isinstance(v, int) else torch.as_tensor(v)) for k, v in it.items()}


def make_darnn_dataset(store: EventStore, anchors: Anchors, max_len: int = MAX_LEN) -> DarnnDataset:
    return DarnnDataset(store, anchors, max_len)
