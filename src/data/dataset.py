"""Per-user journey sequences for sequence models (LSTM, Lu & Kannan Transformer, iTransformer).

One item = one user's full calendar of T windows (T = 62 at 12h, 124 at 6h):

    numeric       float32 [T, 6]  log1p(impressions, clicks, #campaigns, cost), converted, active
    camp_window   int64   [n]     window of each (window, campaign) token entry
    camp_idx      int64   [n]     campaign index 0..C-1
    camp_weight   float32 [n]     log1p(impressions of that campaign in that window)
    y_conversion  float32 [T]     label at t = conversion in window t+1
    mask          bool    [T]     True where (user, t) is a prediction position of the chosen split
    next_window   int64   [m]     sparse multi-hot campaign targets: (t, campaign) pairs
    next_campaign int64   [m]

IMPORTANT: inputs cover all T windows, including windows after t. A model trained on this
must use causal (lower-triangular) attention / a left-to-right RNN so the prediction at t only
sees windows <= t. ``causal_mask(T)`` returns the mask to pass to nn.MultiheadAttention.

Loss and metrics are computed only where ``mask`` is True. With the chronological protocol,
validation users still feed their training-period windows as context; only val label windows
are scored, which is the standard forecasting setup.

The numpy core needs no PyTorch; ``JourneyTorchDataset`` and ``collate_journeys`` import it
lazily so this module also works in environments without torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import load_meta, load_shard, shard_ids

NUMERIC_FEATURES = ["log_impressions", "log_clicks", "log_campaigns", "log_cost",
                    "converted", "is_active"]
PROTOCOL_COLUMN = {"chronological": "time_split", "user_disjoint": "user_group"}


def _csr(uids_sorted: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row ranges [start, end) of each user in a uid-sorted table."""
    start = np.searchsorted(uids_sorted, users, side="left")
    end = np.searchsorted(uids_sorted, users, side="right")
    return start, end


class JourneySequences:
    """Framework-agnostic store of user sequences for one protocol/split."""

    def __init__(self, tables: dict, n_windows: int, n_campaigns: int,
                 protocol: str = "chronological", split: str = "train",
                 max_users: int | None = None, seed: int = 42):
        self.T, self.C = int(n_windows), int(n_campaigns)
        self.protocol, self.split = protocol, split
        pos = tables["positions"]
        col = PROTOCOL_COLUMN[protocol]
        pos = pos[pos[col].astype(str) == split]
        users = np.sort(pos["uid"].unique())
        if max_users is not None and len(users) > max_users:
            users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
        self.users = users

        def prep(df, cols):
            df = df[df["uid"].isin(users)].sort_values(["uid", "window"], kind="mergesort")
            return {c: df[c].to_numpy() for c in ["uid", *cols]}

        uw = tables["user_windows"]
        self._uw = prep(uw, ["window", "n_impressions", "n_clicks", "n_campaigns", "cost_sum",
                             "n_conversions", "is_active"])
        self._wc = prep(tables["window_campaigns"], ["window", "campaign_idx", "n_impressions"])
        self._pos = prep(pos, ["window", "y_conversion"])
        self._nx = prep(tables["next_campaign_labels"].merge(
            pos[["uid", "window"]], on=["uid", "window"], how="inner"), ["window", "campaign_idx"])
        self._ranges = {k: _csr(getattr(self, k)["uid"], users)
                        for k in ("_uw", "_wc", "_pos", "_nx")}

    def __len__(self) -> int:
        return len(self.users)

    def _rows(self, key: str, i: int) -> slice:
        s, e = self._ranges[key]
        return slice(int(s[i]), int(e[i]))

    def __getitem__(self, i: int) -> dict:
        T = self.T
        r = self._rows("_uw", i)
        uw = {k: v[r] for k, v in self._uw.items()}
        w = uw["window"]
        numeric = np.zeros((T, len(NUMERIC_FEATURES)), dtype=np.float32)
        numeric[w, 0] = np.log1p(uw["n_impressions"])
        numeric[w, 1] = np.log1p(uw["n_clicks"])
        numeric[w, 2] = np.log1p(uw["n_campaigns"])
        numeric[w, 3] = np.log1p(np.maximum(uw["cost_sum"], 0) * 1000.0)  # cost is tiny/transformed
        numeric[w, 4] = (uw["n_conversions"] > 0)
        numeric[w, 5] = uw["is_active"]

        r = self._rows("_wc", i)
        r_pos = self._rows("_pos", i)
        y = np.zeros(T, dtype=np.float32)
        mask = np.zeros(T, dtype=bool)
        pw = self._pos["window"][r_pos]
        y[pw] = self._pos["y_conversion"][r_pos]
        mask[pw] = True
        r_nx = self._rows("_nx", i)
        return {
            "uid": int(self.users[i]),
            "numeric": numeric,
            "camp_window": self._wc["window"][r].astype(np.int64),
            "camp_idx": self._wc["campaign_idx"][r].astype(np.int64),
            "camp_weight": np.log1p(self._wc["n_impressions"][r]).astype(np.float32),
            "y_conversion": y,
            "mask": mask,
            "next_window": self._nx["window"][r_nx].astype(np.int64),
            "next_campaign": self._nx["campaign_idx"][r_nx].astype(np.int64),
        }

    def positive_rate(self) -> float:
        return float(self._pos["y_conversion"].mean()) if len(self._pos["y_conversion"]) else 0.0


def load_sequences(data_dir: Path, protocol: str = "chronological", split: str = "train",
                   shards: list[int] | None = None, max_users: int | None = None,
                   seed: int = 42) -> JourneySequences:
    """Load one or more user shards from a processed dataset folder."""
    meta = load_meta(data_dir)
    shards = shards if shards is not None else shard_ids(data_dir)
    parts = [load_shard(data_dir, s) for s in shards]
    tables = {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}
    return JourneySequences(tables, meta["n_windows"], meta["n_campaigns"], protocol, split,
                            max_users, seed)


# ------------------------------------------------------------------ PyTorch glue (lazy)

def causal_mask(T: int):
    """Boolean [T, T] mask for nn.MultiheadAttention: True = position may NOT be attended."""
    import torch
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)


def collate_journeys(items: list[dict], n_campaigns: int, dense_campaign_targets: bool = True) -> dict:
    """Batch items into tensors.

    Returns numeric [B,T,F], mask [B,T], y_conversion [B,T], y_campaign [B,T,C] (dense multi-hot,
    or sparse index lists if dense_campaign_targets=False), and EmbeddingBag inputs
    bag_indices / bag_offsets / bag_weights with B*T bags in (b, t) order:

        emb = nn.EmbeddingBag(C, d, mode="sum")
        x = emb(bag_indices, bag_offsets, per_sample_weights=bag_weights).view(B, T, d)
    """
    import torch
    B, T = len(items), items[0]["numeric"].shape[0]
    idx, wts, counts = [], [], np.zeros((B, T), dtype=np.int64)
    for b, it in enumerate(items):
        order = np.argsort(it["camp_window"], kind="stable")
        idx.append(it["camp_idx"][order])
        wts.append(it["camp_weight"][order])
        counts[b] = np.bincount(it["camp_window"], minlength=T)
    offsets = np.concatenate([[0], np.cumsum(counts.ravel())[:-1]])
    out = {
        "uid": torch.tensor([it["uid"] for it in items]),
        "numeric": torch.from_numpy(np.stack([it["numeric"] for it in items])),
        "mask": torch.from_numpy(np.stack([it["mask"] for it in items])),
        "y_conversion": torch.from_numpy(np.stack([it["y_conversion"] for it in items])),
        "bag_indices": torch.from_numpy(np.concatenate(idx) if idx else np.zeros(0, np.int64)),
        "bag_offsets": torch.from_numpy(offsets.astype(np.int64)),
        "bag_weights": torch.from_numpy(np.concatenate(wts) if wts else np.zeros(0, np.float32)),
    }
    if dense_campaign_targets:
        y = np.zeros((B, T, n_campaigns), dtype=np.float32)
        for b, it in enumerate(items):
            y[b, it["next_window"], it["next_campaign"]] = 1.0
        out["y_campaign"] = torch.from_numpy(y)
    else:
        out["y_campaign_sparse"] = [(it["next_window"], it["next_campaign"]) for it in items]
    return out


def make_torch_dataset(seqs: JourneySequences):
    """Wrap a JourneySequences as a torch Dataset (use with collate_journeys)."""
    from torch.utils.data import Dataset

    class JourneyTorchDataset(Dataset):
        def __len__(self):
            return len(seqs)

        def __getitem__(self, i):
            return seqs[i]

    return JourneyTorchDataset()


def make_loader(seqs: JourneySequences, batch_size: int = 256, shuffle: bool = True,
                num_workers: int = 0, dense_campaign_targets: bool = True):
    from functools import partial
    from torch.utils.data import DataLoader
    return DataLoader(make_torch_dataset(seqs), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=partial(collate_journeys, n_campaigns=seqs.C,
                                         dense_campaign_targets=dense_campaign_targets))
