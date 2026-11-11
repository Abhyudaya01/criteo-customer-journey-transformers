"""In-memory window data for the window-based sequence models (iTransformer, Lu & Kannan, PatchTST).

Port of the shared cells of the Colab training notebooks (notebooks/colab/training/), so a
script run and a notebook run see identical inputs:

* one row per prediction position (uid, t); history = the last ``lookback`` windows <= t, built
  from ``user_windows`` (7 activity series) and ``window_campaigns`` (campaign bags);
* static features: recency, time of day / week, plus the past-only history features of
  ``src/data/features.py`` (standardised with chronological-training statistics);
* training sample: every positive plus a random ``neg_rate`` share of negatives, drawn once
  with ``neg_seed``; ``to_true_prob`` undoes the sampling;
* campaign targets and the "seen before" matrix for the new-to-user campaign metrics.

Nothing here reads windows after t: ``make_batch`` masks every lag with window < 0 and only
looks back from t (tests/test_train_transformer.py perturbs future rows to check this).

Window length comes from ``meta.json``. The lookback is given in hours, so a 6-hour-input model
sees the same history span as the 12-hour one (168 h = 14 windows of 12 h or 28 of 6 h).
Calendar features use ``24 / window_hours`` windows per day; for 12-hour windows they are
exactly the notebook's ``t % 2`` and ``t % 14``.
"""

from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

from src.data.features import FEATURE_COLUMNS, build_position_features, feature_matrix
from src.data.io import load_meta, load_shard, shard_ids
from src.evaluation.metrics import campaign_ranking_metrics

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
FEATS = ["log_impressions", "log_clicks", "log_campaigns", "log_cost", "log_conversions",
         "is_active", "has_record"]
POSITION_COLUMNS = ["uid", "window", "label_window", "windows_since_active", "is_active",
                    "y_conversion", "n_next_campaigns", "time_split", "user_group"]


class WindowData:
    def __init__(self, data_dir: Path, lookback_hours: float = 168, neg_rate: float = 0.01,
                 neg_seed: int = 42, data_seed: int = 42, history_features: bool = True,
                 hist_drop=("day_index",), max_shards: int | None = None, log=print):
        t0 = time.time()
        self.data_dir = Path(data_dir)
        self.meta = load_meta(self.data_dir)
        self.n_windows = int(self.meta["n_windows"])
        self.n_campaigns = int(self.meta["n_campaigns"])
        self.window_hours = float(self.meta.get("window_hours", 12))
        self.target_hours = float(self.meta.get("target_hours", self.window_hours))
        lb = lookback_hours / self.window_hours
        if abs(lb - round(lb)) > 1e-9:
            raise ValueError(f"lookback {lookback_hours}h is not a whole number of {self.window_hours:g}h windows")
        self.lookback = int(round(lb))
        self.per_day = max(1, int(round(24 / self.window_hours)))
        self.neg_rate, self.data_seed = neg_rate, data_seed
        self.hist_cols = [c for c in FEATURE_COLUMNS if c not in hist_drop] if history_features else []
        hist_keep = [FEATURE_COLUMNS.index(c) for c in self.hist_cols]

        shards = shard_ids(self.data_dir)[:max_shards]
        pos_parts, uw_parts, wc_parts, ncl_parts, h_parts = [], [], [], [], []
        for s in shards:                                   # tables of one shard share the same users
            tb = load_shard(self.data_dir, s)
            p = tb["positions"]
            keep = POSITION_COLUMNS + [c for c in ("coarse_window",) if c in p.columns]
            p = p[keep]
            u = tb["user_windows"]
            if self.hist_cols:                             # past-only (as-of merge on windows <= t)
                f = build_position_features(p[["uid", "window", "windows_since_active", "is_active"]],
                                            u, self.window_hours)
                h_parts.append(feature_matrix(f)[:, hist_keep].astype(np.float16))
                del f
            pos_parts.append(p)
            uw_parts.append(u)
            wc_parts.append(tb["window_campaigns"][["uid", "window", "campaign_idx", "n_impressions"]])
            ncl_parts.append(tb["next_campaign_labels"][["uid", "window", "campaign_idx"]])
        pos = pd.concat(pos_parts, ignore_index=True)
        uw = pd.concat(uw_parts, ignore_index=True)
        self.H = np.concatenate(h_parts) if self.hist_cols else np.zeros((len(pos), 0), dtype=np.float16)
        del pos_parts, uw_parts, h_parts

        # Contiguous user index and a dense (user, window) -> user_windows row lookup.
        self.uids = np.unique(uw["uid"].to_numpy())
        self.W = 1 << max(6, math.ceil(math.log2(self.n_windows)))
        uw_u = np.searchsorted(self.uids, uw["uid"].to_numpy()).astype(np.int64)
        self.n_uw = len(uw)
        self.dense = np.full(len(self.uids) * self.W, self.n_uw, dtype=np.int32)   # n_uw = no record
        self.dense[uw_u * self.W + uw["window"].to_numpy()] = np.arange(self.n_uw, dtype=np.int32)

        # Per-window activity features (past/present only; conversions on their own timestamp).
        Fw = np.zeros((self.n_uw + 1, len(FEATS)), dtype=np.float32)
        Fw[:-1, 0] = np.log1p(uw["n_impressions"].to_numpy())
        Fw[:-1, 1] = np.log1p(uw["n_clicks"].to_numpy())
        Fw[:-1, 2] = np.log1p(uw["n_campaigns"].to_numpy())
        Fw[:-1, 3] = np.log1p(uw["cost_sum"].to_numpy().clip(min=0))
        Fw[:-1, 4] = np.log1p(uw["n_conversions"].to_numpy())
        Fw[:-1, 5] = uw["is_active"].to_numpy()
        Fw[:-1, 6] = 1.0
        self.Fw = Fw

        pos_u = np.searchsorted(self.uids, pos["uid"].to_numpy())
        if not (self.uids[np.minimum(pos_u, len(self.uids) - 1)] == pos["uid"].to_numpy()).all():
            raise ValueError("a position's user is missing from user_windows")
        self.pos_u = pos_u.astype(np.int64)
        self.pos_w = pos["window"].to_numpy().astype(np.int64)
        self.pos_uid = pos["uid"].to_numpy().astype(np.int64)
        self.pos_key_window = self._prediction_windows(pos)
        self.pos_wsa = pos["windows_since_active"].to_numpy().astype(np.float32)
        self.n_next = pos["n_next_campaigns"].to_numpy()
        self.y = pos["y_conversion"].to_numpy().astype(np.int8)
        self.time_split = _split_codes(pos["time_split"])
        self.user_group = _split_codes(pos["user_group"])
        self.n_pos = len(pos)
        del pos, uw, uw_u

        self.lags = np.arange(self.lookback - 1, -1, -1)              # oldest ... newest (t)
        rng = np.random.default_rng(neg_seed)
        self.keep_neg = rng.random(self.n_pos) < neg_rate             # one negative sample per seed

        # Standardise history features with (sampled) chronological-training rows only.
        ref = np.flatnonzero((self.time_split == SPLIT_CODE["train"]) & self.keep_neg)
        if self.H.shape[1]:
            self.H_mu = self.H[ref].astype(np.float32).mean(0)
            self.H_sd = self.H[ref].astype(np.float32).std(0) + 1e-6
        else:
            self.H_mu, self.H_sd = np.zeros(0, np.float32), np.ones(0, np.float32)
        self.n_static = 4 + self.H.shape[1]

        self._build_campaign_tables(wc_parts, ncl_parts)
        log(f"{self.data_dir.name}: {self.n_pos:,} positions | {int(self.y.sum()):,} positives | "
            f"{len(self.uids):,} users | {self.n_uw:,} user-windows | {self.window_hours:g}h windows, "
            f"lookback {self.lookback} | history features: {self.hist_cols or 'off'} | "
            f"loaded in {time.time() - t0:.0f}s")

    # --- keys -----------------------------------------------------------------------------
    def _prediction_windows(self, pos: pd.DataFrame) -> np.ndarray:
        """Anchor window in target-window units (12h keys for the 6h-input / 12h-target data)."""
        w = pos["window"].to_numpy().astype(np.int64)
        if self.target_hours == self.window_hours:
            return w
        from src.data.multiscale import to_coarse_keys, window_ratio
        k = to_coarse_keys(w, window_ratio(self.target_hours, self.window_hours)).astype(np.int64)
        if "coarse_window" in pos.columns and not (k == pos["coarse_window"].to_numpy()).all():
            raise ValueError("re-anchored windows do not map back to coarse_window")
        return k

    # --- splits and samples ---------------------------------------------------------------
    def split_masks(self, protocol: str) -> dict:
        if protocol == "chronological":
            return {s: self.time_split == c for s, c in SPLIT_CODE.items()}
        if protocol == "user_disjoint":                    # unseen users, all windows (run_baselines.py)
            return {s: self.user_group == c for s, c in SPLIT_CODE.items()}
        if protocol == "user_disjoint_future":             # unseen users and future windows
            return {s: (self.time_split == c) & (self.user_group == c) for s, c in SPLIT_CODE.items()}
        raise ValueError(protocol)

    def train_sample(self, masks: dict) -> np.ndarray:
        return np.flatnonzero(masks["train"] & ((self.y == 1) | self.keep_neg))

    def val_sample(self, val_idx: np.ndarray) -> np.ndarray:
        """Validation rows used for early stopping (same sampling as training)."""
        return val_idx[(self.y[val_idx] == 1) | self.keep_neg[val_idx]]

    def to_true_prob(self, p: np.ndarray) -> np.ndarray:
        """Undo negative subsampling: true odds = sampled odds * neg_rate."""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return p * self.neg_rate / (p * self.neg_rate + 1 - p)

    def campaign_sample(self, masks: dict, max_n: int) -> np.ndarray:
        cand = np.flatnonzero(masks["test"] & (self.n_next > 0))
        rs = np.random.default_rng(self.data_seed)
        return np.sort(rs.choice(cand, size=min(max_n, len(cand)), replace=False))

    # --- features -------------------------------------------------------------------------
    def static_features(self, idx: np.ndarray) -> np.ndarray:
        t = self.pos_w[idx]
        week = 7 * self.per_day
        slot = (t % self.per_day).astype(np.float32) / max(1, self.per_day - 1)
        base = np.stack([self.pos_wsa[idx] / self.lookback, slot,
                         np.sin(2 * np.pi * (t % week) / week).astype(np.float32),
                         np.cos(2 * np.pi * (t % week) / week).astype(np.float32)], axis=1)
        if not self.H.shape[1]:
            return base.astype(np.float32)
        return np.concatenate([base, (self.H[idx].astype(np.float32) - self.H_mu) / self.H_sd],
                              axis=1).astype(np.float32)

    def history_rows(self, idx: np.ndarray):
        """(n, lookback) user_windows rows for windows t-lookback+1 .. t (n_uw = no record)."""
        u, t = self.pos_u[idx], self.pos_w[idx]
        w = t[:, None] - self.lags[None, :]
        rows = self.dense[u[:, None] * self.W + np.clip(w, 0, None)]
        rows[w < 0] = self.n_uw
        return rows, w

    # --- campaigns ------------------------------------------------------------------------
    def _build_campaign_tables(self, wc_parts, ncl_parts) -> None:
        wc = pd.concat(wc_parts, ignore_index=True)
        wc_row = self.dense[np.searchsorted(self.uids, wc["uid"].to_numpy()).astype(np.int64) * self.W
                            + wc["window"].to_numpy()]
        if (wc_row >= self.n_uw).any():
            raise ValueError("window_campaigns row without a user_windows row")
        order = np.argsort(wc_row, kind="stable")
        self.camp_idx = wc["campaign_idx"].to_numpy()[order].astype(np.int64)
        self.camp_wt = np.log1p(wc["n_impressions"].to_numpy()[order]).astype(np.float32)
        self.camp_off = np.searchsorted(wc_row[order], np.arange(self.n_uw + 2)).astype(np.int64)
        ncl = pd.concat(ncl_parts, ignore_index=True)
        key = np.searchsorted(self.uids, ncl["uid"].to_numpy()).astype(np.int64) * self.W + ncl["window"].to_numpy()
        o = np.argsort(key, kind="stable")
        self.ncl_key, self.ncl_camp = key[o], ncl["campaign_idx"].to_numpy()[o].astype(np.int64)

    def campaign_bags(self, rows: np.ndarray):
        """Flattened EmbeddingBag inputs (campaign, weight, offsets, lengths) for user_windows rows."""
        starts, ends = self.camp_off[rows], self.camp_off[rows + 1]
        lens = ends - starts
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        j = np.repeat(starts - offsets, lens) + np.arange(int(lens.sum()))
        return self.camp_idx[j], self.camp_wt[j], offsets, lens

    def next_campaigns(self, idx: np.ndarray):
        """(row_in_batch, campaign) pairs for the campaigns in each position's label window."""
        keys = self.pos_u[idx] * self.W + self.pos_w[idx]
        lo, hi = np.searchsorted(self.ncl_key, keys, "left"), np.searchsorted(self.ncl_key, keys, "right")
        lens = hi - lo
        rows = np.repeat(np.arange(len(idx)), lens)
        j = np.repeat(lo - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(int(lens.sum()))
        return rows, self.ncl_camp[j]

    def seen_matrix(self, idx: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        """True where the user saw the campaign in any window <= t ("not new")."""
        seen = np.zeros((len(idx), self.n_campaigns), dtype=bool)
        wins = np.arange(self.n_windows)
        for s in range(0, len(idx), chunk):
            j = idx[s:s + chunk]
            rows = self.dense[self.pos_u[j][:, None] * self.W + wins[None, :]]
            rows[wins[None, :] > self.pos_w[j][:, None]] = self.n_uw
            camp, _, _, lens = self.campaign_bags(rows.reshape(-1))
            seen[s + np.repeat(np.arange(rows.size), lens) // self.n_windows, camp] = True
        return seen

    def ranking_all_and_new(self, y_true: np.ndarray, seen: np.ndarray, scores: np.ndarray) -> dict:
        out = {k: v for k, v in campaign_ranking_metrics(y_true, scores, ks=(1, 5, 10)).items()
               if k != "frac_positions_nonempty"}
        y_new = (y_true.astype(bool) & ~seen).astype(np.int8)
        keep = y_new.sum(1) > 0
        new = campaign_ranking_metrics(y_new[keep], scores[keep], ks=(1, 5, 10)) if keep.any() else {}
        out.update({f"new_{k}": v for k, v in new.items() if k != "frac_positions_nonempty"})
        out["frac_positions_with_new_campaign"] = float(keep.mean())
        out["frac_next_campaigns_new"] = float(y_new.sum() / max(y_true.sum(), 1))
        return out

    # --- batches --------------------------------------------------------------------------
    def make_batch(self, idx: np.ndarray, with_labels: bool = False, device: str = "cpu") -> dict:
        """Tensors for one batch: activity series + calendar, campaign bags, static features,
        padding mask, user index and the recency-weighted campaign history term."""
        import torch
        rows, w = self.history_rows(idx)
        week = 7 * self.per_day
        cal = np.stack([(w % self.per_day) / max(1, self.per_day - 1),
                        np.sin(2 * np.pi * (w % week) / week),
                        np.cos(2 * np.pi * (w % week) / week)], -1)
        Xnum = np.concatenate([self.Fw[rows], cal.astype(np.float32)], axis=-1)
        S = self.static_features(idx)
        camp, wt, off, lens = self.campaign_bags(rows.reshape(-1))
        elem_row = np.repeat(np.arange(rows.size), lens)
        # 0.8 per 12 hours, whatever the window length (0.8 per window for 12h windows).
        decay = 0.8 ** (self.lags * self.window_hours / 12.0)
        elem_w = wt * decay[elem_row % self.lookback].astype(np.float32)
        b = {"x": torch.from_numpy(Xnum), "camp": torch.from_numpy(camp), "wt": torch.from_numpy(wt),
             "off": torch.from_numpy(off), "s": torch.from_numpy(S), "pad": torch.from_numpy(w < 0),
             "u": torch.from_numpy(self.pos_u[idx]), "h_row": torch.from_numpy(elem_row // self.lookback),
             "h_w": torch.from_numpy(elem_w)}
        if with_labels:
            b["y"] = torch.from_numpy(self.y[idx].astype(np.float32))
            r, c = self.next_campaigns(idx)
            b["yc_r"], b["yc_c"] = torch.from_numpy(r), torch.from_numpy(c)
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    def top_campaigns(self, tr: np.ndarray, k: int, max_rows: int = 1_000_000) -> np.ndarray:
        """Most frequent campaigns (by log-impressions) in the input windows of a training sample."""
        j = tr if len(tr) <= max_rows else np.random.default_rng(self.data_seed).choice(
            tr, size=max_rows, replace=False)
        rows, _ = self.history_rows(j)
        camp, wt, _, _ = self.campaign_bags(rows.reshape(-1))
        counts = np.bincount(camp, weights=wt, minlength=self.n_campaigns)
        return np.argsort(-counts, kind="stable")[:min(k, self.n_campaigns)]


def _split_codes(col: pd.Series) -> np.ndarray:
    """Compact int8 codes (0 train, 1 val, 2 test); avoids 126M Python strings in memory."""
    c = col.astype(str).astype("category")
    lut = np.array([SPLIT_CODE.get(str(k), -1) for k in c.cat.categories], dtype=np.int8)
    return lut[c.cat.codes.to_numpy()]
