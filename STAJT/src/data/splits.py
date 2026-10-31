"""Dataset split helpers.

Splits operate on *window indices*, not raw rows. A prediction position (uid, t) is
assigned to a split by its **label window** (t + horizon), so every label of a split lies
strictly inside that split's window range and no label ever crosses a split boundary.
Inputs may look back into earlier periods (validation inputs use training-period history),
which is the normal forecasting setup and is not leakage.

Two protocols, matching the proposal:

* Chronological:  train | val | test are contiguous blocks of label windows.
* User-disjoint:  users are hashed into train / val / test groups. Combine with the
                  chronological boundaries via ``user_disjoint_chronological_split`` to get
                  "unseen customers in the future period".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowBoundaries:
    """Label-window ranges: train [first, val_start), val [val_start, test_start), test [test_start, n)."""

    first_label_window: int
    val_start: int
    test_start: int
    n_windows: int

    def as_dict(self) -> dict:
        return {
            "train_label_windows": [self.first_label_window, self.val_start - 1],
            "val_label_windows": [self.val_start, self.test_start - 1],
            "test_label_windows": [self.test_start, self.n_windows - 1],
        }


def window_boundaries(n_windows: int, train_frac: float = 0.70, val_frac: float = 0.15,
                      horizon: int = 1) -> WindowBoundaries:
    """Cut the label windows [horizon, n_windows) into contiguous train/val/test blocks.

    Fractions are of label windows (not rows), so boundaries are identical across models and
    do not move when a filter changes the number of rows. Each block gets at least one window.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Need 0 < train_frac, 0 <= val_frac and train_frac + val_frac < 1.")
    first = horizon
    n_label = n_windows - first
    if n_label < 3:
        raise ValueError(f"Only {n_label} label windows; cannot make three splits.")
    n_train = max(1, int(round(n_label * train_frac)))
    n_val = max(1, int(round(n_label * val_frac))) if val_frac > 0 else 0
    n_train = min(n_train, n_label - n_val - 1)
    val_start = first + n_train
    test_start = val_start + n_val
    return WindowBoundaries(first, val_start, test_start, n_windows)


def assign_chronological_split(df: pd.DataFrame, bounds: WindowBoundaries,
                               label_window_col: str = "label_window") -> pd.Series:
    """Return a 'train'/'val'/'test' label per row based on its label window."""
    lw = df[label_window_col].to_numpy()
    split = np.where(lw < bounds.val_start, "train",
                     np.where(lw < bounds.test_start, "val", "test"))
    split = np.where(lw < bounds.first_label_window, "none", split)
    return pd.Series(split, index=df.index, name="split")


def chronological_split(df: pd.DataFrame, n_windows: int, train_frac: float = 0.70,
                        val_frac: float = 0.15, horizon: int = 1,
                        window_col: str = "window", label_window_col: str = "label_window"):
    """Split prediction positions by label-window index. Returns (train, val, test).

    ``df`` needs either a label-window column or an anchor ``window`` column (label window is
    then ``window + horizon``).
    """
    data = df
    if label_window_col not in data.columns:
        data = data.assign(**{label_window_col: data[window_col] + horizon})
    bounds = window_boundaries(n_windows, train_frac, val_frac, horizon)
    split = assign_chronological_split(data, bounds, label_window_col)
    return tuple(data[split == s].copy() for s in ("train", "val", "test"))


def user_groups(users, train_frac: float = 0.70, val_frac: float = 0.15,
                seed: int = 42) -> pd.Series:
    """Deterministically assign each user to 'train'/'val'/'test' by hashing the id.

    Hash-based (not a shuffle of the observed user list), so a user lands in the same group
    for 6h and 12h windows and after any row filtering.
    """
    users = pd.Index(pd.unique(np.asarray(users)))
    h = pd.util.hash_array(users.to_numpy(), hash_key=f"{seed:016d}"[-16:], categorize=False)
    u = (h % np.uint64(1_000_000)).astype(np.float64) / 1_000_000
    grp = np.where(u < train_frac, "train", np.where(u < train_frac + val_frac, "val", "test"))
    return pd.Series(grp, index=users, name="user_group")


def user_disjoint_split(df: pd.DataFrame, user_col: str = "uid", train_frac: float = 0.70,
                        val_frac: float = 0.15, seed: int = 42):
    """Hold out entire users. Returns (train, val, test) with no user in two parts."""
    groups = user_groups(df[user_col], train_frac, val_frac, seed)
    g = df[user_col].map(groups)
    return tuple(df[g == s].copy() for s in ("train", "val", "test"))


def user_disjoint_chronological_split(df: pd.DataFrame, n_windows: int, user_col: str = "uid",
                                      user_train_frac: float = 0.70, user_val_frac: float = 0.15,
                                      time_train_frac: float = 0.70, time_val_frac: float = 0.15,
                                      horizon: int = 1, seed: int = 42,
                                      label_window_col: str = "label_window"):
    """Unseen users *and* future windows.

    train: train users, train label windows
    val:   val users,   val label windows
    test:  test users,  test label windows
    Rows outside these combinations are dropped, so neither users nor label windows overlap.
    """
    bounds = window_boundaries(n_windows, time_train_frac, time_val_frac, horizon)
    time_split = assign_chronological_split(df, bounds, label_window_col)
    groups = df[user_col].map(user_groups(df[user_col], user_train_frac, user_val_frac, seed))
    return tuple(df[(groups == s) & (time_split == s)].copy() for s in ("train", "val", "test"))


def make_splits(positions: pd.DataFrame, n_windows: int, split_cfg: dict, seed: int = 42,
                horizon: int = 1) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Build every split protocol named in the ``splits`` config section."""
    tf, vf = split_cfg.get("train_frac", 0.70), split_cfg.get("val_frac", 0.15)
    out = {"chronological": chronological_split(positions, n_windows, tf, vf, horizon)}
    if split_cfg.get("user_disjoint_evaluation", False):
        utf = split_cfg.get("user_train_frac", 0.70)
        uvf = split_cfg.get("user_val_frac", 0.15)
        # All windows, users held out (Lu & Kannan-style holdout customers).
        out["user_disjoint"] = user_disjoint_split(positions, "uid", utf, uvf, seed)
        # Stricter: unseen users evaluated only on future label windows.
        out["user_disjoint_future"] = user_disjoint_chronological_split(
            positions, n_windows, user_train_frac=utf, user_val_frac=uvf,
            time_train_frac=tf, time_val_frac=vf, horizon=horizon, seed=seed)
    return out
