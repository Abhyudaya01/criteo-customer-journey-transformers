"""Criteo Attribution preprocessing: raw impressions -> windowed customer journeys + labels.

Pipeline (all pure DataFrame -> DataFrame functions, so they are easy to test):

    load_criteo            read the raw TSV(.gz), keep only the columns we need, cast dtypes
    validate_schema        fail loudly if the file does not match configs/base.yaml
    add_windows            window index per impression and per conversion
    build_campaign_vocab   raw campaign id -> contiguous index
    build_window_campaigns sparse (uid, window, campaign_idx) token contents
    build_conversion_events one row per distinct conversion (deduplicated across impressions)
    build_user_windows     per (uid, window) aggregate features
    build_positions        prediction positions t, with next-window conversion label (t+1)
    build_next_campaign_labels  sparse multi-label targets: campaigns seen in window t+1

Leakage rules (important, see docs/design_notes.md):
    * ``conversion``, ``conversion_timestamp``, ``conversion_id``, ``attribution``,
      ``click_pos``, ``click_nb`` and ``cpo`` on an impression row describe a conversion
      that can happen up to 30 days *after* the impression. They are only ever used to
      build labels (and past-conversion features keyed on the conversion's own timestamp),
      never as features of the impression window.
    * Prediction positions are chosen from past activity only (a user is "at risk" at t if
      they had activity in the last ``max_inactive_windows`` windows), never from labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd

# Documented column order of criteo_attribution_dataset.tsv.gz (tab separated, gzip).
CRITEO_COLUMNS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click", "click_pos", "click_nb", "cost", "cpo", "time_since_last_click",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
]

# Columns that describe the *future* conversion linked to an impression. Never use as inputs.
LEAKY_COLUMNS = [
    "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click_pos", "click_nb", "cpo",
]

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class DataColumns:
    """Column names, read from the ``data`` section of configs/base.yaml."""

    user: str = "uid"
    timestamp: str = "timestamp"
    campaign: str = "campaign"
    conversion: str = "conversion"
    conversion_timestamp: str = "conversion_timestamp"
    conversion_id: str = "conversion_id"
    click: str = "click"
    cost: str = "cost"

    @classmethod
    def from_config(cls, data_cfg: dict) -> "DataColumns":
        return cls(
            user=data_cfg.get("user_column", "uid"),
            timestamp=data_cfg.get("timestamp_column", "timestamp"),
            campaign=data_cfg.get("campaign_column", "campaign"),
            conversion=data_cfg.get("conversion_column", "conversion"),
            conversion_timestamp=data_cfg.get("conversion_timestamp_column", "conversion_timestamp"),
            conversion_id=data_cfg.get("conversion_id_column", "conversion_id"),
            click=data_cfg.get("click_column", "click"),
            cost=data_cfg.get("cost_column", "cost"),
        )

    def required(self) -> list[str]:
        return [self.timestamp, self.user, self.campaign, self.conversion,
                self.conversion_timestamp, self.conversion_id, self.click, self.cost]


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------

def load_criteo(path: Path, sep: str = "\t", cols: DataColumns | None = None,
                nrows: int | None = None, extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Load the raw Criteo attribution file (``.tsv`` or ``.tsv.gz``; compression inferred).

    Only the columns the pipeline needs are read, with compact dtypes, so the full
    16.5M-row file fits comfortably in a 16 GB job.
    """
    cols = cols or DataColumns()
    usecols = list(dict.fromkeys(cols.required() + (extra_columns or [])))
    dtypes = {
        cols.timestamp: "int64", cols.user: "int64", cols.campaign: "int64",
        cols.conversion: "int8", cols.conversion_timestamp: "int64",
        cols.conversion_id: "int64", cols.click: "int8", cols.cost: "float32",
    }
    header = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
    validate_schema(header, cols)
    return pd.read_csv(
        path, sep=sep, usecols=usecols, nrows=nrows,
        dtype={k: v for k, v in dtypes.items() if k in usecols},
    )


def validate_schema(columns: list[str], cols: DataColumns | None = None) -> None:
    """Raise with a helpful message if configured columns are missing from the file."""
    cols = cols or DataColumns()
    missing = [c for c in cols.required() if c not in columns]
    if missing:
        hint = ""
        if len(columns) == 1:
            hint = (" The header parsed as a single column, so the separator is probably "
                    "wrong (the official file is tab separated: sep='\\t').")
        raise ValueError(f"Columns {missing} not found in file header {columns}.{hint}")


def infer_timestamp_unit(df: pd.DataFrame, cols: DataColumns | None = None,
                         expected_days: float = 30.0) -> str:
    """Return 'seconds' if the impression span looks like ~30 days of seconds.

    The dataset page documents ``time_since_last_click`` in seconds and says timestamps
    start at 0, but does not state the unit of ``timestamp``. We check it rather than assume.
    """
    cols = cols or DataColumns()
    span_days = (df[cols.timestamp].max() - df[cols.timestamp].min()) / 86400
    if 0.5 * expected_days <= span_days <= 1.5 * expected_days:
        return "seconds"
    raise ValueError(
        f"Impression span is {span_days:.2f} days if timestamps are seconds; expected about "
        f"{expected_days}. Check the timestamp unit before windowing."
    )


def sort_events(df: pd.DataFrame, cols: DataColumns | None = None) -> pd.DataFrame:
    """Sort customer events chronologically within user (stable, so ties keep file order)."""
    cols = cols or DataColumns()
    return df.sort_values([cols.user, cols.timestamp], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------------------

def window_seconds(hours: float) -> int:
    return int(round(hours * SECONDS_PER_HOUR))


def n_windows_for(df: pd.DataFrame, hours: float, cols: DataColumns | None = None) -> int:
    """Number of global windows covering all impressions (window 0 starts at timestamp 0)."""
    cols = cols or DataColumns()
    return int(df[cols.timestamp].max() // window_seconds(hours)) + 1


def add_windows(df: pd.DataFrame, hours: float, cols: DataColumns | None = None) -> pd.DataFrame:
    """Add ``window`` (impression window) and ``conversion_window`` (-1 if no conversion).

    Windows are global and aligned to timestamp 0 so every user shares the same calendar,
    which is what makes chronological splitting by window index possible.
    """
    cols = cols or DataColumns()
    w = window_seconds(hours)
    out = df.copy()
    out["window"] = (out[cols.timestamp] // w).astype("int32")
    conv_ts = out[cols.conversion_timestamp]
    out["conversion_window"] = np.where(conv_ts >= 0, conv_ts // w, -1).astype("int32")
    return out


def build_campaign_vocab(df: pd.DataFrame, cols: DataColumns | None = None) -> pd.DataFrame:
    """Map raw campaign ids to contiguous indices 0..C-1 (ordered by raw id for stability)."""
    cols = cols or DataColumns()
    counts = df.groupby(cols.campaign).size().rename("n_impressions").reset_index()
    counts = counts.sort_values(cols.campaign).reset_index(drop=True)
    counts["campaign_idx"] = np.arange(len(counts), dtype="int32")
    return counts.rename(columns={cols.campaign: "campaign"})[["campaign", "campaign_idx", "n_impressions"]]


def build_window_campaigns(df: pd.DataFrame, vocab: pd.DataFrame,
                           cols: DataColumns | None = None) -> pd.DataFrame:
    """Sparse token contents: one row per (uid, window, campaign) with impression/click counts."""
    cols = cols or DataColumns()
    mapping = pd.Series(vocab["campaign_idx"].values, index=vocab["campaign"].values)
    tmp = pd.DataFrame({
        "uid": df[cols.user].values,
        "window": df["window"].values,
        "campaign_idx": df[cols.campaign].map(mapping).astype("int32").values,
        "click": df[cols.click].values,
    })
    out = (tmp.groupby(["uid", "window", "campaign_idx"], sort=True)
              .agg(n_impressions=("click", "size"), n_clicks=("click", "sum"))
              .reset_index())
    out["n_impressions"] = out["n_impressions"].astype("int32")
    out["n_clicks"] = out["n_clicks"].astype("int32")
    return out


def build_conversion_events(df: pd.DataFrame, n_windows: int,
                            cols: DataColumns | None = None) -> pd.DataFrame:
    """One row per distinct conversion: (uid, conversion_id, conversion_timestamp, window).

    A conversion is repeated on every impression that precedes it, so we deduplicate on
    (uid, conversion_id). Conversions after the last impression window are dropped because
    there is no window left to predict them from.
    """
    cols = cols or DataColumns()
    conv = df.loc[df[cols.conversion_id] >= 0,
                  [cols.user, cols.conversion_id, cols.conversion_timestamp, "conversion_window"]]
    conv = (conv.drop_duplicates([cols.user, cols.conversion_id])
                .rename(columns={cols.user: "uid", cols.conversion_id: "conversion_id",
                                 cols.conversion_timestamp: "conversion_timestamp",
                                 "conversion_window": "window"}))
    conv = conv[(conv["window"] >= 0) & (conv["window"] < n_windows)]
    return conv.sort_values(["uid", "conversion_timestamp"]).reset_index(drop=True)


def build_user_windows(df: pd.DataFrame, conversions: pd.DataFrame,
                       cols: DataColumns | None = None) -> pd.DataFrame:
    """Per (uid, window) features. Includes windows with only a conversion and no impression.

    Columns: n_impressions, n_clicks, n_campaigns, cost_sum, n_conversions, is_active
    (is_active = had at least one impression). ``n_conversions`` counts conversions whose
    *own* timestamp falls in the window, so it is a legitimate past feature for later windows.
    """
    cols = cols or DataColumns()
    imp = (df.groupby([cols.user, "window"], sort=False)
             .agg(n_impressions=(cols.timestamp, "size"),
                  n_clicks=(cols.click, "sum"),
                  n_campaigns=(cols.campaign, "nunique"),
                  cost_sum=(cols.cost, "sum"))
             .reset_index()
             .rename(columns={cols.user: "uid"}))
    conv = conversions.groupby(["uid", "window"]).size().rename("n_conversions").reset_index()
    out = imp.merge(conv, on=["uid", "window"], how="outer")
    for c in ["n_impressions", "n_clicks", "n_campaigns", "n_conversions"]:
        out[c] = out[c].fillna(0).astype("int32")
    out["cost_sum"] = out["cost_sum"].fillna(0).astype("float32")
    out["is_active"] = (out["n_impressions"] > 0).astype("int8")
    out["window"] = out["window"].astype("int32")
    return out.sort_values(["uid", "window"]).reset_index(drop=True)


def build_positions(user_windows: pd.DataFrame, conversions: pd.DataFrame, n_windows: int,
                    max_inactive_windows: int, horizon: int = 1) -> pd.DataFrame:
    """Prediction positions and next-window conversion labels.

    A position (uid, t) means "using everything up to and including window t, predict
    window t + horizon". User u is at risk at t if u had an impression in some window a with
    t - max_inactive_windows <= a <= t. This rule only looks backwards, so it cannot leak
    labels into sample selection. Positions stop at n_windows - 1 - horizon so every label
    window exists.

    Columns: uid, window (t), label_window (t + horizon), is_active (impression at t),
    windows_since_active, y_conversion (1 if a conversion falls in label_window).
    """
    last_t = n_windows - 1 - horizon
    act = user_windows.loc[user_windows["is_active"] == 1, ["uid", "window"]]
    act = act.sort_values(["uid", "window"]).reset_index(drop=True)
    starts = act["window"].to_numpy(np.int64)
    same_user_next = act["uid"].shift(-1).to_numpy() == act["uid"].to_numpy()
    next_start = np.where(same_user_next, act["window"].shift(-1).fillna(0).to_numpy(np.int64), np.iinfo(np.int64).max)
    ends = np.minimum.reduce([starts + max_inactive_windows, next_start - 1,
                              np.full_like(starts, last_t)])
    keep = ends >= starts
    starts, ends, uids = starts[keep], ends[keep], act["uid"].to_numpy()[keep]
    lengths = ends - starts + 1
    rep_start = np.repeat(starts, lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    pos = pd.DataFrame({
        "uid": np.repeat(uids, lengths),
        "window": (rep_start + offsets).astype("int32"),
        "windows_since_active": offsets.astype("int16"),
    })
    pos["is_active"] = (pos["windows_since_active"] == 0).astype("int8")
    pos["label_window"] = (pos["window"] + horizon).astype("int32")

    conv_keys = conversions[["uid", "window"]].drop_duplicates().rename(columns={"window": "label_window"})
    conv_keys["y_conversion"] = np.int8(1)
    pos = pos.merge(conv_keys, on=["uid", "label_window"], how="left")
    pos["y_conversion"] = pos["y_conversion"].fillna(0).astype("int8")
    return pos[["uid", "window", "label_window", "is_active", "windows_since_active", "y_conversion"]] \
        .sort_values(["uid", "window"]).reset_index(drop=True)


def build_next_campaign_labels(positions: pd.DataFrame, window_campaigns: pd.DataFrame) -> pd.DataFrame:
    """Sparse multi-label campaign targets: (uid, window t, campaign_idx) for campaigns in t+h.

    Also returns nothing for positions with an empty next window; use
    ``add_next_campaign_counts`` to attach ``n_next_campaigns`` to the positions table.
    """
    wc = window_campaigns[["uid", "window", "campaign_idx"]].rename(columns={"window": "label_window"})
    lab = positions[["uid", "window", "label_window"]].merge(wc, on=["uid", "label_window"], how="inner")
    return lab[["uid", "window", "label_window", "campaign_idx"]].sort_values(
        ["uid", "window", "campaign_idx"]).reset_index(drop=True)


def add_next_campaign_counts(positions: pd.DataFrame, next_campaigns: pd.DataFrame) -> pd.DataFrame:
    counts = next_campaigns.groupby(["uid", "window"]).size().rename("n_next_campaigns").reset_index()
    out = positions.merge(counts, on=["uid", "window"], how="left")
    out["n_next_campaigns"] = out["n_next_campaigns"].fillna(0).astype("int16")
    return out


# --------------------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------------------

def build_windowed_dataset(raw: pd.DataFrame, hours: float, max_inactive_hours: float,
                           horizon: int = 1, cols: DataColumns | None = None,
                           vocab: pd.DataFrame | None = None, n_windows: int | None = None,
                           timestamp_unit: str | None = None) -> dict:
    """Run the whole pipeline for one window length. Returns tables + metadata dict.

    When processing user shards, pass the global ``vocab``, ``n_windows`` and
    ``timestamp_unit`` computed on the full file so every shard shares the same calendar
    and campaign indices.
    """
    cols = cols or DataColumns()
    unit = timestamp_unit or infer_timestamp_unit(raw, cols)
    events = add_windows(sort_events(raw, cols), hours, cols)
    n_windows = n_windows or n_windows_for(events, hours, cols)
    max_inactive_windows = max(0, int(math.ceil(max_inactive_hours / hours)))

    vocab = vocab if vocab is not None else build_campaign_vocab(events, cols)
    window_campaigns = build_window_campaigns(events, vocab, cols)
    conversions = build_conversion_events(events, n_windows, cols)
    user_windows = build_user_windows(events, conversions, cols)
    positions = build_positions(user_windows, conversions, n_windows, max_inactive_windows, horizon)
    next_campaigns = build_next_campaign_labels(positions, window_campaigns)
    positions = add_next_campaign_counts(positions, next_campaigns)

    covered, n_conv_windows = _coverage(positions, conversions)
    all_conv = events.loc[events[cols.conversion_id] >= 0].drop_duplicates([cols.user, cols.conversion_id])
    meta = {
        "window_hours": hours,
        "horizon_windows": horizon,
        "timestamp_unit": unit,
        "n_windows": n_windows,
        "max_inactive_windows": max_inactive_windows,
        "n_impressions": int(len(events)),
        "n_users": int(events[cols.user].nunique()),
        "n_campaigns": int(len(vocab)),
        "n_conversions_total": int(len(all_conv)),
        "n_conversions_in_horizon": int(len(conversions)),
        "n_positions": int(len(positions)),
        "n_users_with_positions": int(positions["uid"].nunique()),
        "n_positive_positions": int(positions["y_conversion"].sum()),
        "n_positions_with_next_campaign": int((positions["n_next_campaigns"] > 0).sum()),
        "n_conversion_windows_covered": covered,
        "n_conversion_windows": n_conv_windows,
        "leaky_columns_never_used_as_features": LEAKY_COLUMNS,
    }
    meta = finalize_meta(meta)
    return {
        "campaign_vocab": vocab,
        "window_campaigns": window_campaigns,
        "conversions": conversions,
        "user_windows": user_windows,
        "positions": positions,
        "next_campaign_labels": next_campaigns,
        "meta": meta,
    }


def _coverage(positions: pd.DataFrame, conversions: pd.DataFrame) -> tuple[int, int]:
    """(covered, total) user-windows with a conversion that some position's label window hits."""
    keys = conversions[["uid", "window"]].drop_duplicates()
    hit = keys.merge(positions[["uid", "label_window"]].rename(columns={"label_window": "window"})
                     .drop_duplicates(), on=["uid", "window"], how="inner")
    return int(len(hit)), int(len(keys))


def finalize_meta(meta: dict) -> dict:
    """Add rate fields derived from the integer counts (works on summed shard metadata)."""
    out = dict(meta)
    n = max(out["n_positions"], 1)
    out["positive_rate_conversion"] = out["n_positive_positions"] / n
    out["frac_positions_with_next_campaign"] = out["n_positions_with_next_campaign"] / n
    out["frac_conversion_windows_covered"] = (
        out["n_conversion_windows_covered"] / max(out["n_conversion_windows"], 1))
    return out


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write parquet; fall back to compressed CSV if no parquet engine is installed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".csv.gz")
        df.to_csv(alt, index=False)
        return alt


def save_dataset(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in result.items():
        if name == "meta":
            continue
        written.append(write_table(table, out_dir / f"{name}.parquet"))
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(result["meta"], indent=2))
    written.append(meta_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort raw Criteo events (see scripts/prepare_criteo.py for the full pipeline).")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    df = sort_events(load_criteo(args.input))
    write_table(df, args.output)


if __name__ == "__main__":
    main()
