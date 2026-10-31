"""Read the processed dataset written by scripts/prepare_criteo.py.

Layout of data/processed/criteo_w{hours}h[_{tag}]/:
    meta.json, campaign_vocab.parquet
    {window_campaigns,user_windows,conversions,positions,next_campaign_labels}/part-XXX.parquet

All tables for one shard (part-XXX) cover the same set of users, so a shard can be processed
on its own. Falls back to .csv.gz parts when parquet was unavailable at write time.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

TABLES = ["window_campaigns", "user_windows", "conversions", "positions", "next_campaign_labels"]


def load_meta(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


def _read_any(path_no_suffix: Path) -> pd.DataFrame:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        p = path_no_suffix.with_name(path_no_suffix.name + suffix)
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"{path_no_suffix}.parquet / .csv.gz not found")


def shard_ids(data_dir: Path) -> list[int]:
    folder = Path(data_dir) / "positions"
    ids = sorted({int(p.name.split("-")[1].split(".")[0]) for p in folder.glob("part-*")})
    if not ids:
        raise FileNotFoundError(f"No positions/part-* files in {data_dir}")
    return ids


def load_shard(data_dir: Path, shard: int, tables=TABLES, columns: dict | None = None) -> dict:
    """Load the listed tables for one user shard. ``columns`` maps table -> list of columns."""
    out = {}
    for t in tables:
        df = _read_any(Path(data_dir) / t / f"part-{shard:03d}")
        if columns and t in columns:
            df = df[columns[t]]
        if t == "positions":
            for c in ("time_split", "user_group"):
                if c in df.columns:
                    df[c] = df[c].astype(str)
        out[t] = df
    return out


def load_vocab(data_dir: Path) -> pd.DataFrame:
    return _read_any(Path(data_dir) / "campaign_vocab")
