"""Build windowed customer journeys, labels and split assignments from the raw Criteo file.

Examples (run from the repo root):

    # 1) Verify the header/separator/timestamp unit against configs/base.yaml (fast)
    python scripts/prepare_criteo.py --check-only

    # 2) Quick smoke run on the first 500k rows
    python scripts/prepare_criteo.py --nrows 500000 --tag smoke

    # 3) Full run for the primary (12h) and comparison (6h) windows
    python scripts/prepare_criteo.py --all-windows

    # Lower memory (users are hashed into shards; each shard is processed independently)
    python scripts/prepare_criteo.py --all-windows --n-shards 16

Outputs go to data/processed/criteo_w{hours}h[_{tag}]/:
    campaign_vocab.parquet
    window_campaigns/ user_windows/ conversions/ positions/ next_campaign_labels/
        part-000.parquet ... (one part per user shard; read a folder with pd.read_parquet(folder))
    meta.json
positions has the split columns `time_split` (train/val/test by label window) and
`user_group` (train/val/test by hashed user id).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (  # noqa: E402
    CRITEO_COLUMNS, DataColumns, build_campaign_vocab, build_windowed_dataset, finalize_meta,
    infer_timestamp_unit, load_criteo, n_windows_for, validate_schema, write_table,
)

SUMMED_META = ["n_impressions", "n_users", "n_conversions_total", "n_conversions_in_horizon",
               "n_positions", "n_users_with_positions", "n_positive_positions",
               "n_positions_with_next_campaign", "n_conversion_windows_covered",
               "n_conversion_windows"]
TABLES = ["window_campaigns", "user_windows", "conversions", "positions", "next_campaign_labels"]
from src.data.splits import (  # noqa: E402
    assign_chronological_split, user_groups, window_boundaries,
)


def find_raw_file(cfg: dict, override: Path | None) -> Path:
    if override:
        return override
    data = cfg["data"]
    candidate = REPO_ROOT / data["raw_path"] / data.get("raw_file", "")
    if candidate.is_file():
        return candidate
    hits = sorted((REPO_ROOT / data["raw_path"]).rglob("criteo_attribution_dataset.tsv*"))
    if not hits:
        raise FileNotFoundError(
            f"No criteo_attribution_dataset.tsv(.gz) under {REPO_ROOT / data['raw_path']}. "
            "Run scripts/download_criteo.sh first.")
    return hits[0]


def check_schema(path: Path, sep: str, cols: DataColumns) -> None:
    head = pd.read_csv(path, sep=sep, nrows=5)
    print(f"File: {path}")
    print(f"Separator {sep!r} -> {head.shape[1]} columns")
    print("Header:", list(head.columns))
    validate_schema(list(head.columns), cols)
    extra = [c for c in head.columns if c not in CRITEO_COLUMNS]
    missing_doc = [c for c in CRITEO_COLUMNS if c not in head.columns]
    print("Matches documented schema:", not extra and not missing_doc,
          "| extra:", extra, "| missing:", missing_doc)
    print(head.to_string())
    sample = pd.read_csv(path, sep=sep, usecols=[cols.timestamp], nrows=None)
    unit = infer_timestamp_unit(sample, cols)
    span = sample[cols.timestamp].max() - sample[cols.timestamp].min()
    print(f"Timestamp unit: {unit} (span {span / 86400:.2f} days, min {sample[cols.timestamp].min()})")
    print("Config columns OK.")


def add_split_columns(positions: pd.DataFrame, bounds, cfg: dict, seed: int) -> pd.DataFrame:
    s = cfg["splits"]
    positions = positions.copy()
    positions["time_split"] = assign_chronological_split(positions, bounds).astype("category")
    groups = user_groups(positions["uid"], s.get("user_train_frac", 0.70),
                         s.get("user_val_frac", 0.15), seed)
    positions["user_group"] = positions["uid"].map(groups).astype("category")
    return positions


def split_counts(positions: pd.DataFrame) -> dict:
    """Additive counts per split (users are disjoint across shards, so they sum exactly)."""
    out = {}
    for col in ["time_split", "user_group"]:
        out[col] = {str(k): {"n_positions": int(len(v)), "n_users": int(v["uid"].nunique()),
                             "n_positive": int(v["y_conversion"].sum())}
                    for k, v in positions.groupby(col, observed=True)}
    return out


def merge_counts(acc: dict, new: dict) -> dict:
    for col, groups in new.items():
        for g, d in groups.items():
            tgt = acc.setdefault(col, {}).setdefault(g, {k: 0 for k in d})
            for k, v in d.items():
                tgt[k] += v
    return acc


def shard_ids(uids: pd.Series, n_shards: int, seed: int):
    h = pd.util.hash_array(uids.to_numpy(), hash_key=f"{seed + 1:016d}"[-16:], categorize=False)
    return (h % np.uint64(n_shards)).astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "base.yaml")
    p.add_argument("--input", type=Path, default=None, help="raw file (default: found under data/raw/)")
    p.add_argument("--hours", type=float, nargs="*", default=None, help="window lengths (default: primary)")
    p.add_argument("--all-windows", action="store_true", help="primary + comparison window lengths")
    p.add_argument("--nrows", type=int, default=None, help="read only the first N rows (smoke test)")
    p.add_argument("--tag", default="", help="suffix for the output folder")
    p.add_argument("--check-only", action="store_true", help="verify schema and exit")
    p.add_argument("--n-shards", type=int, default=8, help="user shards processed one at a time")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cols = DataColumns.from_config(cfg["data"])
    sep = cfg["data"].get("separator", "\t")
    seed = cfg["project"].get("seed", 42)
    win = cfg["windowing"]
    horizon = int(win.get("horizon_windows", 1))
    raw_path = find_raw_file(cfg, args.input)

    if args.check_only:
        check_schema(raw_path, sep, cols)
        return

    hours_list = args.hours or [win["primary_hours"]]
    if args.all_windows:
        hours_list = [win["primary_hours"], *win.get("comparison_hours", [])]

    t0 = time.time()
    raw = load_criteo(raw_path, sep=sep, cols=cols, nrows=args.nrows)
    print(f"Loaded {len(raw):,} rows from {raw_path.name} in {time.time() - t0:.0f}s")

    unit = infer_timestamp_unit(raw, cols)
    vocab = build_campaign_vocab(raw, cols)
    shards = shard_ids(raw[cols.user], args.n_shards, seed)
    max_inactive = win.get("max_inactive_hours", 168)

    for hours in hours_list:
        t1 = time.time()
        n_windows = n_windows_for(raw, hours, cols)
        s_cfg = cfg["splits"]
        bounds = window_boundaries(n_windows, s_cfg.get("train_frac", 0.70),
                                   s_cfg.get("val_frac", 0.15), horizon)
        name = f"criteo_w{hours:g}h" + (f"_{args.tag}" if args.tag else "")
        out_dir = REPO_ROOT / cfg["data"]["processed_path"] / name
        write_table(vocab, out_dir / "campaign_vocab.parquet")

        totals = {k: 0 for k in SUMMED_META}
        counts: dict = {}
        meta = {}
        for i in range(args.n_shards):
            part = raw[shards == i]
            if part.empty:
                continue
            res = build_windowed_dataset(part, hours, max_inactive, horizon, cols,
                                         vocab=vocab, n_windows=n_windows, timestamp_unit=unit)
            res["positions"] = add_split_columns(res["positions"], bounds, cfg, seed)
            counts = merge_counts(counts, split_counts(res["positions"]))
            for t in TABLES:
                write_table(res[t], out_dir / t / f"part-{i:03d}.parquet")
            meta = res["meta"]
            for k in SUMMED_META:
                totals[k] += meta[k]
            print(f"  [{hours:g}h] shard {i + 1}/{args.n_shards}: {meta['n_positions']:,} positions")

        meta.update(totals)
        meta["n_campaigns"] = int(len(vocab))
        meta = finalize_meta(meta)
        meta["splits"] = {"boundaries": bounds.as_dict(), **counts}
        meta.update({"seed": seed, "n_shards": args.n_shards, "source_file": raw_path.name,
                     "nrows": args.nrows})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n[{hours:g}h] built in {time.time() - t1:.0f}s -> {out_dir.relative_to(REPO_ROOT)}")
        print(json.dumps({k: v for k, v in meta.items()
                          if k != "leaky_columns_never_used_as_features"}, indent=2))

if __name__ == "__main__":
    main()
