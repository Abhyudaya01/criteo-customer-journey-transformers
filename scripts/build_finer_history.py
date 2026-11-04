"""Build the 6h-input / 12h-target dataset for the window-length ablation.

Reads two folders written by scripts/prepare_criteo.py (same user shards) and writes a third:

    python scripts/build_finer_history.py \\
        --coarse data/processed/criteo_w12h --fine data/processed/criteo_w6h \\
        --out data/processed/criteo_w6h_in_w12h

CPU only, a few minutes on the full data (one shard in memory at a time). No GPU.

Output: fine tables (user_windows, window_campaigns, conversions) + 12h positions and
next-window campaign labels re-anchored to 6h window indices; meta.json has the fine
n_windows and window_hours (so loaders build 124-step sequences) plus target_hours = 12 and the
12h split boundaries. See src/data/multiscale.py for the design.

Models trained on it save predictions keyed by 12h windows:
    from src.data.multiscale import to_coarse_keys
    preds["window"] = to_coarse_keys(preds["window"], ratio=2)
then scripts/evaluate_predictions.py scores them on the same positions as every other model.
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "2")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from pathlib import Path
import argparse
import json
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.io import TABLES, load_meta, load_shard, shard_ids  # noqa: E402
from src.data.multiscale import build_shard, window_ratio  # noqa: E402
from src.data.preprocess import write_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", type=Path, default=REPO_ROOT / "data/processed/criteo_w12h")
    ap.add_argument("--fine", type=Path, default=REPO_ROOT / "data/processed/criteo_w6h")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/criteo_w6h_in_w12h")
    args = ap.parse_args()

    t0 = time.time()
    cm, fm = load_meta(args.coarse), load_meta(args.fine)
    ratio = window_ratio(cm["window_hours"], fm["window_hours"])
    for k in ("n_shards", "seed", "user_sample", "nrows"):
        if cm.get(k) != fm.get(k):
            raise ValueError(f"coarse and fine folders differ in {k}: {cm.get(k)} vs {fm.get(k)}; "
                             "rebuild them in one prepare_criteo.py --all-windows run")
    if fm["n_windows"] != cm["n_windows"] * ratio:
        raise ValueError(f"fine n_windows {fm['n_windows']} != {ratio} x coarse {cm['n_windows']}")
    shards = shard_ids(args.coarse)
    if shards != shard_ids(args.fine):
        raise ValueError("coarse and fine folders have different shard ids")

    args.out.mkdir(parents=True, exist_ok=True)
    vocab = next(args.fine.glob("campaign_vocab.*"))
    shutil.copy(vocab, args.out / vocab.name)
    n_pos = n_pos_pos = 0
    for s in shards:
        res = build_shard(load_shard(args.coarse, s), load_shard(args.fine, s), ratio)
        for t in TABLES:
            write_table(res[t], args.out / t / f"part-{s:03d}.parquet")
        n_pos += len(res["positions"])
        n_pos_pos += int(res["positions"]["y_conversion"].sum())
        print(f"  shard {s}: {len(res['positions']):,} positions ({time.time() - t0:,.0f}s)", flush=True)

    if n_pos != cm["n_positions"] or n_pos_pos != cm["n_positive_positions"]:
        raise RuntimeError(f"position counts changed: {n_pos:,}/{n_pos_pos:,} vs coarse "
                           f"{cm['n_positions']:,}/{cm['n_positive_positions']:,}")
    meta = {k: v for k, v in fm.items() if k not in ("splits",)}
    meta.update({
        "dataset": "fine_history_coarse_target",
        "history_hours": fm["window_hours"], "target_hours": cm["window_hours"], "ratio": ratio,
        "n_positions": cm["n_positions"], "n_positive_positions": cm["n_positive_positions"],
        "positive_rate_conversion": cm["positive_rate_conversion"],
        "splits": cm["splits"],
        "coarse_source": str(args.coarse), "fine_source": str(args.fine),
        "note": "positions/labels/splits are the 12h ones, anchored at 6h window 2k+1; "
                "map predictions back with src.data.multiscale.to_coarse_keys",
    })
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {n_pos:,} positions, {n_pos_pos:,} positive, {time.time() - t0:,.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
