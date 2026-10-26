"""Run the non-sequential baselines on a processed dataset and save a results table.

Examples (repo root, conda env active):

    # seconds: the 2% user sample built with prepare_criteo.py --user-sample 50 --tag smoke
    python scripts/run_baselines.py --data-dir data/processed/criteo_w12h_smoke

    # full 12h data on a Zaratan debug node (4 CPUs, see docs/design_notes.md section 7)
    srun -A msml612-class -p debug -t 15 -c 4 --mem=48G \\
        python scripts/run_baselines.py --data-dir data/processed/criteo_w12h

Protocols (see src/data/splits.py):
    chronological  train/val/test = label windows 1-43 / 44-52 / 53-61 (12h), all users
    user_disjoint  train/val/test = hashed user groups 70/15/15, all label windows

Training uses every positive and a random ``--neg-rate`` share of negatives (default 2%);
probabilities are corrected back before calibration metrics. Evaluation uses every position.
The F1 threshold is chosen on validation and applied to test.

Writes results/tables/baselines_<dataset>.json and .md
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP threads before numpy loads. Zaratan nodes have 128 cores and a 256-thread
# per-user limit on login nodes; uncapped, every process starts 128 threads. Inside a Slurm job
# this uses the CPUs you asked for (--cpus-per-task / -c).
_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.features import build_position_features, feature_matrix  # noqa: E402
from src.data.io import load_meta, load_shard, shard_ids  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    best_f1_threshold, conversion_metrics, ranking_conversion_metrics, ranking_metrics_from_topk,
)
from src.models.baselines import (  # noqa: E402
    campaign_popularity, campaign_topk, correct_for_negative_sampling, make_conversion_model,
    recency_frequency_score,
)

PROTOCOLS = {"chronological": "time_split", "user_disjoint": "user_group"}
SPLITS = ("train", "val", "test")


def popularity_by_protocol(data_dir: Path, shards: list[int], meta: dict, n_campaigns: int) -> dict:
    """Training-only campaign popularity for each protocol (first pass over window_campaigns)."""
    val_start = meta["splits"]["boundaries"]["val_label_windows"][0]
    parts = {"chronological": [], "user_disjoint": []}
    for s in shards:
        t = load_shard(data_dir, s, tables=["window_campaigns", "positions"],
                       columns={"positions": ["uid", "user_group"]})
        wc = t["window_campaigns"]
        parts["chronological"].append(wc[wc["window"] < val_start])
        train_users = t["positions"].loc[t["positions"]["user_group"] == "train", "uid"].unique()
        parts["user_disjoint"].append(wc[wc["uid"].isin(train_users)])
    return {p: campaign_popularity(pd.concat(v), n_campaigns) for p, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument("--models", nargs="+", default=["logistic", "gbdt"], choices=["logistic", "gbdt"])
    ap.add_argument("--neg-rate", type=float, default=0.02, help="share of training negatives kept")
    ap.add_argument("--max-shards", type=int, default=None, help="use only the first N user shards")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--skip-campaign", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = ap.parse_args()

    t0 = time.time()
    data_dir = args.data_dir if args.data_dir.is_absolute() else REPO_ROOT / args.data_dir
    meta = load_meta(data_dir)
    shards = shard_ids(data_dir)[: args.max_shards]
    n_campaigns, hours = meta["n_campaigns"], meta["window_hours"]
    kmax = max(args.k)
    rng = np.random.default_rng(args.seed)
    print(f"{data_dir.name}: {len(shards)} shard(s), {hours:g}h windows, {n_campaigns} campaigns")

    popular = {} if args.skip_campaign else popularity_by_protocol(data_dir, shards, meta, n_campaigns)

    train = {p: {"X": [], "y": []} for p in args.protocols}
    evals = {p: {s: {"X": [], "y": [], "rf": []} for s in ("val", "test")} for p in args.protocols}
    camp = {p: {s: {"topk": {}, "truth": []} for s in ("val", "test")} for p in args.protocols}

    for s in shards:
        t1 = time.time()
        tb = load_shard(data_dir, s)
        pos = build_position_features(tb["positions"], tb["user_windows"], hours)
        X = feature_matrix(pos)
        y = pos["y_conversion"].to_numpy()
        rf = recency_frequency_score(pos)
        keep_neg = rng.random(len(pos)) < args.neg_rate
        truth = None if args.skip_campaign else (
            tb["next_campaign_labels"].groupby(["uid", "window"])["campaign_idx"].apply(list))
        for p in args.protocols:
            split = pos[PROTOCOLS[p]].to_numpy()
            m = (split == "train") & ((y == 1) | keep_neg)
            train[p]["X"].append(X[m]); train[p]["y"].append(y[m])
            for sp in ("val", "test"):
                m = split == sp
                e = evals[p][sp]
                e["X"].append(X[m]); e["y"].append(y[m]); e["rf"].append(rf[m])
                if args.skip_campaign:
                    continue
                q = pos.loc[m & (pos["n_next_campaigns"].to_numpy() > 0), ["uid", "window"]]
                if q.empty:
                    continue
                topk = campaign_topk(q, tb["window_campaigns"], popular[p], kmax)
                keys = list(zip(q["uid"].to_numpy(), q["window"].to_numpy()))
                camp[p][sp]["truth"].extend(truth.loc[keys].tolist())
                for name, arr in topk.items():
                    camp[p][sp]["topk"].setdefault(name, []).append(arr)
        print(f"  shard {s}: {len(pos):,} positions in {time.time() - t1:.0f}s")

    results = {"dataset": data_dir.name, "n_shards_used": len(shards), "neg_rate": args.neg_rate,
               "seed": args.seed, "window_hours": hours, "protocols": {}}
    for p in args.protocols:
        Xtr, ytr = np.concatenate(train[p]["X"]), np.concatenate(train[p]["y"])
        ev = {sp: {k: np.concatenate(v) for k, v in evals[p][sp].items()} for sp in ("val", "test")}
        res = {"n_train_rows_sampled": int(len(ytr)), "n_train_positives": int(ytr.sum()),
               "n_val": int(len(ev["val"]["y"])), "n_test": int(len(ev["test"]["y"])),
               "conversion": {}, "campaign": {}}
        res["conversion"]["recency_frequency"] = {
            sp: ranking_conversion_metrics(ev[sp]["y"], ev[sp]["rf"]) for sp in ("val", "test")}
        for name in args.models:
            t1 = time.time()
            model = make_conversion_model(name, args.seed).fit(Xtr, ytr)
            probs = {sp: correct_for_negative_sampling(model.predict_proba(ev[sp]["X"])[:, 1],
                                                       args.neg_rate) for sp in ("val", "test")}
            thr, _ = best_f1_threshold(ev["val"]["y"], probs["val"])
            res["conversion"][name] = {sp: conversion_metrics(ev[sp]["y"], probs[sp], threshold=thr)
                                       for sp in ("val", "test")}
            res["conversion"][name]["threshold_from_val"] = thr
            res["conversion"][name]["fit_seconds"] = round(time.time() - t1, 1)
            print(f"  [{p}] {name}: test PR-AUC {res['conversion'][name]['test']['pr_auc']:.4f}")
        if not args.skip_campaign:
            for sp in ("val", "test"):
                c = camp[p][sp]
                for name, arrs in c["topk"].items():
                    res["campaign"].setdefault(name, {})[sp] = ranking_metrics_from_topk(
                        np.concatenate(arrs), c["truth"], args.k)
        results["protocols"][p] = res
    results["seconds"] = round(time.time() - t0, 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"baselines_{data_dir.name}"
    stem.with_suffix(".json").write_text(json.dumps(results, indent=2))
    stem.with_suffix(".md").write_text(to_markdown(results, args.k))
    print(to_markdown(results, args.k))
    print(f"Saved {stem.with_suffix('.json')} and .md in {results['seconds']:.0f}s")


def to_markdown(r: dict, ks) -> str:
    lines = [f"# Baselines: {r['dataset']} ({r['window_hours']:g}h windows)", "",
             f"Shards used: {r['n_shards_used']} | training negatives kept: {r['neg_rate']:.0%} | "
             f"seed {r['seed']}", ""]
    for p, res in r["protocols"].items():
        lines += [f"## {p} (test split)", "",
                  f"Test positions: {res['n_test']:,}; base rate "
                  f"{next(iter(res['conversion'].values()))['test']['base_rate']:.4%}", "",
                  "| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, m in res["conversion"].items():
            t = m["test"]
            f = lambda k, fmt="{:.4f}": fmt.format(t[k]) if k in t else "–"  # noqa: E731
            lines.append(f"| {name} | {f('pr_auc')} | {f('roc_auc')} | {f('top_decile_lift', '{:.2f}')} | "
                         f"{f('top_decile_gain', '{:.3f}')} | {f('binary_cross_entropy')} | "
                         f"{f('expected_calibration_error')} | {f('f1')} |")
        if res["campaign"]:
            k = max(ks)
            lines += ["", f"| Campaign baseline | P@{k} | R@{k} | HitRate@1 | HitRate@{k} | NDCG@{k} | MAP@{k} |",
                      "|---|---|---|---|---|---|---|"]
            for name, m in res["campaign"].items():
                t = m["test"]
                lines.append(f"| {name} | {t[f'precision_at_{k}']:.4f} | {t[f'recall_at_{k}']:.4f} | "
                             f"{t['hit_rate_at_1']:.4f} | {t[f'hit_rate_at_{k}']:.4f} | "
                             f"{t[f'ndcg_at_{k}']:.4f} | {t[f'map_at_{k}']:.4f} |")
            n = next(iter(res["campaign"].values()))["test"]["n_positions"]
            lines.append(f"\nCampaign metrics use the {int(n):,} test positions whose next window has "
                         "at least one campaign.")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
