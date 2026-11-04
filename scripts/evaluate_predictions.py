"""Score saved test-set predictions: bootstrap CIs over users and subgroup results.

No model or GPU is needed: this reads prediction files and the processed tables only, so it
runs on a CPU node.

Prediction files (one per model x seed x split), under results/predictions/:

    <model>__seed<k>__<split>.parquet     e.g. stajt__seed1__chrono_test.parquet
    columns: uid, window (or label_window), score, [y_conversion]

``split`` is ``chrono_test`` (time_split == test) or ``user_test`` (user_group == test).
``score`` is the predicted conversion probability for label window t + 1 (any monotone score
works for PR-AUC/ROC-AUC/gain). Every file for one split must cover the same positions.

    # smoke test on the 2% sample
    python scripts/evaluate_predictions.py --data-dir data/processed/criteo_w12h_smoke \\
        --split chrono_test --n-boot 50
    # full data, through Slurm (CPU only)
    sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test

Writes to results/tables/:
    conversion_ci_<split>.csv           point, seed sd and 95% CI per model x metric
    conversion_diff_<split>.csv         paired difference vs --reference, with CI
    conversion_subgroups_<split>.csv    the same per subgroup (5+ active windows, recent click,
                                        next window has a new-to-user campaign)
    conversion_summary_<split>.md       report-ready table
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "2")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from pathlib import Path
import argparse
import re
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.io import load_meta, load_shard, shard_ids  # noqa: E402
from src.evaluation.bootstrap import format_ci, user_bootstrap  # noqa: E402
from src.evaluation.subgroups import HISTORY_ONLY, SUBGROUPS, position_subgroup_flags  # noqa: E402

NAME_RE = re.compile(r"^(?P<model>.+?)__seed(?P<seed>\d+)__(?P<split>[a-z_]+)$")
SPLIT_FILTER = {"chrono_test": ("time_split", "test"), "user_test": ("user_group", "test"),
                "chrono_val": ("time_split", "val"), "user_val": ("user_group", "val")}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_predictions(pred_dir: Path, split: str) -> dict[tuple[str, int], pd.DataFrame]:
    out = {}
    for p in sorted(pred_dir.glob(f"*__{split}.*")):
        m = NAME_RE.match(p.name.split(".")[0])
        if not m:
            print(f"skip {p.name}: name is not <model>__seed<k>__<split>", flush=True)
            continue
        df = _read(p)
        if "window" not in df.columns:
            df["window"] = df["label_window"] - 1
        missing = {"uid", "window", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} is missing columns {missing}")
        out[(m["model"], int(m["seed"]))] = df[["uid", "window", "score"]]
    if not out:
        raise FileNotFoundError(f"No prediction files for split {split!r} in {pred_dir}")
    return out


def load_flags(data_dir: Path, split: str) -> pd.DataFrame:
    meta = load_meta(data_dir)
    col, val = SPLIT_FILTER[split]
    parts = []
    for s in shard_ids(data_dir):
        tb = load_shard(data_dir, s, tables=["positions", "user_windows", "window_campaigns",
                                             "next_campaign_labels"])
        pos = tb["positions"]
        pos = pos[pos[col] == val]
        f = position_subgroup_flags(pos, tb["user_windows"], tb["window_campaigns"],
                                    tb["next_campaign_labels"], meta["window_hours"])
        f["y_conversion"] = pos["y_conversion"].to_numpy()
        parts.append(f)
        print(f"  shard {s}: {len(f):,} {split} positions", flush=True)
    return pd.concat(parts, ignore_index=True)


def align(flags: pd.DataFrame, preds: dict) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Put every prediction file in the row order of ``flags``; fail if any position is missing."""
    keys = flags[["uid", "window"]]
    by_model: dict[str, list[np.ndarray]] = {}
    for (model, seed), df in sorted(preds.items()):
        if df.duplicated(["uid", "window"]).any():
            raise ValueError(f"{model} seed {seed}: duplicate (uid, window) rows")
        m = keys.merge(df, on=["uid", "window"], how="left")
        n_missing = int(m["score"].isna().sum())
        if n_missing or len(df) != len(keys):
            raise ValueError(f"{model} seed {seed}: {n_missing:,} positions missing and "
                             f"{len(df) - (len(keys) - n_missing):,} extra; every file must cover the "
                             f"same {len(keys):,} positions")
        by_model.setdefault(model, []).append(m["score"].to_numpy(np.float64))
    return flags, by_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/processed/criteo_w12h")
    ap.add_argument("--pred-dir", type=Path, default=REPO_ROOT / "results/predictions")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/tables")
    ap.add_argument("--split", choices=sorted(SPLIT_FILTER), default="chrono_test")
    ap.add_argument("--reference", default="itransformer",
                    help="model the paired differences are taken against")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--subgroup-n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=612)
    ap.add_argument("--neg-frac", type=float, default=0.1,
                    help="share of negatives used inside replicates (weighted 1/neg-frac); "
                         "point estimates always use every row")
    ap.add_argument("--n-jobs", type=int, default=int(_THREADS))
    args = ap.parse_args()

    t0 = time.time()
    preds = load_predictions(args.pred_dir, args.split)
    print(f"{len(preds)} prediction files: {sorted(preds)}", flush=True)
    flags, by_model = align(load_flags(args.data_dir, args.split), preds)
    reference = args.reference if args.reference in by_model else None
    y, users = flags["y_conversion"].to_numpy(), flags["uid"].to_numpy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.split

    print(f"overall: {len(y):,} positions, {y.sum():,} positives "
          f"({time.time() - t0:,.0f}s)", flush=True)
    ci, diffs = user_bootstrap(y, by_model, users, n_boot=args.n_boot, seed=args.seed,
                               reference=reference, neg_frac=args.neg_frac, n_jobs=args.n_jobs)
    ci.insert(0, "subgroup", "all")
    ci["base_rate"] = float(y.mean())
    ci.to_csv(args.out_dir / f"conversion_ci_{tag}.csv", index=False)
    diffs.to_csv(args.out_dir / f"conversion_diff_{tag}.csv", index=False)

    sub_rows = [ci]
    for g in SUBGROUPS[1:]:
        m = flags[g].to_numpy()
        if m.sum() == 0 or y[m].sum() == 0:
            print(f"  subgroup {g}: no positives, skipped", flush=True)
            continue
        print(f"  subgroup {g}: {m.sum():,} positions, {y[m].sum():,} positives", flush=True)
        sci, _ = user_bootstrap(y[m], {k: [a[m] for a in v] for k, v in by_model.items()}, users[m],
                                n_boot=args.subgroup_n_boot, seed=args.seed,
                                neg_frac=args.neg_frac, n_jobs=args.n_jobs)
        sci.insert(0, "subgroup", g)
        sci["base_rate"] = float(y[m].mean())
        sci["share_of_positions"] = float(m.mean())
        sci["history_only"] = HISTORY_ONLY[g]
        sub_rows.append(sci)
    subs = pd.concat(sub_rows, ignore_index=True)
    subs.to_csv(args.out_dir / f"conversion_subgroups_{tag}.csv", index=False)

    # Report table: PR-AUC with CI per model x subgroup, base rate next to every PR-AUC.
    pr = subs[subs.metric == "pr_auc"].copy()
    pr["cell"] = [format_ci(r.point, r.ci_low, r.ci_high) for r in pr.itertuples()]
    table = pr.pivot(index="model", columns="subgroup", values="cell")
    table = table[[g for g in SUBGROUPS if g in table.columns]]
    rates = pr.groupby("subgroup")["base_rate"].first()
    lines = [f"# Conversion PR-AUC with 95% user-bootstrap CI ({tag})", "",
             "| Model | " + " | ".join(f"{g} (base rate {100 * rates[g]:.3f}%)" for g in table.columns) + " |",
             "|---" * (len(table.columns) + 1) + "|"]
    for model, row in table.iterrows():
        lines.append(f"| {model} | " + " | ".join(row.fillna("")) + " |")
    if not diffs.empty:
        lines += ["", f"Paired difference vs {reference} (all positions):", "",
                  "| Model | Metric | Difference [95% CI] | Share of replicates not better |",
                  "|---|---|---|---|"]
        for r in diffs.itertuples():
            lines.append(f"| {r.model} | {r.metric} | {format_ci(r.diff, r.ci_low, r.ci_high)} | {r.p_not_better:.3f} |")
    lift = subs[subs.metric == "pr_auc_lift"].copy()
    if not lift.empty:
        lift["cell"] = [f"{r.point:.1f}x [{r.ci_low:.1f}, {r.ci_high:.1f}]" for r in lift.itertuples()]
        lt = lift.pivot(index="model", columns="subgroup", values="cell")
        lt = lt[[g for g in SUBGROUPS if g in lt.columns]]
        lines += ["", "PR-AUC divided by the subgroup's base rate (1.0x = random ranking), 95% CI. "
                      "PR-AUC rises with prevalence, so compare subgroups with this table, not the one above.", "",
                  "| Model | " + " | ".join(lt.columns) + " |", "|---" * (len(lt.columns) + 1) + "|"]
        for model, row in lt.iterrows():
            lines.append(f"| {model} | " + " | ".join(row.fillna("")) + " |")
    lines += ["", "next_new_campaign uses the label window, so it is a descriptive slice, not a "
                  "segment selectable in advance."]
    (args.out_dir / f"conversion_summary_{tag}.md").write_text("\n".join(lines) + "\n")
    print(f"done in {time.time() - t0:,.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
