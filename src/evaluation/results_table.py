"""The team's one results table: results/tables/results.csv (Final Stretch Plan, Week 1).

One row per prediction file, i.e. per model x seed x test split, with point metrics on every
test position plus how the run was made. ``scripts/evaluate_predictions.py`` appends the rows;
nobody edits the CSV by hand. Seed-averaged CIs and subgroups stay in the conversion_* tables.

Run metadata comes from a sidecar next to each prediction file, written by the training script:

    results/predictions/stajt__seed1__chrono_test.parquet
    results/predictions/stajt__seed1__chrono_test.meta.json

    from src.evaluation.results_table import write_run_meta
    write_run_meta(pred_path, config="team_aligned_neg1pct", history="events64",
                   source="STAJT/scripts/train_stajt_fair_chrono.py", gpu_hours=6.5,
                   hardware="A100", notes="checkpoint epoch 5")

Rows are keyed by (model, seed, split, config, history): re-scoring replaces the old row, so
the table never holds duplicates. A file without a sidecar is still scored, with its metadata
marked "unknown" so the gap is visible.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["model", "seed", "split", "config", "history"]
METRIC_COLUMNS = ["pr_auc", "pr_auc_lift", "roc_auc", "top_decile_gain", "base_rate",
                  "n_positions", "n_positives"]
META_COLUMNS = ["source", "run_where", "hardware", "gpu_hours", "git_commit", "trained_on", "notes"]
COLUMNS = KEY + METRIC_COLUMNS + META_COLUMNS + ["scored_on"]
UNKNOWN = "unknown"


def sidecar_path(pred_path: Path | str) -> Path:
    p = Path(pred_path)
    stem = p.name.split(".")[0]  # works for .parquet and .csv.gz
    return p.with_name(f"{stem}.meta.json")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parents[2])
        return out.stdout.strip() or UNKNOWN
    except Exception:
        return UNKNOWN


def _run_where() -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return f"zaratan job {os.environ['SLURM_JOB_ID']}"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "colab"
    return "local"


def write_run_meta(pred_path: Path | str, *, config: str, history: str, source: str,
                   gpu_hours: float, hardware: str = "", notes: str = "",
                   run_where: str | None = None, git_commit: str | None = None) -> Path:
    """Record how a prediction file was made. Call it right after saving the predictions.

    config    training setup, e.g. "team_aligned_neg1pct", "optimized_10neg", "neg1pct"
    history   input representation, e.g. "w12h", "w6h_in_w12h", "events64"
    source    the script that trained the model (one run source per model)
    gpu_hours GPU time for training + test scoring (0 for CPU-only models)
    """
    meta = {
        "config": config, "history": history, "source": source, "gpu_hours": float(gpu_hours),
        "hardware": hardware, "notes": notes,
        "run_where": run_where or _run_where(), "git_commit": git_commit or _git_commit(),
        "trained_on": _dt.date.today().isoformat(),
    }
    path = sidecar_path(pred_path)
    path.write_text(json.dumps(meta, indent=2))
    return path


def read_run_meta(pred_path: Path | str) -> dict:
    path = sidecar_path(pred_path)
    if not path.exists():
        return {"config": UNKNOWN, "history": UNKNOWN, "source": UNKNOWN, "run_where": UNKNOWN,
                "hardware": "", "gpu_hours": np.nan, "git_commit": UNKNOWN, "trained_on": UNKNOWN,
                "notes": "no .meta.json sidecar"}
    return json.loads(path.read_text())


def make_row(model: str, seed: int, split: str, metrics: dict, meta: dict,
             n_positions: int, n_positives: int) -> dict:
    row = {"model": model, "seed": int(seed), "split": split,
           "config": meta.get("config", UNKNOWN), "history": meta.get("history", UNKNOWN),
           "n_positions": int(n_positions), "n_positives": int(n_positives),
           "scored_on": _dt.date.today().isoformat()}
    for m in ("pr_auc", "pr_auc_lift", "roc_auc", "top_decile_gain", "base_rate"):
        row[m] = metrics.get(m, np.nan)
    for c in META_COLUMNS:
        if c not in ("trained_on",):
            row[c] = meta.get(c, UNKNOWN)
    row["trained_on"] = meta.get("trained_on", UNKNOWN)
    return row


def upsert_results(path: Path | str, rows: list[dict]) -> pd.DataFrame:
    """Add rows to the CSV, replacing any existing row with the same key. Returns the table."""
    path = Path(path)
    new = pd.DataFrame(rows, columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        missing = set(COLUMNS) - set(old.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}; it was not written by this helper")
        old = old[COLUMNS]
        k_old = old[KEY].astype(str).agg("|".join, axis=1)
        k_new = set(new[KEY].astype(str).agg("|".join, axis=1))
        table = pd.concat([old[~k_old.isin(k_new)], new], ignore_index=True)
    else:
        table = new
    table = table.sort_values(["split", "model", "config", "history", "seed"], kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    table.to_csv(tmp, index=False, float_format="%.6g")
    tmp.replace(path)  # atomic: a crashed run never leaves a half-written table
    return table.reset_index(drop=True)
