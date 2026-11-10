"""Train a window-based Transformer and save test results and predictions.

Script version of notebooks/colab/training/Criteo_iTransformer_Colab.ipynb (same data, model,
training sample and evaluation), so the team has one run source on Zaratan and Colab.

Examples (repo root, env active):

    # seconds on the 2% sample (prepare_criteo.py --user-sample 50 --tag smoke)
    python scripts/train_transformer.py --data-dir data/processed/criteo_w12h_smoke --epochs 1

    # full 12h data, one seed, both protocols, predictions for scripts/evaluate_predictions.py
    python scripts/train_transformer.py --data-dir data/processed/criteo_w12h --seed 1 \\
        --save-predictions results/predictions

    # window-length ablation: 6h input tokens, 12h targets (docs/window_length_ablation.md)
    python scripts/train_transformer.py --data-dir data/processed/criteo_w6h_in_w12h --seed 1 \\
        --save-predictions results/predictions          # saves itransformer_w6h__seed1__*.parquet

Seeds. ``--seed`` draws the 1% negative training sample and seeds the model, like the baseline
seeds (scripts/zaratan/baselines_seeds.sbatch): each seed is a different negative sample.
``--model-seed`` sets the model seed separately; ``--seed 42 --model-seed 0`` reproduces the
Colab run in results/tables/colab_v2/.

Lookback is in hours (default 168 = 7 days): 14 windows of 12h, or 28 windows of 6h, so the 6h
model sees the same history span as the 12h one.

Outputs (``--out-dir``, default results/tables/<model>_<dataset>/):
    raw/<protocol>_<name>_h<heads>_seed<seed>.json   one run: conversion + campaign metrics
    conversion_<model>.csv, campaign_<model>.csv      every run in raw/
    run_config.json
With ``--save-predictions DIR``: DIR/<pred-name>__seed<seed>__<chrono_test|user_test>.parquet
(columns uid, window, score; one row per test position, 12h window keys), see
results/predictions/README.md. Runs whose raw JSON exists are skipped unless ``--force``.
Model checkpoints go to ``--checkpoint-dir`` only if given (they are not committed).
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "8")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from pathlib import Path
import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.data.preprocess import write_table  # noqa: E402
from src.evaluation.metrics import best_f1_threshold, conversion_metrics  # noqa: E402
from src.training.loop import TrainConfig, predict, train  # noqa: E402
from src.training.window_data import FEATS, WindowData  # noqa: E402

PRED_SPLIT_NAME = {"chronological": "chrono_test", "user_disjoint": "user_test"}
CONV_COLS = ["protocol", "model", "seed", "pr_auc", "roc_auc", "f1", "balanced_accuracy",
             "binary_cross_entropy", "brier", "expected_calibration_error", "top_decile_conversion_rate",
             "top_decile_lift", "top_decile_gain", "base_rate", "val_pr_auc", "n_test", "n_test_pos",
             "fit_seconds"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["itransformer"], default="itransformer")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["chronological", "user_disjoint"],
                    choices=["chronological", "user_disjoint", "user_disjoint_future"])
    ap.add_argument("--seed", type=int, default=1, help="negative sample and model seed")
    ap.add_argument("--model-seed", type=int, default=None, help="model seed if different from --seed")
    ap.add_argument("--neg-rate", type=float, default=0.01)
    ap.add_argument("--lookback-hours", type=float, default=168)
    ap.add_argument("--no-history-features", action="store_true")
    ap.add_argument("--top-campaigns", type=int, default=100)
    ap.add_argument("--heads", type=int, nargs="+", default=[4])
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-copy-history", action="store_true")
    ap.add_argument("--campaign-loss-weight", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--campaign-eval-max", type=int, default=200_000)
    ap.add_argument("--skip-campaign-eval", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None, help="debug: first N user shards only")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save-predictions", type=Path, default=None, metavar="DIR")
    ap.add_argument("--pred-name", default=None,
                    help="model name in prediction files (default itransformer, itransformer_w6h for 6h input)")
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def resolve(p: Path | None) -> Path | None:
    return None if p is None else (p if p.is_absolute() else REPO_ROOT / p)


def run_name(args, data: WindowData) -> str:
    name = f"{args.model}_top{args.top_campaigns}"
    if data.lookback != 14:
        name += f"_L{data.lookback}"
    if data.window_hours != data.target_hours:
        name += f"_w{data.window_hours:g}h"
    return name


def save_predictions(out_dir: Path, name: str, seed: int, protocol: str, data: WindowData,
                     idx: np.ndarray, scores: np.ndarray) -> Path:
    df = pd.DataFrame({"uid": data.pos_uid[idx], "window": data.pos_key_window[idx].astype("int32"),
                       "score": np.asarray(scores, dtype="float64")})
    if df.duplicated(["uid", "window"]).any():
        raise RuntimeError(f"{name}/{protocol}: duplicate (uid, window) test rows")
    return write_table(df, out_dir / f"{name}__seed{seed}__{PRED_SPLIT_NAME[protocol]}.parquet")


def run_one(args, data: WindowData, protocol: str, n_heads: int, cfg: TrainConfig, device: str,
            amp: bool, out_dir: Path) -> dict:
    from src.models.itransformer import ITransformer
    name = run_name(args, data)
    model_seed = args.seed if args.model_seed is None else args.model_seed
    tag = f"{protocol}_{name}_h{n_heads}_seed{args.seed}"
    path = out_dir / "raw" / f"{tag}.json"
    pred_dir = resolve(args.save_predictions)
    pred_name = args.pred_name or ("itransformer" if data.window_hours == data.target_hours
                                   else f"itransformer_w{data.window_hours:g}h")
    pred_path = None if pred_dir is None or protocol not in PRED_SPLIT_NAME else \
        pred_dir / f"{pred_name}__seed{args.seed}__{PRED_SPLIT_NAME[protocol]}.parquet"
    if path.exists() and not args.force and (pred_path is None or pred_path.exists()):
        print(f"  {tag}: already done (skipping; --force to rerun)")
        return json.loads(path.read_text())

    print(f"  {tag}: training (model seed {model_seed})")
    m = data.split_masks(protocol)
    tr = data.train_sample(m)
    val_idx, test_idx = np.flatnonzero(m["val"]), np.flatnonzero(m["test"])
    t0 = time.time()
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    top_idx = data.top_campaigns(tr, args.top_campaigns)        # this protocol's training sample only
    model = ITransformer(top_idx, n_static=data.n_static, n_camp=data.n_campaigns, n_heads=n_heads,
                         lookback=data.lookback, n_activity=len(FEATS), d=args.d_model, layers=args.layers,
                         dropout=args.dropout, copy_history=not args.no_copy_history).to(device)
    state = train(model, data, tr, val_idx, cfg, model_seed, device, amp)
    if args.checkpoint_dir is not None:
        ck = resolve(args.checkpoint_dir)
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(state, ck / f"{tag}.pt")
    fit_s = time.time() - t0

    p_val, _ = predict(model, data, val_idx, cfg, device, amp)
    thr, _ = best_f1_threshold(data.y[val_idx], p_val)
    p_test, _ = predict(model, data, test_idx, cfg, device, amp)
    res = {"task": "conversion", "protocol": protocol, "model": name, "top_campaigns": args.top_campaigns,
           "heads": n_heads, "lookback": data.lookback, "window_hours": data.window_hours,
           "seed": args.seed, "model_seed": model_seed, "neg_rate": args.neg_rate,
           "val_pr_auc": float(average_precision_score(data.y[val_idx], p_val)),
           **conversion_metrics(data.y[test_idx], p_test, threshold=thr),
           "f1_threshold_from_val": thr, "n_test": int(len(test_idx)), "n_test_pos": int(data.y[test_idx].sum()),
           "n_params": int(sum(p.numel() for p in model.parameters())), "fit_seconds": round(fit_s, 1)}
    if pred_path is not None:
        saved = save_predictions(pred_dir, pred_name, args.seed, protocol, data, test_idx, p_test)
        res["predictions_file"] = str(saved.relative_to(REPO_ROOT)) if saved.is_relative_to(REPO_ROOT) else str(saved)
        print(f"    saved {len(test_idx):,} test predictions -> {saved}")
    if not args.skip_campaign_eval:
        cidx = data.campaign_sample(m, args.campaign_eval_max)
        _, scores = predict(model, data, cidx, cfg, device, amp, campaign=True)
        y_true = np.zeros(scores.shape, dtype=np.int8)
        r, c = data.next_campaigns(cidx)
        y_true[r, c] = 1
        res["campaign"] = data.ranking_all_and_new(y_true, data.seen_matrix(cidx), scores)
    res["total_seconds"] = round(time.time() - t0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    camp = res.get("campaign", {})
    print(f"  {tag}: test PR-AUC {res['pr_auc']:.4f} | ROC-AUC {res['roc_auc']:.4f} | "
          f"campaign Recall@10 {camp.get('recall_at_10', float('nan')):.4f} | {res['total_seconds']:.0f}s")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def write_tables(out_dir: Path, model: str) -> None:
    results = [json.loads(f.read_text()) for f in sorted((out_dir / "raw").glob("*.json"))]
    if not results:
        return
    df = pd.DataFrame(results)
    extra = [c for c in ("top_campaigns", "heads", "lookback", "window_hours", "neg_rate", "n_params")
             if c in df.columns]
    df[[c for c in CONV_COLS if c in df.columns] + extra].to_csv(out_dir / f"conversion_{model}.csv", index=False)
    camp = [{"protocol": r["protocol"], "model": r["model"], "seed": r["seed"], **r["campaign"]}
            for r in results if "campaign" in r]
    if camp:
        pd.DataFrame(camp).to_csv(out_dir / f"campaign_{model}.csv", index=False)


def main() -> None:
    args = parse_args()
    data_dir = resolve(args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    out_dir = resolve(args.out_dir) or REPO_ROOT / "results" / "tables" / f"{args.model}_{data_dir.name}"
    print(f"Device {device} | bf16 {amp} | torch {torch.__version__} | data {data_dir} | out {out_dir}")

    data = WindowData(data_dir, lookback_hours=args.lookback_hours, neg_rate=args.neg_rate,
                      neg_seed=args.seed, history_features=not args.no_history_features,
                      max_shards=args.max_shards)
    for prot in args.protocols:
        m = data.split_masks(prot)
        tr = data.train_sample(m)
        print(f"{prot:<22} train sample {len(tr):,} ({int(data.y[tr].sum()):,} pos) | "
              f"val {int(m['val'].sum()):,} | test {int(m['test'].sum()):,} ({int(data.y[m['test']].sum()):,} pos)")
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience, batch=args.batch, eval_batch=args.eval_batch,
                      lr=args.lr, campaign_loss_weight=args.campaign_loss_weight)
    for prot in args.protocols:
        print(f"\n=== {prot} ===")
        for h in args.heads:
            run_one(args, data, prot, h, cfg, device, amp, out_dir)

    write_tables(out_dir, args.model)
    (out_dir / "run_config.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "device": device, "bf16": amp,
        "data_dir": data_dir.name, "window_hours": data.window_hours, "target_hours": data.target_hours,
        "lookback": data.lookback, "history_features": data.hist_cols, **{
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
