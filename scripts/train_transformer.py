"""Train a window-based sequence model and save test results and predictions.

Script version of the Colab training notebooks (same data, models, training sample and
evaluation), so the team has one run source on Zaratan and Colab:

    --model itransformer   notebooks/colab/training/Criteo_iTransformer_Colab.ipynb
    --model lu_kannan      notebooks/colab/training/Criteo_LuKannan_Colab.ipynb
    --model patchtst       notebooks/colab/training/Criteo_PatchTST_Colab.ipynb
    --model lstm           the LSTM of notebooks/colab/training/Criteo_Baselines_Colab.ipynb

Examples (repo root, env active):

    # seconds on the 2% sample (prepare_criteo.py --user-sample 50 --tag smoke)
    python scripts/train_transformer.py --model lu_kannan --data-dir data/processed/criteo_w12h_smoke --epochs 1

    # full 12h data, one seed, both protocols, predictions for scripts/evaluate_predictions.py
    python scripts/train_transformer.py --model lu_kannan --data-dir data/processed/criteo_w12h --seed 1 \\
        --save-predictions results/predictions

    # ablations (proposal): attention heads 1-4 and customer-specific weighting (Lu & Kannan)
    python scripts/train_transformer.py --model lu_kannan --data-dir data/processed/criteo_w12h --seed 1 \\
        --heads 1 2 3 4 --save-predictions results/predictions
    python scripts/train_transformer.py --model lu_kannan --data-dir data/processed/criteo_w12h --seed 1 \\
        --weighting per_user uniform amortized --save-predictions results/predictions

    # window-length ablation: 6h input tokens, 12h targets (docs/window_length_ablation.md)
    python scripts/train_transformer.py --data-dir data/processed/criteo_w6h_in_w12h --seed 1 \\
        --save-predictions results/predictions          # saves itransformer_w6h__seed1__*.parquet

Training budget. Every model: at most ``--epochs`` 10, early stopping after ``--patience`` 2
epochs without a better validation-sample PR-AUC, best epoch kept. The LSTM keeps the notebook's
optimizer (Adam, no weight decay, no clipping), batch 2048 and fp32; only its epoch cap changes
from the notebook's 8 to the shared 10.

Seeds. ``--seed`` draws the 1% negative training sample and seeds the model, like the baseline
seeds (scripts/zaratan/baselines_seeds.sbatch): each seed is a different negative sample.
``--model-seed`` sets the model seed separately; ``--seed 42 --model-seed 0`` reproduces the
Colab runs in results/tables/colab_v2/ (LSTM: with ``--epochs 8``).

Lookback is in hours (default 168 = 7 days): 14 windows of 12h, or 28 windows of 6h, so the 6h
model sees the same history span as the 12h one.

Outputs (``--out-dir``, default results/tables/<model>_<dataset>/):
    raw/<protocol>_<name>[_h<heads>]_seed<seed>.json   one run: conversion (+ campaign) metrics
    conversion_<model>.csv, campaign_<model>.csv        every run in raw/
    run_config.json
With ``--save-predictions DIR``: DIR/<pred-name>__seed<seed>__<chrono_test|user_test>.parquet
(columns uid, window, score; one row per test position, 12h window keys), see
results/predictions/README.md, plus the ``.meta.json`` sidecar that puts the run in the shared
results table (docs/results_table.md): config ``neg1pct``, history ``w12h_14lags`` or
``w6h_28lags_in_w12h``, GPU-hours, hardware, best epoch. Prediction names are the model name
(``itransformer``, ``lu_kannan``, ``patchtst``, ``lstm``) plus a suffix for every setting that
differs from the main configuration: ``_uniform`` / ``_amortized`` (weighting), ``_p4s2`` (patch),
``_h1`` .. ``_h3`` (heads; 4 is the main setting), ``_w6h`` (6h input). Runs whose raw JSON
exists are skipped unless ``--force``.
Model checkpoints go to ``--checkpoint-dir`` only if given (they are not committed).
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "8")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from dataclasses import dataclass, field
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

MODELS = ("itransformer", "lu_kannan", "patchtst", "lstm")
MAIN_HEADS = 4                       # main configuration of every Transformer (other values = head ablation)
MAIN_PATCH = (2, 2)
PRED_SPLIT_NAME = {"chronological": "chrono_test", "user_disjoint": "user_test"}
CONV_COLS = ["protocol", "model", "seed", "pr_auc", "roc_auc", "f1", "balanced_accuracy",
             "binary_cross_entropy", "brier", "expected_calibration_error", "top_decile_conversion_rate",
             "top_decile_lift", "top_decile_gain", "base_rate", "val_pr_auc", "n_test", "n_test_pos",
             "fit_seconds"]


@dataclass
class Variant:
    """One configuration to train: heads plus the model-specific setting."""
    heads: int = MAIN_HEADS
    weighting: str = "per_user"                    # lu_kannan
    patch: tuple = MAIN_PATCH                      # patchtst
    info: dict = field(default_factory=dict)       # extra fields for the raw JSON


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="itransformer")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["chronological", "user_disjoint"],
                    choices=["chronological", "user_disjoint", "user_disjoint_future"])
    ap.add_argument("--seed", type=int, default=1, help="negative sample and model seed")
    ap.add_argument("--model-seed", type=int, default=None, help="model seed if different from --seed")
    ap.add_argument("--neg-rate", type=float, default=0.01)
    ap.add_argument("--lookback-hours", type=float, default=168)
    ap.add_argument("--no-history-features", action="store_true")
    ap.add_argument("--heads", type=int, nargs="+", default=[MAIN_HEADS],
                    help="attention heads; several values = head ablation (ignored for lstm)")
    ap.add_argument("--top-campaigns", type=int, default=100, help="itransformer: campaigns with their own token")
    ap.add_argument("--weighting", nargs="+", default=["per_user"], choices=["per_user", "uniform", "amortized"],
                    help="lu_kannan: head weighting; several values = customer-weighting ablation")
    ap.add_argument("--patches", nargs="+", default=["2:2"], metavar="LEN:STRIDE",
                    help="patchtst: patch length and stride in windows, e.g. 2:2 4:2")
    ap.add_argument("--lstm-hidden", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-copy-history", action="store_true")
    ap.add_argument("--campaign-loss-weight", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=None, help="default 1024 (lstm: 2048, as in the notebook)")
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--campaign-eval-max", type=int, default=200_000)
    ap.add_argument("--skip-campaign-eval", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None, help="debug: first N user shards only")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save-predictions", type=Path, default=None, metavar="DIR")
    ap.add_argument("--pred-name", default=None,
                    help="model name in prediction files (default: automatic, see above); one configuration only")
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        args.patches = [tuple(int(v) for v in p.split(":")) for p in args.patches]
        assert all(len(p) == 2 and min(p) > 0 for p in args.patches)
    except (ValueError, AssertionError):
        ap.error("--patches takes LEN:STRIDE pairs of positive integers, e.g. 2:2 4:2")
    if args.pred_name and len(variants(args)) > 1:
        ap.error("--pred-name names one configuration; drop it to use the automatic names")
    return args


def variants(args) -> list[Variant]:
    if args.model == "lstm":
        return [Variant()]
    if args.model == "lu_kannan":
        return [Variant(heads=h, weighting=w, info={"weighting": w}) for w in args.weighting for h in args.heads]
    if args.model == "patchtst":
        return [Variant(heads=h, patch=p, info={"patch_len": p[0], "stride": p[1]})
                for p in args.patches for h in args.heads]
    return [Variant(heads=h, info={"top_campaigns": args.top_campaigns}) for h in args.heads]


def resolve(p: Path | None) -> Path | None:
    return None if p is None else (p if p.is_absolute() else REPO_ROOT / p)


def window_suffix(data: WindowData) -> str:
    return "" if data.window_hours == data.target_hours else f"_w{data.window_hours:g}h"


def run_name(args, data: WindowData, v: Variant) -> str:
    """Name in the raw JSON / CSV tables (same as the notebooks' model names)."""
    name = {"itransformer": f"itransformer_top{args.top_campaigns}", "lu_kannan": f"lu_kannan_{v.weighting}",
            "patchtst": f"patchtst_p{v.patch[0]}s{v.patch[1]}", "lstm": "lstm"}[args.model]
    if data.lookback != 14:
        name += f"_L{data.lookback}"
    return name + window_suffix(data)


def pred_name(args, data: WindowData, v: Variant) -> str:
    """Model name in the prediction files and the shared results table: main configuration = bare name."""
    if args.pred_name:
        return args.pred_name
    name = args.model
    if args.model == "itransformer" and args.top_campaigns != 100:
        name += f"_top{args.top_campaigns}"
    if args.model == "lu_kannan" and v.weighting != "per_user":
        name += f"_{v.weighting}"
    if args.model == "patchtst" and tuple(v.patch) != MAIN_PATCH:
        name += f"_p{v.patch[0]}s{v.patch[1]}"
    if args.model != "lstm" and v.heads != MAIN_HEADS:
        name += f"_h{v.heads}"
    return name + window_suffix(data)


def build_model(args, data: WindowData, v: Variant, tr: np.ndarray):
    common = dict(n_static=data.n_static, n_camp=data.n_campaigns)
    tf = dict(lookback=data.lookback, d=args.d_model, layers=args.layers, dropout=args.dropout,
              copy_history=not args.no_copy_history)
    if args.model == "itransformer":
        from src.models.itransformer import ITransformer
        top_idx = data.top_campaigns(tr, args.top_campaigns)     # this protocol's training sample only
        return ITransformer(top_idx, n_heads=v.heads, n_activity=len(FEATS), **common, **tf)
    if args.model == "lu_kannan":
        from src.models.lu_kannan import LuKannanTransformer
        model = LuKannanTransformer(n_num=data.Fw.shape[1] + 3, n_users=len(data.uids), n_heads=v.heads,
                                    weighting=v.weighting, **common, **tf)
        model.mark_seen(np.unique(data.pos_u[tr]))              # training customers (per_user weights)
        return model
    if args.model == "patchtst":
        from src.models.patchtst import PatchTST
        return PatchTST(len(FEATS), patch_len=v.patch[0], stride=v.patch[1], n_heads=v.heads, **common, **tf)
    from src.models.lstm import LSTMNet
    return LSTMNet(len(FEATS), data.n_static, hidden=args.lstm_hidden)


def train_config(args) -> TrainConfig:
    common = dict(epochs=args.epochs, patience=args.patience, eval_batch=args.eval_batch, lr=args.lr)
    if args.model == "lstm":                       # the notebook's LSTM loop
        return TrainConfig(batch=args.batch or 2048, campaign_loss_weight=0.0, grad_clip=None,
                           optimizer="adam", shuffle="torch", **common)
    return TrainConfig(batch=args.batch or 1024, campaign_loss_weight=args.campaign_loss_weight, **common)


def save_predictions(out_dir: Path, name: str, seed: int, protocol: str, data: WindowData,
                     idx: np.ndarray, scores: np.ndarray) -> Path:
    df = pd.DataFrame({"uid": data.pos_uid[idx], "window": data.pos_key_window[idx].astype("int32"),
                       "score": np.asarray(scores, dtype="float64")})
    if df.duplicated(["uid", "window"]).any():
        raise RuntimeError(f"{name}/{protocol}: duplicate (uid, window) test rows")
    return write_table(df, out_dir / f"{name}__seed{seed}__{PRED_SPLIT_NAME[protocol]}.parquet")


def history_label(data: WindowData) -> str:
    """Input representation for the results table, e.g. w12h_14lags or w6h_28lags_in_w12h."""
    lab = f"w{data.window_hours:g}h_{data.lookback}lags"
    return lab if data.window_hours == data.target_hours else f"{lab}_in_w{data.target_hours:g}h"


def write_meta(pred_path: Path, **meta) -> Path | None:
    """Sidecar for results/tables/results.csv (docs/results_table.md), if the results table is in the repo.

    Without src/evaluation/results_table.py the scorer would read the sidecar as a prediction file,
    so nothing is written; tag such files later with scripts/tag_predictions.py.
    """
    try:
        from src.evaluation.results_table import write_run_meta
    except ImportError:
        print("    (no results table in this checkout: .meta.json not written)")
        return None
    return write_run_meta(pred_path, **meta)


def run_one(args, data: WindowData, protocol: str, v: Variant, cfg: TrainConfig, device: str,
            amp: bool, out_dir: Path) -> dict:
    name = run_name(args, data, v)
    model_seed = args.seed if args.model_seed is None else args.model_seed
    tag = f"{protocol}_{name}" + ("" if args.model == "lstm" else f"_h{v.heads}") + f"_seed{args.seed}"
    path = out_dir / "raw" / f"{tag}.json"
    pred_dir = resolve(args.save_predictions)
    pname = pred_name(args, data, v)
    pred_path = None if pred_dir is None or protocol not in PRED_SPLIT_NAME else \
        pred_dir / f"{pname}__seed{args.seed}__{PRED_SPLIT_NAME[protocol]}.parquet"
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
    model = build_model(args, data, v, tr).to(device)
    state, info = train(model, data, tr, val_idx, cfg, model_seed, device, amp)
    if args.checkpoint_dir is not None:
        ck = resolve(args.checkpoint_dir)
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(state, ck / f"{tag}.pt")
    fit_s = time.time() - t0

    p_val, _ = predict(model, data, val_idx, cfg, device, amp)
    thr, _ = best_f1_threshold(data.y[val_idx], p_val)
    p_test, _ = predict(model, data, test_idx, cfg, device, amp)
    res = {"task": "conversion", "protocol": protocol, "model": name, **v.info,
           **({} if args.model == "lstm" else {"heads": v.heads}),
           "lookback": data.lookback, "window_hours": data.window_hours,
           "seed": args.seed, "model_seed": model_seed, "neg_rate": args.neg_rate,
           "epoch_cap": cfg.epochs, "patience": cfg.patience,
           "val_pr_auc": float(average_precision_score(data.y[val_idx], p_val)),
           **conversion_metrics(data.y[test_idx], p_test, threshold=thr),
           "f1_threshold_from_val": thr, "n_test": int(len(test_idx)), "n_test_pos": int(data.y[test_idx].sum()),
           "n_params": int(sum(p.numel() for p in model.parameters())), "fit_seconds": round(fit_s, 1), **info}
    if pred_path is not None:
        saved = save_predictions(pred_dir, pname, args.seed, protocol, data, test_idx, p_test)
        on_gpu = device == "cuda"
        side = write_meta(saved, config=f"neg{args.neg_rate * 100:g}pct", history=history_label(data),
                          source="scripts/train_transformer.py",
                          gpu_hours=round((time.time() - t0) / 3600, 4) if on_gpu else 0.0,
                          hardware=torch.cuda.get_device_name(0) if on_gpu else "cpu",
                          notes=f"best epoch {info['best_epoch']}/{info['epochs_run']}, model seed {model_seed}")
        res["predictions_file"] = str(saved.relative_to(REPO_ROOT)) if saved.is_relative_to(REPO_ROOT) else str(saved)
        print(f"    saved {len(test_idx):,} test predictions -> {saved}" + (f" (+ {side.name})" if side else ""))
    if not args.skip_campaign_eval and args.model != "lstm":       # LSTM is conversion-only
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
    extra = [c for c in ("top_campaigns", "weighting", "patch_len", "stride", "heads", "lookback", "window_hours",
                         "neg_rate", "epoch_cap", "best_epoch", "epochs_run", "n_params") if c in df.columns]
    df[[c for c in CONV_COLS if c in df.columns] + extra].to_csv(out_dir / f"conversion_{model}.csv", index=False)
    camp = [{"protocol": r["protocol"], "model": r["model"], "seed": r["seed"], **r["campaign"]}
            for r in results if "campaign" in r]
    if camp:
        pd.DataFrame(camp).to_csv(out_dir / f"campaign_{model}.csv", index=False)


def main(argv=None) -> None:
    args = parse_args(argv)
    data_dir = resolve(args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast for the Transformers on GPU; the notebook trained the LSTM in fp32.
    amp = device == "cuda" and torch.cuda.is_bf16_supported() and args.model != "lstm"
    out_dir = resolve(args.out_dir) or REPO_ROOT / "results" / "tables" / f"{args.model}_{data_dir.name}"
    print(f"Model {args.model} | device {device} | bf16 {amp} | torch {torch.__version__} | data {data_dir} | "
          f"out {out_dir}")

    data = WindowData(data_dir, lookback_hours=args.lookback_hours, neg_rate=args.neg_rate,
                      neg_seed=args.seed, history_features=not args.no_history_features,
                      max_shards=args.max_shards)
    for prot in args.protocols:
        m = data.split_masks(prot)
        tr = data.train_sample(m)
        print(f"{prot:<22} train sample {len(tr):,} ({int(data.y[tr].sum()):,} pos) | "
              f"val {int(m['val'].sum()):,} | test {int(m['test'].sum()):,} ({int(data.y[m['test']].sum()):,} pos)")
    cfg = train_config(args)
    for prot in args.protocols:
        print(f"\n=== {prot} ===")
        for v in variants(args):
            run_one(args, data, prot, v, cfg, device, amp, out_dir)

    write_tables(out_dir, args.model)
    (out_dir / "run_config.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "device": device, "bf16": amp,
        "data_dir": data_dir.name, "window_hours": data.window_hours, "target_hours": data.target_hours,
        "lookback": data.lookback, "history_features": data.hist_cols, "train_config": vars(cfg), **{
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
