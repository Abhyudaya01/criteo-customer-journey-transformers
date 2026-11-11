"""Train DARNN under our chronological protocol and save its test predictions.

Same inputs as team-aligned STAJT (docs/darnn_protocol.md):
  * events + vocab: STAJT's full event cache (every user), STAJT/data/stajt_full_eval_cache
  * training anchors: STAJT's team-aligned anchors (all positives + 1% of negatives),
    STAJT/data/ujt_fair_chrono_cache/train_anchors_neg_rate_0.0100.parquet
  * checkpoint selection: the plan's fixed validation subsample (all validation positives + 10%
    of negatives, negatives weighted x10 in PR-AUC), not the full 20.4M-position validation set
  * test: every chronological test position, scored once with the selected checkpoint

    # smoke test first (1% of users, 1 epoch; CPU or MIG, a few minutes)
    python scripts/train_darnn.py --smoke --epochs 1 --out checkpoints/darnn_smoke --no-save-predictions
    # full run on a MIG slice (scripts/zaratan/train_darnn.sbatch)
    python scripts/train_darnn.py --seed 1

Writes checkpoints/<out>/darnn_epoch_XX.pt (resumable with --resume), history.json, and
results/predictions/darnn__seed<k>__chrono_test.parquet + .meta.json (GPU-hours included), ready
for scripts/evaluate_predictions.py and the shared results table.
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.events import (  # noqa: E402
    EVENT_COLS, Anchors, EventStore, load_stajt_train_anchors, make_darnn_dataset,
)
from src.data.io import load_shard, shard_ids  # noqa: E402
from src.evaluation.results_table import write_run_meta  # noqa: E402
from src.models.darnn import DARNN, darnn_loss  # noqa: E402

STAJT_ROOT = REPO_ROOT / "STAJT"


def smoke_users(uid: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Deterministic ~pct% user sample (same users across runs and splits)."""
    h = pd.util.hash_array(np.asarray(uid, dtype=np.int64), hash_key="darnn-smoke-0001", categorize=False)
    return (h % np.uint64(10_000)) < np.uint64(int(pct * 100))


def load_events(event_dir: Path, keep_uid: np.ndarray | None) -> EventStore:
    parts = sorted(event_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no part-*.parquet in {event_dir}; run STAJT/evaluate_stajt_full.py "
                                "once to build the full event cache")
    frames = []
    keep = None if keep_uid is None else pd.Index(np.unique(keep_uid))
    for p in parts:
        ev = pd.read_parquet(p, columns=EVENT_COLS)
        if keep is not None:
            ev = ev[ev["uid"].isin(keep)]
        frames.append(ev)
    ev = pd.concat(frames, ignore_index=True)
    print(f"events: {len(ev):,} from {len(parts)} parts", flush=True)
    return EventStore(ev)


def load_positions(data_dir: Path, split: str) -> pd.DataFrame:
    parts = []
    for s in shard_ids(data_dir):
        pos = load_shard(data_dir, s, tables=["positions"])["positions"]
        parts.append(pos.loc[pos["time_split"].astype(str) == split,
                             ["uid", "window", "label_window", "y_conversion"]])
    return pd.concat(parts, ignore_index=True)


def val_subsample(val: pd.DataFrame, neg_frac: float = 0.10, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Plan compute rule 1: all positives + a fixed share of negatives, negatives weighted 1/neg_frac."""
    rng = np.random.default_rng(seed)
    keep = (val["y_conversion"].to_numpy() == 1) | (rng.random(len(val)) < neg_frac)
    sub = val.loc[keep].reset_index(drop=True)
    w = np.where(sub["y_conversion"].to_numpy() == 1, 1.0, 1.0 / neg_frac)
    return sub, w


def weighted_scores(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted PR-AUC and ROC-AUC; NaN when the sample has only one class (tiny smoke runs)."""
    if y.min() == y.max():
        print("warning: validation sample has a single class; metrics are NaN", flush=True)
        return float("nan"), float("nan")
    return (float(average_precision_score(y, s, sample_weight=w)),
            float(roc_auc_score(y, s, sample_weight=w)))


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast(device_type=device.type, enabled=amp):
            out.append(torch.sigmoid(model(b)["conv_logit"].float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/processed/criteo_w12h")
    ap.add_argument("--event-cache", type=Path, default=STAJT_ROOT / "data/stajt_full_eval_cache")
    ap.add_argument("--anchor-cache", type=Path, default=STAJT_ROOT / "data/ujt_fair_chrono_cache")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/darnn")
    ap.add_argument("--pred-dir", type=Path, default=REPO_ROOT / "results/predictions")
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3, help="stop after this many epochs without a better val PR-AUC")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="click-loss weight: L = L_conv + alpha * L_click")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--no-last-click-step", action="store_true", help="paper-faithful decoder (ablation)")
    ap.add_argument("--num-workers", type=int, default=int(_THREADS))
    ap.add_argument("--smoke", action="store_true", help="1%% of users, for a quick end-to-end check")
    ap.add_argument("--smoke-pct", type=float, default=1.0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.smoke and not args.no_save_predictions:
        sys.exit("--smoke needs --no-save-predictions: smoke-run predictions must not land in the shared folder")
    t_start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = device.type == "cuda"
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device: {device} ({hardware})", flush=True)

    meta = json.loads((args.event_cache / "vocab_meta.json").read_text())
    context_sizes = [meta["context_vocab_sizes"][f"cat{i}"] for i in range(1, 10)]

    train_a = load_stajt_train_anchors(args.anchor_cache)
    val_pos = load_positions(args.data_dir, "val")
    test_pos = load_positions(args.data_dir, "test")
    if args.smoke:
        keep = smoke_users(train_a.uid, args.smoke_pct)
        train_a = Anchors(train_a.uid[keep], train_a.anchor_ts[keep], train_a.y[keep], train_a.window[keep])
        val_pos = val_pos[smoke_users(val_pos["uid"].to_numpy(), args.smoke_pct)]
        test_pos = test_pos[smoke_users(test_pos["uid"].to_numpy(), args.smoke_pct)]
    val_sub, val_w = val_subsample(val_pos)
    val_a, test_a = Anchors.from_frame(val_sub), Anchors.from_frame(test_pos.reset_index(drop=True))
    print(f"anchors: train {len(train_a):,} ({int(train_a.y.sum()):,} pos) | val subsample {len(val_a):,} "
          f"({int(val_a.y.sum()):,} pos) | test {len(test_a):,} ({int(test_a.y.sum()):,} pos)", flush=True)

    users = np.concatenate([train_a.uid, val_a.uid, test_a.uid]) if args.smoke else None
    store = load_events(args.event_cache / "events", users)

    mk = lambda a, bs, shuffle: DataLoader(  # noqa: E731
        make_darnn_dataset(store, a), batch_size=bs, shuffle=shuffle, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = mk(train_a, args.batch_size, True)
    val_loader = mk(val_a, args.eval_batch_size, False)

    model = DARNN(meta["n_entities"], context_sizes, d_model=args.d_model,
                  last_click_step=not args.no_last_click_step).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:  # torch >= 2.3
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    history, start_epoch, best, stale = [], 1, (-1.0, None), 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        history, start_epoch = ck["history"], ck["epoch"] + 1
        for r in history:
            if r["val_pr_auc_weighted"] > best[0]:
                best = (r["val_pr_auc_weighted"], r["checkpoint"])
        print(f"resumed from {args.resume}: next epoch {start_epoch}, best {best}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, parts = 0.0, {"conversion": 0.0, "click": 0.0}
        for bi, b in enumerate(train_loader, start=1):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(b)
            loss, p = darnn_loss(out, b["y"], args.alpha)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {bi}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); parts = {k: parts[k] + p[k] for k in parts}
            if bi == 1 or bi % 500 == 0:
                print(f"epoch {epoch:02d} batch {bi:06d}/{len(train_loader):06d} loss {float(loss):.5f} "
                      f"(conv {p['conversion']:.5f}, click {p['click']:.5f}) {(time.time() - t0) / 60:.1f}m",
                      flush=True)
        nb = max(len(train_loader), 1)
        scores = predict(model, val_loader, device, amp)
        val_pr, val_roc = weighted_scores(val_a.y, scores, val_w)
        row = {"epoch": epoch, "train_loss": tot / nb, "train_conv_loss": parts["conversion"] / nb,
               "train_click_loss": parts["click"] / nb,
               "val_pr_auc_weighted": val_pr, "val_roc_auc_weighted": val_roc,
               "lambda": float(model.lam.detach().cpu()), "minutes": (time.time() - t0) / 60}
        ckpt = args.out / f"darnn_epoch_{epoch:02d}.pt"
        row["checkpoint"] = str(ckpt)
        history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history, "meta": meta, "config": config}, ckpt)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if best[1] is None or row["val_pr_auc_weighted"] > best[0]:  # NaN never wins after the first
            best, stale = (row["val_pr_auc_weighted"], str(ckpt)), 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no improvement for {stale} epochs; stopping", flush=True)
                break

    print(f"selected {best[1]} (weighted val PR-AUC {best[0]:.5f})", flush=True)
    model.load_state_dict(torch.load(best[1], map_location="cpu")["model"])
    test_loader = mk(test_a, args.eval_batch_size, False)
    scores = predict(model, test_loader, device, amp)
    pr = float(average_precision_score(test_a.y, scores))
    print(f"test (chronological, {len(scores):,} positions): PR-AUC {pr:.5f}, base rate {test_a.y.mean():.5f}",
          flush=True)
    gpu_hours = (time.time() - t_start) / 3600 if device.type == "cuda" else 0.0
    summary = {"selected": best[1], "val_pr_auc_weighted": best[0], "test_pr_auc": pr,
               "test_positions": int(len(scores)), "gpu_hours": gpu_hours, "hardware": hardware,
               "smoke": args.smoke}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_save_predictions:
        args.pred_dir.mkdir(parents=True, exist_ok=True)
        model_name = "darnn" if not args.no_last_click_step else "darnn_paper_decoder"
        path = args.pred_dir / f"{model_name}__seed{args.seed}__chrono_test.parquet"
        pd.DataFrame({"uid": test_a.uid, "window": test_a.window.astype("int32"), "score": scores}).to_parquet(path)
        write_run_meta(path, config="team_aligned_neg1pct", history="events64",
                       source="scripts/train_darnn.py", gpu_hours=gpu_hours, hardware=hardware,
                       notes=f"selected {Path(best[1]).name}; last_click_step={not args.no_last_click_step}")
        print(f"saved {len(scores):,} test predictions -> {path}", flush=True)


if __name__ == "__main__":
    main()
