#!/usr/bin/env python3
import argparse
import gc
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


SEED = 42
HOURS = 12
WINDOW_SECONDS = HOURS * 3600
N_SHARDS = 64
TRAIN_CUTOFF = 43 * WINDOW_SECONDS
CAT_COLS = [f"cat{i}" for i in range(1, 10)]
CAT_ID_COLS = [f"cat{i}_id" for i in range(1, 10)]
SAFE_RAW_COLS = ["timestamp", "uid", "campaign", "click", "cost", *CAT_COLS]
BASE_EVENT_COLS = [
    "uid", "timestamp", "entity_id", "raw_action_id", "behavior_family",
    "log_cost", *CAT_ID_COLS,
]
FINAL_EVENT_COLS = [
    "uid", "timestamp", "entity_id", "raw_action_id", "behavior_family",
    "log_cost", "log_delta_hours", "entity_seen_before", "log_entity_gap_hours",
    *CAT_ID_COLS,
]


def load_train_module(root: Path):
    path = root / "scripts" / "train_stajt_full.py"
    spec = importlib.util.spec_from_file_location("train_stajt_full", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gain_at_10(y, p):
    k = max(1, int(math.ceil(0.10 * len(y))))
    idx = np.argpartition(-p, k - 1)[:k]
    total = y.sum()
    return float(y[idx].sum() / total) if total > 0 else float("nan")


def metrics(y, p):
    g10 = gain_at_10(y, p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "gain_at_10": g10,
        "lift_at_10": g10 / 0.10,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
    }


def shard_ids(uids: pd.Series):
    # Must exactly match scripts/prepare_criteo.py with seed=42 and n_shards=64.
    key = f"{SEED + 1:016d}"[-16:]
    h = pd.util.hash_array(uids.to_numpy(), hash_key=key, categorize=False)
    return (h % np.uint64(N_SHARDS)).astype(np.int16)


def build_full_event_cache(raw_path: Path, cache_dir: Path, ckpt_meta: dict):
    final_dir = cache_dir / "events"
    meta_path = cache_dir / "vocab_meta.json"
    done_marker = cache_dir / "_SUCCESS"

    if done_marker.exists() and meta_path.exists():
        files = sorted(final_dir.glob("part-*.parquet"))
        if len(files) == N_SHARDS:
            print("Loading existing full evaluation event cache", flush=True)
            with open(meta_path) as f:
                meta = json.load(f)
            if meta != ckpt_meta:
                raise RuntimeError(f"Cached vocab meta does not match checkpoint meta.\ncache={meta}\nckpt={ckpt_meta}")
            return

    print("Building full evaluation event cache (one-time job)", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = cache_dir / "chunks"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True)

    # Pass 1: exactly reproduce the training-only vocabularies.
    print("Pass 1/2: reconstructing training-period vocabularies", flush=True)
    vocab_sets = {"campaign": set(), **{c: set() for c in CAT_COLS}}
    reader = pd.read_csv(
        raw_path, sep="\t", usecols=["timestamp", "campaign", *CAT_COLS],
        chunksize=1_000_000,
    )
    for ci, chunk in enumerate(reader, start=1):
        tr = chunk.loc[chunk["timestamp"] < TRAIN_CUTOFF]
        for c in ["campaign", *CAT_COLS]:
            vocab_sets[c].update(tr[c].dropna().unique().tolist())
        print(f"  vocab chunk {ci}", flush=True)

    campaign_values = sorted(vocab_sets["campaign"])
    campaign_map = {v: i + 1 for i, v in enumerate(campaign_values)}
    context_maps = {}
    context_sizes = {}
    for c in CAT_COLS:
        vals = sorted(vocab_sets[c])
        context_maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        context_sizes[c] = len(vals) + 1

    meta = {
        "n_entities": len(campaign_values) + 1,
        "context_vocab_sizes": context_sizes,
    }
    print("Reconstructed vocab meta:", meta, flush=True)
    if meta != ckpt_meta:
        raise RuntimeError(f"Reconstructed vocabulary does not match checkpoint.\nreconstructed={meta}\nckpt={ckpt_meta}")

    # Pass 2: encode all raw events and partition by the exact same user shard.
    print("Pass 2/2: encoding and partitioning all raw events", flush=True)
    reader = pd.read_csv(raw_path, sep="\t", usecols=SAFE_RAW_COLS, chunksize=1_000_000)
    for ci, chunk in enumerate(reader, start=1):
        chunk["entity_id"] = chunk["campaign"].map(campaign_map).fillna(0).astype(np.int32)
        chunk["raw_action_id"] = np.where(chunk["click"].to_numpy() > 0, 2, 1).astype(np.int8)
        chunk["behavior_family"] = chunk["raw_action_id"].astype(np.int8)
        for c in CAT_COLS:
            chunk[f"{c}_id"] = chunk[c].map(context_maps[c]).fillna(0).astype(np.int32)
        chunk["log_cost"] = np.log1p(
            np.maximum(chunk["cost"].to_numpy(np.float64), 0.0) * 1000.0
        ).astype(np.float32)
        sid = shard_ids(chunk["uid"])

        for s in np.unique(sid):
            s = int(s)
            out_dir = chunk_root / f"part-{s:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            part = chunk.loc[sid == s, BASE_EVENT_COLS]
            part.to_parquet(out_dir / f"chunk-{ci:03d}.parquet", index=False)
        print(f"  encoded raw chunk {ci}", flush=True)

    # Consolidate each user shard and compute the exact same causal event features.
    print("Consolidating 64 event shards", flush=True)
    for s in range(N_SHARDS):
        fragments = sorted((chunk_root / f"part-{s:03d}").glob("*.parquet"))
        if not fragments:
            raise RuntimeError(f"No event fragments for shard {s}")
        events = pd.concat([pd.read_parquet(p) for p in fragments], ignore_index=True)
        events = events.sort_values(["uid", "timestamp"], kind="mergesort").reset_index(drop=True)

        prev_ts = events.groupby("uid", sort=False)["timestamp"].shift(1)
        delta_h = ((events["timestamp"] - prev_ts) / 3600.0).fillna(0.0).clip(lower=0)
        events["log_delta_hours"] = np.log1p(delta_h).astype(np.float32)

        prev_entity_ts = events.groupby(["uid", "entity_id"], sort=False)["timestamp"].shift(1)
        seen = prev_entity_ts.notna()
        gap_h = ((events["timestamp"] - prev_entity_ts) / 3600.0).fillna(0.0).clip(lower=0)
        events["entity_seen_before"] = seen.astype(np.float32)
        events["log_entity_gap_hours"] = np.log1p(gap_h).astype(np.float32)

        events[FINAL_EVENT_COLS].to_parquet(final_dir / f"part-{s:03d}.parquet", index=False)
        shutil.rmtree(chunk_root / f"part-{s:03d}")
        print(f"  consolidated event shard {s+1:02d}/{N_SHARDS}: {len(events):,} events", flush=True)
        del events
        gc.collect()

    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    done_marker.write_text("ok\n")
    print("Full event cache ready", flush=True)


def load_models(train_mod, checkpoint_paths, device):
    models = []
    ckpts = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt["meta"]
        context_sizes = [meta["context_vocab_sizes"][f"cat{i}"] for i in range(1, 10)]
        model = train_mod.STAJT(meta["n_entities"], context_sizes)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        ckpts.append(ckpt)
        print(f"Loaded {path.name} (epoch={ckpt['epoch']})", flush=True)
    return models, ckpts


@torch.inference_mode()
def predict_loader(models, loader, device):
    ys = []
    preds = [[] for _ in models]
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        ys.append(b["y_conversion"].cpu().numpy().astype(np.float32, copy=False))
        for i, model in enumerate(models):
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(b)
            preds[i].append(torch.sigmoid(logits).float().cpu().numpy())
    y = np.concatenate(ys)
    p = [np.concatenate(parts) for parts in preds]
    return y, p


def anchors_for_shard(path: Path, split: str):
    df = pd.read_parquet(
        path,
        columns=["uid", "label_window", "y_conversion", "time_split"],
    )
    m = df["time_split"].astype(str).eq(split)
    a = df.loc[m, ["uid", "label_window", "y_conversion"]].copy()
    a["anchor_timestamp"] = a["label_window"].astype(np.int64) * WINDOW_SECONDS
    return a


def evaluate_split(
    train_mod,
    models,
    proc: Path,
    event_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device,
):
    all_y = []
    all_p = [[] for _ in models]
    t0 = time.time()

    position_files = sorted((proc / "positions").glob("part-*.parquet"))
    if len(position_files) != N_SHARDS:
        raise RuntimeError(f"Expected {N_SHARDS} position shards, found {len(position_files)}")

    for s, pos_path in enumerate(position_files):
        anchors = anchors_for_shard(pos_path, split)
        events = pd.read_parquet(event_dir / f"part-{s:03d}.parquet")
        store = train_mod.EventStore(events)
        ds = train_mod.JourneyDataset(anchors, store, train_mod.MAX_LEN)

        if len(ds) != len(anchors):
            raise RuntimeError(
                f"Shard {s}: dataset dropped {len(anchors)-len(ds):,} {split} anchors; "
                "full event cache should contain every user."
            )

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        y, pred_list = predict_loader(models, loader, device)
        all_y.append(y)
        for i, p in enumerate(pred_list):
            all_p[i].append(p)

        done = sum(len(x) for x in all_y)
        print(
            f"[{split}] shard {s+1:02d}/{N_SHARDS} "
            f"anchors={len(ds):,} cumulative={done:,} elapsed={(time.time()-t0)/60:.1f}m",
            flush=True,
        )

        del loader, ds, store, events, anchors, y, pred_list
        gc.collect()

    y = np.concatenate(all_y)
    preds = [np.concatenate(parts) for parts in all_p]
    return y, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--rebuild-event-cache", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed" / "criteo_w12h_full"
    raw = root / "data" / "raw" / "criteo_attribution_dataset" / "criteo_attribution_dataset.tsv.gz"
    ckpt_dir = root / "checkpoints" / "ujt_full"
    eval_cache = root / "data" / "stajt_full_eval_cache"
    out_dir = root / "results" / "stajt_full_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Load the Zaratan PyTorch/CUDA module first.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train_mod = load_train_module(root)

    checkpoint_paths = [
        ckpt_dir / "ujt_criteo_epoch_01.pt",
        ckpt_dir / "ujt_criteo_epoch_02.pt",
        ckpt_dir / "ujt_criteo_epoch_03.pt",
    ]
    for p in checkpoint_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    # Use checkpoint metadata as the source of truth for vocab dimensions.
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")
    ckpt_meta = first_ckpt["meta"]
    del first_ckpt

    if args.rebuild_event_cache and eval_cache.exists():
        shutil.rmtree(eval_cache)

    build_full_event_cache(raw, eval_cache, ckpt_meta)

    # Load all three small models at once so validation data is read only once.
    models, ckpts = load_models(train_mod, checkpoint_paths, device)

    print("\n=== FULL CHRONOLOGICAL VALIDATION: epochs 1-3 ===", flush=True)
    y_val, val_preds = evaluate_split(
        train_mod, models, proc, eval_cache / "events", "val",
        args.batch_size, args.num_workers, device,
    )

    val_rows = []
    for path, ckpt, p in zip(checkpoint_paths, ckpts, val_preds):
        m = metrics(y_val, p)
        row = {"checkpoint": path.name, "epoch": int(ckpt["epoch"]), **m}
        val_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    val_df = pd.DataFrame(val_rows).sort_values("epoch")
    val_df.to_csv(out_dir / "full_validation_epochs.csv", index=False)
    best = max(val_rows, key=lambda r: r["pr_auc"])
    best_idx = [r["epoch"] for r in val_rows].index(best["epoch"])
    print(
        f"\nBEST FULL-VAL CHECKPOINT: epoch {best['epoch']} "
        f"PR-AUC={best['pr_auc']:.6f} Gain@10={best['gain_at_10']:.4f}",
        flush=True,
    )

    # Free the two models we do not need for the final test.
    best_model = models[best_idx]
    models = [best_model]
    del val_preds, y_val
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== FULL CHRONOLOGICAL TEST: selected checkpoint only ===", flush=True)
    y_test, test_preds = evaluate_split(
        train_mod, models, proc, eval_cache / "events", "test",
        args.batch_size, args.num_workers, device,
    )
    test_m = metrics(y_test, test_preds[0])
    test_row = {
        "checkpoint": best["checkpoint"],
        "epoch": int(best["epoch"]),
        **test_m,
    }
    print(json.dumps(test_row, indent=2), flush=True)

    with open(out_dir / "full_test_best.json", "w") as f:
        json.dump(test_row, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"validation": val_rows, "selected": best, "test": test_row}, f, indent=2)

    print("\nSaved:", flush=True)
    print(out_dir / "full_validation_epochs.csv", flush=True)
    print(out_dir / "full_test_best.json", flush=True)
    print(out_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
