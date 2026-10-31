#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


def load_eval_module(root):
    path = root / "scripts" / "evaluate_stajt_full.py"
    spec = importlib.util.spec_from_file_location("stajt_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ev = load_eval_module(root)

    proc = root / "data" / "processed" / "criteo_w12h_full"
    raw = (
        root / "data" / "raw" /
        "criteo_attribution_dataset" /
        "criteo_attribution_dataset.tsv.gz"
    )

    ckpt_dir = root / args.ckpt_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared FULL event cache. This is read-only after it has been built.
    eval_cache = root / "data" / "stajt_full_eval_cache"

    paths = [
        ckpt_dir / f"ujt_criteo_epoch_{e:02d}.pt"
        for e in args.epochs
    ]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Checkpoint directory:", ckpt_dir, flush=True)
    print("Epochs:", args.epochs, flush=True)

    train_mod = ev.load_train_module(root)

    first = torch.load(paths[0], map_location="cpu")
    ckpt_meta = first["meta"]
    del first

    ev.build_full_event_cache(raw, eval_cache, ckpt_meta)

    models, ckpts = ev.load_models(
        train_mod,
        paths,
        device,
    )

    print(
        "\n=== FULL CHRONOLOGICAL VALIDATION ONLY ===",
        flush=True,
    )
    print(
        "Expected: 20,421,186 positions / 34,430 positives",
        flush=True,
    )

    y, preds = ev.evaluate_split(
        train_mod,
        models,
        proc,
        eval_cache / "events",
        "val",
        args.batch_size,
        args.num_workers,
        device,
    )

    rows = []

    for path, ckpt, pred in zip(paths, ckpts, preds):
        m = ev.metrics(y, pred)

        row = {
            "checkpoint": path.name,
            "epoch": int(ckpt["epoch"]),
            **m,
        }

        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    df = pd.DataFrame(rows).sort_values("epoch")
    csv_path = out_dir / "full_validation_epochs.csv"
    df.to_csv(csv_path, index=False)

    best = max(rows, key=lambda x: x["pr_auc"])

    with open(out_dir / "best_validation.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBEST CHECKPOINT:", flush=True)
    print(json.dumps(best, indent=2), flush=True)

    print("\nSaved:", flush=True)
    print(csv_path, flush=True)
    print(out_dir / "best_validation.json", flush=True)

    print("\nNO TEST SET WAS EVALUATED.", flush=True)


if __name__ == "__main__":
    main()
