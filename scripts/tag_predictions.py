"""Write the .meta.json sidecar for prediction files that were saved without one.

The results table (results/tables/results.csv) takes each run's config, input representation,
training script and GPU-hours from these sidecars. New runs should call
src.evaluation.results_table.write_run_meta right after saving predictions; this tool covers
files that already exist.

    # the baseline files from Nov 4 (seed 42, 1% negatives, CPU)
    python scripts/tag_predictions.py results/predictions/{gbdt,logistic,recency}__seed42__*.parquet \\
        --config neg1pct --history w12h_features --source scripts/run_baselines.py \\
        --gpu-hours 0 --hardware cpu --run-where "zaratan job 23779722"

Existing sidecars are left alone unless --overwrite is given.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.results_table import sidecar_path, write_run_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gpu-hours", type=float, required=True)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--run-where", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for f in args.files:
        if not f.exists():
            sys.exit(f"{f} does not exist")
        if sidecar_path(f).exists() and not args.overwrite:
            print(f"keep  {sidecar_path(f).name} (exists; --overwrite to replace)")
            continue
        notes = args.notes or ("deterministic rule, no training" if f.name.startswith("recency__") else "")
        out = write_run_meta(f, config=args.config, history=args.history, source=args.source,
                             gpu_hours=args.gpu_hours, hardware=args.hardware, notes=notes,
                             run_where=args.run_where)
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
