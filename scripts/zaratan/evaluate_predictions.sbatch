#!/bin/bash
# Bootstrap CIs over users + subgroup results from saved predictions (CPU only, no GPU).
#
# Expects results/predictions/<model>__seed<k>__<split>.parquet (see scripts/evaluate_predictions.py).
# Cost estimate: ~0.17 s per replicate per prediction file on 10% of negatives, so 15 files x
# 1000 replicates is ~45 CPU-minutes, about 10 minutes wall time on 8 CPUs plus data loading.
#
# Submit from the repo root with the conda env active:
#   mkdir -p logs && sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test
#   mkdir -p logs && sbatch scripts/zaratan/evaluate_predictions.sbatch user_test
# Optional second argument = reference model for paired differences (default itransformer):
#   sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test gbdt
# Smoke test first (1% user sample or the smoke dataset, a few seconds):
#   python scripts/evaluate_predictions.py --data-dir data/processed/criteo_w12h_smoke --n-boot 50
#SBATCH --job-name=eval-preds
#SBATCH --account=msml612-class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
SPLIT="${1:-chrono_test}"
REF="${2:-itransformer}"
python scripts/evaluate_predictions.py --data-dir data/processed/criteo_w12h --split "${SPLIT}" \
    --reference "${REF}" --n-boot 1000 --subgroup-n-boot 500 --neg-frac 0.1 --n-jobs 8
