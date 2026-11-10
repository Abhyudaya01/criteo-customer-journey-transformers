#!/bin/bash
# Seeds 2 and 3 for logistic + GBDT (CPU only), then re-score with seed-averaged CIs.
#
# Seed 1 is the existing seed-42 run from baselines_ci.sbatch. Each seed draws a different 1%
# negative training sample (and GBDT random_state). evaluate_predictions.py groups files by
# model name, so it averages all three seeds and reports their standard deviation.
#
# Cost: ~22 min of baselines per seed + ~5 min scoring per split, about 55 min in total.
#
# Submit from the repo root with the conda env active:
#   mkdir -p logs && sbatch scripts/zaratan/baselines_seeds.sbatch
#SBATCH --job-name=baselines-seeds
#SBATCH --account=msml612-class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
DATA=data/processed/criteo_w12h

for SEED in 2 3; do
    python scripts/run_baselines.py --data-dir "${DATA}" --neg-rate 0.01 --skip-campaign \
        --seed "${SEED}" --save-predictions results/predictions \
        --out-dir "results/tables/baselines_neg001_seed${SEED}"
done

for SPLIT in chrono_test user_test; do
    python scripts/evaluate_predictions.py --data-dir "${DATA}" --split "${SPLIT}" \
        --reference gbdt --n-boot 1000 --subgroup-n-boot 500 --neg-frac 0.1 --n-jobs 8
done
