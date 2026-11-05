#!/bin/bash
# Baselines with saved test predictions, then user-bootstrap CIs + subgroups (CPU only, no GPU).
#
# First real-data run of scripts/evaluate_predictions.py, before the Transformer prediction
# files exist. Trains logistic + GBDT on all positives + 1% of negatives (the team's sample),
# saves test predictions for both protocols, then scores them. Campaign baselines are skipped
# (already in results/tables/baselines_criteo_w12h.md).
#
# Cost: ~25 min for the baselines (October run: 21 min on 4 CPUs) + ~10 min for scoring.
#
# Submit from the repo root with the conda env active:
#   mkdir -p logs results/predictions && sbatch scripts/zaratan/baselines_ci.sbatch
#SBATCH --job-name=baselines-ci
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
DATA=data/processed/criteo_w12h

python scripts/run_baselines.py --data-dir "${DATA}" --neg-rate 0.01 --skip-campaign \
    --save-predictions results/predictions --out-dir results/tables/baselines_neg001

for SPLIT in chrono_test user_test; do
    python scripts/evaluate_predictions.py --data-dir "${DATA}" --split "${SPLIT}" \
        --reference gbdt --n-boot 1000 --subgroup-n-boot 500 --neg-frac 0.1 --n-jobs 8
done
