#!/bin/bash
# iTransformer: train, evaluate and save test predictions for one or more seeds (GPU).
#
# Arguments (both optional):
#   1  dataset folder name under data/processed   (default criteo_w12h)
#   2  seeds, space-separated in quotes           (default "1")
#
# Submit from the repo root with the conda env active:
#   mkdir -p logs results/predictions
#   sbatch scripts/zaratan/train_itransformer.sbatch                          # 12h, seed 1
#   sbatch scripts/zaratan/train_itransformer.sbatch criteo_w12h "1 2 3"      # 12h, seeds 1-3
#   sbatch scripts/zaratan/train_itransformer.sbatch criteo_w6h_in_w12h 1     # window ablation
# Then score: sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test (and user_test).
#
# Smoke test first (2% user sample, about a minute on CPU):
#   python scripts/train_transformer.py --data-dir data/processed/criteo_w12h_smoke --epochs 1
#
# Cost: on a Colab A100 one seed took 7-8 min per protocol (training + evaluation, see
# results/tables/colab_v2/itransformer), plus data loading once per call. The model is small
# (0.2M parameters), so a MIG slice (a100_1g.5gb) should be enough for the GPU part. Host
# memory is the limit: Colab's High-RAM A100 runtime (83 GB) was enough for the full 12h data;
# 96 GB is requested here. The 6h-input data has twice as many user-windows.
# If the MIG slice is busy, use --gres=gpu:a100:1 instead.
#SBATCH --job-name=itransformer
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
DATA="${1:-criteo_w12h}"
SEEDS="${2:-1}"

for SEED in ${SEEDS}; do
    python scripts/train_transformer.py --model itransformer --data-dir "data/processed/${DATA}" \
        --seed "${SEED}" --save-predictions results/predictions
done
