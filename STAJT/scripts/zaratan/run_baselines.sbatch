#!/bin/bash
# Non-sequential baselines on the full 12h dataset (CPU only, ~21 min, peak 18.4 GB, a few SU).
#
# Submit from the repo root with the conda env active:
#   mkdir -p logs && sbatch scripts/zaratan/run_baselines.sbatch
# Faster to start, half the users (fits the 15-min debug limit):
#   srun -A msml612-class -p debug -t 15 -c 4 --mem=32G \
#       python scripts/run_baselines.py --data-dir data/processed/criteo_w12h --max-shards 4
#SBATCH --job-name=baselines
#SBATCH --account=msml612-class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
python scripts/run_baselines.py --data-dir data/processed/criteo_w12h
