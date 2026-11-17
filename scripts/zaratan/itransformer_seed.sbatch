#!/bin/bash
# iTransformer, one seed, both protocols, saves test predictions (GPU, MIG slice).
# Usage: sbatch scripts/zaratan/itransformer_seed.sbatch <seed>
#SBATCH --job-name=itrans-seed
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
SEED="${1:?seed}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/train_transformer.py --model itransformer --data-dir data/processed/criteo_w12h \
    --seed "${SEED}" --save-predictions results/predictions
