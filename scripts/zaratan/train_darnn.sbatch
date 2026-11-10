#!/bin/bash
# DARNN under our chronological protocol, on one MIG slice (1/7 of an A100, 5 GiB).
#
# DARNN is two small LSTMs, so a MIG slice is enough: 7 SU/hr vs 48 SU/hr for a full A100
# (hpcc.umd.edu/home/clusters/zaratan). Data loading is the bottleneck, hence 8 CPUs.
#
# Before the first submit, confirm the MIG GRES name on Zaratan:
#   sinfo -p gpu -o "%G" | tr ',' '\n' | sort -u        # expect gpu:a100_1g.5gb
#
# Plan compute rule: smoke-test first (1% of users, 1 epoch, ~10-15 min):
#   mkdir -p logs && sbatch --export=ALL,SMOKE=1 scripts/zaratan/train_darnn.sbatch
# Full run (seed 1, up to 10 epochs with early stopping; estimate 1.5-3 h):
#   sbatch scripts/zaratan/train_darnn.sbatch
# Resume after a timeout:
#   sbatch --export=ALL,RESUME=checkpoints/darnn/darnn_epoch_04.pt scripts/zaratan/train_darnn.sbatch
#SBATCH --job-name=darnn
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [[ "${SMOKE:-0}" == "1" ]]; then
    python scripts/train_darnn.py --smoke --epochs 1 --out checkpoints/darnn_smoke \
        --no-save-predictions --num-workers 8
else
    python scripts/train_darnn.py --seed 1 --out checkpoints/darnn --num-workers 8 \
        ${RESUME:+--resume "${RESUME}"}
fi
