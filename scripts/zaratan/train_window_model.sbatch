#!/bin/bash
# Any window model (itransformer, lu_kannan, patchtst, lstm): train, evaluate and save test
# predictions for one or more seeds (GPU). Same as train_itransformer.sbatch with a model argument.
#
# Arguments:
#   1  model: itransformer | lu_kannan | patchtst | lstm          (required)
#   2  dataset folder name under data/processed                   (default criteo_w12h)
#   3  seeds, space-separated in quotes                           (default "1")
#   4  extra arguments for scripts/train_transformer.py, in quotes (optional)
#
# Submit from the repo root with the conda env active:
#   mkdir -p logs results/predictions
#   sbatch scripts/zaratan/train_window_model.sbatch lu_kannan criteo_w12h "1 2 3"     # main table seeds
#   sbatch scripts/zaratan/train_window_model.sbatch lstm      criteo_w12h "1 2 3"
#   sbatch scripts/zaratan/train_window_model.sbatch lu_kannan criteo_w12h 1 "--heads 1 2 3"            # head ablation
#   sbatch scripts/zaratan/train_window_model.sbatch lu_kannan criteo_w12h 1 "--weighting uniform amortized"
# (heads 4 / per_user = the seed-1 main run, so the ablations only add the other settings.)
# Then score: sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test (and user_test).
#
# Cost: on a Colab A100 one Transformer run took 5-9 min per protocol, the LSTM about 2 min
# (results/tables/colab_v2), plus data loading once per call. The models are small, so a MIG
# slice is enough for the GPU part; host memory is the limit (96 GB, as for iTransformer).
# The head ablation (3 settings x 2 protocols) fits in the 3-hour limit.
# If the MIG slice is busy, use --gres=gpu:a100:1 instead.
#SBATCH --job-name=window-model
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
MODEL="${1:?model: itransformer, lu_kannan, patchtst or lstm}"
DATA="${2:-criteo_w12h}"
SEEDS="${3:-1}"
EXTRA="${4:-}"

for SEED in ${SEEDS}; do
    # shellcheck disable=SC2086  # EXTRA holds several arguments
    python scripts/train_transformer.py --model "${MODEL}" --data-dir "data/processed/${DATA}" \
        --seed "${SEED}" --save-predictions results/predictions ${EXTRA}
done
