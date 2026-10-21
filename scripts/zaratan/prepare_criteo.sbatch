#!/bin/bash
# Build the 12h and 6h windowed datasets on a Zaratan compute node (CPU only).
#
# Submit from the repo root (Slurm needs logs/ to exist before the job starts):
#   mkdir -p logs && sbatch scripts/zaratan/prepare_criteo.sbatch
# Check:  squeue -u $USER      Log: logs/prepare_criteo-<jobid>.out
#
# Before the first run, fill in your allocation account below (find it with `sbalance`
# or `sacctmgr show assoc user=$USER format=account%30`), and confirm the partition name
# with `sinfo -s` (CPU jobs normally go to the standard partition).
#SBATCH --job-name=prepare_criteo
#SBATCH --account=CHANGE_ME
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Activate the project environment (created once with: conda env create -f environment.yml).
# If conda is provided as a module on your account, load it first, e.g. `module load python`
# or the miniforge/anaconda module listed by `module avail conda`.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate criteo-journey

python -c "import pandas, pyarrow; print('pandas', pandas.__version__, 'pyarrow', pyarrow.__version__)"
python scripts/prepare_criteo.py --check-only
python scripts/prepare_criteo.py --all-windows
