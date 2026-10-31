# Scripts

Use this folder for thin command-line entry points that call reusable functions from `src/`.

Available:
- `download_criteo.sh` — download + unzip the Criteo Attribution dataset into `data/raw/`
- `prepare_criteo.py` — schema check (`--check-only`), 12h/6h windows, labels, split columns
- `zaratan/prepare_criteo.sbatch` — run the preparation on a Zaratan compute node
- `run_baselines.py` — non-sequential conversion and campaign baselines -> `results/tables/baselines_*.{json,md}`
- `dataset_stats.py` — dataset statistics and report figures -> `results/tables/dataset_stats_*`, `results/figures/`
- `zaratan/run_baselines.sbatch` — baselines on the full 12h data (CPU, ~15 min)

To add later:
- `train_lstm.py`
- `train_transformer.py`
- `evaluate.py`
