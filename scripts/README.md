# Scripts

Use this folder for thin command-line entry points that call reusable functions from `src/`.

Available:
- `download_criteo.sh` — download + unzip the Criteo Attribution dataset into `data/raw/`
- `prepare_criteo.py` — schema check (`--check-only`), 12h/6h windows, labels, split columns
- `zaratan/prepare_criteo.sbatch` — run the preparation on a Zaratan compute node

To add later:
- `train_hmm.py`
- `train_lstm.py`
- `train_transformer.py`
- `evaluate.py`
