# Models

Window models (trained with `scripts/train_transformer.py`, data from `src/training/window_data.py`,
loop in `src/training/loop.py`); each matches its Colab notebook in `notebooks/colab/training/`:

- `lu_kannan.py` — primary model: Lu & Kannan heterogeneous multi-head Transformer with
  customer-specific head weights (`per_user`; ablations `uniform`, `amortized`)
- `itransformer.py` — comparative Transformer (inverted attention across campaigns and activity series)
- `patchtst.py` — comparative Transformer (patched, channel-independent)
- `lstm.py` — recurrent baseline (conversion only)

Event-level and non-sequential models:

- `darnn.py` — DARNN (`scripts/train_darnn.py`)
- `baselines.py` — prior, recency, logistic, GBDT (`scripts/run_baselines.py`)

The HMM baseline is in `notebooks/colab/training/Criteo_Baselines_Colab.ipynb`; STAJT is in `STAJT/`.
