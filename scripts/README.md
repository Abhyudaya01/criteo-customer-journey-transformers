# Scripts

Use this folder for thin command-line entry points that call reusable functions from `src/`.

Available:
- `download_criteo.sh` — download + unzip the Criteo Attribution dataset into `data/raw/`
- `prepare_criteo.py` — schema check (`--check-only`), 12h/6h windows, labels, split columns
- `zaratan/prepare_criteo.sbatch` — run the preparation on a Zaratan compute node
- `run_baselines.py` — non-sequential conversion and campaign baselines -> `results/tables/baselines_*.{json,md}`
- `dataset_stats.py` — dataset statistics and report figures -> `results/tables/dataset_stats_*`, `results/figures/`
- `zaratan/run_baselines.sbatch` — baselines on the full 12h data (CPU, ~15 min)
- `train_transformer.py` — window models (`--model itransformer | lu_kannan | patchtst | lstm`):
  train, evaluate, save test predictions for `evaluate_predictions.py`; same models and sample as
  the Colab notebooks (identical results on synthetic data with the Colab seeds). Every model has
  the same budget: at most 10 epochs, early stopping with patience 2. Ablations: `--heads 1 2 3 4`,
  `--weighting per_user uniform amortized` (Lu & Kannan), `--patches 2:2 4:2` (PatchTST); each
  setting gets its own prediction name (`lu_kannan_h1`, `lu_kannan_uniform`, ...)
- `zaratan/train_window_model.sbatch` — any window model, seeds and extra arguments, on the GPU partition
- `zaratan/train_itransformer.sbatch` — iTransformer seeds on the GPU partition (12h or 6h-input data)

On Colab (no Zaratan account), from a clone of the repo with the data on Drive, save outside the
clone so nothing is lost when the runtime stops:

```bash
D=/content/drive/MyDrive/DATA612_data
python scripts/train_transformer.py --model lu_kannan --data-dir $D/processed/criteo_w12h --seed 1 \
    --out-dir $D/results/script/lu_kannan --save-predictions $D/predictions
```

Benchmark datasets (proposal: models x datasets table). `notebooks/colab/data_prep/` builds
`retailrocket_w12h`, `taobao_w3h` and `yoochoose_w5m` in the same format as Criteo, so the same
scripts run on them. Use 14 windows of history on every dataset (`--lookback-windows 14`), keep
each dataset's predictions in its own folder, and score each folder into its own table:

```bash
B=retailrocket_w12h          # or taobao_w3h, yoochoose_w5m
for m in lu_kannan itransformer patchtst lstm; do
  python scripts/train_transformer.py --model $m --data-dir $D/processed/$B --lookback-windows 14 \
      --seed 1 --out-dir $D/results/benchmarks/$B/$m --save-predictions $D/predictions/$B
done
python scripts/run_baselines.py --data-dir $D/processed/$B --neg-rate 0.01 \
    --out-dir $D/results/benchmarks/$B --save-predictions $D/predictions/$B
for sp in chrono_test user_test; do
  python scripts/evaluate_predictions.py --data-dir $D/processed/$B --split $sp \
      --pred-dir $D/predictions/$B --out-dir $D/results/benchmarks/$B/eval
done
```
