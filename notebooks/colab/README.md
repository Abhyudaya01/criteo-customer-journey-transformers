# Colab notebooks (Abhyudaya)

Colab Pro versions of the data preparation and Tier 1 training runs. They reuse the repo's own
code (`src/data`, `src/evaluation/metrics.py`, `scripts/prepare_criteo.py`), so tables, labels,
splits and metrics match the Zaratan runs.

Google Drive layout: `MyDrive/DATA612_data/{raw,processed,results}`.

## Data preparation (CPU, High-RAM)

| Notebook | Output (`processed/`) | Notes |
|---|---|---|
| `data_prep/Criteo_DataPrep_Colab.ipynb` | `criteo_w12h` | Matches the Zaratan build exactly (126.0M positions) |
| `data_prep/Retailrocket_DataPrep_Colab.ipynb` | `retailrocket_w12h` | conversion = transaction; campaign = item category |
| `data_prep/Taobao_DataPrep_Colab.ipynb` | `taobao_w3h` | conversion = buy; 3-hour windows (9-day log) |
| `data_prep/YOOCHOOSE_DataPrep_Colab.ipynb` | `yoochoose_w5m` | sessions as journeys; 5-min session-relative windows; session-disjoint split |

## Training (GPU: A100 or L4, bf16)

Run in this order; each notebook skips runs whose result JSON already exists.

1. `training/Criteo_Baselines_Colab.ipynb`: prior, recency, logistic, GBDT, HMM, LSTM, campaign baselines
2. `training/Criteo_LuKannan_Colab.ipynb`: Lu & Kannan heterogeneous multi-head Transformer
3. `training/Criteo_PatchTST_Colab.ipynb`
4. `training/Criteo_iTransformer_Colab.ipynb`

Shared setup: every positive plus 1% of negatives (probabilities corrected back), 14-window
lookback, `src/data/features.py` history features, F1 threshold chosen on validation, both
protocols (chronological, user-disjoint). Results go to `results/v2_history/<model>/<dataset>/`.

The training notebooks and the three benchmark data-prep notebooks are the executed Colab copies, with
cell outputs from the runs (26 Oct 2026). Single seed so far; seeds, prediction files for `scripts/evaluate_predictions.py`, and script
versions of these notebooks are next.
