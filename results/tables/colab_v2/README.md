# Colab Tier 1 results (Abhyudaya)

Test-set results from `notebooks/colab/training/`, 12-hour Criteo windows, single seed.

- `colab_v2/`: with the history features from `src/data/features.py` (the results in the interim report, Table 3).
- `colab_v1/`: earlier run without those features (the v1 to v2 ablation). Its unseen-user protocol is the
  stricter `user_disjoint_future`, so only its chronological rows compare with v2.

Each `<model>/criteo_w12h/` folder holds `conversion_*.csv`, `campaign_*.csv`, `run_config.json` and the per-run
`raw/*.json` they were built from. All models: every positive plus 1% of negatives (probabilities corrected),
14-window lookback, F1 threshold chosen on validation. Campaign metrics use 200,000 test positions with a
non-empty next window. Model checkpoints are not committed.

Prediction files for `scripts/evaluate_predictions.py` (bootstrap CIs, subgroups) come with the seed runs.
