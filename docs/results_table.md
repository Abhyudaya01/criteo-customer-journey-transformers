# The shared results table

`results/tables/results.csv` is the team's one results table (Final Stretch Plan, Week 1). It
has one row per prediction file (model × seed × test split) with point metrics on every test
position and how the run was made. Slides and the final report take their numbers from it.

Nobody edits it by hand. `scripts/evaluate_predictions.py` adds or replaces rows each time it
scores. Rows are keyed by model, seed, split, config and history, so re-scoring never duplicates.
Seed-averaged CIs and subgroups stay in `conversion_ci_*` and `conversion_subgroups_*`.

## Columns

| Column | Meaning |
|---|---|
| `model`, `seed`, `split` | from the prediction file name `<model>__seed<k>__<split>.parquet` |
| `config` | training setup, e.g. `team_aligned_neg1pct`, `optimized_10neg`, `neg1pct` |
| `history` | input representation, e.g. `w12h`, `w6h_in_w12h`, `events64`, `w12h_features` |
| `pr_auc`, `pr_auc_lift`, `roc_auc`, `top_decile_gain` | point metrics on every test position |
| `base_rate`, `n_positions`, `n_positives` | test set the metrics were computed on |
| `source` | the script that trained the model (the plan's "one run source") |
| `run_where`, `hardware`, `gpu_hours` | e.g. `zaratan job 23998993`, `A100` / `MIG 3g.20gb` / `cpu`, GPU time |
| `git_commit`, `trained_on`, `scored_on`, `notes` | provenance |

`gpu_hours` is the plan's compute rule 6 and the report's compute-cost row.

## What each training script adds

Right after saving test predictions, write the sidecar next to the file:

```python
from src.evaluation.results_table import write_run_meta

pred_path = "results/predictions/itransformer__seed2__chrono_test.parquet"
preds[["uid", "window", "score"]].to_parquet(pred_path)
write_run_meta(pred_path, config="neg1pct", history="w12h",
               source="scripts/train_itransformer.py", gpu_hours=1.4,
               hardware="MIG 3g.20gb", notes="checkpoint epoch 7")
```

`run_where` (Zaratan job ID or Colab) and `git_commit` are filled in automatically.
`scripts/run_baselines.py --save-predictions` already does this.

A file saved without a sidecar is still scored, but its metadata shows `unknown` and the
scoring log names it. Tag existing files with `scripts/tag_predictions.py` (see its `--help`).

## Updating the table

```bash
# point metrics only, about 2 minutes on CPU per split
python scripts/evaluate_predictions.py --split chrono_test --table-only
# or the full scoring job (CIs + subgroups + table)
sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test gbdt
```
