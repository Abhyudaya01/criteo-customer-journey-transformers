# Saved test-set predictions

Every final model writes its test-set predictions here once, so bootstrap CIs and subgroup
results never need retraining (Final Stretch Plan, compute rule 2).

## File name

`<model>__seed<k>__<split>.parquet`, all lower case, double underscores:

- `model`: `stajt`, `stajt_optimized`, `itransformer`, `lu_kannan`, `patchtst`, `lstm`, `gbdt`, `logistic`, ...
- `k`: the training seed (1, 2, 3)
- `split`: `chrono_test` (positions with `time_split == "test"`) or `user_test` (`user_group == "test"`)

Example: `itransformer__seed2__chrono_test.parquet`

## Columns

| Column | Type | Meaning |
|---|---|---|
| `uid` | int64 | user id |
| `window` | int32 | anchor window t (or give `label_window` = t + 1 instead) |
| `score` | float | predicted probability of conversion in window t + 1 (sampling-corrected for 1%-negative training) |

One row per test position, every position in the split, no duplicates. The evaluation script
stops with an error if two files for the same split cover different positions.

Prediction files are large (20M rows each), so they are gitignored; keep them in the team's
shared scratch folder on Zaratan and symlink this folder to it.

## Scoring

```bash
sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test
sbatch scripts/zaratan/evaluate_predictions.sbatch user_test
```

Outputs land in `results/tables/` (`conversion_ci_*`, `conversion_diff_*`,
`conversion_subgroups_*`, `conversion_summary_*.md`).
