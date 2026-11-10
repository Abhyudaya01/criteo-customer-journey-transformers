# Window-length ablation: 6-hour vs. 12-hour tokens

Owner: Rachel. Final Stretch Plan Must item, due Nov 15.

## Design

The target is unchanged: will the user convert in the next 12 hours, on the same 12-hour test
positions as every other model. Only the input changes: window models read 6-hour tokens instead
of 12-hour tokens.

- **Same information, cut finer.** Each 12-hour position (user, window k) is placed at 6-hour
  window 2k+1, the last 6-hour window inside 12-hour window k. History through 6-hour window 2k+1
  covers exactly the same hours as history through 12-hour window k. `tests/test_multiscale.py`
  checks that cumulative impressions, clicks, conversions and cost match exactly.
- **PR-AUC stays comparable.** A 6-hour *target* would halve the base rate (0.18% to about 0.09%)
  and change the task, so PR-AUC could not be compared across rows.
- **STAJT needs no new run.** STAJT reads raw events up to the anchor, not window tokens, so
  window length does not change its input. Its existing 12-hour result is its row in the
  ablation table. This saves one full A100 run.
- **Paired with every other model.** The 6-hour model is scored on the same positions, so
  `scripts/evaluate_predictions.py` gives a paired CI for "6h tokens minus 12h tokens".

The question this answers for the report: do window models lose signal because their tokens are
too coarse? If 6-hour iTransformer closes the gap to STAJT, finer windows were enough. If not, the
gap comes from aggregating events into windows at all, which supports event-level modeling.

## Also fixed: split boundaries

`window_boundaries` rounds fractions of label windows, so 6-hour validation and test started
6 hours earlier than 12-hour (hour 522 vs. 528, and 630 vs. 636). `prepare_criteo.py` now cuts
comparison windows at the primary window's clock times (`splits.aligned_boundaries`). The
6-hour-input dataset below takes its splits from the 12-hour positions, so it is correct even
with the October 6-hour build.

## Steps on Zaratan

1. Build the dataset (CPU, a few minutes; needs both October folders from the same
   `prepare_criteo.py --all-windows` run):

   ```bash
   srun -A msml612-class -p debug -t 15 -c 2 --mem=16G \
       python scripts/build_finer_history.py \
       --coarse data/processed/criteo_w12h --fine data/processed/criteo_w6h \
       --out data/processed/criteo_w6h_in_w12h
   ```

   It stops with an error if position or positive counts differ from the 12-hour data.

2. Train iTransformer on `criteo_w6h_in_w12h`, 1 seed (ablations use 1 seed). Same model and
   hyperparameters as the 12-hour run; only the data folder changes. Sequences are 124 steps
   instead of 62. `load_sequences` and `build_position_features(..., window_hours=6)` read the
   folder unchanged. Smoke-test on a 1% user sample first, and use a MIG slice.

3. Save test predictions keyed by 12-hour windows:

   ```python
   from src.data.multiscale import to_coarse_keys
   preds["window"] = to_coarse_keys(preds["window"], ratio=2)
   preds[["uid", "window", "score"]].to_parquet(
       "results/predictions/itransformer_w6h__seed1__chrono_test.parquet")
   ```

   Repeat for `user_test`.

4. Score (CPU): `sbatch scripts/zaratan/evaluate_predictions.sbatch chrono_test`. Compare
   `itransformer_w6h` against `itransformer` seed 1, so both rows have one seed.

## Open dependency

The iTransformer training code lives in Abhyudaya's Colab notebooks, not in the repo. Step 2
needs it committed as a script (this is also the plan's "one run source" fix for the 20-point
code score).
