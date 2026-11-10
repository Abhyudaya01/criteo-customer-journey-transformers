# DARNN under our chronological protocol

Owner: Rachel. Final Stretch Plan Should item, due Nov 22. It starts only after Rachel's Must
items are done. The published-protocol reproduction is cut (future work).

DARNN (Ren et al., 2018) is the closest published model to STAJT: it attends over individual
impressions and, separately, over click behavior. The comparison asks whether STAJT's gains
come from event-level modeling in general (DARNN has that too) or from STAJT's own components.

## Shared with STAJT (identical, checked by tests/test_events.py)

| Part | Source |
|---|---|
| Events, campaign and cat1–cat9 vocabularies (training period only) | STAJT's full event cache, `STAJT/data/stajt_full_eval_cache/events/part-XXX.parquet` |
| Prediction points | 12h positions; history = events strictly before the label window |
| History length | last 64 events, left-padded |
| Training sample | STAJT's team-aligned anchors (all positives + 1% of negatives), `train_anchors_neg_rate_0.0100.parquet` |
| Validation and test | every position of the split; predictions saved as `darnn__seed1__chrono_test.parquet` and scored by `scripts/evaluate_predictions.py`, paired with STAJT |

`src/data/events.py` (`EventStore`, `darnn_item`, `split_anchors`, `load_stajt_train_anchors`,
`make_darnn_dataset`) reads these with no second pass over the raw file.

## Faithful to the paper

- Encoder: LSTM over impression features x_j (campaign, cat1–cat9, cost, time since previous
  event, time to prediction point, campaign seen before, time since campaign last seen).
- Decoder: LSTM initialized from the last encoder state, taking the previous click z_{j-1} and
  predicting click z_j; click loss L^c over real events.
- Impression-level and click-level attention, combined by the learned weight λ; conversion loss L^v.

The click flag is **not** an impression feature: it is the decoder's target. (STAJT feeds clicks
in as an input; the loader drops them for DARNN, and a test checks that flipping every click
changes only the click targets.)

## Adapted to our protocol

1. **Last click.** As published, the decoder never consumes the click on the most recent event
   z_m, because it only predicts it. Here every event is before the prediction point, so z_m
   is known, and recent clicks are among the strongest conversion signals in Criteo (interim
   report, Figure 3). The decoder runs one extra step that takes z_m as input. Without this,
   DARNN would be handicapped against every other model, which all see the last click.
2. **Target.** Conversion in the next 12-hour window, not "conversion at the end of the
   sequence", so it is the same task as every other model.
3. **No user ID feature.** The paper lists user ID in x_j. STAJT and the window models use none,
   and a 6M-row user embedding would not transfer to unseen users.
4. **Loss weight.** The paper defines L^c and L^v but not how they are combined. Use
   L = L^v + α · L^c with α = 1, and say so in the report.
5. **Hyperparameters.** The method section does not give hidden size, embedding size, or
   optimizer. Match STAJT's scale (d = 128, AdamW, lr 3e-4) so neither model gets a bigger
   budget, unless the paper's experiments section says otherwise.

## Code

- Model: `src/models/darnn.py` (`DARNN`, `darnn_loss`). `last_click_step=False` gives the paper's
  decoder, for an optional ablation (`--no-last-click-step`, saved as `darnn_paper_decoder`).
- Training + test scoring: `scripts/train_darnn.py`, Zaratan job `scripts/zaratan/train_darnn.sbatch`.
  Writes `results/predictions/darnn__seed1__chrono_test.parquet` plus its `.meta.json`, so
  `scripts/evaluate_predictions.py` scores it against STAJT and adds it to the results table.
- Tests: `tests/test_darnn.py` (needs torch; run `pytest -q` on Zaratan). They check that the
  most recent click reaches the conversion output only with `last_click_step`, that a click is
  never used to predict itself, that padding length doesn't change predictions, and a full CPU
  run of the training script on synthetic data, including resume.

## Compute

DARNN is two small LSTMs, so use a MIG slice (`gpu:a100_1g.5gb`, 7 SU/hr), not a full A100 (48 SU/hr). Smoke-test on a 1% user sample,
select checkpoints on the fixed validation subsample (all positives + 10% of negatives), score
the test set once, 1 seed.

## Reference

Ren, K., Fang, Y., Zhang, W., Liu, S., Li, J., Zhang, Y., Yu, Y., & Wang, J. (2018). Learning
multi-touch conversion attribution with dual-attention mechanisms for online advertising.
CIKM 2018, 1433–1442. https://arxiv.org/abs/1808.03737
