# Design notes: data, scope, architecture, metrics

Working notes for the interim report (due 10/29). Written in response to the TA's proposal
feedback: *strong data/evaluation foundation, but too many projects combined*. Everything
below is a proposal for the group to agree on, not a final decision.

---

## 1. Data pipeline (implemented on `feature/criteo-preprocessing`)

### Source file
`criteo_attribution_dataset.tsv.gz` from the official Criteo release: **tab separated,
gzip, with a header row**, one row per impression, 22 columns:

| Group | Columns | How we use them |
|---|---|---|
| Keys | `timestamp`, `uid`, `campaign` | windowing, sequences, campaign tokens |
| Impression features | `click`, `cost`, `time_since_last_click`, `cat1`–`cat9` | inputs (we currently use `click`, `cost`) |
| **Future conversion** | `conversion`, `conversion_timestamp`, `conversion_id`, `attribution`, `click_pos`, `click_nb`, `cpo` | **labels only, never inputs** |

`configs/base.yaml` column names match the documented schema. `timestamp` starts at 0;
its unit is not documented, so `prepare_criteo.py --check-only` verifies a ~30-day span in
seconds before anything is windowed.

**Biggest leakage trap.** Every impression row carries the conversion that follows it
(up to 30 days later). Using `conversion` or `click_nb` from window *t* as a feature would
hand the model the answer for *t+1*. The pipeline builds conversion features and labels
only from `conversion_timestamp`, deduplicated on `(uid, conversion_id)`, and a test checks
the feature tables contain none of the leaky columns.

### Tables produced (`data/processed/criteo_w12h/`, same for 6h)

| Table | Grain | Purpose |
|---|---|---|
| `campaign_vocab` | campaign | raw id → `campaign_idx` 0..C-1 |
| `window_campaigns` | uid × window × campaign | token contents (impressions, clicks) |
| `user_windows` | uid × window | numeric token features: impressions, clicks, #campaigns, cost, conversions in window |
| `conversions` | distinct conversion | deduplicated conversion events |
| `positions` | uid × window *t* | prediction points; `y_conversion` for *t+1*, `n_next_campaigns`, `time_split`, `user_group` |
| `next_campaign_labels` | uid × *t* × campaign | sparse multi-label campaign targets for *t+1* |

**Which (user, t) pairs are prediction positions?** A user is "at risk" at window *t* if
they had an impression in the last `max_inactive_hours` (default 168 h = 7 days). This
looks only backwards, so sample selection can never peek at labels (a test injects future
conversions and checks the position set does not change). Lu & Kannan also include
inactive periods in the sequence; the 7-day cap keeps the table from exploding with
millions of one-impression users. Check `frac_conversion_windows_covered` in `meta.json`
after the first real run and raise the cap if it drops too low.

### Splits (`src/data/splits.py`)
Every position is assigned by its **label window** (*t+1*), so no label ever crosses a
boundary. Inputs may look back into earlier periods (normal forecasting).

* **Chronological**: label windows 1–41 train, 42–50 val, 51–59 test (12 h; 70/15/15).
* **User-disjoint**: users hashed 70/15/15 into train/val/test. Hashing (not shuffling)
  keeps a user in the same group for 6 h and 12 h runs. Two variants are available:
  `user_disjoint` (all windows, like Lu & Kannan's holdout customers) and
  `user_disjoint_future` (unseen users *and* future windows; the strictest test).

### Known dataset caveats (worth a sentence in the report)
* An impression row holds at most one conversion, so a second conversion with no impression
  since the previous one cannot appear. Conversion labels are a lower bound.
* Conversions after the last impression window are dropped (no window to predict from).
* The final window may be partial; `cost`/`cpo` are transformed, not real prices.
* Positive rate will be very low (likely well under 1% of positions). PR-AUC is the
  primary metric for that reason.

---

## 2. Scope tiers (proposal for the group to agree on)

The TA's concern is delivery risk. Suggest committing publicly to Tier 1 only, and
treating everything else as conditional.

| Tier | Contents | Gate |
|---|---|---|
| **1 – must ship** (interim 10/29 → final) | Criteo 12 h; simple baselines (§4) + HMM + LSTM; Lu & Kannan heterogeneous Transformer; **one** comparative Transformer (§3); chronological + user-disjoint evaluation; core metrics; attention-head ablation (1–4) | — |
| **2 – should ship** | 6 h vs 12 h; customer-specific weighting ablation; 3 seeds with confidence intervals; attention visualizations; multi-step horizon | Tier 1 results exist by ~11/15 |
| **3 – stretch, cut first** | MoCo/TS2Vec contrastive pretraining; LoRA; second dataset; DeepCausalMMM on synthetic data | Tier 2 done by ~11/22 |

Recommendation: move DeepCausalMMM to "future work" now. It uses different data
(synthetic, aggregate), different metrics and a different question, so it is the most
separable piece and the clearest answer to "too many projects". LoRA only makes sense if
pretraining exists, so it lives or dies with the contrastive work.

---

## 3. PatchTST or iTransformer → **recommend iTransformer**

| | PatchTST | iTransformer |
|---|---|---|
| Token | a patch of time steps of **one** channel | the **whole lookback of one variable** (campaign, conversion, …) |
| Attention across | time patches | variables |
| Cross-campaign effects | none by design (channel independence) | exactly what it models |
| Fit to our sequences | weak: 60 windows (12 h) → a handful of patches, mostly zeros for sparse users | fine: lookback of 60 is an MLP input per variable |
| Interpretability | patch-to-patch maps | campaign × campaign attention maps |
| Overlap with Lu & Kannan | both attend over time | complementary (L&K: time; iTransformer: campaigns) |

PatchTST's patching pays off on long, dense series (hundreds of steps). Our per-user series
are short and sparse, and the research question is about cross-campaign relationships, which
channel-independent PatchTST cannot express. iTransformer gives a clean contrast to the
time-attention Lu & Kannan model.

Practical notes for iTransformer: 700 variable tokens per sample gives 700² attention, so
use the top ~100 campaigns by training-period impressions plus one "other" token, plus
tokens for conversions, clicks and cost. It is not causal over time, so each (user, t) is a
separate sample with a fixed lookback; subsample positions per epoch if training is slow.

---

## 4. Architecture proposal for the main model on Criteo

```
window t of user n
  ├─ campaigns in t  → EmbeddingBag(C, d, mode="sum", weights=log1p(impressions))
  └─ numeric: log1p(impressions, clicks, cost), conversions, is_active
        concat → Linear → x_t (d_model)
x_t + sinusoidal PE(global window index)            # shared calendar for all users
  → shared layers: causal masked self-attention     # one pass predicts every t → t+1
       with mixture heads: z̄_t = Σ_h ω_{n,h} z_{t,h},  ω_n = softmax(θ_n)
  → task encoders (as in L&K "shared embedding, separate encoders"):
       conversion branch → 1 logit        (y_conversion at t+1)
       campaign branch   → C logits       (next_campaign_labels at t+1)
loss = BCE(conv, pos_weight) + λ · BCE(campaigns), only on positions in the split
```

Choices worth deciding as a group:

* **Separate encoders per target.** Lu & Kannan use one encoder per target, which is
  infeasible for 700 campaigns. Use two task branches (conversion, campaign) on a shared
  trunk. This keeps their idea and stays cheap.
* **Customer head weights ω_n.** Lu & Kannan learn a free θ_n per user and give unseen
  users the population mean. On Criteo most users have very short histories, so per-user
  θ_n will be poorly identified. Suggested ablation (this *is* the "customer-specific
  weighting" ablation from the proposal):
  (a) uniform ω (standard Transformer), (b) free θ_n per user (L&K), (c) ω from a small
  network over the user's history so far (amortized; works for unseen users without
  falling back to the mean). Only (c) is new, and it is a small change.
* **Causal masking** makes one forward pass over a user's sequence produce predictions for
  every position, which matches Lu & Kannan's "predict X_{t+1} for every t" setup and is far
  cheaper than one sample per position.
* **Imbalance.** BCE with `pos_weight` around √(neg/pos), then temperature scaling on the
  validation set so the probabilities are calibrated (reported with ECE).
* **Next step in code:** a PyTorch `Dataset` that turns one user's rows from
  `window_campaigns` + `user_windows` into a padded `[T, …]` tensor plus a position mask
  and labels. That is the bridge between this pipeline and every model.

### Baselines to add (cheap, and they keep the comparison honest)
1. **Repeat-last**: campaigns of the user's last active window as next-window scores.
2. **Popularity**: training-period campaign frequency.
3. **Logistic regression / gradient boosting** on recency, frequency, clicks, time since last
   click and windows since active for conversion.
A Transformer that does not beat (3) on PR-AUC is a finding worth reporting.

---

## 5. Metrics

Already in the proposal: PR-AUC (primary), ROC-AUC, balanced accuracy, F1, BCE,
Precision@K / Recall@K, top-decile conversion rate, lift, gain.

Added to `src/evaluation/metrics.py`:

| Metric | Why |
|---|---|
| `best_f1_threshold` (fit on val, apply to test) | at <1% positives a fixed 0.5 threshold makes F1/balanced accuracy meaningless |
| `top_fraction_metrics` (top-decile CR, lift, gain), `gain_curve` | Lu & Kannan's targeting view; were listed but not implemented |
| `expected_calibration_error`, Brier | the outputs are probabilities a bidder would consume |
| `campaign_ranking_metrics`: HitRate@K, NDCG@K, MAP@K | rank-aware; computed only on positions whose next window is non-empty, with that share reported |
| `campaign_bce` | full multi-label BCE |

Reporting practices (no extra code needed):
* **Stratify** by history length (1 vs 2+ active windows) and by `windows_since_active`.
  Transformers should only help when there is a sequence to attend over.
* **"Any activity next window" AUC**: most next windows are empty; predicting whether the
  user shows up at all is a useful intermediate task.
* **Multi-step horizon** (Tier 2): AUC versus steps ahead, as in Lu & Kannan's holdout period.
* **Uncertainty**: bootstrap over users for 95% CIs; 3 seeds for the headline table.
* **Cost**: parameters, GPU memory, time per epoch on Zaratan (needed anyway if LoRA survives).

---

## 6. Candidate datasets

| Dataset | What it is | User journey across windows? | Campaign/channel analogue | Verdict |
|---|---|---|---|---|
| **Criteo Attribution** | 30 days, ~16.5M impressions, ~45K conversions, ~700 campaigns | yes (`uid`) | campaigns | **Primary** |
| Retailrocket (Kaggle) | ~4.5 months of e-commerce events (view, add-to-cart, transaction), hashed ids | yes (`visitorid`) | event types / item categories | Best secondary if Tier 3 happens |
| Taobao UserBehavior (Tianchi) | ~1M users, 25 Nov–3 Dec 2017; click, cart, favorite, purchase | yes, but only 9 days (18 windows at 12 h) | behavior types / categories | Large; short span; needs Tianchi login to download |
| YOOCHOOSE (RecSys 2015) | 6 months of click sessions, some with buys | **no persistent user id**: session-level only | none | Drop: cannot build multi-window journeys or a user-disjoint split |
| PySiMMMulator | synthetic aggregate channel spend/response | n/a | channels | MMM only (Tier 3 / future work) |

Recommendation: Criteo only for Tiers 1–2. If a second dataset is used, Retailrocket is the
closest fit (persistent visitor id, real conversions, a long enough span for 12 h windows).

---

## 7. Zaratan runbook

```bash
# once: put the repo on scratch (home quota is small), create the env
cd <your scratch dir>
git clone https://github.com/Abhyudaya01/criteo-customer-journey-transformers.git
cd criteo-customer-journey-transformers
git checkout feature/criteo-preprocessing
conda env create -f environment.yml           # load the cluster's conda module first if needed

# data (login node is fine for a single download)
bash scripts/download_criteo.sh
python scripts/prepare_criteo.py --check-only # header, separator, timestamp unit

# build 12 h + 6 h on a compute node (edit --account first)
mkdir -p logs && sbatch scripts/zaratan/prepare_criteo.sbatch
squeue -u $USER; tail -f logs/prepare_criteo-*.out
```

`environment.yml` has no PyTorch yet; add the CUDA build that matches Zaratan's GPU module
before model training.

After the first real run, paste `meta.json` (both window sizes) into the interim report:
number of users, positions, positive rate, share of positions with a next-window campaign,
and conversion coverage. Those numbers settle `max_inactive_hours` and the top-K campaign
cut-off.
