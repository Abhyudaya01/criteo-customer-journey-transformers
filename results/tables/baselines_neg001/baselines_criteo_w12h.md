# Baselines: criteo_w12h (12h windows)

Shards used: 8 | training negatives kept: 1% | seed 42

## chronological (test split)

Test positions: 20,121,228; base rate 0.1876%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0069 | 0.7448 | 3.58 | 0.358 | – | – | – |
| logistic | 0.0182 | 0.8161 | 4.76 | 0.476 | 0.0124 | 0.0008 | 0.0624 |
| gbdt | 0.0224 | 0.8434 | 5.29 | 0.529 | 0.0118 | 0.0003 | 0.0720 |

## user_disjoint (test split)

Test positions: 18,878,625; base rate 0.1830%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0064 | 0.7319 | 3.51 | 0.351 | – | – | – |
| logistic | 0.0193 | 0.8191 | 4.77 | 0.477 | 0.0119 | 0.0003 | 0.0646 |
| gbdt | 0.0245 | 0.8464 | 5.40 | 0.540 | 0.0115 | 0.0000 | 0.0751 |
