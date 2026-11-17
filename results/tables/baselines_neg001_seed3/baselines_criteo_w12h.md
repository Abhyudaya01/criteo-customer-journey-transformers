# Baselines: criteo_w12h (12h windows)

Shards used: 8 | training negatives kept: 1% | seed 3

## chronological (test split)

Test positions: 20,121,228; base rate 0.1876%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0069 | 0.7448 | 3.58 | 0.358 | – | – | – |
| gbdt | 0.0235 | 0.8435 | 5.31 | 0.531 | 0.0118 | 0.0003 | 0.0730 |

## user_disjoint (test split)

Test positions: 18,878,625; base rate 0.1830%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0064 | 0.7319 | 3.51 | 0.351 | – | – | – |
| gbdt | 0.0251 | 0.8464 | 5.40 | 0.540 | 0.0115 | 0.0000 | 0.0751 |
