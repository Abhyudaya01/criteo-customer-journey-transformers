# Baselines: criteo_w12h (12h windows)

Shards used: 8 | training negatives kept: 1% | seed 2

## chronological (test split)

Test positions: 20,121,228; base rate 0.1876%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0069 | 0.7448 | 3.58 | 0.358 | – | – | – |
| gbdt | 0.0221 | 0.8433 | 5.31 | 0.531 | 0.0118 | 0.0003 | 0.0711 |

## user_disjoint (test split)

Test positions: 18,878,625; base rate 0.1830%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0064 | 0.7319 | 3.51 | 0.351 | – | – | – |
| gbdt | 0.0247 | 0.8465 | 5.41 | 0.541 | 0.0115 | 0.0000 | 0.0749 |
