# Baselines: criteo_w12h_smoke (12h windows)

Shards used: 8 | training negatives kept: 2% | seed 42

## chronological (test split)

Test positions: 401,676; base rate 0.1775%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0069 | 0.7482 | 3.58 | 0.358 | – | – | – |
| logistic | 0.0196 | 0.8118 | 4.64 | 0.464 | 0.0117 | 0.0007 | 0.0684 |
| gbdt | 0.0198 | 0.8309 | 4.87 | 0.487 | 0.0115 | 0.0006 | 0.0844 |

| Campaign baseline | P@10 | R@10 | HitRate@1 | HitRate@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0200 | 0.1941 | 0.0140 | 0.1997 | 0.0924 | 0.0604 |
| repeat_last | 0.0822 | 0.7959 | 0.7698 | 0.8157 | 0.7825 | 0.7722 |
| user_history | 0.0919 | 0.8898 | 0.7627 | 0.8993 | 0.8372 | 0.8162 |

Campaign metrics use the 19,502 test positions whose next window has at least one campaign.

## user_disjoint (test split)

Test positions: 377,856; base rate 0.1823%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0075 | 0.7278 | 3.60 | 0.360 | – | – | – |
| logistic | 0.0196 | 0.8166 | 4.67 | 0.467 | 0.0118 | 0.0004 | 0.0678 |
| gbdt | 0.0203 | 0.8361 | 5.11 | 0.511 | 0.0116 | 0.0002 | 0.0712 |

| Campaign baseline | P@10 | R@10 | HitRate@1 | HitRate@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0204 | 0.1968 | 0.0328 | 0.2028 | 0.1028 | 0.0729 |
| repeat_last | 0.0812 | 0.7833 | 0.7595 | 0.8062 | 0.7723 | 0.7617 |
| user_history | 0.0893 | 0.8611 | 0.7643 | 0.8746 | 0.8212 | 0.8033 |

Campaign metrics use the 18,687 test positions whose next window has at least one campaign.
