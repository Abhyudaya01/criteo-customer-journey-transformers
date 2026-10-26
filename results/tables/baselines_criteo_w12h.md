# Baselines: criteo_w12h (12h windows)

Shards used: 8 | training negatives kept: 2% | seed 42

## chronological (test split)

Test positions: 20,121,228; base rate 0.1876%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0069 | 0.7448 | 3.58 | 0.358 | – | – | – |
| logistic | 0.0191 | 0.8156 | 4.75 | 0.475 | 0.0123 | 0.0006 | 0.0641 |
| gbdt | 0.0233 | 0.8432 | 5.31 | 0.531 | 0.0118 | 0.0003 | 0.0731 |

| Campaign baseline | P@10 | R@10 | HitRate@1 | HitRate@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0192 | 0.1858 | 0.0161 | 0.1914 | 0.0892 | 0.0589 |
| repeat_last | 0.0815 | 0.7888 | 0.7650 | 0.8094 | 0.7769 | 0.7668 |
| user_history | 0.0915 | 0.8857 | 0.7592 | 0.8952 | 0.8333 | 0.8123 |

Campaign metrics use the 963,604 test positions whose next window has at least one campaign.

## user_disjoint (test split)

Test positions: 18,878,625; base rate 0.1830%

| Conversion model | PR-AUC | ROC-AUC | Top-decile lift | Top-decile gain | BCE | ECE | F1 (val thr) |
|---|---|---|---|---|---|---|---|
| recency_frequency | 0.0064 | 0.7319 | 3.51 | 0.351 | – | – | – |
| logistic | 0.0198 | 0.8187 | 4.76 | 0.476 | 0.0119 | 0.0003 | 0.0656 |
| gbdt | 0.0251 | 0.8465 | 5.41 | 0.541 | 0.0115 | 0.0000 | 0.0763 |

| Campaign baseline | P@10 | R@10 | HitRate@1 | HitRate@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0200 | 0.1925 | 0.0301 | 0.1991 | 0.0979 | 0.0682 |
| repeat_last | 0.0813 | 0.7832 | 0.7598 | 0.8061 | 0.7719 | 0.7612 |
| user_history | 0.0891 | 0.8592 | 0.7633 | 0.8727 | 0.8194 | 0.8016 |

Campaign metrics use the 940,089 test positions whose next window has at least one campaign.

Run: Zaratan job 23587289, standard partition, 4 CPUs, 20 min 53 s, peak memory 18.4 GB (25 Oct 2026).
