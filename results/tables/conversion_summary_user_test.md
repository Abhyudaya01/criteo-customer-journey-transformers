# Conversion PR-AUC with 95% user-bootstrap CI (user_test)

| Model | all (base rate 0.183%) | active_5plus (base rate 0.543%) | recent_click (base rate 0.776%) | next_new_campaign (base rate 1.568%) |
|---|---|---|---|---|
| gbdt | 0.0248 [0.0225, 0.0275] | 0.0484 [0.0416, 0.0570] | 0.0470 [0.0420, 0.0573] | 0.0309 [0.0272, 0.0342] |
| itransformer | 0.0299 [0.0271, 0.0326] | 0.0559 [0.0481, 0.0649] | 0.0559 [0.0494, 0.0661] | 0.0319 [0.0286, 0.0368] |
| itransformer_w6h | 0.0296 [0.0270, 0.0320] | 0.0548 [0.0475, 0.0626] | 0.0548 [0.0493, 0.0642] | 0.0321 [0.0282, 0.0367] |
| logistic | 0.0193 [0.0173, 0.0218] | 0.0388 [0.0334, 0.0459] | 0.0366 [0.0308, 0.0443] | 0.0269 [0.0239, 0.0304] |
| recency | 0.0064 [0.0059, 0.0068] | 0.0116 [0.0105, 0.0128] | 0.0139 [0.0128, 0.0152] | 0.0193 [0.0176, 0.0202] |

Paired difference vs itransformer (all positions):

| Model | Metric | Difference [95% CI] | Share of replicates not better |
|---|---|---|---|
| gbdt | pr_auc | -0.0052 [-0.0058, -0.0041] | 1.000 |
| gbdt | pr_auc_lift | -2.8266 [-3.1844, -2.2752] | 1.000 |
| gbdt | roc_auc | -0.0190 [-0.0199, -0.0180] | 1.000 |
| gbdt | top_decile_gain | -0.0584 [-0.0620, -0.0546] | 1.000 |
| itransformer_w6h | pr_auc | -0.0003 [-0.0010, 0.0003] | 0.805 |
| itransformer_w6h | pr_auc_lift | -0.1887 [-0.5547, 0.1459] | 0.805 |
| itransformer_w6h | roc_auc | 0.0000 [-0.0002, 0.0003] | 0.320 |
| itransformer_w6h | top_decile_gain | -0.0008 [-0.0024, 0.0017] | 0.694 |
| logistic | pr_auc | -0.0106 [-0.0114, -0.0093] | 1.000 |
| logistic | pr_auc_lift | -5.8102 [-6.2419, -5.1154] | 1.000 |
| logistic | roc_auc | -0.0463 [-0.0476, -0.0449] | 1.000 |
| logistic | top_decile_gain | -0.1219 [-0.1261, -0.1164] | 1.000 |
| recency | pr_auc | -0.0236 [-0.0259, -0.0211] | 1.000 |
| recency | pr_auc_lift | -12.8867 [-14.1333, -11.5766] | 1.000 |
| recency | roc_auc | -0.1335 [-0.1360, -0.1310] | 1.000 |
| recency | top_decile_gain | -0.2480 [-0.2539, -0.2425] | 1.000 |

PR-AUC divided by the subgroup's base rate (1.0x = random ranking), 95% CI. PR-AUC rises with prevalence, so compare subgroups with this table, not the one above.

| Model | all | active_5plus | recent_click | next_new_campaign |
|---|---|---|---|---|
| gbdt | 13.5x [12.3, 14.9] | 8.9x [7.7, 10.3] | 6.1x [5.5, 7.3] | 2.0x [1.8, 2.2] |
| itransformer | 16.4x [14.9, 17.7] | 10.3x [8.9, 11.9] | 7.2x [6.4, 8.4] | 2.0x [1.8, 2.3] |
| itransformer_w6h | 16.2x [14.8, 17.4] | 10.1x [8.8, 11.4] | 7.1x [6.4, 8.2] | 2.0x [1.8, 2.3] |
| logistic | 10.5x [9.5, 11.8] | 7.1x [6.2, 8.3] | 4.7x [4.0, 5.6] | 1.7x [1.5, 1.9] |
| recency | 3.5x [3.3, 3.7] | 2.1x [2.0, 2.3] | 1.8x [1.7, 1.9] | 1.2x [1.1, 1.3] |

next_new_campaign uses the label window, so it is a descriptive slice, not a segment selectable in advance.
