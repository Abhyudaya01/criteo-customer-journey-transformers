# Conversion PR-AUC with 95% user-bootstrap CI (chrono_test)

| Model | all (base rate 0.188%) | active_5plus (base rate 0.468%) | recent_click (base rate 0.804%) | next_new_campaign (base rate 1.421%) |
|---|---|---|---|---|
| gbdt | 0.0227 [0.0219, 0.0243] | 0.0355 [0.0339, 0.0387] | 0.0451 [0.0410, 0.0480] | 0.0289 [0.0251, 0.0358] |
| itransformer | 0.0291 [0.0280, 0.0310] | 0.0442 [0.0412, 0.0463] | 0.0549 [0.0503, 0.0592] | 0.0301 [0.0253, 0.0358] |
| itransformer_w6h | 0.0286 [0.0274, 0.0306] | 0.0432 [0.0398, 0.0449] | 0.0542 [0.0494, 0.0586] | 0.0305 [0.0254, 0.0362] |
| logistic | 0.0182 [0.0172, 0.0196] | 0.0292 [0.0268, 0.0306] | 0.0350 [0.0308, 0.0377] | 0.0248 [0.0215, 0.0273] |
| lstm | 0.0267 [0.0259, 0.0293] | 0.0427 [0.0395, 0.0451] | 0.0513 [0.0471, 0.0567] | 0.0299 [0.0251, 0.0350] |
| lu_kannan | 0.0254 [0.0244, 0.0273] | 0.0382 [0.0353, 0.0399] | 0.0490 [0.0445, 0.0529] | 0.0289 [0.0248, 0.0334] |
| recency | 0.0069 [0.0067, 0.0077] | 0.0104 [0.0099, 0.0110] | 0.0151 [0.0141, 0.0167] | 0.0181 [0.0166, 0.0209] |

Paired difference vs itransformer (all positions):

| Model | Metric | Difference [95% CI] | Share of replicates not better |
|---|---|---|---|
| gbdt | pr_auc | -0.0064 [-0.0071, -0.0057] | 1.000 |
| gbdt | pr_auc_lift | -3.4077 [-3.7926, -3.0170] | 1.000 |
| gbdt | roc_auc | -0.0203 [-0.0214, -0.0195] | 1.000 |
| gbdt | top_decile_gain | -0.0592 [-0.0630, -0.0557] | 1.000 |
| itransformer_w6h | pr_auc | -0.0005 [-0.0011, 0.0001] | 0.962 |
| itransformer_w6h | pr_auc_lift | -0.2700 [-0.6019, 0.0271] | 0.962 |
| itransformer_w6h | roc_auc | 0.0005 [0.0002, 0.0008] | 0.000 |
| itransformer_w6h | top_decile_gain | 0.0006 [-0.0013, 0.0027] | 0.211 |
| logistic | pr_auc | -0.0109 [-0.0121, -0.0103] | 1.000 |
| logistic | pr_auc_lift | -5.8096 [-6.4578, -5.4988] | 1.000 |
| logistic | roc_auc | -0.0477 [-0.0491, -0.0464] | 1.000 |
| logistic | top_decile_gain | -0.1140 [-0.1186, -0.1094] | 1.000 |
| lstm | pr_auc | -0.0023 [-0.0025, -0.0012] | 1.000 |
| lstm | pr_auc_lift | -1.2401 [-1.3438, -0.6578] | 1.000 |
| lstm | roc_auc | -0.0146 [-0.0155, -0.0139] | 1.000 |
| lstm | top_decile_gain | -0.0390 [-0.0423, -0.0359] | 1.000 |
| lu_kannan | pr_auc | -0.0037 [-0.0044, -0.0031] | 1.000 |
| lu_kannan | pr_auc_lift | -1.9472 [-2.3383, -1.6559] | 1.000 |
| lu_kannan | roc_auc | 0.0003 [-0.0005, 0.0009] | 0.303 |
| lu_kannan | top_decile_gain | -0.0017 [-0.0045, 0.0015] | 0.854 |
| recency | pr_auc | -0.0222 [-0.0238, -0.0212] | 1.000 |
| recency | pr_auc_lift | -11.8162 [-12.7004, -11.2886] | 1.000 |
| recency | roc_auc | -0.1190 [-0.1213, -0.1168] | 1.000 |
| recency | top_decile_gain | -0.2315 [-0.2377, -0.2261] | 1.000 |

PR-AUC divided by the subgroup's base rate (1.0x = random ranking), 95% CI. PR-AUC rises with prevalence, so compare subgroups with this table, not the one above.

| Model | all | active_5plus | recent_click | next_new_campaign |
|---|---|---|---|---|
| gbdt | 12.1x [11.7, 13.0] | 7.6x [7.2, 8.2] | 5.6x [5.1, 5.9] | 2.0x [1.8, 2.5] |
| itransformer | 15.5x [14.9, 16.5] | 9.4x [8.8, 9.8] | 6.8x [6.3, 7.3] | 2.1x [1.8, 2.5] |
| itransformer_w6h | 15.2x [14.6, 16.3] | 9.2x [8.5, 9.6] | 6.7x [6.2, 7.2] | 2.1x [1.8, 2.6] |
| logistic | 9.7x [9.2, 10.4] | 6.2x [5.7, 6.5] | 4.4x [3.8, 4.6] | 1.7x [1.6, 1.9] |
| lstm | 14.3x [13.8, 15.6] | 9.1x [8.5, 9.6] | 6.4x [5.9, 7.0] | 2.1x [1.8, 2.5] |
| lu_kannan | 13.5x [13.0, 14.5] | 8.2x [7.6, 8.5] | 6.1x [5.6, 6.5] | 2.0x [1.8, 2.4] |
| recency | 3.7x [3.6, 4.1] | 2.2x [2.1, 2.3] | 1.9x [1.8, 2.1] | 1.3x [1.2, 1.5] |

next_new_campaign uses the label window, so it is a descriptive slice, not a segment selectable in advance.
