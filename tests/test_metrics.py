import numpy as np

from src.evaluation.metrics import (
    best_f1_threshold, campaign_ranking_metrics, conversion_metrics,
    expected_calibration_error, precision_recall_at_k, top_fraction_metrics,
)


def test_top_decile_lift_and_gain_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)
    m = top_fraction_metrics(y, p, 0.10)
    assert m["top_decile_conversion_rate"] == 1.0
    assert m["top_decile_lift"] == 10.0
    assert m["top_decile_gain"] == 1.0


def test_conversion_metrics_keys():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.05).astype(int)
    p = np.clip(y * 0.3 + rng.random(1000) * 0.2, 0, 1)
    m = conversion_metrics(y, p)
    for k in ["pr_auc", "roc_auc", "brier", "expected_calibration_error", "top_decile_lift"]:
        assert k in m
    assert m["roc_auc"] > 0.9


def test_ece_zero_when_calibrated_constant():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) < 1e-9


def test_best_f1_threshold():
    y = np.array([0, 0, 1, 1])
    thr, f1 = best_f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert thr == 0.8 and f1 == 1.0


def test_campaign_ranking_metrics():
    y = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
    s = np.array([[0.9, 0.1, 0.2, 0.8], [0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.9, 0.3]])
    m = campaign_ranking_metrics(y, s, ks=(1, 2))
    assert abs(m["frac_positions_nonempty"] - 2 / 3) < 1e-9
    assert m["hit_rate_at_1"] == 0.5
    assert m["ndcg_at_2"] < 1.0
    p, r = precision_recall_at_k(y[[0, 2]], s[[0, 2]], 2)
    assert p == 0.5 and abs(r - 2 / 3) < 1e-9
