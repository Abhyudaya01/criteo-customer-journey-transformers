import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.bootstrap import SortedPreds, sorted_metrics, user_bootstrap
from src.evaluation.metrics import top_fraction_metrics


def _data(n=5000, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)
    s = y * 0.4 + rng.random(n)
    if ties:
        s = np.round(s, 1)  # many tied scores
    users = rng.integers(0, n // 8, n)
    return y, s, users


def test_unweighted_matches_sklearn_and_metrics_module():
    for ties in (False, True):
        y, s, _ = _data(ties=ties)
        m = sorted_metrics(SortedPreds.build(y, s))
        assert abs(m["pr_auc"] - average_precision_score(y, s)) < 1e-10
        assert abs(m["roc_auc"] - roc_auc_score(y, s)) < 1e-10
        assert abs(m["top_decile_gain"] - top_fraction_metrics(y, s)["top_decile_gain"]) < 1e-10


def test_weighted_matches_sklearn_sample_weight():
    for ties in (False, True):
        y, s, _ = _data(seed=1, ties=ties)
        w = np.random.default_rng(2).integers(0, 4, len(y)).astype(float)
        m = sorted_metrics(SortedPreds.build(y, s), w)
        assert abs(m["pr_auc"] - average_precision_score(y, s, sample_weight=w)) < 1e-10
        assert abs(m["roc_auc"] - roc_auc_score(y, s, sample_weight=w)) < 1e-10


def test_integer_weights_equal_row_duplication():
    y, s, _ = _data(n=800, seed=3)
    w = np.random.default_rng(4).integers(0, 3, len(y))
    m = sorted_metrics(SortedPreds.build(y, s), w.astype(float))
    yd, sd = np.repeat(y, w), np.repeat(s, w)
    assert abs(m["top_decile_gain"] - top_fraction_metrics(yd, sd)["top_decile_gain"]) < 0.02


def test_bootstrap_ci_covers_point_and_paired_diff_sign():
    y, s, users = _data(n=20000, seed=5)
    noisy = s + np.random.default_rng(6).normal(0, 0.6, len(s))
    ci, diffs = user_bootstrap(y, {"good": s, "noisy": noisy}, users, n_boot=60, seed=0,
                               reference="good")
    for _, r in ci.iterrows():
        assert r.ci_low <= r.point <= r.ci_high
    d = diffs[(diffs.model == "noisy") & (diffs.metric == "pr_auc")].iloc[0]
    assert d["diff"] < 0 and d.ci_high < 0 and d.p_not_better == 1.0


def test_identical_models_have_zero_paired_diff():
    y, s, users = _data(n=3000, seed=7)
    _, diffs = user_bootstrap(y, {"a": s, "b": s.copy()}, users, n_boot=20, reference="a")
    assert np.allclose(diffs[["diff", "ci_low", "ci_high"]].to_numpy(), 0)


def test_seed_lists_average_and_report_sd():
    y, s, users = _data(n=3000, seed=8)
    rng = np.random.default_rng(9)
    seeds = [s + rng.normal(0, 0.05, len(s)) for _ in range(3)]
    ci, _ = user_bootstrap(y, {"m": seeds}, users, n_boot=10)
    row = ci[ci.metric == "pr_auc"].iloc[0]
    expected = np.mean([average_precision_score(y, a) for a in seeds])
    assert row.n_seeds == 3 and abs(row.point - expected) < 1e-10 and row.seed_sd > 0


def test_negative_subsample_keeps_ci_close_and_parallel_matches_serial():
    y, s, users = _data(n=40000, seed=10)
    full, _ = user_bootstrap(y, {"m": s}, users, n_boot=80, seed=1)
    sub, _ = user_bootstrap(y, {"m": s}, users, n_boot=80, seed=1, neg_frac=0.3)
    a, b = full.set_index("metric"), sub.set_index("metric")
    assert (a["point"] == b["point"]).all()  # points always use every row
    width = a.loc["pr_auc", "ci_high"] - a.loc["pr_auc", "ci_low"]
    assert abs(b.loc["pr_auc", "ci_low"] - a.loc["pr_auc", "ci_low"]) < 0.25 * width
    par, _ = user_bootstrap(y, {"m": s, "n": s * 2}, users, n_boot=10, seed=2, n_jobs=2)
    ser, _ = user_bootstrap(y, {"m": s, "n": s * 2}, users, n_boot=10, seed=2, n_jobs=1)
    assert np.allclose(par[["ci_low", "ci_high"]], ser[["ci_low", "ci_high"]])


def test_pr_auc_lift_is_pr_auc_over_base_rate_and_one_for_random():
    y, s, users = _data(n=30000, seed=12)
    m = sorted_metrics(SortedPreds.build(y, s), metrics=("pr_auc", "pr_auc_lift", "base_rate"))
    assert abs(m["pr_auc_lift"] - m["pr_auc"] / y.mean()) < 1e-9
    rnd = np.random.default_rng(13).random(len(y))
    ci, _ = user_bootstrap(y, {"random": rnd}, users, n_boot=50, seed=0, metrics=("pr_auc_lift",))
    r = ci.iloc[0]
    # Random ranking: AP sits slightly above the base rate in small samples (~1.07x here with
    # ~900 positives; the excess shrinks as positives grow), far below any real model.
    assert abs(r.point - average_precision_score(y, rnd) / y.mean()) < 1e-9
    assert 0.9 < r.point < 1.2
    good, _ = user_bootstrap(y, {"m": s}, users, n_boot=20, seed=0, metrics=("pr_auc_lift",))
    assert good.iloc[0].ci_low > 3 * r.ci_high
