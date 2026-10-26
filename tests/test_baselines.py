"""Leakage and correctness tests for features, campaign baselines and the baseline script."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.features import FEATURE_COLUMNS, build_position_features
from src.data.preprocess import build_windowed_dataset, save_dataset
from src.models.baselines import (
    campaign_popularity, campaign_topk, correct_for_negative_sampling,
)
from tests.synthetic import make_synthetic_criteo

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return build_windowed_dataset(make_synthetic_criteo(n_users=800, conv_prob=0.15), 12, 48)


def test_features_use_only_past_windows(data):
    cutoff = 30
    pos = data["positions"]
    early = pos[pos["window"] <= cutoff]
    full = build_position_features(early, data["user_windows"], 12)
    uw_past = data["user_windows"][data["user_windows"]["window"] <= cutoff]
    past_only = build_position_features(early, uw_past, 12)
    pd.testing.assert_frame_equal(full[FEATURE_COLUMNS], past_only[FEATURE_COLUMNS])


def test_feature_values_match_bruteforce(data):
    pos, uw = data["positions"], data["user_windows"]
    f = build_position_features(pos, uw, 12)
    for _, row in f.sample(200, random_state=0).iterrows():
        h = uw[(uw.uid == row.uid) & (uw.window <= row.window)]
        assert row.cum_impressions == h.n_impressions.sum()
        assert row.cum_conversions == h.n_conversions.sum()
        cur = h[h.window == row.window]
        assert row.cur_impressions == (cur.n_impressions.sum() if len(cur) else 0)


def test_campaign_topk_uses_only_past(data):
    pos, wc = data["positions"], data["window_campaigns"]
    q = pos[(pos.n_next_campaigns > 0) & (pos.window <= 30)][["uid", "window"]]
    pop = campaign_popularity(wc[wc.window <= 20], int(wc.campaign_idx.max()) + 1)
    a = campaign_topk(q, wc, pop, 5)
    b = campaign_topk(q, wc[wc.window <= 30], pop, 5)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_repeat_last_puts_last_window_campaigns_first(data):
    pos, wc = data["positions"], data["window_campaigns"]
    q = pos[pos.is_active == 1].head(50)[["uid", "window"]]
    pop = campaign_popularity(wc, int(wc.campaign_idx.max()) + 1)
    top = campaign_topk(q, wc, pop, 10)["repeat_last"]
    for i, (u, t) in enumerate(q.itertuples(index=False)):
        current = set(wc[(wc.uid == u) & (wc.window == t)].campaign_idx)
        assert set(top[i, : len(current)]) == current
        assert len(set(top[i])) == 10  # no duplicates after popularity fill


def test_negative_sampling_correction():
    # p_sampled from odds * (1/r) maps back to the true p
    p_true = np.array([0.001, 0.01, 0.2])
    r = 0.02
    odds_s = p_true / (1 - p_true) / r
    p_s = odds_s / (1 + odds_s)
    np.testing.assert_allclose(correct_for_negative_sampling(p_s, r), p_true, rtol=1e-6)


def test_run_baselines_script_end_to_end(data):
    from src.data.splits import assign_chronological_split, user_groups, window_boundaries
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pos = data["positions"].copy()
        b = window_boundaries(data["meta"]["n_windows"])
        pos["time_split"] = assign_chronological_split(pos, b)
        pos["user_group"] = pos["uid"].map(user_groups(pos["uid"]))
        res = dict(data, positions=pos)
        ds = tmp / "criteo_w12h_test"
        for name, table in res.items():
            if name not in ("meta", "campaign_vocab"):
                (ds / name).mkdir(parents=True)
                table.to_csv(ds / name / "part-000.csv.gz", index=False)
        res["campaign_vocab"].to_csv(ds / "campaign_vocab.csv.gz", index=False)
        meta = dict(data["meta"], splits={"boundaries": b.as_dict()})
        (ds / "meta.json").write_text(json.dumps(meta))
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "run_baselines.py"),
                              "--data-dir", str(ds), "--neg-rate", "0.5", "--out-dir", str(tmp)],
                             capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr[-2000:]
        r = json.loads((tmp / "baselines_criteo_w12h_test.json").read_text())
        for p in ("chronological", "user_disjoint"):
            conv = r["protocols"][p]["conversion"]
            assert {"recency_frequency", "logistic", "gbdt"} <= set(conv)
            assert 0 <= conv["gbdt"]["test"]["pr_auc"] <= 1
            assert set(r["protocols"][p]["campaign"]) == {"popularity", "repeat_last", "user_history"}
        assert (tmp / "baselines_criteo_w12h_test.md").exists()
