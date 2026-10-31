"""Shape and content tests for the journey sequence dataset."""

import importlib.util

import numpy as np
import pytest

from src.data.dataset import NUMERIC_FEATURES, JourneySequences
from src.data.preprocess import build_windowed_dataset
from src.data.splits import assign_chronological_split, user_groups, window_boundaries
from tests.synthetic import make_synthetic_criteo

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def tables():
    res = build_windowed_dataset(make_synthetic_criteo(n_users=500, conv_prob=0.2), 12, 48)
    pos = res["positions"]
    pos["time_split"] = assign_chronological_split(pos, window_boundaries(res["meta"]["n_windows"]))
    pos["user_group"] = pos["uid"].map(user_groups(pos["uid"]))
    return res


@pytest.fixture(scope="module")
def seqs(tables):
    return JourneySequences(tables, tables["meta"]["n_windows"], tables["meta"]["n_campaigns"],
                            "chronological", "train")


def test_item_shapes(seqs):
    it = seqs[0]
    T = seqs.T
    assert it["numeric"].shape == (T, len(NUMERIC_FEATURES))
    assert it["y_conversion"].shape == (T,) and it["mask"].shape == (T,)
    assert len(it["camp_idx"]) == len(it["camp_window"]) == len(it["camp_weight"])
    assert len(it["next_window"]) == len(it["next_campaign"])


def test_mask_and_labels_match_positions(tables, seqs):
    pos = tables["positions"]
    pos = pos[pos["time_split"] == "train"]
    assert sum(seqs[i]["mask"].sum() for i in range(len(seqs))) == len(pos)
    assert sum(seqs[i]["y_conversion"][seqs[i]["mask"]].sum() for i in range(len(seqs))) \
        == pos["y_conversion"].sum()


def test_inputs_match_user_windows(tables, seqs):
    uw = tables["user_windows"]
    for i in range(0, len(seqs), max(1, len(seqs) // 25)):
        it = seqs[i]
        u = uw[(uw.uid == it["uid"]) & (uw.is_active == 1)]
        assert set(np.flatnonzero(it["numeric"][:, 5])) == set(u.window)
        np.testing.assert_allclose(it["numeric"][u.window, 0], np.log1p(u.n_impressions), rtol=1e-6)


def test_next_campaign_targets_are_next_window(tables, seqs):
    wc = tables["window_campaigns"]
    it = next(seqs[i] for i in range(len(seqs)) if len(seqs[i]["next_window"]))
    for t, c in zip(it["next_window"], it["next_campaign"]):
        assert it["mask"][t]
        assert ((wc.uid == it["uid"]) & (wc.window == t + 1) & (wc.campaign_idx == c)).any()


def test_user_disjoint_val_has_no_train_users(tables):
    n, C = tables["meta"]["n_windows"], tables["meta"]["n_campaigns"]
    tr = JourneySequences(tables, n, C, "user_disjoint", "train")
    va = JourneySequences(tables, n, C, "user_disjoint", "val")
    assert not set(tr.users) & set(va.users)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_collate_shapes(seqs):
    import torch
    from src.data.dataset import causal_mask, collate_journeys
    items = [seqs[i] for i in range(8)]
    b = collate_journeys(items, seqs.C)
    B, T = 8, seqs.T
    assert b["numeric"].shape == (B, T, len(NUMERIC_FEATURES))
    assert b["y_campaign"].shape == (B, T, seqs.C)
    assert b["bag_offsets"].shape == (B * T,)
    emb = torch.nn.EmbeddingBag(seqs.C, 4, mode="sum")
    out = emb(b["bag_indices"], b["bag_offsets"], per_sample_weights=b["bag_weights"])
    assert out.view(B, T, 4).shape == (B, T, 4)
    assert causal_mask(T)[0, 1] and not causal_mask(T)[1, 0]
