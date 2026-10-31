"""Split leakage and user-disjointness tests."""

import pandas as pd
import pytest

from src.data.preprocess import build_windowed_dataset
from src.data.splits import (
    chronological_split, make_splits, user_disjoint_chronological_split,
    user_disjoint_split, user_groups, window_boundaries,
)
from tests.synthetic import make_synthetic_criteo


@pytest.fixture(scope="module")
def data():
    res = build_windowed_dataset(make_synthetic_criteo(n_users=600), 12, 48)
    return res["positions"], res["meta"]["n_windows"]


def test_boundaries_cover_all_label_windows_once():
    b = window_boundaries(60, 0.7, 0.15)
    assert b.first_label_window == 1
    assert 1 < b.val_start < b.test_start < 60
    assert (b.val_start - 1) - 1 + 1 == 41  # 70% of the 59 label windows


def test_chronological_labels_never_cross_boundaries(data):
    pos, n = data
    train, val, test = chronological_split(pos, n)
    assert len(train) + len(val) + len(test) == len(pos)
    assert train["label_window"].max() < val["label_window"].min()
    assert val["label_window"].max() < test["label_window"].min()
    # every label window belongs to exactly one split
    sets = [set(x["label_window"]) for x in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[1] & sets[2]) and not (sets[0] & sets[2])


def test_chronological_split_from_anchor_window_only(data):
    pos, n = data
    a = chronological_split(pos.drop(columns="label_window"), n)
    b = chronological_split(pos, n)
    for x, y in zip(a, b):
        assert len(x) == len(y)


def test_user_disjoint_has_validation_and_no_overlap(data):
    pos, _ = data
    train, val, test = user_disjoint_split(pos, train_frac=0.7, val_frac=0.15)
    users = [set(x["uid"]) for x in (train, val, test)]
    assert all(users)
    assert not (users[0] & users[1]) and not (users[1] & users[2]) and not (users[0] & users[2])
    assert len(train) + len(val) + len(test) == len(pos)


def test_user_groups_are_stable_across_subsets():
    users = pd.Series(range(5000))
    full = user_groups(users)
    sub = user_groups(users.iloc[::7])
    assert (full.loc[sub.index] == sub).all()
    share = full.value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.03 and abs(share["val"] - 0.15) < 0.03


def test_user_disjoint_future_split(data):
    pos, n = data
    train, val, test = user_disjoint_chronological_split(pos, n)
    assert train["label_window"].max() < val["label_window"].min() <= val["label_window"].max() < test["label_window"].min()
    assert not (set(train.uid) & set(test.uid)) and not (set(val.uid) & set(test.uid))


def test_make_splits_from_config(data):
    pos, n = data
    out = make_splits(pos, n, {"train_frac": 0.7, "val_frac": 0.15, "user_disjoint_evaluation": True})
    assert set(out) == {"chronological", "user_disjoint", "user_disjoint_future"}
    assert all(len(part) == 3 for part in out.values())
