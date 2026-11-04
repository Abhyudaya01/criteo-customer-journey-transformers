import pandas as pd

from src.data.preprocess import build_windowed_dataset
from src.evaluation.subgroups import (
    position_subgroup_flags, recent_click_windows, split_truth_by_novelty,
)
from tests.synthetic import make_synthetic_criteo


def _tiny():
    # One user; 12-hour windows. Impressions in windows 0, 2, 3, 4, 6 (click only in window 3).
    uw = pd.DataFrame({
        "uid": [1] * 5, "window": [0, 2, 3, 4, 6],
        "n_impressions": [1] * 5, "n_clicks": [0, 0, 1, 0, 0], "is_active": [1] * 5,
    })
    wc = pd.DataFrame({"uid": [1] * 5, "window": [0, 2, 3, 4, 6], "campaign_idx": [10, 10, 11, 10, 12]})
    pos = pd.DataFrame({"uid": [1] * 7, "window": list(range(7))})
    pos["label_window"] = pos["window"] + 1
    nx = wc.rename(columns={"window": "label_window"}).assign(window=lambda d: d["label_window"] - 1)
    return pos, uw, wc, nx


def test_flags_on_hand_built_journey():
    pos, uw, wc, nx = _tiny()
    f = position_subgroup_flags(pos, uw, wc, nx, window_hours=12).set_index("window")
    assert f["cum_active_windows"].tolist() == [1, 1, 2, 3, 4, 4, 5]
    assert f["active_5plus"].tolist() == [False] * 6 + [True]
    # Click in window 3 counts for t = 3 and t = 4 (last 24h = 2 windows), not t = 5.
    assert f["recent_click"].tolist() == [False, False, False, True, True, False, False]
    # Label windows 3 (campaign 11) and 6 (campaign 12) carry campaigns new to the user.
    assert f["next_new_campaign"].tolist() == [False, False, True, False, False, True, False]


def test_recent_click_windows():
    assert recent_click_windows(12) == 2 and recent_click_windows(6) == 4


def test_history_flags_ignore_future_rows():
    raw = make_synthetic_criteo(n_users=150, seed=1)
    ds = build_windowed_dataset(raw, hours=12, max_inactive_hours=168)
    base = position_subgroup_flags(ds["positions"], ds["user_windows"], ds["window_campaigns"],
                                   ds["next_campaign_labels"], 12)
    t_cut = 30
    uw = ds["user_windows"].copy()
    future = uw["window"] > t_cut
    uw.loc[future, "n_clicks"] += 5
    uw.loc[future, "is_active"] = 1
    pert = position_subgroup_flags(ds["positions"], uw, ds["window_campaigns"],
                                   ds["next_campaign_labels"], 12)
    past = (base["window"] <= t_cut).to_numpy()
    for c in ("cum_active_windows", "recent_click"):
        assert (base.loc[past, c].to_numpy() == pert.loc[past, c].to_numpy()).all()


def test_split_truth_by_novelty():
    _, _, wc, nx = _tiny()
    t = split_truth_by_novelty(nx, wc).set_index("window")
    assert t.loc[2, "new"] == [11] and t.loc[2, "seen"] == []
    assert t.loc[3, "new"] == [] and t.loc[3, "seen"] == [10]
