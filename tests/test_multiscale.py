import numpy as np
import pandas as pd
import pytest

from src.data.dataset import JourneySequences
from src.data.features import build_position_features
from src.data.multiscale import build_shard, fine_anchor, to_coarse_keys, window_ratio
from src.data.preprocess import build_windowed_dataset
from src.data.splits import (
    aligned_boundaries, assign_chronological_split, user_groups, window_boundaries,
)
from tests.synthetic import make_synthetic_criteo


def test_aligned_boundaries_cut_at_same_hours():
    b12 = window_boundaries(62)
    b6 = aligned_boundaries(b12, 12, 6, 124)
    assert (b6.val_start * 6, b6.test_start * 6) == (b12.val_start * 12, b12.test_start * 12)
    # Unaligned rounding would cut 6 hours earlier: the bug this guards against.
    assert window_boundaries(124).val_start * 6 != b12.val_start * 12
    with pytest.raises(ValueError):
        aligned_boundaries(b12, 12, 5, 149)


def test_anchor_mapping_round_trip():
    k = np.arange(61)
    assert (to_coarse_keys(fine_anchor(k, 2), 2) == k).all()
    assert window_ratio(12, 6) == 2
    with pytest.raises(ValueError):
        to_coarse_keys(np.array([2]), 2)


@pytest.fixture(scope="module")
def built():
    raw = make_synthetic_criteo(n_users=300, seed=4)
    coarse = build_windowed_dataset(raw, hours=12, max_inactive_hours=168)
    fine = build_windowed_dataset(raw, hours=6, max_inactive_hours=168,
                                  vocab=coarse["campaign_vocab"])
    pos = coarse["positions"]
    b = window_boundaries(coarse["meta"]["n_windows"])
    pos["time_split"] = assign_chronological_split(pos, b).astype(str)
    pos["user_group"] = pos["uid"].map(user_groups(pos["uid"])).astype(str)
    return coarse, fine, build_shard(coarse, fine, 2)


def test_same_positions_labels_and_splits(built):
    coarse, _, out = built
    c, f = coarse["positions"], out["positions"]
    assert len(c) == len(f)
    assert (f["window"] % 2 == 1).all()
    assert (f["coarse_window"].to_numpy() == c["window"].to_numpy()).all()
    for col in ("y_conversion", "time_split", "user_group"):
        assert (f[col].to_numpy() == c[col].to_numpy()).all()


def test_fine_history_holds_exactly_the_coarse_information(built):
    """Cumulative history at 6h anchor 2k+1 == cumulative history at 12h window k: same hours."""
    coarse, fine, out = built
    fc = build_position_features(coarse["positions"], coarse["user_windows"], 12)
    ff = build_position_features(out["positions"], fine["user_windows"], 6)
    for col in ("cum_impressions", "cum_clicks", "cum_conversions", "cum_cost", "day_index"):
        assert np.allclose(fc[col].to_numpy(), ff[col].to_numpy()), col
    # Finer recency: an impression in the last 6h implies one in the last 12h, not vice versa.
    assert (ff["is_active"].to_numpy() <= fc["is_active"].to_numpy()).all()


def test_sequence_loader_reads_it_unchanged(built):
    coarse, fine, out = built
    seq = JourneySequences(out, n_windows=fine["meta"]["n_windows"],
                           n_campaigns=len(coarse["campaign_vocab"]), split="test")
    cpos = coarse["positions"]
    for i in range(min(len(seq), 25)):
        item = seq[i]
        w = np.flatnonzero(item["mask"])
        assert (w % 2 == 1).all()
        cp = cpos[(cpos.uid == item["uid"]) & (cpos.time_split == "test")].sort_values("window")
        assert (to_coarse_keys(w, 2) == cp["window"].to_numpy()).all()
        assert (item["y_conversion"][w] == cp["y_conversion"].to_numpy()).all()
        # Next-window campaign targets sit on the same fine anchors as the conversion labels.
        assert set(item["next_window"]) <= set(w)


def test_mixed_integer_widths_from_disk(built):
    """Tables reloaded from CSV come back int64 while built tables are int32 (real crash)."""
    coarse, fine, out = built
    fine64 = {**fine, "user_windows": fine["user_windows"].astype({"uid": "int64", "window": "int64"})}
    again = build_shard(coarse, fine64, 2)
    assert (again["positions"]["windows_since_active"].to_numpy()
            == out["positions"]["windows_since_active"].to_numpy()).all()
