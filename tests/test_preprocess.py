"""Window construction and label construction tests (checked against a brute-force reference)."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    LEAKY_COLUMNS, add_windows, build_windowed_dataset, validate_schema,
)
from tests.synthetic import make_synthetic_criteo

HOURS = 12
W = HOURS * 3600
MAX_INACTIVE_HOURS = 48  # 4 windows


@pytest.fixture(scope="module")
def raw():
    return make_synthetic_criteo()


@pytest.fixture(scope="module")
def result(raw):
    return build_windowed_dataset(raw, HOURS, MAX_INACTIVE_HOURS)


def test_schema_error_mentions_separator():
    with pytest.raises(ValueError, match="separator"):
        validate_schema(["timestamp,uid,campaign"])


def test_window_index_and_count(raw, result):
    ev = add_windows(raw, HOURS)
    assert (ev["window"] == raw["timestamp"] // W).all()
    assert result["meta"]["n_windows"] == 60  # 30 days of 12h windows
    assert result["meta"]["timestamp_unit"] == "seconds"


def test_window_campaigns_counts_match_raw(raw, result):
    wc = result["window_campaigns"]
    assert wc["n_impressions"].sum() == len(raw)
    assert wc["n_clicks"].sum() == raw["click"].sum()
    assert not wc.duplicated(["uid", "window", "campaign_idx"]).any()
    assert wc["campaign_idx"].max() == result["meta"]["n_campaigns"] - 1


def test_conversions_deduplicated(raw, result):
    conv = result["conversions"]
    n_unique = raw.loc[raw.conversion_id >= 0, ["uid", "conversion_id"]].drop_duplicates().shape[0]
    assert result["meta"]["n_conversions_total"] == n_unique
    assert not conv.duplicated(["uid", "conversion_id"]).any()
    assert (conv["window"] < result["meta"]["n_windows"]).all()


def _reference_positions(raw, n_windows, max_inactive_windows):
    """Slow, obviously-correct version of build_positions + labels."""
    active = {(u, w) for u, w in zip(raw.uid, raw.timestamp // W)}
    conv = raw[raw.conversion_id >= 0].drop_duplicates(["uid", "conversion_id"])
    conv_windows = {(u, ts // W) for u, ts in zip(conv.uid, conv.conversion_timestamp)}
    camps = {}
    for u, w, c in zip(raw.uid, raw.timestamp // W, raw.campaign):
        camps.setdefault((u, w), set()).add(c)
    rows = []
    for u in sorted(raw.uid.unique()):
        for t in range(n_windows - 1):
            if any((u, a) in active for a in range(t - max_inactive_windows, t + 1)):
                rows.append((u, t, int((u, t + 1) in conv_windows), len(camps.get((u, t + 1), ()))))
    return pd.DataFrame(rows, columns=["uid", "window", "y_conversion", "n_next_campaigns"])


def test_positions_and_labels_match_reference(raw, result):
    meta = result["meta"]
    ref = _reference_positions(raw, meta["n_windows"], meta["max_inactive_windows"])
    got = result["positions"][["uid", "window", "y_conversion", "n_next_campaigns"]]
    pd.testing.assert_frame_equal(got.reset_index(drop=True).astype("int64"),
                                  ref.astype("int64"), check_dtype=False)


def test_label_window_is_next_window_and_in_range(result):
    pos = result["positions"]
    assert (pos["label_window"] == pos["window"] + 1).all()
    assert pos["label_window"].max() <= result["meta"]["n_windows"] - 1


def test_next_campaign_labels_come_from_next_window(result):
    lab = result["next_campaign_labels"]
    wc = result["window_campaigns"].rename(columns={"window": "label_window"})
    merged = lab.merge(wc, on=["uid", "label_window", "campaign_idx"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_sample_selection_ignores_future(raw):
    """Adding a conversion in the future must not change which positions exist."""
    base = build_windowed_dataset(raw, HOURS, MAX_INACTIVE_HOURS)["positions"][["uid", "window"]]
    raw2 = raw.copy()
    rows = raw2.index[raw2.conversion_id < 0][:50]
    raw2.loc[rows, "conversion_id"] = 10_000 + np.arange(len(rows))
    raw2.loc[rows, "conversion_timestamp"] = raw2.loc[rows, "timestamp"] + 20 * 86400
    raw2.loc[rows, "conversion"] = 1
    other = build_windowed_dataset(raw2, HOURS, MAX_INACTIVE_HOURS)["positions"][["uid", "window"]]
    pd.testing.assert_frame_equal(base, other)


def test_feature_tables_do_not_contain_leaky_columns(result):
    for name in ["window_campaigns", "user_windows"]:
        assert not set(result[name].columns) & set(LEAKY_COLUMNS), name


def test_user_window_conversion_feature_uses_conversion_time(result):
    uw = result["user_windows"]
    conv = result["conversions"]
    assert uw["n_conversions"].sum() == len(conv)


def test_six_hour_windows(raw):
    six = build_windowed_dataset(raw, 6, MAX_INACTIVE_HOURS)
    assert six["meta"]["n_windows"] == 120
    assert six["meta"]["max_inactive_windows"] == 8


def test_user_sharding_matches_full_build(raw):
    """Processing user shards with the global vocab/calendar gives identical positions."""
    from src.data.preprocess import build_campaign_vocab, n_windows_for
    full = build_windowed_dataset(raw, HOURS, MAX_INACTIVE_HOURS)
    vocab, n = build_campaign_vocab(raw), n_windows_for(raw, HOURS)
    parts = []
    for s in range(3):
        part = raw[raw.uid % 3 == s]
        parts.append(build_windowed_dataset(part, HOURS, MAX_INACTIVE_HOURS, vocab=vocab,
                                            n_windows=n, timestamp_unit="seconds")["positions"])
    sharded = pd.concat(parts).sort_values(["uid", "window"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(full["positions"], sharded)


def test_timestamp_check_partial_read_warns_not_fails(raw):
    import warnings
    from src.data.preprocess import infer_timestamp_unit
    first_day = raw[raw.timestamp < 86400]
    with pytest.raises(ValueError):
        infer_timestamp_unit(first_day)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert infer_timestamp_unit(first_day, strict=False) == "seconds"
        assert w
