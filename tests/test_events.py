"""The DARNN loader must give exactly STAJT's history, minus the click as an input."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.events import (
    CAT_ID_COLS, EventStore, Anchors, anchor_timestamps, darnn_item, split_anchors,
)
from src.data.preprocess import build_windowed_dataset
from src.data.splits import assign_chronological_split, window_boundaries
from tests.synthetic import make_synthetic_criteo

REPO = Path(__file__).resolve().parents[1]


def _torch_stub() -> types.ModuleType:
    """Just enough of torch to import STAJT's training script and run its numpy data path."""
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda a: a
    torch.tensor = lambda x, dtype=None: np.asarray(x)
    torch.no_grad = lambda: (lambda f: f)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float32 = np.float32
    torch.Tensor = type("Tensor", (), {})  # scipy/sklearn probe sys.modules["torch"].Tensor
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    functional = types.ModuleType("torch.nn.functional")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset, data.DataLoader = object, object
    torch.nn, nn.functional, torch.utils, utils.data = nn, functional, utils, data
    return {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional,
            "torch.utils": utils, "torch.utils.data": data}


@pytest.fixture(scope="module")
def stajt():
    try:
        import torch  # noqa: F401  (real torch on Zaratan)
        stubs = {}
    except ImportError:
        stubs = _torch_stub()
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        path = REPO / "STAJT/scripts/train_stajt_fair_chrono.py"
        spec = importlib.util.spec_from_file_location("stajt_train_for_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _encode_like_stajt(raw: pd.DataFrame) -> pd.DataFrame:
    """STAJT's event encoding (evaluate_stajt_full.build_full_event_cache), on synthetic rows."""
    ev = raw.sort_values(["uid", "timestamp"], kind="mergesort").reset_index(drop=True)
    camps = {v: i + 1 for i, v in enumerate(sorted(ev["campaign"].unique()))}
    ev["entity_id"] = ev["campaign"].map(camps).astype(np.int32)
    ev["raw_action_id"] = np.where(ev["click"] > 0, 2, 1).astype(np.int8)
    ev["behavior_family"] = ev["raw_action_id"]
    for i in range(1, 10):
        vals = {v: j + 1 for j, v in enumerate(sorted(ev[f"cat{i}"].unique()))}
        ev[f"cat{i}_id"] = ev[f"cat{i}"].map(vals).astype(np.int32)
    ev["log_cost"] = np.log1p(np.maximum(ev["cost"], 0) * 1000.0).astype(np.float32)
    prev = ev.groupby("uid", sort=False)["timestamp"].shift(1)
    ev["log_delta_hours"] = np.log1p(((ev["timestamp"] - prev) / 3600).fillna(0).clip(lower=0)).astype(np.float32)
    prev_c = ev.groupby(["uid", "entity_id"], sort=False)["timestamp"].shift(1)
    ev["entity_seen_before"] = prev_c.notna().astype(np.float32)
    ev["log_entity_gap_hours"] = np.log1p(((ev["timestamp"] - prev_c) / 3600).fillna(0).clip(lower=0)).astype(np.float32)
    return ev


@pytest.fixture(scope="module")
def data():
    raw = make_synthetic_criteo(n_users=250, seed=11)
    # Some users get long histories so the 64-event truncation is exercised.
    heavy = raw[raw["uid"] < 5].copy()
    extra = pd.concat([heavy.assign(timestamp=heavy["timestamp"] + k * 7) for k in range(1, 12)])
    raw = pd.concat([raw, extra], ignore_index=True).sort_values("timestamp", kind="mergesort")
    events = _encode_like_stajt(raw)
    ds = build_windowed_dataset(raw.reset_index(drop=True), hours=12, max_inactive_hours=168)
    pos = ds["positions"]
    pos["time_split"] = assign_chronological_split(pos, window_boundaries(ds["meta"]["n_windows"])).astype(str)
    return events, pos


def test_history_matches_stajt_loader_exactly(stajt, data):
    events, pos = data
    store_ours = EventStore(events)
    store_stajt = stajt.EventStore(events)
    anchors = pos.sample(n=600, random_state=0).reset_index(drop=True)
    anchors["anchor_timestamp"] = anchor_timestamps(anchors["label_window"])
    assert (anchors["anchor_timestamp"].to_numpy()
            == anchors["label_window"].to_numpy() * stajt.WINDOW_SECONDS).all()
    ds = stajt.JourneyDataset(anchors, store_stajt, stajt.MAX_LEN)
    a = Anchors.from_frame(anchors)
    long_hist = 0
    for i in range(len(a)):
        s = ds[i]
        d = darnn_item(store_ours, a.uid[i], a.anchor_ts[i], a.y[i])
        assert (d["campaign"] == s["entity_id"]).all()
        assert (d["context"] == s["context"]).all()
        assert np.allclose(d["numeric"], s["numeric"])
        assert (d["mask"] == s["mask"]).all()
        assert (d["click"] == (s["raw_action_id"] == 2)).all()
        assert float(d["y"]) == float(s["y_conversion"])
        long_hist += d["n_events"] == 64
    assert long_hist > 0  # truncation path covered


def test_every_position_has_history_and_none_from_the_label_window(data):
    events, pos = data
    store = EventStore(events)
    a = split_anchors(pos, "test")
    for i in range(0, len(a), 7):
        d = darnn_item(store, a.uid[i], a.anchor_ts[i], a.y[i])
        assert d["n_events"] >= 1  # positions require an impression in windows <= t
        s, e = store.ranges[int(a.uid[i])]
        used = store.timestamp[s:e][store.timestamp[s:e] < a.anchor_ts[i]][-64:]
        assert (used < a.anchor_ts[i]).all()


def test_future_events_do_not_change_inputs(data):
    events, pos = data
    a = split_anchors(pos, "val")
    cut = int(a.anchor_ts.min())
    later = events["timestamp"] >= cut
    perturbed = events.copy()
    perturbed.loc[later, "raw_action_id"] = 2
    perturbed.loc[later, "entity_id"] = 999
    perturbed.loc[later, CAT_ID_COLS] = 7
    s0, s1 = EventStore(events), EventStore(perturbed)
    for i in np.flatnonzero(a.anchor_ts == cut)[:50]:
        d0 = darnn_item(s0, a.uid[i], a.anchor_ts[i], a.y[i])
        d1 = darnn_item(s1, a.uid[i], a.anchor_ts[i], a.y[i])
        for k in ("campaign", "context", "numeric", "click", "mask"):
            assert np.array_equal(d0[k], d1[k]), k


def test_click_is_only_a_target_not_an_impression_feature(data):
    events, pos = data
    flipped = events.copy()
    flipped["raw_action_id"] = 3 - flipped["raw_action_id"]  # swap click / no click
    flipped["behavior_family"] = flipped["raw_action_id"]
    s0, s1 = EventStore(events), EventStore(flipped)
    a = split_anchors(pos, "train")
    for i in range(0, len(a), 13):
        d0 = darnn_item(s0, a.uid[i], a.anchor_ts[i], a.y[i])
        d1 = darnn_item(s1, a.uid[i], a.anchor_ts[i], a.y[i])
        for k in ("campaign", "context", "numeric"):
            assert np.array_equal(d0[k], d1[k]), k
        m = d0["mask"]
        assert (d0["click"][m] + d1["click"][m] == 1).all()


def test_anchor_timestamp_mismatch_is_rejected():
    df = pd.DataFrame({"uid": [1], "label_window": [5], "y_conversion": [0], "anchor_timestamp": [5 * 3600]})
    with pytest.raises(ValueError):
        Anchors.from_frame(df)
