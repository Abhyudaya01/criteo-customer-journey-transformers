"""DARNN model and training script. Needs torch (runs on Zaratan; skipped where torch is missing)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from src.data.events import Anchors, EventStore, darnn_item, make_darnn_dataset  # noqa: E402
from src.data.preprocess import build_windowed_dataset, write_table  # noqa: E402
from src.data.splits import assign_chronological_split, user_groups, window_boundaries  # noqa: E402
from src.models.darnn import DARNN, darnn_loss, left_to_right_padded  # noqa: E402
from tests.synthetic import make_synthetic_criteo  # noqa: E402
from tests.test_events import _encode_like_stajt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    raw = make_synthetic_criteo(n_users=400, seed=21)
    events = _encode_like_stajt(raw)
    ds = build_windowed_dataset(raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True),
                                hours=12, max_inactive_hours=168)
    pos = ds["positions"]
    pos["time_split"] = assign_chronological_split(pos, window_boundaries(ds["meta"]["n_windows"])).astype(str)
    pos["user_group"] = pos["uid"].map(user_groups(pos["uid"])).astype(str)
    sizes = [int(events[f"cat{i}_id"].max()) + 1 for i in range(1, 10)]
    return {"raw": raw, "events": events, "ds": ds, "pos": pos,
            "n_entities": int(events["entity_id"].max()) + 1, "sizes": sizes}


def _batch(world, n=48, max_len=64, seed=0):
    store = EventStore(world["events"])
    sample = world["pos"].sample(n=n, random_state=seed).reset_index(drop=True)
    a = Anchors.from_frame(sample)
    items = [make_darnn_dataset(store, a, max_len)[i] for i in range(len(a))]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def _model(world, **kw):
    torch.manual_seed(0)
    return DARNN(world["n_entities"], world["sizes"], d_model=32, **kw).eval()


def test_left_to_right_padding():
    x = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4], [0, 0, 0, 9]])
    n = torch.tensor([2, 4, 1])
    assert left_to_right_padded(x, n).tolist() == [[5, 6, 0, 0], [1, 2, 3, 4], [9, 0, 0, 0]]


def test_forward_shapes_and_finite_loss(world):
    b = _batch(world)
    out = _model(world)(b)
    assert out["conv_logit"].shape == (48,)
    assert out["click_logit"].shape == (48, 64) and out["click_mask"].shape == (48, 64)
    assert (out["click_mask"].sum(1) == b["n_events"]).all()
    loss, parts = darnn_loss(out, b["y"])
    assert torch.isfinite(loss) and parts["click"] > 0
    # Attention weights sum to 1 over real positions.
    assert torch.allclose(out["att_impression"].sum(1), torch.ones(48), atol=1e-5)


def test_padding_length_does_not_change_predictions(world):
    """The same history, left-padded to 64 or to 96 steps, gives the same output."""
    m = _model(world)
    with torch.no_grad():
        a = m(_batch(world, max_len=64))["conv_logit"]
        b = m(_batch(world, max_len=96))["conv_logit"]
    assert torch.allclose(a, b, atol=1e-5)


def _flip_last_click(b):
    b2 = {k: v.clone() for k, v in b.items()}
    b2["click"][:, -1] = 1 - b2["click"][:, -1]  # left-padded: last column = most recent event
    return b2


def test_last_click_reaches_conversion_only_with_last_click_step(world):
    b = _batch(world)
    with torch.no_grad():
        ours = _model(world, last_click_step=True)
        paper = _model(world, last_click_step=False)
        d_ours = (ours(b)["conv_logit"] - ours(_flip_last_click(b))["conv_logit"]).abs()
        d_paper = (paper(b)["conv_logit"] - paper(_flip_last_click(b))["conv_logit"]).abs()
    assert (d_ours > 1e-6).all()      # our adaptation uses the most recent click
    assert (d_paper < 1e-6).all()     # the paper's decoder never consumes it


def test_click_prediction_never_sees_its_own_click(world):
    """Click logit for event j may depend on clicks before j, never on z_j itself or later."""
    b = _batch(world)
    m = _model(world)
    with torch.no_grad():
        base = m(b)["click_logit"]
        flipped = m(_flip_last_click(b))["click_logit"]
    n = b["n_events"]
    last = n - 1  # right-padded index of the most recent event
    rows = torch.arange(len(n))
    assert torch.allclose(base[rows, last], flipped[rows, last], atol=1e-6)


def test_tiny_overfit(world):
    b = _batch(world, n=32, seed=3)
    b["y"] = (torch.arange(32) % 2).float()
    m = DARNN(world["n_entities"], world["sizes"], d_model=32, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        loss, _ = darnn_loss(m(b), b["y"])
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * first


def test_training_script_end_to_end_on_cpu(world, tmp_path):
    ds, pos, events = world["ds"], world["pos"], world["events"]
    data = tmp_path / "processed"
    for t in ("user_windows", "window_campaigns", "conversions", "next_campaign_labels"):
        write_table(ds[t], data / t / "part-000.parquet")
    write_table(pos, data / "positions" / "part-000.parquet")
    (data / "meta.json").write_text(json.dumps({**ds["meta"], "window_hours": 12}))
    cache = tmp_path / "stajt_full_eval_cache"
    (cache / "events").mkdir(parents=True)
    events.to_parquet(cache / "events" / "part-000.parquet")
    (cache / "vocab_meta.json").write_text(json.dumps({
        "n_entities": world["n_entities"],
        "context_vocab_sizes": {f"cat{i}": s for i, s in enumerate(world["sizes"], start=1)}}))
    anchors = tmp_path / "anchor_cache"
    anchors.mkdir()
    tr = pos[pos.time_split == "train"]
    tr = pd.concat([tr[tr.y_conversion == 1], tr[tr.y_conversion == 0].sample(frac=0.2, random_state=0)])
    tr[["uid", "window", "label_window", "y_conversion", "time_split"]].to_parquet(
        anchors / "train_anchors_neg_rate_0.0100.parquet")
    preds = tmp_path / "predictions"
    cmd = [sys.executable, str(REPO / "scripts/train_darnn.py"), "--cpu", "--epochs", "2",
           "--num-workers", "0", "--d-model", "32", "--batch-size", "64",
           "--data-dir", str(data), "--event-cache", str(cache), "--anchor-cache", str(anchors),
           "--out", str(tmp_path / "ckpt"), "--pred-dir", str(preds)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    p = pd.read_parquet(preds / "darnn__seed1__chrono_test.parquet")
    test = pos[pos.time_split == "test"]
    assert len(p) == len(test) and set(p.columns) == {"uid", "window", "score"}
    assert p.merge(test, on=["uid", "window"]).shape[0] == len(test)
    meta = json.loads((preds / "darnn__seed1__chrono_test.meta.json").read_text())
    assert meta["history"] == "events64" and meta["source"] == "scripts/train_darnn.py"
    hist = json.loads((tmp_path / "ckpt" / "history.json").read_text())
    assert len(hist) == 2 and all(0 <= h["lambda"] <= 1 for h in hist)
    # Resume continues from the last checkpoint instead of starting over.
    r2 = subprocess.run(cmd[:-2] + ["--no-save-predictions", "--epochs", "3",
                                    "--resume", str(tmp_path / "ckpt" / "darnn_epoch_02.pt")],
                        capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert "next epoch 3" in r2.stdout
    # Smoke runs refuse to write into the shared predictions folder.
    r3 = subprocess.run(cmd + ["--smoke"], capture_output=True, text=True, timeout=600)
    assert r3.returncode != 0 and "no-save-predictions" in (r3.stdout + r3.stderr)
