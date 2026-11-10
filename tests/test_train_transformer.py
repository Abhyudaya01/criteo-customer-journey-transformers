"""Tests for the window-model data path and scripts/train_transformer.py (iTransformer).

* history batches use only windows <= t (future rows perturbed, batches unchanged);
* each seed draws its own negative sample, and the same seed draws the same one;
* the script runs end to end and writes one prediction per test position, in the format
  scripts/evaluate_predictions.py reads;
* on the 6h-input / 12h-target data the predictions are keyed by 12h windows and cover exactly
  the 12h test positions (window-length ablation, docs/window_length_ablation.md).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.multiscale import build_shard
from src.data.preprocess import build_windowed_dataset
from src.data.splits import assign_chronological_split, user_groups, window_boundaries
from tests.synthetic import make_synthetic_criteo

torch = pytest.importorskip("torch")
from src.training.window_data import WindowData  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TINY = ["--epochs", "1", "--d-model", "16", "--layers", "1", "--top-campaigns", "5", "--heads", "2",
        "--batch", "128", "--eval-batch", "1024", "--campaign-eval-max", "300", "--neg-rate", "0.5",
        "--device", "cpu"]


def _write(ds: Path, tables: dict, meta: dict) -> Path:
    for name, table in tables.items():
        if name in ("meta", "campaign_vocab"):
            continue
        (ds / name).mkdir(parents=True, exist_ok=True)
        table.to_csv(ds / name / "part-000.csv.gz", index=False)
    tables["campaign_vocab"].to_csv(ds / "campaign_vocab.csv.gz", index=False)
    (ds / "meta.json").write_text(json.dumps(meta))
    return ds


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("window_data")
    raw = make_synthetic_criteo(n_users=500, conv_prob=0.15, seed=3)
    coarse = build_windowed_dataset(raw, hours=12, max_inactive_hours=168)
    fine = build_windowed_dataset(raw, hours=6, max_inactive_hours=168, vocab=coarse["campaign_vocab"])
    b = window_boundaries(coarse["meta"]["n_windows"])
    pos = coarse["positions"]
    pos["time_split"] = assign_chronological_split(pos, b).astype(str)
    pos["user_group"] = pos["uid"].map(user_groups(pos["uid"])).astype(str)
    c_meta = dict(coarse["meta"], splits={"boundaries": b.as_dict()})
    d12 = _write(tmp / "criteo_w12h_test", coarse, c_meta)
    sh = build_shard(coarse, fine, 2)
    f_meta = dict(fine["meta"], target_hours=12, ratio=2, splits=c_meta["splits"])
    d6 = _write(tmp / "criteo_w6h_in_w12h_test", dict(sh, campaign_vocab=fine["campaign_vocab"]), f_meta)
    return {"d12": d12, "d6": d6, "coarse": coarse, "tmp": tmp}


def test_batches_use_only_past_windows(datasets, tmp_path):
    coarse = datasets["coarse"]
    cut = 30
    uw, wc = coarse["user_windows"].copy(), coarse["window_campaigns"].copy()
    fut = uw["window"] > cut
    uw.loc[fut, ["n_impressions", "n_clicks", "n_conversions"]] *= 7
    uw.loc[fut, "cost_sum"] += 3.0
    wc.loc[wc["window"] > cut, "n_impressions"] *= 5
    wc.loc[wc["window"] > cut, "campaign_idx"] = (wc.loc[wc["window"] > cut, "campaign_idx"] + 1) % \
        coarse["meta"]["n_campaigns"]
    pert = dict(coarse, user_windows=uw, window_campaigns=wc)
    meta = json.loads((datasets["d12"] / "meta.json").read_text())
    d_pert = _write(tmp_path / "pert", pert, meta)

    a, b = WindowData(datasets["d12"], log=lambda *_: None), WindowData(d_pert, log=lambda *_: None)
    idx = np.flatnonzero(a.pos_w <= cut)
    assert len(idx) > 100 and (a.pos_w == b.pos_w).all()
    ba, bb = a.make_batch(idx), b.make_batch(idx)
    for k in ("x", "camp", "wt", "off", "pad", "h_row", "h_w"):
        assert torch.equal(ba[k], bb[k]), k
    np.testing.assert_array_equal(a.H[idx], b.H[idx])          # past-only history features
    np.testing.assert_array_equal(a.seen_matrix(idx[:200]), b.seen_matrix(idx[:200]))


def test_each_seed_draws_its_own_negative_sample(datasets):
    quiet = {"log": lambda *_: None, "history_features": False}
    s1, s1b, s2 = (WindowData(datasets["d12"], neg_seed=s, neg_rate=0.3, **quiet) for s in (1, 1, 2))
    assert (s1.keep_neg == s1b.keep_neg).all()
    assert (s1.keep_neg != s2.keep_neg).any()
    assert abs(s1.keep_neg.mean() - 0.3) < 0.02


def _run(args):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "train_transformer.py"), *args],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    return out.stdout


def test_script_end_to_end_and_prediction_format(datasets, tmp_path):
    d12 = datasets["d12"]
    _run(["--data-dir", str(d12), "--seed", "1", "--out-dir", str(tmp_path / "out"),
          "--save-predictions", str(tmp_path / "pred"), *TINY])
    pos = datasets["coarse"]["positions"]
    for prot, split, col in (("chronological", "chrono_test", "time_split"),
                             ("user_disjoint", "user_test", "user_group")):
        r = json.loads((tmp_path / "out" / "raw" / f"{prot}_itransformer_top5_h2_seed1.json").read_text())
        assert 0 <= r["pr_auc"] <= 1 and 0 <= r["roc_auc"] <= 1 and "campaign" in r
        p = pd.read_parquet(tmp_path / "pred" / f"itransformer__seed1__{split}.parquet")
        test = pos.loc[pos[col] == "test", ["uid", "window"]]
        assert list(p.columns) == ["uid", "window", "score"]
        assert len(p) == len(test) == r["n_test"] and not p.duplicated(["uid", "window"]).any()
        assert set(map(tuple, p[["uid", "window"]].to_numpy())) == set(map(tuple, test.to_numpy()))
        assert p["score"].between(0, 1).all()
    assert (tmp_path / "out" / "conversion_itransformer.csv").exists()
    assert (tmp_path / "out" / "campaign_itransformer.csv").exists()

    # A second call skips finished runs.
    assert "already done" in _run(["--data-dir", str(d12), "--seed", "1", "--out-dir", str(tmp_path / "out"),
                                   "--save-predictions", str(tmp_path / "pred"), *TINY])

    # The shared scorer reads the files with no changes.
    ev = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate_predictions.py"),
                         "--data-dir", str(d12), "--split", "chrono_test", "--pred-dir", str(tmp_path / "pred"),
                         "--out-dir", str(tmp_path / "eval"), "--n-boot", "5", "--subgroup-n-boot", "5"],
                        capture_output=True, text=True, cwd=REPO)
    assert ev.returncode == 0, ev.stderr[-3000:]
    ci = pd.read_csv(tmp_path / "eval" / "conversion_ci_chrono_test.csv")
    assert "itransformer" in set(ci["model"])


def test_6h_input_predictions_use_12h_keys(datasets, tmp_path):
    _run(["--data-dir", str(datasets["d6"]), "--seed", "1", "--protocols", "chronological",
          "--out-dir", str(tmp_path / "out"), "--save-predictions", str(tmp_path / "pred"),
          "--skip-campaign-eval", *TINY])
    r = json.loads((tmp_path / "out" / "raw" / "chronological_itransformer_top5_L28_w6h_h2_seed1.json").read_text())
    assert r["lookback"] == 28 and r["window_hours"] == 6
    p = pd.read_parquet(tmp_path / "pred" / "itransformer_w6h__seed1__chrono_test.parquet")
    pos = datasets["coarse"]["positions"]
    test = pos.loc[pos["time_split"] == "test", ["uid", "window"]]
    assert len(p) == len(test)
    assert set(map(tuple, p[["uid", "window"]].to_numpy())) == set(map(tuple, test.to_numpy()))
