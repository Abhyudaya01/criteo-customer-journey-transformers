"""Tests for the window-model data path and scripts/train_transformer.py.

* history batches use only windows <= t (future rows perturbed, batches unchanged);
* each seed draws its own negative sample, and the same seed draws the same one;
* the script runs end to end for every model (iTransformer, Lu & Kannan, PatchTST, LSTM) and
  writes one prediction per test position, in the format scripts/evaluate_predictions.py reads;
* ablation settings (heads, customer weighting, patches) get their own prediction names, so they
  never overwrite the main runs;
* Lu & Kannan gives customers outside the training sample the mean head weights;
* on the 6h-input / 12h-target data the predictions are keyed by 12h windows and cover exactly
  the 12h test positions (window-length ablation, docs/window_length_ablation.md).

Equivalence with the Colab notebooks (same data, seeds 42 / model seed 0) was checked outside the
test suite, see the PR: identical metrics for iTransformer, Lu & Kannan and PatchTST; identical
LSTM training, with final probabilities equal to about 6 significant digits.
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
HAS_RESULTS_TABLE = (REPO / "src" / "evaluation" / "results_table.py").exists()   # sidecars need it
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


def _check_predictions(path: Path, test: pd.DataFrame, n_test: int):
    p = pd.read_parquet(path)
    assert list(p.columns) == ["uid", "window", "score"]
    assert len(p) == len(test) == n_test and not p.duplicated(["uid", "window"]).any()
    assert set(map(tuple, p[["uid", "window"]].to_numpy())) == set(map(tuple, test.to_numpy()))
    assert p["score"].between(0, 1).all()


def test_script_end_to_end_and_prediction_format(datasets, tmp_path):
    d12 = datasets["d12"]
    _run(["--data-dir", str(d12), "--seed", "1", "--out-dir", str(tmp_path / "out"),
          "--save-predictions", str(tmp_path / "pred"), *TINY])
    pos = datasets["coarse"]["positions"]
    for prot, split, col in (("chronological", "chrono_test", "time_split"),
                             ("user_disjoint", "user_test", "user_group")):
        r = json.loads((tmp_path / "out" / "raw" / f"{prot}_itransformer_top5_h2_seed1.json").read_text())
        assert 0 <= r["pr_auc"] <= 1 and 0 <= r["roc_auc"] <= 1 and "campaign" in r
        assert r["epoch_cap"] == 1 and r["best_epoch"] == 1
        test = pos.loc[pos[col] == "test", ["uid", "window"]]
        # 2 heads is a head-ablation setting (main = 4), so the name says so.
        _check_predictions(tmp_path / "pred" / f"itransformer_top5_h2__seed1__{split}.parquet", test, r["n_test"])
        side_path = tmp_path / "pred" / f"itransformer_top5_h2__seed1__{split}.meta.json"
        assert side_path.exists() == HAS_RESULTS_TABLE
        if HAS_RESULTS_TABLE:
            side = json.loads(side_path.read_text())
            assert side["config"] == "neg50pct" and side["history"] == "w12h_14lags"
            assert side["source"] == "scripts/train_transformer.py" and side["hardware"] == "cpu"
            assert side["gpu_hours"] == 0.0
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
    assert "itransformer_top5_h2" in set(ci["model"])


def test_6h_input_predictions_use_12h_keys(datasets, tmp_path):
    _run(["--data-dir", str(datasets["d6"]), "--seed", "1", "--protocols", "chronological",
          "--out-dir", str(tmp_path / "out"), "--save-predictions", str(tmp_path / "pred"),
          "--skip-campaign-eval", *TINY])
    r = json.loads((tmp_path / "out" / "raw" / "chronological_itransformer_top5_L28_w6h_h2_seed1.json").read_text())
    assert r["lookback"] == 28 and r["window_hours"] == 6
    pos = datasets["coarse"]["positions"]
    test = pos.loc[pos["time_split"] == "test", ["uid", "window"]]
    _check_predictions(tmp_path / "pred" / "itransformer_top5_h2_w6h__seed1__chrono_test.parquet", test, len(test))
    if HAS_RESULTS_TABLE:
        side = json.loads((tmp_path / "pred" / "itransformer_top5_h2_w6h__seed1__chrono_test.meta.json").read_text())
        assert side["history"] == "w6h_28lags_in_w12h"


@pytest.mark.parametrize("model,run,pred", [
    ("lu_kannan", "lu_kannan_per_user_h4", "lu_kannan"),
    ("patchtst", "patchtst_p2s2_h4", "patchtst"),
    ("lstm", "lstm", "lstm"),
])
def test_other_models_end_to_end(datasets, tmp_path, model, run, pred):
    """Main configuration of each model: bare prediction name, both protocols, scorer reads them."""
    tiny = [a for a in TINY if a not in ("--heads", "2")] + ["--heads", "4"]
    _run(["--model", model, "--data-dir", str(datasets["d12"]), "--seed", "1", "--out-dir", str(tmp_path / "out"),
          "--save-predictions", str(tmp_path / "pred"), *tiny])
    pos = datasets["coarse"]["positions"]
    for prot, split, col in (("chronological", "chrono_test", "time_split"),
                             ("user_disjoint", "user_test", "user_group")):
        r = json.loads((tmp_path / "out" / "raw" / f"{prot}_{run}_seed1.json").read_text())
        assert 0 <= r["pr_auc"] <= 1 and r["epoch_cap"] == 1
        assert ("campaign" in r) == (model != "lstm")           # LSTM is conversion-only
        _check_predictions(tmp_path / "pred" / f"{pred}__seed1__{split}.parquet",
                           pos.loc[pos[col] == "test", ["uid", "window"]], r["n_test"])
        if HAS_RESULTS_TABLE:
            side = json.loads((tmp_path / "pred" / f"{pred}__seed1__{split}.meta.json").read_text())
            assert side["history"] == "w12h_14lags" and side["source"] == "scripts/train_transformer.py"
    assert (tmp_path / "out" / f"conversion_{model}.csv").exists()
    ev = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate_predictions.py"),
                         "--data-dir", str(datasets["d12"]), "--split", "user_test", "--pred-dir", str(tmp_path / "pred"),
                         "--out-dir", str(tmp_path / "eval"), "--n-boot", "5", "--subgroup-n-boot", "5"],
                        capture_output=True, text=True, cwd=REPO)
    assert ev.returncode == 0, ev.stderr[-3000:]
    assert pred in set(pd.read_csv(tmp_path / "eval" / "conversion_ci_user_test.csv")["model"])


def test_ablation_settings_get_their_own_prediction_files(datasets, tmp_path):
    """Heads 1-4 and customer weighting (proposal ablations) must not overwrite the main run."""
    tiny = [a for a in TINY if a not in ("--heads", "2")]
    _run(["--model", "lu_kannan", "--data-dir", str(datasets["d12"]), "--seed", "1", "--protocols", "chronological",
          "--heads", "1", "4", "--weighting", "per_user", "uniform", "amortized", "--skip-campaign-eval",
          "--out-dir", str(tmp_path / "out"), "--save-predictions", str(tmp_path / "pred"), *tiny])
    names = sorted(p.name.split("__")[0] for p in (tmp_path / "pred").glob("*.parquet"))
    assert names == sorted(["lu_kannan", "lu_kannan_h1", "lu_kannan_uniform", "lu_kannan_uniform_h1",
                            "lu_kannan_amortized", "lu_kannan_amortized_h1"])
    assert len(list((tmp_path / "out" / "raw").glob("*.json"))) == 6
    conv = pd.read_csv(tmp_path / "out" / "conversion_lu_kannan.csv")
    assert set(conv["weighting"]) == {"per_user", "uniform", "amortized"} and set(conv["heads"]) == {1, 4}


def test_prediction_names():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tt", REPO / "scripts" / "train_transformer.py")
    tt = importlib.util.module_from_spec(spec)
    sys.modules["tt"] = tt                               # dataclasses look the module up by name
    spec.loader.exec_module(tt)

    class D:                                             # 12h data, or 6h input with 12h targets
        def __init__(self, w): self.window_hours, self.target_hours, self.lookback = w, 12.0, int(168 / w)

    def names(argv, w=12.0):
        a = tt.parse_args(["--data-dir", "x", *argv])
        return [tt.pred_name(a, D(w), v) for v in tt.variants(a)]

    assert names([]) == ["itransformer"] and names([], 6.0) == ["itransformer_w6h"]
    assert names(["--model", "lu_kannan", "--heads", "1", "2", "3", "4"]) == \
        ["lu_kannan_h1", "lu_kannan_h2", "lu_kannan_h3", "lu_kannan"]
    assert names(["--model", "patchtst", "--patches", "2:2", "4:2"]) == ["patchtst", "patchtst_p4s2"]
    assert names(["--model", "lstm", "--heads", "1", "2"]) == ["lstm"]      # heads do not apply
    with pytest.raises(SystemExit):
        tt.parse_args(["--data-dir", "x", "--heads", "1", "4", "--pred-name", "one"])


def test_lookback_in_windows_for_benchmarks(datasets):
    """--lookback-windows fixes the token count whatever the window length (Taobao 3h, YOOCHOOSE 5 min)."""
    quiet = {"log": lambda *_: None, "history_features": False}
    d = WindowData(datasets["d12"], lookback_windows=5, **quiet)
    assert d.lookback == 5
    b = d.make_batch(np.arange(64))
    assert tuple(b["x"].shape[:2]) == (64, 5)
    assert WindowData(datasets["d6"], lookback_windows=14, **quiet).lookback == 14     # 6h data, 14 tokens
    with pytest.raises(ValueError):
        WindowData(datasets["d12"], lookback_hours=18, **quiet)                     # 1.5 windows of 12h


def test_short_window_labels():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tt2", REPO / "scripts" / "train_transformer.py")
    tt = importlib.util.module_from_spec(spec)
    sys.modules["tt2"] = tt
    spec.loader.exec_module(tt)
    assert [tt.hours_label(h) for h in (12, 6, 3, 5 / 60)] == ["12h", "6h", "3h", "5m"]

    class D:
        window_hours, target_hours, lookback = 5 / 60, 5 / 60, 14
    assert tt.history_label(D()) == "w5m_14lags"
    with pytest.raises(SystemExit):                      # hours or windows, not both
        tt.parse_args(["--data-dir", "x", "--lookback-hours", "168", "--lookback-windows", "14"])


def test_lu_kannan_unseen_customers_get_mean_weights():
    from src.models.lu_kannan import LuKannanTransformer
    torch.manual_seed(0)
    m = LuKannanTransformer(n_num=10, n_static=3, n_camp=6, n_users=5, n_heads=3, weighting="per_user")
    torch.nn.init.normal_(m.theta.weight)
    m.mark_seen([0, 1, 2])
    tok = torch.zeros(5, 4, 8)
    m.eval()
    om = m.omega(tok, {"u": torch.arange(5)})
    assert torch.allclose(om[:3], m.theta.weight[:3].softmax(-1))
    mean = m.theta.weight[:3].mean(0).softmax(-1)
    assert torch.allclose(om[3], mean) and torch.allclose(om[4], mean)
    m.train()                                            # training: every customer uses its own weights
    assert torch.allclose(m.omega(tok, {"u": torch.arange(5)}), m.theta.weight.softmax(-1))
