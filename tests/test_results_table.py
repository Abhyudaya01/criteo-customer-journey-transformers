import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.results_table import (
    COLUMNS, UNKNOWN, make_row, read_run_meta, sidecar_path, upsert_results, write_run_meta,
)


def _row(model="gbdt", seed=1, split="chrono_test", pr=0.02, config="neg1pct", history="w12h"):
    meta = {"config": config, "history": history, "source": "s.py", "gpu_hours": 0.0}
    return make_row(model, seed, split, {"pr_auc": pr, "base_rate": 0.002}, meta, 100, 2)


def test_sidecar_path_for_parquet_and_csv_gz(tmp_path):
    assert sidecar_path(tmp_path / "gbdt__seed2__chrono_test.parquet").name == "gbdt__seed2__chrono_test.meta.json"
    assert sidecar_path(tmp_path / "gbdt__seed2__chrono_test.csv.gz").name == "gbdt__seed2__chrono_test.meta.json"


def test_write_and_read_meta_round_trip(tmp_path):
    p = tmp_path / "stajt__seed1__user_test.parquet"
    write_run_meta(p, config="team_aligned_neg1pct", history="events64",
                   source="STAJT/scripts/train_stajt_fair_chrono.py", gpu_hours=6.5, hardware="A100",
                   run_where="zaratan job 1", git_commit="abc123")
    meta = read_run_meta(p)
    assert meta["gpu_hours"] == 6.5 and meta["history"] == "events64" and meta["run_where"] == "zaratan job 1"
    assert json.loads(sidecar_path(p).read_text())["git_commit"] == "abc123"


def test_missing_sidecar_is_marked_unknown(tmp_path):
    meta = read_run_meta(tmp_path / "lstm__seed1__chrono_test.parquet")
    assert meta["config"] == UNKNOWN and np.isnan(meta["gpu_hours"])


def test_upsert_replaces_same_key_and_keeps_others(tmp_path):
    path = tmp_path / "results.csv"
    upsert_results(path, [_row(seed=1, pr=0.020), _row(seed=2, pr=0.021)])
    t = upsert_results(path, [_row(seed=2, pr=0.030), _row(model="lstm", pr=0.025)])
    assert len(t) == 3 and list(t.columns) == COLUMNS
    assert t.loc[(t.model == "gbdt") & (t.seed == 2), "pr_auc"].item() == pytest.approx(0.030)
    assert t.loc[(t.model == "gbdt") & (t.seed == 1), "pr_auc"].item() == pytest.approx(0.020)
    # Same model + seed but a different config or input is a different run, not a replacement.
    t = upsert_results(path, [_row(seed=1, config="optimized_10neg", pr=0.033)])
    assert len(t) == 4
    assert len(pd.read_csv(path)) == 4


def test_foreign_csv_is_not_overwritten(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame({"model": ["x"], "pr": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        upsert_results(path, [_row()])
    assert list(pd.read_csv(path).columns) == ["model", "pr"]


def test_scoring_ignores_sidecars(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "evaluate_predictions", Path(__file__).resolve().parents[1] / "scripts/evaluate_predictions.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    f = tmp_path / "gbdt__seed1__chrono_test.csv"
    pd.DataFrame({"uid": [1, 2], "window": [3, 4], "score": [0.1, 0.2]}).to_csv(f, index=False)
    write_run_meta(f, config="c", history="h", source="s", gpu_hours=0)
    preds, paths = ev.load_predictions(tmp_path, "chrono_test")
    assert list(preds) == [("gbdt", 1)] and paths[("gbdt", 1)] == f
