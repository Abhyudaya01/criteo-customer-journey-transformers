"""End-to-end check for scripts/dataset_stats.py on synthetic data."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.data.preprocess import build_windowed_dataset
from src.data.splits import assign_chronological_split, user_groups, window_boundaries
from tests.synthetic import make_synthetic_criteo

REPO = Path(__file__).resolve().parents[1]


def test_dataset_stats_end_to_end():
    res = build_windowed_dataset(make_synthetic_criteo(n_users=600, conv_prob=0.2), 12, 48)
    b = window_boundaries(res["meta"]["n_windows"])
    pos = res["positions"]
    pos["time_split"] = assign_chronological_split(pos, b)
    pos["user_group"] = pos["uid"].map(user_groups(pos["uid"]))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds = tmp / "criteo_w12h_test"
        for name, table in res.items():
            if name not in ("meta", "campaign_vocab"):
                (ds / name).mkdir(parents=True)
                table.to_csv(ds / name / "part-000.csv.gz", index=False)
        (ds / "meta.json").write_text(json.dumps(dict(res["meta"], splits={"boundaries": b.as_dict()})))
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "dataset_stats.py"),
                              "--data-dir", str(ds), "--tables-dir", str(tmp), "--figures-dir", str(tmp)],
                             capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr[-2000:]
        s = json.loads((tmp / "dataset_stats_criteo_w12h_test.json").read_text())
        assert s["n_users"] == res["user_windows"].loc[res["user_windows"].is_active == 1, "uid"].nunique()
        assert sum(s["series"]["positions_per_label_window"]) == len(pos)
        assert sum(s["series"]["positives_per_label_window"]) == pos["y_conversion"].sum()
        assert abs(s["conversion_rate"]["overall"] - pos["y_conversion"].mean()) < 1e-12
        assert 0 <= s["campaigns"]["next_window_campaigns_novel_share"] <= 1
        assert len(list(tmp.glob("criteo_w12h_test_*.png"))) == 4
