"""Dataset statistics and report figures for a processed Criteo dataset.

    # 2% sample, seconds
    python scripts/dataset_stats.py --data-dir data/processed/criteo_w12h_smoke
    # full data (CPU, a few minutes; run through Slurm, not on the login node)
    srun -A msml612-class -p debug -t 15 -c 2 --mem=16G \\
        python scripts/dataset_stats.py --data-dir data/processed/criteo_w12h

Writes
    results/tables/dataset_stats_<dataset>.json   every number
    results/tables/dataset_stats_<dataset>.md     report-ready tables
    results/figures/<dataset>_journey_length.png      active windows / impressions per user
    results/figures/<dataset>_conversion_by_window.png active users and conversion rate per window
    results/figures/<dataset>_recency.png             positions and conversion rate by recency
    results/figures/<dataset>_campaign_concentration.png  cumulative impression share by campaign
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "2")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.io import load_meta, load_shard, shard_ids  # noqa: E402

# Light-mode chart tokens (dataviz reference palette); report figures are printed/PDF.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SPLIT_SHADE = {"val": "#f1f0ec", "test": "#e6e5df"}


def quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())}


def collect(data_dir: Path, meta: dict) -> dict:
    T, C = meta["n_windows"], meta["n_campaigns"]
    K = meta["max_inactive_windows"]
    acc = {k: np.zeros(T, dtype=np.int64) for k in
           ("active_users", "impressions", "conversions", "positions", "positives",
            "positions_next_campaign")}
    acc["recency_positions"] = np.zeros(K + 1, dtype=np.int64)
    acc["recency_positives"] = np.zeros(K + 1, dtype=np.int64)
    acc["campaign_impressions"] = np.zeros(C, dtype=np.int64)
    per_user = {k: [] for k in ("impressions", "active_windows", "span_windows", "campaigns",
                                "conversions")}
    next_total = next_novel = pos_next = pos_next_with_novel = 0
    active_cells = span_cells = 0

    for s in shard_ids(data_dir):
        tb = load_shard(data_dir, s)
        uw, pos, wc = tb["user_windows"], tb["positions"], tb["window_campaigns"]
        conv, nx = tb["conversions"], tb["next_campaign_labels"]

        act = uw[uw["is_active"] == 1]
        acc["active_users"] += np.bincount(act["window"], minlength=T)
        acc["impressions"] += np.bincount(act["window"], weights=act["n_impressions"],
                                          minlength=T).astype(np.int64)
        acc["conversions"] += np.bincount(conv["window"], minlength=T)
        acc["positions"] += np.bincount(pos["label_window"], minlength=T)
        acc["positives"] += np.bincount(pos["label_window"], weights=pos["y_conversion"],
                                        minlength=T).astype(np.int64)
        acc["positions_next_campaign"] += np.bincount(
            pos["label_window"], weights=(pos["n_next_campaigns"] > 0), minlength=T).astype(np.int64)
        r = pos["windows_since_active"].clip(upper=K)
        acc["recency_positions"] += np.bincount(r, minlength=K + 1)
        acc["recency_positives"] += np.bincount(r, weights=pos["y_conversion"],
                                                minlength=K + 1).astype(np.int64)
        acc["campaign_impressions"] += np.bincount(wc["campaign_idx"], weights=wc["n_impressions"],
                                                   minlength=C).astype(np.int64)

        g = act.groupby("uid")
        per_user["impressions"].append(g["n_impressions"].sum().to_numpy())
        per_user["active_windows"].append(g.size().to_numpy())
        span = (g["window"].max() - g["window"].min() + 1).to_numpy()
        per_user["span_windows"].append(span)
        active_cells += len(act)
        span_cells += int(span.sum())
        per_user["campaigns"].append(wc.groupby("uid")["campaign_idx"].nunique().to_numpy())
        users = g.size().index
        per_user["conversions"].append(
            conv.groupby("uid").size().reindex(users, fill_value=0).to_numpy())

        # Novel campaigns: a next-window campaign the user had never been shown before.
        first = wc.groupby(["uid", "campaign_idx"])["window"].min().rename("first_seen").reset_index()
        lab = nx.merge(first, on=["uid", "campaign_idx"], how="left")
        novel = (lab["first_seen"] == lab["label_window"]).to_numpy()
        next_total += len(lab)
        next_novel += int(novel.sum())
        per_pos = lab.assign(novel=novel).groupby(["uid", "window"])["novel"].any()
        pos_next += len(per_pos)
        pos_next_with_novel += int(per_pos.sum())
        print(f"  shard {s}: {len(pos):,} positions", flush=True)

    per_user = {k: np.concatenate(v) for k, v in per_user.items()}
    acc["novel"] = {"next_window_campaigns": next_total, "novel_campaigns": next_novel,
                    "positions_with_next_campaign": pos_next,
                    "positions_with_a_novel_campaign": pos_next_with_novel}
    acc["active_cells"], acc["span_cells"] = active_cells, span_cells
    return {"acc": acc, "per_user": per_user}


def _min_max(rate: np.ndarray, positions: np.ndarray) -> dict:
    """Min/max conversion rate over label windows with enough positions to be stable."""
    ok = positions >= max(1, 0.05 * np.median(positions[positions > 0]))
    idx = np.flatnonzero(ok)
    r = rate[idx]
    return {"min": float(r.min()), "max": float(r.max()),
            "argmin": int(idx[r.argmin()]), "argmax": int(idx[r.argmax()]),
            "windows_excluded_as_too_small": np.flatnonzero(~ok & (positions > 0)).tolist()}


def summarize(meta: dict, acc: dict, per_user: dict) -> dict:
    pu = per_user
    n_users = len(pu["impressions"])
    aw = pu["active_windows"]
    camp = np.sort(acc["campaign_impressions"])[::-1]
    cum = np.cumsum(camp) / camp.sum()
    rec_rate = np.divide(acc["recency_positives"], acc["recency_positions"],
                         out=np.zeros(len(acc["recency_positions"])), where=acc["recency_positions"] > 0)
    nv = acc["novel"]
    b = meta["splits"]["boundaries"]
    rate = np.divide(acc["positives"], acc["positions"], out=np.zeros(len(acc["positions"])),
                     where=acc["positions"] > 0)
    split_rates = {}
    for name, key in (("train", "train_label_windows"), ("val", "val_label_windows"),
                      ("test", "test_label_windows")):
        lo, hi = b[key]
        split_rates[name] = float(acc["positives"][lo:hi + 1].sum() / acc["positions"][lo:hi + 1].sum())
    per_day = max(1, int(round(24 / meta["window_hours"])))
    slot_rate = {int(sl): float(acc["positives"][sl::per_day].sum() / max(acc["positions"][sl::per_day].sum(), 1))
                 for sl in range(per_day)}
    return {
        "dataset": None,
        "window_hours": meta["window_hours"], "n_windows": meta["n_windows"],
        "n_users": int(n_users),
        "per_user": {
            "impressions": quantiles(pu["impressions"]),
            "active_windows": quantiles(aw),
            "span_windows": quantiles(pu["span_windows"]),
            "distinct_campaigns": quantiles(pu["campaigns"]),
            "share_one_active_window": float((aw == 1).mean()),
            "share_one_impression": float((pu["impressions"] == 1).mean()),
            "share_ge_5_active_windows": float((aw >= 5).mean()),
            "share_users_converting": float((pu["conversions"] > 0).mean()),
        },
        "sparsity": {
            "active_share_of_journey_span": acc["active_cells"] / acc["span_cells"],
            "active_share_of_all_user_windows": acc["active_cells"] / (n_users * meta["n_windows"]),
            "positions_inactive_share": float(1 - acc["recency_positions"][0] / acc["recency_positions"].sum()),
            "positions_with_next_campaign_share": float(acc["positions_next_campaign"].sum() / acc["positions"].sum()),
            "conversion_rate_active_vs_inactive": [float(rec_rate[0]),
                float(acc["recency_positives"][1:].sum() / max(acc["recency_positions"][1:].sum(), 1))],
        },
        "conversion_rate": {
            "overall": float(acc["positives"].sum() / acc["positions"].sum()),
            "by_split": split_rates,
            "by_label_window": _min_max(rate, acc["positions"]),
            "by_slot_of_day": slot_rate,
        },
        "campaigns": {
            "n_campaigns": int(len(camp)),
            "top10_impression_share": float(cum[min(9, len(cum) - 1)]),
            "top50_impression_share": float(cum[min(49, len(cum) - 1)]),
            "top100_impression_share": float(cum[min(99, len(cum) - 1)]),
            "next_window_campaigns_novel_share": nv["novel_campaigns"] / max(nv["next_window_campaigns"], 1),
            "positions_with_a_novel_campaign_share":
                nv["positions_with_a_novel_campaign"] / max(nv["positions_with_next_campaign"], 1),
        },
        "series": {
            "active_users_per_window": acc["active_users"].tolist(),
            "impressions_per_window": acc["impressions"].tolist(),
            "conversions_per_window": acc["conversions"].tolist(),
            "positions_per_label_window": acc["positions"].tolist(),
            "positives_per_label_window": acc["positives"].tolist(),
            "positions_by_windows_since_active": acc["recency_positions"].tolist(),
            "conversion_rate_by_windows_since_active": rec_rate.tolist(),
            "active_windows_histogram_1_to_15plus":
                np.bincount(np.minimum(aw, 15), minlength=16)[1:].tolist(),
            "campaign_cumulative_impression_share": cum.tolist(),
        },
    }


# ------------------------------------------------------------------ figures

def _style(ax, ylabel: str | None = None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)


def _fig(nrows=1, ncols=1, h=2.6, sharex=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, h), sharex=sharex,
                             facecolor=SURFACE, squeeze=False)
    return plt, fig, axes.ravel()


def _title(fig, title: str, subtitle: str):
    fig.text(0.01, 0.985, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, subtitle, fontsize=8.5, color=INK_2, va="top")


def _pct(ax, decimals=0):
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def fig_journey_length(s: dict, path: Path):
    plt, fig, (a1, a2) = _fig(1, 2, h=2.7)
    hist = np.array(s["series"]["active_windows_histogram_1_to_15plus"], dtype=float)
    hist = np.concatenate([hist[:9], [hist[9:].sum()]])   # 1..9, 10+
    share = hist / hist.sum()
    labels = [str(i) for i in range(1, 10)] + ["10+"]
    a1.bar(range(len(share)), share, width=0.72, color=BLUE)
    a1.set_xticks(range(len(share)), labels, fontsize=7.5)
    _style(a1, "Share of users"); _pct(a1)
    a1.set_xlabel("Active windows per user", color=INK_2, fontsize=8.5)
    a1.text(0.6, share[0], f"{share[0]:.0%} of users are active\nin a single window only",
            fontsize=8, color=INK, va="top", ha="left")
    pu = s["per_user"]
    rows = [("Impressions", pu["impressions"]), ("Active windows", pu["active_windows"]),
            ("Span (windows)", pu["span_windows"]), ("Campaigns", pu["distinct_campaigns"])]
    a2.axis("off")
    cell = [[name, f"{q['mean']:.1f}", f"{q['p50']:.0f}", f"{q['p90']:.0f}", f"{q['p99']:.0f}"] for name, q in rows]
    tbl = a2.table(cellText=cell, colLabels=["Per user", "Mean", "Med.", "P90", "P99"],
                   loc="center", cellLoc="right", colLoc="right", colWidths=[0.40, 0.15, 0.15, 0.15, 0.15])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor(GRID); cl.set_linewidth(0.5); cl.set_facecolor(SURFACE)
        cl.get_text().set_color(INK_2 if r == 0 else INK)
        if c == 0:
            cl.get_text().set_ha("left"); cl._loc = "left"
    tbl.scale(1, 1.5)
    _title(fig, "Journey length per user",
           f"{s['n_users']:,} users, {s['window_hours']:g}h windows over {s['n_windows']} windows")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def _shade_splits(ax, b: dict):
    for name, key in (("val", "val_label_windows"), ("test", "test_label_windows")):
        lo, hi = b[key]
        ax.axvspan(lo - 0.5, hi + 0.5, color=SPLIT_SHADE[name], lw=0, zorder=0)


def fig_conversion_by_window(s: dict, b: dict, path: Path):
    plt, fig, (a1, a2) = _fig(2, 1, h=4.0, sharex=True)
    T = s["n_windows"]
    x = np.arange(T)
    act = np.array(s["series"]["active_users_per_window"], dtype=float)
    pos = np.array(s["series"]["positions_per_label_window"], dtype=float)
    pp = np.array(s["series"]["positives_per_label_window"], dtype=float)
    # Hide windows with too few positions to give a stable rate (e.g. window 1).
    enough = pos >= max(1.0, 0.05 * np.median(pos[pos > 0]))
    rate = np.divide(pp, pos, out=np.full(T, np.nan), where=enough)
    for ax in (a1, a2):
        _shade_splits(ax, b)
    a1.bar(x, act / 1e3, width=0.8, color=BLUE)
    _style(a1, "Active users (thousands)")
    a2.plot(x[1:], rate[1:], color=BLUE, lw=1.4)
    overall = s["conversion_rate"]["overall"]
    a2.axhline(overall, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    _style(a2, "Conversion rate in window"); _pct(a2, 2)
    a2.set_xlabel(f"Window index ({s['window_hours']:g}h each; label window for the rate)", color=INK_2, fontsize=8.5)
    a2.set_ylim(bottom=0)
    for name, key in (("train", "train_label_windows"), ("val", "val_label_windows"),
                      ("test", "test_label_windows")):
        lo, hi = b[key]
        a1.text((lo + hi) / 2, 1.02, name, transform=a1.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK_2)
    days = s["n_windows"] * s["window_hours"] / 24
    _title(fig, f"Activity and conversion rate across the {days:.0f} days",
           f"Shaded: validation and test label windows. Dashed line: overall rate {overall:.2%}")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def fig_recency(s: dict, path: Path):
    plt, fig, (a1, a2) = _fig(1, 2, h=2.7)
    n = np.array(s["series"]["positions_by_windows_since_active"], dtype=float)
    r = np.array(s["series"]["conversion_rate_by_windows_since_active"], dtype=float)
    x = np.arange(len(n))
    a1.bar(x, n / n.sum(), width=0.72, color=BLUE)
    _style(a1, "Share of prediction positions"); _pct(a1)
    a1.set_xlabel("Windows since last impression", color=INK_2, fontsize=8.5)
    a2.bar(x, r, width=0.72, color=BLUE)
    _style(a2, "Conversion rate in next window"); _pct(a2, 2)
    for ax in (a1, a2):
        ax.set_xticks(x[::2], [str(v) for v in x[::2]])
    a2.set_xlabel("Windows since last impression", color=INK_2, fontsize=8.5)
    cutoff_h = (len(n) - 1) * s["window_hours"]
    _title(fig, "Prediction positions and conversion rate by recency",
           f"0 = impression in the current window; positions kept up to the {cutoff_h / 24:g}-day activity cutoff")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def fig_campaign_concentration(s: dict, path: Path):
    plt, fig, (ax,) = _fig(1, 1, h=2.6)
    cum = np.array(s["series"]["campaign_cumulative_impression_share"])
    x = np.arange(1, len(cum) + 1)
    ax.plot(x, cum, color=BLUE, lw=1.4)
    for k in (10, 100):  # labelled points; full curve is in the JSON
        if k <= len(cum):
            ax.plot([k], [cum[k - 1]], "o", ms=5, color=BLUE, mec=SURFACE, mew=1.2)
            ax.annotate(f"top {k}: {cum[k - 1]:.0%}", (k, cum[k - 1]), xytext=(8, -12),
                        textcoords="offset points", fontsize=8, color=INK)
    _style(ax, "Cumulative share of impressions"); _pct(ax)
    ax.set_xlabel("Campaigns ranked by impressions", color=INK_2, fontsize=8.5)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, len(cum) + 5)
    _title(fig, "Concentration of impressions across campaigns",
           f"{len(cum)} campaigns ranked by total impressions over all windows")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def to_markdown(s: dict, meta: dict) -> str:
    pu, sp, cr, cp = s["per_user"], s["sparsity"], s["conversion_rate"], s["campaigns"]
    q = lambda d: f"{d['mean']:.2f} | {d['p50']:.0f} | {d['p90']:.0f} | {d['p99']:.0f} | {d['max']:.0f}"  # noqa: E731
    bs = cr["by_split"]
    return "\n".join([
        f"# Dataset statistics: {s['dataset']} ({s['window_hours']:g}h windows, {s['n_windows']} windows)", "",
        "## Journey length per user", "",
        "| Per user | Mean | Median | P90 | P99 | Max |", "|---|---|---|---|---|---|",
        f"| Impressions | {q(pu['impressions'])} |",
        f"| Active windows | {q(pu['active_windows'])} |",
        f"| Journey span (first to last active window) | {q(pu['span_windows'])} |",
        f"| Distinct campaigns | {q(pu['distinct_campaigns'])} |", "",
        f"- Users: {s['n_users']:,}; active in one window only: {pu['share_one_active_window']:.1%}; "
        f"one impression only: {pu['share_one_impression']:.1%}; 5+ active windows: {pu['share_ge_5_active_windows']:.1%}",
        f"- Users with at least one conversion: {pu['share_users_converting']:.2%}", "",
        "## Sparsity", "",
        f"- Active share of user-windows within each journey span: {sp['active_share_of_journey_span']:.1%}",
        f"- Active share of all user x window cells: {sp['active_share_of_all_user_windows']:.2%}",
        f"- Prediction positions with no impression in the current window: {sp['positions_inactive_share']:.1%}",
        f"- Positions with any campaign in the next window: {sp['positions_with_next_campaign_share']:.1%}",
        f"- Conversion rate in next window: {sp['conversion_rate_active_vs_inactive'][0]:.3%} if active now, "
        f"{sp['conversion_rate_active_vs_inactive'][1]:.3%} if not", "",
        "## Conversion rate", "",
        f"- Overall (per position): {cr['overall']:.3%}",
        f"- Train / val / test label windows: {bs['train']:.3%} / {bs['val']:.3%} / {bs['test']:.3%}",
        f"- Per label window: min {cr['by_label_window']['min']:.3%} (window {cr['by_label_window']['argmin']}), "
        f"max {cr['by_label_window']['max']:.3%} (window {cr['by_label_window']['argmax']}); "
        f"excluded as too small: {cr['by_label_window']['windows_excluded_as_too_small'] or 'none'}",
        "- By slot of day: " + ", ".join(f"slot {k}: {v:.3%}" for k, v in cr["by_slot_of_day"].items()), "",
        "## Campaigns", "",
        f"- Top 10 / 50 / 100 campaigns' share of impressions: {cp['top10_impression_share']:.1%} / "
        f"{cp['top50_impression_share']:.1%} / {cp['top100_impression_share']:.1%} (of {cp['n_campaigns']})",
        f"- Next-window campaigns the user had never been shown before: {cp['next_window_campaigns_novel_share']:.1%}",
        f"- Positions (with a next-window campaign) that include a new-to-user campaign: "
        f"{cp['positions_with_a_novel_campaign_share']:.1%}", "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args()
    t0 = time.time()
    data_dir = args.data_dir if args.data_dir.is_absolute() else REPO_ROOT / args.data_dir
    meta = load_meta(data_dir)
    print(f"{data_dir.name}: collecting statistics", flush=True)
    c = collect(data_dir, meta)
    s = summarize(meta, c["acc"], c["per_user"])
    s["dataset"] = data_dir.name
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tables_dir / f"dataset_stats_{data_dir.name}"
    stem.with_suffix(".json").write_text(json.dumps(s, indent=2))
    md = to_markdown(s, meta)
    stem.with_suffix(".md").write_text(md)
    b = meta["splits"]["boundaries"]
    figs = {
        "journey_length": lambda p: fig_journey_length(s, p),
        "conversion_by_window": lambda p: fig_conversion_by_window(s, b, p),
        "recency": lambda p: fig_recency(s, p),
        "campaign_concentration": lambda p: fig_campaign_concentration(s, p),
    }
    for name, fn in figs.items():
        fn(args.figures_dir / f"{data_dir.name}_{name}.png")
    print(md)
    print(f"Saved {stem}.json/.md and {len(figs)} figures to {args.figures_dir} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
