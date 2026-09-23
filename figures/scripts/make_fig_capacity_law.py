"""Generates fig_capacity_law.pdf: injection effect vs. model capacity.

Three panels:
  (a) BETWEEN architectures, VOLUME -- all ten published/reimplemented architectures at 30 folds,
      absolute RMSE, plain (open circle) to uniform-injected (filled circle), joined by an arrow
      colored by direction (green = improved, red = worsened). Rows ordered by parameter count.
  (b) BETWEEN architectures, SPEED -- same design, speed task.
  (c) WITHIN one architecture -- the STID capacity sweep (hidden 64/128/256, 30 folds each, two
      seeds at hidden 64), showing the same decay (in % effect) with architecture, training recipe
      and injected signal all held fixed.

(a)/(b) replace the earlier single %-change-vs-log-parameters scatter: that design buried the
plain-vs-injected comparison inside a single derived percentage, which was hard to read at a
glance. Splitting by task and showing the actual before/after RMSE values with an arrow makes the
capacity effect (small models improve, large models regress) directly visible without requiring
the reader to mentally invert a percentage's sign convention. The underlying capacity-law
correlation (Spearman rho vs. log parameters) is preserved as annotated text on each panel.

No single conda env on this machine has matplotlib, pandas and scipy together, so this runs in two
stages: `--stage stats` (trajtok: pandas/scipy) writes fig_capacity_law_data.json, and
`--stage plot` (playground: matplotlib/numpy) renders it. `make_fig_capacity_law.sh` runs both.

Style matches make_figs.py so the figure drops into the two-column IEEE layout unchanged.
"""
import argparse
import glob
import json

GTS = "/home/ncrc/work/gts"
OUT = f"{GTS}/paper_tkde/figs"
DATA = f"{GTS}/paper_tkde/fig_capacity_law_data.json"

NAME = {"stid": "STID", "stid_fixed": "STID", "gman": "GMAN", "stgcn": "STGCN", "pdformer": "PDFormer",
        "staeformer": "STAEformer", "mtgnn": "MTGNN", "dcrnn": "DCRNN", "gts": "GTS",
        "gwnet": "GWNET", "agcrn": "AGCRN"}


def stage_stats():
    import numpy as np
    import pandas as pd
    from scipy import stats

    rows = []
    for f in sorted(glob.glob(f"{GTS}/multi_fold_*_results_ext30.json")):
        if "multipath" in f:
            continue
        # stid was reimplemented with the correct architecture (see NIGHT_LOOP_NOTES.md
        # 2026-09-21) -- use ONLY the corrected file, skip the original buggy one so it
        # doesn't appear as a separate/duplicate point.
        if f.endswith("multi_fold_baseline_stid_volume_results_ext30.json") or \
           f.endswith("multi_fold_baseline_stid_speed_results_ext30.json"):
            continue
        try:
            r = json.load(open(f))
            by = {}
            for e in r:
                by.setdefault(e["fold_start"], {})[e["variant"]] = e
            k0 = list(by)[0]
            odk = next((v for v in by[k0] if v != "plain"), None)
            if odk is None:
                continue
            P, O = [], []
            for k, v in sorted(by.items()):
                if "plain" in v and odk in v:
                    P.append(v["plain"]["test_mse"])
                    O.append(v[odk]["test_mse"])
            if len(P) < 5:
                continue
            P, O = pd.Series(P), pd.Series(O)
            nm = (f.split("/")[-1].replace("multi_fold_", "")
                   .replace("_results_ext30.json", "").replace("baseline_", ""))
            m, t = nm.rsplit("_", 1)
            rows.append({"model": m, "name": NAME.get(m, m), "task": t,
                         "params": int(by[k0]["plain"]["n_params"]),
                         "pct": float(((O - P) / P * 100).mean()),
                         "p": float(stats.ttest_rel(O, P)[1]),
                         "rmse_plain": float(np.sqrt(P).mean()),
                         "rmse_od": float(np.sqrt(O).mean())})
        except Exception:
            pass

    x = np.log10([r["params"] for r in rows])
    y = np.array([r["pct"] for r in rows])
    rho, prho = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    fit_by_task = {}
    for task in ["volume", "speed"]:
        sel = [r for r in rows if r["task"] == task]
        xt = np.log10([r["params"] for r in sel])
        yt = np.array([r["pct"] for r in sel])
        rho_t, prho_t = stats.spearmanr(xt, yt)
        fit_by_task[task] = {"rho": float(rho_t), "p_rho": float(prho_t)}

    sw = pd.read_csv(f"{GTS}/stid_fixed_volume_capacity_sweep_summary.csv")
    agg = (sw.groupby("plain_params")
             .agg(pct=("injection_mean_pct", "mean"), lo=("injection_mean_pct", "min"),
                  hi=("injection_mean_pct", "max"), p=("injection_p", "max"))
             .reset_index().sort_values("plain_params"))

    out = {"between": rows,
           "fit": {"slope": float(slope), "intercept": float(intercept),
                   "rho": float(rho), "p_rho": float(prho), "n": len(rows)},
           "fit_by_task": fit_by_task,
           "within_stid": agg.to_dict("records")}
    json.dump(out, open(DATA, "w"), indent=1)
    print(f"wrote {DATA}")
    print(f"  between-architecture Spearman rho={rho:.3f} (p={prho:.2g}), n={len(rows)}")
    print(agg.to_string(index=False))


def stage_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    d = json.load(open(DATA))
    rows, fit, within = d["between"], d["fit"], d["within_stid"]
    fit_by_task = d["fit_by_task"]

    fig = plt.figure(figsize=(7.1, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.65,
                          left=0.105, right=0.985, top=0.76, bottom=0.20)
    axa, axb, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    def dumbbell(ax, task, title, rho, p_rho):
        # ordered by parameter count ascending, matching Table absperf/general
        sel = sorted([r for r in rows if r["task"] == task], key=lambda r: r["params"])
        ys = np.arange(len(sel))
        for y, r in zip(ys, sel):
            improved = r["rmse_od"] < r["rmse_plain"]
            col = "#2f7d52" if improved else "#b0402f"
            ax.annotate("", xy=(r["rmse_od"], y), xytext=(r["rmse_plain"], y),
                        arrowprops=dict(arrowstyle="-|>", color=col, alpha=0.8, lw=1.3,
                                         shrinkA=4, shrinkB=4, mutation_scale=9), zorder=2)
        ax.scatter([r["rmse_plain"] for r in sel], ys, marker="o", s=32,
                   facecolors="white", edgecolors="#333333", linewidths=1.1, zorder=3,
                   label="plain")
        ax.scatter([r["rmse_od"] for r in sel], ys, marker="o", s=32,
                   c=["#2f7d52" if r["rmse_od"] < r["rmse_plain"] else "#b0402f" for r in sel],
                   edgecolors="#333333", linewidths=0.5, zorder=3, label="injected (uniform)")
        ax.set_yticks(ys)
        ax.set_yticklabels([r["name"] for r in sel], fontsize=7.3)
        ax.set_ylim(-0.7, len(sel) - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE", fontsize=8)
        ax.set_title(f"{title}\ncapacity-law $\\rho$={rho:.2f} (log params, p={p_rho:.1g})",
                     fontsize=7.8)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)

    dumbbell(axa, "volume", "(a) volume: plain $\\to$ injected RMSE",
             fit_by_task["volume"]["rho"], fit_by_task["volume"]["p_rho"])
    dumbbell(axb, "speed", "(b) speed: plain $\\to$ injected RMSE",
             fit_by_task["speed"]["rho"], fit_by_task["speed"]["p_rho"])
    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, framealpha=0.9, handletextpad=0.4, columnspacing=1.2)

    px = [w["plain_params"] for w in within]
    yc_lo = min(w["lo"] for w in within)
    yc_hi = max(w["hi"] for w in within)
    yc_pad = 0.12 * (yc_hi - yc_lo)
    axc.axhspan(yc_lo - yc_pad, 0, color="#2f7d52", alpha=0.07, zorder=0)
    axc.axhspan(0, yc_hi + yc_pad, color="#b0402f", alpha=0.07, zorder=0)
    axc.set_ylim(yc_lo - yc_pad, yc_hi + yc_pad)
    axc.plot(px, [w["pct"] for w in within], "-o", color="#2f7d52", lw=1.3, ms=5, zorder=3)
    axc.fill_between(px, [w["lo"] for w in within], [w["hi"] for w in within],
                     color="#2f7d52", alpha=0.18, zorder=2)
    for i, w in enumerate(within):
        ns_ = w["p"] >= 0.05
        dx = -22 if i == len(within) - 1 else 3
        axc.annotate("n.s." if ns_ else "sig.", (w["plain_params"], w["pct"]), fontsize=6,
                     xytext=(dx, -9 if ns_ else 5), textcoords="offset points", color="#333333")
    axc.axhline(0, color="black", lw=0.8, zorder=1)
    axc.set_xscale("log")
    axc.margins(x=0.15)
    axc.set_xlabel("plain-model parameters\n(log scale)", fontsize=8)
    axc.set_ylabel("injection effect (% $\\Delta$MSE)", fontsize=8)
    axc.set_title("(c) within STID\n(volume, 30 folds)", fontsize=8.5)

    fig.savefig(f"{OUT}/fig_capacity_law.pdf")
    print(f"wrote {OUT}/fig_capacity_law.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stats", "plot"], required=True)
    a = ap.parse_args()
    stage_stats() if a.stage == "stats" else stage_plot()
