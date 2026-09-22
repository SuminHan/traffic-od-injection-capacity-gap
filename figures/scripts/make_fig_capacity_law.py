"""Generates fig_capacity_law.pdf: injection effect vs. model capacity.

Two panels sharing a log-parameter x-axis:
  (a) BETWEEN architectures -- all ten published/reimplemented architectures at 30 folds, both
      tasks, with an OLS fit and Spearman rho annotated.
  (b) WITHIN one architecture -- the STID capacity sweep (hidden 64/128/256, 30 folds each, two
      seeds at hidden 64), showing the same decay with architecture, training recipe and injected
      signal all held fixed.

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
                         "p": float(stats.ttest_rel(O, P)[1])})
        except Exception:
            pass

    import numpy as np
    x = np.log10([r["params"] for r in rows])
    y = np.array([r["pct"] for r in rows])
    rho, prho = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    sw = pd.read_csv(f"{GTS}/stid_fixed_volume_capacity_sweep_summary.csv")
    agg = (sw.groupby("plain_params")
             .agg(pct=("injection_mean_pct", "mean"), lo=("injection_mean_pct", "min"),
                  hi=("injection_mean_pct", "max"), p=("injection_p", "max"))
             .reset_index().sort_values("plain_params"))

    out = {"between": rows,
           "fit": {"slope": float(slope), "intercept": float(intercept),
                   "rho": float(rho), "p_rho": float(prho), "n": len(rows)},
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

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.4, 4.0),
                                   gridspec_kw={"height_ratios": [1.35, 1]})

    marker_name = {"o": "circle", "^": "triangle"}
    for task, mark, col in [("volume", "o", "#2f4f8f"), ("speed", "^", "#b8860b")]:
        sel = [r for r in rows if r["task"] == task]
        sig = [r for r in sel if r["p"] < 0.05]
        ns = [r for r in sel if r["p"] >= 0.05]
        shape_word = marker_name[mark]
        if sig:
            axa.scatter([r["params"] for r in sig], [r["pct"] for r in sig], marker=mark,
                        s=36, c=col, label=f"{shape_word} = {task}, filled = sig.", zorder=3)
        if ns:
            axa.scatter([r["params"] for r in ns], [r["pct"] for r in ns], marker=mark, s=36,
                        facecolors="none", edgecolors=col, linewidths=1.2,
                        label=f"{shape_word} = {task}, open = n.s.", zorder=3)

    xs = np.linspace(np.log10(min(r["params"] for r in rows)),
                     np.log10(max(r["params"] for r in rows)), 50)
    axa.plot(10 ** xs, fit["slope"] * xs + fit["intercept"], "--", color="#888888", lw=1.1, zorder=2)
    axa.annotate("small models:\ninjection helps", xy=(0.30, 0.04), xycoords="axes fraction",
                 fontsize=6.3, color="#2f7d52", style="italic", ha="left", va="bottom")
    axa.annotate("large models:\ninjection hurts", xy=(0.97, 0.94), xycoords="axes fraction",
                 fontsize=6.3, color="#b0402f", style="italic", ha="right", va="top")
    axa.axhline(0, color="black", lw=0.8, zorder=1)
    axa.set_xscale("log")
    axa.set_ylabel("injection effect\n(% $\\Delta$MSE)")
    axa.set_title(f"(a) between architectures  ($\\rho$={fit['rho']:.2f}, "
                  f"p={fit['p_rho']:.1g})", fontsize=8.5)
    axa.legend(fontsize=5.6, loc="upper left", framealpha=0.9, ncol=1, handletextpad=0.4,
               labelspacing=0.3)
    for r in rows:
        if r["task"] == "volume" and r["model"] in ("stid_fixed", "gman", "gwnet", "agcrn", "mtgnn"):
            axa.annotate(r["name"], (r["params"], r["pct"]), fontsize=5.8,
                         xytext=(2, 3.5), textcoords="offset points", color="#333333")

    px = [w["plain_params"] for w in within]
    axb.plot(px, [w["pct"] for w in within], "-o", color="#2f7d52", lw=1.3, ms=5, zorder=3)
    axb.fill_between(px, [w["lo"] for w in within], [w["hi"] for w in within],
                     color="#2f7d52", alpha=0.18, zorder=2)
    for w in within:
        ns_ = w["p"] >= 0.05
        axb.annotate("n.s." if ns_ else "sig.", (w["plain_params"], w["pct"]), fontsize=6,
                     xytext=(3, -9 if ns_ else 5), textcoords="offset points", color="#333333")
    axb.axhline(0, color="black", lw=0.8, zorder=1)
    axb.set_xscale("log")
    axb.set_xlabel("plain-model parameters (log scale)")
    axb.set_ylabel("injection effect\n(% $\\Delta$MSE)")
    axb.set_title("(b) within STID (volume, 30 folds)", fontsize=8.5)
    axa.set_xlim(axb.get_xlim()[0], axa.get_xlim()[1])

    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_capacity_law.pdf")
    print(f"wrote {OUT}/fig_capacity_law.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stats", "plot"], required=True)
    a = ap.parse_args()
    stage_stats() if a.stage == "stats" else stage_plot()
