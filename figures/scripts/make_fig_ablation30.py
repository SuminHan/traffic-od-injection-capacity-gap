"""Summary comparison of the four signal-decomposition ablations (Hybrid, Climatology, Residual,
Dual-channel) at full 30-fold resolution -- companion figure to Tables confound/residual/dualsignal,
in the same style as make_figs.py (matplotlib, serif, single-column).

A per-fold line-chart version (4 series x 30 points) was tried first and found unreadable -- too
much crossing/noise to parse at a glance. This version shows the one number each table's text
actually leans on: each condition's 30-fold MEAN, with 1-SEM error bars and its recovery %
relative to Hybrid annotated directly on the bar."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GTS = "/home/ncrc/work/gts"
OUT = f"{GTS}/paper_tkde/figs"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                      "grid.alpha": 0.3, "axes.axisbelow": True})

existing = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
only = {e["fold_start"]: e["test_mse"] for e in existing if e["model"] == "traffic_only"}
hyb = {e["fold_start"]: e["test_mse"] for e in existing if e["model"] == "traffic_hybrid_routed"}
clim = {e["fold_start"]: e["test_mse"] for e in json.load(open(f"{GTS}/climatology_ablation_results.json"))}
resid = {e["fold_start"]: e["test_mse"] for e in json.load(open(f"{GTS}/od_residual_ablation_results.json"))}
dual = {e["fold_start"]: e["test_mse"] for e in json.load(open(f"{GTS}/dual_signal_ablation_results.json"))}

folds = sorted(set(only) & set(hyb) & set(clim) & set(resid) & set(dual))
pct = lambda d: [(d[f] - only[f]) / only[f] * 100 for f in folds]

vol_series = [
    ("Hybrid\n(real)", pct(hyb), "#1a1a1a"),
    ("Clima-\ntology", pct(clim), "#2f6ea3"),
    ("Resid-\nual", pct(resid), "#c8781f"),
    ("Dual-\nchannel", pct(dual), "#7a4fa3"),
]

speed_base = json.load(open(f"{GTS}/multi_fold_speed_results_ext.json"))
s_only = {e["fold_start"]: e["test_mse"] for e in speed_base if e["model"] == "speed_only"}
s_hyb = {e["fold_start"]: e["test_mse"] for e in speed_base if e["model"] == "speed_hybrid_routed"}
speed_ctl = json.load(open(f"{GTS}/speed_calendar_controls_results.json"))
s_clim = {e["fold_start"]: e["test_mse"] for e in speed_ctl if e["kind"] == "climatology"}
s_resid = {e["fold_start"]: e["test_mse"] for e in speed_ctl if e["kind"] == "residual"}
s_folds = sorted(set(s_only) & set(s_hyb) & set(s_clim) & set(s_resid))
s_pct = lambda d: [(d[f] - s_only[f]) / s_only[f] * 100 for f in s_folds]

speed_series = [
    ("Hybrid\n(real)", s_pct(s_hyb), "#1a1a1a"),
    ("Clima-\ntology", s_pct(s_clim), "#2f6ea3"),
    ("Resid-\nual", s_pct(s_resid), "#c8781f"),
]

fig, (axv, axs) = plt.subplots(1, 2, figsize=(4.6, 2.3), width_ratios=[4, 3])
for ax, series, title in [(axv, vol_series, "Volume"), (axs, speed_series, "Speed")]:
    means = [float(np.mean(v)) for _, v, _ in series]
    sems = [float(np.std(v, ddof=1) / np.sqrt(len(v))) for _, v, _ in series]
    labels = [t[0] for t in series]
    colors = [t[2] for t in series]
    recovery = [100.0 * m / means[0] for m in means]
    x = np.arange(len(series))
    ax.bar(x, means, yerr=sems, capsize=3, color=colors, width=0.6, alpha=0.9,
           error_kw={"linewidth": 1})
    ax.axhline(0, color="black", linewidth=0.7)
    for i, (m, r) in enumerate(zip(means, recovery)):
        label = "100%\n(ref.)" if i == 0 else f"{r:.0f}%\nrecov."
        ax.annotate(label, (i, m), xytext=(0, -20), textcoords="offset points",
                    ha="center", fontsize=6.0, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(min(means) - 3.2, 1.2)
axv.set_ylabel("Mean MSE $\\Delta$ vs. Only (%)")
fig.suptitle("Signal-decomposition ablations, 30-fold", fontsize=8.8, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ablation30.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/fig_ablation30.pdf")

for label, vals, _ in vol_series + speed_series:
    print(f"{label!r:26s} mean={np.mean(vals):+.2f}%  min={min(vals):+.2f}%  max={max(vals):+.2f}%")
