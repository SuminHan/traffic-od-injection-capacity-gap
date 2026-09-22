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

series = [
    ("Hybrid\n(real signal)", pct(hyb), "#1a1a1a"),
    ("Climatology\nonly", pct(clim), "#2f6ea3"),
    ("Residual\nonly", pct(resid), "#c8781f"),
    ("Dual-\nchannel", pct(dual), "#7a4fa3"),
]

means = [float(np.mean(v)) for _, v, _ in series]
sems = [float(np.std(v, ddof=1) / np.sqrt(len(v))) for _, v, _ in series]
labels = [s[0] for s in series]
colors = [s[2] for s in series]
recovery = [100.0 * m / means[0] for m in means]

fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = np.arange(len(series))
ax.bar(x, means, yerr=sems, capsize=3, color=colors, width=0.6, alpha=0.9,
       error_kw={"linewidth": 1})
ax.axhline(0, color="black", linewidth=0.7)
for i, (m, r) in enumerate(zip(means, recovery)):
    label = "100% (ref.)" if i == 0 else f"{r:.0f}% recovered"
    ax.annotate(label, (i, m), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=6.6, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Mean MSE $\\Delta$ vs. Only (%)")
ax.set_title("Signal-decomposition ablations (volume, 30-fold)", fontsize=8.5)
ax.set_ylim(min(means) - 2.3, 1.2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ablation30.pdf")
plt.close(fig)
print("wrote", f"{OUT}/fig_ablation30.pdf")

for label, vals, _ in series:
    print(f"{label!r:26s} mean={np.mean(vals):+.2f}%  min={min(vals):+.2f}%  max={max(vals):+.2f}%")
