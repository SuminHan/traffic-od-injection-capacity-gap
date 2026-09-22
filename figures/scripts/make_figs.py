"""Generate real-data figures for the TKDE paper draft, in a clean serif/mono style suitable for
a two-column IEEE layout."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GTS = "/home/ncrc/work/gts"
OUT = f"{GTS}/paper_tkde/figs"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                      "grid.alpha": 0.3, "axes.axisbelow": True})

# ---------------------------------------------------------------------------
# Fig 1: per-fold MSE % improvement, volume 30-fold (extended range), our own model
# ---------------------------------------------------------------------------
d = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
by_fold = {}
for r in d:
    by_fold.setdefault(r["fold_start"], {})[r["model"]] = r
folds = sorted(by_fold.keys())
pct = []
for fs in folds:
    o = by_fold[fs]["traffic_only"]["test_mse"]
    h = by_fold[fs]["traffic_hybrid_routed"]["test_mse"]
    pct.append((h - o) / o * 100)

fig, ax = plt.subplots(figsize=(3.4, 2.1))
colors = ["#c0392b" if p > 0 else "#2f7d52" for p in pct]
ax.bar(folds, pct, color=colors, width=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Fold index (rolling-origin, 1-month stride)")
ax.set_ylabel("MSE $\\Delta$ (%)")
ax.set_title("Volume, 30-fold: Hybrid vs. Only", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_volume_perfold.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: generalization bar chart across architectures/mechanisms
# ---------------------------------------------------------------------------
rows = [
    ("Ours-vol.", -7.17, 29, 30),
    ("DCRNN-vol.", 0.95, 14, 30),
    ("DCRNN-speed", 1.76, 11, 30),
    ("DCRNN-oracle", 1.31, 4, 9),
    ("GWNET-vol.", 25.45, 1, 30),
    ("GWNET-speed", 3.77, 5, 30),
    ("GWNET-adj.", 1.17, 10, 23),
    ("GTS-vol.", 0.71, 3, 7),
    ("STAEf.-vol.", 0.74, 14, 30),
    ("STAEf.-speed", 4.71, 1, 30),
    ("GMAN-vol.", -7.53, 22, 30),
    ("GMAN-speed", 0.19, 14, 30),
]
labels = [r[0] for r in rows]
means = [r[1] for r in rows]
wins = [f"{r[2]}/{r[3]}" for r in rows]
colors2 = ["#2f7d52" if m < 0 else "#9a4a3a" for m in means]

fig, ax = plt.subplots(figsize=(3.4, 2.9))
x = np.arange(len(labels))
bars = ax.bar(x, means, color=colors2, width=0.65)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6, rotation=55, ha="right")
ax.set_ylabel("Mean MSE $\\Delta$ (%)")
ax.set_title("OD injection: mean effect by model & mechanism", fontsize=7.3)
for xi, m, w in zip(x, means, wins):
    ax.text(xi, m + (0.8 if m >= 0 else -2.0), w, ha="center", fontsize=5.5, color="#444")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_generalization.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: anomaly-quintile diagnostic
# ---------------------------------------------------------------------------
quint = [8.90, 7.48, 5.15, 4.42, 2.66]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.plot(range(5), quint, marker="o", color="#3a6ea5", linewidth=1.8, markersize=5)
ax.set_xticks(range(5))
ax.set_xticklabels(["Q0\n(typical)", "Q1", "Q2", "Q3", "Q4\n(anomalous)"], fontsize=7)
ax.set_ylabel("Mean error reduction\n(Only $-$ Hybrid)")
ax.set_title("Injection benefit vs. OD-anomaly quintile", fontsize=8.5)
ax.set_ylim(0, 10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_diagnostic.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: capacity sweep -- gap to DCRNN vs. hidden size, injection effect overlay
# ---------------------------------------------------------------------------
hiddens = [48, 64, 96]
gap_pct = [23.4, 14.9, -0.3]
inj_pct = [-11.4, -11.2, -1.7]
fig, ax1 = plt.subplots(figsize=(3.4, 2.1))
ax1.plot(hiddens, gap_pct, marker="s", color="#9a4a3a", label="Gap to DCRNN (\\%)")
ax1.set_xlabel("Hidden size")
ax1.set_ylabel("Remaining gap to\nDCRNN (\\%)", color="#9a4a3a")
ax1.tick_params(axis="y", labelcolor="#9a4a3a")
ax2 = ax1.twinx()
ax2.plot(hiddens, inj_pct, marker="o", color="#2f7d52", label="OD injection effect (\\%)")
ax2.set_ylabel("OD injection\neffect (\\%)", color="#2f7d52")
ax2.tick_params(axis="y", labelcolor="#2f7d52")
ax1.set_xticks(hiddens)
ax1.set_title("Capacity sweep: gap closes and\ninjection effect collapses together", fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_capacity.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: qualitative case study -- (a) one sensor's true/Only/Hybrid time series over a
# representative week, (b) hour-of-day mean absolute-error reduction (all sensors, fold 0)
# ---------------------------------------------------------------------------
import pandas as pd
wk = pd.read_csv(f"{GTS}/case_study_week.csv", parse_dates=["ts"])
hod_df = pd.read_csv(f"{GTS}/case_study_hod.csv")
hod = hod_df.set_index("hour")["0"]

fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.4, 3.6), gridspec_kw={"height_ratios": [1.15, 1]})

# Full 7-day week, not a cherry-picked window: an earlier version zoomed to the single 2-day
# span containing the week's largest favorable gap, which -- honestly, on inspection -- makes the
# effect look cleaner and more one-sided than it is (only 90/168 hours in the full week favor
# Hybrid; several unshown hours favor Only by a larger margin than the favorable peaks shown).
# Plotting |error| instead of raw predicted value also fixes the earlier readability problem
# (raw Only/Hybrid/truth lines visually overlap almost everywhere at this scale) without needing
# to crop the window: the two error curves and their sign relative to each other are legible
# across the whole week, cherry-picking nothing.
wk["err_only"] = (wk["true_volume"] - wk["pred_volume_only"]).abs()
wk["err_hybrid"] = (wk["true_volume"] - wk["pred_volume_hybrid"]).abs()
axa.fill_between(wk["ts"], wk["err_only"], wk["err_hybrid"],
                  where=(wk["err_hybrid"] <= wk["err_only"]), color="#2f7d52", alpha=0.25,
                  interpolate=True, label="Hybrid better")
axa.fill_between(wk["ts"], wk["err_only"], wk["err_hybrid"],
                  where=(wk["err_hybrid"] > wk["err_only"]), color="#9a4a3a", alpha=0.25,
                  interpolate=True, label="Only better")
axa.plot(wk["ts"], wk["err_only"], color="#9a4a3a", linewidth=0.9, label="Traffic-Only |error|")
axa.plot(wk["ts"], wk["err_hybrid"], color="#2f7d52", linewidth=0.9, label="Traffic-Hybrid |error|")
axa.set_ylabel("Abs. error\n(vehicles/h)", fontsize=7.5)
axa.set_title("Sensor C-13, full test week, absolute error (fold 0)", fontsize=8)
axa.tick_params(axis="x", labelrotation=30, labelsize=6)
axa.legend(fontsize=5.4, loc="upper left", frameon=False, ncol=2)

axb.bar(hod.index.astype(int), hod.values, color="#3a6ea5", width=0.7)
axb.set_xlabel("Hour of day", fontsize=7.5)
axb.set_ylabel("Mean |error|\nreduction (veh.)", fontsize=7.5)
axb.set_title("When injection helps: hour-of-day pattern\n(all sensors, fold 0)", fontsize=8)
axb.set_xticks(range(0, 24, 3))

fig.tight_layout()
fig.savefig(f"{OUT}/fig_case_study.pdf")
plt.close(fig)

print("wrote 5 figures to", OUT)
