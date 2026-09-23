"""Guide-only illustration (not part of the paper itself): a dumbbell chart showing each model's
absolute RMSE moving from plain (open circle) to injected (filled circle) performance, connected
by an arrow -- literally "before vs after" rather than a normalized %-change. Split into two
subplots (volume / speed) since the two tasks' RMSE live on completely different scales
(~300-500 for volume vs ~3-4 for speed) and can't share one axis.

Data: Table 9's own numbers (see main.tex) -- RMSE (inj.) is validation-gated selective injection
for all eleven models, including Ours (Section 7.1/Table 9), with no uniform-only exceptions left.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/figpng"

plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# (name, params_label, rmse_plain_vol, rmse_inj_vol, rmse_plain_spd, rmse_inj_spd, uniform_only)
DATA = [
    ("STID",        "31-35k",  360.7, 342.5, 3.30, 3.25, False),
    ("GMAN",        "55-59k",  504.8, 468.5, 3.46, 3.36, False),
    ("Ours",        "62k",     410.6, 390.7, 4.03, 3.84, False),
    ("STGCN",       "62-67k",  382.5, 367.6, 3.33, 3.26, False),
    ("PDFormer",    "95-157k", 367.8, 355.6, 3.20, 3.17, False),
    ("STAEformer",  "160-217k",357.1, 349.2, 3.18, 3.16, False),
    ("MTGNN",       "335-349k",350.1, 347.6, 3.19, 3.16, False),
    ("DCRNN",       "382-386k",355.8, 348.9, 3.21, 3.18, False),
    ("GTS",         "395-422k",357.0, 347.9, 3.21, 3.19, False),
    ("GWNET",       "673-683k",367.8, 366.8, 3.21, 3.18, False),
    ("AGCRN",       "774-777k",396.3, 392.4, 3.21, 3.18, False),
]

fig, (axv, axs) = plt.subplots(1, 2, figsize=(11, 5.5))

for ax, plain_i, inj_i, title, ylabel in [
    (axv, 2, 3, "Volume: RMSE, plain → injected", "RMSE (vehicles/hour)"),
    (axs, 4, 5, "Speed: RMSE, plain → injected", "RMSE (km/h)"),
]:
    y = np.arange(len(DATA))[::-1]
    for i, row in enumerate(DATA):
        name, params, *_ = row
        plain_v, inj_v = row[plain_i], row[inj_i]
        uniform_only = row[6]
        improved = inj_v < plain_v
        color = "#2f7d52" if improved else "#b0402f"
        yi = y[i]
        ax.annotate("", xy=(inj_v, yi), xytext=(plain_v, yi),
                    arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.8, lw=1.8,
                                     shrinkA=6, shrinkB=6, mutation_scale=14), zorder=2)
        ax.scatter([plain_v], [yi], s=70, facecolors="white", edgecolors="#333333",
                   linewidths=1.4, zorder=3)
        ax.scatter([inj_v], [yi], s=70, facecolors=color, edgecolors=color, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}{'  (uniform)' if r[6] else ''}" for r in DATA])
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.7, len(DATA) - 0.3)

# Legend (shared, placed once)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor="#333333", markersize=9, label="plain (no injection)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#2f7d52",
           markeredgecolor="#2f7d52", markersize=9, label="injected, improved"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#b0402f",
           markeredgecolor="#b0402f", markersize=9, label="injected, worse"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle("How much does injection actually move each model's absolute error?", fontsize=13, y=1.02)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f"{OUT}/fig_dumbbell_guide.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/fig_dumbbell_guide.png")
