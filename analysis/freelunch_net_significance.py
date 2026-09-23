"""Paired significance test of the free-lunch-corrected (net) selective-injection effect.
Per fold f: raw_f = selective-injection MSE change vs plain (selective_injection_*_fold_results.csv);
fl_f = seed-only selector's change vs the better whole checkpoint (seed_control_*_fold_results.csv).
net_f = raw_f - fl_f; one-sample t-test and Wilcoxon signed-rank on net_f (H0: mean 0), one per
(architecture, task), plus Benjamini-Hochberg across all combinations."""
import numpy as np, pandas as pd
from scipy import stats

ARCHS = ["gman", "stid_fixed", "stgcn", "gts", "pdformer", "dcrnn", "staeformer", "mtgnn", "gwnet", "agcrn", "ours"]
rows = []
for m in ARCHS:
    for t in ["volume", "speed"]:
        s = pd.read_csv(f"selective_injection_{m}_{t}_fold_results.csv").set_index("fold")
        c = pd.read_csv(f"seed_control_{m}_{t}_fold_results.csv").set_index("fold")
        f = s.index.intersection(c.index)
        raw = (s.loc[f, "mse_selective"] - s.loc[f, "mse_plain"]) / s.loc[f, "mse_plain"] * 100
        fl = (c.loc[f, "mse_selective"] - c.loc[f, "mse_globalpick"]) / c.loc[f, "mse_globalpick"] * 100
        net = raw - fl
        rows.append({"model": m, "task": t, "n": len(f), "raw": raw.mean(), "free_lunch": fl.mean(),
                     "net": net.mean(), "net_wins": int((net < 0).sum()),
                     "p_t": stats.ttest_1samp(net, 0)[1], "p_wilcoxon": stats.wilcoxon(net)[1]})
df = pd.DataFrame(rows)
p = df.p_t.values; o = np.argsort(p); q = np.empty_like(p)
q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
df["q_bh"] = np.minimum(q, 1)
df.to_csv("freelunch_net_significance.csv", index=False)
pd.set_option("display.width", 200)
print(df.sort_values("net").round(4).to_string(index=False))
