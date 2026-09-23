"""Is selective injection just ensembling? For every (architecture, task) and fold, compare on the
test month:
  plain            -- plain checkpoint (seed 0)
  selective        -- per-sensor validation-gated choice between plain and injected (paper Table 8)
  avg(plain, inj)  -- naive average of the plain and injected checkpoints' predictions
  avg(plain, s1)   -- naive average of two plain checkpoints differing only in seed: the gain any
                      two-model ensemble gets, with no OD signal at all
All MSE changes are relative to plain. Uses existing prediction files only (no training).
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

GTS = "/home/ncrc/work/gts"
KEY = ["window_start_hour", "decode_step", "sensor_idx"]


def load(d):
    df = pd.read_csv(f"{GTS}/{d}/test_predictions.csv").sort_values(KEY).reset_index(drop=True)
    y = df[[c for c in df.columns if c.startswith("true_")][0]].values
    p = df[[c for c in df.columns if c.startswith("pred_")][0]].values
    return df.sensor_idx.values, y, p


def dirs(model, task):
    if model == "ours":
        rf = "multi_fold_traffic_results_ext.json" if task == "volume" else "multi_fold_speed_results_ext.json"
        pk, ok = ("traffic_only", "traffic_hybrid_routed") if task == "volume" else ("speed_only", "speed_hybrid_routed")
        by = {}
        for e in json.load(open(f"{GTS}/{rf}")):
            by.setdefault(e["fold_start"], {})[e["model"]] = e["out"]
        return {f: (v[pk], v[ok], v[pk] + "_s1") for f, v in by.items() if pk in v and ok in v}
    for rf in (f"multi_fold_baseline_{model}_{task}_results_ext30.json", f"multi_fold_{model}_{task}_results_ext30.json"):
        if os.path.exists(f"{GTS}/{rf}"):
            break
    by = {}
    for e in json.load(open(f"{GTS}/{rf}")):
        by.setdefault(e["fold_start"], {})[e["variant"]] = e["out"]
    out = {}
    for f, v in by.items():
        odk = next((k for k in v if k != "plain"), None)
        if "plain" in v and odk:
            out[f] = (v["plain"], v[odk], f"seedctl_{model}_{task}_fold{f}_s1_ext30")
    return out


ARCHS = ["stid_fixed", "gman", "stgcn", "pdformer", "staeformer", "mtgnn", "dcrnn", "gts", "gwnet", "agcrn", "ours"]
summary = []
for m in ARCHS:
    for t in ["volume", "speed"]:
        sel = pd.read_csv(f"{GTS}/selective_injection_{m}_{t}_fold_results.csv").set_index("fold")
        rows = []
        for f, (dp, di, ds) in sorted(dirs(m, t).items()):
            if f not in sel.index or not os.path.exists(f"{GTS}/{ds}/test_predictions.csv"):
                continue
            s0, y, pp = load(dp); s1, y1, pi = load(di); s2, y2, ps = load(ds)
            assert (s0 == s1).all() and (s0 == s2).all() and np.allclose(y, y1)
            mp = np.mean((y - pp) ** 2)
            rows.append({"fold": f, "mse_plain": mp,
                         "mse_selective": sel.loc[f, "mse_selective"] * mp / sel.loc[f, "mse_plain"],
                         "mse_avg_inj": np.mean((y - (pp + pi) / 2) ** 2),
                         "mse_avg_seed": np.mean((y - (pp + ps) / 2) ** 2)})
        df = pd.DataFrame(rows)
        df.to_csv(f"{GTS}/ensemble_check_{m}_{t}_fold_results.csv", index=False)
        pct = lambda c: (df[c] - df.mse_plain) / df.mse_plain * 100
        r = {"model": m, "task": t, "n": len(df)}
        for c in ["selective", "avg_inj", "avg_seed"]:
            r[c] = pct(f"mse_{c}").mean()
        r["sel_vs_avginj_p"] = stats.ttest_rel(df.mse_selective, df.mse_avg_inj)[1]
        r["sel_beats_avginj_folds"] = int((df.mse_selective < df.mse_avg_inj).sum())
        r["avginj_vs_avgseed_p"] = stats.ttest_rel(df.mse_avg_inj, df.mse_avg_seed)[1]
        summary.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(summary).to_csv(f"{GTS}/ensemble_check_summary.csv", index=False)
print("ENSEMBLE_CHECK_DONE")
