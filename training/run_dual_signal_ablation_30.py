"""
Injects climatology and its residual (real OD minus climatology) as two SEPARATELY-normalized
channels, instead of their raw sum (which is exactly what the real routed signal is, and what
Hybrid already injects as one pre-summed, jointly-normalized channel).

Motivation. run_od_residual_ablation.py found that the residual alone -- by construction containing
zero calendar information -- recovers 78.6% of Hybrid's average improvement, almost matching
climatology's 82%. But injected together as the raw sum (i.e. Hybrid), the improvement is only
modestly larger than either piece alone, and worse than the residual alone in one fold (fold 1).
The likely cause, confirmed in the loading code: each injected signal is z-scored by its OWN
train-window std before injection. Climatology explains 99.4% of the real signal's variance
(corr=0.997), so the residual occupies only ~0.6% of Hybrid's raw variance -- when Hybrid is
normalized as one combined signal, the residual's real information becomes a small ripple on a
much larger calendar-scale wave, plausibly hard for a fixed-capacity gate to extract. Injected as
its own file, it gets rescaled to unit variance like everything else.

This ablation tests whether presenting both components on equal normalized footing -- as two
channels, gated jointly via od_channels=2 -- recovers more than the sum (Hybrid) does, using the
existing 2-channel fusion path (already used for outflow/inflow) via train_traffic_model_dualsignal.py
(a data-loading patch only, diffed against the original).

Same 5 fold boundaries as Only/Hybrid/Climatology/Residual, so all five are directly comparable.

Interpretation:
  - Dual-channel beats Hybrid -> the paper's headline injection mechanism is leaving value on the
    table by summing components with very different scales before normalizing; a cheap fix (no
    retraining of the OD forecaster, just a 2-channel gate) could raise the headline result.
  - Dual-channel ~= Hybrid -> the joint-normalization scale effect isn't actually costing much in
    practice; the separate-channel result was informative about WHY residual-alone looks strong,
    but doesn't change what to inject in the main pipeline.
"""
import json
import subprocess

GTS = "/home/ncrc/work/gts"
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"

existing = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
hyb = {e["fold_start"]: e for e in existing if e["model"] == "traffic_hybrid_routed"}
only = {e["fold_start"]: e for e in existing if e["model"] == "traffic_only"}
clim = {e["fold_start"]: e for e in json.load(open(f"{GTS}/climatology_ablation_results.json"))}
resid = {e["fold_start"]: e for e in json.load(open(f"{GTS}/od_residual_ablation_results.json"))}

import os
results = json.load(open(f"{GTS}/dual_signal_ablation_results.json")) if os.path.exists(f"{GTS}/dual_signal_ablation_results.json") else []
done_folds = {e["fold_start"] for e in results}
for fold in range(30):
    if fold in done_folds:
        continue
    a = hyb[fold]["args"]
    out = f"tfold{fold}_hybrid_dualsignal_ext"
    cmd = [PY, "-u", "train_traffic_model_dualsignal.py",
           "--dataset_suffix", "_ext", "--use_od_injection", "1", "--od_channels", "2",
           "--dual_signal_a_suffix", f"_climatology_fold{fold}_ext",
           "--dual_signal_b_suffix", f"_residual_fold{fold}_ext",
           "--train_lo", str(a["train_lo"]), "--train_hi", str(a["train_hi"]),
           "--val_lo", str(a["val_lo"]), "--val_hi", str(a["val_hi"]),
           "--test_lo", str(a["test_lo"]), "--test_hi", str(a["test_hi"]),
           "--hidden", str(a["hidden"]), "--epochs", "40", "--patience", "8", "--out", out]
    print(f"\n=== fold {fold} dual-signal (climatology + residual, separately normalized) ===\n"
          f"{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=GTS, check=True)
    summ = json.load(open(f"{GTS}/{out}/summary.json"))
    results.append({"fold_start": fold, "model": "traffic_dual_signal", "out": out, **summ})
    json.dump(results, open(f"{GTS}/dual_signal_ablation_results.json", "w"), indent=2)

results_by_fold = {e['fold_start']: e for e in results}
print("\n\n=== SUMMARY: Hybrid(sum) vs Climatology vs Residual vs Dual-channel(same two parts, "
      "separately normalized), folds 0-4 ===")
for fold in range(30):
    o = only[fold]["test_mse"]
    h = hyb[fold]["test_mse"]
    c = clim[fold]["test_mse"]
    r = resid[fold]["test_mse"]
    d = results_by_fold[fold]["test_mse"]
    print(f"fold {fold}: Hybrid={(h-o)/o*100:+.2f}%  Climatology={(c-o)/o*100:+.2f}%  "
          f"Residual={(r-o)/o*100:+.2f}%  Dual={(d-o)/o*100:+.2f}%")

print("\nDUAL_SIGNAL_ABLATION_30_ALL_DONE")
