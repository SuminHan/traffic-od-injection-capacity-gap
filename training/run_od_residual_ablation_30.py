"""
Runs the lightweight model with the OD-residual (de-climatologized) signal built by
build_od_residual_signal.py: real routed OD minus its own train-window climatology, on the same 5
fold boundaries as Traffic-Only, Traffic-Hybrid, the naive-routing ablation and the climatology
ablation -- so all four are directly comparable.

Where the climatology ablation asks "how much of the benefit is calendar-shaped?", this asks the
complementary question directly rather than by subtraction: injected with NO calendar content of
its own (by construction, its own per-bucket mean is ~0), does the residual alone recover a
meaningful fraction of Hybrid's improvement?

Interpretation:
  - residual alone recovers a comparable fraction of Hybrid's benefit -> genuine day-specific OD
    content is doing real work, independent of and additional to calendar structure.
  - residual alone recovers little/nothing -> real OD's advantage over climatology (folds 0, 4)
    was itself calendar-adjacent structure the coarse 168-bucket lookup missed, not anomaly-
    specific demand information; the "genuine OD value" claim weakens further.
"""
import json
import subprocess

GTS = "/home/ncrc/work/gts"
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"

existing = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
hyb = {e["fold_start"]: e for e in existing if e["model"] == "traffic_hybrid_routed"}
only = {e["fold_start"]: e for e in existing if e["model"] == "traffic_only"}
clim = json.load(open(f"{GTS}/climatology_ablation_results.json"))
clim_by_fold = {e["fold_start"]: e for e in clim}

import os
results = json.load(open(f"{GTS}/od_residual_ablation_results.json")) if os.path.exists(f"{GTS}/od_residual_ablation_results.json") else []
done_folds = {e["fold_start"] for e in results}
for fold in range(30):
    if fold in done_folds:
        continue
    a = hyb[fold]["args"]
    out = f"tfold{fold}_hybrid_residual_ext"
    cmd = [PY, "-u", "train_traffic_model.py",
           "--dataset_suffix", "_ext", "--routed_signal_suffix", f"_residual_fold{fold}_ext",
           "--use_od_injection", "1",
           "--train_lo", str(a["train_lo"]), "--train_hi", str(a["train_hi"]),
           "--val_lo", str(a["val_lo"]), "--val_hi", str(a["val_hi"]),
           "--test_lo", str(a["test_lo"]), "--test_hi", str(a["test_hi"]),
           "--hidden", str(a["hidden"]), "--epochs", "40", "--patience", "8", "--out", out]
    print(f"\n=== fold {fold} OD residual (de-climatologized) ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=GTS, check=True)
    summ = json.load(open(f"{GTS}/{out}/summary.json"))
    results.append({"fold_start": fold, "model": "traffic_od_residual_signal", "out": out, **summ})
    json.dump(results, open(f"{GTS}/od_residual_ablation_results.json", "w"), indent=2)

results_by_fold = {e['fold_start']: e for e in results}
print("\n\n=== SUMMARY: Only vs Hybrid(real) vs Climatology vs Residual(real-climatology), folds 0-4 ===")
for fold in range(30):
    o = only[fold]["test_mse"]
    h = hyb[fold]["test_mse"]
    c = clim_by_fold[fold]["test_mse"]
    r = results_by_fold[fold]["test_mse"]
    print(f"fold {fold}: Only={o:.1f}  Hybrid={h:.1f} ({(h-o)/o*100:+.2f}%)  "
          f"Climatology={c:.1f} ({(c-o)/o*100:+.2f}%)  "
          f"Residual={r:.1f} ({(r-o)/o*100:+.2f}%)")

print("\nOD_RESIDUAL_ABLATION_30_ALL_DONE")
