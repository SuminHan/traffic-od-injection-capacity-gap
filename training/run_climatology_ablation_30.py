"""
Runs the lightweight model with the per-fold, train-window-only (day-of-week, hour-of-day)
climatology signal built by build_climatology_signal_perfold.py, on the SAME 5 fold boundaries
already used for Traffic-Only, Traffic-Hybrid (real routed), and the naive-routing ablation --
so this is a direct, matched, real fold-based comparison. Answers the calendar-confound question
flagged for the published-baseline mechanism-(i) runs (Section 4.3's shuffled-OD note) but never
checked for this paper's own headline result: does the headline -7 to -8% MSE improvement survive
when the injected signal carries ONLY calendar information, no real OD forecast content?

Interpretation:
  - climatology recovers most of Hybrid's improvement -> headline result is largely a
    calendar-embedding effect, not genuine OD-signal value. Requires reframing the paper.
  - climatology recovers little/none of it -> OD's genuine (non-calendar) contribution is
    quantitatively demonstrated, which strengthens the paper's central claim.
"""
import json, subprocess

GTS = "/home/ncrc/work/gts"  # 30-fold extension: path only, else identical
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"

existing = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
hyb = {e["fold_start"]: e for e in existing if e["model"] == "traffic_hybrid_routed"}
only = {e["fold_start"]: e for e in existing if e["model"] == "traffic_only"}

import os
results = json.load(open(f"{GTS}/climatology_ablation_results.json")) if os.path.exists(f"{GTS}/climatology_ablation_results.json") else []
done_folds = {e["fold_start"] for e in results}
for fold in range(30):
    if fold in done_folds:
        continue
    a = hyb[fold]["args"]
    out = f"tfold{fold}_hybrid_climatology_ext"
    cmd = [PY, "-u", "train_traffic_model.py",
           "--dataset_suffix", "_ext", "--routed_signal_suffix", f"_climatology_fold{fold}_ext",
           "--use_od_injection", "1",
           "--train_lo", str(a["train_lo"]), "--train_hi", str(a["train_hi"]),
           "--val_lo", str(a["val_lo"]), "--val_hi", str(a["val_hi"]),
           "--test_lo", str(a["test_lo"]), "--test_hi", str(a["test_hi"]),
           "--hidden", str(a["hidden"]), "--epochs", "40", "--patience", "8", "--out", out]
    print(f"\n=== fold {fold} climatology ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=GTS, check=True)
    summ = json.load(open(f"{GTS}/{out}/summary.json"))
    results.append({"fold_start": fold, "model": "traffic_climatology_only_signal", "out": out, **summ})
    json.dump(results, open(f"{GTS}/climatology_ablation_results.json", "w"), indent=2)

results_by_fold = {e['fold_start']: e for e in results}
print("\n\n=== SUMMARY: Only vs Hybrid(real routed) vs Climatology(dow,hour lookup, train-only), folds 0-4 ===")
for fold in range(30):
    o_mse = only[fold]["test_mse"]
    h_mse = hyb[fold]["test_mse"]
    c_mse = results_by_fold[fold]["test_mse"]
    print(f"fold {fold}: Only={o_mse:.1f}  Hybrid(routed)={h_mse:.1f} ({(h_mse-o_mse)/o_mse*100:+.2f}%)  "
          f"Climatology={c_mse:.1f} ({(c_mse-o_mse)/o_mse*100:+.2f}%)")

print("\nCLIMATOLOGY_ABLATION_30_ALL_DONE")
