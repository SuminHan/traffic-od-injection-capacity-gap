"""
Properly validates the paper's currently-unquantified routing-necessity claim (Section 4.2:
"a naive same-district outflow value pasted onto the nearest sensor... carries no useful
signal", cited only as "early, unsuccessful pilot experiments"). Runs the lightweight model with
the naive (nearest-centroid, no road-network routing) injection signal built by
build_naive_signal_volume.py, on the SAME 5 fold boundaries already used for Traffic-Only and
Traffic-Hybrid (real routed) in multi_fold_traffic_results_ext.json -- so this is a direct,
matched, real fold-based comparison, not an anecdote. Matches the Section 7 ablation's own
"5 folds" convention for this kind of secondary check.
"""
import json, subprocess

GTS = "/home/ncrc/work/gts"  # 30-fold extension: path only, else identical
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"

existing = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
hyb = {e["fold_start"]: e for e in existing if e["model"] == "traffic_hybrid_routed"}
only = {e["fold_start"]: e for e in existing if e["model"] == "traffic_only"}

import os
results = json.load(open(f"{GTS}/naive_signal_ablation_results.json")) if os.path.exists(f"{GTS}/naive_signal_ablation_results.json") else []
done_folds = {e["fold_start"] for e in results}
for fold in range(30):
    if fold in done_folds:
        continue
    a = hyb[fold]["args"]
    out = f"tfold{fold}_hybrid_naive_ext"
    cmd = [PY, "-u", "train_traffic_model.py",
           "--dataset_suffix", "_ext", "--routed_signal_suffix", "_naive_ext",
           "--use_od_injection", "1",
           "--train_lo", str(a["train_lo"]), "--train_hi", str(a["train_hi"]),
           "--val_lo", str(a["val_lo"]), "--val_hi", str(a["val_hi"]),
           "--test_lo", str(a["test_lo"]), "--test_hi", str(a["test_hi"]),
           "--hidden", str(a["hidden"]), "--epochs", "40", "--patience", "8", "--out", out]
    print(f"\n=== fold {fold} naive ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=GTS, check=True)
    summ = json.load(open(f"{GTS}/{out}/summary.json"))
    results.append({"fold_start": fold, "model": "traffic_naive_unrouted", "out": out, **summ})
    json.dump(results, open(f"{GTS}/naive_signal_ablation_results.json", "w"), indent=2)

results_by_fold = {e['fold_start']: e for e in results}
print("\n\n=== SUMMARY: Only vs Hybrid(real routed) vs Naive(nearest-centroid, no routing), folds 0-4 ===")
for fold in range(30):
    o_mse = only[fold]["test_mse"]
    h_mse = hyb[fold]["test_mse"]
    n_mse = results_by_fold[fold]["test_mse"]
    print(f"fold {fold}: Only={o_mse:.1f}  Hybrid(routed)={h_mse:.1f} ({(h_mse-o_mse)/o_mse*100:+.2f}%)  "
          f"Naive(unrouted)={n_mse:.1f} ({(n_mse-o_mse)/o_mse*100:+.2f}%)")

print("\nNAIVE_SIGNAL_ABLATION_30_ALL_DONE")
