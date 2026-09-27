"""Pilot: does warm-starting a large model's OD-injection pathway from an already-trained small
model (instead of learning it from scratch, jointly, inside the large model) let a large model
benefit from injection where standard from-scratch training does not?

Motivation (user's alternative hypothesis to the paper's capacity-gap/redundancy account): maybe
large models don't fail to benefit from injection because the signal is redundant, but because
end-to-end joint optimization from random init makes it hard to *find* how to use it. STID at
hidden=256 is the paper's own reversal case (uniform injection: -9.02% at hidden=64 -> +3.37% at
hidden=256, Table VIII). We already know hidden=64's ODInjectionWrapper learned to use the signal
well. This script keeps the backbone at hidden=256 (as in the paper) but gives it a
BottleneckODInjectionWrapper (models_baselines.py) whose injection pathway runs at a fixed width
of 64, either (a) randomly initialized (control -- isolates whether just shrinking the pathway's
width helps, independent of warm start) or (b) warm-started from hidden=64's already-trained
pathway (the actual test), then fine-tuned end-to-end on the full backbone.

5 folds only (0-4), full 40-epoch/patience-8 training per fold as normal -- a pilot in fold count,
not training rigor, matching this project's existing convention for pre-registered quick checks
(e.g. the fusion-architecture ablation, Appendix A) before committing to a full 30-fold sweep.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

GTS = "/home/ncrc/work/gts"
PY = sys.executable
FOLDS = [0, 1, 2, 3, 4]

jobs = []
for fold in FOLDS:
    a = json.load(open(f"{GTS}/scap_stid_fixed_volume_h256_s0_od_simple_fold{fold}_ext30/summary.json"))["args"]
    small_ckpt = f"{GTS}/pmfold{fold}_stid_fixed_volume_od_simple_ext30/best.pt"
    common = ["--model", "stid_fixed", "--task", "volume", "--dataset_suffix", "_ext",
              "--use_od_injection", "1", "--fusion_style", "simple", "--od_graph", "0",
              "--node_se", "none", "--routing_source", "single", "--shuffle_od", "0",
              "--od_source", "forecast", "--od_signal_mode", "raw",
              "--hidden", "256", "--p_len", "12", "--q_len", "12", "--epochs", "40", "--lr", "0.002",
              "--batch_size", "32", "--patience", "8",
              "--train_lo", str(a["train_lo"]), "--train_hi", str(a["train_hi"]),
              "--val_lo", str(a["val_lo"]), "--val_hi", str(a["val_hi"]),
              "--test_lo", str(a["test_lo"]), "--test_hi", str(a["test_hi"]), "--seed", "0",
              "--inject_bottleneck_dim", "64"]
    for cond, extra in [("random", []),
                        ("warmstart", ["--inject_warmstart_from", small_ckpt])]:
        out = f"pilot_bneck_{cond}_fold{fold}"
        if os.path.exists(f"{GTS}/{out}/summary.json"):
            continue
        jobs.append((out, [PY, "-u", "train_baseline_model_extra2.py"] + common + extra + ["--out", out]))

print(f"{len(jobs)} jobs to run", flush=True)


def run(job):
    out, cmd = job
    with open(f"{GTS}/logs_{out}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=GTS, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{out}: rc={rc}", flush=True)
    return rc


with ThreadPoolExecutor(3) as ex:
    rcs = list(ex.map(run, jobs))

print("PILOT_BOTTLENECK_ALL_DONE", "failures:", sum(r != 0 for r in rcs), flush=True)

# summarize
rows = []
for fold in FOLDS:
    plain_mse = json.load(open(f"{GTS}/scap_stid_fixed_volume_h256_s0_plain_fold{fold}_ext30/summary.json"))["test_mse"]
    full256_mse = json.load(open(f"{GTS}/scap_stid_fixed_volume_h256_s0_od_simple_fold{fold}_ext30/summary.json"))["test_mse"]
    row = {"fold": fold, "plain_mse": plain_mse,
           "full256_injected_mse": full256_mse,
           "full256_pct": (full256_mse - plain_mse) / plain_mse * 100}
    for cond in ("random", "warmstart"):
        p = f"{GTS}/pilot_bneck_{cond}_fold{fold}/summary.json"
        if os.path.exists(p):
            mse = json.load(open(p))["test_mse"]
            row[f"{cond}_bneck_mse"] = mse
            row[f"{cond}_bneck_pct"] = (mse - plain_mse) / plain_mse * 100
    rows.append(row)

import csv
with open(f"{GTS}/pilot_bottleneck_warmstart_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

means = {}
for key in ("full256_pct", "random_bneck_pct", "warmstart_bneck_pct"):
    vals = [r[key] for r in rows if key in r]
    if vals:
        means[key] = sum(vals) / len(vals)
print("\nMean %% MSE change vs plain, over folds 0-4:")
for k, v in means.items():
    print(f"  {k}: {v:+.2f}%")
print("\nwrote pilot_bottleneck_warmstart_summary.csv")
