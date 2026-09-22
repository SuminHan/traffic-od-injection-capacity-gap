"""
Second-seed plain-checkpoint training for the seed-control ("free lunch") correction, extended to
the three architectures never given this treatment (PDFormer, MTGNN, AGCRN) -- needed for Table 13
(freelunch) and Table 18 (adaptive threshold) to cover the same ten architectures Table 9/12 now
do. Mirrors run_extra_models_ext30_sweep.py's fold construction exactly (12mo train/1mo val/1mo
test, stride 1, 30 folds over the _ext 1308-day range) but trains only the "plain" (no OD
injection) variant, at --seed 1 instead of the default --seed 0, writing to
seedctl_<model>_<task>_fold<N>_s1_ext30 to match the naming convention of the original five
architectures' seed-control checkpoints. Resumable (skips any (model,task,fold) already on disk).
"""
import json, subprocess, datetime as dt, os
import numpy as np

GTS = "/home/ncrc/work/gts"
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"
STRIDE = 1
WINDOW = 14
DATASET_SUFFIX = "_ext"
N_DAYS = 1308

MODELS_TASKS = [
    ("pdformer", "volume"), ("pdformer", "speed"),
    ("mtgnn", "volume"), ("mtgnn", "speed"),
    ("agcrn", "volume"), ("agcrn", "speed"),
]


def month_bounds():
    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    days = d["days"][:N_DAYS]
    day_objs = [dt.datetime.strptime("20" + str(s), "%Y%m%d").date() for s in days]
    month_start = {}
    for i, do in enumerate(day_objs):
        key = (do.year, do.month)
        if key not in month_start:
            month_start[key] = i
    months_sorted = sorted(month_start.keys())
    bounds = []
    for k, key in enumerate(months_sorted):
        start = month_start[key]
        end = month_start[months_sorted[k + 1]] if k + 1 < len(months_sorted) else len(days)
        bounds.append((key[0], key[1], start, end))
    return bounds


bounds = month_bounds()
n_months = len(bounds)
starts = list(range(0, n_months - WINDOW + 1, STRIDE))
print(f"{n_months} months, {len(starts)} folds at starts {starts}", flush=True)


def run(cmd, tag):
    print(f"\n=== {tag} ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=GTS, check=True)


for model, task in MODELS_TASKS:
    for s in starts:
        out = f"seedctl_{model}_{task}_fold{s}_s1_ext30"
        if os.path.exists(f"{GTS}/{out}/summary.json"):
            continue
        train_m = bounds[s:s + 12]
        val_m = bounds[s + 12:s + 13]
        test_m = bounds[s + 13:s + 14]
        train_lo, train_hi = train_m[0][2], train_m[-1][3]
        val_lo, val_hi = val_m[0][2], val_m[-1][3]
        test_lo, test_hi = test_m[0][2], test_m[-1][3]

        cmd = [PY, "-u", "train_baseline_model_extra.py", "--model", model, "--task", task,
               "--dataset_suffix", DATASET_SUFFIX, "--use_od_injection", "0", "--seed", "1",
               "--train_lo", str(train_lo), "--train_hi", str(train_hi),
               "--val_lo", str(val_lo), "--val_hi", str(val_hi),
               "--test_lo", str(test_lo), "--test_hi", str(test_hi),
               "--epochs", "40", "--patience", "8", "--out", out]
        run(cmd, f"{model}/{task} fold{s} seed1")

print("\nSEEDCTL_EXTRA_EXT30_ALL_DONE")
