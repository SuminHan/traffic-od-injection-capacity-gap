"""
Critical robustness check for the selective-injection finding (Section 8.3): is the
validation-based per-sensor selector actually doing informed work, or would ANY per-sensor
mix of the plain and od_simple checkpoints (chosen without looking at validation data at all)
produce a similar-looking "improvement" -- e.g. because averaging/mixing two independently
trained checkpoints has some generic ensembling benefit regardless of which sensor gets which?

For each of the 8 (model,task) combos and each fold, draw 200 RANDOM per-sensor selection masks
(matched to the same number of sensors assigned to "od" as the real validation-based selector used
that fold, from selective_injection_{model}_{task}_fold_results.csv's n_sensors_chose_od column),
apply each to that fold's already-computed test_predictions.csv, and compute the resulting MSE.
This gives a proper null distribution per fold. Then:
  1. Per combo: what fraction of (fold, random-draw) pairs beat plain at all, and what's the mean
     %MSE change of the random baseline vs. the real validation-based selector's known result?
  2. A directly interpretable empirical p-value per combo: P(random selector's mean-across-folds
     %MSE change <= the real selector's actual mean %MSE change), using per-fold-averaged random
     draws so the comparison is apples-to-apples with how the real result was computed.

No GPU, no retraining -- reuses the exact same test_predictions.csv files already on disk from
the original 30-fold matrix sweep.
"""
import numpy as np
import pandas as pd

GTS = "/home/ncrc/work/gts"
RNG = np.random.RandomState(0)
N_RANDOM = 200

COMBOS = [
    ("dcrnn", "volume", "multi_fold_baseline_dcrnn_volume_results_ext30.json"),
    ("dcrnn", "speed", "multi_fold_baseline_dcrnn_speed_results_ext30.json"),
    ("gwnet", "volume", "multi_fold_baseline_gwnet_volume_results_ext30.json"),
    ("gwnet", "speed", "multi_fold_baseline_gwnet_speed_results_ext30.json"),
    ("gman", "volume", "multi_fold_baseline_gman_volume_results_ext30.json"),
    ("gman", "speed", "multi_fold_baseline_gman_speed_results_ext30.json"),
    ("staeformer", "volume", "multi_fold_staeformer_volume_results_ext30.json"),
    ("staeformer", "speed", "multi_fold_staeformer_speed_results_ext30.json"),
]

import json

all_rows = []
for model_name, task, results_file in COMBOS:
    vkey = "true_volume" if task == "volume" else "true_speed"
    pkey = "pred_volume" if task == "volume" else "pred_speed"

    real = pd.read_csv(f"{GTS}/selective_injection_{model_name}_{task}_fold_results.csv")
    results = json.load(open(f"{GTS}/{results_file}"))
    by_fold = {}
    for e in results:
        by_fold.setdefault(e["fold_start"], {})[e["variant"]] = e
    od_key = "od_simple" if "od_simple" in by_fold[list(by_fold.keys())[0]] else "od"

    fold_random_pct = []  # per fold: mean %change across 200 random draws
    for _, row in real.iterrows():
        fold = int(row["fold"])
        n_chose_od = int(row["n_sensors_chose_od"])
        plain_e, od_e = by_fold[fold]["plain"], by_fold[fold][od_key]
        tp = pd.read_csv(f"{GTS}/{plain_e['out']}/test_predictions.csv")
        to = pd.read_csv(f"{GTS}/{od_e['out']}/test_predictions.csv")
        tp = tp.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
        to = to.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
        n_sensors = int(tp["sensor_idx"].max()) + 1
        true_vals = tp[vkey].values
        pred_p = tp[pkey].values
        pred_o = to[pkey].values
        sensor_idx = tp["sensor_idx"].values
        mse_plain = float(np.mean((true_vals - pred_p) ** 2))

        draws_pct = np.empty(N_RANDOM)
        for i in range(N_RANDOM):
            chosen_od_sensors = RNG.choice(n_sensors, size=n_chose_od, replace=False)
            mask = np.isin(sensor_idx, chosen_od_sensors)
            pred_rand = np.where(mask, pred_o, pred_p)
            mse_rand = float(np.mean((true_vals - pred_rand) ** 2))
            draws_pct[i] = (mse_rand - mse_plain) / mse_plain * 100
        fold_random_pct.append(draws_pct)

    fold_random_pct = np.array(fold_random_pct)  # (n_folds, N_RANDOM)
    # per-draw, average %change across folds (matches how the real result's mean %change was computed)
    per_draw_mean_across_folds = fold_random_pct.mean(axis=0)  # (N_RANDOM,)

    real_mean_pct = real_pct = None
    summary_csv = pd.read_csv(f"{GTS}/selective_injection_all_combos_summary.csv")
    real_row = summary_csv[(summary_csv.model == model_name) & (summary_csv.task == task)].iloc[0]
    real_mean_pct = float(real_row["selective_mean_pct"])

    p_empirical = float((per_draw_mean_across_folds <= real_mean_pct).mean())
    random_mean = float(per_draw_mean_across_folds.mean())
    random_std = float(per_draw_mean_across_folds.std())

    print(f"{model_name}/{task}: REAL selective mean %MSE change = {real_mean_pct:+.2f}%  |  "
          f"RANDOM baseline: mean={random_mean:+.2f}% std={random_std:.2f}%  |  "
          f"empirical p(random <= real) = {p_empirical:.4f}", flush=True)

    all_rows.append({"model": model_name, "task": task, "real_selective_pct": real_mean_pct,
                      "random_mean_pct": random_mean, "random_std_pct": random_std,
                      "empirical_p": p_empirical})

out_df = pd.DataFrame(all_rows)
out_df.to_csv(f"{GTS}/selective_injection_random_control_summary.csv", index=False)
print("\n" + out_df.to_string(index=False))
print("\nwrote selective_injection_random_control_summary.csv")
