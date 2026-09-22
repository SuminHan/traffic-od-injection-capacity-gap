"""
Extension of selective_injection_random_control.py to the six (architecture, task) combinations
added to Table~selective/freelunch after the original 8 (GTS x2, STGCN x2, STID_fixed x2) --
same random-selection negative control (200 per-fold random per-sensor masks, matched sensor
count, compared to the real validation-based selector's result), just computing real_mean_pct
directly from each combo's own fold_results CSV instead of the original 8-combo master summary
(which doesn't have these rows). No GPU, reuses existing test_predictions.csv files.
"""
import json
import numpy as np
import pandas as pd

GTS = "/home/ncrc/work/gts"
RNG = np.random.RandomState(0)
N_RANDOM = 200

COMBOS = [
    ("gts", "volume", "multi_fold_baseline_gts_volume_results_ext30.json"),
    ("gts", "speed", "multi_fold_baseline_gts_speed_results_ext30.json"),
    ("stgcn", "volume", "multi_fold_baseline_stgcn_volume_results_ext30.json"),
    ("stgcn", "speed", "multi_fold_baseline_stgcn_speed_results_ext30.json"),
    ("stid_fixed", "volume", "multi_fold_baseline_stid_fixed_volume_results_ext30.json"),
    ("stid_fixed", "speed", "multi_fold_baseline_stid_fixed_speed_results_ext30.json"),
]

all_rows = []
for model_name, task, results_file in COMBOS:
    vkey = "true_volume" if task == "volume" else "true_speed"
    pkey = "pred_volume" if task == "volume" else "pred_speed"

    real = pd.read_csv(f"{GTS}/selective_injection_{model_name}_{task}_fold_results.csv")
    real_mean_pct = float(((real["mse_selective"] - real["mse_plain"]) / real["mse_plain"] * 100).mean())

    results = json.load(open(f"{GTS}/{results_file}"))
    by_fold = {}
    for e in results:
        by_fold.setdefault(e["fold_start"], {})[e["variant"]] = e
    od_key = "od_simple" if "od_simple" in by_fold[list(by_fold.keys())[0]] else "od"

    fold_random_pct = []
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

    fold_random_pct = np.array(fold_random_pct)
    per_draw_mean_across_folds = fold_random_pct.mean(axis=0)

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
out_df.to_csv(f"{GTS}/selective_injection_random_control_ext_summary.csv", index=False)
print("\n" + out_df.to_string(index=False))
print("\nwrote selective_injection_random_control_ext_summary.csv")
