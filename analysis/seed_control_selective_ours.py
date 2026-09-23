"""
Seed-control ("free lunch") measurement for our own lightweight model, the same protocol the
paper's Table freelunch applies to the ten published architectures (seed_control_selective_extra.py):

  1. two PLAIN checkpoints per fold differing only in seed (tfold*/sfold*_only_ext and the *_s1 twins
     trained by run_seedctl_ours_ext30.py),
  2. per-sensor selection between them on that fold's validation window only,
  3. free lunch = selective-between-plain-seeds vs. whichever whole checkpoint is better on validation.

Net effect for Ours = raw selective gain (selective_injection_ours_summary.csv) - free lunch.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from scipy import stats

from selective_injection_ours import (GTS, TASKS, device, load_series, load_graph,
                                      eval_persensor_mse)
from train_traffic_model import TrafficModel


def main():
    summary = []
    for task, results_file, plain_key, _, vkey, pkey, tensor_prefix, tensor_vkey, graph_path in TASKS:
        results = json.load(open(f"{GTS}/{results_file}"))
        plains = sorted((e for e in results if e["model"] == plain_key), key=lambda e: e["fold_start"])
        rows, t0 = [], time.time()
        for e in plains:
            out1 = e["out"] + "_s1"
            if not os.path.exists(f"{GTS}/{out1}/test_predictions.csv"):
                print(f"  missing {out1}, skipped", flush=True)
                continue
            a = argparse.Namespace(**e["args"])
            vol_t, time_t, mu, sd, link_ids, n_sensors, _ = load_series(a, tensor_prefix, tensor_vkey)
            ei, ew = load_graph(graph_path, link_ids)
            ms = []
            for out in (e["out"], out1):
                m = TrafficModel(n_sensors, ei, ew, a.hidden, False).to(device)
                m.load_state_dict(torch.load(f"{GTS}/{out}/best.pt", weights_only=True))
                ms.append(m)
            v0, v1 = (eval_persensor_mse(m, a, vol_t, time_t, None, mu, sd, n_sensors,
                                         a.val_lo, a.val_hi, False) for m in ms)
            choose1 = v1 < v0
            key = ["window_start_hour", "decode_step", "sensor_idx"]
            d0 = pd.read_csv(f"{GTS}/{e['out']}/test_predictions.csv").sort_values(key).reset_index(drop=True)
            d1 = pd.read_csv(f"{GTS}/{out1}/test_predictions.csv").sort_values(key).reset_index(drop=True)
            assert (d0.sensor_idx.values == d1.sensor_idx.values).all()
            y = d0[vkey].values
            psel = np.where(choose1[d0.sensor_idx.values], d1[pkey].values, d0[pkey].values)
            mse0 = float(np.mean((y - d0[pkey].values) ** 2))
            mse1 = float(np.mean((y - d1[pkey].values) ** 2))
            msel = float(np.mean((y - psel) ** 2))
            gp1 = bool(v1.mean() < v0.mean())
            rows.append({"fold": e["fold_start"], "mse_seed0": mse0, "mse_seed1": mse1,
                         "mse_selective": msel, "mse_globalpick": mse1 if gp1 else mse0,
                         "globalpick_is_seed1": gp1, "n_sensors_chose_seed1": int(choose1.sum()),
                         "n_sensors": n_sensors})
            print(f"  {task} fold {e['fold_start']:2d}: sel vs gp "
                  f"{(msel - rows[-1]['mse_globalpick']) / rows[-1]['mse_globalpick'] * 100:+.2f}% "
                  f"({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(f"{GTS}/seed_control_ours_{task}_fold_results.csv", index=False)
        fl = (df.mse_selective - df.mse_globalpick) / df.mse_globalpick * 100
        swing = ((df.mse_seed1 - df.mse_seed0) / df.mse_seed0 * 100).abs()
        raw = pd.read_csv(f"{GTS}/selective_injection_ours_summary.csv").set_index("task").loc[task,
                                                                                             "selective_mean_pct"]
        summary.append({"model": "ours", "task": task, "n_folds": len(df),
                        "raw_selective_pct": float(raw), "free_lunch_pct": float(fl.mean()),
                        "free_lunch_p": float(stats.ttest_rel(df.mse_selective, df.mse_globalpick)[1]),
                        "net_pct": float(raw - fl.mean()), "seed_swing_abs_mean_pct": float(swing.mean())})
        print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(f"{GTS}/seed_control_ours_summary.csv", index=False)


if __name__ == "__main__":
    main()
