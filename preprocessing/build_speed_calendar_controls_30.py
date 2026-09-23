"""Speed-task version of the calendar-confound controls (Table controls): for each of the 30 folds,
a train-window-only (day-of-week, hour-of-day) climatology of the REAL routed speed signal, and the
residual (real - climatology). Same construction as build_climatology_signal_perfold_30.py and
build_od_residual_signal_30.py, applied to routed_od_signal_speed_ext.npz."""
import json, os
import numpy as np
from datetime import date, timedelta

GTS = "/home/ncrc/work/gts"
sig = np.load(f"{GTS}/routed_od_signal_speed_ext.npz", allow_pickle=True)
real, ids = sig["routed_forecast"], sig["sensor_link_ids"]
T, S = real.shape
n_days = T // 24
dow = np.repeat([(date(2023, 1, 1) + timedelta(days=d)).weekday() for d in range(n_days)], 24)
bucket = dow * 24 + np.tile(np.arange(24), n_days)
hyb = {e["fold_start"]: e for e in json.load(open(f"{GTS}/multi_fold_speed_results_ext.json"))
       if e["model"] == "speed_hybrid_routed"}
for fold in range(30):
    a = hyb[fold]["args"]
    lo, hi = a["train_lo"] * 24, a["train_hi"] * 24
    look = np.stack([real[lo:hi][bucket[lo:hi] == b].mean(0) for b in range(168)]).astype(np.float32)
    clim = look[bucket]
    np.savez_compressed(f"{GTS}/routed_od_signal_speed_climatology_fold{fold}_ext.npz",
                        routed_forecast=clim, sensor_link_ids=ids)
    np.savez_compressed(f"{GTS}/routed_od_signal_speed_residual_fold{fold}_ext.npz",
                        routed_forecast=(real - clim).astype(np.float32), sensor_link_ids=ids)
    print(fold, "corr(clim,real)=%.4f" % np.corrcoef(clim.ravel(), real.ravel())[0, 1], flush=True)
print("SPEED_CONTROL_SIGNALS_DONE")
