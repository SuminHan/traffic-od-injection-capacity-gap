"""
Builds the calendar-confound control that main.tex's own Section 4.3 prose flags as a real risk
but has never actually run for the paper's headline lightweight-model result: Section 4.3 states
"an initial version lacking [a] future-calendar term was found statistically indistinguishable
from injecting shuffled ... OD signal" for the published-baseline mechanism-(i) runs, but the
headline Table~main result (our own model, -7.17%/-8.17% MSE) has never been checked against a
climatology-only control. If a pure (day-of-week, hour-of-day) climatology of the routed OD signal
-- computed from the TRAINING window only, no forecast, no OD model, just a calendar lookup table
-- recovers a similar improvement, the headline result would be a calendar-embedding effect rather
than genuine OD-signal value.

For each of the same 5 folds already used for Traffic-Only/Traffic-Hybrid (multi_fold_traffic_
results_ext.json) and the naive-routing ablation, this builds a per-fold climatology signal file:
for every sensor and (day-of-week, hour-of-day) bucket, the mean of the REAL routed OD signal
(routed_od_signal_ext.npz) over that fold's training window only, then broadcast across the full
1308-day range (train/val/test alike -- the leakage-safety comes from the climatology VALUES being
train-only, not from withholding it elsewhere, exactly matching build_od_diagnostic.py's existing
leakage-safe methodology). Output schema matches routed_od_signal_ext.npz so it drops straight into
train_traffic_model.py via --routed_signal_suffix climatology_fold{N}_ext.
"""
import json
import numpy as np
from datetime import date, timedelta

GTS = "/home/ncrc/work/gts"  # 30-fold extension: path only, else identical
N_HOURS_PER_DAY = 24
DAY0 = date(2023, 1, 1)

sig = np.load(f"{GTS}/routed_od_signal_ext.npz", allow_pickle=True)
real = sig["routed_forecast"]  # (T, S)
sensor_link_ids = sig["sensor_link_ids"]
T, S = real.shape
print(f"real routed signal: T={T} S={S}")

n_days = T // N_HOURS_PER_DAY
dow_per_day = np.array([(DAY0 + timedelta(days=d)).weekday() for d in range(n_days)])  # 0=Mon
dow_per_hour = np.repeat(dow_per_day, N_HOURS_PER_DAY)  # (T,)
hour_of_day = np.tile(np.arange(N_HOURS_PER_DAY), n_days)  # (T,)
bucket = dow_per_hour * N_HOURS_PER_DAY + hour_of_day  # (T,) in [0, 168)

existing = json.load(open(f"{GTS}/multi_fold_traffic_results_ext.json"))
hyb = {e["fold_start"]: e for e in existing if e["model"] == "traffic_hybrid_routed"}

for fold in range(30):
    out_path_check = f"{GTS}/routed_od_signal_climatology_fold{fold}_ext.npz"
    import os
    if os.path.exists(out_path_check):
        print(f"fold {fold}: [skip] already built")
        continue
    a = hyb[fold]["args"]
    tr_lo_h, tr_hi_h = a["train_lo"] * N_HOURS_PER_DAY, a["train_hi"] * N_HOURS_PER_DAY
    train_bucket = bucket[tr_lo_h:tr_hi_h]
    train_signal = real[tr_lo_h:tr_hi_h]  # (T_train, S)

    clim_lookup = np.zeros((168, S), dtype=np.float32)
    for b in range(168):
        mask = train_bucket == b
        n = mask.sum()
        if n > 0:
            clim_lookup[b] = train_signal[mask].mean(axis=0)
        else:
            clim_lookup[b] = train_signal.mean(axis=0)  # fallback, should not trigger (>=52 weeks/bucket)

    climatology_full = clim_lookup[bucket]  # (T, S), broadcast across the FULL range incl. val/test
    assert climatology_full.shape == real.shape

    out_path = f"{GTS}/routed_od_signal_climatology_fold{fold}_ext.npz"
    np.savez_compressed(out_path, routed_forecast=climatology_full.astype(np.float32),
                         sensor_link_ids=sensor_link_ids)

    corr_clim_real = np.corrcoef(climatology_full.ravel(), real.ravel())[0, 1]
    print(f"fold {fold}: train window [{a['train_lo']},{a['train_hi']}) days -> "
          f"saved {out_path}, corr(climatology, real routed signal) = {corr_clim_real:.4f}")

print("\nCLIMATOLOGY_SIGNAL_BUILD_30_ALL_DONE")
