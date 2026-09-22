"""
Congestion-aware OD injection signal (backlog item from the calibration thread: "congestion-
aware routing" -- current injection sources, single-path and multipath, both do STATIC linear
routing (origin-outflow @ fixed weight matrix W, see build_routing_weights_multipath.py's
docstring) with NO capacity/BPR feedback. The calibrated static-assignment simulator (the OTHER
research track in this session, .../simulator/) DOES model capacity feedback via BPR + MSA
equilibrium, and is validated against real TOPIS sensor counts.

Rather than re-running full MSA for all 31392 (day,hour) slots in the OD-injection dataset
(timed at ~140-220s/hour earlier this session -> 50+ days of GPU time, infeasible), this reuses
the pooled sensor-hour records ALREADY collected across this session's calibration experiments
(103+ distinct days incl. two dedicated gap-fill runs covering all 24 hours, corrected via the
current best pipeline: rank-affine + geh_production_correction_v4.json's GEH-optimal group
correction). For each pooled record we have both (a) the calibrated-simulator's best estimate
of true traffic at that sensor/hour, and (b) the naive static-routing value for that SAME
(day,hour,sensor) from routed_od_signal_ext.npz (single-path).

v1 of this script used a MULTIPLICATIVE ratio (corrected_sim / naive_val) per (sensor,
hour_of_day) -- this blew up badly once night-hour data was added: at low-traffic hours (0-6,
22-23) the naive denominator is near zero, so small absolute differences produce huge relative
ratios (up to 60% of sensors clipped at the 10x ceiling for hour 23). Switched to an ADDITIVE
delta (corrected_sim - naive_val) instead, matching the additive-correction approach used
throughout the calibration track in this session (median(real-sim) style) -- additive deltas
don't blow up when the naive baseline is near zero, they just add a bounded absolute
congestion-redistribution amount. Grounds the delta per (sensor, hour_of_day) [category
dropped -- 0-t showed weekday/weekend variation beyond hour-of-day adds mostly noise at this
sample size] with a fallback to the (hour_of_day)-only global median for sensors with too few
pooled samples, then applies that delta additively (clamped so the result stays >= 0) to the
FULL single-path routed_actual/routed_forecast arrays (all 31392 hours).

This is an approximation (assumes the hour-of-day congestion SHAPE generalizes across days
within the same hour-of-day, doesn't capture day-specific congestion swings), not a substitute
for true per-hour re-simulation -- documented honestly as such.
"""
import json, glob, sys
import numpy as np

GTS = "/home/ncrc/work/gts"
SIM = f"{GTS}/simulator"
sys.path.insert(0, SIM)

# --- 1. load pooled calibration records + apply the current best correction pipeline ---
pool = {}
for fn in glob.glob(f"{SIM}/*_raw.json"):
    try:
        recs = json.load(open(fn))
    except Exception:
        continue
    for r in recs:
        if not {"rank", "sim", "real"} <= r.keys():
            continue
        pool[(r["day"], r["hour_of_day"], r["sensor_id"])] = r
all_records = list(pool.values())
print(f"pooled {len(all_records)} calibration records from {len(glob.glob(f'{SIM}/*_raw.json'))} files")

corr = json.load(open(f"{SIM}/geh_production_correction_v4.json"))
rank_coefs = corr["rank_affine_coefs"]
group_corr = corr["sensor_hour_category_correction_geh_optimal"]
fallback_corr = corr["sensor_hour_fallback_correction_geh_optimal"]

from datetime import date, timedelta
sys.path.insert(0, GTS)
from train_gts_od import KR_HOLIDAYS
EPOCH = date(2023, 1, 1)
holiday_set = set((date(int(h[:4]), int(h[4:6]), int(h[6:8])) - EPOCH).days for h in KR_HOLIDAYS)


def category3(day):
    if day in holiday_set:
        return "holiday"
    d = EPOCH + timedelta(days=day)
    return "weekend" if d.weekday() >= 5 else "weekday"


def apply_correction(r):
    a, b = rank_coefs.get(r["rank"], (1.0, 0.0))
    sim_ra = max(a * r["sim"] + b, 0.0)
    cat = category3(r["day"])
    gkey = f"{r['sensor_id']}|{r['hour_of_day']}|{cat}"
    fkey = f"{r['sensor_id']}|{r['hour_of_day']}"
    if gkey in group_corr:
        c = group_corr[gkey]
    elif fkey in fallback_corr:
        c = fallback_corr[fkey]
    else:
        c = 0.0
    return max(sim_ra + c, 0.0)


for r in all_records:
    r["sim_corrected"] = apply_correction(r)

# --- 2. naive static-routing reference for the SAME (day,hour,sensor) ---
rs = np.load(f"{GTS}/routed_od_signal_ext.npz", allow_pickle=True)
sensor_link_ids = list(rs["sensor_link_ids"])
sensor_pos = {sid: i for i, sid in enumerate(sensor_link_ids)}
routed_actual_full = rs["routed_actual"]  # (T=31392, 121)
routed_forecast_full = rs["routed_forecast"]
n_hours_per_day = 24

deltas_by_bucket = {}  # (sensor_idx, hour_of_day) -> list of additive deltas
deltas_by_hod = {}      # hour_of_day -> list of additive deltas (fallback pool)
n_missing_sensor, n_oob = 0, 0
for r in all_records:
    sid = r["sensor_id"]
    if sid not in sensor_pos:
        n_missing_sensor += 1
        continue
    sidx = sensor_pos[sid]
    t = r["day"] * n_hours_per_day + r["hour_of_day"]
    if t >= routed_actual_full.shape[0]:
        n_oob += 1
        continue
    naive_val = float(routed_actual_full[t, sidx])
    delta = r["sim_corrected"] - naive_val
    # naive static routing (no capacity limit) occasionally spikes to 10-20x a typical sensor
    # reading when OD demand piles onto one link -- clip the per-record delta so a handful of
    # these pathological naive-routing outliers can't dominate a bucket's median (real calibrated
    # volumes here range roughly up to ~14000, so +-8000 is a generous but bounded correction)
    delta = float(np.clip(delta, -8000.0, 8000.0))
    key = (sidx, r["hour_of_day"])
    deltas_by_bucket.setdefault(key, []).append(delta)
    deltas_by_hod.setdefault(r["hour_of_day"], []).append(delta)
print(f"skipped: {n_missing_sensor} sensor-not-found, {n_oob} out-of-range-for-routed-signal")

# --- 3. build a (121, 24) median-delta table with fallback ---
n_sensors = len(sensor_link_ids)
delta_table = np.zeros((n_sensors, n_hours_per_day), dtype=np.float32)
n_direct, n_fallback, n_default = 0, 0, 0
hod_fallback_median = {h: float(np.median(v)) for h, v in deltas_by_hod.items()}
for sidx in range(n_sensors):
    for hod in range(n_hours_per_day):
        key = (sidx, hod)
        if key in deltas_by_bucket and len(deltas_by_bucket[key]) >= 3:
            delta_table[sidx, hod] = float(np.median(deltas_by_bucket[key]))
            n_direct += 1
        elif hod in hod_fallback_median:
            delta_table[sidx, hod] = hod_fallback_median[hod]
            n_fallback += 1
        else:
            n_default += 1  # stays 0.0 (no adjustment)
print(f"delta table: {n_direct} direct (sensor,hour) buckets, {n_fallback} hour-only fallback, {n_default} default(0.0)")
print(f"delta table stats: mean={delta_table.mean():.3f}, min={delta_table.min():.3f}, max={delta_table.max():.3f}")

# --- 4. apply additively to the FULL 31392-hour arrays, clamped >= 0 ---
hours_arr = np.arange(routed_actual_full.shape[0]) % n_hours_per_day
add = delta_table[:, hours_arr].T  # (T, n_sensors)
congestion_actual = np.maximum(routed_actual_full + add, 0.0).astype(np.float32)
congestion_forecast = np.maximum(routed_forecast_full + add, 0.0).astype(np.float32)

np.savez_compressed(f"{GTS}/routed_od_signal_congestion_aware_ext.npz",
                     routed_actual=congestion_actual, routed_forecast=congestion_forecast,
                     sensor_link_ids=np.array(sensor_link_ids), delta_table=delta_table)
print(f"wrote routed_od_signal_congestion_aware_ext.npz: shape {congestion_forecast.shape}")
corr_vs_single = np.corrcoef(congestion_forecast.ravel(), routed_forecast_full.ravel())[0, 1]
print(f"correlation(congestion_forecast, single-path routed_forecast) = {corr_vs_single:.4f} "
      f"(should be high but <1.0 -- same demand signal, congestion-reshaped)")
