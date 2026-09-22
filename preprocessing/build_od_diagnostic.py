"""
Diagnose WHY OD injection helps our own simple model but not DCRNN: test the "redundancy"
hypothesis two ways, using ONLY already-saved per-sample CSVs + the routed OD signal (no GPU).

(1) Does OD injection help MORE on samples where the true routed-OD signal deviates a lot from
    its (day-of-week, hour-of-day) typical baseline ("anomalous" hours) vs samples close to
    normal? If the hypothesis is right, improvement should correlate with anomaly magnitude.
(2) Does OD injection help MORE at sensors with WEAK spatial autocorrelation with their kernel-
    graph neighbors (where the model has less to infer from pure spatial structure) vs sensors
    with strong neighbor correlation (where the model can already infer flow from neighbors)?

Uses DCRNN/volume's full 23-fold plain-vs-OD sweep (the most complete not-significant result).
"""
import glob
import numpy as np
import pandas as pd

GTS = "/home/ncrc/work/gts"

# --- OD anomaly magnitude per (sensor_link_id, global_hour) ---
rs = np.load(f"{GTS}/routed_od_signal.npz", allow_pickle=True)
sensor_ids = list(rs["sensor_link_ids"])
routed = rs["routed_forecast"]  # (T, 121)
T, S = routed.shape
hours_arr = np.arange(T) % 24
# day-of-week needs actual calendar; reuse od_tensor_full.npz's days + KR holiday logic isn't
# needed here, just dow, which we can get via the same date parsing used throughout the session
import datetime as dt
d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
days = d["days"][:T // 24]
day_objs = [dt.datetime.strptime("20" + str(s), "%Y%m%d").date() for s in days]
dows_daily = np.array([do.weekday() for do in day_objs])
dows_arr = np.repeat(dows_daily, 24)

baseline = np.zeros_like(routed)
for dow in range(7):
    for hr in range(24):
        mask = (dows_arr == dow) & (hours_arr == hr)
        if mask.sum() == 0:
            continue
        baseline[mask] = routed[mask].mean(axis=0)
anomaly = np.abs(routed - baseline)  # (T, 121) -- magnitude of deviation from typical
# per-sensor z-score the anomaly so magnitudes are comparable across sensors of different scale
anomaly_z = (anomaly - anomaly.mean(axis=0)) / (anomaly.std(axis=0) + 1e-6)
sensor_pos = {sid: i for i, sid in enumerate(sensor_ids)}

# --- spatial autocorrelation per sensor: corr(sensor's own traffic, neighbor-weighted-avg traffic) ---
tt = np.load(f"{GTS}/traffic_tensor.npz", allow_pickle=True)
link_ids = list(tt["link_ids"])
vol = tt["volume"].reshape(len(link_ids), -1)  # (121, T_full)
g = np.load(f"{GTS}/volume_point_graph.npz", allow_pickle=True)
g_ids = list(g["link_ids"]); gpos = {lid: i for i, lid in enumerate(g_ids)}
sel = np.array([gpos[l] for l in link_ids])
A = g["A"][np.ix_(sel, sel)]  # (121,121) reordered to match traffic_tensor's link order
neighbor_avg = (A @ vol) / (A.sum(axis=1, keepdims=True) + 1e-6)
autocorr = np.array([np.corrcoef(vol[i], neighbor_avg[i])[0, 1] for i in range(len(link_ids))])
autocorr_by_link = dict(zip(link_ids, autocorr))
print("spatial autocorrelation: mean", np.nanmean(autocorr), "min", np.nanmin(autocorr), "max", np.nanmax(autocorr))

# --- aggregate per-sample improvement across all 23 folds ---
plain_dirs = sorted(glob.glob(f"{GTS}/bfold*_dcrnn_volume_plain")) + [f"{GTS}/baseline_dcrnn_volume_fold0"]
anomaly_bins = [[] for _ in range(5)]  # quintile buckets of anomaly_z -> list of improvements
sensor_improve_sum = {}
sensor_improve_n = {}
n_folds_done = 0

for plain_dir in plain_dirs:
    od_dir = plain_dir.replace("_plain", "_od") if "_plain" in plain_dir else plain_dir + "_od"
    import os
    if not os.path.exists(f"{od_dir}/test_predictions.csv") or not os.path.exists(f"{plain_dir}/test_predictions.csv"):
        continue
    po = pd.read_csv(f"{plain_dir}/test_predictions.csv", usecols=["global_hour", "sensor_link_id", "true_volume", "pred_volume"])
    oo = pd.read_csv(f"{od_dir}/test_predictions.csv", usecols=["global_hour", "sensor_link_id", "true_volume", "pred_volume"])
    po = po.rename(columns={"pred_volume": "pred_plain"})
    oo = oo.rename(columns={"pred_volume": "pred_od"})
    m = po.merge(oo[["global_hour", "sensor_link_id", "pred_od"]], on=["global_hour", "sensor_link_id"])
    m["err_plain"] = (m["true_volume"] - m["pred_plain"]).abs()
    m["err_od"] = (m["true_volume"] - m["pred_od"]).abs()
    m["improve"] = m["err_plain"] - m["err_od"]  # positive = OD helped this sample

    gh = m["global_hour"].values
    valid_gh = gh < T
    m = m[valid_gh]
    gh = gh[valid_gh]
    sidx = np.array([sensor_pos.get(s, -1) for s in m["sensor_link_id"].values])
    valid_s = sidx >= 0
    m = m[valid_s]; gh = gh[valid_s]; sidx = sidx[valid_s]

    az = anomaly_z[gh, sidx]
    bins = pd.qcut(az, 5, labels=False, duplicates="drop")
    for b in range(bins.max() + 1 if len(bins) else 0):
        anomaly_bins[b].extend(m["improve"].values[bins == b].tolist())

    for sid, grp in m.groupby("sensor_link_id"):
        sensor_improve_sum[sid] = sensor_improve_sum.get(sid, 0.0) + grp["improve"].sum()
        sensor_improve_n[sid] = sensor_improve_n.get(sid, 0) + len(grp)
    n_folds_done += 1
    print(f"  processed {os.path.basename(plain_dir)} ({n_folds_done} folds so far)")

print(f"\n=== Diagnostic 1: improvement by OD-anomaly quintile ({n_folds_done} folds) ===")
for b in range(5):
    if anomaly_bins[b]:
        print(f"  quintile {b} (n={len(anomaly_bins[b])}): mean improve(err_plain-err_od) = {np.mean(anomaly_bins[b]):+.3f}")

print(f"\n=== Diagnostic 2: per-sensor mean improvement vs spatial autocorrelation ===")
rows = []
for sid in sensor_improve_sum:
    if sid in autocorr_by_link and sensor_improve_n[sid] > 0:
        rows.append((autocorr_by_link[sid], sensor_improve_sum[sid] / sensor_improve_n[sid]))
rows = np.array(rows)
corr = np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]
print(f"  n_sensors={len(rows)}, corr(autocorr, mean_improve) = {corr:.3f}")
# also bin by autocorr tercile
order = np.argsort(rows[:, 0])
terciles = np.array_split(order, 3)
for i, name in enumerate(["weak-autocorr", "mid-autocorr", "strong-autocorr"]):
    vals = rows[terciles[i], 1]
    print(f"  {name} sensors (n={len(vals)}): mean improve = {vals.mean():+.3f}")

np.savez(f"{GTS}/od_diagnostic_results.npz",
         anomaly_bins=np.array([np.mean(b) if b else np.nan for b in anomaly_bins]),
         sensor_autocorr=rows[:, 0], sensor_improve=rows[:, 1])
print("\nsaved od_diagnostic_results.npz")
