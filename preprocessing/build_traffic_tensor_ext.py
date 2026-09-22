"""Same as build_traffic_tensor.py but reading the extended volume cache (through 2026-07-31)."""
import numpy as np
import pandas as pd

GTS = "/home/ncrc/work/gts"
SCRATCH = "/home/ncrc/work"

sd = pd.read_csv(f"{SCRATCH}/sensor_dong_match.csv")
od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [int(c) for c in od["codes"]]
code_to_idx = {c: i for i, c in enumerate(od_codes)}
sd = sd[sd.code8.isin(code_to_idx)].reset_index(drop=True)
print(f"{len(sd)} sensors matched to OD's 500-node graph")

vc = np.load(f"{GTS}/volume_hourly_cache_ext.npz", allow_pickle=True)
link_ids = list(vc["link_ids"])
keep_mask = np.array([lid in set(sd.link_id) for lid in link_ids])
sel = np.where(keep_mask)[0]
link_order = [link_ids[i] for i in sel]
sd_idx = sd.set_index("link_id")
sd = sd_idx.loc[link_order].reset_index()

dates = vc["dates"]; hours = vc["hours"]; volume = vc["volume"][sel]
n_sensors = volume.shape[0]
uniq_dates = np.unique(dates)
uniq_dates.sort()
n_days = len(uniq_dates)
date_to_row = {d: i for i, d in enumerate(uniq_dates)}

vol_tensor = np.full((n_sensors, n_days, 24), np.nan, dtype=np.float32)
for t in range(volume.shape[1]):
    di = date_to_row[dates[t]]
    h = int(hours[t])
    vol_tensor[:, di, h] = volume[:, t]

flat = vol_tensor.reshape(n_sensors, n_days * 24)
valid_mask = ~np.isnan(flat)
for i in range(n_sensors):
    x = flat[i]
    nanidx = np.isnan(x)
    if nanidx.any() and (~nanidx).sum() > 1:
        good = np.where(~nanidx)[0]
        x[nanidx] = np.interp(np.where(nanidx)[0], good, x[good])
    flat[i] = np.nan_to_num(x, nan=0.0)
vol_tensor = flat.reshape(n_sensors, n_days, 24)

node_idx = np.array([code_to_idx[c] for c in sd.code8], dtype=np.int64)
day_strs = np.array([str(d)[2:] for d in uniq_dates])

np.savez_compressed(
    f"{GTS}/traffic_tensor_ext.npz",
    volume=vol_tensor.astype(np.float32),
    valid_mask=valid_mask.reshape(n_sensors, n_days, 24),
    link_ids=np.array(sd.link_id),
    od_node_idx=node_idx,
    days=day_strs,
)
print(f"saved traffic_tensor_ext.npz: volume {vol_tensor.shape}, days {day_strs[0]}..{day_strs[-1]}, "
      f"valid_frac={valid_mask.mean():.4f}")
