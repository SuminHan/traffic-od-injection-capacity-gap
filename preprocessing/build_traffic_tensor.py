"""
Build a (n_days, 24, n_sensors) hourly traffic-volume tensor from smhan's durable
volume_hourly_cache.npz (124 TOPIS point sensors, 2023-01-01..2025-12-31, ~12.2% NaN),
restricted to the 121 sensors whose matched dong (sensor_dong_match.csv) is one of the
500 capital-region nodes already used for the OD graph -- so sensor i's traffic can be
fused with OD node j = dong_node_idx[i]'s predicted flow.
NaNs are forward/back-filled per-sensor (short sensor outages), any still-NaN left as 0
with a companion validity mask so the training loss can ignore them.
"""
import numpy as np
import pandas as pd

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
SCRATCH = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad"

sd = pd.read_csv(f"{SCRATCH}/sensor_dong_match.csv")
od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [int(c) for c in od["codes"]]
code_to_idx = {c: i for i, c in enumerate(od_codes)}
sd = sd[sd.code8.isin(code_to_idx)].reset_index(drop=True)
print(f"{len(sd)} sensors matched to OD's 500-node graph")

vc = np.load(f"{SCRATCH.rsplit('/',1)[0]}/uve_experiment/pipeline_nowcast/volume_hourly_cache.npz"
             if False else "/home/smhan/uve_experiment/pipeline_nowcast/volume_hourly_cache.npz", allow_pickle=True)
link_ids = list(vc["link_ids"])
keep_mask = np.array([lid in set(sd.link_id) for lid in link_ids])
sel = np.where(keep_mask)[0]
# reorder sd to match the volume cache's link order (restricted to kept sensors)
link_order = [link_ids[i] for i in sel]
sd_idx = sd.set_index("link_id")
sd = sd_idx.loc[link_order].reset_index()

dates = vc["dates"]; hours = vc["hours"]; volume = vc["volume"][sel]  # (n_sensors, T)
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

# per-sensor linear interpolation over the flattened (day,hour) axis for short gaps,
# any still-NaN (leading/trailing or fully-missing sensor stretches) -> 0 + invalid mask
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

node_idx = np.array([code_to_idx[c] for c in sd.code8], dtype=np.int64)  # sensor -> OD node index
day_strs = np.array([str(d)[2:] for d in uniq_dates])  # YYMMDD to match od_tensor_full's days format

np.savez_compressed(
    f"{GTS}/traffic_tensor.npz",
    volume=vol_tensor.astype(np.float32),          # (n_sensors, n_days, 24)
    valid_mask=valid_mask.reshape(n_sensors, n_days, 24),
    link_ids=np.array(sd.link_id),
    od_node_idx=node_idx,                            # index into od_tensor_full's 500 nodes
    days=day_strs,
)
print(f"saved traffic_tensor.npz: volume {vol_tensor.shape}, days {day_strs[0]}..{day_strs[-1]}, "
      f"valid_frac={valid_mask.mean():.4f}")
