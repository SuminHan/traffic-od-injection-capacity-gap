"""Build a (n_sensors, n_days, 24) hourly speed tensor from smhan's durable speed_hourly_cache.npz
(396 TOPIS link sensors, 2023-01-01..2025-12-31, only 0.14% NaN -- much cleaner than volume's
12%). Same interpolation/valid-mask treatment as build_traffic_tensor.py."""
import numpy as np

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"

vc = np.load("/home/smhan/uve_experiment/pipeline_nowcast/speed_hourly_cache.npz", allow_pickle=True)
link_ids = list(vc["link_ids"])
dates = vc["dates"]; hours = vc["hours"]; speed = vc["speed"]  # (396, T)
n_sensors = speed.shape[0]
uniq_dates = np.unique(dates); uniq_dates.sort()
n_days = len(uniq_dates)
date_to_row = {d: i for i, d in enumerate(uniq_dates)}

spd_tensor = np.full((n_sensors, n_days, 24), np.nan, dtype=np.float32)
for t in range(speed.shape[1]):
    di = date_to_row[dates[t]]
    h = int(hours[t])
    spd_tensor[:, di, h] = speed[:, t]

flat = spd_tensor.reshape(n_sensors, n_days * 24)
valid_mask = ~np.isnan(flat)
for i in range(n_sensors):
    x = flat[i]
    nanidx = np.isnan(x)
    if nanidx.any() and (~nanidx).sum() > 1:
        good = np.where(~nanidx)[0]
        x[nanidx] = np.interp(np.where(nanidx)[0], good, x[good])
    flat[i] = np.nan_to_num(x, nan=0.0)
spd_tensor = flat.reshape(n_sensors, n_days, 24)

day_strs = np.array([str(d)[2:] for d in uniq_dates])
np.savez_compressed(
    f"{GTS}/speed_tensor.npz",
    speed=spd_tensor.astype(np.float32),
    valid_mask=valid_mask.reshape(n_sensors, n_days, 24),
    link_ids=np.array(link_ids),
    days=day_strs,
)
print(f"saved speed_tensor.npz: speed {spd_tensor.shape}, days {day_strs[0]}..{day_strs[-1]}, "
      f"valid_frac={valid_mask.mean():.4f}")
