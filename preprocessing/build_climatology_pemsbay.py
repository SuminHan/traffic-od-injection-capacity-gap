"""Builds a calendar-climatology auxiliary signal for PEMS-BAY, so the capacity law can be tested
on a second, public dataset.

The law reported in Section 8 -- auxiliary-signal benefit decays with model capacity -- currently
rests on one city and one signal (routed OD). PEMS-BAY has no origin-destination data, so the OD
signal cannot be carried over. What can be carried over is the control signal from Section 6.1: a
train-window-only (day-of-week, time-of-day) climatology, which on Seoul recovered roughly 82% of
the routed signal's benefit. That makes it a legitimate stand-in, and a deliberately conservative
one: it contains no privileged information whatsoever, only a calendar lookup table, so any
capacity-dependent pattern it produces cannot be attributed to the quality of an OD forecast.

The question this answers is therefore sharper than the Seoul study can ask alone: is "cheap
auxiliary signal helps small models and hurts large ones" a property of routed OD in Seoul, or a
property of auxiliary-signal injection as such? Replicating the decay here would make the law a
two-dataset, two-signal result.

Leakage safety matches build_climatology_signal_perfold.py: the climatology VALUES come only from
the training window, then are broadcast across train/val/test alike, exactly as a deployed system
would apply a lookup table fitted on history.

Output: routed_od_signal_speed_pemsbay.npz, the filename train_benchmark_model.py resolves for
--task speed --dataset_suffix _pemsbay --routing_source single.
"""
import numpy as np

GTS = "/home/ncrc/work/gts"
STEPS_PER_DAY = 288
TRAIN_HI_DAY = 125          # matches prep_pemsbay.py's day-aligned 70/10/20 split
N_BUCKETS = 7 * STEPS_PER_DAY

tt = np.load(f"{GTS}/speed_tensor_pemsbay.npz", allow_pickle=True)
speed = tt["speed"]                       # (S, days, steps)
sensor_ids = tt["link_ids"]
days = tt["days"]
S, n_days, steps = speed.shape
assert steps == STEPS_PER_DAY

flat = speed.transpose(1, 2, 0).reshape(n_days * steps, S)   # (T, S)
T = flat.shape[0]
print(f"PEMS-BAY speed: T={T} S={S} ({n_days} days x {steps} steps)")

# PEMS-BAY starts 2017-01-01, a Sunday; days[] carries YYMMDD so derive weekday from it directly.
import datetime as dt
dow_per_day = np.array([dt.datetime.strptime(f"20{d:06d}", "%Y%m%d").weekday() for d in days])
dow = np.repeat(dow_per_day, steps)
tod = np.tile(np.arange(steps), n_days)
bucket = dow * steps + tod
assert bucket.max() < N_BUCKETS

train_hi = TRAIN_HI_DAY * steps
tb, ts = bucket[:train_hi], flat[:train_hi]

lookup = np.zeros((N_BUCKETS, S), dtype=np.float32)
counts = np.zeros(N_BUCKETS, dtype=np.int64)
for b in range(N_BUCKETS):
    m = tb == b
    counts[b] = m.sum()
    # Missing values are zero-coded; average over observed entries so the climatology is not
    # dragged toward zero by dropouts.
    if counts[b] > 0:
        w = ts[m]
        obs = w != 0
        denom = obs.sum(axis=0)
        lookup[b] = np.where(denom > 0, (w * obs).sum(axis=0) / np.maximum(denom, 1),
                             ts[ts != 0].mean())
    else:
        lookup[b] = ts[ts != 0].mean()
print(f"buckets: {N_BUCKETS}, samples/bucket min={counts.min()} median={int(np.median(counts))}")

clim = lookup[bucket]                      # (T, S), broadcast over the full range
assert clim.shape == flat.shape

obs = flat != 0
corr = np.corrcoef(clim[obs].ravel(), flat[obs].ravel())[0, 1]
print(f"corr(climatology, observed speed) = {corr:.4f}")

out = f"{GTS}/routed_od_signal_speed_pemsbay.npz"
np.savez_compressed(out, routed_forecast=clim.astype(np.float32),
                    sensor_link_ids=sensor_ids)
print(f"wrote {out} {clim.shape}")
print("\nPEMSBAY_CLIMATOLOGY_DONE")
