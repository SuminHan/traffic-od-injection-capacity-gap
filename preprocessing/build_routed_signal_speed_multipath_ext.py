"""Extended version of build_routed_signal_speed_multipath.py: applies
routing_weights_speed_multipath.npz's W to the FULL 2023-01..2026-07 od_signal_ext.npz range,
matching build_routed_signal_speed_ext.py's extension of the single-path version. Needed for
--routing_source multipath --dataset_suffix _ext --task speed in train_baseline_model.py."""
import numpy as np

GTS = "/home/ncrc/work/gts"
TRAFFIC_DAYS = 1308  # speed_tensor_ext.npz covers 2023-01-01..2026-07-31 (43 months)

rw = np.load(f"{GTS}/routing_weights_speed_multipath.npz", allow_pickle=True)
W = rw["W"]
rw_codes = list(rw["codes"])

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [str(int(c)) for c in od["codes"]]
assert rw_codes == od_codes

sig = np.load(f"{GTS}/od_signal_ext.npz")
outflow_forecast = sig["pred_outflow"].T[:TRAFFIC_DAYS * 24]
routed = outflow_forecast @ W

np.savez_compressed(f"{GTS}/routed_od_signal_speed_multipath_ext.npz",
                     routed_forecast=routed.astype(np.float32),
                     sensor_link_ids=rw["sensor_link_ids"])
print(f"saved routed_od_signal_speed_multipath_ext.npz: {routed.shape}")
print(f"stats: mean={routed.mean():.2f} std={routed.std():.2f}")
