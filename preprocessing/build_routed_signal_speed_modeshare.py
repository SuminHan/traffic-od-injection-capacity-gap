"""Apply routing_weights_speed_modeshare.npz's W to the OD model's forecast (od_signal_ext.npz), producing
a routed signal aligned to the 396 speed-sensor links."""
import numpy as np

GTS = "/home/ncrc/work/gts"
TRAFFIC_DAYS = 1096  # speed_tensor.npz covers 2023-01-01..2025-12-31 (36 months, like the original volume run)

rw = np.load(f"{GTS}/routing_weights_speed_modeshare.npz", allow_pickle=True)
W = rw["W"]
rw_codes = list(rw["codes"])

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [str(int(c)) for c in od["codes"]]
assert rw_codes == od_codes

sig = np.load(f"{GTS}/od_signal_ext.npz")
outflow_forecast = sig["pred_outflow"].T[:TRAFFIC_DAYS * 24]  # (T,N)
routed = outflow_forecast @ W  # (T, 396)

np.savez_compressed(f"{GTS}/routed_od_signal_speed_modeshare.npz",
                     routed_forecast=routed.astype(np.float32),
                     sensor_link_ids=rw["sensor_link_ids"])
print(f"saved routed_od_signal_speed_modeshare.npz: {routed.shape}")
print(f"stats: mean={routed.mean():.2f} std={routed.std():.2f}")
