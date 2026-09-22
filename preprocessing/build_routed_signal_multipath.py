"""Multipath version of build_routed_signal.py -- applies routing_weights_multipath.npz's W
(Monte Carlo perturbed-Dijkstra stochastic assignment) instead of the single-shortest-path W."""
import numpy as np

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
TRAFFIC_DAYS = 1096

rw = np.load(f"{GTS}/routing_weights_multipath.npz", allow_pickle=True)
W = rw["W"]
rw_codes = list(rw["codes"])

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [str(int(c)) for c in od["codes"]]
assert rw_codes == od_codes

od_arr = od["od"]
n_days, n_hours, N, _ = od_arr.shape
T_total = TRAFFIC_DAYS * n_hours
outflow_actual = od_arr.reshape(n_days * n_hours, N, N).sum(axis=2)[:T_total]
routed_actual = outflow_actual @ W

sig = np.load(f"{GTS}/od_signal.npz")
outflow_forecast = sig["pred_outflow"].T[:T_total]
routed_forecast = outflow_forecast @ W

np.savez_compressed(f"{GTS}/routed_od_signal_multipath.npz",
                     routed_actual=routed_actual.astype(np.float32),
                     routed_forecast=routed_forecast.astype(np.float32),
                     sensor_link_ids=rw["sensor_link_ids"])
print(f"saved routed_od_signal_multipath.npz: routed_forecast {routed_forecast.shape}")
print(f"stats: mean={routed_forecast.mean():.1f} std={routed_forecast.std():.1f}")
