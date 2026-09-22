"""Extended version of build_routed_signal_multipath.py: applies routing_weights_multipath.npz's
W (Monte Carlo perturbed-Dijkstra stochastic assignment, unchanged/static) to the FULL
2023-01..2026-07 od_signal_ext.npz range, matching what build_routed_signal_ext.py did for the
single-path W. Needed because train_baseline_model.py looks for
routed_od_signal_multipath_ext.npz when --routing_source multipath --dataset_suffix _ext, which
didn't exist yet (only the non-extended 1096-day routed_od_signal_multipath.npz did)."""
import numpy as np

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
TRAFFIC_DAYS = 1308

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

sig = np.load(f"{GTS}/od_signal_ext.npz")
outflow_forecast = sig["pred_outflow"].T[:T_total]
routed_forecast = outflow_forecast @ W

np.savez_compressed(f"{GTS}/routed_od_signal_multipath_ext.npz",
                     routed_actual=routed_actual.astype(np.float32),
                     routed_forecast=routed_forecast.astype(np.float32),
                     sensor_link_ids=rw["sensor_link_ids"])
print(f"saved routed_od_signal_multipath_ext.npz: routed_actual {routed_actual.shape}, routed_forecast {routed_forecast.shape}")
corr = np.corrcoef(routed_actual.ravel(), routed_forecast.ravel())[0, 1]
print(f"correlation(routed_actual, routed_forecast) = {corr:.4f}")
