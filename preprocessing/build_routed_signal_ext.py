"""Apply the static routing_weights.npz matrix W (500 dongs x 121 sensors) to per-node outflow
to get a properly ROUTED traffic signal per sensor -- both from ground-truth OD (for any
legitimate use as a known/past feature) and from the OD model's own forecast (od_signal_ext.npz, for
a leakage-free decode-time covariate)."""
import numpy as np

GTS = "/home/ncrc/work/gts"
TRAFFIC_DAYS = 1308

rw = np.load(f"{GTS}/routing_weights.npz", allow_pickle=True)
W = rw["W"]  # (500, 121), rows ordered same as od_tensor_full's codes
rw_codes = list(rw["codes"])

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [str(int(c)) for c in od["codes"]]
assert rw_codes == od_codes, "routing_weights.npz codes must align 1:1 with od_tensor_full.npz codes"

od_arr = od["od"]
n_days, n_hours, N, _ = od_arr.shape
T_total = TRAFFIC_DAYS * n_hours
outflow_actual = od_arr.reshape(n_days * n_hours, N, N).sum(axis=2)[:T_total]  # (T,N)
routed_actual = outflow_actual @ W  # (T,121)

sig = np.load(f"{GTS}/od_signal_ext.npz")
outflow_forecast = sig["pred_outflow"].T  # (T,N) -- pred_outflow saved as (N,T)
outflow_forecast = outflow_forecast[:T_total]
routed_forecast = outflow_forecast @ W  # (T,121)

np.savez_compressed(f"{GTS}/routed_od_signal_ext.npz",
                     routed_actual=routed_actual.astype(np.float32),
                     routed_forecast=routed_forecast.astype(np.float32),
                     sensor_link_ids=rw["sensor_link_ids"])
print(f"saved routed_od_signal_ext.npz: routed_actual {routed_actual.shape}, routed_forecast {routed_forecast.shape}")
print(f"routed_actual stats: mean={routed_actual.mean():.1f} std={routed_actual.std():.1f}")
print(f"routed_forecast stats: mean={routed_forecast.mean():.1f} std={routed_forecast.std():.1f}")
corr = np.corrcoef(routed_actual.ravel(), routed_forecast.ravel())[0, 1]
print(f"correlation(routed_actual, routed_forecast) = {corr:.4f}  (sanity: forecast should track actual)")
