"""Apply BOTH routing_weights_v2.npz matrices (W=outflow-based, V=inflow-based) to the OD model's
forecast (od_signal_ext.npz), producing a 2-channel routed signal: routed_outflow (via W, same as
routed_forecast before) and routed_inflow (via V, new)."""
import numpy as np

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
TRAFFIC_DAYS = 1308

rw = np.load(f"{GTS}/routing_weights_v2.npz", allow_pickle=True)
W, V = rw["W"], rw["V"]
rw_codes = list(rw["codes"])

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
od_codes = [str(int(c)) for c in od["codes"]]
assert rw_codes == od_codes

sig = np.load(f"{GTS}/od_signal_ext.npz")
outflow_forecast = sig["pred_outflow"].T[:TRAFFIC_DAYS * 24]  # (T,N)
inflow_forecast = sig["pred_inflow"].T[:TRAFFIC_DAYS * 24]    # (T,N)

routed_outflow = outflow_forecast @ W  # (T,121)
routed_inflow = inflow_forecast @ V    # (T,121)

np.savez_compressed(f"{GTS}/routed_od_signal_v2.npz",
                     routed_outflow=routed_outflow.astype(np.float32),
                     routed_inflow=routed_inflow.astype(np.float32),
                     sensor_link_ids=rw["sensor_link_ids"])
print(f"saved routed_od_signal_v2.npz: routed_outflow {routed_outflow.shape}, routed_inflow {routed_inflow.shape}")
print(f"routed_outflow: mean={routed_outflow.mean():.1f} std={routed_outflow.std():.1f}")
print(f"routed_inflow:  mean={routed_inflow.mean():.1f} std={routed_inflow.std():.1f}")
corr = np.corrcoef(routed_outflow.ravel(), routed_inflow.ravel())[0, 1]
print(f"correlation(routed_outflow, routed_inflow) = {corr:.4f}  (sanity: related but not identical)")
