"""
Builds the "naive same-district outflow pasted onto nearest sensor" injection signal for
volume, to properly (re-)validate with real fold-based numbers a claim this paper currently only
supports with a one-sentence, unquantified mention of "early, unsuccessful pilot experiments"
(Section 4.2) -- inconsistent with this paper's own rigor standard everywhere else.

Same schema as routed_od_signal_ext.npz (routed_forecast (T,121), sensor_link_ids) so it drops
straight into train_traffic_model.py via --routed_signal_suffix _naive_ext, but instead of
outflow_forecast @ W (real road-network routing), each sensor simply receives the outflow
forecast of whichever of the 500 origin zones has its centroid geographically nearest to that
sensor -- exactly the naive baseline the paper's prose describes, now actually built and testable.

Reuses the identical origin-zone-centroid and sensor-coordinate methodology as
build_routing_weights.py (same admdongkor dong/SGG-union geometry, same EPSG:5186 projection,
same volume_hourly_cache.npz sensor coordinates) so the only thing that differs from the real
routed signal is the routing step itself.
"""
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.ops import unary_union

SC = "/home/ncrc/work"
GTS = f"{SC}/gts"
TRAFFIC_DAYS = 1308

rw = np.load(f"{GTS}/routing_weights.npz", allow_pickle=True)
our_codes = list(rw["codes"])  # same order as od_tensor_full.npz codes, and as outflow_forecast's N axis
sensor_link_ids = list(rw["sensor_link_ids"])

dong = gpd.read_file(f"{SC}/admdongkor/ver20260701/HangJeongDong_ver20260701.geojson")
dong["code8"] = dong["adm_cd2"].str[:-2]
seoul_geom = dong[dong["code8"].notna()].set_index("code8")["geometry"].to_dict()
sgg_union = {sgg: unary_union(g.geometry) for sgg, g in dong.groupby("sgg")}
t_proj = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)

origin_xy = []
valid_origin_idx = []
for i, code in enumerate(our_codes):
    geom = seoul_geom.get(code) if code[:2] == "11" else sgg_union.get(code[:5])
    if geom is None:
        continue
    cen = geom.centroid
    x, y = t_proj.transform(cen.x, cen.y)
    origin_xy.append((x, y))
    valid_origin_idx.append(i)
origin_xy = np.array(origin_xy)
valid_origin_idx = np.array(valid_origin_idx)
print(f"{len(valid_origin_idx)}/{len(our_codes)} origins have geometry")

vc = np.load("/path/to/raw_data/pipeline_nowcast/volume_hourly_cache.npz", allow_pickle=True)
vc_ids = list(vc["link_ids"])
vc_pos = {lid: i for i, lid in enumerate(vc_ids)}
sensor_xy = np.array([t_proj.transform(vc["lon"][vc_pos[lid]], vc["lat"][vc_pos[lid]]) for lid in sensor_link_ids])

tree = cKDTree(origin_xy)
dists, nearest_local = tree.query(sensor_xy)
nearest_origin_idx = valid_origin_idx[nearest_local]  # (121,) index into the full 500-origin outflow array
print(f"sensor->nearest-origin-centroid distances: mean={dists.mean():.0f}m max={dists.max():.0f}m "
      f"(cf. real routing's mean snap dist ~{rw['snap_dist_m'].mean():.0f}m for context)")

sig = np.load(f"{GTS}/od_signal_ext.npz")
outflow_forecast = sig["pred_outflow"].T  # (T,500)
outflow_forecast = outflow_forecast[:TRAFFIC_DAYS * 24]

naive_forecast = outflow_forecast[:, nearest_origin_idx]  # (T,121) -- just gather, no routing math at all

np.savez_compressed(f"{GTS}/routed_od_signal_naive_ext.npz",
                     routed_forecast=naive_forecast.astype(np.float32),
                     sensor_link_ids=np.array(sensor_link_ids))
print(f"saved routed_od_signal_naive_ext.npz: {naive_forecast.shape}")

# sanity: how correlated is this naive signal with the REAL routed signal, and with ground truth?
real = np.load(f"{GTS}/routed_od_signal_ext.npz")["routed_forecast"]
corr_naive_real = np.corrcoef(naive_forecast.ravel(), real.ravel())[0, 1]
print(f"corr(naive, real routed signal) = {corr_naive_real:.4f}")

tt = np.load(f"{GTS}/traffic_tensor_ext.npz", allow_pickle=True)
vol = tt["volume"]  # (121, days, 24)
n_sensors, n_days, n_hours = vol.shape
true_flat = vol.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)[:TRAFFIC_DAYS * 24]
corr_naive_true = np.corrcoef(naive_forecast.ravel(), true_flat.ravel())[0, 1]
corr_real_true = np.corrcoef(real.ravel(), true_flat.ravel())[0, 1]
print(f"corr(naive signal, ground-truth sensor volume) = {corr_naive_true:.4f}")
print(f"corr(real routed signal, ground-truth sensor volume) = {corr_real_true:.4f}")
