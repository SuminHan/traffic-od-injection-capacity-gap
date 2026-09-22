"""
Same static routing-weight approach as build_routing_weights.py (A*-style road-hierarchy
shortest-path assignment, 2023 train-period destination split), but targeting the 396 TOPIS SPEED
sensor links instead of the 121 volume sensors. Speed sensors map onto the road graph via their
OWN LineString geometry (topis_link_geometry.geojson, EPSG:5181 -> reprojected to 5186) snapped to
the nearest graph edge midpoint -- same snapping method as the volume sensors, but these are real
road-link sensors (not point counters), so the snap should be tighter.
"""
import time
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.ops import unary_union

t0 = time.time()
SC = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad"
GTS = f"{SC}/gts"
NL = "/home/smhan/uve_experiment/seoul_buildings/nodelink"
BBOX = (-10052.5, 477236.5, 275064.6, 632421.6)
ALPHA = 0.9
RANK_BASE = {"101": 0.50, "102": 0.60, "103": 0.75, "104": 0.95, "105": 0.90, "106": 0.95, "107": 1.00}

links = gpd.read_file(f"{NL}/MOCT_LINK.shp", encoding="cp949", bbox=BBOX)
nodes = gpd.read_file(f"{NL}/MOCT_NODE.shp", encoding="cp949", bbox=BBOX)
node_ids = nodes["NODE_ID"].values
node_idx = {nid: i for i, nid in enumerate(node_ids)}
n = len(node_ids)
print(f"{len(links)} links, {n} nodes ({time.time()-t0:.0f}s)")

rows, cols, length_w, rank_arr, link_id_arr = [], [], [], [], []
for r in links.itertuples():
    f, t_, length = r.F_NODE, r.T_NODE, r.LENGTH
    if f not in node_idx or t_ not in node_idx or length is None or length <= 0:
        continue
    rows.append(node_idx[f]); cols.append(node_idx[t_]); length_w.append(length)
    rank_arr.append(str(r.ROAD_RANK)); link_id_arr.append(r.LINK_ID)
rows = np.array(rows); cols = np.array(cols); length_w = np.array(length_w, dtype=np.float64)
rank_arr = np.array(rank_arr); link_id_arr = np.array(link_id_arr)
base_w = np.array([RANK_BASE.get(rk, 1.0) for rk in rank_arr])
w = length_w * (base_w ** ALPHA)
edge_map = {(rows[k], cols[k]): k for k in range(len(rows))}
n_links = len(link_id_arr)
graph = csr_matrix((w, (rows, cols)), shape=(n, n))
print(f"{n_links} directed edges, graph built ({time.time()-t0:.0f}s)")

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
our_codes = [str(int(c)) for c in od["codes"]]

dong = gpd.read_file(f"{SC}/admdongkor/ver20260701/HangJeongDong_ver20260701.geojson")
dong["code8"] = dong["adm_cd2"].str[:-2]
seoul_geom = dong[dong["code8"].notna()].set_index("code8")["geometry"].to_dict()
sgg_union = {sgg: unary_union(g.geometry) for sgg, g in dong.groupby("sgg")}
t_proj = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
node_xy = np.array([(g.x, g.y) for g in nodes.geometry])
node_tree = cKDTree(node_xy)

code_to_nodeidx = {}
for code in our_codes:
    if code[:2] == "11" and code in seoul_geom:
        geom = seoul_geom[code]
    else:
        geom = sgg_union.get(code[:5])
    if geom is None:
        continue
    cen = geom.centroid
    x, y = t_proj.transform(cen.x, cen.y)
    dist, idx = node_tree.query([x, y])
    if dist <= 15000:
        code_to_nodeidx[code] = idx
print(f"{len(code_to_nodeidx)}/{len(our_codes)} codes mapped to graph nodes ({time.time()-t0:.0f}s)")

# --- snap 396 SPEED sensors (LineString geometry, EPSG:5181) to nearest graph EDGE (by midpoint) ---
spd = np.load(f"{GTS}/speed_tensor.npz", allow_pickle=True)
sensor_link_ids = list(spd["link_ids"])
geo = gpd.read_file("/home/ncrc/ksa_uve_backup/topis_link_geometry.geojson").set_index("LINK_ID")
t_5181 = Transformer.from_crs("EPSG:5181", "EPSG:5186", always_xy=True)
sensor_xy = []
for lid in sensor_link_ids:
    geom = geo.loc[lid, "geometry"]
    cen = geom.centroid
    x, y = t_5181.transform(cen.x, cen.y)
    sensor_xy.append((x, y))
sensor_xy = np.array(sensor_xy)

edge_mid = np.stack([(node_xy[rows[k]] + node_xy[cols[k]]) / 2 for k in range(n_links)])
edge_tree = cKDTree(edge_mid)
dists, target_link_idx = edge_tree.query(sensor_xy)
print(f"speed sensor->link snap distances: mean={dists.mean():.0f}m max={dists.max():.0f}m ({time.time()-t0:.0f}s)")
n_sensors = len(sensor_link_ids)
target_local = {li: k for k, li in enumerate(target_link_idx)}

n_days, n_hours, N, _ = od["od"].shape
train_od_sum = od["od"][:365].sum(axis=(0, 1)).astype(np.float64)
codes_arr = our_codes
code_pos = {c: i for i, c in enumerate(codes_arr)}

# MODE-SHARE VARIANT: see build_routing_weights_modeshare.py for the full rationale -- same
# distance-based car-mode-share discount (plateau=0.35, midpoint=2km) applied here for consistency.
MID_KM, SCALE_KM, MAX_CAR_SHARE = 2.0, 1.0, 0.35
mapped_codes_all = list(code_to_nodeidx.keys())
xy_all = np.array([node_xy[code_to_nodeidx[c]] for c in mapped_codes_all])
idx_all = np.array([code_pos[c] for c in mapped_codes_all])
d_km = np.linalg.norm(xy_all[:, None, :] - xy_all[None, :, :], axis=-1) / 1000.0
car_share = MAX_CAR_SHARE / (1 + np.exp(-(d_km - MID_KM) / SCALE_KM))
for i, oi in enumerate(idx_all):
    train_od_sum[oi, idx_all] *= car_share[i]
print(f"applied car-mode-share weighting to destination shares (mean weight={car_share.mean():.3f})")

outflow_by_dest_share = {}
for code in code_to_nodeidx:
    i = code_pos[code]
    row = train_od_sum[i].copy(); row[i] = 0
    tot = row.sum()
    if tot > 0:
        outflow_by_dest_share[code] = row / tot

W = np.zeros((N, n_sensors), dtype=np.float32)
mapped_codes = list(code_to_nodeidx.keys())
for oi, o_code in enumerate(mapped_codes):
    o_i = code_to_nodeidx[o_code]
    shares = outflow_by_dest_share.get(o_code)
    if shares is None:
        continue
    _, predecessors = dijkstra(graph, directed=True, indices=o_i, return_predecessors=True)
    row_out = code_pos[o_code]
    for d_code, d_i in code_to_nodeidx.items():
        if d_code == o_code:
            continue
        share = shares[code_pos[d_code]]
        if share <= 0:
            continue
        cur = d_i
        hit = set()
        steps = 0
        while cur != o_i and predecessors[cur] >= 0 and steps < 5000:
            prev = predecessors[cur]
            li = edge_map.get((prev, cur))
            if li is not None and li in target_local:
                hit.add(target_local[li])
            cur = prev
            steps += 1
        for sj in hit:
            W[row_out, sj] += share
    if oi % 50 == 0:
        print(f"  {oi}/{len(mapped_codes)} origins routed ({time.time()-t0:.0f}s)")

print(f"routing done ({time.time()-t0:.0f}s). W nonzero: {(W>0).sum()}/{W.size}")
np.savez_compressed(f"{GTS}/routing_weights_speed_modeshare.npz", W=W, codes=np.array(codes_arr),
                     sensor_link_ids=np.array(sensor_link_ids), alpha=ALPHA, snap_dist_m=dists)
