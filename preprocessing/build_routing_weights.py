"""
Build a STATIC routing weight matrix W (500 OD-nodes x 121 TOPIS sensors), reusing the exact
capital-region road network + A*-style road-hierarchy-weighted shortest path approach from
capital_multipath_assign.py (this session's earlier validated "OD -> assignment -> traffic"
methodology) -- instead of the naive "just paste the matched dong's raw outflow onto its nearest
sensor" signal used in the first (unsuccessful) injection attempt.

W[origin, sensor] = share of origin dong's total historical (2023 train-period) outbound trips
whose shortest path crosses that sensor's road link, aggregated over all destinations. This is a
STATIC structural matrix (road network + historical destination split only, no daily/hourly OD
values baked in) -- apply it to ANY hour's per-node outflow (ground truth OR the OD model's own
forecast) via a simple matmul to get a properly ROUTED traffic signal for that hour.

Simplification vs. the original 9-alpha multi-path ensemble: single alpha=0.9 (strong road-
hierarchy preference) for tractability -- single shortest path per OD pair, not a weighted blend
of several candidate paths. Noted explicitly; can be upgraded to multi-path later if this helps.
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
SC = "/home/ncrc/work"
GTS = f"{SC}/gts"
NL = "/path/to/raw_data/seoul_buildings/nodelink"
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

# --- map our 500 od_tensor_full.npz codes to graph nodes (Seoul: exact dong; else: gu/si union) ---
od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
our_codes = [str(int(c)) for c in od["codes"]]
print(f"{len(our_codes)} OD-tensor codes to map")

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

# --- snap 121 TOPIS sensors (lat/lon, WGS84) to nearest graph EDGE (by midpoint) ---
tt = np.load(f"{GTS}/traffic_tensor.npz", allow_pickle=True)
sensor_link_ids = list(tt["link_ids"])
vc = np.load("/path/to/raw_data/pipeline_nowcast/volume_hourly_cache.npz", allow_pickle=True)
vc_ids = list(vc["link_ids"])
vc_pos = {lid: i for i, lid in enumerate(vc_ids)}
sensor_xy = []
for lid in sensor_link_ids:
    i = vc_pos[lid]
    x, y = t_proj.transform(vc["lon"][i], vc["lat"][i])
    sensor_xy.append((x, y))
sensor_xy = np.array(sensor_xy)

edge_mid = np.stack([(node_xy[rows[k]] + node_xy[cols[k]]) / 2 for k in range(n_links)])
edge_tree = cKDTree(edge_mid)
dists, target_link_idx = edge_tree.query(sensor_xy)
print(f"sensor->link snap distances: mean={dists.mean():.0f}m max={dists.max():.0f}m ({time.time()-t0:.0f}s)")
n_sensors = len(sensor_link_ids)
target_set = set(target_link_idx.tolist())
target_local = {li: k for k, li in enumerate(target_link_idx)}  # graph link-index -> sensor column

# --- historical (2023 train period) destination split per origin, from actual OD ground truth ---
n_days, n_hours, N, _ = od["od"].shape
train_od_sum = od["od"][:365].sum(axis=(0, 1))  # (N,N) total 2023 flow
codes_arr = our_codes
code_pos = {c: i for i, c in enumerate(codes_arr)}
outflow_by_dest_share = {}
for code in code_to_nodeidx:
    i = code_pos[code]
    row = train_od_sum[i].copy()
    row[i] = 0
    tot = row.sum()
    if tot > 0:
        outflow_by_dest_share[code] = row / tot  # (N,) destination shares

# --- for each origin, dijkstra -> for each dest dong, backtrack & mark which sensor links are used ---
W = np.zeros((N, n_sensors), dtype=np.float32)
mapped_codes = list(code_to_nodeidx.keys())
for oi, o_code in enumerate(mapped_codes):
    o_i = code_to_nodeidx[o_code]
    if o_code not in outflow_by_dest_share:
        continue
    shares = outflow_by_dest_share[o_code]
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

print(f"routing done ({time.time()-t0:.0f}s). W nonzero entries: {(W>0).sum()} / {W.size}")
np.savez_compressed(f"{GTS}/routing_weights.npz", W=W, codes=np.array(codes_arr),
                     sensor_link_ids=np.array(sensor_link_ids), alpha=ALPHA,
                     snap_dist_m=dists)
print(f"saved routing_weights.npz, total {time.time()-t0:.0f}s")
