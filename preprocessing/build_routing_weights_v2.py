"""
v2: same as build_routing_weights.py, but computes TWO matrices in one pass (same expensive
per-origin dijkstra, reused) -- W (outflow-based, original) and V (inflow-based, new):

  W[origin, sensor]      = share of origin's outbound trips whose path crosses that sensor's link
  V[destination, sensor] = share of destination's INBOUND trips whose path crosses that sensor's
                            link (same paths, weighted by the destination's origin-composition
                            instead of the origin's destination-composition)

Rationale for adding V: W alone only captures "people leaving dong X" fanning out through the
network; V captures "people arriving at dong Y" converging through the network -- a link near a
job/shopping hub, say, could see heavy inbound-driven traffic that W's outflow-only view misses.
Apply od_signal's pred_outflow @ W and pred_inflow @ V, then feed the model BOTH routed channels.

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
inflow_by_origin_share = {}
for code in code_to_nodeidx:
    i = code_pos[code]
    row = train_od_sum[i].copy(); row[i] = 0
    tot = row.sum()
    if tot > 0:
        outflow_by_dest_share[code] = row / tot  # (N,) this dong's outbound trips, split by destination
    col = train_od_sum[:, i].copy(); col[i] = 0
    tot_in = col.sum()
    if tot_in > 0:
        inflow_by_origin_share[code] = col / tot_in  # (N,) this dong's inbound trips, split by origin

# --- for each origin, ONE dijkstra -> accumulate BOTH W (outflow-weighted, at origin's row) and
# V (inflow-weighted, at destination's row) from the same paths ---
W = np.zeros((N, n_sensors), dtype=np.float32)
V = np.zeros((N, n_sensors), dtype=np.float32)
mapped_codes = list(code_to_nodeidx.keys())
for oi, o_code in enumerate(mapped_codes):
    o_i = code_to_nodeidx[o_code]
    out_shares = outflow_by_dest_share.get(o_code)
    if out_shares is None:
        continue
    _, predecessors = dijkstra(graph, directed=True, indices=o_i, return_predecessors=True)
    row_out = code_pos[o_code]
    for d_code, d_i in code_to_nodeidx.items():
        if d_code == o_code:
            continue
        out_share = out_shares[code_pos[d_code]]
        in_shares_d = inflow_by_origin_share.get(d_code)
        in_share = in_shares_d[row_out] if in_shares_d is not None else 0.0
        if out_share <= 0 and in_share <= 0:
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
        row_dest = code_pos[d_code]
        for sj in hit:
            if out_share > 0:
                W[row_out, sj] += out_share
            if in_share > 0:
                V[row_dest, sj] += in_share
    if oi % 50 == 0:
        print(f"  {oi}/{len(mapped_codes)} origins routed ({time.time()-t0:.0f}s)")

print(f"routing done ({time.time()-t0:.0f}s). W nonzero: {(W>0).sum()}/{W.size}  V nonzero: {(V>0).sum()}/{V.size}")
np.savez_compressed(f"{GTS}/routing_weights_v2.npz", W=W, V=V, codes=np.array(codes_arr),
                     sensor_link_ids=np.array(sensor_link_ids), alpha=ALPHA,
                     snap_dist_m=dists)
print(f"saved routing_weights.npz, total {time.time()-t0:.0f}s")
