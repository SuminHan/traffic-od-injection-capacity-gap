"""
For EACH of the 396 speed sensors, compute its top-K contributing (origin,destination) OD pairs
by routed share (same dijkstra/alpha=0.9 assignment as build_routing_weights_speed.py) -- powers
an interactive "click a sensor, see which OD flows route through it" map. Also saves the actual
routed path polyline (lat/lon) for each sensor's #1-ranked OD pair only (to keep the embedded
JSON a reasonable size -- 396 x full-path-for-every-top-K would be too large for a static page).
"""
import json
import numpy as np
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.ops import unary_union
import heapq

SC = "/home/ncrc/work"
GTS = f"{SC}/gts"
NL = "/path/to/raw_data/seoul_buildings/nodelink"
BBOX = (-10052.5, 477236.5, 275064.6, 632421.6)
ALPHA = 0.9
RANK_BASE = {"101": 0.50, "102": 0.60, "103": 0.75, "104": 0.95, "105": 0.90, "106": 0.95, "107": 1.00}
TOPK = 8

links = gpd.read_file(f"{NL}/MOCT_LINK.shp", encoding="cp949", bbox=BBOX)
nodes = gpd.read_file(f"{NL}/MOCT_NODE.shp", encoding="cp949", bbox=BBOX)
node_ids = nodes["NODE_ID"].values
node_idx = {nid: i for i, nid in enumerate(node_ids)}
n = len(node_ids)

rows, cols, length_w, rank_arr = [], [], [], []
for r in links.itertuples():
    f, t_, length = r.F_NODE, r.T_NODE, r.LENGTH
    if f not in node_idx or t_ not in node_idx or length is None or length <= 0:
        continue
    rows.append(node_idx[f]); cols.append(node_idx[t_]); length_w.append(length)
    rank_arr.append(str(r.ROAD_RANK))
rows = np.array(rows); cols = np.array(cols); length_w = np.array(length_w, dtype=np.float64)
base_w = np.array([RANK_BASE.get(rk, 1.0) for rk in rank_arr])
w = length_w * (base_w ** ALPHA)
edge_map = {(rows[k], cols[k]): k for k in range(len(rows))}
n_links = len(rows)
graph = csr_matrix((w, (rows, cols)), shape=(n, n))
node_xy = np.array([(g.x, g.y) for g in nodes.geometry])
print(f"graph ready: {n_links} edges, {n} nodes")

od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
our_codes = [str(int(c)) for c in od["codes"]]
dong = gpd.read_file(f"{SC}/admdongkor/ver20260701/HangJeongDong_ver20260701.geojson")
dong["code8"] = dong["adm_cd2"].str[:-2]
seoul_geom = dong[dong["code8"].notna()].set_index("code8")["geometry"].to_dict()
sgg_union = {sgg: unary_union(g.geometry) for sgg, g in dong.groupby("sgg")}
name_by_code = dict(zip(dong["adm_cd2"].str[:-2], dong["adm_nm"])) if "adm_nm" in dong.columns else {}
t_proj = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
t_inv = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
node_tree = cKDTree(node_xy)

code_to_nodeidx = {}
for code in our_codes:
    geom = seoul_geom.get(code) if code[:2] == "11" else None
    if geom is None:
        geom = sgg_union.get(code[:5])
    if geom is None:
        continue
    cen = geom.centroid
    x, y = t_proj.transform(cen.x, cen.y)
    dist, idx = node_tree.query([x, y])
    if dist <= 15000:
        code_to_nodeidx[code] = idx

spd = np.load(f"{GTS}/speed_tensor.npz", allow_pickle=True)
sensor_link_ids = list(spd["link_ids"])
geo = gpd.read_file("/home/ncrc/ksa_uve_backup/topis_link_geometry.geojson").set_index("LINK_ID")
t_5181 = Transformer.from_crs("EPSG:5181", "EPSG:5186", always_xy=True)
sensor_xy = np.array([t_5181.transform(geo.loc[lid, "geometry"].centroid.x, geo.loc[lid, "geometry"].centroid.y) for lid in sensor_link_ids])
edge_mid = np.stack([(node_xy[rows[k]] + node_xy[cols[k]]) / 2 for k in range(n_links)])
edge_tree = cKDTree(edge_mid)
_, target_link_idx = edge_tree.query(sensor_xy)
target_local = {li: k for k, li in enumerate(target_link_idx)}
n_sensors = len(sensor_link_ids)

train_od_sum = od["od"][:365].sum(axis=(0, 1))
code_pos = {c: i for i, c in enumerate(our_codes)}
outflow_by_dest_share = {}
for code in code_to_nodeidx:
    i = code_pos[code]
    row = train_od_sum[i].copy(); row[i] = 0
    tot = row.sum()
    if tot > 0:
        outflow_by_dest_share[code] = row / tot

# per-sensor min-heap of (share, origin_code, dest_code, n_hops) -- keep only TOPK per sensor
sensor_heaps = [[] for _ in range(n_sensors)]
mapped_codes = list(code_to_nodeidx.keys())
t0 = __import__("time").time()
for oi, o_code in enumerate(mapped_codes):
    o_i = code_to_nodeidx[o_code]
    shares = outflow_by_dest_share.get(o_code)
    if shares is None:
        continue
    _, predecessors = dijkstra(graph, directed=True, indices=o_i, return_predecessors=True)
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
        if cur != o_i or not hit:
            continue
        for sj in hit:
            entry = (float(share), o_code, d_code, steps)
            if len(sensor_heaps[sj]) < TOPK:
                heapq.heappush(sensor_heaps[sj], entry)
            elif entry[0] > sensor_heaps[sj][0][0]:
                heapq.heapreplace(sensor_heaps[sj], entry)
    if oi % 50 == 0:
        print(f"  {oi}/{len(mapped_codes)} origins ({__import__('time').time()-t0:.0f}s)")

print(f"top-K scan done ({__import__('time').time()-t0:.0f}s)")

# for each sensor's #1 entry, reconstruct the actual path polyline
sensor_data = {}
for sj, lid in enumerate(sensor_link_ids):
    heap = sorted(sensor_heaps[sj], key=lambda e: -e[0])
    if not heap:
        sensor_data[lid] = {"latlon": list(t_inv.transform(*sensor_xy[sj])[::-1]), "topk": []}
        continue
    topk_list = [{"origin_code": o, "origin_name": name_by_code.get(o, o),
                   "dest_code": d, "dest_name": name_by_code.get(d, d),
                   "share": s, "hops": h} for (s, o, d, h) in heap]
    # reconstruct path for rank-1
    s1, o1, d1, h1 = heap[0]
    o_i, d_i = code_to_nodeidx[o1], code_to_nodeidx[d1]
    _, predecessors = dijkstra(graph, directed=True, indices=o_i, return_predecessors=True)
    path_nodes = [d_i]
    cur = d_i
    while cur != o_i and predecessors[cur] >= 0:
        cur = predecessors[cur]
        path_nodes.append(cur)
    path_nodes.reverse()
    latlon_path = [list(t_inv.transform(*node_xy[ni])[::-1]) for ni in path_nodes[::max(1, len(path_nodes)//60)]]
    sensor_data[lid] = {"latlon": list(t_inv.transform(*sensor_xy[sj])[::-1]), "topk": topk_list,
                         "top1_path": latlon_path}

with open(f"{GTS}/sensor_topk_od.json", "w") as f:
    json.dump(sensor_data, f)
print(f"saved sensor_topk_od.json: {len(sensor_data)} sensors, "
      f"{sum(1 for v in sensor_data.values() if v['topk'])} with >=1 OD contributor")
