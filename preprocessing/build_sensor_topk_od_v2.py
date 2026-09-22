"""
v2: for each sensor's top-K contributing OD pairs, reconstruct the ACTUAL routed path for
EVERY entry (not just rank-1), and also export the origin/destination dong BOUNDARY polygons
(simplified, reprojected to WGS84) for every code that appears -- powers a richer interactive map
(dong shapes highlighted + all top-K paths drawn, not just the #1).
"""
import json, time, heapq
import numpy as np
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.ops import unary_union, transform as shp_transform

SC = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad"
GTS = f"{SC}/gts"
NL = "/home/smhan/uve_experiment/seoul_buildings/nodelink"
BBOX = (-10052.5, 477236.5, 275064.6, 632421.6)
ALPHA = 0.9
RANK_BASE = {"101": 0.50, "102": 0.60, "103": 0.75, "104": 0.95, "105": 0.90, "106": 0.95, "107": 1.00}
TOPK = 8
t0 = time.time()

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
print(f"graph ready: {n_links} edges, {n} nodes ({time.time()-t0:.0f}s)")

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

code_to_nodeidx, code_to_geom = {}, {}
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
        code_to_geom[code] = geom  # WGS84 already (admdongkor is lat/lon)

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

# ---- pass 1: same top-K heap scan as before ----
sensor_heaps = [[] for _ in range(n_sensors)]
mapped_codes = list(code_to_nodeidx.keys())
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
        print(f"  pass1 {oi}/{len(mapped_codes)} origins ({time.time()-t0:.0f}s)")
print(f"pass1 done ({time.time()-t0:.0f}s)")

# ---- pass 2: reconstruct paths for EVERY (origin,dest) pair that made ANY sensor's top-K ----
needed_pairs = {}  # origin -> set of dest codes needing a path
final_topk = {}  # sensor_link_id -> sorted list of entries
for sj, lid in enumerate(sensor_link_ids):
    heap = sorted(sensor_heaps[sj], key=lambda e: -e[0])
    final_topk[lid] = heap
    for (s, o, d, h) in heap:
        needed_pairs.setdefault(o, set()).add(d)

print(f"pass2: {len(needed_pairs)} distinct origins need path reconstruction")
path_cache = {}  # (o,d) -> latlon path (decimated)
for oi, (o_code, dests) in enumerate(needed_pairs.items()):
    o_i = code_to_nodeidx[o_code]
    _, predecessors = dijkstra(graph, directed=True, indices=o_i, return_predecessors=True)
    for d_code in dests:
        d_i = code_to_nodeidx[d_code]
        path_nodes = [d_i]
        cur = d_i
        steps = 0
        while cur != o_i and predecessors[cur] >= 0 and steps < 5000:
            cur = predecessors[cur]
            path_nodes.append(cur)
            steps += 1
        if cur != o_i:
            continue
        path_nodes.reverse()
        step = max(1, len(path_nodes) // 40)
        latlon = [list(t_inv.transform(*node_xy[ni])[::-1]) for ni in path_nodes[::step]]
        path_cache[(o_code, d_code)] = latlon
    if oi % 50 == 0:
        print(f"  pass2 {oi}/{len(needed_pairs)} origins ({time.time()-t0:.0f}s)")
print(f"pass2 done ({time.time()-t0:.0f}s), {len(path_cache)} paths cached")

# ---- assemble sensor data with full top-K paths ----
sensor_data = {}
all_codes_used = set()
for lid in sensor_link_ids:
    sj = sensor_link_ids.index(lid)  # small n_sensors, fine
    heap = final_topk[lid]
    topk_list = []
    for (s, o, d, h) in heap:
        all_codes_used.add(o); all_codes_used.add(d)
        topk_list.append({"origin_code": o, "origin_name": name_by_code.get(o, o),
                           "dest_code": d, "dest_name": name_by_code.get(d, d),
                           "share": s, "hops": h, "path": path_cache.get((o, d), [])})
    si = sensor_link_ids.index(lid)
    sensor_data[lid] = {"latlon": list(t_inv.transform(*sensor_xy[si])[::-1]), "topk": topk_list}

# ---- dong boundary polygons for all codes used, simplified + reprojected ----
def poly_rings(geom):
    """extract [ [ [lat,lon], ... ], ... ] exterior ring(s), simplified"""
    geom = geom.simplify(0.0005, preserve_topology=True)
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    rings = []
    for p in polys:
        coords = list(p.exterior.coords)
        rings.append([[lat, lon] for lon, lat in coords])
    return rings

dong_boundaries = {}
for code in all_codes_used:
    geom = code_to_geom.get(code)
    if geom is None:
        continue
    try:
        dong_boundaries[code] = poly_rings(geom)
    except Exception as e:
        print(f"skip boundary for {code}: {e}")

with open(f"{GTS}/sensor_topk_od_v2.json", "w") as f:
    json.dump({"sensors": sensor_data, "dong_boundaries": dong_boundaries}, f)
print(f"saved sensor_topk_od_v2.json: {len(sensor_data)} sensors, {len(dong_boundaries)} dong boundaries, "
      f"total {time.time()-t0:.0f}s")
