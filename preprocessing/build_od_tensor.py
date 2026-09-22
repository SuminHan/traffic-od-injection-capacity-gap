"""
Build the (day, hour, origin, dest) OD tensor for the 491-node (Seoul 427 dong + Gyeonggi/Incheon
64 gu-si) capital-region graph, from the 31 contiguous days of July 2026 movement data.

Temporal resolution: HOURLY. 31 days x 24 hours = 744 timesteps, contiguous (unlike the earlier
4 non-contiguous sample days used for traffic-assignment validation).
"""
import zipfile, pickle, time
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree

SC = "/home/ncrc/work"
MOVE_DIR = f"{SC}/movement_202607"
OUT_DIR = f"{SC}/gts"
t0 = time.time()

DAYS = [f"2607{d:02d}" for d in range(1, 32)]

# --- node set: reuse the exact same Seoul-dong (exact) + Gyeonggi/Incheon-gu (sgg) crosswalk
# used for the capital-region traffic assignment, so this graph's node ordering is consistent
# with everything built so far.
nodes = gpd.read_file(f"{SC}/seoul_nodes.gpkg")  # just for CRS reference / not routing here
dong = gpd.read_file(f"{SC}/admdongkor/ver20260701/HangJeongDong_ver20260701.geojson")
dong["code8"] = dong["adm_cd2"].str[:-2]
seoul_geom = dong[dong["code8"].notna()].set_index("code8")["geometry"].to_dict()
from shapely.ops import unary_union
sgg_union = {sgg: unary_union(g.geometry) for sgg, g in dong.groupby("sgg")}

BBOX_5186 = (-10052.5, 477236.5, 275064.6, 632421.6)
t = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)

# collect all codes seen across the 31 days
codes = set()
for day in DAYS:
    with zipfile.ZipFile(f"{MOVE_DIR}/movement_{day}.zip") as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            for chunk in pd.read_csv(f, usecols=["o_admdong_cd", "d_admdong_cd"], dtype=str, chunksize=3_000_000):
                codes.update(chunk["o_admdong_cd"].unique()); codes.update(chunk["d_admdong_cd"].unique())
print(f"{len(codes)} unique codes across 31 days ({time.time()-t0:.0f}s)")

valid_codes = []
for code in codes:
    if code[:2] == "11" and code in seoul_geom:
        geom = seoul_geom[code]
    else:
        geom = sgg_union.get(code[:5])
    if geom is None:
        continue
    cen = geom.centroid
    x, y = t.transform(cen.x, cen.y)
    if BBOX_5186[0] - 15000 <= x <= BBOX_5186[2] + 15000 and BBOX_5186[1] - 15000 <= y <= BBOX_5186[3] + 15000:
        valid_codes.append(code)
valid_codes = sorted(set(valid_codes))
print(f"{len(valid_codes)} codes kept as graph nodes ({time.time()-t0:.0f}s)")

code_idx = {c: i for i, c in enumerate(valid_codes)}
N = len(valid_codes)
n_days, n_hours = len(DAYS), 24

# dense (day, hour, N, N) is 31*24*491*491*4bytes = ~717MB float32 -- fine to hold in memory
od = np.zeros((n_days, n_hours, N, N), dtype=np.float32)

for di, day in enumerate(DAYS):
    with zipfile.ZipFile(f"{MOVE_DIR}/movement_{day}.zip") as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            parts = []
            for chunk in pd.read_csv(f, usecols=["o_admdong_cd", "d_admdong_cd", "st_time_cd", "cnt"],
                                      dtype={"st_time_cd": str, "o_admdong_cd": str, "d_admdong_cd": str},
                                      chunksize=3_000_000):
                chunk = chunk[chunk["o_admdong_cd"].isin(code_idx) & chunk["d_admdong_cd"].isin(code_idx)]
                if not len(chunk):
                    continue
                chunk["hour"] = chunk["st_time_cd"].str[:2].astype(int)
                parts.append(chunk.groupby(["o_admdong_cd", "d_admdong_cd", "hour"])["cnt"].sum())
    # aggregate across chunks -> each (o,d,hour) key now unique, so a single fancy-index
    # ASSIGNMENT (not accumulation) suffices -- np.add.at on raw ungrouped rows was the
    # bottleneck; this groupby-then-assign pattern is what the earlier 4-day script used and it
    # ran in seconds per file.
    day_od = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
    o_codes, d_codes, h_vals = zip(*day_od.index)
    oi = np.array([code_idx[c] for c in o_codes])
    dj = np.array([code_idx[c] for c in d_codes])
    h = np.array(h_vals)
    od[di, h, oi, dj] = day_od.values.astype(np.float32)
    print(f"  {day}: {len(day_od)} OD-hour combos done ({time.time()-t0:.0f}s)")

np.savez_compressed(f"{OUT_DIR}/od_tensor_202607.npz", od=od, codes=np.array(valid_codes), days=np.array(DAYS))
print(f"saved od tensor {od.shape} ({time.time()-t0:.0f}s), nonzero frac={float((od>0).mean()):.4f}")
