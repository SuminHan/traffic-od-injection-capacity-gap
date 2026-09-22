"""kNN + Gaussian-kernel spatial graph for the 396 speed sensors (no precomputed equivalent of
smhan's volume_point_graph.npz exists for speed) -- k=4 nearest neighbors by Euclidean distance
in EPSG:5186 meters, adaptive bandwidth (median nearest-neighbor distance), symmetrized."""
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree

GTS = "/home/ncrc/work/gts"
K = 4

spd = np.load(f"{GTS}/speed_tensor.npz", allow_pickle=True)
link_ids = list(spd["link_ids"])
geo = gpd.read_file("/home/ncrc/ksa_uve_backup/topis_link_geometry.geojson").set_index("LINK_ID")
t_proj = Transformer.from_crs("EPSG:5181", "EPSG:5186", always_xy=True)
xy = np.array([t_proj.transform(geo.loc[lid, "geometry"].centroid.x, geo.loc[lid, "geometry"].centroid.y) for lid in link_ids])

tree = cKDTree(xy)
dists, idxs = tree.query(xy, k=K + 1)  # includes self at k=0
ls = np.median(dists[:, 1:])  # adaptive bandwidth = median nearest-neighbor distance
print(f"adaptive length-scale: {ls:.1f}m")

n = len(link_ids)
A = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    for k in range(1, K + 1):
        j = idxs[i, k]
        w = np.exp(-(dists[i, k] ** 2) / (2 * ls ** 2))
        A[i, j] = max(A[i, j], w)
        A[j, i] = max(A[j, i], w)  # symmetrize

np.savez_compressed(f"{GTS}/speed_point_graph.npz", A=A, link_ids=np.array(link_ids), k=K, length_scale=ls)
print(f"saved speed_point_graph.npz: A {A.shape}, nonzero frac={(A>1e-4).mean():.4f}")
