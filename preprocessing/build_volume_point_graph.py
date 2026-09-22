"""kNN + Gaussian-kernel spatial graph for the 121 volume sensors, mirroring
build_speed_point_graph.py -- k=4 nearest neighbors by Euclidean distance in EPSG:5186
meters, adaptive bandwidth (median nearest-neighbor distance), symmetrized. Needed as the
station-station adjacency for the ported baseline models (GWNET/DCRNN/GMAN), which expect
a fixed graph distinct from the OD-routing weight matrix."""
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
K = 4

tt = np.load(f"{GTS}/traffic_tensor.npz", allow_pickle=True)
link_ids = list(tt["link_ids"])
vc = np.load("/home/smhan/uve_experiment/pipeline_nowcast/volume_hourly_cache.npz", allow_pickle=True)
vc_ids = list(vc["link_ids"])
vc_pos = {lid: i for i, lid in enumerate(vc_ids)}
t_proj = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
xy = np.array([t_proj.transform(vc["lon"][vc_pos[lid]], vc["lat"][vc_pos[lid]]) for lid in link_ids])

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

np.savez_compressed(f"{GTS}/volume_point_graph.npz", A=A, link_ids=np.array(link_ids), k=K, length_scale=ls)
print(f"saved volume_point_graph.npz: A {A.shape}, nonzero frac={(A>1e-4).mean():.4f}")
