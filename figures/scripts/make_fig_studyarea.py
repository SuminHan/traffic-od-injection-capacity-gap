"""Study-area map: Seoul capital-region boundary, the 500 OD administrative units, and the 121
volume-sensor locations, so a reader unfamiliar with Seoul's geography can immediately see the
study's spatial scope. Companion to Table 2 (dataset summary)."""
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad"
GTS = f"{SC}/gts"
OUT = f"{GTS}/paper_tkde/figs"

plt.rcParams.update({"font.size": 9, "font.family": "serif"})

# --- OD administrative units (500) and Seoul boundary ---
od = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
our_codes = set(str(int(c)) for c in od["codes"])

dong = gpd.read_file(f"{SC}/admdongkor/ver20260701/HangJeongDong_ver20260701.geojson")
dong["code8"] = dong["adm_cd2"].str[:-2]
dong["sgg5"] = dong["code8"].str[:5]
dong_sel = dong[dong["code8"].isin(our_codes) | dong["sgg5"].isin({c[:5] for c in our_codes})].copy()
seoul_boundary = unary_union(dong[dong["code8"].str[:2] == "11"].geometry)

# --- volume sensor locations ---
vc = np.load("/home/smhan/uve_experiment/pipeline_nowcast/volume_hourly_cache.npz", allow_pickle=True)
tt = np.load(f"{GTS}/traffic_tensor.npz", allow_pickle=True)
sensor_link_ids = list(tt["link_ids"])
vc_pos = {lid: i for i, lid in enumerate(list(vc["link_ids"]))}
sensor_lat = np.array([vc["lat"][vc_pos[lid]] for lid in sensor_link_ids])
sensor_lon = np.array([vc["lon"][vc_pos[lid]] for lid in sensor_link_ids])

fig, ax = plt.subplots(figsize=(4.8, 3.3))
dong_sel.boundary.plot(ax=ax, linewidth=0.15, color="#999999", zorder=1)
gpd.GeoSeries([seoul_boundary], crs=dong.crs).boundary.plot(ax=ax, linewidth=1.0, color="#333333", zorder=3)
ax.scatter(sensor_lon, sensor_lat, s=5, c="#c0392b", marker="o", linewidths=0,
           zorder=4, label=f"volume sensors ($n$=121)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Study area: Seoul capital region", fontsize=9)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9, markerscale=1.5)
ax.set_xlim(126.55, 127.35)
ax.set_ylim(37.35, 37.75)
ax.set_aspect("equal")
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=6.5)


fig.tight_layout()
fig.savefig(f"{OUT}/fig_studyarea.pdf")
print(f"wrote {OUT}/fig_studyarea.pdf")
print(f"{len(dong_sel)} OD-unit polygons shown, {len(sensor_lat)} volume sensors")
