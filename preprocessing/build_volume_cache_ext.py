"""
Extend the traffic volume dataset to 2026-07 (raw xlsx already downloaded through 2026-07, but
the processed volume_hourly_cache.npz smhan's pipeline built stops at 2025-12 -- its months()
helper is hardcoded to range(2023,2026)). Parses the 7 missing months (2026_01..2026_07) directly
from the raw xlsx, restricted to the SAME 124 site codes already in volume_hourly_cache.npz (reuse
their lat/lon -- physical sensors, coordinates don't change), sums inbound+outbound directions per
site/date (matching the existing cache's per-site-not-per-direction granularity), and concatenates
onto the existing cache. Writes to THIS project's own directory -- never touches smhan's original
cache or raw files.
"""
import re
import numpy as np
import pandas as pd

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
TOPIS_DIR = "/home/smhan/uve_experiment/topis_volume"
SRC_CACHE = "/home/smhan/uve_experiment/pipeline_nowcast/volume_hourly_cache.npz"

VOL_HOURS = [f"{i}시" for i in range(24)]
SHEET_RE = re.compile(r"^\d{4}년 \d{2}월$")
EXT_MONTHS = [(2026, m) for m in range(1, 8)]  # Jan..Jul 2026

orig = np.load(SRC_CACHE, allow_pickle=True)
known_ids = set(orig["link_ids"].tolist())
print(f"existing cache: {len(known_ids)} sites, dates {orig['dates'].min()}..{orig['dates'].max()}")

new_rows = []  # (date_int, site, hour) -> volume
for (y, m) in EXT_MONTHS:
    path = f"{TOPIS_DIR}/{y}_{m:02d}.xlsx"
    xls = pd.ExcelFile(path)
    sheets = [s for s in xls.sheet_names if SHEET_RE.match(s)]
    assert len(sheets) == 1, f"{path}: expected 1 data sheet, got {sheets}"
    df = pd.read_excel(xls, sheet_name=sheets[0])
    df = df[df["지점번호"].isin(known_ids)]
    # sum inbound(유입) + outbound(유출) directions per (지점번호, 일자)
    g = df.groupby(["지점번호", "일자"])[VOL_HOURS].sum(min_count=1)
    for (site, date), row in g.iterrows():
        for h in range(24):
            v = row.iloc[h]
            if np.isfinite(v):
                new_rows.append((int(date), site, h, float(v)))
    print(f"{y}_{m:02d}: {len(df)} rows -> {len(g)} (site,date) groups, running total {len(new_rows)}")

new_df = pd.DataFrame(new_rows, columns=["date", "site", "hour", "volume"])
print(f"total new (date,site,hour) rows: {len(new_df)}")

# rebuild arrays in the SAME site order as the original cache, appending new dates
link_ids = list(orig["link_ids"])
site_pos = {s: i for i, s in enumerate(link_ids)}
new_dates_sorted = sorted(new_df["date"].unique())
n_new_days = len(new_dates_sorted)
date_pos = {d: i for i, d in enumerate(new_dates_sorted)}
n_sites = len(link_ids)

new_dates_arr = np.repeat(new_dates_sorted, 24)
new_hours_arr = np.tile(np.arange(24), n_new_days)
new_volume = np.full((n_sites, n_new_days * 24), np.nan, dtype=np.float32)
for row in new_df.itertuples(index=False):
    if row.site not in site_pos:
        continue
    si = site_pos[row.site]
    di = date_pos[row.date]
    new_volume[si, di * 24 + row.hour] = row.volume

ext_dates = np.concatenate([orig["dates"], new_dates_arr])
ext_hours = np.concatenate([orig["hours"], new_hours_arr])
ext_volume = np.concatenate([orig["volume"], new_volume], axis=1)

np.savez_compressed(
    f"{GTS}/volume_hourly_cache_ext.npz",
    volume=ext_volume, link_ids=orig["link_ids"], lat=orig["lat"], lon=orig["lon"],
    dates=ext_dates, hours=ext_hours,
)
print(f"saved volume_hourly_cache_ext.npz: volume {ext_volume.shape}, dates {ext_dates.min()}..{ext_dates.max()}")
print(f"new-range valid fraction: {(~np.isnan(new_volume)).mean():.4f}")
