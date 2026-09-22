# Data

Raw data is **not** included in this repository (size, and some sources require their own
access/download process). This page describes what each dataset actually contains — down to the
raw column/file names the `preprocessing/` scripts parse — and where to get it.

## 1. TOPIS traffic sensors (volume + speed)

- 121 point sensors reporting hourly vehicle volume (vehicles/hour); 396 link-level sensors
  reporting hourly speed (km/h). Both from Seoul's TOPIS (Transport Operation & Information
  Service).
- Download pages (not the Open API):
  - [속도 정보 | 서울시 교통정보 시스템 - TOPIS](https://topis.seoul.go.kr/refRoom/openRefRoom_1.do) (speed)
  - [교통량 정보 | 서울시 교통정보 시스템 - TOPIS](https://topis.seoul.go.kr/refRoom/openRefRoom_2.do) (volume)

**Raw file format.** One `.xlsx` per month, named `<year>_<month>.xlsx` (e.g. `2026_07.xlsx`),
under separate `topis_volume/` and `topis_speed/` directories.

- **Volume**: single data sheet per file, sheet name matching `<year>년 <month>월`. Columns
  include `지점번호` (site ID), `일자` (date), and 24 hourly columns `0시`..`23시` (volume for
  that hour). `preprocessing/build_volume_cache_ext.py` sums inbound + outbound directions per
  `(site, date)` to get one volume series per site.
- **Speed**: one row per (date, link), read with `openpyxl` in `read_only` mode. Column 0 is the
  date, column 3 is the link ID, columns 12–35 are the 24 hourly speed values.
  `preprocessing/build_speed_cache_ext.py` parses this directly (no pandas — the raw files are
  large enough that row-by-row `openpyxl` iteration was faster).

**Note on repo scope**: `build_volume_cache_ext.py` / `build_speed_cache_ext.py` in this repo
*extend* an already-aggregated `volume_hourly_cache.npz` / `speed_hourly_cache.npz` (built by an
earlier, upstream stage of the pipeline that is not included here) with newer months parsed
directly from the raw `.xlsx` files. If you're starting from scratch, you'll need to write the
initial raw-xlsx → `{volume,speed}_hourly_cache.npz` aggregation step yourself, matching the
column layout described above.

## 2. Population-mobility (OD) data

- Hourly origin–destination movement counts between the 500 administrative units (dong-level in
  Seoul, sgg-level just outside it) in the Seoul capital region, 2023-01 through 2026-08.
- Source: Seoul's "생활이동" (Living/Population Movement) OD dataset, jointly produced by the
  Seoul Metropolitan Government and KT.
  - [수도권 생활이동 (성 연령별, 도착지 기준) — Seoul Open Data Plaza](https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do)
  - [수도권 생활이동 (overview/visualization)](https://data.seoul.go.kr/dataVisual/seoul/capitalRegionLivingMigration.do)

**Raw file format.** One zip per day, `movement_<YYMMDD>.zip`, under a `movement_od/<YYYYMM>/`
directory (e.g. `movement_od/202607/movement_260701.zip`); each zip contains one CSV. Encoding is
inconsistent across the date range — older files are `cp949`, newer ones `utf-8`
(`preprocessing/build_od_tensor_full.py` tries `utf-8` first and falls back to `cp949`). Columns
used:

| column | meaning |
|---|---|
| `o_admdong_cd` | origin administrative-dong code |
| `d_admdong_cd` | destination administrative-dong code |
| `st_time_cd` | start time code (first 2 digits = hour) |
| `move_purpose` | trip purpose, 1–7, stored as float-formatted text (`"1.0"`, not `"1"`) — commute/school/shopping/tourism/hospital/home/other; see Appendix E of the paper for a purpose-split analysis using this field |
| `cnt` | trip count |

`preprocessing/build_od_tensor_full.py` filters to the 500 administrative-unit codes used in the
paper (`o_admdong_cd`/`d_admdong_cd` both in the fixed codeset), buckets by hour, and sums `cnt`
into the `(day, hour, 500, 500)` OD tensor.

## 3. Road network

- South Korea's national standard node-link road network dataset (표준노드링크), maintained by
  ITS Korea / the Ministry of Land, Infrastructure and Transport. Used by
  `preprocessing/build_routing_weights*.py` to route OD flow onto sensors (Section 4.2 of the
  paper).
  - [전국표준노드링크 자료실 (ITS Korea)](https://www.its.go.kr/nodelink/nodelinkRef) — the download page
  - Mirrored on the national portal: [국토교통부_표준노드링크 | 공공데이터포털](https://www.data.go.kr/data/15025526/fileData.do)

**Raw file format.** Two shapefiles, `MOCT_LINK.shp` and `MOCT_NODE.shp` (`cp949` encoding),
read with `geopandas`. Links carry `F_NODE`/`T_NODE` (from/to node ID), `LENGTH`, `ROAD_RANK`,
and `LINK_ID`; nodes carry `NODE_ID`. `build_routing_weights.py` builds a directed graph from
these (weighted by length and a road-hierarchy preference on `ROAD_RANK`) and runs Dijkstra from
every origin dong's centroid to every TOPIS sensor to get the static routing matrix $W$.

## Directory layout the preprocessing scripts expect

The scripts under `preprocessing/` currently hardcode paths from our own server (see the note in
the top-level README) and the placeholder `/path/to/raw_data`. At minimum you will need, locally:

```
/path/to/raw_data/movement_od/<YYYYMM>/movement_<YYMMDD>.zip   # population-mobility, daily
/path/to/raw_data/topis_volume/<year>_<month>.xlsx              # TOPIS volume, monthly
/path/to/raw_data/topis_speed/<year>_<month>.xlsx                # TOPIS speed, monthly
/path/to/raw_data/seoul_buildings/nodelink/MOCT_LINK.shp          # road network
/path/to/raw_data/seoul_buildings/nodelink/MOCT_NODE.shp
```

Adjust the path constants at the top of each `build_*.py` script to point at your own copy of
the raw data before running.
