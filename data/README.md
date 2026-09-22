# Data

Raw data is **not** included in this repository (size, and some sources require their own
access/download process). This describes what each dataset is and where to get it; the
`preprocessing/` scripts turn the raw downloads into the tensors the model code consumes.

## 1. TOPIS traffic sensors (volume + speed)

- 121 point sensors reporting hourly vehicle volume (vehicles/hour).
- 396 link-level sensors reporting hourly speed (km/h).
- Source: Seoul's TOPIS (Transport Operation & Information Service) traffic data, accessed via
  the Seoul Open Data Plaza (data.seoul.go.kr) / T-Data Seoul Traffic Big Data Platform open
  APIs — an API key from data.seoul.go.kr is required.
  - [TOPIS Open API portal](https://topis.seoul.go.kr/refRoom/openRefRoom_4.do)
  - [Volume ("교통량 정보") API](https://topis.seoul.go.kr/refRoom/openRefRoom_2.do)
  - [Speed ("속도 정보") API](https://topis.seoul.go.kr/refRoom/openRefRoom_1.do)
  - [T-Data Seoul Traffic Big Data Platform](https://t-data.seoul.go.kr/)
  - Mirrored on the national public data portal: [실시간 도로 소통 정보 | 공공데이터포털](https://www.data.go.kr/data/15058364/openapi.do)

**Verify before relying on this**: I (the assistant) found these via web search rather than
having used them myself to build this dataset -- please confirm these are the same source /
fields the original download used before citing them as the paper's data source.

## 2. Population-mobility (OD) data

- Hourly origin-destination movement counts between the 500 administrative units (dong-level in
  Seoul, sgg-level just outside it) in the Seoul capital region, 2023-01 through 2026-08.
- Carries a `move_purpose` field (used in the purpose-split exploratory analysis, Appendix E).
- This matches the profile of Seoul's "생활이동" (Living/Population Movement) OD dataset,
  jointly produced by the Seoul Metropolitan Government and KT, distributed with exactly this
  kind of daily/hourly, purpose-coded (commute/school/shopping/tourism/hospital/home/other),
  administrative-dong-to-dong structure:
  - [수도권 생활이동 (Seoul Open Data Plaza)](https://data.seoul.go.kr/dataVisual/seoul/capitalRegionLivingMigration.do)
  - [서울 생활이동 (Seoul Open Data Plaza)](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingMigration.do)
  - [수도권 생활이동 (성 연령별, 도착지 기준) 데이터셋](https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do)
  - Mirrored on the national portal: [서울특별시_생활이동 인구 데이터 | 공공데이터포털](https://www.data.go.kr/data/15087677/fileData.do)

**Verify before relying on this**: same caveat as above -- I have not confirmed this is the
exact dataset/vintage the `movement_*.zip` files came from, only that the field structure
matches. Please confirm (or correct) before treating this as the paper's actual data source.

## 3. Road network

- Used by `build_routing_weights*.py` (which reference a local `.../seoul_buildings/nodelink`
  directory) to route OD flow onto sensors (Section 4.2 of the paper).
- Likely South Korea's national standard node-link road network dataset (표준노드링크),
  maintained by ITS Korea / the Ministry of Land, Infrastructure and Transport:
  - [ITS Korea 표준노드링크 (National Transport Information Center)](https://www.its.go.kr/nodelink/)
  - [전국표준노드링크 자료실](https://www.its.go.kr/nodelink/nodelinkRef)
  - Mirrored on the national portal: [국토교통부_표준노드링크 | 공공데이터포털](https://www.data.go.kr/data/15025526/fileData.do)

**Verify before relying on this**: same caveat -- the directory name is a strong hint, not a
confirmed source; please double-check before citing.

## Directory layout the preprocessing scripts expect

The scripts under `preprocessing/` currently hardcode paths from our own server (see the note in
the top-level README). At minimum you will need, locally:

```
<raw_data_root>/movement_od/<YYYYMM>/movement_<YYYYMMDD>.zip   # population-mobility, daily
<raw_data_root>/topis_volume/...                                # TOPIS volume sensor logs
<raw_data_root>/topis_speed/...                                 # TOPIS speed sensor logs
```

Adjust the path constants at the top of each `build_*.py` script to point at your own copy of
the raw data before running.
