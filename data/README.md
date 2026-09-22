# Data

Raw data is **not** included in this repository (size, and some sources require their own
access/download process). This describes what each dataset is and where to get it; the
`preprocessing/` scripts turn the raw downloads into the tensors the model code consumes.

## 1. TOPIS traffic sensors (volume + speed)

- 121 point sensors reporting hourly vehicle volume (vehicles/hour).
- 396 link-level sensors reporting hourly speed (km/h).
- Source: Seoul's TOPIS (Transport Operation & Information Service) traffic data.
- **TODO (fill in before making this public / final release):** the exact TOPIS open-data
  portal endpoint / API used to download the raw sensor logs.

## 2. Population-mobility (OD) data

- Hourly origin-destination movement counts between the 500 administrative units (dong-level in
  Seoul, sgg-level just outside it) in the Seoul capital region, 2023-01 through 2026-08.
- Carries a `move_purpose` field (used in the purpose-split exploratory analysis, Appendix E).
- **TODO (fill in before making this public / final release):** the data provider and
  access/download process for the mobility (`movement_*.zip`) files.

## 3. Road network

- Used by `build_routing_weights*.py` to route OD flow onto sensors (Section 4.2 of the paper).
- **TODO:** source of the road-network graph (e.g. OSM extract) used to compute shortest paths.

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
