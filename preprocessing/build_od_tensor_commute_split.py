"""
Data-enrichment experiment: the raw movement_od zips have a `move_purpose` column (1-7) our
existing od_tensor_full.npz throws away entirely (it only keeps o/d/hour/cnt). Empirically
characterized (sample day 2026-08-06, 6.3M rows) via hour-of-day distribution:
  code 1 (19% of trips): sharp 6-9am peak, near-zero at night -> classic commute-to-work signature
  code 2 (1.8%): similar AM peak, smaller/narrower population, lowest foreign share -> commute-to-
                 school-like
  code 3 (35%): flat-to-rising through the day, broad 16-20h peak -> classic return-home signature
  codes 4,5,6,7: smaller/mixed signatures (shopping/hospital/tourism/other), not cleanly separable
                 without the official code layout doc (not found on this server; Seoul Open Data
                 Plaza's "수도권 생활이동(목적별) 레이아웃" doc would have the authoritative mapping
                 but isn't downloaded here).
Given that ambiguity, this builds the SAFEST actionable split: COMMUTE-LIKE (codes 1+2, unambiguous
sharp-AM-peak signature regardless of their exact official label) vs REST (everything else) --
rather than guessing all 7 labels. Produces two separate (day,hour,500,500) OD tensors for the
2023 period (the destination-share estimation window every routing-weight script already uses),
so a "commute-weighted" routing matrix can be built and compared against the current all-purpose one.
"""
import glob, os, time, zipfile
import numpy as np
import pandas as pd
from multiprocessing import Pool

SC = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
MOVE_ROOT = "/home/smhan/uve_experiment/movement_od"
N_WORKERS = 20
COMMUTE_CODES = {"1.0", "2.0"}  # move_purpose is stored as float-formatted text ("1.0"), not bare "1"

d0 = np.load(f"{SC}/od_tensor_202607.npz", allow_pickle=True)
CODES = list(d0["codes"])
CODE_IDX = {c: i for i, c in enumerate(CODES)}
N = len(CODES)


def find_2023_zips():
    paths = sorted(glob.glob(f"{MOVE_ROOT}/2023*/movement_*.zip"))
    days = []
    for p in paths:
        seq = os.path.basename(p).replace("movement_", "").replace(".zip", "")
        if len(seq) == 6 and seq.isdigit():
            days.append(seq)
    return sorted(set(days))


def process_day(seq):
    yyyymm = "20" + seq[:4]
    path = f"{MOVE_ROOT}/{yyyymm}/movement_{seq}.zip"
    try:
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            raw = z.read(name)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp949")
        import io
        parts_commute, parts_rest = [], []
        for chunk in pd.read_csv(io.StringIO(text),
                                  usecols=["o_admdong_cd", "d_admdong_cd", "st_time_cd", "move_purpose", "cnt"],
                                  dtype={"st_time_cd": str, "o_admdong_cd": str, "d_admdong_cd": str,
                                         "move_purpose": str},
                                  chunksize=3_000_000):
            chunk = chunk[chunk["o_admdong_cd"].isin(CODE_IDX) & chunk["d_admdong_cd"].isin(CODE_IDX)]
            if not len(chunk):
                continue
            chunk["hour"] = chunk["st_time_cd"].str[:2].astype(int)
            is_commute = chunk["move_purpose"].isin(COMMUTE_CODES)
            g_commute = chunk[is_commute].groupby(["o_admdong_cd", "d_admdong_cd", "hour"])["cnt"].sum()
            g_rest = chunk[~is_commute].groupby(["o_admdong_cd", "d_admdong_cd", "hour"])["cnt"].sum()
            if len(g_commute): parts_commute.append(g_commute)
            if len(g_rest): parts_rest.append(g_rest)

        def finalize(parts):
            if not parts:
                return None
            day_od = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
            o_codes, d_codes, h_vals = zip(*day_od.index)
            oi = np.array([CODE_IDX[c] for c in o_codes], dtype=np.int32)
            dj = np.array([CODE_IDX[c] for c in d_codes], dtype=np.int32)
            h = np.array(h_vals, dtype=np.int8)
            v = day_od.values.astype(np.float32)
            return (h, oi, dj, v)

        return (seq, finalize(parts_commute), finalize(parts_rest))
    except Exception as e:
        return (seq, ("ERROR", str(e)), None)


if __name__ == "__main__":
    t0 = time.time()
    all_days = find_2023_zips()
    print(f"{len(all_days)} day-zips found for 2023 ({time.time()-t0:.0f}s)")

    od_commute = np.zeros((len(all_days), 24, N, N), dtype=np.float32)
    od_rest = np.zeros((len(all_days), 24, N, N), dtype=np.float32)
    day_pos = {seq: i for i, seq in enumerate(all_days)}
    failed = []

    with Pool(N_WORKERS) as pool:
        for i, (seq, r_commute, r_rest) in enumerate(pool.imap_unordered(process_day, all_days, chunksize=4)):
            if isinstance(r_commute, tuple) and isinstance(r_commute[0], str) and r_commute[0] == "ERROR":
                failed.append((seq, r_commute[1]))
                continue
            di = day_pos[seq]
            if r_commute is not None:
                h, oi, dj, v = r_commute
                od_commute[di, h, oi, dj] = v
            if r_rest is not None:
                h, oi, dj, v = r_rest
                od_rest[di, h, oi, dj] = v
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(all_days)} days done ({time.time()-t0:.0f}s)")

    print(f"all days processed ({time.time()-t0:.0f}s), {len(failed)} failed: {failed[:5]}")
    np.savez_compressed(f"{SC}/od_tensor_2023_commute_split.npz",
                         od_commute=od_commute, od_rest=od_rest,
                         codes=np.array(CODES), days=np.array(all_days))
    print(f"saved od_tensor_2023_commute_split.npz, total {time.time()-t0:.0f}s")
    print(f"commute share of total volume: {od_commute.sum()/(od_commute.sum()+od_rest.sum()):.3f}")
