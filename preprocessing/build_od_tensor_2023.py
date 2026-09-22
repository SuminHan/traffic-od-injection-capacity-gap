"""Build the (365 days, 24h, 500, 500) OD tensor for all of 2023 -- used as the sweep dataset
instead of the 1-month July 2026 sample, for a more robust architecture-selection comparison."""
import glob, os, time, zipfile
import numpy as np
import pandas as pd
from multiprocessing import Pool

SC = "/home/ncrc/work/gts"
MOVE_ROOT = "/path/to/raw_data/movement_od"
N_WORKERS = 20

d0 = np.load(f"{SC}/od_tensor_202607.npz", allow_pickle=True)
CODES = list(d0["codes"])
CODE_IDX = {c: i for i, c in enumerate(CODES)}
N = len(CODES)


def process_day(seq):
    yyyymm = "20" + seq[:4]
    path = f"{MOVE_ROOT}/{yyyymm}/movement_{seq}.zip"
    try:
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            raw = z.read(name)
        # encoding changed somewhere along the way (older files are cp949, newer are utf-8) --
        # try utf-8 first, fall back to cp949 rather than guessing which year switched where
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp949")
        import io
        parts = []
        for chunk in pd.read_csv(io.StringIO(text), usecols=["o_admdong_cd", "d_admdong_cd", "st_time_cd", "cnt"],
                                  dtype={"st_time_cd": str, "o_admdong_cd": str, "d_admdong_cd": str},
                                  chunksize=3_000_000):
            chunk = chunk[chunk["o_admdong_cd"].isin(CODE_IDX) & chunk["d_admdong_cd"].isin(CODE_IDX)]
            if not len(chunk):
                continue
            chunk["hour"] = chunk["st_time_cd"].str[:2].astype(int)
            parts.append(chunk.groupby(["o_admdong_cd", "d_admdong_cd", "hour"])["cnt"].sum())
        if not parts:
            return (seq, None)
        day_od = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
        o_codes, d_codes, h_vals = zip(*day_od.index)
        oi = np.array([CODE_IDX[c] for c in o_codes], dtype=np.int32)
        dj = np.array([CODE_IDX[c] for c in d_codes], dtype=np.int32)
        h = np.array(h_vals, dtype=np.int8)
        v = day_od.values.astype(np.float32)
        return (seq, (h, oi, dj, v))
    except Exception as e:
        return (seq, ("ERROR", str(e)))


if __name__ == "__main__":
    t0 = time.time()
    all_days = sorted(os.path.basename(p).replace("movement_", "").replace(".zip", "")
                       for p in glob.glob(f"{MOVE_ROOT}/2023*/movement_23*.zip"))
    print(f"{len(all_days)} 2023 day-zips found ({time.time()-t0:.0f}s)")

    od = np.zeros((len(all_days), 24, N, N), dtype=np.float32)
    day_pos = {seq: i for i, seq in enumerate(all_days)}
    failed = []

    with Pool(N_WORKERS) as pool:
        for i, (seq, result) in enumerate(pool.imap_unordered(process_day, all_days, chunksize=4)):
            if result is None:
                continue
            if isinstance(result[0], str) and result[0] == "ERROR":
                failed.append((seq, result[1])); continue
            h, oi, dj, v = result
            od[day_pos[seq], h, oi, dj] = v
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(all_days)} days done ({time.time()-t0:.0f}s)")

    print(f"all days processed ({time.time()-t0:.0f}s), {len(failed)} failed: {failed[:10]}")
    np.savez_compressed(f"{SC}/od_tensor_2023.npz", od=od, codes=np.array(CODES), days=np.array(all_days))
    print(f"saved od_tensor_2023.npz {od.shape} ({time.time()-t0:.0f}s)")
