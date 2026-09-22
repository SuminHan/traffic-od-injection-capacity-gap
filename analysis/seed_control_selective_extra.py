"""Seed-control negative control, extended to every (architecture, task) combination that has a
selective-injection result on disk.

Why this matters for the paper. Section 8.3 reports selective, validation-gated injection as the
paper's practical contribution: per fold, per sensor, keep whichever of the plain and injected
checkpoints has lower validation MSE. The random-mask control already in the paper asks "does a
RANDOM per-sensor mask help?", which never touches validation data and therefore cannot answer the
question a reviewer actually asks:

    "Does choosing, per sensor, whichever of two checkpoints scores better on a one-month
     validation window hand you a free test-set gain regardless of what those checkpoints are?"

seed_control_selective.py answered that for two combinations and found the free lunch is real but
architecture-dependent: -4.16% for GMAN/volume (seed swing 9.67%) versus -1.21% for DCRNN/speed
(seed swing 1.56%). Because the size of the free lunch tracks how much the two checkpoints disagree,
it cannot be subtracted from Section 8.3 as a single constant -- every combination needs its own
measurement. This script produces that measurement for the remaining combinations, so each row of
Table~selective can be reported net of its own selection-procedure baseline.

Protocol per fold, identical to selective_injection_all_combos.py except that BOTH arms are plain
(no-injection) models differing only in random seed:
  1. train a seed-1 twin of the existing seed-0 plain checkpoint,
  2. choose per sensor using ONLY that fold's validation window,
  3. apply the fixed per-fold selector to the already-computed test predictions.

Also reports `selective_vs_globalpick`: the gain over simply keeping whichever whole checkpoint is
better on validation. That isolates what PER-SENSOR selection adds beyond ordinary model selection,
which is the sharpest form of the reviewer's objection.

Shared code is imported read-only; the training scripts are invoked as subprocesses exactly as the
sweeps invoke them.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows

GTS = "/home/ncrc/work/gts"
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"
CONTROL_SEED = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# Architectures live in three registries with three matching training entry points.
EXTRA2_MODELS = {"stid", "stgcn"}
EXTRA_MODELS = {"pdformer", "mtgnn", "agcrn"}

DEFAULT_COMBOS = [
    "pdformer:volume", "pdformer:speed", "mtgnn:volume", "mtgnn:speed",
    "agcrn:volume", "agcrn:speed",
]


def registry_for(model):
    if model in EXTRA2_MODELS:
        from models_baselines_extra2 import MODELS_EXTRA2
        return MODELS_EXTRA2, "train_baseline_model_extra2.py"
    if model in EXTRA_MODELS:
        from models_baselines_extra import MODELS_EXTRA
        return MODELS_EXTRA, "train_baseline_model_extra.py"
    from models_baselines import MODELS
    return MODELS, "train_baseline_model.py"


def results_file(model, task):
    for cand in (f"multi_fold_baseline_{model}_{task}_results_ext30.json",
                 f"multi_fold_{model}_{task}_results_ext30.json"):
        if os.path.exists(f"{GTS}/{cand}"):
            return cand
    raise FileNotFoundError(f"no results file for {model}/{task}")


def load_series(args, task):
    vkey = "volume" if task == "volume" else "speed"
    tfile = "traffic_tensor" if task == "volume" else "speed_tensor"
    tt = np.load(f"{GTS}/{tfile}{args.dataset_suffix}.npz", allow_pickle=True)
    vol = tt[vkey]
    link_ids = list(tt["link_ids"])
    n_sensors, n_days, n_hours = vol.shape
    vol_flat = vol.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)
    mu = vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].mean()
    sd = vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].std() + 1e-3
    vol_n = (vol_flat - mu) / sd

    g = np.load(f"{GTS}/{vkey}_point_graph.npz", allow_pickle=True)
    gpos = {lid: i for i, lid in enumerate(list(g["link_ids"]))}
    sel = np.array([gpos[l] for l in link_ids])
    A = torch.tensor(g["A"][np.ix_(sel, sel)], dtype=torch.float32, device=device)

    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    days = d["days"]
    day_objs = [dt.datetime.strptime("20" + str(dd), "%Y%m%d").date() for dd in days[:n_days]]
    hours_arr = np.tile(np.arange(n_hours), n_days)
    dows = np.repeat(np.array([do.weekday() for do in day_objs]), n_hours)
    fmt = lambda dd: dd.strftime("%Y%m%d")
    is_hol = np.repeat(np.array([fmt(x) in KR_HOLIDAYS for x in day_objs], dtype=np.float32), n_hours)
    is_bef = np.repeat(np.array([fmt(x + dt.timedelta(days=1)) in KR_HOLIDAYS for x in day_objs], dtype=np.float32), n_hours)
    is_aft = np.repeat(np.array([fmt(x - dt.timedelta(days=1)) in KR_HOLIDAYS for x in day_objs], dtype=np.float32), n_hours)
    time_feat = np.stack([
        np.sin(2 * np.pi * hours_arr / 24), np.cos(2 * np.pi * hours_arr / 24),
        np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7),
        is_bef, is_hol, is_aft,
    ], axis=-1).astype(np.float32)

    return (torch.tensor(vol_n, dtype=torch.float32, device=device),
            torch.tensor(time_feat, device=device),
            float(mu), float(sd), n_sensors, n_hours, A)


def eval_persensor_mse(model, args, vol_t, time_t, mu, sd, n_sensors, lo, hi):
    P, Q = args.p_len, args.q_len
    idx_list = build_windows(n_hours_per_day=24, day_range=(lo, hi), p_len=P, q_len=Q)
    stn_idx = torch.arange(n_sensors, device=device)
    sq, cnt = np.zeros(n_sensors), np.zeros(n_sensors)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx_list), 32):
            batch = idx_list[i:i + 32]
            xe = torch.stack([vol_t[s:s + P] for s in batch]).unsqueeze(-1)
            te = torch.stack([time_t[s:s + P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            xd = torch.stack([vol_t[s + P:s + P + Q] for s in batch])
            pred = model(torch.cat([xe, te], dim=-1).transpose(1, 2), stn_idx)
            se = (xd.cpu().numpy() * sd + mu - (pred.transpose(1, 2).cpu().numpy() * sd + mu)) ** 2
            sq += se.sum(axis=(0, 1))
            cnt += se.shape[0] * se.shape[1]
    return sq / cnt


def train_twin(args, model, task, train_script, out_dir):
    if os.path.exists(f"{GTS}/{out_dir}/test_predictions.csv"):
        return True
    cmd = [PY, "-u", train_script, "--model", model, "--task", task,
           "--dataset_suffix", args.dataset_suffix, "--use_od_injection", "0",
           "--hidden", str(args.hidden), "--p_len", str(args.p_len), "--q_len", str(args.q_len),
           "--epochs", str(args.epochs), "--lr", str(args.lr),
           "--batch_size", str(args.batch_size), "--patience", str(args.patience),
           "--loss_fn", getattr(args, "loss_fn", "smooth_l1"),
           "--train_lo", str(args.train_lo), "--train_hi", str(args.train_hi),
           "--val_lo", str(args.val_lo), "--val_hi", str(args.val_hi),
           "--test_lo", str(args.test_lo), "--test_hi", str(args.test_hi),
           "--out", out_dir, "--seed", str(CONTROL_SEED)]
    r = subprocess.run(cmd, cwd=GTS, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"      !! FAILED {out_dir}\n{r.stdout[-900:]}\n{r.stderr[-900:]}", flush=True)
        return False
    return True


def run_combo(model, task, max_folds, summary_path, all_summary):
    MODELS, train_script = registry_for(model)
    vkey = "true_volume" if task == "volume" else "true_speed"
    pkey = "pred_volume" if task == "volume" else "pred_speed"
    print(f"\n{'=' * 72}\nSEED CONTROL: {model} / {task}\n{'=' * 72}", flush=True)

    results = json.load(open(f"{GTS}/{results_file(model, task)}"))
    plain_by_fold = {e["fold_start"]: e for e in results if e["variant"] == "plain"}
    csv_path = f"{GTS}/seed_control_{model}_{task}_fold_results.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
    done = {r["fold"] for r in rows}
    t0 = time.time()

    for fold in sorted(plain_by_fold)[:max_folds]:
        if fold in done:
            continue
        p0 = plain_by_fold[fold]
        a = argparse.Namespace(**p0["args"])
        a.model = model
        out1 = f"seedctl_{model}_{task}_fold{fold}_s{CONTROL_SEED}_ext30"
        if not train_twin(a, model, task, train_script, out1):
            continue

        vol_t, time_t, mu, sd, n_sensors, n_hours, A = load_series(a, task)
        mk = lambda: MODELS[model](A, 1 + TIME_FEAT_DIM, n_sensors, H=a.hidden, q_len=a.q_len).to(device)
        m0, m1 = mk(), mk()
        m0.load_state_dict(torch.load(f"{GTS}/{p0['out']}/best.pt", weights_only=True))
        m1.load_state_dict(torch.load(f"{GTS}/{out1}/best.pt", weights_only=True))

        v0 = eval_persensor_mse(m0, a, vol_t, time_t, mu, sd, n_sensors, a.val_lo, a.val_hi)
        v1 = eval_persensor_mse(m1, a, vol_t, time_t, mu, sd, n_sensors, a.val_lo, a.val_hi)
        choose1 = v1 < v0

        t0df = pd.read_csv(f"{GTS}/{p0['out']}/test_predictions.csv").sort_values(
            ["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
        t1df = pd.read_csv(f"{GTS}/{out1}/test_predictions.csv").sort_values(
            ["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
        assert (t0df["sensor_idx"].values == t1df["sensor_idx"].values).all()

        true_v = t0df[vkey].values
        pred_sel = np.where(choose1[t0df["sensor_idx"].values], t1df[pkey].values, t0df[pkey].values)
        mse0 = float(np.mean((true_v - t0df[pkey].values) ** 2))
        mse1 = float(np.mean((true_v - t1df[pkey].values) ** 2))
        msel = float(np.mean((true_v - pred_sel) ** 2))
        gp_is1 = bool(v1.mean() < v0.mean())

        rows.append({"fold": fold, "mse_seed0": mse0, "mse_seed1": mse1, "mse_selective": msel,
                     "mse_globalpick": mse1 if gp_is1 else mse0, "globalpick_is_seed1": gp_is1,
                     "n_sensors_chose_seed1": int(choose1.sum()), "n_sensors": n_sensors})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  fold {fold:2d}: s0={mse0:11.3f} s1={mse1:11.3f} sel={msel:11.3f} "
              f"({choose1.sum()}/{n_sensors}) [{(msel - mse0) / mse0 * 100:+.2f}%] "
              f"({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        return
    df = pd.DataFrame(rows)
    n = len(df)
    pct = (df.mse_selective - df.mse_seed0) / df.mse_seed0 * 100
    pct_gp = (df.mse_selective - df.mse_globalpick) / df.mse_globalpick * 100
    pct_s1 = (df.mse_seed1 - df.mse_seed0) / df.mse_seed0 * 100
    p = stats.ttest_rel(df.mse_selective, df.mse_seed0)[1] if n >= 2 else np.nan
    p_gp = stats.ttest_rel(df.mse_selective, df.mse_globalpick)[1] if n >= 2 else np.nan
    all_summary.append({
        "model": model, "task": task, "n_folds": n,
        "seedctl_selective_mean_pct": float(pct.mean()),
        "seedctl_selective_wins": int((df.mse_selective < df.mse_seed0).sum()),
        "seedctl_selective_p": float(p),
        "selective_vs_globalpick_mean_pct": float(pct_gp.mean()),
        "selective_vs_globalpick_wins": int((df.mse_selective < df.mse_globalpick).sum()),
        "selective_vs_globalpick_p": float(p_gp),
        "seed_swing_abs_mean_pct": float(pct_s1.abs().mean()),
        "mean_sensors_chose_seed1": float(df.n_sensors_chose_seed1.mean()),
        "n_sensors_total": int(df.n_sensors.iloc[0]),
    })
    pd.DataFrame(all_summary).to_csv(summary_path, index=False)
    print(f"\n>>> {model}/{task}: free lunch {pct_gp.mean():+.2f}% (p={p_gp:.3g}); "
          f"vs seed0 {pct.mean():+.2f}%; seed swing |{pct_s1.abs().mean():.2f}%|", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default=",".join(DEFAULT_COMBOS),
                    help="comma-separated model:task pairs")
    ap.add_argument("--max_folds", type=int, default=30)
    opt = ap.parse_args()

    # NOTE: writes to our own file, not seed_control_all_summary.csv -- that file is owned by a
    # different account on this shared server (smhan) and we don't have write permission on it;
    # merge these rows into it manually/in-prose instead of trying to write there directly.
    summary_path = f"{GTS}/seed_control_extra_summary.csv"
    all_summary = []
    if os.path.exists(summary_path):
        all_summary = pd.read_csv(summary_path).to_dict("records")
    have = {(r["model"], r["task"]) for r in all_summary}

    for c in opt.combos.split(","):
        model, task = c.strip().split(":")
        if (model, task) in have:
            print(f"[skip] {model}/{task} already summarized", flush=True)
            continue
        try:
            run_combo(model, task, opt.max_folds, summary_path, all_summary)
        except Exception as e:
            print(f"!! {model}/{task} failed: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 72)
    print("SEED CONTROL (ALL COMBOS) SUMMARY")
    print("=" * 72)
    if all_summary:
        print(pd.DataFrame(all_summary).to_string(index=False))
    print("\nSEED_CONTROL_ALL_COMBOS_DONE")


if __name__ == "__main__":
    main()
