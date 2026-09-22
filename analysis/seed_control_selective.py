"""
Seed-control negative control for the validation-gated selective injection of Section 8.3.

The external review's second critical objection: the random-mask control already in the paper
only asks "does a RANDOM per-sensor mask help?", which never touches the validation data. The
question a reviewer actually asks is different:

    "Does the act of picking, per sensor, whichever of two checkpoints has lower VALIDATION MSE
     hand you a free test-set gain regardless of what those two checkpoints are?"

With 121-396 sensors and a one-month validation window, per-sensor validation MSE is noisy and
validation/test are correlated (seasonality, sensor-specific bias), so the selection procedure
itself could manufacture gains with no OD signal involved at all.

This script measures exactly that. For each fold we train a SECOND plain (no-injection)
checkpoint that differs from the existing one only in random seed, then run the identical
per-sensor validation-gated selection between plain(seed 0) and plain(seed 1), and report the
test %MSE change against plain(seed 0) -- the same reference Section 8.3 reports against.

Interpretation:
  seed-control ~ -0.3%  -> Section 8.3's -1.5%..-14% is mostly real signal selection.
  seed-control ~ -2..3% -> the small effects (DCRNN/speed -1.51%, GWNET/speed -1.87%,
                           STAEformer/speed -1.23%) lose their explanatory power.

Selection protocol is copied verbatim in behaviour from selective_injection_all_combos.py
(validation-window forward pass from the saved best.pt, fixed per-fold selector applied to the
already-computed test predictions), with the only change being that BOTH arms are plain models.

Nothing in this file modifies shared code: train_gts_od / models_baselines are imported read-only
and train_baseline_model.py is invoked as a subprocess, exactly as the live sweeps do.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from models_baselines import MODELS

GTS = "/home/ncrc/work/gts"
PY = "/home/ncrc/miniconda3/envs/trajtok/bin/python3"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ordered cheapest-first so the fast combo yields an early read on the answer.
COMBOS = [
    ("gman", "volume", "multi_fold_baseline_gman_volume_results_ext30.json"),
    ("dcrnn", "speed", "multi_fold_baseline_dcrnn_speed_results_ext30.json"),
]
CONTROL_SEED = 1


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

    graph_path = f"{GTS}/{'volume' if task == 'volume' else 'speed'}_point_graph.npz"
    g = np.load(graph_path, allow_pickle=True)
    g_ids = list(g["link_ids"])
    gpos = {lid: i for i, lid in enumerate(g_ids)}
    sel = np.array([gpos[l] for l in link_ids])
    A = torch.tensor(g["A"][np.ix_(sel, sel)], dtype=torch.float32, device=device)

    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    days = d["days"]
    day_objs = [dt.datetime.strptime("20" + str(dd), "%Y%m%d").date() for dd in days[:n_days]]
    hours_arr = np.tile(np.arange(n_hours), n_days)
    dows = np.repeat(np.array([do.weekday() for do in day_objs]), n_hours)

    def _fmt(dd):
        return dd.strftime("%Y%m%d")

    is_hol = np.repeat(np.array([_fmt(dd) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    is_before = np.repeat(np.array([_fmt(dd + dt.timedelta(days=1)) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    is_after = np.repeat(np.array([_fmt(dd - dt.timedelta(days=1)) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    time_feat = np.stack([
        np.sin(2 * np.pi * hours_arr / 24), np.cos(2 * np.pi * hours_arr / 24),
        np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7),
        is_before, is_hol, is_after,
    ], axis=-1).astype(np.float32)

    vol_t = torch.tensor(vol_n, dtype=torch.float32, device=device)
    time_t = torch.tensor(time_feat, device=device)
    return vol_t, time_t, float(mu), float(sd), link_ids, n_sensors, n_hours, A


def build_plain_model(args, A, n_sensors):
    Fday = 1 + TIME_FEAT_DIM
    return MODELS[args.model](A, Fday, n_sensors, H=args.hidden, q_len=args.q_len).to(device)


def eval_persensor_mse(model, args, vol_t, time_t, mu, sd, n_sensors, n_hours, lo, hi):
    P, Q = args.p_len, args.q_len
    idx_list = build_windows(n_hours_per_day=24, day_range=(lo, hi), p_len=P, q_len=Q)
    stn_idx = torch.arange(n_sensors, device=device)
    sq_err_sum = np.zeros(n_sensors, dtype=np.float64)
    n_count = np.zeros(n_sensors, dtype=np.float64)
    bs = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx_list), bs):
            batch = idx_list[i:i + bs]
            xe = torch.stack([vol_t[s:s + P] for s in batch]).unsqueeze(-1)
            te = torch.stack([time_t[s:s + P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            xd = torch.stack([vol_t[s + P:s + P + Q] for s in batch])
            x_in = torch.cat([xe, te], dim=-1).transpose(1, 2)
            pred = model(x_in, stn_idx)
            true_denorm = xd.cpu().numpy() * sd + mu
            pred_denorm = pred.transpose(1, 2).cpu().numpy() * sd + mu
            se = (true_denorm - pred_denorm) ** 2
            sq_err_sum += se.sum(axis=(0, 1))
            n_count += se.shape[0] * se.shape[1]
    return sq_err_sum / n_count


def train_control_checkpoint(args, model_name, task, out_dir):
    """Train the seed-CONTROL_SEED plain twin of an existing plain fold run."""
    if os.path.exists(f"{GTS}/{out_dir}/test_predictions.csv"):
        print(f"    [skip] {out_dir} already present", flush=True)
        return True
    cmd = [
        PY, "-u", "train_baseline_model_seedctl.py",
        "--model", model_name, "--task", task,
        "--dataset_suffix", args.dataset_suffix,
        "--use_od_injection", "0",
        "--hidden", str(args.hidden), "--p_len", str(args.p_len), "--q_len", str(args.q_len),
        "--epochs", str(args.epochs), "--lr", str(args.lr),
        "--batch_size", str(args.batch_size), "--patience", str(args.patience),
        "--loss_fn", getattr(args, "loss_fn", "smooth_l1"),
        "--train_lo", str(args.train_lo), "--train_hi", str(args.train_hi),
        "--val_lo", str(args.val_lo), "--val_hi", str(args.val_hi),
        "--test_lo", str(args.test_lo), "--test_hi", str(args.test_hi),
        "--out", out_dir, "--seed", str(CONTROL_SEED),
    ]
    print(f"    training {out_dir} (seed {CONTROL_SEED})", flush=True)
    r = subprocess.run(cmd, cwd=GTS, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    !! FAILED {out_dir}: {r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_folds", type=int, default=30)
    opt = ap.parse_args()

    all_summary = []
    for model_name, task, results_file in COMBOS:
        t0 = time.time()
        vkey = "true_volume" if task == "volume" else "true_speed"
        pkey = "pred_volume" if task == "volume" else "pred_speed"
        print(f"\n{'=' * 72}\nSEED CONTROL: {model_name} / {task}\n{'=' * 72}", flush=True)

        results = json.load(open(f"{GTS}/{results_file}"))
        plain_by_fold = {e["fold_start"]: e for e in results if e["variant"] == "plain"}

        fold_rows = []
        for fold in sorted(plain_by_fold.keys())[:opt.max_folds]:
            p0 = plain_by_fold[fold]
            args_p = argparse.Namespace(**p0["args"])
            args_p.model = model_name
            out_s1 = f"seedctl_{model_name}_{task}_fold{fold}_s{CONTROL_SEED}_ext30"

            print(f"  fold {fold}:", flush=True)
            if not train_control_checkpoint(args_p, model_name, task, out_s1):
                continue

            vol_t, time_t, mu, sd, link_ids, n_sensors, n_hours, A = load_series(args_p, task)

            m0 = build_plain_model(args_p, A, n_sensors)
            m0.load_state_dict(torch.load(f"{GTS}/{p0['out']}/best.pt", weights_only=True))
            m1 = build_plain_model(args_p, A, n_sensors)
            m1.load_state_dict(torch.load(f"{GTS}/{out_s1}/best.pt", weights_only=True))

            val0 = eval_persensor_mse(m0, args_p, vol_t, time_t, mu, sd, n_sensors, n_hours, args_p.val_lo, args_p.val_hi)
            val1 = eval_persensor_mse(m1, args_p, vol_t, time_t, mu, sd, n_sensors, n_hours, args_p.val_lo, args_p.val_hi)
            choose_s1 = val1 < val0

            t0df = pd.read_csv(f"{GTS}/{p0['out']}/test_predictions.csv")
            t1df = pd.read_csv(f"{GTS}/{out_s1}/test_predictions.csv")
            t0df = t0df.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
            t1df = t1df.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
            assert (t0df["sensor_idx"].values == t1df["sensor_idx"].values).all()

            sel_mask = choose_s1[t0df["sensor_idx"].values]
            pred_sel = np.where(sel_mask, t1df[pkey].values, t0df[pkey].values)
            true_vals = t0df[vkey].values

            mse_s0 = float(np.mean((true_vals - t0df[pkey].values) ** 2))
            mse_s1 = float(np.mean((true_vals - t1df[pkey].values) ** 2))
            mse_sel = float(np.mean((true_vals - pred_sel) ** 2))

            # Honest baseline for "does PER-SENSOR selection add anything?": a practitioner with
            # the same validation window would first just keep whichever whole checkpoint is
            # better on validation. Per-sensor selection only earns its keep if it beats that.
            global_pick_is_s1 = bool(val1.mean() < val0.mean())
            mse_globalpick = mse_s1 if global_pick_is_s1 else mse_s0

            fold_rows.append({
                "fold": fold, "mse_seed0": mse_s0, "mse_seed1": mse_s1,
                "mse_selective": mse_sel, "mse_globalpick": mse_globalpick,
                "globalpick_is_seed1": global_pick_is_s1,
                "n_sensors_chose_seed1": int(choose_s1.sum()),
                "n_sensors": n_sensors,
            })
            print(f"    seed0={mse_s0:.3f} seed1={mse_s1:.3f} selective={mse_sel:.3f} "
                  f"({choose_s1.sum()}/{n_sensors} chose seed1) "
                  f"[{(mse_sel - mse_s0) / mse_s0 * 100:+.2f}%] ({time.time() - t0:.0f}s)", flush=True)

            pd.DataFrame(fold_rows).to_csv(
                f"{GTS}/seed_control_{model_name}_{task}_fold_results.csv", index=False)

        if not fold_rows:
            print(f"  no folds completed for {model_name}/{task}", flush=True)
            continue

        df = pd.DataFrame(fold_rows)
        n = len(df)
        pct_sel = (df["mse_selective"] - df["mse_seed0"]) / df["mse_seed0"] * 100
        pct_s1 = (df["mse_seed1"] - df["mse_seed0"]) / df["mse_seed0"] * 100
        pct_sel_vs_gp = (df["mse_selective"] - df["mse_globalpick"]) / df["mse_globalpick"] * 100
        row = {"model": model_name, "task": task, "n_folds": n}
        if n >= 2:
            t_sel, p_sel = stats.ttest_rel(df["mse_selective"], df["mse_seed0"])
            t_gp, p_gp = stats.ttest_rel(df["mse_selective"], df["mse_globalpick"])
        else:
            p_sel = p_gp = float("nan")
        wins = int((df["mse_selective"] < df["mse_seed0"]).sum())
        wins_gp = int((df["mse_selective"] < df["mse_globalpick"]).sum())
        row.update({
            "seedctl_selective_mean_pct": float(pct_sel.mean()),
            "seedctl_selective_wins": wins, "seedctl_selective_p": float(p_sel),
            "seed1_alone_mean_pct": float(pct_s1.mean()),
            "selective_vs_globalpick_mean_pct": float(pct_sel_vs_gp.mean()),
            "selective_vs_globalpick_wins": wins_gp,
            "selective_vs_globalpick_p": float(p_gp),
            "mean_sensors_chose_seed1": float(df["n_sensors_chose_seed1"].mean()),
            "n_sensors_total": int(df["n_sensors"].iloc[0]),
        })
        all_summary.append(row)
        pd.DataFrame(all_summary).to_csv(f"{GTS}/seed_control_summary.csv", index=False)
        print(f"\n>>> {model_name}/{task} SEED-CONTROL: {pct_sel.mean():+.2f}% vs seed0 "
              f"({wins}/{n} folds, p={p_sel:.3g}); seed1-alone {pct_s1.mean():+.2f}%; "
              f"selective vs best-whole-checkpoint {pct_sel_vs_gp.mean():+.2f}% "
              f"({wins_gp}/{n}, p={p_gp:.3g})", flush=True)

    print("\n\n" + "=" * 72)
    print("SEED CONTROL SUMMARY")
    print("=" * 72)
    if all_summary:
        print(pd.DataFrame(all_summary).to_string(index=False))
    print("\nSEED_CONTROL_ALL_DONE")


if __name__ == "__main__":
    main()
