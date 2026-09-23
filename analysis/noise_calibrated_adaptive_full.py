"""
Continuation of noise_calibrated_selective.py after the user chose "adaptive, per-architecture
threshold" (AskUserQuestion answer, session 016dcbdc, 2026-09-20 13:58, cut off by session limit
immediately after). The original fixed-quantile version applied the SAME quantile of each
architecture's own noise distribution (q in {0, 0.5, 0.75, 0.9, 0.95}) to every architecture --
already somewhat adaptive in raw MSE units (since the noise distribution itself differs by
architecture), but the QUANTILE LEVEL was hand-picked identically for all.

Only 2 combinations (GMAN/volume, GWNET/volume) were tested in the original prototype. Fitting any
data-driven mapping from "how much real signal exists" to "what quantile to use" on n=2 would be
pure overfitting -- indistinguishable from picking a formula that reproduces exactly those two
known answers. This script avoids that trap by using a PARAMETER-FREE adaptive rule instead of a
tuned one: for each architecture, threshold in units of that architecture's OWN noise standard
deviation (a z-score cutoff), and apply the SAME fixed z-critical value across every architecture.
The raw-MSE threshold this produces is architecture-specific (adaptive) by construction, but the
*rule* (a single z-critical value, chosen a priori from standard one-/two-sided normal quantiles,
not fit to any of this paper's own results) has zero free parameters tuned to this data -- it is
just "how many noise-SDs of margin do we require before trusting a per-sensor validation
difference," the direct per-sensor analogue of a significance test.

Runs across every (architecture, task) combination that already has a seed-1 (noise-only)
checkpoint on disk from seed_control_selective_all.py / the original seed-control experiments
(10 of 12 as of this run; dcrnn/volume and staeformer/speed are excluded, their seed1 checkpoints
still being produced by the live seed_control_selective_all.py sweep -- rerun this script once
those land to get the full 12). Pure re-selection-and-rescoring on already-saved checkpoints and
test predictions; no new training.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from models_baselines import MODELS as MODELS_MAIN
from models_baselines_extra2 import MODELS_EXTRA2
from models_baselines_extra import MODELS_EXTRA

MODELS_ALL = {**MODELS_MAIN, **MODELS_EXTRA2, **MODELS_EXTRA}

GTS = "/home/ncrc/work/gts"
device = "cuda" if torch.cuda.is_available() else "cpu"

# (model, task, results_file, od_variant_key) -- the six combinations added after the original
# twelve (PDFormer/MTGNN/AGCRN, the three architectures uniform injection harmed most).
COMBOS = [
    ("gman", "volume", "multi_fold_baseline_gman_volume_results_ext30.json", "od"),
    ("gman", "speed", "multi_fold_baseline_gman_speed_results_ext30.json", "od"),
    ("dcrnn", "speed", "multi_fold_baseline_dcrnn_speed_results_ext30.json", "od"),
    ("dcrnn", "volume", "multi_fold_baseline_dcrnn_volume_results_ext30.json", "od"),
    ("gwnet", "volume", "multi_fold_baseline_gwnet_volume_results_ext30.json", "od"),
    ("gwnet", "speed", "multi_fold_baseline_gwnet_speed_results_ext30.json", "od"),
    ("staeformer", "volume", "multi_fold_staeformer_volume_results_ext30.json", "od_simple"),
    ("staeformer", "speed", "multi_fold_staeformer_speed_results_ext30.json", "od_simple"),
    ("stid_fixed", "volume", "multi_fold_baseline_stid_fixed_volume_results_ext30.json", "od_simple"),
    ("stid_fixed", "speed", "multi_fold_baseline_stid_fixed_speed_results_ext30.json", "od_simple"),
    ("stgcn", "volume", "multi_fold_baseline_stgcn_volume_results_ext30.json", "od_simple"),
    ("stgcn", "speed", "multi_fold_baseline_stgcn_speed_results_ext30.json", "od_simple"),
    ("pdformer", "volume", "multi_fold_baseline_pdformer_volume_results_ext30.json", "od_simple"),
    ("pdformer", "speed", "multi_fold_baseline_pdformer_speed_results_ext30.json", "od_simple"),
    ("mtgnn", "volume", "multi_fold_baseline_mtgnn_volume_results_ext30.json", "od_simple"),
    ("mtgnn", "speed", "multi_fold_baseline_mtgnn_speed_results_ext30.json", "od_simple"),
    ("agcrn", "volume", "multi_fold_baseline_agcrn_volume_results_ext30.json", "od_simple"),
    ("agcrn", "speed", "multi_fold_baseline_agcrn_speed_results_ext30.json", "od_simple"),
]
Z_CRITS = [0.0, 1.0, 1.645, 1.96, 2.576]  # noise-SD multiples; NOT fit to this paper's data


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
    import datetime as dt
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


def load_od(args, n_hours, task):
    suffix = "" if task == "volume" else "_speed"
    rs = np.load(f"{GTS}/routed_od_signal{suffix}{args.dataset_suffix}.npz", allow_pickle=True)
    ch = rs["routed_forecast"]
    cmu = ch[args.train_lo * n_hours:args.train_hi * n_hours].mean()
    csd = ch[args.train_lo * n_hours:args.train_hi * n_hours].std() + 1e-3
    return torch.tensor((ch - cmu) / csd, dtype=torch.float32, device=device)


def build_model(model_name, args, A, n_sensors, injected):
    from models_baselines import ODInjectionWrapper
    Fday = 1 + TIME_FEAT_DIM
    base = MODELS_ALL[model_name](A, Fday, n_sensors, H=args.hidden, q_len=args.q_len).to(device)
    return ODInjectionWrapper(base, args.hidden, args.q_len, TIME_FEAT_DIM).to(device) if injected else base


def eval_persensor_mse(model, args, vol_t, time_t, od_dec_t, mu, sd, n_sensors, n_hours, lo, hi, injected):
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
            x_in = torch.cat([xe, te], dim=-1).transpose(1, 2)
            if injected:
                td = torch.stack([time_t[s + P:s + P + Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1).transpose(1, 2)
                od_dec = torch.stack([od_dec_t[s + P:s + P + Q] for s in batch]).transpose(1, 2).unsqueeze(-1)
                pred = model(x_in, stn_idx, od_dec, td)
            else:
                pred = model(x_in, stn_idx)
            se = (xd.cpu().numpy() * sd + mu - (pred.transpose(1, 2).cpu().numpy() * sd + mu)) ** 2
            sq += se.sum(axis=(0, 1))
            cnt += se.shape[0] * se.shape[1]
    return sq / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_folds", type=int, default=30)
    ap.add_argument("--only", type=str, default=None,
                     help="comma-separated model:task filter, e.g. gman:speed,staeformer:speed")
    opt = ap.parse_args()
    only_set = set(opt.only.split(",")) if opt.only else None

    all_rows = []
    for model_name, task, results_file, od_key in COMBOS:
        if only_set is not None and f"{model_name}:{task}" not in only_set:
            continue
        print(f"\n{'=' * 72}\n{model_name}/{task}\n{'=' * 72}", flush=True)
        results = json.load(open(f"{GTS}/{results_file}"))
        by_fold = {}
        for e in results:
            by_fold.setdefault(e["fold_start"], {})[e["variant"]] = e

        vkey, pkey = ("true_volume", "pred_volume") if task == "volume" else ("true_speed", "pred_speed")
        fold_rows = []
        for fold in sorted(by_fold.keys())[:opt.max_folds]:
            seed1_dir = f"seedctl_{model_name}_{task}_fold{fold}_s1_ext30"
            if "plain" not in by_fold[fold] or od_key not in by_fold[fold] or not os.path.exists(f"{GTS}/{seed1_dir}/best.pt"):
                continue
            plain_e, od_e = by_fold[fold]["plain"], by_fold[fold][od_key]
            args_p = argparse.Namespace(**plain_e["args"])
            args_o = argparse.Namespace(**od_e["args"])

            vol_t, time_t, mu, sd, n_sensors, n_hours, A = load_series(args_p, task)
            od_dec_t = load_od(args_o, n_hours, task)

            m_plain = build_model(model_name, args_p, A, n_sensors, injected=False)
            m_plain.load_state_dict(torch.load(f"{GTS}/{plain_e['out']}/best.pt", weights_only=True))
            m_od = build_model(model_name, args_o, A, n_sensors, injected=True)
            m_od.load_state_dict(torch.load(f"{GTS}/{od_e['out']}/best.pt", weights_only=True))
            m_seed1 = build_model(model_name, args_p, A, n_sensors, injected=False)
            m_seed1.load_state_dict(torch.load(f"{GTS}/{seed1_dir}/best.pt", weights_only=True))

            val_plain = eval_persensor_mse(m_plain, args_p, vol_t, time_t, None, mu, sd, n_sensors, n_hours, args_p.val_lo, args_p.val_hi, False)
            val_od = eval_persensor_mse(m_od, args_o, vol_t, time_t, od_dec_t, mu, sd, n_sensors, n_hours, args_o.val_lo, args_o.val_hi, True)
            val_seed1 = eval_persensor_mse(m_seed1, args_p, vol_t, time_t, None, mu, sd, n_sensors, n_hours, args_p.val_lo, args_p.val_hi, False)

            real_diff = val_plain - val_od
            noise_diff = val_plain - val_seed1

            tp = pd.read_csv(f"{GTS}/{plain_e['out']}/test_predictions.csv").sort_values(
                ["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
            to = pd.read_csv(f"{GTS}/{od_e['out']}/test_predictions.csv").sort_values(
                ["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
            assert (tp["sensor_idx"].values == to["sensor_idx"].values).all()
            true_v = tp[vkey].values

            noise_sd = float(np.std(noise_diff))  # this fold's estimate of the architecture's noise scale
            row = {"fold": fold, "n_sensors": n_sensors, "noise_sd": noise_sd}
            for z in Z_CRITS:
                tau = z * noise_sd
                choose_od = real_diff > tau
                pred_sel = np.where(choose_od[tp["sensor_idx"].values], to[pkey].values, tp[pkey].values)
                mse_sel = float(np.mean((true_v - pred_sel) ** 2))
                row[f"mse_z{z}"] = mse_sel
                row[f"n_chose_z{z}"] = int(choose_od.sum())
            row["mse_plain_full"] = float(np.mean((true_v - tp[pkey].values) ** 2))
            fold_rows.append(row)
            print(f"  fold {fold}: noise_sd={noise_sd:.1f} " + " ".join(
                f"z{z}={row[f'n_chose_z{z}']}/{n_sensors}" for z in Z_CRITS), flush=True)

        if not fold_rows:
            print(f"  SKIPPED (no seed1 checkpoints found for {model_name}/{task})", flush=True)
            continue

        df = pd.DataFrame(fold_rows)
        df.to_csv(f"{GTS}/noise_calibrated_adaptive_{model_name}_{task}_fold_results.csv", index=False)
        summary = {"model": model_name, "task": task, "n_folds": len(df)}
        base = df["mse_plain_full"]
        for z in Z_CRITS:
            pct = (df[f"mse_z{z}"] - base) / base * 100
            p = stats.ttest_rel(df[f"mse_z{z}"], base)[1] if len(df) >= 2 else float("nan")
            summary[f"pct_z{z}"] = float(pct.mean())
            summary[f"p_z{z}"] = float(p)
            summary[f"mean_n_chose_z{z}"] = float(df[f"n_chose_z{z}"].mean())
        all_rows.append(summary)
        print(f"\n>>> {model_name}/{task}:")
        for z in Z_CRITS:
            print(f"    z={z}: {summary[f'pct_z{z}']:+.2f}% (p={summary[f'p_z{z}']:.3g}), "
                  f"mean sensors selected={summary[f'mean_n_chose_z{z}']:.1f}")

    pd.DataFrame(all_rows).to_csv(f"{GTS}/noise_calibrated_adaptive_full_summary.csv", index=False)
    print("\nNOISE_CALIBRATED_ADAPTIVE_ALL_DONE")


if __name__ == "__main__":
    main()
