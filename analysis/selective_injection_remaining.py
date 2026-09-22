"""
Extends the selective-injection (validation-gated per-sensor selector) experiment to the three
remaining architectures never tested with it (PDFormer, MTGNN, AGCRN) -- completes Table 9's
RMSE (inj.) column so no row needs a dagger fallback to uniform injection. Same no-leakage
protocol as selective_injection_stid_stgcn.py: per fold, per sensor, choose plain vs od using
ONLY that fold's validation-window forward pass (fresh inference from the saved best.pt, never
the test window), then apply the fixed per-fold selector to that fold's already-computed test
predictions. GPU work here is inference-only (no training).
"""
import json, argparse, time, datetime as dt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from models_baselines import ODInjectionWrapper
from models_baselines_extra import MODELS_EXTRA

GTS = "/home/ncrc/work/gts"
device = "cuda" if torch.cuda.is_available() else "cpu"

COMBOS = [
    ("pdformer", "volume", "multi_fold_baseline_pdformer_volume_results_ext30.json"),
    ("pdformer", "speed", "multi_fold_baseline_pdformer_speed_results_ext30.json"),
    ("mtgnn", "volume", "multi_fold_baseline_mtgnn_volume_results_ext30.json"),
    ("mtgnn", "speed", "multi_fold_baseline_mtgnn_speed_results_ext30.json"),
    ("agcrn", "volume", "multi_fold_baseline_agcrn_volume_results_ext30.json"),
    ("agcrn", "speed", "multi_fold_baseline_agcrn_speed_results_ext30.json"),
]


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

    graph_path = f"{GTS}/{'volume' if task=='volume' else 'speed'}_point_graph.npz"
    g = np.load(graph_path, allow_pickle=True)
    g_ids = list(g["link_ids"]); gpos = {lid: i for i, lid in enumerate(g_ids)}
    sel = np.array([gpos[l] for l in link_ids])
    A = torch.tensor(g["A"][np.ix_(sel, sel)], dtype=torch.float32, device=device)

    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    days = d["days"]
    day_objs = [dt.datetime.strptime("20" + str(s), "%Y%m%d").date() for s in days[:n_days]]
    hours_arr = np.tile(np.arange(n_hours), n_days)
    dows = np.repeat(np.array([do.weekday() for do in day_objs]), n_hours)

    def _fmt(dd): return dd.strftime("%Y%m%d")
    is_hol = np.repeat(np.array([_fmt(dd) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    is_before = np.repeat(np.array([_fmt(dd + dt.timedelta(days=1)) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    is_after = np.repeat(np.array([_fmt(dd - dt.timedelta(days=1)) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    time_feat = np.stack([
        np.sin(2*np.pi*hours_arr/24), np.cos(2*np.pi*hours_arr/24),
        np.sin(2*np.pi*dows/7), np.cos(2*np.pi*dows/7),
        is_before, is_hol, is_after,
    ], axis=-1).astype(np.float32)

    vol_t = torch.tensor(vol_n, dtype=torch.float32, device=device)
    time_t = torch.tensor(time_feat, device=device)
    return vol_t, time_t, float(mu), float(sd), link_ids, n_sensors, n_hours, A


def load_od(args, n_hours, task):
    suffix = "" if task == "volume" else "_speed"
    rs = np.load(f"{GTS}/routed_od_signal{suffix}{args.dataset_suffix}.npz", allow_pickle=True)
    ch = rs["routed_forecast"]
    cmu = ch[args.train_lo*n_hours:args.train_hi*n_hours].mean()
    csd = ch[args.train_lo*n_hours:args.train_hi*n_hours].std() + 1e-3
    return torch.tensor((ch - cmu) / csd, dtype=torch.float32, device=device)


def build_model(args, A, n_sensors):
    # MODELS_EXTRA (pdformer/mtgnn/agcrn) construction matches train_baseline_model_extra.py
    # exactly: no p_len kwarg (unlike MODELS_EXTRA2's stid/stgcn, these infer P from the input
    # tensor shape at forward time, and AGCRN/MTGNN's __init__ doesn't even accept p_len).
    Fday = 1 + TIME_FEAT_DIM
    model_kwargs = {}
    if getattr(args, "node_se", "none") != "none":
        se = np.load(f"{GTS}/node2vec_se_{args.task}.npz")["emb"]
        model_kwargs = {"se_init": se, "se_frozen": args.node_se == "frozen"}
    base_model = MODELS_EXTRA[args.model](A, Fday, n_sensors, H=args.hidden, q_len=args.q_len, **model_kwargs).to(device)
    if args.use_od_injection:
        model = ODInjectionWrapper(base_model, args.hidden, args.q_len, TIME_FEAT_DIM).to(device)
    else:
        model = base_model
    return model


def eval_persensor_mse(model, args, vol_t, time_t, od_dec_t, mu, sd, n_sensors, lo, hi):
    P, Q = args.p_len, args.q_len
    idx_list = build_windows(n_hours_per_day=24, day_range=(lo, hi), p_len=P, q_len=Q)
    stn_idx = torch.arange(n_sensors, device=device)
    sq_err_sum = np.zeros(n_sensors, dtype=np.float64)
    n_count = np.zeros(n_sensors, dtype=np.float64)
    bs = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx_list), bs):
            batch = idx_list[i:i+bs]
            xe = torch.stack([vol_t[s:s+P] for s in batch]).unsqueeze(-1)
            te = torch.stack([time_t[s:s+P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            xd = torch.stack([vol_t[s+P:s+P+Q] for s in batch])
            td = torch.stack([time_t[s+P:s+P+Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1).transpose(1, 2)
            x_in = torch.cat([xe, te], dim=-1).transpose(1, 2)
            if args.use_od_injection:
                od_dec = torch.stack([od_dec_t[s+P:s+P+Q] for s in batch]).transpose(1, 2).unsqueeze(-1)
                pred = model(x_in, stn_idx, od_dec, td)
            else:
                pred = model(x_in, stn_idx)
            true_denorm = xd.cpu().numpy() * sd + mu
            pred_denorm = pred.transpose(1, 2).cpu().numpy() * sd + mu
            se = (true_denorm - pred_denorm) ** 2
            sq_err_sum += se.sum(axis=(0, 1))
            n_count += se.shape[0] * se.shape[1]
    return sq_err_sum / n_count


all_summary = []
for model_name, task, results_file in COMBOS:
    t0 = time.time()
    vkey = "true_volume" if task == "volume" else "true_speed"
    pkey = "pred_volume" if task == "volume" else "pred_speed"
    print(f"\n{'='*70}\n{model_name} / {task}  ({results_file})\n{'='*70}", flush=True)

    results = json.load(open(f"{GTS}/{results_file}"))
    by_fold = {}
    for e in results:
        by_fold.setdefault(e["fold_start"], {})[e["variant"]] = e
    od_key = "od_simple"

    fold_rows = []
    for fold in sorted(by_fold.keys()):
        if "plain" not in by_fold[fold] or od_key not in by_fold[fold]:
            continue
        plain_e, od_e = by_fold[fold]["plain"], by_fold[fold][od_key]
        args_p = argparse.Namespace(**plain_e["args"])
        args_o = argparse.Namespace(**od_e["args"])

        vol_t, time_t, mu, sd, link_ids, n_sensors, n_hours, A = load_series(args_p, task)
        od_dec_t = load_od(args_o, n_hours, task)

        model_p = build_model(args_p, A, n_sensors)
        model_p.load_state_dict(torch.load(f"{GTS}/{plain_e['out']}/best.pt", weights_only=True))
        model_o = build_model(args_o, A, n_sensors)
        model_o.load_state_dict(torch.load(f"{GTS}/{od_e['out']}/best.pt", weights_only=True))

        val_mse_p = eval_persensor_mse(model_p, args_p, vol_t, time_t, None, mu, sd, n_sensors, args_p.val_lo, args_p.val_hi)
        val_mse_o = eval_persensor_mse(model_o, args_o, vol_t, time_t, od_dec_t, mu, sd, n_sensors, args_o.val_lo, args_o.val_hi)
        choose_od = val_mse_o < val_mse_p

        tp = pd.read_csv(f"{GTS}/{plain_e['out']}/test_predictions.csv")
        to = pd.read_csv(f"{GTS}/{od_e['out']}/test_predictions.csv")
        tp = tp.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
        to = to.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
        assert (tp["sensor_idx"].values == to["sensor_idx"].values).all()

        sel_mask = choose_od[tp["sensor_idx"].values]
        pred_selective = np.where(sel_mask, to[pkey].values, tp[pkey].values)
        true_vals = tp[vkey].values

        mse_plain = float(np.mean((true_vals - tp[pkey].values) ** 2))
        mse_od = float(np.mean((true_vals - to[pkey].values) ** 2))
        mse_selective = float(np.mean((true_vals - pred_selective) ** 2))
        fold_rows.append({"fold": fold, "mse_plain": mse_plain, "mse_od": mse_od,
                           "mse_selective": mse_selective, "n_sensors_chose_od": int(choose_od.sum()),
                           "n_sensors": n_sensors})
        print(f"  fold {fold}: plain={mse_plain:.3f} od={mse_od:.3f} selective={mse_selective:.3f} "
              f"({choose_od.sum()}/{n_sensors}) ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(fold_rows)
    df.to_csv(f"{GTS}/selective_injection_{model_name}_{task}_fold_results.csv", index=False)
    n = len(df)
    pct_sel = (df["mse_selective"] - df["mse_plain"]) / df["mse_plain"] * 100
    pct_od = (df["mse_od"] - df["mse_plain"]) / df["mse_plain"] * 100
    t_sel, p_sel = stats.ttest_rel(df["mse_selective"], df["mse_plain"])
    t_od, p_od = stats.ttest_rel(df["mse_od"], df["mse_plain"])
    wins_sel = int((df["mse_selective"] < df["mse_plain"]).sum())
    wins_od = int((df["mse_od"] < df["mse_plain"]).sum())
    row = {"model": model_name, "task": task, "n_folds": n,
           "selective_mean_pct": float(pct_sel.mean()), "selective_wins": wins_sel, "selective_p": float(p_sel),
           "uniform_od_mean_pct": float(pct_od.mean()), "uniform_od_wins": wins_od, "uniform_od_p": float(p_od),
           "mean_sensors_chose_od": float(df["n_sensors_chose_od"].mean()), "n_sensors_total": int(df["n_sensors"].iloc[0])}
    all_summary.append(row)
    print(f"\n>>> {model_name}/{task}: SELECTIVE {pct_sel.mean():+.2f}% ({wins_sel}/{n}, p={p_sel:.3g})  |  "
          f"UNIFORM-OD {pct_od.mean():+.2f}% ({wins_od}/{n}, p={p_od:.3g})")

summary_df = pd.DataFrame(all_summary)
summary_df.to_csv(f"{GTS}/selective_injection_remaining_summary.csv", index=False)
print("\n\n" + "="*70)
print("SUMMARY (pdformer, mtgnn, agcrn)")
print("="*70)
print(summary_df.to_string(index=False))
print("\nwrote selective_injection_remaining_summary.csv")
