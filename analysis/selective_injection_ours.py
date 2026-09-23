"""
Extends the selective-injection (validation-gated per-sensor selector) experiment to our own
lightweight model ("Ours" in Table 9) -- the one row that had never been tested with it, since
uniform injection already works well there (Section 6's own headline result). Added for full
parity across all eleven models in Table 9 / consistency with Table 12 treating selective
injection as this paper's standard practical recipe.

Same no-leakage protocol as the other selective_injection_*.py scripts: per fold, per sensor,
choose plain vs injected using ONLY that fold's validation-window forward pass (fresh inference
from the saved best.pt, never the test window), then apply the fixed per-fold selector to that
fold's already-computed test predictions. Inference-only, no training -- reuses the existing
tfold*/sfold* checkpoints from Section 6's own 30-fold sweep.
"""
import json, argparse, time, datetime as dt
import numpy as np
import pandas as pd
import torch

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from train_traffic_model import TrafficModel

GTS = "/home/ncrc/work/gts"
device = "cuda" if torch.cuda.is_available() else "cpu"

TASKS = [
    ("volume", "multi_fold_traffic_results_ext.json", "traffic_only", "traffic_hybrid_routed",
     "true_volume", "pred_volume", f"{GTS}/traffic_tensor", "volume",
     "/home/smhan/uve_experiment/pipeline_nowcast/volume_point_graph.npz"),
    ("speed", "multi_fold_speed_results_ext.json", "speed_only", "speed_hybrid_routed",
     "true_volume", "pred_volume", f"{GTS}/speed_tensor", "speed",
     f"{GTS}/speed_point_graph.npz"),
]


def load_series(args_ns, tensor_prefix, vkey):
    tt = np.load(f"{tensor_prefix}{args_ns.dataset_suffix}.npz", allow_pickle=True)
    vol = tt[vkey]
    link_ids = list(tt["link_ids"])
    n_sensors, n_days, n_hours = vol.shape
    vol_flat = vol.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)
    mu = vol_flat[args_ns.train_lo * n_hours:args_ns.train_hi * n_hours].mean()
    sd = vol_flat[args_ns.train_lo * n_hours:args_ns.train_hi * n_hours].std() + 1e-3
    vol_n = (vol_flat - mu) / sd

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
    return vol_t, time_t, float(mu), float(sd), link_ids, n_sensors, n_hours


def load_graph(graph_path, link_ids):
    g = np.load(graph_path, allow_pickle=True)
    g_ids = list(g["link_ids"]); gpos = {lid: i for i, lid in enumerate(g_ids)}
    sel = np.array([gpos[l] for l in link_ids])
    A = g["A"][np.ix_(sel, sel)]
    src, dst = np.where(A > 1e-4)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    edge_w = torch.tensor(A[src, dst], dtype=torch.float32, device=device)
    return edge_index, edge_w


def load_od(args_ns, n_hours, suffix):
    rs = np.load(f"{GTS}/routed_od_signal{suffix}{args_ns.dataset_suffix}.npz", allow_pickle=True)
    ch = rs["routed_forecast"]
    cmu = ch[args_ns.train_lo*n_hours:args_ns.train_hi*n_hours].mean()
    csd = ch[args_ns.train_lo*n_hours:args_ns.train_hi*n_hours].std() + 1e-3
    return torch.tensor((ch - cmu) / csd, dtype=torch.float32, device=device)


def eval_persensor_mse(model, args_ns, vol_t, time_t, od_dec_t, mu, sd, n_sensors, lo, hi, use_od):
    P, Q = args_ns.p_len, args_ns.q_len
    idx_list = build_windows(n_hours_per_day=24, day_range=(lo, hi), p_len=P, q_len=Q)
    sq_err_sum = np.zeros(n_sensors, dtype=np.float64)
    n_count = np.zeros(n_sensors, dtype=np.float64)
    bs = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx_list), bs):
            batch = idx_list[i:i+bs]
            xe = torch.stack([vol_t[s:s+P] for s in batch]).unsqueeze(-1)
            te = torch.stack([time_t[s:s+P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            xd = torch.stack([vol_t[s+P:s+P+Q] for s in batch]).unsqueeze(-1)
            td = torch.stack([time_t[s+P:s+P+Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            od_dec = torch.stack([od_dec_t[s+P:s+P+Q] for s in batch]).unsqueeze(-1) if use_od else None
            pred = model(xe, te, xd, td, od_dec, teacher_forcing=0.0)
            true_denorm = xd.squeeze(-1).cpu().numpy() * sd + mu
            pred_denorm = pred.cpu().numpy() * sd + mu
            se = (true_denorm - pred_denorm) ** 2
            sq_err_sum += se.sum(axis=(0, 1))
            n_count += se.shape[0] * se.shape[1]
    return sq_err_sum / n_count


all_summary = []
for task, results_file, plain_key, od_key, vkey, pkey, tensor_prefix, tensor_vkey, graph_path in TASKS:
    t0 = time.time()
    print(f"\n{'='*70}\nOurs / {task}  ({results_file})\n{'='*70}", flush=True)

    suffix = "" if task == "volume" else "_speed"
    results = json.load(open(f"{GTS}/{results_file}"))
    by_fold = {}
    for e in results:
        by_fold.setdefault(e["fold_start"], {})[e["model"]] = e

    fold_rows = []
    for fold in sorted(by_fold.keys()):
        if plain_key not in by_fold[fold] or od_key not in by_fold[fold]:
            continue
        plain_e, od_e = by_fold[fold][plain_key], by_fold[fold][od_key]
        args_p = argparse.Namespace(**plain_e["args"])
        args_o = argparse.Namespace(**od_e["args"])

        vol_t, time_t, mu, sd, link_ids, n_sensors, n_hours = load_series(args_p, tensor_prefix, tensor_vkey)
        edge_index, edge_w = load_graph(graph_path, link_ids)
        od_dec_t = load_od(args_o, n_hours, suffix)

        model_p = TrafficModel(n_sensors, edge_index, edge_w, args_p.hidden, False).to(device)
        model_p.load_state_dict(torch.load(f"{GTS}/{plain_e['out']}/best.pt", weights_only=True))
        model_o = TrafficModel(n_sensors, edge_index, edge_w, args_o.hidden, True,
                                gate_calendar=bool(getattr(args_o, "gate_calendar", False)),
                                encoder_od_injection=bool(getattr(args_o, "encoder_od_injection", False)),
                                od_channels=getattr(args_o, "od_channels", 1)).to(device)
        model_o.load_state_dict(torch.load(f"{GTS}/{od_e['out']}/best.pt", weights_only=True))

        val_mse_p = eval_persensor_mse(model_p, args_p, vol_t, time_t, None, mu, sd, n_sensors, args_p.val_lo, args_p.val_hi, False)
        val_mse_o = eval_persensor_mse(model_o, args_o, vol_t, time_t, od_dec_t, mu, sd, n_sensors, args_o.val_lo, args_o.val_hi, True)
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
    df.to_csv(f"{GTS}/selective_injection_ours_{task}_fold_results.csv", index=False)
    n = len(df)
    from scipy import stats
    pct_sel = (df["mse_selective"] - df["mse_plain"]) / df["mse_plain"] * 100
    pct_od = (df["mse_od"] - df["mse_plain"]) / df["mse_plain"] * 100
    t_sel, p_sel = stats.ttest_rel(df["mse_selective"], df["mse_plain"])
    t_od, p_od = stats.ttest_rel(df["mse_od"], df["mse_plain"])
    wins_sel = int((df["mse_selective"] < df["mse_plain"]).sum())
    wins_od = int((df["mse_od"] < df["mse_plain"]).sum())
    row = {"model": "ours", "task": task, "n_folds": n,
           "selective_mean_pct": float(pct_sel.mean()), "selective_wins": wins_sel, "selective_p": float(p_sel),
           "uniform_od_mean_pct": float(pct_od.mean()), "uniform_od_wins": wins_od, "uniform_od_p": float(p_od),
           "mean_sensors_chose_od": float(df["n_sensors_chose_od"].mean()), "n_sensors_total": int(df["n_sensors"].iloc[0])}
    all_summary.append(row)
    print(f"\n>>> Ours/{task}: SELECTIVE {pct_sel.mean():+.2f}% ({wins_sel}/{n}, p={p_sel:.3g})  |  "
          f"UNIFORM-OD {pct_od.mean():+.2f}% ({wins_od}/{n}, p={p_od:.3g})")

summary_df = pd.DataFrame(all_summary)
summary_df.to_csv(f"{GTS}/selective_injection_ours_summary.csv", index=False)
print("\n\n" + "="*70)
print("SUMMARY (ours)")
print("="*70)
print(summary_df.to_string(index=False))
print("\nwrote selective_injection_ours_summary.csv")
