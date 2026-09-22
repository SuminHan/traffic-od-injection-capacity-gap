"""
User's proposal (2026-09-17), following the per-sensor existence test (31/396 STAEformer/speed
sensors significantly HELPED by OD injection, 45 significantly HURT, per hypothesis0_existence_
test_speed.csv): instead of applying injection uniformly, pick per-sensor whether to use the
plain or the od_simple model, and see if that "selective injection" ensemble beats both
uniform-plain and uniform-injection in aggregate.

Critical methodological point the user flagged themselves: the 31/45 split was computed by
looking at TEST-set win rates -- using it directly to decide which model serves which sensor at
test time would be circular (selecting on the answer key). This script instead makes the
selection per-fold using ONLY that fold's VALIDATION window (never seen by the 31/45 test-based
analysis, never touched at training), then applies the resulting fixed-per-fold selector to that
fold's already-computed TEST predictions. Repeated independently over all 30 folds, exactly
mirroring the paper's own rolling-origin protocol -- so if selective injection wins, it's a real,
non-leaky effect.

Model/data-loading code paths are copied verbatim in structure from train_baseline_model.py
(same imports, same windowing/normalization/model-construction logic) so the validation-set
forward pass here is guaranteed consistent with how these checkpoints were actually trained and
evaluated -- nothing about the model or preprocessing is reinvented.
"""
import json, argparse, time
import numpy as np
import pandas as pd
import torch
from scipy import stats

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from models_baselines import MODELS, ODInjectionWrapper

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()


def load_series(args):
    tt = np.load(f"{GTS}/speed_tensor{args.dataset_suffix}.npz", allow_pickle=True)
    vol = tt["speed"]
    link_ids = list(tt["link_ids"])
    n_sensors, n_days, n_hours = vol.shape
    vol_flat = vol.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)
    mu = vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].mean()
    sd = vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].std() + 1e-3
    vol_n = (vol_flat - mu) / sd

    graph_path = f"{GTS}/speed_point_graph.npz"
    g = np.load(graph_path, allow_pickle=True)
    g_ids = list(g["link_ids"]); gpos = {lid: i for i, lid in enumerate(g_ids)}
    sel = np.array([gpos[l] for l in link_ids])
    A = torch.tensor(g["A"][np.ix_(sel, sel)], dtype=torch.float32, device=device)

    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    days = d["days"]
    import datetime as dt
    day_objs = [dt.datetime.strptime("20" + str(dd), "%Y%m%d").date() for dd in days[:n_days]]
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


def load_od(args, n_hours):
    rs = np.load(f"{GTS}/routed_od_signal_speed{args.dataset_suffix}.npz", allow_pickle=True)
    ch = rs["routed_forecast"]
    cmu = ch[args.train_lo*n_hours:args.train_hi*n_hours].mean()
    csd = ch[args.train_lo*n_hours:args.train_hi*n_hours].std() + 1e-3
    return torch.tensor((ch - cmu) / csd, dtype=torch.float32, device=device)


def build_model(args, A, n_sensors):
    Fday = 1 + TIME_FEAT_DIM
    base_model = MODELS[args.model](A, Fday, n_sensors, H=args.hidden, q_len=args.q_len).to(device)
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
            true_denorm = xd.cpu().numpy() * sd + mu          # (B,Q,S)
            pred_denorm = pred.transpose(1, 2).cpu().numpy() * sd + mu  # (B,Q,S)
            se = (true_denorm - pred_denorm) ** 2               # (B,Q,S)
            sq_err_sum += se.sum(axis=(0, 1))
            n_count += se.shape[0] * se.shape[1]
    return sq_err_sum / n_count  # (S,) per-sensor MSE


results = json.load(open(f"{GTS}/multi_fold_staeformer_speed_results_ext30.json"))
by_fold = {}
for e in results:
    by_fold.setdefault(e["fold_start"], {})[e["variant"]] = e

fold_summaries = []
for fold in sorted(by_fold.keys()):
    plain_e, od_e = by_fold[fold]["plain"], by_fold[fold]["od_simple"]
    args_p = argparse.Namespace(**plain_e["args"])
    args_o = argparse.Namespace(**od_e["args"])

    vol_t, time_t, mu, sd, link_ids, n_sensors, n_hours, A = load_series(args_p)
    od_dec_t = load_od(args_o, n_hours)

    model_p = build_model(args_p, A, n_sensors)
    model_p.load_state_dict(torch.load(f"{GTS}/{plain_e['out']}/best.pt", weights_only=True))
    model_o = build_model(args_o, A, n_sensors)
    model_o.load_state_dict(torch.load(f"{GTS}/{od_e['out']}/best.pt", weights_only=True))

    val_mse_p = eval_persensor_mse(model_p, args_p, vol_t, time_t, None, mu, sd, n_sensors, args_p.val_lo, args_p.val_hi)
    val_mse_o = eval_persensor_mse(model_o, args_o, vol_t, time_t, od_dec_t, mu, sd, n_sensors, args_o.val_lo, args_o.val_hi)
    choose_od = val_mse_o < val_mse_p  # (S,) bool, decided from VALIDATION only

    # load already-computed TEST predictions for both variants (from the original 30-fold run)
    tp = pd.read_csv(f"{GTS}/{plain_e['out']}/test_predictions.csv")
    to = pd.read_csv(f"{GTS}/{od_e['out']}/test_predictions.csv")
    assert len(tp) == len(to)
    tp = tp.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
    to = to.sort_values(["window_start_hour", "decode_step", "sensor_idx"]).reset_index(drop=True)
    assert (tp["sensor_idx"].values == to["sensor_idx"].values).all()
    assert (tp["true_speed"].values == to["true_speed"].values).all()

    sel_mask = choose_od[tp["sensor_idx"].values]  # per-row: use od pred if this sensor was val-selected
    pred_selective = np.where(sel_mask, to["pred_speed"].values, tp["pred_speed"].values)
    true_vals = tp["true_speed"].values

    mse_plain = float(np.mean((true_vals - tp["pred_speed"].values) ** 2))
    mse_od = float(np.mean((true_vals - to["pred_speed"].values) ** 2))
    mse_selective = float(np.mean((true_vals - pred_selective) ** 2))

    fold_summaries.append({"fold": fold, "mse_plain": mse_plain, "mse_od": mse_od,
                            "mse_selective": mse_selective, "n_sensors_chose_od": int(choose_od.sum())})
    print(f"fold {fold}: plain={mse_plain:.3f} od={mse_od:.3f} selective={mse_selective:.3f} "
          f"({choose_od.sum()}/{n_sensors} sensors chose od on val)  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(fold_summaries)
df.to_csv(f"{GTS}/selective_injection_fold_results.csv", index=False)

pct_vs_plain = (df["mse_selective"] - df["mse_plain"]) / df["mse_plain"] * 100
pct_od_vs_plain = (df["mse_od"] - df["mse_plain"]) / df["mse_plain"] * 100
t_sel, p_sel = stats.ttest_rel(df["mse_selective"], df["mse_plain"])
t_od, p_od = stats.ttest_rel(df["mse_od"], df["mse_plain"])

print(f"\n=== SELECTIVE INJECTION (validation-selected, 30-fold) vs PLAIN ===")
print(f"mean %MSE change = {pct_vs_plain.mean():+.2f}%, wins {(df['mse_selective']<df['mse_plain']).sum()}/30, "
      f"paired t-test p={p_sel:.4g}")
print(f"\n=== UNIFORM OD INJECTION (for reference, same as paper Table 1) vs PLAIN ===")
print(f"mean %MSE change = {pct_od_vs_plain.mean():+.2f}%, wins {(df['mse_od']<df['mse_plain']).sum()}/30, "
      f"paired t-test p={p_od:.4g}")
print(f"\nmean sensors choosing od per fold (val-selected): {df['n_sensors_chose_od'].mean():.0f}/396")
print(f"\ntotal {time.time()-t0:.0f}s")
