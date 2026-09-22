"""
Trains DCRNNSeq2Seq (models_baselines.py) -- the TRUE autoregressive-decoder DCRNN variant with
per-step OD gated fusion, built to fix the diagnosed weakness of DCRNNBaseline+ODInjectionWrapper
(single pooled context + late single-shot fusion, empirically indistinguishable from injecting
shuffled/meaningless OD). Data loading, windowing, standardization, and metrics all match
train_baseline_model.py / train_traffic_model.py exactly for direct comparability.
"""
import argparse, json, time, os, datetime as dt
import numpy as np
import torch
import torch.nn.functional as F

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from models_baselines import DCRNNSeq2Seq

GTS = "/home/ncrc/work/gts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["volume", "speed"])
    p.add_argument("--dataset_suffix", type=str, default="")
    p.add_argument("--use_od", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--p_len", type=int, default=12)
    p.add_argument("--q_len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--train_lo", type=int, required=True)
    p.add_argument("--train_hi", type=int, required=True)
    p.add_argument("--val_lo", type=int, required=True)
    p.add_argument("--val_hi", type=int, required=True)
    p.add_argument("--test_lo", type=int, required=True)
    p.add_argument("--test_hi", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = f"{GTS}/{args.out}"
    os.makedirs(out_dir, exist_ok=True)

    if args.task == "volume":
        tt = np.load(f"{GTS}/traffic_tensor{args.dataset_suffix}.npz", allow_pickle=True)
        vol = tt["volume"]
        graph_path = f"{GTS}/volume_point_graph.npz"
        mape_thresh = 10.0
    else:
        tt = np.load(f"{GTS}/speed_tensor{args.dataset_suffix}.npz", allow_pickle=True)
        vol = tt["speed"]
        graph_path = f"{GTS}/speed_point_graph.npz"
        mape_thresh = 1.0
    link_ids = list(tt["link_ids"])
    n_sensors, n_days, n_hours = vol.shape
    vol_flat = vol.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)  # (T, S)

    mu, sd = vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].mean(), \
             vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].std() + 1e-3
    vol_n = (vol_flat - mu) / sd

    g = np.load(graph_path, allow_pickle=True)
    g_ids = list(g["link_ids"]); gpos = {lid: i for i, lid in enumerate(g_ids)}
    sel = np.array([gpos[l] for l in link_ids])
    A = torch.tensor(g["A"][np.ix_(sel, sel)], dtype=torch.float32, device=device)
    print(f"{n_sensors} sensors, A nonzero frac={(A>1e-4).float().mean().item():.4f}")

    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    days = d["days"]
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
    assert time_feat.shape[1] == TIME_FEAT_DIM

    vol_t = torch.tensor(vol_n, dtype=torch.float32, device=device)  # (T,S)
    time_t = torch.tensor(time_feat, device=device)  # (T,7)
    stn_idx = torch.arange(n_sensors, device=device)

    od_dec_t = None
    if args.use_od:
        rs_path = f"{GTS}/routed_od_signal{args.dataset_suffix}.npz" if args.task == "volume" \
            else f"{GTS}/routed_od_signal_speed{args.dataset_suffix}.npz"
        rs = np.load(rs_path, allow_pickle=True)
        assert list(rs["sensor_link_ids"]) == link_ids
        ch = rs["routed_forecast"]
        cmu, csd = ch[args.train_lo*n_hours:args.train_hi*n_hours].mean(), ch[args.train_lo*n_hours:args.train_hi*n_hours].std()+1e-3
        od_dec_t = torch.tensor((ch - cmu) / csd, dtype=torch.float32, device=device)

    P, Q = args.p_len, args.q_len
    train_idx = build_windows(n_hours, (args.train_lo, args.train_hi), P, Q)
    val_idx = build_windows(n_hours, (args.val_lo, args.val_hi), P, Q)
    test_idx = build_windows(n_hours, (args.test_lo, args.test_hi), P, Q)
    print(f"windows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    def get_batch(idx_list, bs, shuffle=True, return_starts=False):
        idx = idx_list.copy()
        if shuffle: np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = idx[i:i+bs]
            xe = torch.stack([vol_t[s:s+P] for s in batch]).unsqueeze(-1)          # (B,P,S,1)
            te = torch.stack([time_t[s:s+P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)  # (B,P,S,7)
            xe_in = torch.cat([xe, te], dim=-1).transpose(1, 2)                    # (B,S,P,8)
            xd = torch.stack([vol_t[s+P:s+P+Q] for s in batch])                    # (B,Q,S)
            xd = xd.transpose(1, 2)                                                # (B,S,Q)
            td = torch.stack([time_t[s+P:s+P+Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1) \
                .transpose(1, 2)                                                   # (B,S,Q,7)
            od_dec = torch.stack([od_dec_t[s+P:s+P+Q] for s in batch]).transpose(1, 2).unsqueeze(-1) \
                if od_dec_t is not None else None                                  # (B,S,Q,1)
            if return_starts:
                yield xe_in, xd, td, od_dec, batch
                continue
            yield xe_in, xd, td, od_dec

    Fday = 1 + TIME_FEAT_DIM
    model = DCRNNSeq2Seq(A, Fday, n_sensors, H=args.hidden, q_len=Q, use_od=bool(args.use_od)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model=dcrnn_s2s task={args.task} use_od={bool(args.use_od)} params: {n_params:,}  device={device}")

    log_path = f"{out_dir}/train_log.jsonl"; open(log_path, "w").close()
    best_val = float("inf"); epochs_no_improve = 0; stopped_early_at = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for xe_in, xd, td, od_dec in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            pred = model(xe_in, stn_idx, td, xd_true=xd, od_dec=od_dec, teacher_forcing=max(0.3, 1 - epoch / 20))
            loss = F.smooth_l1_loss(pred, xd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses, val_betas = [], []
        with torch.no_grad():
            for xe_in, xd, td, od_dec in get_batch(val_idx, args.batch_size, shuffle=False):
                pred = model(xe_in, stn_idx, td, xd_true=None, od_dec=od_dec, teacher_forcing=0.0)
                val_losses.append(F.smooth_l1_loss(pred, xd).item())
                if args.use_od:
                    val_betas.append(model.last_mean_beta)

        tr_loss, va_loss = float(np.mean(tr_losses)), float(np.mean(val_losses))
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "elapsed_s": round(time.time()-t0, 1)}
        if args.use_od:
            rec["beta_mean"] = float(np.mean(val_betas))
        with open(log_path, "a") as f: f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={va_loss:.4f}  ({rec['elapsed_s']}s)" +
              (f"  beta_mean={rec['beta_mean']:.3f}" if "beta_mean" in rec else ""))
        if va_loss < best_val:
            best_val = va_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), f"{out_dir}/best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                stopped_early_at = epoch
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(f"{out_dir}/best.pt", weights_only=True))
    model.eval()
    test_true, test_pred, test_betas = [], [], []
    with torch.no_grad():
        for xe_in, xd, td, od_dec in get_batch(test_idx, args.batch_size, shuffle=False):
            pred = model(xe_in, stn_idx, td, xd_true=None, od_dec=od_dec, teacher_forcing=0.0)
            true_denorm = (xd.cpu().numpy() * sd + mu)
            pred_denorm = (pred.cpu().numpy() * sd + mu)
            test_true.append(true_denorm.ravel())
            test_pred.append(pred_denorm.ravel())
            if args.use_od:
                test_betas.append(model.last_mean_beta)
    test_true = np.concatenate(test_true); test_pred = np.concatenate(test_pred)

    r = float(np.corrcoef(test_true, test_pred)[0, 1])
    mae = float(np.mean(np.abs(test_true - test_pred)))
    mse = float(np.mean((test_true - test_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = np.sum((test_true - test_true.mean()) ** 2)
    r2 = float(1 - np.sum((test_true - test_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    mape_mask = test_true >= mape_thresh
    mape = float(np.mean(np.abs((test_true[mape_mask] - test_pred[mape_mask]) / test_true[mape_mask])) * 100)
    mape_excluded_frac = float(1 - mape_mask.mean())
    summary = {"args": vars(args), "n_params": n_params,
               "test_mse": mse, "test_rmse": rmse, "test_mae": mae,
               "test_mape": mape, "test_mape_excluded_frac": mape_excluded_frac,
               "test_r": r, "test_r2": r2,
               "best_val_loss": best_val, "stopped_early_at_epoch": stopped_early_at,
               "epochs_requested": args.epochs}
    if args.use_od:
        summary["final_beta_mean"] = float(np.mean(test_betas))
    with open(f"{out_dir}/summary.json", "w") as f: json.dump(summary, f, indent=2)
    print("TEST:", summary)


if __name__ == "__main__":
    main()
