"""
HybridOD on the full dataset, 12 months train / 1 month val / 1 month test, P=Q=12.
Uses the top-K mask determined to be the winner from the preliminary sweep (winner.json).
"""
import argparse, json, time, os, datetime as dt
import numpy as np
import torch
import torch.nn.functional as F

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, HybridOD, build_windows

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_mask", type=int, default=1)
    p.add_argument("--use_time_embed", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--edge_dim", type=int, default=16)
    p.add_argument("--time_dim", type=int, default=16)
    p.add_argument("--p_len", type=int, default=12)
    p.add_argument("--q_len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--train_months", type=int, default=12)
    p.add_argument("--val_months", type=int, default=1)
    p.add_argument("--test_months", type=int, default=1)
    p.add_argument("--start_month", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def month_bounds_from_days(days_arr):
    day_objs = [dt.datetime.strptime("20" + str(s), "%Y%m%d").date() for s in days_arr]
    month_start = {}
    for i, do in enumerate(day_objs):
        key = (do.year, do.month)
        if key not in month_start:
            month_start[key] = i
    months_sorted = sorted(month_start.keys())
    bounds = []
    for k, key in enumerate(months_sorted):
        start = month_start[key]
        end = month_start[months_sorted[k + 1]] if k + 1 < len(months_sorted) else len(days_arr)
        bounds.append((key[0], key[1], start, end))
    return bounds, day_objs


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = f"{GTS}/{args.out}"
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(f"{GTS}/od_tensor_full.npz", allow_pickle=True)
    od, codes, days = d["od"], d["codes"], d["days"]
    n_days, n_hours, N, _ = od.shape
    od_flat = od.reshape(n_days * n_hours, N, N).astype(np.float32)

    bounds, day_objs = month_bounds_from_days(days)
    s = args.start_month
    need = args.train_months + args.val_months + args.test_months
    assert len(bounds) >= s + need, f"only {len(bounds)} months available, need {s + need} (start_month={s})"
    train_m = bounds[s:s + args.train_months]
    val_m = bounds[s + args.train_months:s + args.train_months + args.val_months]
    test_m = bounds[s + args.train_months + args.val_months:s + need]
    train_range = (train_m[0][2], train_m[-1][3])
    val_range = (val_m[0][2], val_m[-1][3])
    test_range = (test_m[0][2], test_m[-1][3])
    print(f"train [{train_range}) {train_m[0][0]}-{train_m[0][1]:02d}..{train_m[-1][0]}-{train_m[-1][1]:02d}  "
          f"val [{val_range}) {val_m[0][0]}-{val_m[0][1]:02d}  test [{test_range}) {test_m[0][0]}-{test_m[0][1]:02d}")

    train_hours = (train_range[0] * n_hours, train_range[1] * n_hours)
    cum = od_flat[train_hours[0]:train_hours[1]].sum(axis=0)
    if args.use_mask:
        K = min(args.topk, N - 1)
        cum_noself = cum.copy(); np.fill_diagonal(cum_noself, -1)
        part = np.argpartition(-cum_noself, K, axis=1)[:, :K]
        edges = set()
        for i in range(N):
            for j in part[i]:
                if i != j:
                    edges.add((i, int(j))); edges.add((int(j), i))
        src = np.array([e[0] for e in edges]); dst = np.array([e[1] for e in edges])
    else:
        iu = np.where(~np.eye(N, dtype=bool)); src, dst = iu
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    n_edges = edge_index.shape[1]
    print(f"N={N} nodes, {n_edges} edges, density={n_edges/(N*(N-1)):.4f}")

    outflow = od_flat.sum(axis=2); inflow = od_flat.sum(axis=1)
    node_feat = np.stack([outflow, inflow], axis=-1)

    hours = np.tile(np.arange(n_hours), n_days)
    dows = np.repeat(np.array([do.weekday() for do in day_objs]), n_hours)

    def _fmt(dd):
        return dd.strftime("%Y%m%d")
    is_hol = np.repeat(np.array([_fmt(dd) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    is_before = np.repeat(np.array([_fmt(dd + dt.timedelta(days=1)) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    is_after = np.repeat(np.array([_fmt(dd - dt.timedelta(days=1)) in KR_HOLIDAYS for dd in day_objs], dtype=np.float32), n_hours)
    time_feat = np.stack([
        np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
        np.sin(2*np.pi*dows/7), np.cos(2*np.pi*dows/7),
        is_before, is_hol, is_after,
    ], axis=-1).astype(np.float32)

    P, Q = args.p_len, args.q_len
    train_idx = build_windows(n_hours, train_range, P, Q)
    val_idx = build_windows(n_hours, val_range, P, Q)
    test_idx = build_windows(n_hours, test_range, P, Q)
    print(f"windows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    mu = node_feat[train_hours[0]:train_hours[1]].mean(axis=(0, 1))
    sd = node_feat[train_hours[0]:train_hours[1]].std(axis=(0, 1)) + 1e-3
    node_feat_n = (node_feat - mu) / sd

    node_feat_t = torch.tensor(node_feat_n, device=device)
    time_feat_t = torch.tensor(time_feat, device=device)
    od_t = torch.tensor(od_flat[:, src, dst], device=device)

    def get_batch(idx_list, bs, shuffle=True):
        idx = idx_list.copy()
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = idx[i:i + bs]
            xe = torch.stack([node_feat_t[s:s+P] for s in batch])
            te = torch.stack([time_feat_t[s:s+P] for s in batch])
            xd = torch.stack([node_feat_t[s+P:s+P+Q] for s in batch])
            td = torch.stack([time_feat_t[s+P:s+P+Q] for s in batch])
            odb = torch.stack([od_t[s+P:s+P+Q] for s in batch])
            yield xe.float(), te.float(), xd.float(), td.float(), odb.float()

    model = HybridOD(N, edge_index, args.hidden, args.edge_dim, args.time_dim, bool(args.use_time_embed)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device={device}")

    log_path = f"{out_dir}/train_log.jsonl"; open(log_path, "w").close()
    best_val = float("inf")
    epochs_no_improve = 0
    stopped_early_at = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for xe, te, xd, td, odb in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            node_pred, od_pred = model(xe, te, xd, td, teacher_forcing=max(0.3, 1 - epoch / 20))
            loss = F.mse_loss(node_pred, xd) + F.smooth_l1_loss(od_pred, odb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses, val_od_mae, val_betas = [], [], []
        with torch.no_grad():
            for xe, te, xd, td, odb in get_batch(val_idx, args.batch_size, shuffle=False):
                node_pred, od_pred = model(xe, te, xd, td, teacher_forcing=0.0)
                val_losses.append((F.mse_loss(node_pred, xd) + F.smooth_l1_loss(od_pred, odb)).item())
                val_od_mae.append((od_pred - odb).abs().mean().item())
                val_betas.append(model.last_mean_beta)

        beta_now = float(np.mean(val_betas))  # mean gate value over the full val set (per-node/step gate, not a single scalar)
        tr_loss, va_loss, va_mae = float(np.mean(tr_losses)), float(np.mean(val_losses)), float(np.mean(val_od_mae))
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_od_mae": va_mae,
               "beta_mean": beta_now, "elapsed_s": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={va_loss:.4f}  val_od_mae={va_mae:.3f}  beta_mean={beta_now:.3f}  ({rec['elapsed_s']}s)")

        if va_loss < best_val:
            best_val = va_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), f"{out_dir}/best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                stopped_early_at = epoch
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(f"{out_dir}/best.pt"))
    model.eval()
    test_losses, test_od_mae, test_od_true, test_od_pred, test_betas = [], [], [], [], []
    with torch.no_grad():
        for xe, te, xd, td, odb in get_batch(test_idx, args.batch_size, shuffle=False):
            node_pred, od_pred = model(xe, te, xd, td, teacher_forcing=0.0)
            test_losses.append((F.mse_loss(node_pred, xd) + F.smooth_l1_loss(od_pred, odb)).item())
            test_od_mae.append((od_pred - odb).abs().mean().item())
            test_od_true.append(odb.cpu().numpy().flatten())
            test_od_pred.append(od_pred.cpu().numpy().flatten())
            test_betas.append(model.last_mean_beta)
    test_od_true = np.concatenate(test_od_true); test_od_pred = np.concatenate(test_od_pred)
    r = float(np.corrcoef(test_od_true, test_od_pred)[0, 1]) if test_od_true.std() > 0 else float("nan")
    mse = float(np.mean((test_od_true - test_od_pred) ** 2))
    ss_tot = np.sum((test_od_true - test_od_true.mean()) ** 2)
    r2 = float(1 - np.sum((test_od_true - test_od_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    final_beta = float(np.mean(test_betas))
    summary = {"args": vars(args), "n_edges": n_edges, "test_loss": float(np.mean(test_losses)),
               "test_od_mae": float(np.mean(test_od_mae)), "test_od_mse": mse, "test_od_r": r, "test_od_r2": r2, "best_val_loss": best_val,
               "final_beta_mean": final_beta, "stopped_early_at_epoch": stopped_early_at,
               "epochs_requested": args.epochs}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("TEST:", summary)


if __name__ == "__main__":
    main()
