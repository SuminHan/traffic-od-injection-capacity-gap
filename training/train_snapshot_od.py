"""
Snapshot OD inference (NOT seq2seq): no P/Q history window at all -- for each hour t
independently, look only at "what time/day/holiday is it" (time_feat(t)) and directly infer
the OD matrix for that same hour. No autoregression, no encoder-decoder unroll -- one graph-conv
forward pass per sample, so this trains far faster than train_gts_od.py and isolates how much of
OD structure the masked dynamic-weight graph can explain from calendar context alone (per-node
identity + time-conditioned edges), independent of any temporal-forecasting capability.

Split: 12 months train / 1 month val / 1 month test (contiguous, on the full 2023-01..2026-08
dataset -- a 1-year sample isn't enough hold 12+1+1 months).
"""
import argparse, json, time, os, datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, DynamicEdgeWeight

GTS = "/home/ncrc/work/gts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_mask", type=int, default=1)
    p.add_argument("--use_time_embed", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--edge_dim", type=int, default=16)
    p.add_argument("--time_dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--train_months", type=int, default=12)
    p.add_argument("--val_months", type=int, default=1)
    p.add_argument("--test_months", type=int, default=1)
    p.add_argument("--start_month", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class SnapshotOD(nn.Module):
    """learned static per-node identity + one time-conditioned masked graph-conv step
    (no recurrence -- there is no "previous hour" in this formulation) -> OD(t)."""
    def __init__(self, n_nodes, edge_index, hidden, edge_dim, time_dim, use_time_embed):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        n_edges = edge_index.shape[1]
        self.node_embed = nn.Parameter(torch.randn(n_nodes, hidden) * 0.1)
        self.edge_weight_fn = DynamicEdgeWeight(n_edges, edge_dim, time_dim, use_time_embed)
        self.register_buffer("src", edge_index[0])
        self.register_buffer("dst", edge_index[1])
        self.msg = nn.Linear(hidden, hidden)
        self.combine = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
        deg = torch.zeros(n_nodes, device=edge_index.device)
        deg.index_add_(0, self.dst, torch.ones(n_edges, device=edge_index.device))
        self.register_buffer("inv_deg", 1.0 / deg.clamp(min=1))
        self.od_head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, time_feat):
        """time_feat: (B, TIME_FEAT_DIM) -> od_pred (B, n_edges)"""
        B = time_feat.shape[0]
        ew = self.edge_weight_fn(time_feat)  # (B, n_edges)
        h0 = self.node_embed.unsqueeze(0).expand(B, -1, -1)  # (B,N,hidden)
        msg = self.msg(h0)[:, self.src, :]  # (B,E,hidden)
        weighted = msg * ew.unsqueeze(-1)
        agg = torch.zeros(B, self.n_nodes, self.hidden, device=time_feat.device)
        agg.index_add_(1, self.dst, weighted)
        agg = agg * self.inv_deg.view(1, -1, 1)
        h = self.combine(torch.cat([h0, agg], dim=-1))  # (B,N,hidden), one-shot "scene" state

        hs, hd = h[:, self.src, :], h[:, self.dst, :]
        feat = torch.cat([hs, hd, ew.unsqueeze(-1)], dim=-1)
        return F.softplus(self.od_head(feat)).squeeze(-1)  # (B,E), >=0


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
    print(f"train days [{train_range}) = {train_m[0][0]}-{train_m[0][1]:02d}..{train_m[-1][0]}-{train_m[-1][1]:02d}")
    print(f"val days   [{val_range}) = {val_m[0][0]}-{val_m[0][1]:02d}")
    print(f"test days  [{test_range}) = {test_m[0][0]}-{test_m[0][1]:02d}")

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

    time_feat_t = torch.tensor(time_feat, device=device)
    od_t = torch.tensor(od_flat[:, src, dst], device=device)  # (T,E)

    train_idx = np.arange(train_hours[0], train_hours[1])
    val_idx = np.arange(val_range[0] * n_hours, val_range[1] * n_hours)
    test_idx = np.arange(test_range[0] * n_hours, test_range[1] * n_hours)
    print(f"snapshots: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    def get_batch(idx_arr, bs, shuffle=True):
        idx = idx_arr.copy()
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            yield time_feat_t[b], od_t[b]

    model = SnapshotOD(N, edge_index, args.hidden, args.edge_dim, args.time_dim, bool(args.use_time_embed)).to(device)
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
        for tf, odb in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            pred = model(tf)
            loss = F.smooth_l1_loss(pred, odb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for tf, odb in get_batch(val_idx, args.batch_size, shuffle=False):
                pred = model(tf)
                val_losses.append(F.smooth_l1_loss(pred, odb).item())

        tr_loss, va_loss = float(np.mean(tr_losses)), float(np.mean(val_losses))
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "elapsed_s": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={va_loss:.4f}  ({rec['elapsed_s']}s)")

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
    test_losses, test_true, test_pred = [], [], []
    with torch.no_grad():
        for tf, odb in get_batch(test_idx, args.batch_size, shuffle=False):
            pred = model(tf)
            test_losses.append(F.smooth_l1_loss(pred, odb).item())
            test_true.append(odb.cpu().numpy().flatten())
            test_pred.append(pred.cpu().numpy().flatten())
    test_true = np.concatenate(test_true); test_pred = np.concatenate(test_pred)
    r = float(np.corrcoef(test_true, test_pred)[0, 1]) if test_true.std() > 0 else float("nan")
    mae = float(np.mean(np.abs(test_true - test_pred)))
    mse = float(np.mean((test_true - test_pred) ** 2))
    ss_tot = np.sum((test_true - test_true.mean()) ** 2)
    r2 = float(1 - np.sum((test_true - test_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    summary = {"args": vars(args), "n_edges": n_edges, "test_loss": float(np.mean(test_losses)),
               "test_od_mae": mae, "test_od_mse": mse, "test_od_r": r, "test_od_r2": r2, "best_val_loss": best_val,
               "stopped_early_at_epoch": stopped_early_at, "epochs_requested": args.epochs}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("TEST:", summary)


if __name__ == "__main__":
    main()
