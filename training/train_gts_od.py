"""
Masked, timestamp-embedded dynamic-graph seq2seq OD prediction.

- Graph structure: nodes = 491 capital-region dongs/gu-si. Edges = dong pairs with nonzero
  historical OD interaction (>= --min_od_count total over the training period) -- this is the
  "mask": message passing AND the OD-prediction head only operate over these edges, everything
  else gets no gradient at all (not just a loss mask, the computation graph itself excludes them).
- Edge weights are NOT static: each edge's weight at time t is produced by an MLP over
  [learnable edge embedding, timestamp embedding(hour-of-day, day-of-week)] -- a dynamic,
  time-conditioned graph, ablatable via --use_time_embed.
- Seq2seq (encoder-decoder GRU + masked graph conv per step, GCRN-style), not a per-horizon model.
- Resolution: HOURLY throughout (P=24h encoder window -> Q=24h decoder window by default).

Ablation flags: --use_mask {0,1}, --use_time_embed {0,1}.
"""
import argparse, json, time, os, datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reused verbatim from the earlier UVE embedding work (train_ae_cvae_holiday.py) so holiday
# handling stays consistent across projects -- covers 2023-2026 Korean public holidays.
KR_HOLIDAYS = set(['20230101', '20230121', '20230122', '20230123', '20230124', '20230301', '20230505', '20230527', '20230529', '20230606', '20230815', '20230928', '20230929', '20230930', '20231002', '20231003', '20231009', '20231225', '20240101', '20240209', '20240210', '20240211', '20240212', '20240301', '20240410', '20240505', '20240506', '20240515', '20240606', '20240815', '20240916', '20240917', '20240918', '20241001', '20241003', '20241009', '20241225', '20250101', '20250127', '20250128', '20250129', '20250130', '20250301', '20250303', '20250505', '20250506', '20250603', '20250606', '20250815', '20251003', '20251005', '20251006', '20251007', '20251008', '20251009', '20251225', '20260101', '20260216', '20260217', '20260218', '20260301', '20260302', '20260501', '20260505', '20260524', '20260525', '20260603', '20260606', '20260717', '20260815', '20260817', '20260924', '20260925', '20260926', '20261003', '20261005', '20261009', '20261225'])
TIME_FEAT_DIM = 7  # sin/cos hour, sin/cos dow, before/on/after holiday

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_mask", type=int, default=1)
    p.add_argument("--use_time_embed", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--edge_dim", type=int, default=16)
    p.add_argument("--time_dim", type=int, default=16)
    p.add_argument("--p_len", type=int, default=24)
    p.add_argument("--q_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--min_od_count", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--patience", type=int, default=10,
                    help="stop if val_loss hasn't improved for this many consecutive epochs")
    p.add_argument("--dataset", type=str, default="202607",
                    help="suffix of od_tensor_<dataset>.npz to load, e.g. 202607 or 2023")
    p.add_argument("--out", type=str, default="run1")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_windows(n_hours_per_day, day_range, p_len, q_len):
    """sliding (start) indices within a contiguous day-block, in GLOBAL hour units"""
    lo = day_range[0] * n_hours_per_day
    hi = day_range[1] * n_hours_per_day
    return list(range(lo, hi - (p_len + q_len) + 1))


class DynamicEdgeWeight(nn.Module):
    def __init__(self, n_edges, edge_dim, time_dim, use_time_embed):
        super().__init__()
        self.use_time_embed = use_time_embed
        self.edge_embed = nn.Parameter(torch.randn(n_edges, edge_dim) * 0.1)
        if use_time_embed:
            self.time_proj = nn.Sequential(nn.Linear(TIME_FEAT_DIM, time_dim), nn.ReLU(), nn.Linear(time_dim, time_dim))
            self.gate = nn.Sequential(nn.Linear(edge_dim + time_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1))
        else:
            self.gate = nn.Sequential(nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1))

    def forward(self, time_feat):
        """time_feat: (B,TIME_FEAT_DIM) sin/cos hour + sin/cos dow + holiday(before/on/after).
        returns (B, n_edges) weights in (0,1)"""
        n_e = self.edge_embed.shape[0]
        if self.use_time_embed:
            B = time_feat.shape[0]
            t = self.time_proj(time_feat)  # (B, time_dim)
            e = self.edge_embed.unsqueeze(0).expand(B, -1, -1)  # (B, n_e, edge_dim)
            t = t.unsqueeze(1).expand(-1, n_e, -1)
            w = self.gate(torch.cat([e, t], dim=-1)).squeeze(-1)  # (B, n_e)
        else:
            B = time_feat.shape[0]
            w = self.gate(self.edge_embed).squeeze(-1).unsqueeze(0).expand(B, -1)  # (B, n_e)
        return torch.sigmoid(w)


class MaskedGraphGRUCell(nn.Module):
    """one masked, time-weighted graph-conv step fused into a GRU update (GCRN-style)."""
    def __init__(self, in_dim, hidden, edge_index, n_nodes):
        super().__init__()
        self.hidden = hidden
        self.n_nodes = n_nodes
        self.register_buffer("src", edge_index[0])
        self.register_buffer("dst", edge_index[1])
        self.msg = nn.Linear(hidden, hidden)
        self.gru_x = nn.Linear(in_dim, 3 * hidden)
        self.gru_h = nn.Linear(hidden, 3 * hidden)
        self.gru_m = nn.Linear(hidden, 3 * hidden)
        deg = torch.zeros(n_nodes, device=edge_index.device)
        deg.index_add_(0, self.dst, torch.ones(edge_index.shape[1], device=edge_index.device))
        self.register_buffer("inv_deg", 1.0 / deg.clamp(min=1))

    def forward(self, x, h, edge_w):
        """x:(B,N,in_dim) h:(B,N,hidden) edge_w:(B,E) or (1,E) -> new h:(B,N,hidden)"""
        B = x.shape[0]
        msg = self.msg(h)  # (B,N,hidden)
        msg_src = msg[:, self.src, :]  # (B,E,hidden)
        w = edge_w.unsqueeze(-1)  # (B,E,1) or (1,E,1)
        weighted = msg_src * w
        agg = torch.zeros(B, self.n_nodes, self.hidden, device=x.device, dtype=x.dtype)
        agg.index_add_(1, self.dst, weighted)
        agg = agg * self.inv_deg.view(1, -1, 1)

        gx = self.gru_x(x)
        gh = self.gru_h(h)
        gm = self.gru_m(agg)
        xz, xr, xn = gx.chunk(3, dim=-1)
        hz, hr, hn = gh.chunk(3, dim=-1)
        mz, mr, mn = gm.chunk(3, dim=-1)
        z = torch.sigmoid(xz + hz + mz)
        r = torch.sigmoid(xr + hr + mr)
        n = torch.tanh(xn + r * (hn + mn))
        return (1 - z) * h + z * n


class Seq2SeqOD(nn.Module):
    def __init__(self, n_nodes, edge_index, hidden, edge_dim, time_dim, use_time_embed, in_dim=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        n_edges = edge_index.shape[1]
        self.edge_weight_fn = DynamicEdgeWeight(n_edges, edge_dim, time_dim, use_time_embed)
        self.encoder = MaskedGraphGRUCell(in_dim, hidden, edge_index, n_nodes)
        self.decoder = MaskedGraphGRUCell(in_dim, hidden, edge_index, n_nodes)
        self.node_head = nn.Linear(hidden, in_dim)  # predicts [outflow, inflow] per node
        self.register_buffer("src", edge_index[0])
        self.register_buffer("dst", edge_index[1])
        self.od_head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def od_from_h(self, h, edge_w):
        hs, hd = h[:, self.src, :], h[:, self.dst, :]
        w = edge_w if edge_w.shape[0] == h.shape[0] else edge_w.expand(h.shape[0], -1)
        feat = torch.cat([hs, hd, w.unsqueeze(-1)], dim=-1)
        return F.softplus(self.od_head(feat)).squeeze(-1)  # (B,E), >=0

    def forward(self, x_enc, time_enc, x_dec_true, time_dec, teacher_forcing=1.0):
        """x_enc:(B,P,N,2) time_enc:(B,P,TIME_FEAT_DIM) x_dec_true:(B,Q,N,2) time_dec:(B,Q,TIME_FEAT_DIM)"""
        B = x_enc.shape[0]
        h = torch.zeros(B, self.n_nodes, self.hidden, device=x_enc.device)
        P = x_enc.shape[1]
        for t in range(P):
            ew = self.edge_weight_fn(time_enc[:, t, :])
            h = self.encoder(x_enc[:, t, :, :], h, ew)

        Q = x_dec_true.shape[1]
        node_preds, od_preds = [], []
        dec_in = x_enc[:, -1, :, :]
        for t in range(Q):
            ew = self.edge_weight_fn(time_dec[:, t, :])
            h = self.decoder(dec_in, h, ew)
            node_pred = self.node_head(h)
            od_pred = self.od_from_h(h, ew)
            node_preds.append(node_pred)
            od_preds.append(od_pred)
            if self.training and torch.rand(()) < teacher_forcing:
                dec_in = x_dec_true[:, t, :, :]
            else:
                dec_in = node_pred.detach()
        return torch.stack(node_preds, dim=1), torch.stack(od_preds, dim=1)  # (B,Q,N,2), (B,Q,E)


class HybridOD(nn.Module):
    """Fuses two branches at the embedding level before the final prediction head:
      - h_seq:  the seq2seq (P history -> Q decode) recurrent branch, exactly like Seq2SeqOD
      - h_snap: a "scene-only" branch that looks at nothing but the current timestep's calendar
                context (time_feat) -- a static per-node identity + one time-conditioned graph-conv
                step, no memory of history at all (same idea as train_snapshot_od.py's SnapshotOD)
    h_final = beta * h_seq + (1-beta) * h_snap
    beta is a GATE, not a single global scalar: a small MLP over [h_seq, h_snap] produces a
    per-node, per-timestep beta in (0,1) via sigmoid. This replaces an earlier version that used
    one learned scalar for the whole model -- rolling-fold validation showed that version losing to
    plain Seq2SeqOD in every fold (beta converged to ~0.5, a near-even blend that couldn't
    downweight the much weaker snapshot branch enough). A per-node/per-step gate can in principle
    learn beta->1 (ignore snapshot) wherever it doesn't help, rather than always mixing ~50/50.
    h_final (not h_seq alone) is what the node/OD prediction heads see. The GRU recurrence itself
    still carries h_seq forward untouched, so the temporal dynamics aren't corrupted by blending in
    the memory-free branch.
    """
    def __init__(self, n_nodes, edge_index, hidden, edge_dim, time_dim, use_time_embed, in_dim=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        n_edges = edge_index.shape[1]
        self.edge_weight_fn = DynamicEdgeWeight(n_edges, edge_dim, time_dim, use_time_embed)
        self.encoder = MaskedGraphGRUCell(in_dim, hidden, edge_index, n_nodes)
        self.decoder = MaskedGraphGRUCell(in_dim, hidden, edge_index, n_nodes)

        # scene-only branch: static node identity + one masked, time-weighted graph-conv step
        self.node_embed = nn.Parameter(torch.randn(n_nodes, hidden) * 0.1)
        self.snap_msg = nn.Linear(hidden, hidden)
        self.snap_combine = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())

        # per-node, per-timestep gate: sees both branches' embeddings, decides how much to trust
        # h_snap at THIS node/THIS moment, instead of one global blend ratio for everything.
        self.gate_net = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.node_head = nn.Linear(hidden, in_dim)
        self.register_buffer("src", edge_index[0])
        self.register_buffer("dst", edge_index[1])
        deg = torch.zeros(n_nodes, device=edge_index.device)
        deg.index_add_(0, self.dst, torch.ones(n_edges, device=edge_index.device))
        self.register_buffer("inv_deg", 1.0 / deg.clamp(min=1))
        self.od_head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def snapshot_embed(self, time_feat, ew):
        B = time_feat.shape[0]
        h0 = self.node_embed.unsqueeze(0).expand(B, -1, -1)
        msg = self.snap_msg(h0)[:, self.src, :]
        weighted = msg * ew.unsqueeze(-1)
        agg = torch.zeros(B, self.n_nodes, self.hidden, device=time_feat.device)
        agg.index_add_(1, self.dst, weighted)
        agg = agg * self.inv_deg.view(1, -1, 1)
        return self.snap_combine(torch.cat([h0, agg], dim=-1))

    def od_from_h(self, h, edge_w):
        hs, hd = h[:, self.src, :], h[:, self.dst, :]
        w = edge_w if edge_w.shape[0] == h.shape[0] else edge_w.expand(h.shape[0], -1)
        feat = torch.cat([hs, hd, w.unsqueeze(-1)], dim=-1)
        return F.softplus(self.od_head(feat)).squeeze(-1)

    def forward(self, x_enc, time_enc, x_dec_true, time_dec, teacher_forcing=1.0):
        B = x_enc.shape[0]
        h = torch.zeros(B, self.n_nodes, self.hidden, device=x_enc.device)
        P = x_enc.shape[1]
        for t in range(P):
            ew = self.edge_weight_fn(time_enc[:, t, :])
            h = self.encoder(x_enc[:, t, :, :], h, ew)

        Q = x_dec_true.shape[1]
        node_preds, od_preds, gate_vals = [], [], []
        dec_in = x_enc[:, -1, :, :]
        for t in range(Q):
            ew = self.edge_weight_fn(time_dec[:, t, :])
            h = self.decoder(dec_in, h, ew)          # pure temporal recurrence, uncorrupted
            h_snap = self.snapshot_embed(time_dec[:, t, :], ew)
            beta = torch.sigmoid(self.gate_net(torch.cat([h, h_snap], dim=-1)))  # (B,N,1)
            h_final = beta * h + (1 - beta) * h_snap  # fused embedding used ONLY for prediction
            node_pred = self.node_head(h_final)
            od_pred = self.od_from_h(h_final, ew)
            node_preds.append(node_pred)
            od_preds.append(od_pred)
            gate_vals.append(beta.mean())
            if self.training and torch.rand(()) < teacher_forcing:
                dec_in = x_dec_true[:, t, :, :]
            else:
                dec_in = node_pred.detach()
        self.last_mean_beta = torch.stack(gate_vals).mean().item()
        return torch.stack(node_preds, dim=1), torch.stack(od_preds, dim=1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SC = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"
    out_dir = f"{SC}/{args.out}"
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(f"{SC}/od_tensor_{args.dataset}.npz", allow_pickle=True)
    od, codes, days = d["od"], d["codes"], d["days"]
    n_days, n_hours, N, _ = od.shape
    T = n_days * n_hours
    od_flat = od.reshape(T, N, N).astype(np.float32)  # HOURLY, contiguous, T=744

    cum = od_flat.sum(axis=0)  # (N,N) total OD over the whole month
    if args.use_mask:
        # a raw nonzero/threshold mask is nearly a complete graph at monthly granularity (>99%
        # density even at count>=10) since almost every dong pair exchanges *some* trips over a
        # month -- top-K per node (each node's K biggest real partners, directed, union of both
        # directions) is what actually produces a sparse, meaningful "who can influence whom" graph
        K = min(args.topk, N - 1)
        cum_noself = cum.copy(); np.fill_diagonal(cum_noself, -1)
        part = np.argpartition(-cum_noself, K, axis=1)[:, :K]
        edges = set()
        for i in range(N):
            for j in part[i]:
                if i != j:
                    edges.add((i, int(j)))
                    edges.add((int(j), i))  # symmetrize so influence flows both ways
        src = np.array([e[0] for e in edges]); dst = np.array([e[1] for e in edges])
    else:
        iu = np.where(~np.eye(N, dtype=bool))
        src, dst = iu
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    n_edges = edge_index.shape[1]
    print(f"N={N} nodes, {n_edges} edges (mask={'on' if args.use_mask else 'off/dense'}), "
          f"density={n_edges/(N*(N-1)):.4f}")

    outflow = od_flat.sum(axis=2)  # (T,N)
    inflow = od_flat.sum(axis=1)   # (T,N)
    node_feat = np.stack([outflow, inflow], axis=-1)  # (T,N,2)

    hours = np.tile(np.arange(n_hours), n_days)
    # real calendar dates from the npz's `days` array (YYMMDD strings, e.g. "260701") -- gives
    # true day-of-week and lets us look up KR_HOLIDAYS directly, instead of an arbitrary anchor.
    day_objs = [dt.datetime.strptime("20" + str(d), "%Y%m%d").date() for d in days]
    dows_per_day = np.array([d.weekday() for d in day_objs])  # 0=Mon..6=Sun
    dows = np.repeat(dows_per_day, n_hours)

    def _fmt(d):
        return d.strftime("%Y%m%d")
    is_hol = np.repeat(np.array([_fmt(d) in KR_HOLIDAYS for d in day_objs], dtype=np.float32), n_hours)
    is_before_hol = np.repeat(np.array([_fmt(d + dt.timedelta(days=1)) in KR_HOLIDAYS for d in day_objs], dtype=np.float32), n_hours)
    is_after_hol = np.repeat(np.array([_fmt(d - dt.timedelta(days=1)) in KR_HOLIDAYS for d in day_objs], dtype=np.float32), n_hours)
    print(f"holiday days in range: {int(np.array([_fmt(d) in KR_HOLIDAYS for d in day_objs]).sum())} / {n_days}")

    time_feat = np.stack([
        np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7),
        is_before_hol, is_hol, is_after_hol,
    ], axis=-1).astype(np.float32)  # (T, TIME_FEAT_DIM)

    P, Q = args.p_len, args.q_len
    # 71%/14.5%/14.5% split (matches the original 22/4/5-day split on 31 days), scaled to
    # however many days this dataset actually has
    n_train = round(n_days * 22 / 31)
    n_val = round(n_days * 4 / 31)
    train_days, val_days, test_days = (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_days)
    train_idx = build_windows(n_hours, train_days, P, Q)
    val_idx = build_windows(n_hours, val_days, P, Q)
    test_idx = build_windows(n_hours, test_days, P, Q)
    print(f"windows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    mu, sd = node_feat[:train_days[1]*n_hours].mean(axis=(0,1)), node_feat[:train_days[1]*n_hours].std(axis=(0,1)) + 1e-3
    node_feat_n = (node_feat - mu) / sd

    node_feat_t = torch.tensor(node_feat_n, device=device)
    time_feat_t = torch.tensor(time_feat, device=device)
    od_t = torch.tensor(od_flat[:, src, dst], device=device)  # (T,E) true OD along masked edges only

    def get_batch(idx_list, bs, shuffle=True):
        idx = idx_list.copy()
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = idx[i:i+bs]
            xe = torch.stack([node_feat_t[s:s+P] for s in batch])
            te = torch.stack([time_feat_t[s:s+P] for s in batch])
            xd = torch.stack([node_feat_t[s+P:s+P+Q] for s in batch])
            td = torch.stack([time_feat_t[s+P:s+P+Q] for s in batch])
            odb = torch.stack([od_t[s+P:s+P+Q] for s in batch])
            yield xe.float(), te.float(), xd.float(), td.float(), odb.float()

    model = Seq2SeqOD(N, edge_index, args.hidden, args.edge_dim, args.time_dim, bool(args.use_time_embed)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device={device}")

    log_path = f"{out_dir}/train_log.jsonl"
    open(log_path, "w").close()
    best_val = float("inf")
    epochs_no_improve = 0
    stopped_early_at = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for xe, te, xd, td, odb in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            node_pred, od_pred = model(xe, te, xd, td, teacher_forcing=max(0.3, 1 - epoch / 40))
            loss_node = F.mse_loss(node_pred, xd)
            loss_od = F.smooth_l1_loss(od_pred, odb)
            loss = loss_node + loss_od
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses, val_od_mae = [], []
        with torch.no_grad():
            for xe, te, xd, td, odb in get_batch(val_idx, args.batch_size, shuffle=False):
                node_pred, od_pred = model(xe, te, xd, td, teacher_forcing=0.0)
                loss = F.mse_loss(node_pred, xd) + F.smooth_l1_loss(od_pred, odb)
                val_losses.append(loss.item())
                val_od_mae.append((od_pred - odb).abs().mean().item())

        tr_loss, va_loss, va_mae = float(np.mean(tr_losses)), float(np.mean(val_losses)), float(np.mean(val_od_mae))
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_od_mae": va_mae,
               "elapsed_s": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={va_loss:.4f}  val_od_mae={va_mae:.3f}  ({rec['elapsed_s']}s)")

        if va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{out_dir}/best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                stopped_early_at = epoch
                print(f"early stopping at epoch {epoch} (no val_loss improvement for {args.patience} epochs)")
                break

    # final test eval with best checkpoint
    model.load_state_dict(torch.load(f"{out_dir}/best.pt"))
    model.eval()
    test_losses, test_od_mae, test_od_true, test_od_pred = [], [], [], []
    with torch.no_grad():
        for xe, te, xd, td, odb in get_batch(test_idx, args.batch_size, shuffle=False):
            node_pred, od_pred = model(xe, te, xd, td, teacher_forcing=0.0)
            test_losses.append((F.mse_loss(node_pred, xd) + F.smooth_l1_loss(od_pred, odb)).item())
            test_od_mae.append((od_pred - odb).abs().mean().item())
            test_od_true.append(odb.cpu().numpy().flatten())
            test_od_pred.append(od_pred.cpu().numpy().flatten())
    test_od_true = np.concatenate(test_od_true); test_od_pred = np.concatenate(test_od_pred)
    r = float(np.corrcoef(test_od_true, test_od_pred)[0, 1]) if test_od_true.std() > 0 else float("nan")
    summary = {"args": vars(args), "n_edges": n_edges, "test_loss": float(np.mean(test_losses)),
               "test_od_mae": float(np.mean(test_od_mae)), "test_od_r": r, "best_val_loss": best_val,
               "stopped_early_at_epoch": stopped_early_at, "epochs_requested": args.epochs}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("TEST:", summary)

if __name__ == "__main__":
    main()
