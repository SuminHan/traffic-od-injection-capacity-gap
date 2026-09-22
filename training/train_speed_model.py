"""
Traffic-volume forecasting on the 121 TOPIS point sensors matched to OD dong nodes.
Two variants selected by --use_od_injection:
  0 = TrafficOnly   : P=12h volume history -> Q=12h volume forecast, graph conv over the
                       121-sensor kernel-weighted spatial graph (volume_point_graph.npz), no OD.
  1 = TrafficHybrid  : same traffic branch (h_traffic, GRU recurrence) PLUS, at each decode step,
                       the frozen OD model's own forecast, ROUTED through the real capital-region
                       road network onto that sensor's specific link (routed_od_signal.npz's
                       "routed_forecast" -- build_routing_weights.py's A*-style road-hierarchy
                       shortest-path assignment applied to precompute_od_signal.py's per-node OD
                       forecast; NOT a raw same-dong outflow number pasted onto the nearest sensor,
                       which was tried first and showed no lift) is projected into h_od and fused
                       via a per-node/per-step learned gate (not a single global scalar -- see
                       HybridOD in train_gts_od.py for why):
                           beta = sigmoid(gate([h_traffic, h_od]));  h_final = beta*h_traffic + (1-beta)*h_od
                       This is the "population-OD knowledge injection" arm -- it fuses the OD
                       model's *forecast* of population movement into the traffic decoder, not raw
                       ground-truth OD (which wouldn't be available at real forecast time either).
Rolling-fold parameterized exactly like train_final_fold.py (explicit day-index lo/hi), restricted
to the 2023-01-01..2025-12-31 range the traffic sensor data covers (1096 days).
"""
import argparse, json, time, os, datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows

GTS = "/home/ncrc/work/gts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_suffix", type=str, default="",
                    help="'' -> traffic_tensor.npz (2023-2025, 23-fold run); '_ext' -> traffic_tensor_ext.npz (2023-2026.07)")
    p.add_argument("--routed_signal_suffix", type=str, default=None,
                    help="override the routed_od_signal_speed filename suffix independently of --dataset_suffix "
                         "(e.g. '_modeshare' to use routed_od_signal_speed_modeshare.npz while still training "
                         "on the base speed_tensor.npz). Defaults to --dataset_suffix when unset.")
    p.add_argument("--use_od_injection", type=int, default=0)
    p.add_argument("--gate_calendar", type=int, default=0,
                    help="1 -> feed raw calendar vector (hour/dow/holiday) directly into gate_net, not just h_traffic")
    p.add_argument("--encoder_od_injection", type=int, default=0,
                    help="1 -> also fuse routed OD signal into the P-window encoder (not just the Q-window decoder)")
    p.add_argument("--od_channels", type=int, default=1,
                    help="1 -> routed_od_signal{suffix}.npz's routed_forecast (outflow only). 2 -> routed_od_signal_v2.npz's [routed_outflow, routed_inflow]")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--p_len", type=int, default=12)
    p.add_argument("--q_len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=64)
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


class WeightedGraphGRUCell(nn.Module):
    """one graph-conv step (fixed, precomputed edge weights -- a kernel-distance sensor graph,
    not learned/dynamic like MaskedGraphGRUCell) fused into a GRU update."""
    def __init__(self, in_dim, hidden, edge_index, edge_w, n_nodes):
        super().__init__()
        self.hidden = hidden
        self.register_buffer("src", edge_index[0]); self.register_buffer("dst", edge_index[1])
        self.register_buffer("ew", edge_w)
        n_edges = edge_index.shape[1]
        deg = torch.zeros(n_nodes)
        deg.index_add_(0, edge_index[1].cpu(), edge_w.cpu())
        self.register_buffer("inv_deg", 1.0 / deg.clamp(min=1e-3))
        self.msg = nn.Linear(hidden, hidden)
        self.gate_z = nn.Linear(in_dim + 2 * hidden, hidden)
        self.gate_r = nn.Linear(in_dim + 2 * hidden, hidden)
        self.gate_n = nn.Linear(in_dim + 2 * hidden, hidden)
        self.n_nodes = n_nodes

    def forward(self, x, h):
        """x: (B,N,in_dim) current-step input, h: (B,N,hidden) prev state -> new h"""
        B = x.shape[0]
        msg = self.msg(h)[:, self.src, :] * self.ew.view(1, -1, 1)
        agg = torch.zeros(B, self.n_nodes, self.hidden, device=x.device)
        agg.index_add_(1, self.dst, msg)
        agg = agg * self.inv_deg.view(1, -1, 1)
        z = torch.sigmoid(self.gate_z(torch.cat([x, h, agg], dim=-1)))
        r = torch.sigmoid(self.gate_r(torch.cat([x, h, agg], dim=-1)))
        n = torch.tanh(self.gate_n(torch.cat([x, r * h, agg], dim=-1)))
        return (1 - z) * n + z * h


class TrafficModel(nn.Module):
    def __init__(self, n_nodes, edge_index, edge_w, hidden, use_od_injection, gate_calendar=False,
                 encoder_od_injection=False, od_channels=1):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.use_od = use_od_injection
        self.gate_calendar = gate_calendar
        # if True, the P-window encoder ALSO fuses the routed OD signal at every step (same
        # od_proj/gate_net, shared weights) and carries the FUSED state forward -- not just the
        # Q-window decoder. Tests whether the model can use OD context earlier, not only when
        # generating future predictions.
        self.encoder_od_injection = encoder_od_injection
        in_dim = 1 + TIME_FEAT_DIM
        self.encoder = WeightedGraphGRUCell(in_dim, hidden, edge_index, edge_w, n_nodes)
        self.decoder = WeightedGraphGRUCell(in_dim, hidden, edge_index, edge_w, n_nodes)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.od_channels = od_channels
        if use_od_injection:
            # od_channels=1: routed_forecast (outflow only, via W) -- the OD model's forecasted
            # outflow ROUTED through the real road network onto this sensor's link (see
            # build_routing_weights.py). od_channels=2: ALSO adds routed_inflow (via V,
            # build_routing_weights_v2.py) -- "people converging on this link" as well as "people
            # fanning out from it", since a link near a hub could be inbound-traffic-dominated.
            self.od_proj = nn.Sequential(nn.Linear(od_channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            # per-node, per-timestep gate (not one global scalar -- see HybridOD in train_gts_od.py
            # for why: a single learned beta couldn't downweight an unhelpful branch enough).
            # gate_calendar=True additionally feeds the raw calendar vector (hour/dow/holiday)
            # directly into the gate, not just indirectly through h_traffic -- lets the gate learn
            # explicit calendar-conditioned trust (e.g. "downweight OD signal on ordinary weekdays,
            # upweight it around holidays") instead of only whatever leaks through h_traffic's GRU.
            gate_in = 2 * hidden + (TIME_FEAT_DIM if gate_calendar else 0)
            self.gate_net = nn.Sequential(nn.Linear(gate_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _fuse(self, h_traffic, od_step, time_step):
        h_od = self.od_proj(od_step)
        gate_in = torch.cat([h_traffic, h_od, time_step], dim=-1) if self.gate_calendar \
            else torch.cat([h_traffic, h_od], dim=-1)
        beta = torch.sigmoid(self.gate_net(gate_in))  # (B,N,1)
        return beta * h_traffic + (1 - beta) * h_od, beta

    def forward(self, xe, te, xd_true, td, od_dec, teacher_forcing=1.0, od_enc=None):
        """xe:(B,P,N,1) te:(B,P,N,T) xd_true:(B,Q,N,1) td:(B,Q,N,T) od_dec:(B,Q,N,1) or None
        od_enc:(B,P,N,1) or None -- routed OD signal aligned to the encoder window, only used
        when encoder_od_injection=True."""
        B, P, N, _ = xe.shape
        h = torch.zeros(B, N, self.hidden, device=xe.device)
        enc_gate_vals = []
        for t in range(P):
            inp = torch.cat([xe[:, t], te[:, t]], dim=-1)
            h_traffic = self.encoder(inp, h)
            if self.use_od and self.encoder_od_injection:
                h, beta = self._fuse(h_traffic, od_enc[:, t], te[:, t])
                enc_gate_vals.append(beta.mean())
            else:
                h = h_traffic

        Q = xd_true.shape[1]
        outs = []
        gate_vals = []
        prev = xe[:, -1]  # (B,N,1) last observed volume, autoregressive seed
        for t in range(Q):
            inp = torch.cat([prev, td[:, t]], dim=-1)
            h_traffic = self.decoder(inp, h)
            if self.use_od:
                h_final, beta = self._fuse(h_traffic, od_dec[:, t], td[:, t])
                gate_vals.append(beta.mean())
            else:
                h_final = h_traffic
            pred = self.head(h_final)  # (B,N,1)
            outs.append(pred)
            h = h_traffic  # recurrence always carried by the pure traffic branch
            use_true = torch.rand(()) < teacher_forcing
            prev = xd_true[:, t] if use_true else pred.detach()
        if self.use_od:
            self.last_mean_beta = torch.stack(gate_vals).mean().item()
            if self.encoder_od_injection:
                self.last_mean_beta_enc = torch.stack(enc_gate_vals).mean().item()
        return torch.cat(outs, dim=1).view(B, Q, N)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = f"{GTS}/{args.out}"
    os.makedirs(out_dir, exist_ok=True)

    tt = np.load(f"{GTS}/speed_tensor{args.dataset_suffix}.npz", allow_pickle=True)
    volume = tt["speed"]  # (n_sensors, n_days, 24) -- variable kept as "volume" for minimal diff vs train_traffic_model.py
    link_ids = list(tt["link_ids"])
    n_sensors, n_days, n_hours = volume.shape
    vol_flat = volume.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)  # (T, S)

    mu, sd = vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].mean(), \
             vol_flat[args.train_lo * n_hours:args.train_hi * n_hours].std() + 1e-3
    vol_n = (vol_flat - mu) / sd

    g = np.load(f"{GTS}/speed_point_graph.npz", allow_pickle=True)
    g_ids = list(g["link_ids"]); gpos = {lid: i for i, lid in enumerate(g_ids)}
    sel = np.array([gpos[l] for l in link_ids])
    A = g["A"][np.ix_(sel, sel)]
    src, dst = np.where(A > 1e-4)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    edge_w = torch.tensor(A[src, dst], dtype=torch.float32, device=device)
    print(f"{n_sensors} sensors, {edge_index.shape[1]} weighted edges (kernel graph, threshold 1e-4)")

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

    vol_t = torch.tensor(vol_n, dtype=torch.float32, device=device)  # (T,S)
    time_t = torch.tensor(time_feat, device=device)  # (T,7)

    od_dec_t = None
    if args.use_od_injection:
        rs_suffix = args.routed_signal_suffix if args.routed_signal_suffix is not None else args.dataset_suffix
        rs = np.load(f"{GTS}/routed_od_signal_speed{rs_suffix}.npz", allow_pickle=True)
        assert list(rs["sensor_link_ids"]) == link_ids, "routed_od_signal_speed sensor order must match speed_tensor"
        channels = [rs["routed_forecast"]]  # (T, S) -- outflow only, via W
        norm_channels = []
        for ch in channels:
            cmu, csd = ch[args.train_lo*n_hours:args.train_hi*n_hours].mean(), ch[args.train_lo*n_hours:args.train_hi*n_hours].std()+1e-3
            norm_channels.append((ch - cmu) / csd)
        od_dec_t = torch.tensor(np.stack(norm_channels, axis=-1), dtype=torch.float32, device=device)  # (T,S,od_channels)

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
            xe = torch.stack([vol_t[s:s+P] for s in batch]).unsqueeze(-1)
            te = torch.stack([time_t[s:s+P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            xd = torch.stack([vol_t[s+P:s+P+Q] for s in batch]).unsqueeze(-1)
            td = torch.stack([time_t[s+P:s+P+Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            od_dec = torch.stack([od_dec_t[s+P:s+P+Q] for s in batch]) if od_dec_t is not None else None
            od_enc = torch.stack([od_dec_t[s:s+P] for s in batch]) if od_dec_t is not None else None
            if return_starts:
                yield xe, te, xd, td, od_dec, od_enc, batch
                continue
            yield xe, te, xd, td, od_dec, od_enc

    model = TrafficModel(n_sensors, edge_index, edge_w, args.hidden, bool(args.use_od_injection),
                          gate_calendar=bool(args.gate_calendar),
                          encoder_od_injection=bool(args.encoder_od_injection),
                          od_channels=args.od_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device={device}  use_od={bool(args.use_od_injection)}")

    log_path = f"{out_dir}/train_log.jsonl"; open(log_path, "w").close()
    best_val = float("inf"); epochs_no_improve = 0; stopped_early_at = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for xe, te, xd, td, od_dec, od_enc in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            pred = model(xe, te, xd, td, od_dec, teacher_forcing=max(0.3, 1 - epoch / 20), od_enc=od_enc)
            loss = F.smooth_l1_loss(pred, xd.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses, val_betas = [], []
        with torch.no_grad():
            for xe, te, xd, td, od_dec, od_enc in get_batch(val_idx, args.batch_size, shuffle=False):
                pred = model(xe, te, xd, td, od_dec, teacher_forcing=0.0, od_enc=od_enc)
                val_losses.append(F.smooth_l1_loss(pred, xd.squeeze(-1)).item())
                if bool(args.use_od_injection):
                    val_betas.append(model.last_mean_beta)

        tr_loss, va_loss = float(np.mean(tr_losses)), float(np.mean(val_losses))
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "elapsed_s": round(time.time()-t0, 1)}
        if bool(args.use_od_injection):
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
    # raw per-(window, decode_step, sensor) predictions, for downstream metric recomputation
    # without retraining -- window_start_hour + decode_step gives the true global hour index
    # (window_start_hour + P + decode_step); sensor_idx indexes into link_ids.
    csv_rows = []
    with torch.no_grad():
        for xe, te, xd, td, od_dec, od_enc, starts in get_batch(test_idx, args.batch_size, shuffle=False, return_starts=True):
            pred = model(xe, te, xd, td, od_dec, teacher_forcing=0.0, od_enc=od_enc)
            true_denorm = (xd.squeeze(-1) * sd + mu).cpu().numpy()  # (B,Q,N)
            pred_denorm = (pred * sd + mu).cpu().numpy()            # (B,Q,N)
            test_true.append(true_denorm.ravel())
            test_pred.append(pred_denorm.ravel())
            if bool(args.use_od_injection):
                test_betas.append(model.last_mean_beta)
            for bi, s in enumerate(starts):
                for t in range(true_denorm.shape[1]):
                    global_hour = s + P + t
                    for ni in range(n_sensors):
                        csv_rows.append((s, t, global_hour, ni, link_ids[ni],
                                          true_denorm[bi, t, ni], pred_denorm[bi, t, ni]))
    test_true = np.concatenate(test_true); test_pred = np.concatenate(test_pred)

    import csv as _csv
    with open(f"{out_dir}/test_predictions.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["window_start_hour", "decode_step", "global_hour", "sensor_idx", "sensor_link_id",
                    "true_volume", "pred_volume"])
        w.writerows(csv_rows)
    r = float(np.corrcoef(test_true, test_pred)[0, 1])
    mae = float(np.mean(np.abs(test_true - test_pred)))
    mse = float(np.mean((test_true - test_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = np.sum((test_true - test_true.mean()) ** 2)
    r2 = float(1 - np.sum((test_true - test_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    # MAPE: guard against near-zero true volumes (division blows up) -- exclude true<10 vehicles/hr,
    # report the excluded fraction so a low-traffic-heavy fold doesn't silently skew the metric.
    mape_mask = test_true >= 1  # speed in km/h: near-zero = real traffic jam, not a sparsity artifact
    # like volume's near-zero hours -- only guard against literal division-by-zero, threshold much lower
    mape = float(np.mean(np.abs((test_true[mape_mask] - test_pred[mape_mask]) / test_true[mape_mask])) * 100)
    mape_excluded_frac = float(1 - mape_mask.mean())
    summary = {"args": vars(args), "test_mse": mse, "test_rmse": rmse, "test_mae": mae,
               "test_mape": mape, "test_mape_excluded_frac": mape_excluded_frac,
               "test_r": r, "test_r2": r2,
               "best_val_loss": best_val, "stopped_early_at_epoch": stopped_early_at,
               "epochs_requested": args.epochs}
    if bool(args.use_od_injection):
        summary["final_beta_mean"] = float(np.mean(test_betas))
    with open(f"{out_dir}/summary.json", "w") as f: json.dump(summary, f, indent=2)
    print("TEST:", summary)


if __name__ == "__main__":
    main()
