"""
GMAN (Graph Multi-Attention Network, Zheng et al. AAAI 2020), core model classes PORTED from the
LibCity reference implementation (libcity/model/traffic_speed_prediction/GMAN.py, cloned into
libcity_ref/) rather than hand-rolled -- the first from-scratch attempt (no BatchNorm, no Dropout,
no Xavier init, no LR decay) plateaued at a much worse loss than TrafficModel's simple GRU and
never recovered; those four ingredients are exactly what LibCity's FC/GMAN class includes and mine
omitted. Wired into OUR existing data pipeline (traffic_tensor*.npz, routed_od_signal*.npz) rather
than adopting the full LibCity framework (its own config/data-feature/atomic-file abstractions) --
only the model internals are reused.

Then the SAME per-node/per-step gated OD-injection mechanism (see TrafficModel in
train_traffic_model.py) is applied on top of GMAN's decoder representation, to test whether the
OD-injection finding holds independent of which base traffic model is used.

Scope-appropriate simplifications vs. the reference config (L=5, K=8, node2vec SE):
  - Spatial Embedding (SE): a LEARNED per-node embedding (nn.Parameter), not node2vec pretraining
    on the graph -- node2vec preprocessing wasn't set up for this project. Everything else (BN,
    dropout, Xavier init, LR decay, L stacked blocks) is ported faithfully.
  - L=3 (paper/LibCity default L=5 targets city-scale graphs; our graph is 121 nodes).
"""
import argparse, json, time, os, csv, datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM
from train_traffic_model import build_windows

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_suffix", type=str, default="_ext")
    p.add_argument("--use_od_injection", type=int, default=0)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--p_len", type=int, default=12)
    p.add_argument("--q_len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train_lo", type=int, required=True)
    p.add_argument("--train_hi", type=int, required=True)
    p.add_argument("--val_lo", type=int, required=True)
    p.add_argument("--val_hi", type=int, required=True)
    p.add_argument("--test_lo", type=int, required=True)
    p.add_argument("--test_hi", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---- ported (near-verbatim) from LibCity's GMAN.py: FC uses 1x1 Conv2d + BatchNorm + Xavier init ----
class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn=True, bn_decay=0.1):
        super().__init__()
        if isinstance(units, int):
            units, activations = [units], [activations]
        layers = []
        in_d = input_dims
        for num_unit, act in zip(units, activations):
            conv = nn.Conv2d(in_d, num_unit, (1, 1), stride=1, padding=0, bias=True)
            nn.init.xavier_normal_(conv.weight); nn.init.constant_(conv.bias, 0)
            layers.append(conv)
            if act is not None:
                if bn:
                    layers.append(nn.BatchNorm2d(num_unit, eps=1e-3, momentum=bn_decay))
                layers.append(act())
            in_d = num_unit
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3)  # (B,H,W,C) -> (B,C,H,W)
        x = self.layers(x)
        return x.transpose(2, 3).transpose(1, 3)


class SpatialAttention(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K, self.d = K, D // K
        self.q_fc = FC(2 * D, D, nn.ReLU)
        self.k_fc = FC(2 * D, D, nn.ReLU)
        self.v_fc = FC(2 * D, D, nn.ReLU)
        self.out_fc = FC(D, [D, D], [nn.ReLU, None])

    def forward(self, x, ste):
        h = torch.cat([x, ste], dim=-1)
        q, k, v = self.q_fc(h), self.k_fc(h), self.v_fc(h)
        q = torch.cat(torch.split(q, self.d, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.d, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.d, dim=-1), dim=0)
        attn = F.softmax(torch.matmul(q, k.transpose(2, 3)) / (self.d ** 0.5), dim=-1)
        out = torch.matmul(attn, v)
        out = torch.cat(torch.split(out, out.size(0) // self.K, dim=0), dim=-1)
        return self.out_fc(out)


class TemporalAttention(nn.Module):
    def __init__(self, K, D, mask=False):
        super().__init__()
        self.K, self.d, self.mask = K, D // K, mask
        self.q_fc = FC(2 * D, D, nn.ReLU)
        self.k_fc = FC(2 * D, D, nn.ReLU)
        self.v_fc = FC(2 * D, D, nn.ReLU)
        self.out_fc = FC(D, [D, D], [nn.ReLU, None])

    def forward(self, x, ste):
        h = torch.cat([x, ste], dim=-1)
        q, k, v = self.q_fc(h), self.k_fc(h), self.v_fc(h)
        q = torch.cat(torch.split(q, self.d, dim=-1), dim=0).transpose(1, 2)
        k = torch.cat(torch.split(k, self.d, dim=-1), dim=0).transpose(1, 2).transpose(2, 3)
        v = torch.cat(torch.split(v, self.d, dim=-1), dim=0).transpose(1, 2)
        attn = torch.matmul(q, k) / (self.d ** 0.5)
        if self.mask:
            T = x.size(1)
            causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn = attn.masked_fill(~causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2)
        out = torch.cat(torch.split(out, out.size(0) // self.K, dim=0), dim=-1)
        return self.out_fc(out)


class GatedFusion(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.hs_fc = FC(D, D, None, bn=False)  # LibCity: HS_fc has no bias/act here (use_bias=False, activations=None)
        self.ht_fc = FC(D, D, None, bn=False)
        self.out_fc = FC(D, [D, D], [nn.ReLU, None])

    def forward(self, hs, ht):
        z = torch.sigmoid(self.hs_fc(hs) + self.ht_fc(ht))
        return self.out_fc(z * hs + (1 - z) * ht)


class STAttBlock(nn.Module):
    def __init__(self, K, D, mask=False):
        super().__init__()
        self.sa = SpatialAttention(K, D)
        self.ta = TemporalAttention(K, D, mask=mask)
        self.gf = GatedFusion(D)

    def forward(self, x, ste):
        return x + self.gf(self.sa(x, ste), self.ta(x, ste))


class TransformAttention(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K, self.d = K, D // K
        self.q_fc = FC(D, D, nn.ReLU)
        self.k_fc = FC(D, D, nn.ReLU)
        self.v_fc = FC(D, D, nn.ReLU)
        self.out_fc = FC(D, [D, D], [nn.ReLU, None])

    def forward(self, x_enc, ste_p, ste_q):
        q, k, v = self.q_fc(ste_q), self.k_fc(ste_p), self.v_fc(x_enc)
        q = torch.cat(torch.split(q, self.d, dim=-1), dim=0).transpose(1, 2)
        k = torch.cat(torch.split(k, self.d, dim=-1), dim=0).transpose(1, 2).transpose(2, 3)
        v = torch.cat(torch.split(v, self.d, dim=-1), dim=0).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k) / (self.d ** 0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2)
        out = torch.cat(torch.split(out, out.size(0) // self.K, dim=0), dim=-1)
        return self.out_fc(out)


class STEmbedding(nn.Module):
    """SE: learned per-node (see module docstring re: node2vec simplification). TE: calendar FC."""
    def __init__(self, n_nodes, D):
        super().__init__()
        self.SE_raw = nn.Parameter(torch.randn(n_nodes, D) * 0.1)
        self.se_fc = FC(D, [D, D], [nn.ReLU, None])
        self.te_fc = FC(TIME_FEAT_DIM, [D, D], [nn.ReLU, None])

    def forward(self, time_feat):
        se = self.se_fc(self.SE_raw.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)  # (N,D)
        te = self.te_fc(time_feat)  # (B,T,N,D)
        return se.unsqueeze(0).unsqueeze(0) + te


class GMAN(nn.Module):
    def __init__(self, n_nodes, D=64, K=8, L=3, use_od_injection=False):
        super().__init__()
        self.n_nodes, self.D, self.use_od = n_nodes, D, use_od_injection
        self.ste = STEmbedding(n_nodes, D)
        self.in_fc = FC(1, [D, D], [nn.ReLU, None])
        self.encoder = nn.ModuleList([STAttBlock(K, D) for _ in range(L)])
        self.trans_att = TransformAttention(K, D)
        self.decoder = nn.ModuleList([STAttBlock(K, D) for _ in range(L)])
        if use_od_injection:
            self.od_proj = nn.Sequential(nn.Linear(1, D), nn.ReLU(), nn.Linear(D, D))
            self.gate_net = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))
        self.out_fc1 = FC(D, D, nn.ReLU)
        self.out_fc2 = FC(D, 1, None)

    def forward(self, xe, te_p, te_q, od_dec=None):
        ste_p, ste_q = self.ste(te_p), self.ste(te_q)
        h = self.in_fc(xe)
        for blk in self.encoder:
            h = blk(h, ste_p)
        h = self.trans_att(h, ste_p, ste_q)
        for blk in self.decoder:
            h = blk(h, ste_q)
        if self.use_od:
            h_od = self.od_proj(od_dec)
            beta = torch.sigmoid(self.gate_net(torch.cat([h, h_od], dim=-1)))
            self.last_mean_beta = beta.mean().item()
            h = beta * h + (1 - beta) * h_od
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.out_fc1(h)
        h = F.dropout(h, p=0.1, training=self.training)
        return F.softplus(self.out_fc2(h)).squeeze(-1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = f"{GTS}/{args.out}"
    os.makedirs(out_dir, exist_ok=True)

    tt = np.load(f"{GTS}/traffic_tensor{args.dataset_suffix}.npz", allow_pickle=True)
    volume = tt["volume"]; link_ids = list(tt["link_ids"])
    n_sensors, n_days, n_hours = volume.shape
    vol_flat = volume.transpose(1, 2, 0).reshape(n_days * n_hours, n_sensors)
    mu, sd = vol_flat[args.train_lo*n_hours:args.train_hi*n_hours].mean(), vol_flat[args.train_lo*n_hours:args.train_hi*n_hours].std()+1e-3
    vol_n = (vol_flat - mu) / sd
    vol_t = torch.tensor(vol_n, dtype=torch.float32, device=device)

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
    time_t = torch.tensor(time_feat, device=device)

    od_dec_t = None
    if args.use_od_injection:
        rs = np.load(f"{GTS}/routed_od_signal{args.dataset_suffix}.npz", allow_pickle=True)
        assert list(rs["sensor_link_ids"]) == link_ids
        routed = rs["routed_forecast"]
        rmu, rsd = routed[args.train_lo*n_hours:args.train_hi*n_hours].mean(), routed[args.train_lo*n_hours:args.train_hi*n_hours].std()+1e-3
        od_dec_t = torch.tensor((routed - rmu) / rsd, dtype=torch.float32, device=device).unsqueeze(-1)

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
            te_p = torch.stack([time_t[s:s+P] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            xd = torch.stack([vol_t[s+P:s+P+Q] for s in batch])
            te_q = torch.stack([time_t[s+P:s+P+Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1)
            od_dec = torch.stack([od_dec_t[s+P:s+P+Q] for s in batch]) if od_dec_t is not None else None
            if return_starts:
                yield xe, te_p, xd, te_q, od_dec, batch
            else:
                yield xe, te_p, xd, te_q, od_dec

    model = GMAN(n_sensors, D=args.D, K=args.K, L=args.L, use_od_injection=bool(args.use_od_injection)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=5)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device={device}  use_od={bool(args.use_od_injection)}")

    log_path = f"{out_dir}/train_log.jsonl"; open(log_path, "w").close()
    best_val = float("inf"); epochs_no_improve = 0; stopped_early_at = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for xe, te_p, xd, te_q, od_dec in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            pred = model(xe, te_p, te_q, od_dec)
            loss = F.smooth_l1_loss(pred, xd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xe, te_p, xd, te_q, od_dec in get_batch(val_idx, args.batch_size, shuffle=False):
                pred = model(xe, te_p, te_q, od_dec)
                val_losses.append(F.smooth_l1_loss(pred, xd).item())

        tr_loss, va_loss = float(np.mean(tr_losses)), float(np.mean(val_losses))
        sched.step(va_loss)
        cur_lr = opt.param_groups[0]["lr"]
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "lr": cur_lr, "elapsed_s": round(time.time()-t0, 1)}
        with open(log_path, "a") as f: f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={va_loss:.4f}  lr={cur_lr:.2e}  ({rec['elapsed_s']}s)")
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
    csv_rows = []
    with torch.no_grad():
        for xe, te_p, xd, te_q, od_dec, starts in get_batch(test_idx, args.batch_size, shuffle=False, return_starts=True):
            pred = model(xe, te_p, te_q, od_dec)
            true_d = (xd * sd + mu).cpu().numpy()
            pred_d = (pred * sd + mu).cpu().numpy()
            test_true.append(true_d.ravel()); test_pred.append(pred_d.ravel())
            if bool(args.use_od_injection):
                test_betas.append(model.last_mean_beta)
            for bi, s in enumerate(starts):
                for t in range(true_d.shape[1]):
                    gh = s + P + t
                    for ni in range(n_sensors):
                        csv_rows.append((s, t, gh, ni, link_ids[ni], true_d[bi, t, ni], pred_d[bi, t, ni]))
    test_true = np.concatenate(test_true); test_pred = np.concatenate(test_pred)

    with open(f"{out_dir}/test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_start_hour", "decode_step", "global_hour", "sensor_idx", "sensor_link_id", "true_volume", "pred_volume"])
        w.writerows(csv_rows)

    r = float(np.corrcoef(test_true, test_pred)[0, 1])
    mae = float(np.mean(np.abs(test_true - test_pred)))
    mse = float(np.mean((test_true - test_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = np.sum((test_true - test_true.mean()) ** 2)
    r2 = float(1 - np.sum((test_true - test_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    mape_mask = test_true >= 10
    mape = float(np.mean(np.abs((test_true[mape_mask] - test_pred[mape_mask]) / test_true[mape_mask])) * 100)
    summary = {"args": vars(args), "test_mse": mse, "test_rmse": rmse, "test_mae": mae, "test_mape": mape,
               "test_r": r, "test_r2": r2, "best_val_loss": best_val,
               "stopped_early_at_epoch": stopped_early_at, "epochs_requested": args.epochs}
    if bool(args.use_od_injection):
        summary["final_beta_mean"] = float(np.mean(test_betas))
    with open(f"{out_dir}/summary.json", "w") as f: json.dump(summary, f, indent=2)
    print("TEST:", summary)


if __name__ == "__main__":
    main()
