"""
Published-baseline comparison for TKDE submission: DCRNN / Graph WaveNet (GWNET) / GMAN
(see models_baselines.py for architecture notes and the one documented adaptation --
direct multi-horizon head instead of our own autoregressive decoder). Same data, windowing,
standardization, train/val/test split conventions, and metric definitions as
train_traffic_model.py / train_speed_model.py, so numbers are directly comparable to
TrafficOnly/TrafficHybrid and SpeedOnly/SpeedHybrid. --task selects volume vs speed data.
Plain baselines only (no OD injection) for now -- these ARE the "how does a real published
architecture do without our OD-routing injection" comparison point.
"""
import argparse, json, time, os, datetime as dt
import numpy as np
import torch
import torch.nn.functional as F

from train_gts_od import KR_HOLIDAYS, TIME_FEAT_DIM, build_windows
from models_baselines import ODInjectionWrapper, MatureInjectionWrapper, CrossAttnInjectionWrapper
from models_baselines_extra import MODELS_EXTRA as MODELS

GTS = "/home/ncrc/work/gts"


def geh_loss(pred_n, target_n, mu, sd, eps=1.0, sqrt_eps=1e-6):
    """Mean GEH (FHWA/UK-standard traffic-count validation statistic) computed on denormalized
    (real-scale) values, usable as a training loss in place of / alongside MSE-family losses.
    GEH = sqrt(2*(M-C)^2/(M+C)) is a variance-stabilizing residual appropriate for count data
    (variance scales with the mean, Poisson-like) -- unlike MSE/MAE it doesn't let high-volume
    links dominate the loss, and it directly targets the same statistic the eventual evaluation
    (GEH<5 pass rate) is judged on. Only statistically appropriate for --task volume (a count);
    speed is a continuous physical quantity without the same mean-variance scaling, so GEH-loss
    on speed is not recommended (main() prints a warning if you do it anyway).
    pred_n, target_n: normalized (z-scored) tensors of the same shape. mu, sd: the scalars used
    to build them (so this stays differentiable end-to-end w.r.t. pred_n).
    eps: added to M+C to avoid the M=C=0 zero-flow singularity -- that case has zero residual
    anyway, so eps there just contributes ~0 loss, not a real approximation error.
    sqrt_eps: added INSIDE the sqrt (not just the M+C denominator) -- sqrt'(x)=1/(2*sqrt(x))
    blows up as x->0, and a well-fit point drives GEH^2 exactly to 0, so without this the very
    predictions the loss should reward the most produce inf/nan gradients (confirmed empirically:
    training went to nan after epoch 0 before this was added). This keeps the sqrt argument
    bounded away from exactly zero without materially changing the metric's value elsewhere."""
    M = torch.clamp(pred_n * sd + mu, min=0.0)
    C = torch.clamp(target_n * sd + mu, min=0.0)
    geh_sq = 2.0 * (M - C) ** 2 / (M + C + eps)
    return torch.sqrt(geh_sq + sqrt_eps).mean()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--task", type=str, required=True, choices=["volume", "speed"])
    p.add_argument("--loss_fn", type=str, default="smooth_l1", choices=["smooth_l1", "geh", "combined"],
                    help="smooth_l1 (default, Huber loss on normalized values), geh (mean GEH "
                         "statistic on denormalized values, see geh_loss() docstring -- directly "
                         "targets the FHWA GEH<5 evaluation criterion instead of a generic "
                         "regression loss), or combined (smooth_l1 + --geh_weight * geh_loss, to "
                         "get both a good general fit AND GEH-criterion alignment instead of "
                         "trading one for the other). geh/combined statistically only appropriate "
                         "for --task volume.")
    p.add_argument("--geh_weight", type=float, default=0.02,
                    help="only used when --loss_fn combined: final loss = smooth_l1 + geh_weight * "
                         "geh_loss. The two terms sit on very different scales (smooth_l1 on "
                         "normalized values, ~0.02-0.3 early in training; geh_loss on denormalized "
                         "traffic counts, ~5-30) -- geh_weight~0.01-0.05 roughly balances their "
                         "contribution based on empirical smoke-test magnitudes on this dataset, "
                         "but there's no principled way to set this without sweeping it; treat it "
                         "as a hyperparameter.")
    p.add_argument("--dataset_suffix", type=str, default="")
    p.add_argument("--use_od_injection", type=int, default=0)
    p.add_argument("--fusion_style", type=str, default="simple", choices=["simple", "mature", "crossattn"],
                    help="simple -> single scalar gate (ODInjectionWrapper). mature -> boost/eliminate "
                         "dual per-dim gate adapted from Li et al. CIKM 2020's knowledge adaptation "
                         "module (MatureInjectionWrapper) -- their own ablation showed naive/simple "
                         "fusion hurts and their richer dual-vector gate fixes it; testing whether the "
                         "same fix helps here. crossattn -> multi-head cross-attention letting every "
                         "decode step attend over the full OD decode sequence, not just the "
                         "time-aligned step (CrossAttnInjectionWrapper).")
    p.add_argument("--od_graph", type=int, default=0,
                    help="dcrnn/gwnet only, alternative to --use_od_injection's node-feature gated "
                         "fusion: inject OD by modulating the model's own graph adjacency instead of "
                         "as a node feature (GWNET: its adaptive adjacency; DCRNN: its fixed diffusion "
                         "adjacency). Mutually exclusive with --use_od_injection -- replaces it.")
    p.add_argument("--node_se", type=str, default="none", choices=["none", "frozen", "finetune"],
                    help="use node2vec SE (build_node2vec_se.py) instead of a from-scratch node embedding "
                         "-- applies to whichever --model is selected (DCRNN/GWNET/GMAN all have a "
                         "learned nn.Embedding(N, emb_dim) node identity lookup this can replace)")
    p.add_argument("--routing_source", type=str, default="single",
                    choices=["single", "multipath", "modeshare", "congestion_aware"],
                    help="single -> routed_od_signal(_speed).npz (single shortest path per OD pair). "
                         "multipath -> routed_od_signal(_speed)_multipath.npz (Monte Carlo perturbed-"
                         "Dijkstra stochastic assignment, see build_routing_weights_multipath.py). "
                         "modeshare -> routed_od_signal(_speed)_modeshare.npz (distance-based car-mode-"
                         "share discount applied to destination shares before routing, see "
                         "build_routing_weights_modeshare.py). "
                         "congestion_aware -> routed_od_signal(_speed)_congestion_aware.npz: single-path "
                         "signal reshaped by an empirically-measured (sensor, hour_of_day) ratio between "
                         "the calibrated BPR/MSA equilibrium simulator (.../simulator/, capacity-aware) "
                         "and naive single-path routing (not capacity-aware) -- see "
                         "build_routed_signal_congestion_aware_ext.py. Approximation: the ratio is a "
                         "fixed hour-of-day pattern (not re-simulated per day), so it captures typical "
                         "congestion reshaping but not day-specific congestion swings. volume task only "
                         "for now (no speed-task version built yet).")
    p.add_argument("--shuffle_od", type=int, default=0,
                    help="diagnostic control: randomly permute the routed OD signal along the time axis "
                         "before injection -- same marginal per-sensor distribution and same extra "
                         "param count (od_proj/gate_net/step_head/td_proj) as real OD injection, but the "
                         "signal is decorrelated from the actual hour. If this hurts performance about as "
                         "much as the real signal did, the earlier degradation was likely an optimization/"
                         "capacity artifact, not evidence the real OD content is actively unhelpful.")
    p.add_argument("--od_source", type=str, default="forecast", choices=["forecast", "oracle"],
                    help="forecast -> routed_forecast (the OD model's own forecast, what a real "
                         "deployment would have). oracle -> routed_actual (the TRUE observed OD for "
                         "that same hour, routed the same way) -- an upper-bound diagnostic only, "
                         "NOT a deployable model (not available at real forecast time). Used to test "
                         "whether OD injection's weakness on anomalous hours is a forecast-accuracy "
                         "problem (oracle should fix it) or a deeper OD-to-traffic causal-link "
                         "problem (oracle won't fix it).")
    p.add_argument("--od_signal_mode", type=str, default="raw", choices=["raw", "deviation"],
                    help="raw -> inject the routed OD forecast value as-is. deviation -> subtract a "
                         "per-sensor (day-of-week, hour-of-day) baseline computed from the TRAIN window "
                         "first, then inject the residual -- the idea being that a predictable/typical "
                         "level of OD flow is already implicit in past traffic, so only the ANOMALOUS "
                         "part of the OD signal should carry new information.")
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
    p.add_argument("--warm_start_from", type=str, default="",
                    help="FAST-SCREENING ONLY, not for final rigor numbers: path to a previous "
                         "fold's best.pt to initialize from instead of random init. Breaks fold "
                         "independence (biases/speeds convergence using another fold's trained "
                         "weights) -- use only to quickly gauge whether an idea looks promising "
                         "across a few folds before committing to the real independent-per-fold "
                         "sweep for any number that goes in the paper.")
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
        val_key = "volume"
    else:
        tt = np.load(f"{GTS}/speed_tensor{args.dataset_suffix}.npz", allow_pickle=True)
        vol = tt["speed"]
        graph_path = f"{GTS}/speed_point_graph.npz"
        mape_thresh = 1.0
        val_key = "speed"
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
    if args.use_od_injection or args.od_graph:
        mp_suffix = {"multipath": "_multipath", "modeshare": "_modeshare",
                     "congestion_aware": "_congestion_aware"}.get(args.routing_source, "")
        rs_path = f"{GTS}/routed_od_signal{mp_suffix}{args.dataset_suffix}.npz" if args.task == "volume" \
            else f"{GTS}/routed_od_signal_speed{mp_suffix}{args.dataset_suffix}.npz"
        rs = np.load(rs_path, allow_pickle=True)
        assert list(rs["sensor_link_ids"]) == link_ids, "routed_od_signal sensor order must match tensor"
        if args.od_source == "oracle":
            assert "routed_actual" in rs, f"{rs_path} has no routed_actual (oracle) channel"
            ch = rs["routed_actual"]  # (T, S) -- TRUE observed OD, upper-bound diagnostic only
        else:
            ch = rs["routed_forecast"]  # (T, S)
        if args.od_signal_mode == "deviation":
            train_lo_t, train_hi_t = args.train_lo * n_hours, args.train_hi * n_hours
            baseline = np.zeros_like(ch)  # (T, S), per-sensor (dow,hour) baseline from TRAIN window only
            for dow in range(7):
                for hr in range(n_hours):
                    mask_train = (dows[train_lo_t:train_hi_t] == dow) & (hours_arr[train_lo_t:train_hi_t] == hr)
                    if mask_train.sum() == 0:
                        continue
                    cell_mean = ch[train_lo_t:train_hi_t][mask_train].mean(axis=0)  # (S,)
                    mask_all = (dows == dow) & (hours_arr == hr)
                    baseline[mask_all] = cell_mean
            ch = ch - baseline
        if args.shuffle_od:
            perm = np.random.RandomState(args.seed).permutation(ch.shape[0])
            ch = ch[perm]
        cmu, csd = ch[args.train_lo*n_hours:args.train_hi*n_hours].mean(), ch[args.train_lo*n_hours:args.train_hi*n_hours].std()+1e-3
        od_dec_t = torch.tensor((ch - cmu) / csd, dtype=torch.float32, device=device)  # (T,S)

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
            xd = torch.stack([vol_t[s+P:s+P+Q] for s in batch])                     # (B,Q,S)
            td = torch.stack([time_t[s+P:s+P+Q] for s in batch]).unsqueeze(2).expand(-1, -1, n_sensors, -1) \
                .transpose(1, 2)                                                    # (B,S,Q,7)
            x_in = torch.cat([xe, te], dim=-1).transpose(1, 2)                      # (B,S,P,8)
            od_dec = torch.stack([od_dec_t[s+P:s+P+Q] for s in batch]).transpose(1, 2).unsqueeze(-1) \
                if od_dec_t is not None else None                                   # (B,S,Q,1)
            od_enc = torch.stack([od_dec_t[s:s+P].mean(dim=0) for s in batch]) \
                if od_dec_t is not None else None                                   # (B,S) -- P-window OD summary, for od_graph
            if return_starts:
                yield x_in, xd, od_dec, td, od_enc, batch
                continue
            yield x_in, xd, od_dec, td, od_enc

    assert not (args.use_od_injection and args.od_graph), "--use_od_injection and --od_graph are mutually exclusive"
    Fday = 1 + TIME_FEAT_DIM
    model_kwargs = {}
    if args.node_se != "none":
        se = np.load(f"{GTS}/node2vec_se_{args.task}.npz")["emb"]
        model_kwargs = {"se_init": se, "se_frozen": args.node_se == "frozen"}
    if args.od_graph:
        assert args.model in ("dcrnn", "gwnet", "gts"), "--od_graph is only implemented for dcrnn/gwnet/gts"
        model_kwargs["use_od_graph"] = True
    base_model = MODELS[args.model](A, Fday, n_sensors, H=args.hidden, q_len=Q, **model_kwargs).to(device)
    if args.use_od_injection:
        wrapper_cls = {"mature": MatureInjectionWrapper, "crossattn": CrossAttnInjectionWrapper,
                       "simple": ODInjectionWrapper}[args.fusion_style]
        model = wrapper_cls(base_model, args.hidden, Q, TIME_FEAT_DIM).to(device)
    else:
        model = base_model
    if args.warm_start_from:
        model.load_state_dict(torch.load(args.warm_start_from, weights_only=True))
        print(f"[FAST-SCREENING] warm-started from {args.warm_start_from} -- NOT an independent-fold result")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} task={args.task} use_od={bool(args.use_od_injection)} od_graph={bool(args.od_graph)} "
          f"params: {n_params:,}  device={device}  loss_fn={args.loss_fn}")
    if args.loss_fn in ("geh", "combined") and args.task == "speed":
        print(f"WARNING: --loss_fn {args.loss_fn} on --task speed -- GEH's variance-stabilization assumes "
              "count-like (Poisson) noise, which volume has and speed does not. Proceeding anyway "
              "since you asked, but treat this as an experiment, not a recommended default.")

    def compute_loss(pred, target_n):
        if args.loss_fn == "geh":
            return geh_loss(pred, target_n, mu, sd)
        if args.loss_fn == "combined":
            # smooth_l1 (Huber), not raw MSE -- matches what every other loss_fn in this codebase
            # uses as its regression term (Huber ~= MSE for small errors, more outlier-robust for
            # large ones), kept for consistency rather than switching bases per option.
            return F.smooth_l1_loss(pred, target_n) + args.geh_weight * geh_loss(pred, target_n, mu, sd)
        return F.smooth_l1_loss(pred, target_n)

    def run_model(x_in, od_dec, td, od_enc):
        if args.use_od_injection:
            return model(x_in, stn_idx, od_dec, td)
        if args.od_graph:
            return model(x_in, stn_idx, od_summary=od_enc)
        return model(x_in, stn_idx)

    log_path = f"{out_dir}/train_log.jsonl"; open(log_path, "w").close()
    best_val = float("inf"); epochs_no_improve = 0; stopped_early_at = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for x_in, xd, od_dec, td, od_enc in get_batch(train_idx, args.batch_size):
            opt.zero_grad()
            pred = run_model(x_in, od_dec, td, od_enc)  # (B,S,Q)
            loss = compute_loss(pred, xd.transpose(1, 2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_in, xd, od_dec, td, od_enc in get_batch(val_idx, args.batch_size, shuffle=False):
                pred = run_model(x_in, od_dec, td, od_enc)
                val_losses.append(compute_loss(pred, xd.transpose(1, 2)).item())

        tr_loss, va_loss = float(np.mean(tr_losses)), float(np.mean(val_losses))
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "elapsed_s": round(time.time()-t0, 1)}
        with open(log_path, "a") as f: f.write(json.dumps(rec) + "\n")
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

    model.load_state_dict(torch.load(f"{out_dir}/best.pt", weights_only=True))
    model.eval()
    test_true, test_pred, test_betas = [], [], []
    csv_rows = []
    with torch.no_grad():
        for x_in, xd, od_dec, td, od_enc, starts in get_batch(test_idx, args.batch_size, shuffle=False, return_starts=True):
            pred = run_model(x_in, od_dec, td, od_enc)  # (B,S,Q)
            if args.use_od_injection:
                test_betas.append(model.last_mean_beta)
            true_denorm = (xd.cpu().numpy() * sd + mu)          # (B,Q,S)
            pred_denorm = (pred.transpose(1, 2).cpu().numpy() * sd + mu)  # (B,Q,S)
            test_true.append(true_denorm.ravel())
            test_pred.append(pred_denorm.ravel())
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
                    f"true_{val_key}", f"pred_{val_key}"])
        w.writerows(csv_rows)
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
    if args.task == "volume":
        # always report GEH<5 pass rate on the test set regardless of which loss trained the
        # model, so --loss_fn smooth_l1 vs geh runs are directly comparable on the metric that
        # actually matters (not just whichever loss each one was optimizing).
        M = np.clip(test_pred, 0, None); C = np.clip(test_true, 0, None)
        geh_arr = np.sqrt(2 * (M - C) ** 2 / (M + C + 1.0))
        summary["test_geh_pass_rate"] = float((geh_arr < 5).mean())
        summary["test_geh_median"] = float(np.median(geh_arr))
    if args.use_od_injection:
        summary["final_beta_mean"] = float(np.mean(test_betas))
    with open(f"{out_dir}/summary.json", "w") as f: json.dump(summary, f, indent=2)
    print("TEST:", summary)


if __name__ == "__main__":
    main()
