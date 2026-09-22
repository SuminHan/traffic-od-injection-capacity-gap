"""
Two more published architectures -- STID (Shao et al., CIKM 2022, "Spatial-Temporal Identity: A
Simple yet Effective Baseline") and STGCN (Yu et al., IJCAI 2018, the classic ChebConv
spatio-temporal baseline) -- added in a SEPARATE file so the live-running GTS sweep
(run_gts_ext30_sweep.py) and the live-running PDFormer/AGCRN/MTGNN sweep
(run_extra_models_ext30_sweep.py) are never touched by this addition. Reference implementations
consulted for the core mechanism only (NOT ported, LibCity's own config/data pipeline is
unrelated to ours): libcity_ref/libcity/model/traffic_speed_prediction/{STID,STGCN}.py.

Same interface contract as every class in models_baselines.py / models_baselines_extra.py:
__init__(A, Fday, N, H=64, q_len=12, se_init=None, se_frozen=True, ...) and
forward(x, stn_idx, return_context=False) with x:(B,N,P,Fday) -> (B,N,q_len), or (B,N,H) context
when return_context=True (`self.head = nn.Linear(H, q_len)` applied when not return_context, the
same pattern GMANBaseline/PDFormerBaseline/AGCRNBaseline all use).

Documented simplifications:
  - STID: kept deliberately, faithfully GRAPH-FREE -- this is the entire point of the model (a
    pure per-node MLP with learned spatial-identity + temporal-identity embeddings, no message
    passing at all) and it is being added specifically as a very-low-capacity, non-graph contrast
    point for this paper's capacity-gap narrative, so `A` is accepted (interface contract) but
    genuinely never used in the forward pass.
    CORRECTED 2026-09-21: an earlier version of this file replaced STID's discrete time-of-day/
    day-of-week identity embeddings with the continuous sin/cos calendar channels already present
    in Fday, fed through the same shared linear ts_proj as the raw signal. That is a substantive
    architecture change, not a cosmetic one -- the paper's own ablation attributes a real chunk of
    STID's accuracy to exactly these embeddings being DISCRETE lookup tables (able to learn sharp,
    non-sinusoidal per-bin patterns, e.g. weekday-vs-weekend asymmetry) rather than a smooth basis
    function shared with the continuous regression input. It is the likely cause of a PEMS-BAY
    reproduction gap that grows sharply with horizon (+9.3% MAE at h3 -> +20.4% at h12, see
    NIGHT_LOOP_NOTES.md and pemsbay_eval.log) -- exactly the pattern expected if periodicity
    information is what's missing, since near-horizon errors can coast on autocorrelation but
    far-horizon ones need to know "what hour of what weekday" precisely. Reverted to the paper's
    actual design below: `self.tod_emb`/`self.dow_emb` are real nn.Embedding lookup tables,
    indexed by discrete bins decoded from the sin/cos channels of the last input timestep (which
    is exact up to float32 rounding, since those channels ARE sin/cos of the true bin index) --
    this keeps the same (x, stn_idx) call signature every other model in this file uses, so no
    caller needs to change except to optionally pass `steps_per_day` (constructor kwarg, default
    24 for this pipeline's hourly Seoul data; PEMS-BAY is 5-minute data, 288 steps/day, passed
    explicitly by train_benchmark_model.py) for datasets whose day is not 24 discrete hours.
  - STGCN: keeps the paper's two-block ST-Conv backbone (temporal gated conv -> Chebyshev graph
    conv -> temporal gated conv, x2) with a scaled-Laplacian Chebyshev basis precomputed once from
    `A` (order Ks=3, standard). Uses GLU-gated temporal convolution exactly as the original.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# STID (Shao et al., CIKM 2022) -- deliberately graph-free
# ---------------------------------------------------------------------------
class _STIDResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):  # (B, N, dim)
        h = self.fc2(self.drop(self.act(self.fc1(x))))
        return h + x


class STIDBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, q_len=12, p_len=12, num_block=3, emb_dim=None,
                 se_init=None, se_frozen=True, **kwargs):
        """p_len must match the actual lookback window P used at train time (this pipeline's
        default and the value every sweep script uses is 12) -- built eagerly here, NOT lazily
        on first forward, because train_baseline_model.py constructs the optimizer immediately
        after model construction and before any forward pass; a lazily-created submodule would
        silently never receive gradient updates (its params would not be in the optimizer).

        UNCHANGED as of 2026-09-21: this is the original graph-free/continuous-time-feature
        architecture, kept exactly as-is so every already-trained Seoul checkpoint (capacity law,
        capacity sweep, main tables) keeps loading. The corrected architecture with real discrete
        time-of-day/day-of-week identity embeddings lives in STIDFixedBaseline below (registered
        as MODELS_EXTRA2["stid_fixed"]) -- see that class's docstring for why, and
        NIGHT_LOOP_NOTES.md 2026-09-21 for the PEMS-BAY validation result (closes part, not all,
        of the reproduction gap). Do not merge the two until/unless a full Seoul retrain of STID
        is deliberately decided on -- editing this class in place breaks every existing
        checkpoint's state_dict with no warning beyond a shape-mismatch crash at load time
        (this happened once already, see the same NIGHT_LOOP_NOTES.md entry)."""
        super().__init__()
        self.N = N
        self.p_len = p_len
        emb_dim = emb_dim if emb_dim is not None else H // 4
        ts_emb_dim = H - emb_dim
        assert ts_emb_dim > 0, "H too small relative to emb_dim for STID"
        self.emb = nn.Embedding(N, emb_dim)
        if se_init is not None:
            assert se_init.shape == (N, emb_dim)
            self.emb.weight.data.copy_(torch.as_tensor(se_init, dtype=torch.float32))
            self.emb.weight.requires_grad = not se_frozen
        self.ts_proj = nn.Linear(p_len * Fday, ts_emb_dim)
        self.encoder = nn.Sequential(*[_STIDResBlock(H) for _ in range(num_block)])
        self.head = nn.Linear(H, q_len)

    def forward(self, x, stn_idx, return_context=False):  # x: (B, N, P, Fday)
        B, N, P, Fd = x.shape
        assert P == self.p_len, f"STIDBaseline built for p_len={self.p_len}, got P={P}"
        ts_flat = x.reshape(B, N, P * Fd)
        ts_emb = self.ts_proj(ts_flat)  # (B, N, ts_emb_dim)
        node_emb = self.emb(stn_idx)[None, :, :].expand(B, -1, -1)  # (B, N, emb_dim)
        hidden = torch.cat([ts_emb, node_emb], dim=-1)  # (B, N, H) -- purely per-node, no graph mixing
        hidden = self.encoder(hidden)
        if return_context:
            return hidden
        return self.head(hidden)  # (B, N, Q)


class STIDFixedBaseline(nn.Module):
    """Corrected STID architecture (see module docstring's 2026-09-21 note): real discrete
    time-of-day/day-of-week nn.Embedding lookup tables, matching Shao et al.'s actual mechanism,
    instead of continuous sin/cos channels folded into ts_proj. Kept as a SEPARATE class (not a
    replacement of STIDBaseline) so it never invalidates existing STIDBaseline checkpoints.
    Registered as MODELS_EXTRA2["stid_fixed"] -- used so far only for the PEMS-BAY sanity-check
    retrain (bench_stid_pemsbay_v2fix), not yet for any Seoul main-table/capacity-law run."""
    def __init__(self, A, Fday, N, H=64, q_len=12, p_len=12, num_block=3, emb_dim=None,
                 se_init=None, se_frozen=True, steps_per_day=24, **kwargs):
        super().__init__()
        self.N = N
        self.p_len = p_len
        self.steps_per_day = steps_per_day
        emb_dim = emb_dim if emb_dim is not None else H // 4
        tod_dim = max(H // 8, 1)
        dow_dim = max(H // 8, 1)
        ts_emb_dim = H - emb_dim - tod_dim - dow_dim
        assert ts_emb_dim > 0, "H too small relative to identity-embedding dims for STID"
        self.emb = nn.Embedding(N, emb_dim)
        if se_init is not None:
            assert se_init.shape == (N, emb_dim)
            self.emb.weight.data.copy_(torch.as_tensor(se_init, dtype=torch.float32))
            self.emb.weight.requires_grad = not se_frozen
        self.tod_emb = nn.Embedding(steps_per_day, tod_dim)
        self.dow_emb = nn.Embedding(7, dow_dim)
        self.ts_proj = nn.Linear(p_len * Fday, ts_emb_dim)
        self.encoder = nn.Sequential(*[_STIDResBlock(H) for _ in range(num_block)])
        self.head = nn.Linear(H, q_len)

    def forward(self, x, stn_idx, return_context=False):  # x: (B, N, P, Fday)
        B, N, P, Fd = x.shape
        assert P == self.p_len, f"STIDFixedBaseline built for p_len={self.p_len}, got P={P}"
        ts_flat = x.reshape(B, N, P * Fd)
        ts_emb = self.ts_proj(ts_flat)  # (B, N, ts_emb_dim)
        node_emb = self.emb(stn_idx)[None, :, :].expand(B, -1, -1)  # (B, N, emb_dim)

        # Discrete temporal identity, decoded from the last input step's sin/cos calendar
        # channels (x[..., 1:5] = sin_hour, cos_hour, sin_dow, cos_dow). Exact up to float32
        # rounding, since those channels ARE sin/cos of the true discrete bin index.
        sin_h, cos_h = x[:, 0, -1, 1], x[:, 0, -1, 2]  # (B,) -- identical across sensors N
        sin_d, cos_d = x[:, 0, -1, 3], x[:, 0, -1, 4]
        hour_idx = torch.remainder(
            torch.round(torch.atan2(sin_h, cos_h) * self.steps_per_day / (2 * np.pi)),
            self.steps_per_day).long()
        dow_idx = torch.remainder(
            torch.round(torch.atan2(sin_d, cos_d) * 7 / (2 * np.pi)), 7).long()
        tod_e = self.tod_emb(hour_idx)[:, None, :].expand(-1, N, -1)  # (B, N, tod_dim)
        dow_e = self.dow_emb(dow_idx)[:, None, :].expand(-1, N, -1)  # (B, N, dow_dim)

        hidden = torch.cat([ts_emb, node_emb, tod_e, dow_e], dim=-1)  # (B, N, H)
        hidden = self.encoder(hidden)
        if return_context:
            return hidden
        return self.head(hidden)  # (B, N, Q)


# ---------------------------------------------------------------------------
# STGCN (Yu et al., IJCAI 2018)
# ---------------------------------------------------------------------------
def _scaled_laplacian_cheb(A, ks=3):
    """Returns (ks, N, N) Chebyshev polynomial basis of the scaled graph Laplacian, as in the
    original STGCN paper. A: (N,N) tensor (any nonnegative weighted adjacency)."""
    a = A.detach().cpu().numpy().astype(np.float64)
    n = a.shape[0]
    d = a.sum(axis=1)
    lap = np.diag(d) - a
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            for j in range(n):
                if d[i] > 0 and d[j] > 0:
                    lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    eigvals = np.linalg.eigvalsh(lap)
    lam_max = max(eigvals.max(), 1e-6)
    lap_scaled = 2 * lap / lam_max - np.eye(n)

    cheb = [np.eye(n), lap_scaled.copy()]
    for _ in range(2, ks):
        cheb.append(2 * lap_scaled @ cheb[-1] - cheb[-2])
    cheb = np.stack(cheb[:ks], axis=0)  # (ks, N, N)
    return torch.tensor(cheb, dtype=torch.float32)


class _Align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        if c_in != c_out:
            self.proj = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):  # (B, C, T, N)
        if self.c_in == self.c_out:
            return x
        return self.proj(x)


class _TemporalGLU(nn.Module):
    def __init__(self, kt, c_in, c_out):
        super().__init__()
        self.kt = kt
        self.align = _Align(c_in, c_out)
        self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1))

    def forward(self, x):  # (B, C, T, N) -> (B, c_out, T-kt+1, N)
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        p, q = torch.split(self.conv(x), self.conv.out_channels // 2, dim=1)
        return (p + x_in) * torch.sigmoid(q)


class _SpatioCheb(nn.Module):
    def __init__(self, c_in, c_out, cheb):
        super().__init__()
        self.register_buffer("cheb", cheb)  # (ks, N, N)
        ks = cheb.shape[0]
        self.theta = nn.Parameter(torch.empty(ks, c_in, c_out))
        nn.init.xavier_uniform_(self.theta)
        self.align = _Align(c_in, c_out)

    def forward(self, x):  # (B, C, T, N) -> (B, c_out, T, N)
        # x_g[k] = x @ cheb[k]^T over the node axis, then mix channels with theta[k]
        out = 0.0
        for k in range(self.cheb.shape[0]):
            xk = torch.einsum("bctn,mn->bctm", x, self.cheb[k])  # graph-conv along node axis
            out = out + torch.einsum("bctm,co->botm", xk, self.theta[k])
        return F.relu(out + self.align(x))


class _STConvBlock(nn.Module):
    def __init__(self, c_in, c_hid, c_out, kt, cheb):
        super().__init__()
        self.t1 = _TemporalGLU(kt, c_in, c_hid)
        self.s = _SpatioCheb(c_hid, c_hid, cheb)
        self.t2 = _TemporalGLU(kt, c_hid, c_out)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.t1(x)
        x = self.s(x)
        x = self.t2(x)
        return self.bn(x)


class STGCNBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, q_len=12, ks=3, kt=3, emb_dim=16,
                 se_init=None, se_frozen=True, **kwargs):
        super().__init__()
        self.N = N
        self.emb = nn.Embedding(N, emb_dim)
        if se_init is not None:
            assert se_init.shape == (N, emb_dim)
            self.emb.weight.data.copy_(torch.as_tensor(se_init, dtype=torch.float32))
            self.emb.weight.requires_grad = not se_frozen
        F_in = Fday + emb_dim
        cheb = _scaled_laplacian_cheb(A.to(torch.float32), ks=ks)
        self.block1 = _STConvBlock(F_in, H // 2, H, kt, cheb)
        self.block2 = _STConvBlock(H, H // 2, H, kt, cheb)
        # kt=3 halves receptive field by 2 each temporal conv, x2 blocks x2 convs -> -8 total;
        # final 1x1 temporal conv collapses whatever time steps remain to a single pooled step.
        self.out_temporal = nn.Conv2d(H, H, (1, 1))
        self.head = nn.Linear(H, q_len)

    def forward(self, x, stn_idx, return_context=False):  # x: (B, N, P, Fday)
        B, N, P, _ = x.shape
        emb = self.emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        x = torch.cat([x, emb], dim=-1)  # (B, N, P, F_in)
        x = x.permute(0, 3, 2, 1)  # (B, F_in, P, N)
        x = self.block1(x)
        x = self.block2(x)  # (B, H, T', N), T' = P - 8 if P > 8 else collapsed by padding below
        if x.size(2) < 1:
            x = F.pad(x, (0, 0, 1 - x.size(2), 0))
        x = F.relu(self.out_temporal(x))
        ctx = x.mean(dim=2)  # (B, H, N) -- pool over whatever time steps survived
        ctx = ctx.permute(0, 2, 1)  # (B, N, H)
        if return_context:
            return ctx
        return self.head(ctx)  # (B, N, Q)


MODELS_EXTRA2 = {"stid": STIDBaseline, "stid_fixed": STIDFixedBaseline, "stgcn": STGCNBaseline}
