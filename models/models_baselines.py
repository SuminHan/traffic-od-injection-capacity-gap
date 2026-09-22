"""
Published-baseline architectures (DCRNN, Graph WaveNet, GMAN), adapted from smhan's clean
standalone reimplementations at /path/to/raw_data/deep_baselines/{dcrnn,gwnet,gman}.py
(themselves distilled from the LibCity reference implementations -- see that directory's
REFERENCES.md). Those originals target a DIFFERENT task (174 stations, daily P=14/Q=1 direct
next-day regression); this file keeps every core mechanism (diffusion convolution / WaveNet
gated dilated TCN with adaptive adjacency / spatial+temporal attention blocks) UNCHANGED, and
makes exactly one adaptation across all three: the single-step head `Linear(H, 1)` becomes a
direct multi-horizon head `Linear(H, Q)` that regresses all Q=12 future hourly steps at once
from the same pooled per-node context representation the original used for its single next-step
prediction. This is a standard "direct multi-horizon" simplification (as opposed to our own
TrafficModel's autoregressive decoder) -- documented here rather than hidden, since it's the one
real difference from the reference architectures.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_common import DiffGRUCell, diffusion_supports, apply_diffusion_supports, \
    gumbel_softmax_sample, symmetrize


def _init_node_emb(emb, se_init, se_frozen, N, emb_dim):
    """Shared by all three baselines: optionally replace a from-scratch nn.Embedding(N, emb_dim)
    node-identity lookup with a precomputed spatial embedding (e.g. node2vec, see
    build_node2vec_se.py). se_frozen=True keeps it fixed; False lets it fine-tune from that init.
    Confirmed on GMAN (fold0, both volume and speed) that frozen hurts but finetune-from-node2vec
    consistently helps over from-scratch random init -- generalizing that to DCRNN/GWNET here."""
    if se_init is not None:
        assert se_init.shape == (N, emb_dim)
        emb.weight.data.copy_(torch.as_tensor(se_init, dtype=torch.float32))
        emb.weight.requires_grad = not se_frozen


# ---------------------------------------------------------------------------
# DCRNN (Li et al., ICLR 2018)
# ---------------------------------------------------------------------------
class DCRNNBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, n_enc=2, emb_dim=16, q_len=12, se_init=None, se_frozen=True,
                 use_od_graph=False):
        """use_od_graph=True: same idea as GWNETBaseline's use_od_graph -- inject OD by modulating
        the diffusion adjacency itself (per-node gate outer-product scaling the fixed A) rather
        than as a node feature. DCRNN has no learned adaptive adjacency slot like GWNET's nodevec1/2,
        so this modulates its one fixed structural graph directly instead."""
        super().__init__()
        self.N = N
        self.emb = nn.Embedding(N, emb_dim)
        _init_node_emb(self.emb, se_init, se_frozen, N, emb_dim)
        F_in = Fday + emb_dim
        self.enc = nn.ModuleList()
        self.enc.append(DiffGRUCell(A, F_in, H))
        for _ in range(n_enc - 1):
            self.enc.append(DiffGRUCell(A, H, H))
        self.dec = DiffGRUCell(A, H, H)
        self.start = nn.Parameter(torch.zeros(H))
        self.head = nn.Linear(H, q_len)
        self.use_od_graph = use_od_graph
        if use_od_graph:
            self.register_buffer("_A_raw", A.to(torch.float32))
            self.od_gate_proj = nn.Linear(1, 1)

    def forward(self, x, stn_idx, return_context=False, od_summary=None):  # x: (B, N, P, Fday)
        B, N, P, _ = x.shape
        emb = self.emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        x = torch.cat([x, emb], dim=-1)
        x = x.transpose(1, 2)  # (B, P, N, F)

        A_override = None
        if self.use_od_graph and od_summary is not None:
            gate = torch.sigmoid(self.od_gate_proj(od_summary.unsqueeze(-1)))  # (B,N,1)
            mod = torch.bmm(gate, gate.transpose(1, 2))  # (B,N,N)
            A_override = self._A_raw.unsqueeze(0) * mod  # (B,N,N)

        prev_seq = None
        h_final = None
        for l, cell in enumerate(self.enc):
            h_cur = torch.zeros((B, N, cell.H), device=x.device)
            cur_seq = []
            for t in range(P):
                inp = x[:, t, :, :] if l == 0 else prev_seq[:, t, :, :]
                h_cur = cell(inp, h_cur, A=A_override)
                cur_seq.append(h_cur)
            prev_seq = torch.stack(cur_seq, dim=1)
            h_final = h_cur

        d = self.dec(self.start.expand(B, N, -1), h_final, A=A_override)  # (B, N, H)
        if return_context:
            return d
        return self.head(d)  # (B, N, Q)


# ---------------------------------------------------------------------------
# DCRNN, seq2seq variant: a TRUE Q-step autoregressive decoder (matching both the
# original DCRNN paper and our own TrafficModel), built to give OD injection a proper
# per-step home instead of ODInjectionWrapper's single-shot post-hoc bolt-on.
# Diagnosis that motivated this: DCRNNBaseline+ODInjectionWrapper was statistically
# indistinguishable from injecting SHUFFLED (meaningless) OD (fold0: -0.23% for shuffled vs
# +1.83% mean/not-significant for real OD across 23 folds) -- the single pooled context vector
# + late single-shot fusion likely just doesn't give the signal anywhere useful to act. Here OD
# is fused at EVERY decode step via the same per-node/per-step gate design already validated in
# TrafficModel: beta = sigmoid(gate([h_dec, h_od])); h_final = beta*h_dec + (1-beta)*h_od, with
# recurrence always carried forward by the PURE decoder branch (not the fused one).
# ---------------------------------------------------------------------------
class DCRNNSeq2Seq(nn.Module):
    def __init__(self, A, Fday, N, H=64, n_enc=2, emb_dim=16, q_len=12, use_od=False,
                 se_init=None, se_frozen=True):
        super().__init__()
        self.N = N
        self.q_len = q_len
        self.use_od = use_od
        self.emb = nn.Embedding(N, emb_dim)
        _init_node_emb(self.emb, se_init, se_frozen, N, emb_dim)
        F_in = Fday + emb_dim
        self.enc = nn.ModuleList()
        self.enc.append(DiffGRUCell(A, F_in, H))
        for _ in range(n_enc - 1):
            self.enc.append(DiffGRUCell(A, H, H))
        self.dec = DiffGRUCell(A, F_in, H)  # real per-step [value, calendar, node-emb] input, not a start vector
        self.head = nn.Linear(H, 1)
        if use_od:
            self.od_proj = nn.Sequential(nn.Linear(1, H), nn.ReLU(), nn.Linear(H, H))
            self.gate_net = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 1))

    def _fuse(self, h_dec, od_step):
        h_od = self.od_proj(od_step)
        beta = torch.sigmoid(self.gate_net(torch.cat([h_dec, h_od], dim=-1)))
        return beta * h_dec + (1 - beta) * h_od, beta

    def forward(self, xe, stn_idx, td, xd_true=None, od_dec=None, teacher_forcing=0.0):
        """xe: (B,N,P,Fday) encoder window. td: (B,N,Q,Fday-1) future calendar features.
        xd_true: (B,N,Q) ground truth future values (for teacher forcing; None at pure inference).
        od_dec: (B,N,Q,1) routed OD forecast per decode step, only used if use_od=True."""
        B, N, P, _ = xe.shape
        Q = self.q_len
        emb = self.emb(stn_idx)[None, :, None, :]  # (1,N,1,E)
        enc_in = torch.cat([xe, emb.expand(B, N, P, -1)], dim=-1)

        prev_seq = None
        h_final = None
        for l, cell in enumerate(self.enc):
            h_cur = torch.zeros((B, N, cell.H), device=xe.device)
            cur_seq = []
            for t in range(P):
                inp = enc_in[:, :, t, :] if l == 0 else prev_seq[:, :, t, :]
                h_cur = cell(inp, h_cur)
                cur_seq.append(h_cur)
            prev_seq = torch.stack(cur_seq, dim=2)
            h_final = h_cur

        h = h_final
        prev_val = xe[:, :, -1, 0:1]  # (B,N,1) last observed value, autoregressive seed
        emb_b = emb[:, :, 0, :].expand(B, -1, -1)  # (B,N,E)
        outs = []
        gate_vals = []
        for t in range(Q):
            dec_in = torch.cat([prev_val, td[:, :, t, :], emb_b], dim=-1)
            h_dec = self.dec(dec_in, h)
            if self.use_od:
                h_fused, beta = self._fuse(h_dec, od_dec[:, :, t, :])
                gate_vals.append(beta.mean())
            else:
                h_fused = h_dec
            pred = self.head(h_fused)  # (B,N,1)
            outs.append(pred)
            h = h_dec  # recurrence always carried by the pure decoder branch
            use_true = (xd_true is not None) and (torch.rand(()) < teacher_forcing)
            prev_val = xd_true[:, :, t:t+1] if use_true else pred.detach()
        if self.use_od:
            self.last_mean_beta = torch.stack(gate_vals).mean().item()
        return torch.cat(outs, dim=-1)  # (B,N,Q)


# ---------------------------------------------------------------------------
# Graph WaveNet (Wu et al., NeurIPS 2019)
# ---------------------------------------------------------------------------
def _asym_transition(A):
    d = A.sum(1).clamp(min=1.0)
    return A / d.unsqueeze(1)


class _NConv(nn.Module):
    def forward(self, x, adj):
        # adj: (N,N) shared across the batch, OR (B,N,N) batch-specific (OD-modulated graph)
        if adj.dim() == 2:
            return torch.einsum("bcjt,ij->bcit", x, adj).contiguous()
        return torch.einsum("bcjt,bij->bcit", x, adj).contiguous()


class _GCN(nn.Module):
    def __init__(self, c_in, c_out, support_len=3, order=2):
        super().__init__()
        self.nconv = _NConv()
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=(1, 1))
        self.order = order

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        return self.mlp(torch.cat(out, dim=1))


class GWNETBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, blocks=4, layers=2, kernel_size=2,
                 skip_mult=4, end_mult=8, emb_dim=16, nv_dim=10, q_len=12,
                 se_init=None, se_frozen=True, use_od_graph=False):
        """use_od_graph=True: instead of (or in addition to) node-feature gated fusion, inject the
        routed OD signal directly into the GRAPH STRUCTURE GWNET already learns (its adaptive
        adjacency `adp`). A per-node OD-activity gate g_i=sigmoid(od_proj(od_i)) is outer-producted
        into a (B,N,N) co-modulation and multiplied into adp, so message passing follows "where
        traffic is actually being routed right now" rather than only the fixed learned topology.
        This is the natural place for flow information in a GNN -- the edges -- rather than one
        more node feature competing with calendar/history for the model's attention."""
        super().__init__()
        self.use_od_graph = use_od_graph
        if use_od_graph:
            self.od_gate_proj = nn.Linear(1, 1)
        self.N = N
        self.blocks, self.layers, self.k = blocks, layers, kernel_size
        C, Cskip, Cend = H, H * skip_mult, H * end_mult
        self.emb = nn.Embedding(N, emb_dim)
        _init_node_emb(self.emb, se_init, se_frozen, N, emb_dim)
        self.start_conv = nn.Conv2d(Fday + emb_dim, C, kernel_size=(1, 1))

        self.register_buffer("supp_f", _asym_transition(A.to(torch.float32)))
        self.register_buffer("supp_b", _asym_transition(A.to(torch.float32).t()))
        self.nodevec1 = nn.Parameter(torch.randn(N, nv_dim))
        self.nodevec2 = nn.Parameter(torch.randn(nv_dim, N))
        support_len = 3

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.receptive_field = 1
        for _ in range(blocks):
            scope = kernel_size - 1
            dil = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(C, C, (1, kernel_size), dilation=dil))
                self.gate_convs.append(nn.Conv2d(C, C, (1, kernel_size), dilation=dil))
                self.skip_convs.append(nn.Conv2d(C, Cskip, kernel_size=(1, 1)))
                self.gconv.append(_GCN(C, C, support_len=support_len, order=2))
                self.bn.append(nn.BatchNorm2d(C))
                dil *= 2
                self.receptive_field += scope
                scope *= 2
        self.end_conv_1 = nn.Conv2d(Cskip, Cend, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(Cend, q_len, kernel_size=(1, 1))
        self.ctx_proj = nn.Linear(Cend, H)  # only used when return_context=True

    def forward(self, x, stn_idx, return_context=False, od_summary=None):  # x: (B, N, P, Fday)
        B, N, P, _ = x.shape
        emb = self.emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        x = torch.cat([x, emb], dim=-1)
        x = x.permute(0, 3, 1, 2)  # (B, F, N, P)
        x = F.pad(x, (1, 0, 0, 0))
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        x = self.start_conv(x)

        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)  # (N,N)
        if self.use_od_graph and od_summary is not None:
            gate = torch.sigmoid(self.od_gate_proj(od_summary.unsqueeze(-1)))  # (B,N,1)
            mod = torch.bmm(gate, gate.transpose(1, 2))  # (B,N,N) OD-activity co-modulation
            adp = adp.unsqueeze(0) * mod  # (B,N,N) -- now batch-specific
        supports = [self.supp_f, self.supp_b, adp]

        skip = None
        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate
            s = self.skip_convs[i](x)
            skip = s if skip is None else (s + skip[:, :, :, -s.size(3):])
            x = self.gconv[i](x, supports)
            x = x + residual[..., -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))  # (B, Cend, N, 1)
        if return_context:
            return self.ctx_proj(x[:, :, :, -1].permute(0, 2, 1))  # (B, N, H)
        x = self.end_conv_2(x)  # (B, Q, N, L) with L==1 by construction (P+1==receptive_field)
        return x[:, :, :, -1].permute(0, 2, 1)  # (B, N, Q)


# ---------------------------------------------------------------------------
# GMAN (Zheng et al., AAAI 2020)
# ---------------------------------------------------------------------------
class _SpatialAttn(nn.Module):
    def __init__(self, H, mask_A):
        super().__init__()
        self.H = H
        self.mask = mask_A
        self.Wq = nn.Linear(H, H)
        self.Wk = nn.Linear(H, H)
        self.Wv = nn.Linear(H, H)
        self.scale = H ** 0.5

    def forward(self, X):  # (B, P, N, H)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        scores = torch.einsum("bpnh,bpmh->bpnm", Q, K) / self.scale
        scores = scores.masked_fill(~self.mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bpnm,bpmh->bpnh", attn, V)


class _TemporalAttn(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.Wq = nn.Linear(H, H)
        self.Wk = nn.Linear(H, H)
        self.Wv = nn.Linear(H, H)
        self.scale = H ** 0.5

    def forward(self, X):  # (B, P, N, H)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        scores = torch.einsum("bptn,bpmn->bptm", Q, K) / self.scale
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bptm,bpmn->bptn", attn, V)


class _GMANBlock(nn.Module):
    def __init__(self, H, mask_A):
        super().__init__()
        self.spatial = _SpatialAttn(H, mask_A)
        self.temporal = _TemporalAttn(H)
        self.norm_s = nn.LayerNorm(H)
        self.norm_t = nn.LayerNorm(H)

    def forward(self, X):
        X = self.norm_s(X + self.spatial(X))
        X = self.norm_t(X + self.temporal(X))
        return X


class GMANBaseline(nn.Module):
    """GMAN's temporal attention has NO inherent sequential inductive bias (unlike DCRNN's GRU
    recurrence or GWNET's causal dilated convolutions) -- pure self-attention over the P lookback
    steps is permutation-*sensitive* only insofar as the calendar features differ per step, but
    has no direct "how many hours ago" signal. Standard sinusoidal positional encoding (as in the
    original Transformer) is added over the P axis so the temporal attention has an explicit
    recency signal, matching what DCRNN/GWNET get "for free" from their architectures."""
    def __init__(self, A, Fday, N, H=64, L=2, emb_dim=16, q_len=12, max_p=64,
                 se_init=None, se_frozen=True):
        """se_init: optional (N, emb_dim) precomputed spatial embedding (e.g. node2vec, see
        build_node2vec_se.py) to use INSTEAD OF a from-scratch-learned node identity lookup --
        this is what the original GMAN paper's "SE" actually is. se_frozen=True matches the
        paper (SE precomputed offline, not fine-tuned); False lets it fine-tune from that init."""
        super().__init__()
        self.N = N
        self.emb = nn.Embedding(N, emb_dim)
        if se_init is not None:
            assert se_init.shape == (N, emb_dim)
            self.emb.weight.data.copy_(torch.as_tensor(se_init, dtype=torch.float32))
            self.emb.weight.requires_grad = not se_frozen
        F_in = Fday + emb_dim
        self.proj = nn.Linear(F_in, H)
        mask_A = (A > 0)
        eye = torch.eye(N, dtype=torch.bool, device=A.device)
        mask_A = mask_A | eye
        self.blocks = nn.ModuleList([_GMANBlock(H, mask_A) for _ in range(L)])
        self.head = nn.Linear(H, q_len)

        pe = torch.zeros(max_p, H)
        pos = torch.arange(max_p, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, H, 2, dtype=torch.float32) * (-math.log(10000.0) / H))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe)  # (max_p, H)

    def forward(self, x, stn_idx, return_context=False):  # x: (B, N, P, Fday)
        B, N, P, _ = x.shape
        emb = self.emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        x = torch.cat([x, emb], dim=-1)
        X = self.proj(x)
        X = X.permute(0, 2, 1, 3)  # (B, P, N, H)
        X = X + self.pos_enc[:P][None, :, None, :]
        for blk in self.blocks:
            X = blk(X)
        out = X[:, -1, :, :]  # (B, N, H) last lookback step
        if return_context:
            return out
        return self.head(out)  # (B, N, Q)


# ---------------------------------------------------------------------------
# GTS (Shang et al., ICLR 2021 "Discrete Graph Structure Learning") -- same
# diffusion-conv-GRU core as DCRNN, but the adjacency is LEARNED from per-node
# embeddings via a small MLP + Gumbel-softmax, initialized at the fixed
# physical graph (zero-init final MLP layer) and refined during training.
# ---------------------------------------------------------------------------
class GTSBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, n_enc=2, emb_dim=32, q_len=12,
                 se_init=None, se_frozen=True, use_od_graph=False):
        super().__init__()
        self.N = N
        self.emb = nn.Embedding(N, emb_dim)
        _init_node_emb(self.emb, se_init, se_frozen, N, emb_dim)
        F_in = Fday + emb_dim
        self.graph_mlp = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, N))
        nn.init.zeros_(self.graph_mlp[-1].weight)
        nn.init.zeros_(self.graph_mlp[-1].bias)
        A = A.to(torch.float32)
        self.register_buffer("logits_init", torch.log(A + 1e-6))

        self.enc = nn.ModuleList()
        self.enc.append(DiffGRUCell(A, F_in, H))
        for _ in range(n_enc - 1):
            self.enc.append(DiffGRUCell(A, H, H))
        self.dec = DiffGRUCell(A, H, H)
        self.start = nn.Parameter(torch.zeros(H))
        self.head = nn.Linear(H, q_len)
        self.use_od_graph = use_od_graph
        if use_od_graph:
            self.od_gate_proj = nn.Linear(1, 1)

    def learned_adj(self, temperature=1.0, hard=False):
        emb = self.emb(torch.arange(self.N, device=self.emb.weight.device))
        logits = self.logits_init + self.graph_mlp(emb)
        A_hat = gumbel_softmax_sample(logits, temperature=temperature, hard=hard)
        return symmetrize(A_hat)

    def forward(self, x, stn_idx, return_context=False, temperature=1.0, od_summary=None):
        B, N, P, _ = x.shape
        A_hat = self.learned_adj(temperature=temperature, hard=False)  # (N,N)
        if self.use_od_graph and od_summary is not None:
            gate = torch.sigmoid(self.od_gate_proj(od_summary.unsqueeze(-1)))  # (B,N,1)
            mod = torch.bmm(gate, gate.transpose(1, 2))  # (B,N,N)
            A_hat = A_hat.unsqueeze(0) * mod  # (B,N,N) -- OD modulates the LEARNED graph
        emb = self.emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        x = torch.cat([x, emb], dim=-1)
        x = x.transpose(1, 2)  # (B, P, N, F)

        prev_seq = None
        h_final = None
        for l, cell in enumerate(self.enc):
            h_cur = torch.zeros((B, N, cell.H), device=x.device)
            cur_seq = []
            for t in range(P):
                inp = x[:, t, :, :] if l == 0 else prev_seq[:, t, :, :]
                h_cur = cell(inp, h_cur, A=A_hat)
                cur_seq.append(h_cur)
            prev_seq = torch.stack(cur_seq, dim=1)
            h_final = h_cur

        d = self.dec(self.start.expand(B, N, -1), h_final, A=A_hat)
        if return_context:
            return d
        return self.head(d)  # (B, N, Q)


class _STBlock(nn.Module):
    """One (temporal self-attention, spatial self-attention) pair, each a standard pre-LN
    Transformer encoder layer (MHA + FFN + residual). Unlike GMANBaseline's _TemporalAttn above
    (which, on inspection, contracts over the wrong axis and never actually attends across the P
    lookback steps -- likely a real contributor to GMAN's training failure noted in Section 2.2),
    both attentions here are implemented via nn.MultiheadAttention with an explicit reshape, so the
    temporal block provably attends across P (per node, batched over N) and the spatial block
    provably attends across N (per timestep, batched over P)."""
    def __init__(self, H, n_heads=4, ff_mult=2, dropout=0.1):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(H, n_heads, dropout=dropout, batch_first=True)
        self.s_attn = nn.MultiheadAttention(H, n_heads, dropout=dropout, batch_first=True)
        self.norm_t1 = nn.LayerNorm(H)
        self.norm_t2 = nn.LayerNorm(H)
        self.norm_s1 = nn.LayerNorm(H)
        self.norm_s2 = nn.LayerNorm(H)
        self.ff_t = nn.Sequential(nn.Linear(H, H * ff_mult), nn.ReLU(), nn.Linear(H * ff_mult, H))
        self.ff_s = nn.Sequential(nn.Linear(H, H * ff_mult), nn.ReLU(), nn.Linear(H * ff_mult, H))

    def forward(self, X):  # X: (B, N, P, H)
        B, N, P, H = X.shape
        # temporal self-attention: attend across P, independently per (batch, node)
        Xt = X.reshape(B * N, P, H)
        a, _ = self.t_attn(Xt, Xt, Xt, need_weights=False)
        Xt = self.norm_t1(Xt + a)
        Xt = self.norm_t2(Xt + self.ff_t(Xt))
        X = Xt.reshape(B, N, P, H)
        # spatial self-attention: attend across N, independently per (batch, timestep)
        Xs = X.permute(0, 2, 1, 3).reshape(B * P, N, H)
        a, _ = self.s_attn(Xs, Xs, Xs, need_weights=False)
        Xs = self.norm_s1(Xs + a)
        Xs = self.norm_s2(Xs + self.ff_s(Xs))
        X = Xs.reshape(B, P, N, H).permute(0, 2, 1, 3)
        return X


class STAEformerBaseline(nn.Module):
    """Simplified STAEformer (Liu et al., CIKM 2023): a vanilla Transformer backbone with NO
    graph-structure inductive bias at all (the adjacency A is accepted for constructor-signature
    compatibility with the other baselines but is otherwise unused), relying instead on four
    concatenated per-(node,timestep) embeddings: (1) a linear projection of the raw traffic value,
    (2) a linear projection of the calendar features (hour/day-of-week/holiday), (3) a learned
    per-node spatial identity embedding, and (4) a learned, data-independent per-(timestep,node)
    "adaptive" embedding -- the paper's key claim is that this last, purely-learned positional
    signal is what lets attention alone match specialized GNN architectures without ever seeing an
    explicit road-network adjacency. L stacked temporal+spatial attention blocks alternate
    attending across the P lookback axis and the N sensor axis."""
    def __init__(self, A, Fday, N, H=64, L=2, n_heads=4, q_len=12, p_len=12,
                 d_feat=None, d_tod=None, d_spatial=None, d_adp=None,
                 se_init=None, se_frozen=True):
        super().__init__()
        self.N, self.P = N, p_len
        time_feat_dim = Fday - 1
        d_feat = d_feat or H // 4
        d_tod = d_tod or H // 4
        d_spatial = d_spatial or H // 4
        d_adp = d_adp or (H - d_feat - d_tod - d_spatial)
        assert d_feat + d_tod + d_spatial + d_adp == H, "embedding dims must sum to H"

        self.feat_proj = nn.Linear(1, d_feat)
        self.tod_proj = nn.Linear(time_feat_dim, d_tod)
        self.spatial_emb = nn.Embedding(N, d_spatial)
        _init_node_emb(self.spatial_emb, se_init, se_frozen, N, d_spatial)
        self.adaptive_emb = nn.Parameter(torch.randn(p_len, N, d_adp) * 0.02)

        self.blocks = nn.ModuleList([_STBlock(H, n_heads=n_heads) for _ in range(L)])
        self.head = nn.Linear(H, q_len)

    def forward(self, x, stn_idx, return_context=False):  # x: (B, N, P, Fday)
        B, N, P, _ = x.shape
        val, cal = x[..., :1], x[..., 1:]
        e_feat = self.feat_proj(val)                                           # (B,N,P,d_feat)
        e_tod = self.tod_proj(cal)                                             # (B,N,P,d_tod)
        e_sp = self.spatial_emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)  # (B,N,P,d_spatial)
        e_adp = self.adaptive_emb[:P].permute(1, 0, 2)[None].expand(B, N, P, -1)  # (B,N,P,d_adp)
        X = torch.cat([e_feat, e_tod, e_sp, e_adp], dim=-1)                    # (B,N,P,H)
        for blk in self.blocks:
            X = blk(X)
        out = X[:, :, -1, :]  # (B, N, H) last lookback step, matching the other baselines' pooling
        if return_context:
            return out
        return self.head(out)  # (B, N, Q)


MODELS = {"dcrnn": DCRNNBaseline, "gwnet": GWNETBaseline, "gman": GMANBaseline, "gts": GTSBaseline,
          "dcrnn_s2s": DCRNNSeq2Seq, "staeformer": STAEformerBaseline}


# ---------------------------------------------------------------------------
# OD-injection wrapper: applies the SAME road-network-routed-OD gated-fusion
# idea used in train_traffic_model.py's TrafficModel to any of the three
# baseline architectures above, without touching their internals. Each
# baseline already collapses its P-length lookback into ONE pooled per-node
# context vector (B,N,H) before its own head; this wrapper broadcasts that
# context across the Q decode steps, fuses it PER-STEP with the routed OD
# forecast for that step via a per-node/per-step learned gate (same
# beta = sigmoid(gate([h_ctx, h_od])) design as TrafficModel), then applies a
# per-step linear head -- so the injection is architecture-agnostic and the
# gating stays as granular (per node, per hour) as our own model's.
# ---------------------------------------------------------------------------
class ODInjectionWrapper(nn.Module):
    """v3: fixes a real regression in the first version -- collapsing the base model's
    per-horizon head Linear(H,Q) (Q *distinct* learned output rows, one per forecast hour)
    down to a single SHARED Linear(H,1) applied identically at every step destroyed the
    model's horizon-specific specialization, which turned out to hurt far more than the OD
    signal helped (confirmed: both DCRNN and GWNET regressed by ~2.6x on fold0 with the v1
    wrapper, a uniform architecture-independent degradation -- a wrapper bug, not a real
    "OD is redundant" finding). v2 patched this with a generic learned per-step-INDEX
    embedding; v3 replaces that with the actual decode-window calendar features (td: hour/dow/
    holiday sin-cos, same TIME_FEAT_DIM=7 vector our own TrafficModel decoder already receives
    at every step) -- the baselines never saw ANY future calendar info before, only the
    P-window's past calendar features baked into the pooled context. Real calendar identity of
    each future hour is strictly more informative than a bare step-index."""
    def __init__(self, base_model, H, q_len, time_feat_dim):
        super().__init__()
        self.base = base_model
        self.td_proj = nn.Linear(time_feat_dim, H)
        self.od_proj = nn.Sequential(nn.Linear(1, H), nn.ReLU(), nn.Linear(H, H))
        self.gate_net = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 1))
        self.step_head = nn.Linear(H, 1)
        self.q_len = q_len

    def forward(self, x, stn_idx, od_dec=None, td=None):
        """x: (B,N,P,Fday). od_dec: (B,N,Q,1) routed OD forecast for the decode window.
        td: (B,N,Q,time_feat_dim) real calendar features for the decode window.
        Both None -> run the plain (un-injected) baseline."""
        h_ctx = self.base(x, stn_idx, return_context=True)  # (B,N,H)
        if od_dec is None:
            return self.base.head(h_ctx) if hasattr(self.base, "head") else None
        B, N, H = h_ctx.shape
        h_ctx_exp = h_ctx.unsqueeze(2).expand(-1, -1, self.q_len, -1) + self.td_proj(td)  # (B,N,Q,H)
        h_od = self.od_proj(od_dec)  # (B,N,Q,H)
        beta = torch.sigmoid(self.gate_net(torch.cat([h_ctx_exp, h_od], dim=-1)))  # (B,N,Q,1)
        h_fused = beta * h_ctx_exp + (1 - beta) * h_od
        self.last_mean_beta = beta.mean().item()
        return self.step_head(h_fused).squeeze(-1)  # (B,N,Q)


class MatureInjectionWrapper(nn.Module):
    """Adapts the "knowledge adaptation" gate from Li et al., "Knowledge Adaption for Demand
    Prediction based on Multi-task Memory Neural Network" (MATURE, CIKM 2020) -- a paper that
    transfers knowledge from a data-rich source transit mode (bus) to data-sparse target modes
    (train/light-rail/ferry), which is structurally the same shape as our problem (source=OD,
    dense city-wide signal; target=sparse traffic sensors). Their own ablation found naive
    concatenation of source+target features HURTS vs. the target alone (matching what we saw with
    our simple scalar gate on DCRNN/GWNET) -- what fixed it for them was a richer TWO-VECTOR
    adaptive gate instead of a single scalar: a boost vector (what NEW content to bring in from
    the source) and an eliminate vector (how much of the target's OWN representation to suppress
    before absorbing that new content), each learned per-feature-dimension rather than as one
    scalar blend weight. Eq. (9)-(11) of the paper, adapted from their external-memory-matrix
    setting to our per-step hidden-state setting:
        align score:  g = v^T tanh(W_g [h_ctx ; h_od])           (scalar compatibility)
        beta         = sigmoid(g)                                  (how much to adapt at all)
        eliminate:    e = sigmoid(W_e h_ctx + b_e)                  (per-dim, target's own gate)
        boost:        b = tanh(W_b h_od + b_b)                      (per-dim, source's new content)
        h_fused      = h_ctx * (1 - beta*e) + beta*b
    """
    def __init__(self, base_model, H, q_len, time_feat_dim):
        super().__init__()
        self.base = base_model
        self.td_proj = nn.Linear(time_feat_dim, H)
        self.od_proj = nn.Sequential(nn.Linear(1, H), nn.ReLU(), nn.Linear(H, H))
        self.align_v = nn.Linear(H, 1, bias=False)
        self.align_w = nn.Linear(2 * H, H)
        self.eliminate_proj = nn.Linear(H, H)
        self.boost_proj = nn.Linear(H, H)
        self.step_head = nn.Linear(H, 1)
        self.q_len = q_len

    def forward(self, x, stn_idx, od_dec=None, td=None):
        h_ctx = self.base(x, stn_idx, return_context=True)  # (B,N,H)
        if od_dec is None:
            return self.base.head(h_ctx) if hasattr(self.base, "head") else None
        B, N, H = h_ctx.shape
        h_ctx_exp = h_ctx.unsqueeze(2).expand(-1, -1, self.q_len, -1) + self.td_proj(td)  # (B,N,Q,H)
        h_od = self.od_proj(od_dec)  # (B,N,Q,H)

        g = self.align_v(torch.tanh(self.align_w(torch.cat([h_ctx_exp, h_od], dim=-1))))  # (B,N,Q,1)
        beta = torch.sigmoid(g)
        e = torch.sigmoid(self.eliminate_proj(h_ctx_exp))  # (B,N,Q,H) per-dim eliminate gate
        b = torch.tanh(self.boost_proj(h_od))              # (B,N,Q,H) per-dim boost content
        h_fused = h_ctx_exp * (1 - beta * e) + beta * b
        self.last_mean_beta = beta.mean().item()
        return self.step_head(h_fused).squeeze(-1)  # (B,N,Q)


class CrossAttnInjectionWrapper(nn.Module):
    """Mechanism (v): the most expressive of the five fusion mechanisms tested in this paper.
    Rather than fusing the traffic context with the OD signal at the SAME decode step only (the
    scalar gate of ODInjectionWrapper and the dual boost/eliminate gate of MatureInjectionWrapper
    both only ever look at od_dec[:, :, q, :] when producing step q's fused representation), this
    wrapper lets every decode step's traffic query attend, via standard multi-head cross-attention,
    over the OD signal at ALL Q future steps jointly -- so if, say, hour 3's OD surge is actually
    most informative for hour 5's traffic (a lagged, indirect effect a same-step gate cannot
    represent at all), a cross-attention mechanism can in principle learn to use it. Implemented as
    a single pre-LN Transformer cross-attention block: query=traffic context (+calendar), key=
    value=projected OD sequence, residual + FFN, matching the _STBlock pattern used in
    STAEformerBaseline above."""
    def __init__(self, base_model, H, q_len, time_feat_dim, n_heads=4):
        super().__init__()
        self.base = base_model
        self.td_proj = nn.Linear(time_feat_dim, H)
        self.od_proj = nn.Sequential(nn.Linear(1, H), nn.ReLU(), nn.Linear(H, H))
        self.cross_attn = nn.MultiheadAttention(H, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.ReLU(), nn.Linear(2 * H, H))
        self.step_head = nn.Linear(H, 1)
        self.q_len = q_len

    def forward(self, x, stn_idx, od_dec=None, td=None):
        h_ctx = self.base(x, stn_idx, return_context=True)  # (B,N,H)
        if od_dec is None:
            return self.base.head(h_ctx) if hasattr(self.base, "head") else None
        B, N, H = h_ctx.shape
        h_ctx_exp = h_ctx.unsqueeze(2).expand(-1, -1, self.q_len, -1) + self.td_proj(td)  # (B,N,Q,H)
        h_od = self.od_proj(od_dec)  # (B,N,Q,H)

        q = h_ctx_exp.reshape(B * N, self.q_len, H)
        kv = h_od.reshape(B * N, self.q_len, H)
        attn, attn_w = self.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        self.last_attn_weights = attn_w.detach()  # (B*N, Q, Q) -- which OD step each traffic step used
        # diagonal = how much each output step attends to the TIME-ALIGNED OD step, the same-step
        # "trust" quantity ODInjectionWrapper/MatureInjectionWrapper log as last_mean_beta
        self.last_mean_beta = attn_w.diagonal(dim1=-2, dim2=-1).mean().item()
        h = self.norm1(q + attn)
        h = self.norm2(h + self.ff(h))
        h_fused = h.reshape(B, N, self.q_len, H)
        return self.step_head(h_fused).squeeze(-1)  # (B,N,Q)
