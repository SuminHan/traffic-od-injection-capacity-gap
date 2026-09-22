"""
Three additional published architectures -- PDFormer (Jiang et al., AAAI 2023), AGCRN (Bai et
al., NeurIPS 2020), and MTGNN (Wu et al., KDD 2020) -- added in a SEPARATE file from
models_baselines.py so the live-running GTS 30-fold sweep (run_gts_ext30_sweep.py, which imports
train_baseline_model.py -> models_baselines.py/train_gts_od.py) is never touched by this addition.
Reference implementations consulted for the core mechanisms (NOT ported line-by-line, LibCity's
own config/data pipeline is unrelated to ours): libcity_ref/libcity/model/traffic_flow_prediction/
{PDFormer,AGCRN}.py and traffic_speed_prediction/MTGNN.py.

Same interface contract as every class in models_baselines.py: __init__(A, Fday, N, H=64,
q_len=12, se_init=None, se_frozen=True, ...) and forward(x, stn_idx, return_context=False) with
x:(B,N,P,Fday) -> (B,N,q_len), or (B,N,H) context when return_context=True, so each model plugs
into train_baseline_model.py / ODInjectionWrapper exactly like DCRNN/GWNET/GMAN/GTS/STAEformer do
(node-feature injection only; none of the three implement --od_graph, matching how GMAN and
STAEformer are already od_graph-exempt in train_baseline_model.py).

Documented simplifications (each keeps the paper's genuinely distinguishing mechanism, drops
paper-specific machinery that depends on data/config this pipeline doesn't have):
  - PDFormer: keeps the dual geo-masked (short-range) + semantic-masked (long-range) spatial
    self-attention split, which is the paper's headline spatial idea. Drops the offline
    traffic-pattern (k-shape clustering) delay-aware key/query augmentation, which requires a
    precomputed pattern-key bank this pipeline does not build.
  - AGCRN: node-adaptive graph convolution (AVWGCN, Chebyshev order cheb_k) + GRU exactly as the
    paper defines it (no node-identity feature concatenation, since the adaptive per-node conv
    weights already differentiate nodes -- more faithful to the paper than adding an extra
    engineered node-id feature the way DCRNN/GWNET/GTS in this codebase do).
  - MTGNN: directed adaptive graph construction (asymmetric node-embedding product) + mix-hop
    propagation + dilated-inception (multi-kernel [2,3,6,7]) temporal convolution, stacked in the
    same block/skip/end-conv backbone shape as GWNETBaseline. Drops the top-k adjacency
    sparsification (kept dense) for simplicity; receptive-field padding is generously
    over-provisioned rather than precision-tuned to the lookback window (correct, slightly
    wasteful, not incorrect).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_baselines import _init_node_emb, ODInjectionWrapper, MatureInjectionWrapper, \
    CrossAttnInjectionWrapper


# =============================================================================
# PDFormer (Jiang et al., "PDFormer: Propagation Delay-Aware Dynamic Long-Range
# Transformer for Traffic Flow Prediction", AAAI 2023)
# =============================================================================
class _PDFBlock(nn.Module):
    """One spatio-temporal block: temporal self-attention (across P) + a dual-masked spatial
    self-attention (geo: short-range, masked to the road-network adjacency; sem: long-range,
    masked to each node's top-k most similar nodes in a learned embedding space), concatenated
    and projected, then a standard pre-LN FFN residual -- the same _STBlock-style shape used by
    STAEformerBaseline in models_baselines.py, with the single spatial attention there split into
    two differently-masked branches here."""
    def __init__(self, H, geo_mask):
        super().__init__()
        self.Ht = H // 2
        self.Hg = H // 4
        self.Hs = H - self.Ht - self.Hg
        self.t_q = nn.Linear(H, self.Ht); self.t_k = nn.Linear(H, self.Ht); self.t_v = nn.Linear(H, self.Ht)
        self.g_q = nn.Linear(H, self.Hg); self.g_k = nn.Linear(H, self.Hg); self.g_v = nn.Linear(H, self.Hg)
        self.s_q = nn.Linear(H, self.Hs); self.s_k = nn.Linear(H, self.Hs); self.s_v = nn.Linear(H, self.Hs)
        self.proj = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.ReLU(), nn.Linear(2 * H, H))
        self.register_buffer("geo_mask", geo_mask)  # (N,N) bool, True = masked out

    def forward(self, X, sem_mask):  # X: (B,N,P,H), sem_mask: (N,N) bool, True = masked out
        B, N, P, H = X.shape
        Xt = X.reshape(B * N, P, H)
        tq, tk, tv = self.t_q(Xt), self.t_k(Xt), self.t_v(Xt)
        t_attn = torch.softmax((tq @ tk.transpose(-2, -1)) / (self.Ht ** 0.5), dim=-1)
        t_out = (t_attn @ tv).reshape(B, N, P, self.Ht)

        Xs = X.permute(0, 2, 1, 3).reshape(B * P, N, H)
        gq, gk, gv = self.g_q(Xs), self.g_k(Xs), self.g_v(Xs)
        g_scores = (gq @ gk.transpose(-2, -1)) / (self.Hg ** 0.5)
        g_scores = g_scores.masked_fill(self.geo_mask, float("-inf"))
        g_attn = torch.softmax(g_scores, dim=-1)
        g_out = (g_attn @ gv).reshape(B, P, N, self.Hg).permute(0, 2, 1, 3)

        sq, sk, sv = self.s_q(Xs), self.s_k(Xs), self.s_v(Xs)
        s_scores = (sq @ sk.transpose(-2, -1)) / (self.Hs ** 0.5)
        s_scores = s_scores.masked_fill(sem_mask, float("-inf"))
        s_attn = torch.softmax(s_scores, dim=-1)
        s_out = (s_attn @ sv).reshape(B, P, N, self.Hs).permute(0, 2, 1, 3)

        attn_out = self.proj(torch.cat([t_out, g_out, s_out], dim=-1))
        X = self.norm1(X + attn_out)
        X = self.norm2(X + self.ff(X))
        return X


class PDFormerBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, L=2, q_len=12, p_len=12, sem_k=10,
                 se_init=None, se_frozen=True):
        super().__init__()
        self.N, self.P, self.sem_k = N, p_len, sem_k
        time_feat_dim = Fday - 1
        d_feat = H // 4
        d_tod = H // 4
        d_sp = H // 4
        d_adp = H - d_feat - d_tod - d_sp
        self.feat_proj = nn.Linear(1, d_feat)
        self.tod_proj = nn.Linear(time_feat_dim, d_tod)
        self.spatial_emb = nn.Embedding(N, d_sp)
        _init_node_emb(self.spatial_emb, se_init, se_frozen, N, d_sp)
        self.adaptive_emb = nn.Parameter(torch.randn(p_len, N, d_adp) * 0.02)

        geo_mask = ~((A > 0) | torch.eye(N, dtype=torch.bool, device=A.device))  # True = masked out
        self.sem_emb = nn.Embedding(N, 16)
        self.blocks = nn.ModuleList([_PDFBlock(H, geo_mask) for _ in range(L)])
        self.head = nn.Linear(H, q_len)

    def _sem_mask(self):
        e = F.normalize(self.sem_emb.weight, dim=-1)
        sim = e @ e.t()
        k = min(self.sem_k, self.N)
        topk = sim.topk(k, dim=-1).indices
        mask = torch.ones(self.N, self.N, dtype=torch.bool, device=e.device)
        mask.scatter_(1, topk, False)
        return mask

    def forward(self, x, stn_idx, return_context=False):  # x: (B,N,P,Fday)
        B, N, P, _ = x.shape
        val, cal = x[..., :1], x[..., 1:]
        e_feat = self.feat_proj(val)
        e_tod = self.tod_proj(cal)
        e_sp = self.spatial_emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        e_adp = self.adaptive_emb[:P].permute(1, 0, 2)[None].expand(B, N, P, -1)
        X = torch.cat([e_feat, e_tod, e_sp, e_adp], dim=-1)
        sem_mask = self._sem_mask()
        for blk in self.blocks:
            X = blk(X, sem_mask)
        out = X[:, :, -1, :]
        if return_context:
            return out
        return self.head(out)


# =============================================================================
# AGCRN (Bai et al., "Adaptive Graph Convolutional Recurrent Network for
# Traffic Forecasting", NeurIPS 2020)
# =============================================================================
class AVWGCN(nn.Module):
    """Node-Adaptive Parameter Learning + Data-Adaptive Graph Generation: the graph itself
    (softmax(relu(E @ E^T)), a self-adaptive similarity graph over learned node embeddings E) and
    every node's own convolution weights/bias (both generated from E via a shared weight pool) are
    learned jointly, rather than sharing one global weight matrix across all nodes like a standard
    GCN."""
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, dim_in, dim_out) * 0.02)
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, dim_out))

    def forward(self, x, node_embeddings):  # x:(B,N,C_in), node_embeddings:(N,D)
        N = node_embeddings.shape[0]
        supports = F.softmax(F.relu(node_embeddings @ node_embeddings.t()), dim=1)
        support_set = [torch.eye(N, device=x.device, dtype=x.dtype), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(2 * supports @ support_set[-1] - support_set[-2])
        supports_stack = torch.stack(support_set, dim=0)  # (cheb_k,N,N)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool  # (N, C_out)
        x_g = torch.einsum("knm,bmc->bknc", supports_stack, x).permute(0, 2, 1, 3)  # (B,N,cheb_k,C_in)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # (B,N,C_out)


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        input_and_state = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, z * state], dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1 - r) * hc


class AGCRNBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=64, n_enc=2, embed_dim=10, cheb_k=2, q_len=12,
                 se_init=None, se_frozen=True):
        super().__init__()
        self.N = N
        self.node_emb_table = nn.Embedding(N, embed_dim)
        _init_node_emb(self.node_emb_table, se_init, se_frozen, N, embed_dim)
        self.cells = nn.ModuleList()
        self.cells.append(AGCRNCell(Fday, H, cheb_k, embed_dim))
        for _ in range(n_enc - 1):
            self.cells.append(AGCRNCell(H, H, cheb_k, embed_dim))
        self.head = nn.Linear(H, q_len)

    def forward(self, x, stn_idx, return_context=False):  # x: (B,N,P,Fday)
        B, N, P, _ = x.shape
        node_embeddings = self.node_emb_table(stn_idx)  # (N, embed_dim)
        cur = x
        h_final = None
        for cell in self.cells:
            h = torch.zeros(B, N, cell.hidden_dim, device=x.device, dtype=x.dtype)
            seq = []
            for t in range(P):
                h = cell(cur[:, :, t, :], h, node_embeddings)
                seq.append(h)
            cur = torch.stack(seq, dim=2)
            h_final = h
        if return_context:
            return h_final
        return self.head(h_final)


# =============================================================================
# MTGNN (Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting
# with Graph Neural Networks", KDD 2020)
# =============================================================================
class _MTGraphConstructor(nn.Module):
    """Directed adaptive graph from an ASYMMETRIC node-embedding product (e1@e2^T - e2@e1^T),
    the genuine MTGNN differentiator from GWNETBaseline's undirected nodevec1@nodevec2 -- traffic
    influence is directional (upstream affects downstream more than the reverse), which a
    symmetric adjacency cannot represent."""
    def __init__(self, N, dim, alpha=3.0):
        super().__init__()
        self.emb1 = nn.Embedding(N, dim)
        self.emb2 = nn.Embedding(N, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.alpha = alpha

    def forward(self, stn_idx):
        e1 = torch.tanh(self.alpha * self.lin1(self.emb1(stn_idx)))
        e2 = torch.tanh(self.alpha * self.lin2(self.emb2(stn_idx)))
        a = e1 @ e2.t() - e2 @ e1.t()
        return F.relu(torch.tanh(self.alpha * a))  # (N,N), directed, dense (no top-k sparsification)


class _MixProp(nn.Module):
    """Mix-hop propagation: at each hop, retain an alpha-fraction of the ORIGINAL features
    (skip-connection to hop 0) rather than only the previous hop's output, then concatenate all
    hops' features before the output projection -- alleviates the over-smoothing that plain
    stacked graph convolution suffers from at higher hop counts."""
    def __init__(self, c_in, c_out, gdep=2, alpha=0.05):
        super().__init__()
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=(1, 1))
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):  # x: (B,C,N,T), adj: (N,N) row-stochastic w/ self-loop
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum("bcnt,mn->bcmt", h, adj)
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))


class _DilatedInception(nn.Module):
    """Parallel dilated causal convolutions at kernel sizes [2,3,6,7] (the paper's own set),
    truncated to the shortest output length and concatenated channel-wise -- captures multiple
    temporal receptive-field scales in one layer instead of GWNETBaseline's single kernel size."""
    def __init__(self, c_in, c_out, dilation=1):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        assert c_out % len(self.kernel_set) == 0
        co = c_out // len(self.kernel_set)
        self.tconv = nn.ModuleList([nn.Conv2d(c_in, co, (1, k), dilation=(1, dilation)) for k in self.kernel_set])

    def forward(self, x):
        outs = [conv(x) for conv in self.tconv]
        min_len = outs[-1].size(3)
        return torch.cat([o[..., -min_len:] for o in outs], dim=1)


class MTGNNBaseline(nn.Module):
    def __init__(self, A, Fday, N, H=32, blocks=1, layers=2, dilation=2, gdep=2,
                 skip_mult=4, end_mult=8, emb_dim=16, nv_dim=16, q_len=12,
                 se_init=None, se_frozen=True):
        super().__init__()
        self.N = N
        C, Cskip, Cend = H, H * skip_mult, H * end_mult
        self.emb = nn.Embedding(N, emb_dim)
        _init_node_emb(self.emb, se_init, se_frozen, N, emb_dim)
        self.start_conv = nn.Conv2d(Fday + emb_dim, C, kernel_size=(1, 1))
        self.graph_constructor = _MTGraphConstructor(N, nv_dim)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv_fwd = nn.ModuleList()
        self.gconv_bwd = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.receptive_field = 1
        for _ in range(blocks):
            scope = 6  # widest DilatedInception kernel (7) minus 1
            d = 1
            for _ in range(layers):
                self.filter_convs.append(_DilatedInception(C, C, dilation=d))
                self.gate_convs.append(_DilatedInception(C, C, dilation=d))
                self.skip_convs.append(nn.Conv2d(C, Cskip, kernel_size=(1, 1)))
                self.gconv_fwd.append(_MixProp(C, C, gdep=gdep))
                self.gconv_bwd.append(_MixProp(C, C, gdep=gdep))
                self.bn.append(nn.BatchNorm2d(C))
                self.receptive_field += scope
                scope *= dilation
                d *= dilation
        self.end_conv_1 = nn.Conv2d(Cskip, Cend, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(Cend, q_len, kernel_size=(1, 1))
        self.ctx_proj = nn.Linear(Cend, H)

    def forward(self, x, stn_idx, return_context=False):  # x: (B,N,P,Fday)
        B, N, P, _ = x.shape
        emb = self.emb(stn_idx)[None, :, None, :].expand(B, N, P, -1)
        x = torch.cat([x, emb], dim=-1)
        x = x.permute(0, 3, 1, 2)  # (B,F,N,P)
        x = F.pad(x, (1, 0, 0, 0))
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        x = self.start_conv(x)
        adj_raw = self.graph_constructor(stn_idx)
        eye = torch.eye(N, device=x.device, dtype=x.dtype)
        adj_fwd = adj_raw + eye
        adj_fwd = adj_fwd / adj_fwd.sum(1, keepdim=True).clamp(min=1e-6)
        adj_bwd = adj_raw.t() + eye
        adj_bwd = adj_bwd / adj_bwd.sum(1, keepdim=True).clamp(min=1e-6)

        skip = None
        for i in range(len(self.filter_convs)):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate
            s = self.skip_convs[i](x)
            skip = s if skip is None else s + skip[..., -s.size(3):]
            x = self.gconv_fwd[i](x, adj_fwd) + self.gconv_bwd[i](x, adj_bwd)
            x = x + residual[..., -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))  # (B,Cend,N,L)
        if return_context:
            return self.ctx_proj(x[:, :, :, -1].permute(0, 2, 1))
        x = self.end_conv_2(x)  # (B,Q,N,L)
        return x[:, :, :, -1].permute(0, 2, 1)  # (B,N,Q)


MODELS_EXTRA = {"pdformer": PDFormerBaseline, "agcrn": AGCRNBaseline, "mtgnn": MTGNNBaseline}
