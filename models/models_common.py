"""Shared building blocks for the four graph spatiotemporal architectures.

Conventions
-----------
* Batch dimension B = number of target days in the micro-batch.
* Node dimension N = number of stations (174 for this experiment).
* Time dimension P = lookback (14) for the input panel; Q = 1 output step.
* Model input  X : (B, N, P, Fday)  per-node features for each lookback day.
* Model output : (B, N, 1)          predicted next-day (z-scored) ridership.

All graph convolutions / attentions mix information *across nodes* (dim N), so
models consume the full (B, N, ...) panel per target day. The GRU is sequential
over P but vectorized over (B, N). Everything runs in float32 on CUDA.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalize(A: torch.Tensor) -> torch.Tensor:
    """Row-normalize A with the DCRNN degree convention: d_i = max(sum_j A_ij, 1),
    with a self-loop added (A + I). Returns D^{-1}(A+I), row-stochastic.
    A may be (N,N) shared, or (B,N,N) batch-specific (e.g. OD-modulated per window)."""
    N = A.shape[-1]
    eye = torch.eye(N, device=A.device, dtype=A.dtype)
    A = A + eye
    d = A.sum(dim=-1).clamp(min=1.0)
    return A / d.unsqueeze(-1)


def diffusion_supports(A: torch.Tensor):
    """Return the 5 diffusion support matrices for a (possibly learned/batched) adjacency A:
    [I, Nf, Nb, Nf^2, Nb^2] where Nf = D_f^{-1}(A_f), Nb = D_b^{-1}(A_b) and A_f = A + I,
    A_b = A^T + I. (Forward + backward random walks, K=2.) A: (N,N) or (B,N,N)."""
    A = A.to(torch.float32)
    N = A.shape[-1]
    I = torch.eye(N, device=A.device, dtype=torch.float32)
    At = A.transpose(-2, -1)
    Nf = row_normalize(A)          # forward
    Nb = row_normalize(At)         # backward
    Nf2 = Nf @ Nf
    Nb2 = Nb @ Nb
    return [I, Nf, Nb, Nf2, Nb2]


def apply_diffusion_supports(Z: torch.Tensor, supports) -> torch.Tensor:
    """Apply all 5 diffusion supports to a node panel Z (B, N, C).
    Each support S may be (N,N) shared across the batch, or (B,N,N) batch-specific.

    Returns (B, N, 5*C) with the supports concatenated along the feature axis.
    """
    B, N, C = Z.shape
    outs = []
    for S in supports:
        if S.dim() == 2:
            # out_bnc = sum_j S[j, c_idx] ... we want (A @ Z) over the NODE axis:
            # node i' = sum_j S[i', j] Z[j, :]
            outs.append(torch.einsum("ij,bjc->bic", S, Z))
        else:
            outs.append(torch.einsum("bij,bjc->bic", S, Z))
    return torch.cat(outs, dim=-1)


class DiffConv(nn.Module):
    """Diffusion convolution: stack the 5 supports of A, apply Z through them,
    then a linear projection over the 5*C stacked features."""

    def __init__(self, A, in_ch, out_ch):
        super().__init__()
        self.N = A.shape[0]
        # A may be fixed (DCRNN) or a parameter source; we store a callable/buffer.
        self.register_buffer("_A", A.to(torch.float32).detach().clone())
        self.fc = nn.Linear(5 * in_ch, out_ch)

    def _get_A(self):
        return self._A

    def forward(self, Z):  # (B, N, C)
        supports = diffusion_supports(self._get_A())
        stacked = apply_diffusion_supports(Z, supports)
        return self.fc(stacked)


class DiffGRUCell(nn.Module):
    """One GRU step whose gates are diffusion convolutions (DCRNN-style).

    x : (B, N, F)  input at this step
    h : (B, N, H)  previous hidden state
    """

    def __init__(self, A, F, H):
        super().__init__()
        self.H = H
        self.input_proj = nn.Linear(F, H)
        self.fc_gate = nn.Linear(5 * (2 * H), 2 * H)   # from [h, x_proj]
        self.fc_cand = nn.Linear(5 * (2 * H), H)
        self.N = A.shape[0]
        self._A = A

    def _supports(self, A=None):
        return diffusion_supports(A if A is not None else self._A)

    def forward(self, x, h, A=None):
        """A: optional (N,N) adjacency override (used by GTS / AGCRN with a
        self-learned graph); defaults to the fixed A from the constructor."""
        xh = self.input_proj(x)                       # (B,N,H)
        combined = torch.cat([h, xh], dim=-1)         # (B,N,2H)
        sup = self._supports(A)
        gate_in = apply_diffusion_supports(combined, sup)   # (B,N,5*2H)
        gates = torch.sigmoid(self.fc_gate(gate_in))        # (B,N,2H)
        r, u = gates[:, :, : self.H], gates[:, :, self.H:]
        c_in = apply_diffusion_supports(combined, sup)
        c = torch.tanh(self.fc_cand(c_in))             # (B,N,H)
        h_new = (1 - u) * c + u * h
        return h_new


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1.0,
                          n_samples: int = 1, hard: bool = True):
    """Gumbel-softmax over the last axis. `hard` uses the straight-through
    estimator (one-hot forward, soft backward) — the discrete sampling of the
    GTS / AGCRN self-learned graphs."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = logits + gumbels
    y_soft = F.softmax(y / temperature, dim=-1)
    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.t())


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
