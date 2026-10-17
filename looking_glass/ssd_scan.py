"""B4 (partial): Mamba-2 / State-Space-Duality (SSD) *chunked* form of the
first-order affine linear recurrence

    h_t = decay_t * h_{t-1} + drive_t,     decay_t in (0, 1),  h_{-1} = h0

as an ALTERNATIVE to the O(log T) Hillis-Steele affine prefix scan in
`looking_glass.cfm_model._scan`. Same recurrence, reformulated with chunked
matrix multiplication (the SSD equations):

Let T be split into N = ceil(T/C) chunks of size C (pad the tail with
decay=1, drive=0, i.e. "hold state" — neutral). Write chunk c, local index i,
global t = c*C + i, and

    p_{c,i}  = prod_{j<=i} decay_{c*C+j}                (intra-chunk cum. decay)
    h^loc_{c,i} = sum_{j<=i} (prod_{j<k<=i} decay_{c*C+k}) drive_{c*C+j}
                 (intra-chunk scan, zero chunk-entry state)
    P_c = p_{c,C-1}                                     (chunk total decay)
    B_c = h^loc_{c,C-1}                                 (chunk local end state)

**Inter-chunk state passing (matmul).** The chunk-boundary state H_c (state
entering chunk c, zero-init globally) satisfies the *same* affine recurrence at
chunk granularity, H_{c+1} = P_c H_c + B_c, H_0 = 0, so with the lower-triangular
decay-product matrix

    M[c,k] = prod_{k<m<c} P_m   (k < c;  empty product = 1),   else 0

we get H_c = sum_k M[c,k] B_k = (M @ B)_c — one matmul over chunk pairs.

**Reassembly.** h_t (zero-init) = h^loc_{c,i} + p_{c,i} * H_c, and the full
prefix product D_t = p_{c,i} * prod_{k<c} P_k gives the initial-state term, so

    h_t = h^loc_{c,i} + p_{c,i} * H_c + D_t * h0 .

The intra-chunk part uses the exact associative affine scan (no division, no
underflow blow-up); the inter-chunk part is a single lower-triangular matmul
(the "state passing"), exactly the SSD chunked decomposition.
"""

from __future__ import annotations

import math

import torch


def _affine_scan(decay, drive):
    """Hillis-Steele affine prefix scan over the time axis (dim -2).

    Solves h_t = decay_t * h_{t-1} + drive_t with h_{-1}=0 for arbitrary leading
    dims: decay (..., T, 1), drive (..., T, d). Returns (D, H) with
    D = prod_{m<=t} decay_m and H = h_t (zero-init)."""
    a = decay
    b = drive
    T = a.shape[-2]
    s = 1
    while s < T:
        an = a.clone()
        bn = b.clone()
        an[..., s:, :] = a[..., s:, :] * a[..., :-s, :]
        bn[..., s:, :] = a[..., s:, :] * b[..., :-s, :] + b[..., s:, :]
        a, b = an, bn
        s *= 2
    return a, b


def _chunk_matrix(P):
    """Lower-triangular chunk decay-product matrix M (..., N, N).

    M[..., c, k] = prod_{k<m<c} P_m for k < c (empty product = 1), else 0.
    Built division-free via a reverse cumulative product so that strong decay
    underflows to an exact 0 rather than producing NaN."""
    N = P.shape[-1]
    idx = torch.arange(N, device=P.device)
    lower = idx[:, None] > idx[None, :]  # (N, N): c > k
    one = torch.ones_like(P)
    # Tmat[..., c, k] = P_k if k < c else 1
    Tmat = torch.where(lower, P.unsqueeze(-2), one.unsqueeze(-2))
    # reverse cumprod along k: W[..., c, k] = prod_{k<=j} Tmat[..., c, j]
    W = torch.flip(torch.cumprod(torch.flip(Tmat, dims=[-1]), dim=-1), dims=[-1])
    M = torch.zeros_like(Tmat)
    M[..., :, :-1] = W[..., :, 1:]  # M[..., c, k] = W[..., c, k+1]
    return torch.where(lower, M, torch.zeros_like(M))


def ssd_scan(decay, drive, chunk=64, h0=None):
    """Chunked SSD form of h_t = decay_t * h_{t-1} + drive_t.

    Args:
        decay: (..., T, 1) or (..., T) in (0, 1); broadcasts over the feature dim.
        drive: (..., T, d) any leading batch dims.
        chunk: intra-chunk length C (>= 1).
        h0: optional initial state h_{-1} broadcastable to (..., d); the same
            `+ D*h0` (D = full prefix product of decay) term as the callers of
            `_scan`.

    Returns:
        H: (..., T, d) with H[..., t, :] = h_t.
    """
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    if drive.dim() < 2:
        raise ValueError("drive must have shape (..., T, d)")
    if decay.dim() == drive.dim() - 1:
        decay = decay.unsqueeze(-1)
    if decay.shape[-1] != 1:
        decay = decay.unsqueeze(-1)
    if decay.shape[:-2] != drive.shape[:-2]:
        decay = decay.expand(*drive.shape[:-2], *decay.shape[-2:])

    lead = tuple(drive.shape[:-2])
    T, d = drive.shape[-2], drive.shape[-1]
    N = (T + chunk - 1) // chunk
    Tp = N * chunk

    if Tp != T:
        pad = Tp - T
        decay = torch.nn.functional.pad(decay, (0, 0, 0, pad), value=1.0)
        drive = torch.nn.functional.pad(drive, (0, 0, 0, pad), value=0.0)

    # (..., N, C, ...) then flatten leading dims for the intra-chunk scan.
    dc = decay.reshape(*lead, N, chunk, 1)
    bc = drive.reshape(*lead, N, chunk, d)
    Mrows = (math.prod(lead) if lead else 1) * N
    p, hloc = _affine_scan(dc.reshape(Mrows, chunk, 1), bc.reshape(Mrows, chunk, d))
    p = p.reshape(*lead, N, chunk, 1)  # intra-chunk cumulative decay
    hloc = hloc.reshape(*lead, N, chunk, d)  # zero chunk-entry local state

    P = p[..., -1, :]  # (..., N, 1) chunk total decay
    B = hloc[..., -1, :]  # (..., N, d) chunk local end state

    # Inter-chunk state passing: H_c = sum_k M[c,k] B_k  (one matmul).
    Mm = _chunk_matrix(P[..., 0])  # (..., N, N)
    H = torch.einsum("...ck,...kd->...cd", Mm, B)  # (..., N, d)

    # Chunk-prefix product Q_c = prod_{k<c} P_k (Q_0 = 1).
    Q = torch.cumprod(torch.cat([torch.ones_like(P[..., :1, :]), P[..., :-1, :]], dim=-2), dim=-2)

    h = hloc + p * H.unsqueeze(-2)  # (..., N, C, d), zero-init h
    D = p * Q.unsqueeze(-2)  # (..., N, C, 1), full prefix product
    h = h.reshape(*lead, Tp, d)
    D = D.reshape(*lead, Tp, 1)
    if Tp != T:
        h = h[..., :T, :]
        D = D[..., :T, :]
    if h0 is not None:
        # h0 is the state at t=-1 with shape (..., d); align it with the time
        # axis of D (..., T, 1) so D * h0 -> (..., T, d).
        h = h + D * h0.unsqueeze(-2)
    return h
