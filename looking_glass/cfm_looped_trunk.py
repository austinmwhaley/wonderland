"""DEC-613 (branch `looped-transformer`): the Dory-style looped GPT trunk.

A trunk-alternative for the CFM dynamics axis, grafted behind the
`MultiScaleSSM` contract — the single swap-in point of `CFM`:

    forward(x, h0=None, mask=None, ev=None, ev_prev=None, dt=None, f0=None, g0=None)
        -> (y, h)      x=(B,T,chan), y=(B,T,dim), h=(B,dim)

Structure (the Dory mapping at dev-rig scale — Dory is 8/20/8 blocks at 4B;
this is 1/(1 x L)/1 at N=500. The mapping is preserved, the scale is ours,
and the parameter counts of both trunks are recorded in the DEC-613 outcome):

  * I block   one pre-RMSNorm transformer layer, run once (stack slot 0, even -> full attention)
  * R block   ONE shared layer run `n_loops` times (the weight-tied latent loop — Dory's
              recurrent block); after EVERY loop execution the I-block output is re-added
              to the stream (Dory's original-input re-injection)
  * O block   one pre-RMSNorm transformer layer, run once
  * readout   Linear(hidden -> dim) + a final RMSNorm. The final RMSNorm is the nGPT
              constant-depth guarantee at the public interface: the served y/h scale is
              loop-count invariant even though the internal stream grows O(sqrt(L));
              `assert_readout_health` (pre-train step-0 check) bounds exactly this.

Attention interleave (Dory's SWA/full mix): stack slot k (I=0, the L loop
executions=1..L, O=L+1) uses FULL causal attention for even k, sliding-window
causal attention (window = T//2, derived from the forward's own padded length —
no literal) for odd k. Sequences are right-padded (mask=1 on valid positions);
padding keys are masked out of attention and padding query rows keep their
diagonal self-attention so the forward stays finite on fully-masked rows — the
positions are then discarded by the loss mask, never fed to a head. Nothing is
clamped.

h0 mapping: the SSM seeds its recurrence with h0; a transformer has no
recurrence to seed. h0 (B,dim) enters as a LEARNED prior — Linear(dim -> hidden)
broadcast-added to the token stream before the I block. Cross-window state
chaining (`forward_embedding` / `forward_states`) is therefore conditioning,
not an exact seed: the disclosed mapping that keeps the (B,dim) contract.

Accepted and NOT consumed (disclosed, see DEC-613): `ev`, `ev_prev` (event
identity is already in x through the shared `emb_et`; the SSM's identity gating
has no transformer analogue at this scale), `f0`, `g0` (slow-intent machinery),
and `dt` beyond x's `w_dt` token embedding — the SWA window is position-based,
not time-based.

The loop over `n_loops` (<=4) is over the DEPTH knob, not over time/rows/
batches: every slot is vectorized over (B,T) with one scaled-dot-product call
each, and the interleave/re-injection are mask algebra, not Python iteration
over the sequence.

Determinism: attention runs on the MATH SDPA backend (flash / memory-efficient
disabled) — the only bit-reproducible path, matching the SSM trunk's `_scan`
determinism class so the trunk comparison is run-to-run comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from looking_glass.cfm_model import RMSNorm

# Dory's elastic range (the model card: n_recurrent_loops in 1..4, default 2).
N_LOOPS_RANGE = (1, 4)


class _LoopedBlock(nn.Module):
    """One pre-RMSNorm transformer layer (single-head attention + 4x MLP, the
    minimal-footprint Dory sublayer). The attention KIND (full vs sliding-window)
    is a runtime argument, not weights: the shared R block is a single module
    executed L times under the SWA/full interleave, exactly as Dory's latent
    loop reuses its recurrent layers."""

    #: relative-time bias buckets over log1p(seconds). 32 is the documented cap the
    #: HSTU-style bias uses (finite table, bucketised from data); it is not tuned per stream.
    N_TIME_BUCKETS = 32

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        # HSTU-style relative attention bias over TIME (Zhai et al. 2024, `rab`): attention
        # scores gain a learned per-bucket term for log1p(|t_q - t_k|), so "how long ago"
        # is a first-class signal instead of something the token embedding must infer.
        # Zero-init: the trunk starts exactly at the current behaviour and learns the bias.
        self.time_bias = nn.Parameter(torch.zeros(self.N_TIME_BUCKETS))
        self.norm1 = RMSNorm(hidden)
        self.w_qkv = nn.Linear(hidden, 3 * hidden)
        self.w_o = nn.Linear(hidden, hidden)
        self.norm2 = RMSNorm(hidden)
        self.w1 = nn.Linear(hidden, 4 * hidden)
        self.w2 = nn.Linear(4 * hidden, hidden)

    def _attn_mask(self, mask: torch.Tensor, T: int, swa: bool) -> torch.Tensor:
        """(B,1,T,T) bool mask: causal & key-valid (& window). Padding query
        rows keep their diagonal so fully-masked rows stay finite (the
        position is discarded by the loss mask; no clamping, no NaN)."""
        dev = mask.device
        valid = mask > 0.5  # (B,T)
        i = torch.arange(T, device=dev)
        causal = (i.unsqueeze(1) >= i.unsqueeze(0)).unsqueeze(0)  # (1,T,T): q >= k
        m = causal & valid.unsqueeze(1)  # (B,T,T)
        if swa:
            # sliding window = half of the row's OWN valid length: local context is a
            # function of the sequence, not of the batch's padding (padding-invariant),
            # and no window literal is introduced.
            w = (valid.sum(-1, keepdim=True).clamp(min=1) // 2).unsqueeze(1)  # (B,1,1)
            dist = i.unsqueeze(1) - i.unsqueeze(0)  # (T,T) = q - k
            m = m & (dist.unsqueeze(0) < w)  # (B,T,T)
        m = m | torch.eye(T, dtype=torch.bool, device=dev).unsqueeze(0)  # finite rows
        return m.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        swa: bool,
        tpos: torch.Tensor | None = None,
        tbias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = x.shape[1]
        m = self._attn_mask(mask, T, swa)
        if tbias is not None:
            # pre-built (B,1,T,T) additive bias, computed ONCE per forward by the trunk.
            # Building it per block cost 4x the (B,T,T) work per step; the matrix depends
            # only on tpos and the shared table, so it is identical for every block.
            m = torch.where(m, tbias, torch.full_like(tbias, float("-inf")))
        elif tpos is not None:
            # bucket log1p(|t_q - t_k|) into the learned bias table. The mask is
            # bool; SDPA's math backend accepts an ADDITIVE float mask, so the
            # bias is added there and disallowed pairs get -inf.
            gap = (tpos.unsqueeze(2) - tpos.unsqueeze(1)).abs().clamp(min=0.0).log1p()
            # data-derived upper edge: the stream's own max gap (no literal horizon)
            hi = float(gap.detach().amax().clamp(min=1.0).item())
            idx = torch.clamp(
                (gap / hi * (self.N_TIME_BUCKETS - 1)).long(), 0, self.N_TIME_BUCKETS - 1
            )
            bias = self.time_bias[idx].unsqueeze(1)  # (B,1,T,T)
            m = torch.where(m, bias, torch.full_like(bias, float("-inf")))
        h = self.norm1(x)
        q, k, v = self.w_qkv(h).chunk(3, dim=-1)
        # single head at dev-rig scale (minimal footprint; multi-head is a
        # scale-up lever, not a loop-structure axis) — SDPA wants the head dim.
        # Deterministic attention (Principle 5, v6 monotonic stability): the math
        # backend is the only SDPA path that is bit-reproducible across runs —
        # flash / memory-efficient reduce in a non-deterministic order (torch
        # warns on its backward). The SSM trunk's `_scan` is determinism class,
        # so the trunk comparison stays fair; at dev-rig scale (T<=503, single
        # head) the math backend's O(T^2) scratch is negligible
        # (measured ~0.2-0.7 ms/forward on the campaign shapes).
        with sdpa_kernel(SDPBackend.MATH):
            a = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=m
            ).squeeze(1)
        x = x + self.w_o(a)
        x = x + self.w2(F.silu(self.w1(self.norm2(x))))
        return x


class LoopedGPTTrunk(nn.Module):
    """The looped (latent-recurrent) normalized GPT trunk behind `CFM.ssm`.

    Contract (the `MultiScaleSSM` shape):
        forward(x, h0=None, mask=None, ev=None, ev_prev=None, dt=None, f0=None, g0=None)
            -> (y, h)      x=(B,T,chan), y=(B,T,dim), h=(B,dim)
        h = the stream at the LAST VALID position per row (the per-customer
        state; heads and the boundary consume it exactly as the SSM state).
    """

    def __init__(self, chan, dim, n_experts, n_loops=2, max_pos=0):
        super().__init__()
        n_loops = int(n_loops)
        if not (N_LOOPS_RANGE[0] <= n_loops <= N_LOOPS_RANGE[1]):
            raise SystemExit(
                f"LoopedGPTTrunk: n_loops must be in {N_LOOPS_RANGE}, got {n_loops!r} "
                "(the Dory elastic range)"
            )
        if int(max_pos) < 1:
            raise SystemExit(
                "LoopedGPTTrunk: max_pos must be the protocol seq_len (derived upstream, >0)"
            )
        self.chan = int(chan)
        self.n_experts = int(n_experts)
        self.dim = int(dim)
        self.n_loops = n_loops
        H = self.dim
        self.max_pos = int(max_pos)
        # Learned absolute position table, sized to the protocol seq_len (data-
        # derived upstream and part of the resolved architecture). Zero-init:
        # order starts from the causal mask alone and the table learns from
        # gradients — no hand-specified positional encoding.
        self.pos = nn.Parameter(torch.zeros(self.max_pos, H))
        self.w_in = nn.Linear(self.chan, H)
        self.w_prior = nn.Linear(self.dim, H)  # h0 -> learned state prior
        # I / R / O blocks: three distinct layers; R is the SHARED weight-tied
        # loop executed n_loops times (Dory's recurrent block, scaled to ours).
        self.in_block = _LoopedBlock(H)
        self.rec_block = _LoopedBlock(H)
        self.out_block = _LoopedBlock(H)
        self.readout = nn.Linear(H, self.dim)
        # nGPT constant depth at the public interface: the served y/h are
        # RMS-normalized, so the readout scale is loop-count invariant.
        self.final_norm = RMSNorm(self.dim)

    def forward(self, x, h0=None, mask=None, ev=None, ev_prev=None, dt=None, f0=None, g0=None):
        B, T, _ = x.shape
        if T > self.max_pos:
            # fail safe, reject: the position table is sized to the protocol
            # seq_len; a longer forward is a protocol violation, not a value
            # to be clamped through a stale table.
            raise RuntimeError(
                f"LoopedGPTTrunk: forward T={T} exceeds max_pos={self.max_pos} "
                "(protocol seq_len) — the resolved architecture was under-sized"
            )
        dev = x.device
        if mask is None:
            mask = torch.ones(B, T, dtype=x.dtype, device=dev)
        else:
            mask = mask.to(dtype=x.dtype)

        # cumulative wall-clock position, from the dt the CFM already computes
        # (log1p seconds between events). If dt is absent the bias is inert.
        tpos = None
        if dt is not None:
            d = dt.squeeze(-1).to(x.dtype)
            tpos = torch.cumsum(d, dim=1)
            tpos = tpos * mask + (1.0 - mask) * tpos.amax(dim=1, keepdim=True)  # pad rows flat
        tbias = None
        if tpos is not None:
            gap = (tpos.unsqueeze(2) - tpos.unsqueeze(1)).abs().clamp(min=0.0).log1p()
            hi = float(gap.detach().amax().clamp(min=1.0).item())
            nb = self.rec_block.N_TIME_BUCKETS
            idx = torch.clamp((gap / hi * (nb - 1)).long(), 0, nb - 1)
            tbias = self.rec_block.time_bias[idx].unsqueeze(1)  # (B,1,T,T)
        x = self.w_in(x) + self.pos[:T]
        if h0 is not None:
            x = x + self.w_prior(h0.to(x.dtype)).unsqueeze(1)  # learned state prior
        x = x * mask.unsqueeze(-1)  # right-padded rows: zero, finite, discarded

        h_in = self.in_block(x, mask, swa=False, tpos=tpos, tbias=tbias)  # slot 0 (even) -> full
        stream = h_in
        for ell in range(self.n_loops):  # depth-knob loop (<=4), slots 1..L
            stream = self.rec_block(stream, mask, swa=((1 + ell) % 2 == 1), tpos=tpos, tbias=tbias)
            stream = stream + h_in  # Dory: original input re-injected every loop
        stream = self.out_block(
            stream, mask, swa=((1 + self.n_loops) % 2 == 1), tpos=tpos, tbias=tbias
        )  # slot L+1
        y = self.final_norm(self.readout(stream))

        counts = mask.sum(-1, keepdim=True).clamp(min=1).long()  # (B,1)
        last = counts - 1
        h = torch.gather(y, 1, last.unsqueeze(-1).expand(B, 1, y.shape[-1])).squeeze(1)
        return y, h
