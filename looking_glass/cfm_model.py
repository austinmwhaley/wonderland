"""CFM encoder: event vocabulary and the selective multi-scale SSM network."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from looking_glass.cfm_config import SF_PHI, _f, _to_epoch


# ---------------------------------------------------------------------------
# vocab + tokenizer
# ---------------------------------------------------------------------------
class EventVocab:
    def __init__(self, et, brand, ent):
        self.et = {v: i for i, v in enumerate(et)}
        self.brand = {v: i for i, v in enumerate(brand)}
        self.ent = {v: i for i, v in enumerate(ent)}
        self.n_et, self.n_brand, self.n_ent = len(et), len(brand), len(ent)

    @staticmethod
    def build(seqs):
        et, br, en = set(), set(), set()
        for s in seqs:
            et.update(map(str, s["event_type"]))
            br.update(str(b) if b is not None else "none" for b in s["brand"])
            en.update(str(e) if e is not None else "none" for e in s["entity_type"])
        return EventVocab(sorted(et), sorted(br), sorted(en))

    def dumps(self):
        return {"et": list(self.et), "brand": list(self.brand), "ent": list(self.ent)}


# ---------------------------------------------------------------------------
# encoder (selective SSM) + objectives
# ---------------------------------------------------------------------------
def _scan(d, b):
    """Vectorized affine prefix scan (Hillis-Steele, O(log T) Python steps).

    Solves h_t = d_t * h_{t-1} + b_t with h_{-1}=0, for d:(B,T,1), b:(B,T,D).
    Returns (D_t, H_t) where D_t = prod_{m<=t} d_m and H_t = h_t. Fully
    vectorized in T (the operation is associative: f_i∘f_j = (d_i d_j, d_i b_j + b_i))."""
    a = d
    Bb = b
    T = a.shape[1]
    s = 1
    while s < T:
        an = a.clone()
        bn = Bb.clone()
        an[:, s:] = a[:, s:] * a[:, :-s]
        bn[:, s:] = a[:, s:] * Bb[:, :-s] + Bb[:, s:]
        a, Bb = an, bn
        s *= 2
    return a, Bb


class SelectiveSSM(nn.Module):
    def __init__(self, dim, delta_bias=0.0):
        super().__init__()
        self.W_delta = nn.Linear(dim, dim)
        self.W_B = nn.Linear(dim, dim)
        self.W_C = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        # per-channel timescale (log-decay) offset; LEARNED (init equal).
        self.delta_bias = nn.Parameter(torch.tensor([float(delta_bias)]))

    def forward(self, x, h0=None, mask=None):
        delta = F.softplus(self.W_delta(x) + self.delta_bias)
        bx = self.W_B(x)
        decay = torch.exp(-delta)
        if mask is not None:
            m = mask.unsqueeze(-1)
            decay = decay * m + (1.0 - m)  # pad: hold state
            bx = bx * m
        D, H = _scan(decay, (1.0 - decay) * bx)  # h_t (zero-init)
        if h0 is not None:
            h0v = h0 if h0.dim() >= 2 else h0.unsqueeze(0)
            H = H + D * h0v.unsqueeze(1)
        return self.W_C(H), H[:, -1]


class MultiScaleSSM(nn.Module):
    """M1: a bank of selective SSMs with different decay scales (timescales).
    Each channel summarizes a different horizon; their readouts are concatenated
    into the public state (width = chan * n_experts). K=1 is the original SSM."""

    def __init__(self, chan, n_experts, delta_biases):
        super().__init__()
        self.chan = chan
        self.n_experts = n_experts
        self.experts = nn.ModuleList([SelectiveSSM(chan, b) for b in delta_biases[:n_experts]])

    def forward(self, x, h0=None, mask=None):
        # NOTE: context-gated mixtures (M2 softmax gate, M5 sparse top-k) were
        # both REJECTED by the battery — the gate attenuated channels and
        # collapsed their diversity (unique-coverage fell). Experts stay
        # concatenated (M1) and the multi-task objective (M4) is what helped.
        ys, hs = [], []
        for i, e in enumerate(self.experts):
            hi = None if h0 is None else h0[..., i * self.chan : (i + 1) * self.chan]
            yi, ho = e(x, h0=hi, mask=mask)
            ys.append(yi)
            hs.append(ho)
        return torch.cat(ys, dim=-1), torch.cat(hs, dim=-1)


class CFM(nn.Module):
    def __init__(self, vocab: EventVocab, dim: int, n_experts: int = 1, delta_biases=None):
        super().__init__()
        self.vocab = vocab
        self.n_experts = max(1, int(n_experts))
        self.chan = max(1, dim // self.n_experts)
        dim = self.chan * self.n_experts
        self.dim = dim
        self.emb_et = nn.Embedding(vocab.n_et + 1, self.chan)
        self.emb_brand = nn.Embedding(vocab.n_brand + 1, self.chan)
        self.emb_ent = nn.Embedding(vocab.n_ent + 1, self.chan)
        self.w_val = nn.Linear(1, self.chan)
        self.w_dt = nn.Linear(1, self.chan)
        self.w_co = nn.Linear(2, self.chan)  # exogenous covariates
        if delta_biases is None:
            delta_biases = [0.0] * self.n_experts
        self.ssm = MultiScaleSSM(self.chan, self.n_experts, delta_biases)
        self.head_next = nn.Linear(dim, vocab.n_et + 1)  # next event type
        self.head_ent = nn.Linear(dim, vocab.n_ent + 1)  # next entity type
        self.head_dt = nn.Linear(dim, 1)  # log1p(dt_next)
        self.head_occ = nn.Linear(dim, 1)  # next event within horizon?
        self.head_val = nn.Linear(dim, 1)  # next event value (monetary)
        self.head_sf = nn.Sequential(
            nn.Linear(dim + 1, dim), nn.ReLU(), nn.Linear(dim, SF_PHI)
        )  # successor features
        self.order_W = nn.Linear(dim, self.chan, bias=False)  # temporal-order scorer
        self.proj = nn.Linear(dim, dim)  # contrastive / public S
        # M3: entity-aware donor — pools step outputs per entity type.
        self.proj_ent = nn.Linear((vocab.n_ent + 1) * dim, dim)
        # JEPA: EMA target encoder + predictor (latent future prediction).
        import copy as _copy

        self.t_emb_et = _copy.deepcopy(self.emb_et)
        self.t_emb_brand = _copy.deepcopy(self.emb_brand)
        self.t_emb_ent = _copy.deepcopy(self.emb_ent)
        self.t_w_val = _copy.deepcopy(self.w_val)
        self.t_w_dt = _copy.deepcopy(self.w_dt)
        self.t_w_co = _copy.deepcopy(self.w_co)
        self.t_ssm = _copy.deepcopy(self.ssm)
        self.t_proj = _copy.deepcopy(self.proj)
        self.pred = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        for q in self._target_params():
            q.requires_grad_(False)
        # log-variance per task for uncertainty-based weighting.
        self.log_var = nn.ParameterDict(
            {
                k: nn.Parameter(torch.zeros(1))
                for k in (
                    "next",
                    "entity",
                    "dt",
                    "value",
                    "mask",
                    "contrast",
                    "redundancy",
                    "occur",
                    "order",
                    "jepa",
                    "sf",
                )
            }
        )

    def _dev(self):
        return next(self.parameters()).device

    def tokens(self, seq):
        dev = self._dev()
        et = torch.tensor(
            [self.vocab.et.get(str(x), self.vocab.n_et) for x in seq["event_type"]], device=dev
        )
        br = torch.tensor(
            [
                self.vocab.brand.get(str(b) if b is not None else "none", self.vocab.n_brand)
                for b in seq["brand"]
            ],
            device=dev,
        )
        en = torch.tensor(
            [
                self.vocab.ent.get(str(e) if e is not None else "none", self.vocab.n_ent)
                for e in seq["entity_type"]
            ],
            device=dev,
        )
        val = torch.tensor(
            [_f(v) for v in seq["value"]], dtype=torch.float32, device=dev
        ).unsqueeze(1)
        ts = np.array([_to_epoch(x) for x in seq["event_ts"]], dtype=np.float64)
        dt = np.zeros_like(ts)
        dt[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
        dt = torch.tensor(np.log1p(dt), dtype=torch.float32, device=dev).unsqueeze(1)
        co = seq.get("co")
        co = co if co is not None else [[0.0, 0.0]] * len(seq["event_type"])
        co = torch.tensor(co, dtype=torch.float32, device=dev)
        return (
            self.emb_et(et)
            + self.emb_brand(br)
            + self.emb_ent(en)
            + self.w_val(val)
            + self.w_dt(dt)
            + self.w_co(co)
        )

    def forward(self, seq, h0=None):
        y, h = self.ssm(self.tokens(seq).unsqueeze(0), h0=h0)
        return y.squeeze(0), h.squeeze(0)

    def _target_params(self):
        return (
            list(self.t_emb_et.parameters())
            + list(self.t_emb_brand.parameters())
            + list(self.t_emb_ent.parameters())
            + list(self.t_w_val.parameters())
            + list(self.t_w_dt.parameters())
            + list(self.t_w_co.parameters())
            + list(self.t_ssm.parameters())
            + list(self.t_proj.parameters())
        )

    def ema(self, tau):
        pairs = list(
            zip(
                self._target_params(),
                list(self.emb_et.parameters())
                + list(self.emb_brand.parameters())
                + list(self.emb_ent.parameters())
                + list(self.w_val.parameters())
                + list(self.w_dt.parameters())
                + list(self.w_co.parameters())
                + list(self.ssm.parameters())
                + list(self.proj.parameters()),
            )
        )
        with torch.no_grad():
            for pt, ps in pairs:
                pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)

    def tokens_batch(self, t):
        return (
            self.emb_et(t["et"])
            + self.emb_brand(t["br"])
            + self.emb_ent(t["en"])
            + self.w_val(t["val"])
            + self.w_dt(t["dt"])
            + self.w_co(t["co"])
        )

    def target_tokens_batch(self, t):
        return (
            self.t_emb_et(t["et"])
            + self.t_emb_brand(t["br"])
            + self.t_emb_ent(t["en"])
            + self.t_w_val(t["val"])
            + self.t_w_dt(t["dt"])
            + self.t_w_co(t["co"])
        )

    def embed(self, x):
        """Public embedding S from the recurrence STATE (x = h, recency-weighted
        via the decay), or mean-pool if a per-step matrix is passed."""
        v = self.proj(x) if x.dim() == 1 else self.proj(x.mean(0))
        return F.normalize(v, dim=0)

    def donor(self, x):
        """Donor representation for plugins: same projection WITHOUT L2
        normalization, so magnitude (how much / how recent) is preserved."""
        return self.proj(x) if x.dim() == 1 else self.proj(x.mean(0))

    def successor(self, state, gamma, reward_weight=None):
        """Item 3: query the discounted future at ANY horizon (gamma in (0,1)) from
        a state. Returns successor features; value for a reward w is w . sf."""
        with torch.no_grad():
            col = torch.full((state.shape[0], 1), float(gamma), device=state.device)
            sf = self.head_sf(torch.cat([state, col], dim=-1))
        if reward_weight is not None:
            return (sf * torch.as_tensor(reward_weight, device=state.device)).sum(-1)
        return sf

    def donor_seq(self, y, h, seq):
        """M3 multi-entity donor: [timescale state | entity-pooled summary].
        Pools per-step outputs by entity type (customer/product/session/...)."""
        et = seq.get("entity_type")
        if et is None or len(et) == 0:
            return self.donor(h)
        dev = y.device
        ids = torch.tensor(
            [self.vocab.ent.get(str(e) if e is not None else "none", self.vocab.n_ent) for e in et],
            device=dev,
        )
        P = torch.zeros(self.vocab.n_ent + 1, y.shape[-1], device=dev)
        C = torch.zeros(self.vocab.n_ent + 1, device=dev)
        P.index_add_(0, ids, y)
        C.index_add_(0, ids, torch.ones_like(ids, dtype=y.dtype))
        P = P / C.clamp(min=1).unsqueeze(1)
        return torch.cat([self.proj(h), self.proj_ent(P.reshape(-1))])
