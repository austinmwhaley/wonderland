"""CFM encoder: event vocabulary and the selective multi-scale SSM network."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from looking_glass.cfm_config import _f, _to_epoch


# ---------------------------------------------------------------------------
# vocab + tokenizer
# ---------------------------------------------------------------------------
class EventVocab:
    def __init__(self, et, brand, ent, eid=None, attrs=None):
        self.et = {v: i for i, v in enumerate(et)}
        self.brand = {v: i for i, v in enumerate(brand)}
        self.ent = {v: i for i, v in enumerate(ent)}
        self.eid = {v: i for i, v in enumerate(eid or [])}
        self.n_et, self.n_brand, self.n_ent, self.n_eid = (
            len(et),
            len(brand),
            len(ent),
            len(self.eid),
        )
        # shared value encoders (§20.2): field -> {value -> idx}
        self.attrs = {f: {v: i for i, v in enumerate(vals)} for f, vals in (attrs or {}).items()}
        self.n_attr = {f: len(v) for f, v in self.attrs.items()}

    @staticmethod
    def build(seqs, max_eid=0, attr_fields=(), max_attr=0):
        from collections import Counter

        et, br, en = set(), set(), set()
        eidc: Counter = Counter()
        attrc: dict = {f: Counter() for f in attr_fields}
        for s in seqs:
            et.update(map(str, s["event_type"]))
            br.update(str(b) if b is not None else "none" for b in s["brand"])
            en.update(str(e) if e is not None else "none" for e in s["entity_type"])
            _eids = s.get("entity_id")
            if _eids is not None:
                for e in _eids:
                    eidc[str(e) if e is not None else "none"] += 1
            for f, vals in (s.get("attrs") or {}).items():
                for v in vals:
                    attrc[f][str(v) if v is not None else "none"] += 1
        # high-cardinality product/entity identity, capped to the most FREQUENT
        # `max_eid` ids (data-derived); the rest fall to the OOV index n_eid.
        eid = [v for v, _ in eidc.most_common(max_eid)] if max_eid else sorted(eidc)
        attrs = {
            f: ([v for v, _ in c.most_common(max_attr)] if max_attr else sorted(c))
            for f, c in attrc.items()
        }
        return EventVocab(sorted(et), sorted(br), sorted(en), eid, attrs)

    def dumps(self):
        return {
            "et": list(self.et),
            "brand": list(self.brand),
            "ent": list(self.ent),
            "eid": list(self.eid),
            "attrs": {f: list(v) for f, v in self.attrs.items()},
        }


# ---------------------------------------------------------------------------
# encoder (selective SSM) + objectives
# ---------------------------------------------------------------------------
def _scan_eager(d, b):
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


# ---------------------------------------------------------------------------
# A2: accelerated scan. `LG_SCAN` = auto (default) | compiled | eager.
#   * auto / compiled: use a module-cached `torch.compile` of the SAME
#     recurrence (`_scan_eager`); on any compile or first-call failure fall
#     back to the eager path (and stay there — safe, never breaks a forward).
#   * eager: disable acceleration entirely (default-equivalent numerics).
# Both paths compute the identical affine prefix scan (fp32, atol 1e-5); the
# initial-state term (`H + D*h0`) is handled by the callers, not here.
# ---------------------------------------------------------------------------
_SCAN_MODE = os.environ.get("LG_SCAN", "auto").strip().lower()
_scan_compiled = None
_scan_disabled = _SCAN_MODE == "eager"


def _scan(d, b):
    """Affine prefix scan dispatcher (accelerated when `LG_SCAN` allows)."""
    global _scan_compiled, _scan_disabled
    if _scan_disabled:
        return _scan_eager(d, b)
    if _scan_compiled is None:
        try:
            _scan_compiled = torch.compile(_scan_eager)
        except Exception:
            _scan_disabled = True
            return _scan_eager(d, b)
    try:
        return _scan_compiled(d, b)
    except Exception:
        _scan_disabled = True
        return _scan_eager(d, b)


def _tod(ts):
    """Periodic (Fourier) timestamp features (spec §1): time-of-day and
    day-of-week as sin/cos pairs. `ts` = epoch seconds (any shape) -> (..., 4).
    2π-normalized, no calendar literals; works on any absolute clock."""
    ts = np.asarray(ts, dtype=np.float64)
    h = (ts % 86400.0) / 86400.0 * (2.0 * np.pi)
    d = ((ts / 86400.0) % 7.0) / 7.0 * (2.0 * np.pi)
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], axis=-1)


class RMSNorm(nn.Module):
    """Per-sample, per-timestep RMS normalization (v6.1, DEC-047).

    A recurrent state has no intrinsic scale: over long horizons the input
    drives accumulate and the readout magnitude drifts (measured here: readout
    std ~50, max ~900), which makes every downstream LINEAR head ill-conditioned
    and untrainable (softmax saturates; a head on the same state is 4x worse than
    a probe). This normalizes each token's readout vector to unit RMS — a
    structural, per-sample operation (no batch statistics, so it cannot fake
    rank) — so heads ALWAYS see well-conditioned inputs regardless of sequence
    length or training dynamics. It is the architectural invariant that makes
    the CVF's "variance-preserving recurrence" hold at the readout."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def assert_readout_health(y, kappa=2.0):
    """Tensor-health guard (v6.1, DEC-047): fail loudly if a readout feeds the
    heads unnormalized. `y` is the SSM readout (B,T,D). With RMSNorm applied the
    per-token RMS is ~1; `kappa` bounds the tolerated deviation (dimensionless,
    no unit-dependent constant). Returns a stats dict; raises on violation."""
    with torch.no_grad():
        rms = y.float().pow(2).mean(-1).sqrt()  # (B,T) per-token RMS
        r, mx = float(rms.mean()), float(y.float().abs().max())
        dim = int(y.shape[-1])
        bound = kappa * float(np.sqrt(dim))
        stats = {"readout_rms": round(r, 4), "readout_absmax": round(mx, 4), "dim": dim}
        if not (1.0 / kappa <= r <= kappa):
            raise RuntimeError(
                f"readout health: per-token RMS {r:.3f} outside [{1 / kappa:.3f},{kappa}]"
                f" — readout is not normalized (RMSNorm missing/inactive): {stats}"
            )
        if mx > bound:
            raise RuntimeError(
                f"readout health: |y|max {mx:.3f} exceeds {bound:.1f}=kappa*sqrt(dim)"
                f" — unbounded feature growth: {stats}"
            )
    return stats


def model_health(model, tokens, mask=None, kappa=2.0, t=None):
    """Fail-fast Step-0 validation harness (spec §2B / DEC-049). Runs BEFORE
    training and checks, programmatically and per-sample:
      * magnitude + variance  -> the readout is RMS-normalized (assert_readout_health)
      * spectral bound        -> every transition decay in (0,1) (bounded operator;
                                 no orthogonal/unitary recurrence, which cannot forget)
      * rank health           -> the state batch is not rank-1 collapsed
      * determinism           -> the same input yields the same output (twice)
    Returns a stats dict; raises on any violation. Dimensionless bounds only."""
    with torch.no_grad():
        if t is not None:
            y, _h = model.encode(t)
        else:
            y, _h = model.ssm(tokens, mask=mask)
        stats = assert_readout_health(y, kappa=kappa)
        # spectral bound: contractive transition (decay = exp(-softplus(delta_bias)))
        dec_max = 0.0
        for ex in getattr(model.ssm, "experts", []):
            db = ex.delta_bias
            dec_max = max(dec_max, float(torch.exp(-torch.nn.functional.softplus(db)).max()))
        stats["decay_max"] = round(dec_max, 5)
        if not dec_max < 1.0:
            raise RuntimeError(
                f"model health: transition decay max {dec_max:.4f} >= 1"
                f" — operator is not contractive (state cannot forget): {stats}"
            )
        # rank health: effective rank (participation ratio) of the state batch.
        # Guard only against rank-1 collapse here; the graded capacity check is
        # the per-stream geometry gate (a raw MP-null floor would false-fail a
        # genuinely low-rank stream — DEC-034).
        z = y.reshape(-1, y.shape[-1]).float()
        z = z - z.mean(0, keepdim=True)
        cov = (z.t() @ z) / max(z.shape[0] - 1, 1)
        ev = torch.linalg.eigvalsh(cov.double()).clamp(min=0.0)
        pr = float((ev.sum() ** 2) / (ev**2).sum().clamp(min=1e-24))
        stats["eff_rank"] = round(pr, 3)
        if pr <= 1.0:
            raise RuntimeError(
                f"model health: effective rank {pr:.3f} <= 1 — rank-1 collapse: {stats}"
            )
        # determinism: identical input -> identical output
        y2, _h2 = model.encode(t) if t is not None else model.ssm(tokens, mask=mask)
        stats["deterministic"] = bool(torch.equal(y, y2))
        if not stats["deterministic"]:
            raise RuntimeError(f"model health: non-deterministic forward: {stats}")
    return stats


class SelectiveSSM(nn.Module):
    def __init__(
        self,
        dim,
        delta_bias=0.0,
        delta_spectrum=False,
        spectrum=(-8.0, 3.0),
        slow_smooth=False,
        band_isolate=False,
        bilinear=False,
        readout_norm=True,
        input_norm=True,
        readout_skip=False,
        event_residual=False,
        readout_scale=False,
        stats_pathway=False,
        continuous_time=False,
    ):
        super().__init__()
        self.continuous_time = bool(continuous_time)
        # D (v8, "carry everything"): a long-memory STATISTICS pathway — an EMA
        # accumulator of the input drive whose decay is learned (init long) so it
        # integrates the customer's running aggregate (RFM-like), added to the
        # readout. Gives the state a mechanism to HOLD long-run statistics the
        # contractive recurrence forgets.
        self.stats_proj = None
        self.stats_decay_logit = None
        if stats_pathway:
            self.stats_proj = nn.Linear(dim, dim)
            self.stats_decay_logit = nn.Parameter(torch.tensor(4.0))  # sigmoid ~0.982
        # variance-preserving recurrence (v6.1, DEC-047): normalize the token
        # stream so the accumulated drive cannot grow with horizon, and the
        # readout so heads see unit-scale features. Both per-sample.
        self.input_norm = RMSNorm(dim) if input_norm else None
        self.readout_norm = RMSNorm(dim) if readout_norm else None
        # magnitude-preserving readout (v8): RMSNorm erases SCALE, but recent-activity
        # IS scale. Add a learned projection of the pre-norm magnitude as a per-sample
        # offset so heads can read "how active" while per-dim conditioning is kept.
        self.readout_scale = nn.Linear(3, dim) if (readout_norm and readout_scale) else None
        # readout skip (v6.2, DEC-048): the recurrent state is a low-pass
        # aggregate; the per-event transition (e.g. cart->purchase) is a function
        # of the CURRENT token, which survives only weakly/entangled in the state
        # (measured: state->next CE 0.37 vs the 1-gram's 0.25; adding the state to
        # the raw current event even HURTS). A direct token->readout skip (the `D`
        # term of a standard SSM) restores the current event to the readout so a
        # linear head can express the transition table. Identity-init, learnable.
        if readout_skip:
            self.skip = nn.Linear(dim, dim, bias=False)
            with torch.no_grad():
                self.skip.weight.copy_(torch.eye(dim))
        else:
            self.skip = None
        # Event-identity residual (v6.9, DEC-055): a dedicated, clean path for the
        # current and previous EVENT embeddings into the readout, so a linear head
        # can express the Markov transition. The generic token skip above mixes
        # emb(event) with brand/value/dt, so the transition trigger is not
        # linearly isolable (measured: z_t alone CE 0.257 vs 1-gram 0.231; adding
        # raw e_{t-1} -> 0.228). This adds information by construction (no loss
        # hack); the SSM core stays contractive and the readout stays normalized.
        self.ev_residual = bool(event_residual)
        if self.ev_residual:
            self.w_ev = nn.Linear(dim, dim, bias=False)
            self.w_evp = nn.Linear(dim, dim, bias=False)
            # per-channel norm so the event channels stay at FIXED scale in the
            # readout (adding pre-normalization washes them out — the state
            # dominates the vector, measured TMD: z_t still 0.270 vs 1-gram 0.231).
            self.ev_norm = RMSNorm(dim)
        else:
            self.w_ev = self.w_evp = self.ev_norm = None
        self.W_delta = nn.Linear(dim, dim)
        self.W_B = nn.Linear(dim, dim)
        self.W_C = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        # per-channel timescale (log-decay) offset; LEARNED.
        if delta_spectrum:
            self.delta_bias = nn.Parameter(torch.linspace(spectrum[0], spectrum[1], dim))
        else:
            self.delta_bias = nn.Parameter(torch.tensor([float(delta_bias)]))
        # v4.5 (DEC-035): block-diagonal input mixing between the FAST and SLOW
        # timescale bands, so the slow state is driven ONLY by the slow-band
        # input (which CFM routes to content-agnostic trend features). This makes
        # slow-state continuity structural rather than an optimizable loss.
        if band_isolate and self.delta_bias.numel() > 1:
            db = self.delta_bias.detach().cpu().numpy()
            fast = db > float(np.median(db))
            m = np.zeros((dim, dim), dtype=np.float32)
            fi, si = np.where(fast)[0], np.where(~fast)[0]
            m[np.ix_(fi, fi)] = 1.0
            m[np.ix_(si, si)] = 1.0
            self.register_buffer("band_mask", torch.tensor(m))
        # v7 DEC-041: bilinear (multiplicative) state-input interaction — the
        # only measured mechanism that raises TRANSFERABLE cross-sample rank
        # (Linear State Collapsibility fix). Per-sample, rank-changing.
        self.bilinear = bool(bilinear)
        if self.bilinear:
            self.W_nl = nn.Linear(dim, dim)
            nn.init.normal_(self.W_nl.weight, std=0.1)
            nn.init.zeros_(self.W_nl.bias)
            self.beta = nn.Parameter(torch.tensor(0.3))
            self.bilinear_iters = 8  # fixed-point passes (cap); converged earlier on tol
            self.bilinear_tol = 1e-4  # state-change convergence (converge, not count)
            self.bilinear_normalize = True  # Path 1: unit-norm gate (bounded Lipschitz)
        self.smooth = None
        if slow_smooth:
            dbv = self.delta_bias.detach().numpy().astype(float)
            slow = dbv <= np.median(dbv)
            r0 = np.where(slow, 0.95, 0.05)
            self.smooth = IntentFilter(dim, retention_init=r0, use_proj=False)

    def _nl_gate(self, H):
        """Unit-norm tanh gate (v7 DEC-043, Path 1): bounds the bilinear term's
        Lipschitz constant without shrinking W_nl, preserving the rank gain.
        Falls back to a projection (no normalization) via bilinear_normalize."""
        g = torch.tanh(self.W_nl(H))
        if getattr(self, "bilinear_normalize", True):
            g = g / (g.norm(dim=-1, keepdim=True) + 1e-6)
        return g

    def _readout(self, y):
        if self.readout_norm is None:
            return y
        mom = None
        if self.readout_scale is not None:
            rms = y.pow(2).mean(-1, keepdim=True).sqrt()
            mean = y.mean(-1, keepdim=True)
            std = y.std(-1, keepdim=True)
            mom = torch.cat([torch.log1p(rms), mean, std], dim=-1)  # (B,T,3)
        yn = self.readout_norm(y)
        return yn + self.readout_scale(mom) if mom is not None else yn

    def forward(self, x, h0=None, mask=None, ev=None, ev_prev=None, dt=None):
        if self.input_norm is not None:
            x = self.input_norm(x)
        xs = x  # post-input-norm token for the readout skip (current event)
        if self.smooth is not None:
            x = self.smooth(x, mask=mask)
        bm = getattr(self, "band_mask", None)
        if bm is not None:
            wd = F.linear(x, self.W_delta.weight * bm, self.W_delta.bias)
            bx = F.linear(x, self.W_B.weight * bm, self.W_B.bias)
        else:
            wd = self.W_delta(x)
            bx = self.W_B(x)
        delta = F.softplus(wd + self.delta_bias)
        # CONTINUOUS-TIME (v8): scale the per-channel decay by the ELAPSED TIME
        # between events, so a channel's memory is in TIME units (irregular streams).
        # decay = exp(-delta * dt_days); dt is log1p(seconds) -> days.
        if dt is not None and self.continuous_time:
            dt_days = torch.expm1(dt) / 86400.0  # (B,T,1)
            decay = torch.exp(-delta * dt_days)
        else:
            decay = torch.exp(-delta)
        if mask is not None:
            m = mask.unsqueeze(-1)
            decay = decay * m + (1.0 - m)  # pad: hold state
            bx = bx * m
        if self.bilinear:
            # VECTORIZED bilinear recurrence (v7 DEC-041, speed-first): the gate
            # tanh(W_nl h_{t-1}) is non-affine (no exact parallel scan). Solve the
            # fixed point by ITERATED affine scans (each O(log T), fully parallel,
            # no Python loop), iterated to convergence (state change < tol), capped.
            # Converges to the exact non-linear recurrence (measured).
            h0v = None
            if h0 is not None:
                h0v = h0 if h0.dim() >= 2 else h0.unsqueeze(0)
            drive0 = (1.0 - decay) * bx
            # (1) converge the non-linear fixed point under no_grad (cheap memory)
            with torch.no_grad():
                D, H = _scan(decay, drive0)
                if h0v is not None:
                    H = H + D * h0v.unsqueeze(1)
                for _ in range(max(int(self.bilinear_iters), 1)):
                    g = self._nl_gate(H)
                    Dn, Hn = _scan(decay, drive0 + self.beta * (g * bx))
                    if h0v is not None:
                        Hn = Hn + Dn * h0v.unsqueeze(1)
                    conv = float((Hn - H).abs().max()) < self.bilinear_tol
                    H = Hn
                    if conv:
                        break
                Hstar = H
            # (2) ONE differentiable pass with the converged (detached) gate;
            # grad flows through W_nl (tanh) and the drive. Memory = 1 scan graph.
            g = self._nl_gate(Hstar)
            D, Hout = _scan(decay, drive0 + self.beta * (g * bx))
            if h0v is not None:
                Hout = Hout + D * h0v.unsqueeze(1)
            y = self.W_C(Hout)
            if self.skip is not None:
                y = y + self.skip(xs)
            y = self._readout(y)
            if self.stats_proj is not None:
                _dc = torch.sigmoid(self.stats_decay_logit)
                _S, _ = _scan(torch.full_like(xs, float(_dc)), (1.0 - _dc) * xs)
                y = y + self.stats_proj(_S)
            if self.ev_residual:
                if ev is not None:
                    y = y + self.ev_norm(self.w_ev(ev))
                if ev_prev is not None:
                    y = y + self.ev_norm(self.w_evp(ev_prev))
            return y, Hout[:, -1]
        D, H = _scan(decay, (1.0 - decay) * bx)  # h_t (zero-init)
        if h0 is not None:
            h0v = h0 if h0.dim() >= 2 else h0.unsqueeze(0)
            H = H + D * h0v.unsqueeze(1)
        y = self.W_C(H)
        if self.skip is not None:
            y = y + self.skip(xs)
        y = self._readout(y)
        if self.stats_proj is not None:
            _dc = torch.sigmoid(self.stats_decay_logit)
            _S, _ = _scan(torch.full_like(xs, float(_dc)), (1.0 - _dc) * xs)
            y = y + self.stats_proj(_S)
        if self.ev_residual:
            if ev is not None:
                y = y + self.ev_norm(self.w_ev(ev))
            if ev_prev is not None:
                y = y + self.ev_norm(self.w_evp(ev_prev))
        return y, H[:, -1]


def cayley_orthogonal(n, device=None, dtype=None):
    """Cayley-orthogonal frame Q=(I−S)(I+S)⁻¹ (S skew, κ=1) for init (DEC-040)."""
    S = torch.randn(n, n, device=device, dtype=dtype)
    S = S - S.t()
    eye = torch.eye(n, device=device, dtype=dtype)
    return torch.linalg.solve(eye + S, eye - S)


def ns_inv_sqrt(cov, iters=8, eps=1e-2):
    """Approximate Sigma^{-1/2} for a symmetric PSD `cov` via the coupled
    Newton-Schulz iteration, spectral-norm normalized so it converges and `eps`
    capped so the condition number is bounded (near-null directions are not
    amplified). Returns a (D, D) matrix usable as a whitening transform."""
    d = cov.shape[0]
    cd = cov.detach().double()
    with torch.no_grad():  # spectral norm by power iteration
        g = torch.Generator().manual_seed(0)
        v = torch.randn(d, 1, generator=g).to(cd.device, cd.dtype)
        for _ in range(16):
            v = cd @ v
            v = v / v.norm().clamp(min=1e-12)
        scale = max(float((v.T @ cd @ v).item()), 1e-12)
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    A = cov / scale + eps * eye
    Y, Z = A.clone(), eye.clone()
    for _ in range(max(int(iters), 0)):
        T = 3.0 * eye - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
    return Z / torch.sqrt(torch.as_tensor(scale, device=cov.device, dtype=cov.dtype))


def ns_zca(z, iters=8, eps=1e-2):
    """Differentiable batch ZCA: whiten `z` by its own (spectral-norm
    normalized, condition-capped) covariance via Newton-Schulz. Matmuls only —
    no eigh/inverse — so it is differentiable and stable."""
    if z.dim() != 2 or z.shape[0] < 2:
        return z
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (zc.shape[0] - 1)
    return zc @ ns_inv_sqrt(cov, iters=iters, eps=eps)


class IntentFilter(nn.Module):
    """Low-pass intent projection for the slow expert (v3.1.0, DEC-027).

    The slow expert otherwise consumes the raw per-token content, whose
    event-type component flips direction at every event (view -> cart -> view),
    so the slow state's VELOCITY reverses every step (the DEC-024 zigzag). This
    filter gives that expert a temporally smoothed content stream: a learned
    intent projection `W_intent`, then a per-channel EMA whose retention is
    learned and initialized from the expert's own decay — so the slow state is a
    second-order low-pass of the token stream and its velocity is smooth.
    """

    def __init__(self, dim, retention_init, use_proj=True):
        super().__init__()
        self.use_proj = bool(use_proj)
        if self.use_proj:
            self.W_intent = nn.Linear(dim, dim)
        r0 = np.clip(np.asarray(retention_init, dtype=float), 1e-3, 1.0 - 1e-3)
        if r0.ndim == 0:
            r0 = np.full(dim, float(r0))
        self.logit = nn.Parameter(torch.tensor(np.log(r0 / (1.0 - r0)), dtype=torch.float32))

    def forward(self, x, mask=None):
        r = torch.sigmoid(self.logit).view(1, 1, -1)
        e = self.W_intent(x) if self.use_proj else x
        d = r.expand_as(e)
        if mask is not None:
            m = mask.unsqueeze(-1)
            d = d * m + (1.0 - m)  # pad: hold the EMA state
            e = e * m
        _, H = _scan(d, (1.0 - d) * e)
        return H


def signed_log1p(x):
    """Monotone, order-of-magnitude-preserving squash: sign(x)·log1p(|x|). Applied
    ONLY to the counter readout (S1). Unlike RMSNorm it does not erase scale — a
    larger running total stays larger — while keeping the exposed feature bounded."""
    return torch.sign(x) * torch.log1p(x.abs())


class CounterBlock(nn.Module):
    """S1 + S2 — ANCHORED additive counter with content-addressed registers.

    Exact arithmetic inside the net. `R` registers; each event's anchored raw write
    `v_t = g_t ⊙ raw_t` is distributed to registers by a LEARNED content-addressing
    assignment `a_t = softmax(W_a x_t)` (S2); with `R=1` this reduces to S1.

        M_t[r] = M_{t-1}[r] + a_t[r] · v_t            (exact sum; decay ≡ 1)

    `raw_t` = the event's RAW field vector (count 1, one-hot event_type/brand/entity_type/
    attrs, raw value, log|value|, gap, log gap) — actual quantities, NOT a learned
    reconstruction. The gate `g` (bias +4 ⇒ σ≈0.98) starts EXACT and only modulates.
    Readout is signed-log1p of the (flattened) registers — magnitude-preserving, never
    normalized away. S2 adds open-endedness (R registers, learned routing) *without*
    losing S1's anchoring/lawfulness.
    """

    def __init__(self, x_dim, raw_dim, n_registers=1, gate_bias=4.0):
        super().__init__()
        self.R = int(max(1, n_registers))
        self.raw_dim = int(raw_dim)
        self.W_g = nn.Linear(x_dim, raw_dim)
        nn.init.normal_(self.W_g.weight, std=0.01)
        nn.init.zeros_(self.W_g.bias)
        self.W_g.bias.data.fill_(float(gate_bias))  # σ(4)≈0.98 -> starts exact
        self.W_a = nn.Linear(x_dim, self.R) if self.R > 1 else None  # content-addressing

    def forward(self, x, raw, mask=None, m0=None):
        g = torch.sigmoid(self.W_g(x))
        v = g * raw  # (B,T,raw_dim) anchored write
        if self.W_a is None:
            drive = v
        else:
            a = torch.softmax(self.W_a(x), dim=-1)  # (B,T,R) learned routing
            drive = (a.unsqueeze(-1) * v.unsqueeze(-2)).reshape(
                x.shape[0], x.shape[1], self.R * self.raw_dim
            )
        if mask is not None:
            drive = drive * mask.unsqueeze(-1)  # pads: hold
        ones = torch.ones_like(drive[:, :, :1])  # decay ≡ 1 => exact monoid sum
        _D, M = _scan(ones, drive)
        if m0 is not None:
            M = M + m0.unsqueeze(1)  # STREAMING: carry the running total across chunks
        return signed_log1p(M), M


class MultiScaleSSM(nn.Module):
    """M1: a bank of selective SSMs with different decay scales (timescales).
    Each channel summarizes a different horizon; their readouts are concatenated
    into the public state (width = chan * n_experts). K=1 is the original SSM."""

    def __init__(
        self,
        chan,
        n_experts,
        delta_biases,
        slow_intent=False,
        delta_spectrum=False,
        band_isolate=False,
        bilinear=False,
        readout_norm=True,
        input_norm=True,
        readout_skip=False,
        event_residual=False,
        readout_scale=False,
        stats_pathway=False,
        continuous_time=False,
    ):
        super().__init__()
        if delta_spectrum:
            # v4.0: ONE wide selective SSM with a per-channel timescale spectrum.
            self.chan = chan
            self.n_experts = 1
            self.experts = nn.ModuleList(
                [
                    SelectiveSSM(
                        chan,
                        delta_spectrum=True,
                        slow_smooth=True,
                        band_isolate=band_isolate,
                        bilinear=bilinear,
                        readout_norm=readout_norm,
                        input_norm=input_norm,
                        readout_skip=readout_skip,
                        event_residual=event_residual,
                        readout_scale=readout_scale,
                        stats_pathway=stats_pathway,
                        continuous_time=continuous_time,
                    )
                ]
            )
            self.slow_intent = False
            return
        self.chan = chan
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [
                SelectiveSSM(
                    chan,
                    b,
                    readout_norm=readout_norm,
                    input_norm=input_norm,
                    readout_skip=readout_skip,
                    event_residual=event_residual,
                    readout_scale=readout_scale,
                    stats_pathway=stats_pathway,
                    continuous_time=continuous_time,
                )
                for b in delta_biases[:n_experts]
            ]
        )
        # v3.1.0 (DEC-027): the SLOW expert gets a low-pass intent filter so its
        # state VELOCITY is smooth instead of tracking raw event-type flips. Slow
        # = the expert with the longest memory (smallest delta_bias) — DERIVED,
        # never "the last half" (which was in fact the FAST expert).
        self.slow_intent = bool(slow_intent) and n_experts >= 2
        if self.slow_intent:
            self.slow_idx = int(np.argmin(delta_biases[:n_experts]))
            r0 = float(np.exp(-np.logaddexp(0.0, delta_biases[self.slow_idx])))  # residual
            self.intent = IntentFilter(chan, retention_init=r0)

    def forward(self, x, h0=None, mask=None, ev=None, ev_prev=None, dt=None):
        # NOTE: context-gated mixtures (M2 softmax gate, M5 sparse top-k) were
        # both REJECTED by the battery — the gate attenuated channels and
        # collapsed their diversity (unique-coverage fell). Experts stay
        # concatenated (M1) and the multi-task objective (M4) is what helped.
        ys, hs = [], []
        for i, e in enumerate(self.experts):
            xi = x
            if self.slow_intent and i == self.slow_idx:
                xi = self.intent(x, mask=mask)
            hi = None if h0 is None else h0[..., i * self.chan : (i + 1) * self.chan]
            yi, ho = e(xi, h0=hi, mask=mask, ev=ev, ev_prev=ev_prev, dt=dt)
            ys.append(yi)
            hs.append(ho)
        return torch.cat(ys, dim=-1), torch.cat(hs, dim=-1)


class CFM(nn.Module):
    def __init__(
        self,
        vocab: EventVocab,
        dim: int,
        n_experts: int = 1,
        delta_biases=None,
        sf_mode: str = "event_types",
        slow_intent: bool = False,
        unified: bool = False,
        zca: bool = False,
        isometric_boundary: bool = False,
        bilinear: bool = False,
        readout_norm: bool = True,
        input_norm: bool = True,
        readout_skip: bool = False,
        event_residual: bool = False,
        entity_id_embed: bool = False,
        readout_scale: bool = False,
        prod_dim: int = 0,
        stats_pathway: bool = False,
        continuous_time: bool = False,
        accumulator: bool = True,
        counter: bool = False,
        counter_registers: int = 1,
        numeric_fields: tuple = (),
        effect_dim: int = 0,
        mtp_k: int = 0,
    ):
        super().__init__()
        self.vocab = vocab
        self.unified = bool(unified)
        self.zca = bool(zca)
        self.bilinear = bool(bilinear)
        self.isometric_boundary = bool(isometric_boundary)
        self.readout_norm_on = bool(readout_norm)
        self.input_norm_on = bool(input_norm)
        self.readout_skip_on = bool(readout_skip)
        self.event_residual_on = bool(event_residual)
        self.eid_embed_on = bool(entity_id_embed) and vocab.n_eid > 0
        # model-free PRODUCT vectors (co-occurrence; §20.2) for the SF phi — a fixed
        # table aligned to the entity-id vocab, set after construction. 0 cols = off.
        self.register_buffer("prod_vec_table", torch.zeros(vocab.n_eid + 1, int(prod_dim)))
        self.readout_scale_on = bool(readout_scale)
        self.stats_pathway_on = bool(stats_pathway)
        self.continuous_time_on = bool(continuous_time)
        if self.isometric_boundary:
            self.iso_skew = nn.Parameter(torch.zeros(dim, dim))
            # conservation scale: Tr(Sigma_readout) is held to the input band
            # power P_in (v6 Stage 3, R_cons). Derived at fit, frozen at serve.
            self.register_buffer("boundary_scale", torch.tensor(1.0))
        self.n_experts = 1 if self.unified else max(1, int(n_experts))
        self.chan = max(1, dim // self.n_experts)
        dim = self.chan * self.n_experts
        self.dim = dim
        # successor-feature phi shape: agnostic per-event-type counts (default,
        # DEC-009) or the legacy purchase-named 4-dim phi — never both.
        self.sf_mode = sf_mode
        self.sf_dim = 1 + vocab.n_et
        self.emb_et = nn.Embedding(vocab.n_et + 1, self.chan)
        self.emb_brand = nn.Embedding(vocab.n_brand + 1, self.chan)
        self.emb_ent = nn.Embedding(vocab.n_ent + 1, self.chan)
        # high-cardinality product/entity identity (the stream carries it; use it)
        if self.eid_embed_on:
            self.emb_eid = nn.Embedding(vocab.n_eid + 1, self.chan)
        # shared value encoders (§20.2): categorical attributes -> embeddings, which
        # GENERALIZE to unseen entities (an unseen product still has a category/aisle)
        self.attr_fields = tuple(getattr(vocab, "attrs", {}) or {})
        self.numeric_fields = tuple(numeric_fields or ())
        if self.attr_fields:
            self.emb_attr = nn.ModuleDict(
                {f: nn.Embedding(vocab.n_attr[f] + 1, self.chan) for f in self.attr_fields}
            )
        self.w_val = nn.Linear(1, self.chan)
        self.w_dt = nn.Linear(1, self.chan)
        self.w_co = nn.Linear(2, self.chan)  # exogenous covariates
        # v6.4 (DEC-050): periodic (Fourier) timestamp encoding — time-of-day
        # and day-of-week as sin/cos pairs (spec §1). Captures seasonality
        # without calendar literals (2π-normalized, per-sample).
        self.w_tod = nn.Linear(4, self.chan)
        if delta_biases is None:
            delta_biases = [0.0] * self.n_experts
        self.ssm = MultiScaleSSM(
            self.chan,
            self.n_experts,
            delta_biases,
            slow_intent=(slow_intent and not self.unified),
            delta_spectrum=self.unified,
            band_isolate=self.unified,
            bilinear=self.bilinear,
            readout_norm=self.readout_norm_on,
            input_norm=self.input_norm_on,
            readout_skip=self.readout_skip_on,
            event_residual=self.event_residual_on,
            readout_scale=self.readout_scale_on,
            stats_pathway=self.stats_pathway_on,
            continuous_time=self.continuous_time_on,
        )
        self.head_next = nn.Linear(dim, vocab.n_et + 1)  # next event type
        # DEC-049: `query` predicts the next event from a FADED state (little
        # remaining info), so its optimum is the marginal. Sharing head_next with
        # query dragged the head to the constant marginal (measured: head_next
        # mean-softmax == the unigram, skill ~0). A dedicated query head lets
        # `next`/`mask` learn the transition.
        self.head_query = nn.Linear(dim, vocab.n_et + 1)  # query-time next event
        # DEC-057: dedicated masked-reconstruction head. Sharing head_next with
        # `mask` (whose input is the MASKED readout, optimum ~ marginal) dragged
        # the clean-input `next` head to the marginal (measured: model head_next
        # CE 0.287 vs a fresh probe 0.219 on the SAME readout, 1-gram 0.230).
        self.head_mask = nn.Linear(dim, vocab.n_et + 1)
        self.head_ent = nn.Linear(dim, vocab.n_ent + 1)  # next entity type
        self.head_dt = nn.Linear(dim, 1)  # log1p(dt_next)
        self.head_occ = nn.Linear(dim, 1)  # next event within horizon?
        self.head_recency = nn.Linear(dim, 1)  # recent-activity level (velocity)
        self.head_foresight = nn.Linear(dim, dim)  # multi-horizon future-state prediction
        self.head_psr = nn.Linear(dim, 6)  # PSR: 3 horizons x [count, value]
        self.head_recon = nn.Linear(dim, 3)  # reconstruct RFM: [count, value-sum, recency]
        self.head_ledger = nn.Linear(dim, 2)  # running totals [count_t, value_t]
        # B5 multi-token prediction: k heads predicting event type at t+1..t+k from the
        # state at t (denser, longer-horizon SHAPE signal; opt-in).
        self.mtp_k = int(mtp_k)
        self.head_mtp = (
            nn.ModuleList([nn.Linear(dim, vocab.n_et + 1) for _ in range(self.mtp_k)])
            if self.mtp_k
            else None
        )
        # counterfactual OUTCOME head (RL leg, DEC-172): f(state, one-hot(action)) -> reward
        # logit, fitted on the randomized (action, reward) periods. Its only role is to FORCE
        # the state to carry the treatment response (decision-sufficiency). `effect_dim` is
        # the total number of action levels (the one-hot width). MLP (not linear): the state
        # then need only encode the latent response, and the head learns the response SHAPE —
        # a linear head would force the whole per-action shape into the state (measured weak).
        self.head_effect = (
            nn.Sequential(nn.Linear(dim + effect_dim, dim), nn.GELU(), nn.Linear(dim, 1))
            if effect_dim
            else None
        )
        self.effect_dim = int(effect_dim)
        self.head_val = nn.Linear(dim, 1)  # next event value (monetary)
        self.head_sf = nn.Sequential(
            nn.Linear(dim + 1, dim), nn.ReLU(), nn.Linear(dim, self.sf_dim + int(prod_dim))
        )
        # exact multi-horizon window targets (count, value-sum) for `agg`
        self.head_agg = nn.Linear(dim + 1, 2)  # successor features
        self.order_W = nn.Linear(dim, self.chan, bias=False)  # temporal-order scorer
        self.proj = nn.Linear(dim, dim)  # contrastive / public S
        # v8 SUFFICIENCY (reinstate, opt-in default-on): a cumulative accumulator
        # exposed in the consumed embedding — the contractive trunk loses "how much"
        # (freq/monetary); this restores it. Best-measured readout (LTV 0.05->0.21).
        self.accumulator_on = bool(accumulator)
        # S1 ANCHORED COUNTER (inside the net): exact gated sums of RAW event quantities,
        # all fields, magnitude-preserving readout. Co-evolves with the shape (its gate +
        # readout projection train under the same objectives).
        self.counter_on = bool(counter)
        self.counter_registers = int(max(1, counter_registers))
        if self.counter_on:
            raw_dim = 1 + (self.vocab.n_et + 1) + (self.vocab.n_brand + 1) + (self.vocab.n_ent + 1)
            for f in self.attr_fields:
                raw_dim += self.vocab.n_attr.get(f, 0) + 1
            raw_dim += 4  # value, log1p|value|, gap, log1p(gap)
            raw_dim += 2 * len(self.numeric_fields)  # each numeric field: x, x^2
            self.counter_raw_dim = raw_dim
            self.counter = CounterBlock(self.chan, raw_dim, n_registers=self.counter_registers)
            self.w_counter = nn.Linear(self.counter_registers * raw_dim, dim)
        else:
            self.counter_raw_dim = 0
            self.counter = None
            self.w_counter = None

        with torch.no_grad():  # DEC-040: orthogonal readout frame
            self.proj.weight.copy_(
                cayley_orthogonal(dim, self.proj.weight.device, self.proj.weight.dtype)
            )
            if self.proj.bias is not None:
                self.proj.bias.zero_()
        # M3: entity-aware donor — pools step outputs per entity type.
        self.proj_ent = nn.Linear((vocab.n_ent + 1) * dim, dim)
        # JEPA: EMA target encoder + predictor (latent future prediction).
        import copy as _copy

        self.t_emb_et = _copy.deepcopy(self.emb_et)
        self.t_emb_brand = _copy.deepcopy(self.emb_brand)
        self.t_emb_ent = _copy.deepcopy(self.emb_ent)
        if self.eid_embed_on:
            self.t_emb_eid = _copy.deepcopy(self.emb_eid)
        self.t_w_val = _copy.deepcopy(self.w_val)
        self.t_w_dt = _copy.deepcopy(self.w_dt)
        self.t_w_co = _copy.deepcopy(self.w_co)
        self.t_w_tod = _copy.deepcopy(self.w_tod)
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
                    "decorr",
                    "barrier",
                    "occur",
                    "order",
                    "jepa",
                    "sf",
                    "query",
                    "agg",
                    "variance",
                    "rank",
                    "spectrum",
                    "volume",
                    "iso",
                    "trajectory",
                    "commutation",
                    "ortho",
                    "effect",
                    "mtp",
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
        out = (
            self.emb_et(et)
            + self.emb_brand(br)
            + self.emb_ent(en)
            + self.w_val(val)
            + self.w_dt(dt)
            + self.w_co(co)
            + self.w_tod(torch.tensor(_tod(ts), dtype=torch.float32, device=dev))
        )
        if self.eid_embed_on and seq.get("entity_id") is not None:
            eids = seq["entity_id"]
            eid = torch.tensor(
                [
                    self.vocab.eid.get(str(x) if x is not None else "none", self.vocab.n_eid)
                    for x in eids
                ],
                device=dev,
            )
            out = out + self.emb_eid(eid)
        for f in self.attr_fields:
            vals = (seq.get("attrs") or {}).get(f)
            if vals is not None:
                idx = torch.tensor(
                    [
                        self.vocab.attrs[f].get(
                            str(x) if x is not None else "none", self.vocab.n_attr[f]
                        )
                        for x in vals
                    ],
                    device=dev,
                )
                out = out + self.emb_attr[f](idx)
        return out

    def forward(self, seq, h0=None):
        tok = self.tokens(seq).unsqueeze(0)
        if self.event_residual_on:
            dev = self._dev()
            et = torch.tensor(
                [[self.vocab.et.get(str(x), self.vocab.n_et) for x in seq["event_type"]]],
                device=dev,
            )
            t = {"et": et, "mask": torch.ones_like(et, dtype=torch.float32)}
            y, h = self.encode_ev(tok, t, h0=h0)
        else:
            import numpy as _np

            ts = _np.array([_to_epoch(x) for x in seq["event_ts"]], dtype=_np.float64)
            d = _np.zeros_like(ts)
            d[1:] = _np.maximum(ts[1:] - ts[:-1], 0.0)
            dt = (
                torch.tensor(_np.log1p(d), dtype=torch.float32, device=self._dev())
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            y, h = self.ssm(tok, h0=h0, dt=dt)
        return y.squeeze(0), h.squeeze(0)

    def _target_params(self):
        return (
            list(self.t_emb_et.parameters())
            + list(self.t_emb_brand.parameters())
            + list(self.t_emb_ent.parameters())
            + list(self.t_w_val.parameters())
            + list(self.t_w_dt.parameters())
            + list(self.t_w_co.parameters())
            + list(self.t_w_tod.parameters())
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
                + list(self.w_tod.parameters())
                + list(self.ssm.parameters())
                + list(self.proj.parameters()),
            )
        )
        with torch.no_grad():
            for pt, ps in pairs:
                pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)

    def tokens_batch(self, t):
        out = (
            self.emb_et(t["et"])
            + self.emb_brand(t["br"])
            + self.emb_ent(t["en"])
            + self.w_val(t["val"])
            + self.w_dt(t["dt"])
            + self.w_co(t["co"])
            + self.w_tod(t["tod"])
        )
        if self.eid_embed_on and "eid" in t:
            out = out + self.emb_eid(t["eid"])
        for f in self.attr_fields:
            if f in t:
                out = out + self.emb_attr[f](t[f])
        return out

    def target_tokens_batch(self, t):
        out = (
            self.t_emb_et(t["et"])
            + self.t_emb_brand(t["br"])
            + self.t_emb_ent(t["en"])
            + self.t_w_val(t["val"])
            + self.t_w_dt(t["dt"])
            + self.t_w_co(t["co"])
            + self.t_w_tod(t["tod"])
        )
        if self.eid_embed_on and "eid" in t:
            out = out + self.t_emb_eid(t["eid"])
        return out

    def _event_residual_inputs(self, t, emb_et=None):
        """(ev, ev_prev) = the current and previous EVENT embeddings, for the
        event-identity readout residual (DEC-055). None when the residual is off."""
        if not self.event_residual_on:
            return None, None
        emb = emb_et if emb_et is not None else self.emb_et
        et = t["et"]
        ev = emb(et)
        prev = torch.cat([et[:, :1], et[:, :-1]], dim=1)  # e_{t-1} (first = e_0)
        return ev, emb(prev)

    def encode_ev(self, x, t, h0=None, emb_et=None):
        """SSM forward with DEDICATED event-identity channels in the readout.

        Adds a fast bypass (DEC-057): the raw current/previous event one-hots are
        written into the FIRST 2*(n_et+1) readout dims, so a linear head ALWAYS
        has the Markov transition available at full scale (the recurrent state is
        a low-pass aggregate that washes the identity out — measured: any additive
        residual still leaves z_t worse than a 1-gram). Never mixed; no loss hack.
        """
        y, h = self.ssm(x, h0=h0, mask=t["mask"], dt=t["dt"])
        if self.counter_on:
            # S1: add a learned projection of the anchored counter readout to y, so the
            # shape heads / self-supervised losses SEE the totals (co-evolution).
            cre, _M = self.counter_readout(x, t)
            y = y + self.w_counter(cre)
        if self.event_residual_on:
            ne = self.vocab.n_et + 1
            et = t["et"]
            prev = torch.cat([et[:, :1], et[:, :-1]], dim=1)
            y = y.clone()
            oh = torch.nn.functional.one_hot(et.clamp(max=ne - 1), ne).to(y.dtype)
            ohp = torch.nn.functional.one_hot(prev.clamp(max=ne - 1), ne).to(y.dtype)
            y[..., :ne] = oh
            y[..., ne : 2 * ne] = ohp
        return y, h

    def encode(self, t, h0=None):
        """Tokenize + encode a collated batch with the event residual."""
        return self.encode_ev(self.tokens_batch(t), t, h0=h0)

    def set_whitening(self, mean: "torch.Tensor", W: "torch.Tensor") -> None:
        """Frozen whitening at the donor boundary (DEC-022): z = (h-mu) W,
        W = Sigma^{-1/2} computed from held-out states post-train. Linear ->
        lossless for downstream linear heads; guarantees full-rank headroom
        of the CONSUMED representation (the geometry gate grades this)."""
        self.whiten_mean = mean.detach().float().cpu()
        self.whiten_W = W.detach().float().cpu()
        self.whiten_on = True

    def _whiten(self, h: "torch.Tensor") -> "torch.Tensor":
        if not getattr(self, "whiten_on", False):
            return h
        dev = h.device
        m = self.whiten_mean.to(dev)
        W = self.whiten_W.to(dev)
        return (h - m) @ W

    def embed(self, x):
        """Public embedding S from the recurrence STATE (x = h, recency-weighted
        via the decay), or mean-pool if a per-step matrix is passed."""
        v = self.proj(x) if x.dim() == 1 else self.proj(x.mean(0))
        return F.normalize(self._boundary(v), dim=0)

    def iso_matrix(self):
        """Per-sample orthogonal boundary R (Cayley: R=(I−S)(I+S)⁻¹, S skew).
        κ(R)=1, R⁻¹=Rᵀ: an isometry — deterministic per sample (no batch
        coupling), invertible, no conditioning floor (v6 Stage 1, A1)."""
        S = self.iso_skew - self.iso_skew.t()
        eye = torch.eye(self.dim, device=S.device, dtype=S.dtype)
        return torch.linalg.solve(eye + S, eye - S)

    def _boundary(self, v):
        if self.isometric_boundary:
            return (v @ self.iso_matrix().t()) * self.boundary_scale
        return self._whiten(v)

    def donor(self, x):
        """Donor representation for plugins: same projection WITHOUT L2
        normalization, so magnitude (how much / how recent) is preserved.
        Boundary transform (isometry or legacy whitening) at the readout.
        1-D input = single state; 2-D (T, D) = per-step matrix, mean-pooled
        (the donor_seq contract). For (B, D) STATE BATCHES use donor_batch."""
        v = self.proj(x) if x.dim() == 1 else self.proj(x.mean(0))
        return self._boundary(v)

    def set_product_embeddings(self, ids, vec):
        """Attach model-free product vectors (co-occurrence; §20.2) for the SF phi,
        aligned to the entity-id vocab. Returns how many ids matched."""
        import numpy as _np

        d = int(vec.shape[1])
        tab = torch.zeros(self.vocab.n_eid + 1, d)
        hit = 0
        for i, pid in enumerate(ids):
            j = self.vocab.eid.get(str(pid))
            if j is not None:
                tab[j] = torch.tensor(_np.asarray(vec[i], dtype=_np.float32))
                hit += 1
        self.prod_vec_table = tab.to(next(self.parameters()).device)
        return hit

    def accumulator(self, tok, t):
        """Cumulative statistics exposed in the consumed embedding (contractive
        trunk cannot hold running totals). Learned token-drive totals + recency
        windows + raw universal totals (value/time). Universal, no domain literals."""
        mask = t["mask"].unsqueeze(-1)  # (B,T,1)
        ssum = (tok * mask).sum(1)  # (B, chan) TOTAL token drive
        count = mask.sum(1).clamp(min=1.0)  # (B,1)
        feats = [
            ssum,
            ssum / count,
            torch.log1p(ssum.norm(dim=-1, keepdim=True)),
            torch.log1p(count),
            tok[:, -1],
        ]
        hl = float(getattr(self, "half_life_days", 30.0)) or 30.0
        secs = (torch.expm1(t["dt"]).squeeze(-1) * t["mask"]).cumsum(1)
        age = secs[:, -1:] - secs
        for k in (0.3, 1.0, 3.0, 10.0):
            w = 0.5 ** (age / (hl * k * 86400.0)) * t["mask"]
            feats.append((tok * w.unsqueeze(-1)).sum(1))
            feats.append(w.sum(1, keepdim=True))
        m = t["mask"]
        val = torch.log1p(t["val"].abs().squeeze(-1))
        gap = torch.expm1(t["dt"]).squeeze(-1)
        feats += [
            (val * m).sum(1, keepdim=True),
            (gap * m).sum(1, keepdim=True),
            (val * m).mean(1, keepdim=True),
            (gap * m).mean(1, keepdim=True),
            gap[:, -1:],
        ]
        return torch.cat(feats, dim=-1)

    def generic_measure(self, t):
        """Schema-derived, EXHAUSTIVE additive aggregates from the batch — NOT hand-picked.

        Every categorical field -> per-value COUNTS; every numeric field (value, gap) ->
        count/sum/mean/std/last; plus total count. Additive by construction (sums), so it
        exposes the extensive empirical measure in the consumed embedding. Generic across
        schemas (fields come from the vocab), no domain literals. Opt-in (DEC-132)."""
        if not getattr(self, "generic_measure_on", False):
            return None
        m = t["mask"]  # (B,T)
        cnt = m.sum(1, keepdim=True)  # (B,1)
        feats = [torch.log1p(cnt)]
        for key, n in (
            ("et", self.vocab.n_et + 1),
            ("br", self.vocab.n_brand + 1),
            ("en", self.vocab.n_ent + 1),
        ):
            if key in t:
                oh = F.one_hot(t[key].clamp(max=n - 1), n).to(m.dtype)  # (B,T,n)
                feats.append(torch.log1p((oh * m.unsqueeze(-1)).sum(1)))
        # every consumed attribute field -> per-value counts (generic, no domain literals)
        for f in getattr(self, "attr_fields", ()):
            if f in t:
                nf = self.vocab.n_attr.get(f, 0) + 1
                if nf > 1:
                    oh = F.one_hot(t[f].clamp(max=nf - 1), nf).to(m.dtype)
                    feats.append(torch.log1p((oh * m.unsqueeze(-1)).sum(1)))
        gap = torch.expm1(t["dt"]).squeeze(-1)
        last_idx = (m.sum(1).long() - 1).clamp(min=0).unsqueeze(1)
        for x in (torch.log1p(t["val"].abs().squeeze(-1)), gap):
            s = (x * m).sum(1, keepdim=True)
            mu = s / cnt.clamp(min=1)
            var = (((x - mu) ** 2) * m).sum(1, keepdim=True) / cnt.clamp(min=1)
            last = (x * m).gather(1, last_idx)
            feats += [torch.log1p(s.abs()), mu, var.clamp(min=0).sqrt(), last]
        return torch.cat(feats, dim=-1)

    def raw_fields(self, t):
        """Generic per-event RAW vector for the S1 counter (all fields; no hand-picking):
        count(1) + one-hot(event_type/brand/entity_type) + one-hot(attr fields) + raw
        numeric (value, log1p|value|, gap, log1p gap). Pads -> 0 (hold)."""
        dev = t["mask"].device
        B, T = t["mask"].shape
        feats = [torch.ones(B, T, 1, device=dev, dtype=torch.float32)]
        for key, n in (
            ("et", self.vocab.n_et + 1),
            ("br", self.vocab.n_brand + 1),
            ("en", self.vocab.n_ent + 1),
        ):
            if key in t:
                feats.append(F.one_hot(t[key].clamp(max=n - 1), n).to(torch.float32))
            else:
                feats.append(torch.zeros(B, T, n, device=dev, dtype=torch.float32))
        for f in self.attr_fields:
            if f in t:
                nf = self.vocab.n_attr.get(f, 0) + 1
                feats.append(F.one_hot(t[f].clamp(max=nf - 1), nf).to(torch.float32))
        for c in self.numeric_fields:
            if c in t:
                xc = t[c].unsqueeze(-1).to(torch.float32)
                feats += [xc, xc * xc]  # sum + 2nd moment (mean/std derivable)
        v = t["val"].to(torch.float32)
        dt = t["dt"].to(torch.float32)
        feats += [v, torch.log1p(v.abs()), dt, torch.log1p(dt)]
        raw = torch.cat(feats, dim=-1)
        return raw * t["mask"].unsqueeze(-1)

    def counter_readout(self, x, t):
        """S1 counter readout at every position: (signed_log1p(M), M); (None, None) if off."""
        if not self.counter_on:
            return None, None
        raw = self.raw_fields(t)
        return self.counter(x, raw, mask=t.get("mask"))

    def donor_batch(self, h: "torch.Tensor") -> "torch.Tensor":
        """Donor readout for (B, D) state batches — no pooling. The consumed
        representation. v6 Stage 1: boundary is a per-sample orthogonal isometry
        (κ=1, invertible) — no batch coupling, no conditioning literals; the
        legacy global-EMA ZCA is used only when isometric_boundary is off."""
        return self._boundary(self.proj(h))

    def update_zca(self, h, beta=0.1):
        """Update the global boundary ZCA from this batch (v4.1, DEC-030).

        Maintains an EMA of the projected mean and covariance and recomputes the
        frozen whitening W = Sigma^{-1/2} (Newton-Schulz, spectral-norm
        normalized, eps-capped) from the GLOBAL estimate. Per-batch ZCA (v4.0)
        whitened each batch but not the population, so held-out isotropy stayed
        low; a global estimate is what the geometry gate measures."""
        with torch.no_grad():
            z = self.proj(h).detach().float()
            if z.dim() == 1:
                z = z.unsqueeze(0)
            mu = z.mean(0)
            zc = z - mu
            cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
            # entropy-weighted EMA rate (v4.3, DEC-032): scale the update by the
            # local batch's spectral entropy H(B) in [0,1] so covariance whitening
            # is self-stabilizing against bursty (low-entropy) activity windows.
            ev = torch.linalg.eigvalsh(cov.double()).clamp(min=0.0)
            s = float(ev.sum().item())
            if s > 0 and ev.numel() > 1:
                p = (ev / s).clamp(min=1e-12)
                h_frac = float((-(p * p.log()).sum() / np.log(ev.numel())).item())
            else:
                h_frac = 1.0
            beta_eff = beta * (0.5 + max(0.0, min(1.0, h_frac)))
            if not hasattr(self, "_zca_cov"):
                self._zca_mean, self._zca_cov = mu, cov
            else:
                self._zca_mean = (1 - beta_eff) * self._zca_mean + beta_eff * mu
                self._zca_cov = (1 - beta_eff) * self._zca_cov + beta_eff * cov
            W = ns_inv_sqrt(self._zca_cov)
        self.set_whitening(self._zca_mean, W)

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
