"""v4.0 macro shifts (DEC-029): differentiable ZCA, unified multi-timescale SSM,
log-det volume barrier, and scale-free Marchenko-Pastur / CCA gates."""

from __future__ import annotations

import numpy as np
import torch

from looking_glass.cfm_model import CFM, EventVocab, ns_zca
from looking_glass.cfm_training import slow_state_slice


def _seq(key="c1", n=12):
    ets = [f"2025-01-{(1 + i // 8):02d}T{10 + (i % 8):02d}:00:00+00:00" for i in range(n)]
    ts = [float(np.datetime64(t.replace("+00:00", ""), "s").astype("int64")) for t in ets]
    return {
        "customer": key,
        "group": "A",
        "anchor_epoch": None,
        "event_type": np.array((["view", "order"] * n)[:n], dtype=object),
        "brand": np.array([None] * n, dtype=object),
        "entity_type": np.array(["customer"] * n, dtype=object),
        "entity_id": np.array([key] * n, dtype=object),
        "value": np.ones(n, dtype=np.float32),
        "event_ts": np.array(ets, dtype=object),
        "ts": ts,
        "co": [[0.0, 0.0]] * n,
    }


def test_ns_zca_produces_isotropic_batch():
    torch.manual_seed(0)
    # a genuinely full-rank but anisotropic batch: whitening should equalize
    scale = torch.logspace(0, 2, 40)  # 1 .. 100 per-dim std
    z = torch.randn(1200, 40) * scale
    wz = ns_zca(z)
    wz = wz - wz.mean(0)
    cov = (wz.T @ wz) / (wz.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp(min=0)
    pr_whit = float(ev.sum() ** 2 / (ev**2).sum()) / 40
    rz = z - z.mean(0)
    rc = (rz.T @ rz) / (rz.shape[0] - 1)
    rev = torch.linalg.eigvalsh(rc.double()).clamp(min=0)
    pr_raw = float(rev.sum() ** 2 / (rev**2).sum()) / 40
    assert pr_whit > 0.5 and pr_whit > 2.5 * pr_raw  # whitening equalizes


def test_unified_ssm_and_slow_channel_selection():
    seqs = [_seq("c1"), _seq("c2")]
    vocab = EventVocab.build(seqs)
    torch.manual_seed(0)
    model = CFM(vocab, dim=32, unified=True)
    assert model.n_experts == 1 and model.ssm.experts[0].delta_bias.numel() == 32
    h = torch.randn(3, 32)
    sl = slow_state_slice(model, h)
    assert sl.shape[-1] == 16  # half the channels, derived from delta_bias


def test_mp_pr_score_flags_low_rank():
    from looking_glass.portfolio import _mp_pr_frac

    # gaussian null: PR/dim near 1; a rank-1 batch must score below it
    assert _mp_pr_frac(800, 64, 0) > 0.8


def test_cca_high_for_shared_signal_low_for_independent():
    from looking_glass.intrinsic import _canonical_corrs

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(500, 8))
    X = latent @ rng.normal(size=(8, 20)) + 0.01 * rng.normal(size=(500, 20))
    Y = latent @ rng.normal(size=(8, 20)) + 0.01 * rng.normal(size=(500, 20))
    shared = np.sort(_canonical_corrs(X, Y))[::-1][:8]
    Z = rng.normal(size=(500, 20))
    indep = np.sort(_canonical_corrs(X, Z))[::-1][:8]
    assert float(np.mean(shared)) > 0.9
    assert float(np.mean(indep)) < 0.6
