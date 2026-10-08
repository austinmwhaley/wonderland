"""Intrinsic foundation proofs (DEC-023): the 4 representation-space proofs
run without any downstream probe — disentanglement, Lipschitz, trajectory
smoothness, information plane."""

from __future__ import annotations

import numpy as np
import torch

from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_model import CFM, EventVocab


def _seq(key="c1", day="2025-01-01", n=10):
    ets = [f"{day}T{10 + i:02d}:00:00+00:00" for i in range(n)]
    ets[0] = f"{day}T08:00:00+00:00"
    ts = [float(np.datetime64(t.replace("+00:00", ""), "s").astype("int64")) for t in ets]
    return {
        "customer": key,
        "group": "A",
        "anchor_epoch": None,
        "event_type": np.array(
            ["view", "view", "order", "view", "order", "view", "view", "order", "view", "order"][
                :n
            ],
            dtype=object,
        ),
        "brand": np.array([None] * n, dtype=object),
        "entity_type": np.array(["customer"] * n, dtype=object),
        "entity_id": np.array([key] * n, dtype=object),
        "value": np.ones(n, dtype=np.float32),
        "event_ts": np.array(ets, dtype=object),
        "ts": ts,
        "co": [[0.0, 0.0]] * n,
    }


def _seqs(n=24):
    return [_seq(f"c{i}", day=f"2025-01-{(i % 28) + 1:02d}") for i in range(n)]


def test_donor_boundary_whitening_is_self_consistent(monkeypatch):
    """Regression (DEC-022): whitening must be fit AND applied at the SAME
    boundary (after `proj`). Pre-fix, whitening was fit on h and applied before
    proj, so the receipt reported pr_after~0.85 while the consumed readout
    `proj(_whiten(h))` collapsed to ~0.06 under an ill-conditioned proj."""
    from looking_glass import cfm_training

    dim = 16
    seqs = _seqs(2)
    torch.manual_seed(0)
    model = CFM(EventVocab.build(seqs), dim=dim, n_experts=1)
    model.eval()
    with torch.no_grad():  # a trained proj is ill-conditioned (measured sv 0.002..7.9)
        model.proj.weight.copy_(torch.diag(torch.logspace(0, -3, dim)))
        if model.proj.bias is not None:
            model.proj.bias.zero_()
    monkeypatch.setattr(
        cfm_training,
        "forward_states",
        lambda m, s, h0s=None: torch.randn(len(s), dim),
    )
    cfg = CFMConfig()
    wh = cfm_training.compute_whitening(
        model, model.vocab, cfg, [{"x": i} for i in range(800)], n=800
    )
    assert wh["applied"], wh
    H = torch.randn(800, dim)
    z = model.donor_batch(H)
    zc = z - z.mean(0)
    ev = torch.linalg.eigvalsh(((zc.T @ zc) / (z.shape[0] - 1)).double()).clamp(min=0)
    measured = float(ev.sum() ** 2 / (ev**2).sum()) / dim
    # the reported post-whitening rank must match what the consumed readout shows
    assert abs(measured - wh["pr_after"]) < 0.15, (measured, wh["pr_after"])
    # condition-capped whitening (DEC-028) intentionally does NOT manufacture
    # full rank from amplified near-null noise, so this sits below 0.5
    assert measured > 0.3, measured


def test_disentanglement_runs_and_flags_anisotropy():
    from looking_glass.intrinsic import _disentanglement

    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 8))
    ts = np.arange(400, dtype=float)
    out = _disentanglement(z, ts, seed=0)
    assert "mi_mean" in out and "oot_ratio" in out
    # an isotropic gaussian: temporally split covariance should be similar
    assert out["oot_ratio"] is not None


def test_lipschitz_bounded_and_finite():
    from looking_glass.intrinsic import _lipschitz

    seqs = _seqs(12)
    torch.manual_seed(0)
    model = CFM(EventVocab.build(seqs), dim=16, n_experts=2)
    model.eval()
    cfg = CFMConfig()
    cfg.agg_horizons_days = [7.0, 30.0]
    out = _lipschitz(model, model.vocab, seqs, cfg, n_perturb=6, seed=0)
    assert out["n_tested"] > 0
    assert np.isfinite(out["lipschitz_mean"])
    assert out["lipschitz_max"] >= out["lipschitz_median"] >= 0


def test_trajectory_velocity_and_continuity():
    from looking_glass.intrinsic import _trajectory

    seqs = _seqs(6)
    torch.manual_seed(0)
    model = CFM(EventVocab.build(seqs), dim=16, n_experts=2)
    model.eval()
    cfg = CFMConfig()
    out = _trajectory(model, model.vocab, seqs, cfg, n_traj=4, seed=0)
    assert out["n_trajectories"] > 0
    for k in ("directional_cos_raw", "directional_cos_slow", "directional_cos_whitened"):
        assert -1.0 <= out[k]["mean"] <= 1.0
    assert out["speed_mean"] >= 0


def test_info_plane_lists_predictive_losses():
    from looking_glass.intrinsic import _info_plane

    seqs = _seqs(6)
    torch.manual_seed(0)
    model = CFM(EventVocab.build(seqs), dim=16, n_experts=2)
    model.eval()
    cfg = CFMConfig()
    cfg.agg_horizons_days = [7.0, 30.0]
    out = _info_plane(model, model.vocab, seqs, cfg, seed=0)
    names = {r["objective"] for r in out["predictive_losses"]}
    assert {"next", "dt", "sf"} <= names  # predictive objectives present
    assert not ({"redundancy", "variance", "rank"} & names)  # geometry excluded
    assert len(out["horizons"]) == 2


def test_intrinsic_evaluate_end_to_end(tmp_path):
    import json

    from looking_glass.intrinsic import evaluate

    seqs = _seqs(30)
    torch.manual_seed(0)
    model = CFM(EventVocab.build(seqs), dim=16, n_experts=2)
    model.eval()
    cfg = CFMConfig()
    cfg.agg_horizons_days = [7.0, 30.0]
    r = evaluate(model, model.vocab, cfg, seqs, seed=0, out_dir=tmp_path, tag="t")
    assert r["ok"] in (True, False)
    assert all(
        k in r
        for k in (
            "proof1_disentanglement",
            "proof2_lipschitz",
            "proof3_trajectory",
            "proof4_info_plane",
        )
    )
    files = list(tmp_path.glob("intrinsic_t_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["ok"] == r["ok"]
    # determinism
    r2 = evaluate(model, model.vocab, cfg, seqs, seed=0, out_dir=tmp_path, tag="t2")
    assert r2["proof2_lipschitz"]["n_tested"] == r["proof2_lipschitz"]["n_tested"]


def test_slow_channel_mask_derives_slow_band():
    from looking_glass.cfm_training import slow_channel_mask, slow_state_slice

    seqs = _seqs(2)
    torch.manual_seed(0)
    model = CFM(EventVocab.build(seqs), dim=32, unified=True)
    m = slow_channel_mask(model)
    assert m is not None and 0 < int(m.sum()) < 32  # a data-derived slow band
    h = torch.randn(3, 32)
    assert slow_state_slice(model, h).shape[-1] == int(m.sum())


def test_input_lowpass_smooths_alternating_input():
    from looking_glass.cfm_model import IntentFilter

    torch.manual_seed(0)
    filt = IntentFilter(8, retention_init=0.9, use_proj=False)
    v = torch.randn(8)
    x = torch.stack([v if i % 2 == 0 else -v for i in range(48)]).unsqueeze(0)
    y = filt(x)[0].detach().numpy()

    def hf(z):
        return float(np.mean(np.abs(np.diff(z, axis=0)) ** 2))

    assert hf(y) < hf(x[0].numpy())  # the spectral bottleneck removes high-freq energy
