"""Smoke tests for the v2.2.0 objective portfolio (DEC-008/DEC-009):
query, agg, sf_mode, masked-forward redaction + causality, shared val split."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_model import CFM, EventVocab
from looking_glass.cfm_training import (
    _collate,
    _loss,
    _masked_forward,
    _task_losses,
    _val_split,
    agg_window_targets,
)


def _seq(key="c1", day="2025-01-01", n=8):
    ets = [f"{day}T{10 + i:02d}:00:00+00:00" for i in range(n)]
    return {
        "customer": key,
        "group": "B",
        "anchor_epoch": None,
        "event_type": np.array(
            ["view", "view", "order", "view", "order", "view", "view", "order"][:n],
            dtype=object,
        ),
        "brand": np.array([None] * n, dtype=object),
        "entity_type": np.array(["customer"] * n, dtype=object),
        "entity_id": np.array([key] * n, dtype=object),
        "value": np.zeros(n, dtype=np.float32),
        "event_ts": np.array(ets, dtype=object),
        "co": [[0.0, 0.0]] * n,
    }


def _cfg(**kw):
    cfg = CFMConfig(**kw) if kw else CFMConfig()
    cfg.agg_horizons_days = [1.0, 7.0]  # derived in resolve_cfm; set for tests
    return cfg


def test_val_split_is_a_partition_and_deterministic():
    a_tr, a_va = _val_split(100, seed=7)
    b_tr, b_va = _val_split(100, seed=7)
    assert sorted(np.concatenate([a_tr, a_va]).tolist()) == list(range(100))
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)
    assert len(a_va) == 15  # 15% of rows


def test_agg_window_targets_hand_case():
    # one sequence: events at t = 0, 10, 20, 100 (seconds), all real
    secs = torch.tensor([[0.0, 10.0, 20.0, 100.0]])
    vals = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.ones(1, 4)
    h = torch.tensor([[30.0]])
    tgt = agg_window_targets(secs, vals, mask, h)
    # counts in (t, t+30]: pos0 -> {10,20}=2 ; pos1 -> {20}=1 ; pos2 -> {} ; pos3 -> {}
    assert torch.allclose(
        tgt[0, :, 0],
        torch.tensor([np.log1p(2), np.log1p(1), 0.0, 0.0], dtype=torch.float32),
    )
    assert torch.allclose(
        tgt[0, 1, 1], torch.tensor(np.log1p(2.0), dtype=torch.float32)
    )  # vsum over (10,40]: only the event at t=20 (value 2); query itself excluded
    # pads (mask=0) never count
    mask2 = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    tgt2 = agg_window_targets(secs, vals, mask2, h)
    assert tgt2[0, 0, 0].item() == pytest.approx(np.log1p(2))


def test_query_agg_sf_losses_present_and_train():
    seqs = [_seq("c1"), _seq("c2", day="2025-01-02"), _seq("c3", day="2025-01-03")]
    vocab = EventVocab.build(seqs)
    torch.manual_seed(0)
    model = CFM(vocab, dim=16, n_experts=1)  # default sf_mode=event_types
    cfg = _cfg()
    T_ = _task_losses(model, vocab, seqs, cfg)
    for k in ("query", "agg", "sf", "next", "mask", "jepa"):
        assert k in T_, k
        assert torch.isfinite(T_[k]), (k, float(T_[k]))
    loss = _loss(model, vocab, seqs, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head_agg.weight.grad is not None
    assert torch.isfinite(model.head_agg.weight.grad).all()
    assert model.head_next.weight.grad is not None


def test_query_respects_config_toggle():
    seqs = [_seq("c1"), _seq("c2", day="2025-01-02")]
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1)
    cfg = _cfg(objectives=tuple(o for o in CFMConfig.objectives if o != "query"))
    T_ = _task_losses(model, vocab, seqs, cfg)
    assert "query" not in T_


def test_sf_mode_shapes_and_phi():
    seqs = [_seq("c1"), _seq("c2", day="2025-01-02")]
    vocab = EventVocab.build(seqs)
    m_evt = CFM(vocab, dim=16, n_experts=1, sf_mode="event_types")
    m_pur = CFM(vocab, dim=16, n_experts=1, sf_mode="purchase")
    assert m_evt.head_sf[-1].out_features == 1 + vocab.n_et  # value + per-type
    assert m_pur.head_sf[-1].out_features == 4
    assert CFMConfig().sf_mode == "event_types"  # agnostic default (DEC-009)
    for model in (m_evt, m_pur):
        T_ = _task_losses(model, vocab, seqs, _cfg(sf_mode=model.sf_mode))
        assert torch.isfinite(T_["sf"])


def test_masked_forward_causal_and_redacted():
    seqs = [_seq("c1")]
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1)
    cfg = _cfg(mask_frac=1.0)  # config, not a literal: every real position masked
    t = _collate(seqs, vocab, torch.device("cpu"))
    y2, tgt, rand = _masked_forward(model, vocab, t, cfg, torch.device("cpu"))
    assert rand.sum() == (t["mask"] > 0).sum()  # frac wired from config

    # content redaction: permuting the TRUE values must not move the logits
    t2 = dict(t)
    t2["val"] = t["val"] + 5.0
    y2b, _, _ = _masked_forward(model, vocab, t2, cfg, torch.device("cpu"))
    assert torch.allclose(y2, y2b, atol=1e-6)

    # causality: perturbing the LAST event cannot change earlier states
    cfg2 = _cfg(mask_frac=0.5)
    torch.manual_seed(3)
    ya, _, _ = _masked_forward(model, vocab, t, cfg2, torch.device("cpu"))
    t3 = dict(t)
    t3["et"] = t["et"].clone()
    t3["et"][0, -1] = (t["et"][0, -1] + 1) % vocab.n_et  # future token changed
    torch.manual_seed(3)  # identical mask draws
    yb, _, _ = _masked_forward(model, vocab, t3, cfg2, torch.device("cpu"))
    assert torch.allclose(ya[:, :-1], yb[:, :-1], atol=1e-6)  # prefix unaffected
    assert not torch.allclose(ya[:, -1], yb[:, -1], atol=1e-6)  # at the change


# ---------------------------------------------------------------------------
# portfolio grade end-to-end (integration): deterministic, complete, gated
# ---------------------------------------------------------------------------
def test_portfolio_evaluate_end_to_end(tmp_path):
    import json

    from looking_glass.portfolio import evaluate

    seqs = [_seq(f"c{i}", day=f"2025-01-{(i % 28) + 1:02d}") for i in range(40)]
    # give the canary something to (try to) find: two A/B groups, several months
    for i, s in enumerate(seqs):
        s["group"] = "A" if i % 2 else "B"
    torch.manual_seed(0)
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1)
    model.eval()
    cfg = _cfg()
    r1 = evaluate(model, vocab, cfg, seqs, seed=0, folds=3, out_dir=tmp_path, tag="t1")
    assert r1["ok"] in (True, False)  # graded, not crashed
    assert r1["objectives"], "per-objective readouts present"
    names = {o["objective"] for o in r1["objectives"]}
    assert {"next", "dt", "agg", "query", "sf"} <= names  # fielded combo measured
    assert r1["geometry"]["eff_rank"] > 0
    assert any("canary" in row["check"] for row in r1["rows"])
    assert r1["yardstick_pending"] == []  # sf back in the gate via R² (DEC-018)
    sf_row = next(r for r in r1["rows"] if "sf" in r["check"])
    assert "sf" in sf_row["check"]
    files = list(tmp_path.glob("portfolio_t1_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["ok"] == r1["ok"]
    # deterministic: same seed -> same skills
    r2 = evaluate(model, vocab, cfg, seqs, seed=0, folds=3, out_dir=tmp_path, tag="t2")
    a = {o["objective"]: o["skill"] for o in r1["objectives"]}
    b = {o["objective"]: o["skill"] for o in r2["objectives"]}
    for k in a:
        assert abs(a[k] - b[k]) < 1e-4, k


def test_portfolio_canary_catches_leak(tmp_path):
    # a state that ENCODES the A/B arm must fail the canary row
    import numpy as _np

    from looking_glass.portfolio import _canaries

    n = 200
    rng = _np.random.default_rng(0)
    states = rng.normal(size=(n, 8)).astype(_np.float32)
    groups = _np.array(["A"] * n, dtype=object)
    groups[::2] = "B"
    # leak: group signal planted in dim 0
    states[:, 0] += _np.where(groups == "A", 3.0, -3.0)
    seqs = [{"group": g, "event_ts": ["2025-01-05T00:00:00+00:00"]} for g in groups]
    c = _canaries(states, seqs, seed=0)
    assert c["group_auc"] > 0.9  # the planted leak is found


# ---------------------------------------------------------------------------
# DWA balancer (DEC-014): dynamic weights from loss-improvement rates
# ---------------------------------------------------------------------------
def test_dwa_warmup_then_plateau_boost():
    from looking_glass.cfm_training import DWA

    b = DWA(["fast", "plateau"], temp=2.0)
    assert all(w == 1.0 for w in b.weights().values())  # warmup: equal
    b.update({"fast": 10.0, "plateau": 5.0})
    assert all(w == 1.0 for w in b.weights().values())  # still warmup
    b.update({"fast": 2.0, "plateau": 5.0})  # fast improving (r=0.2), plateau r=1.0
    assert b.weights()["plateau"] > b.weights()["fast"]  # plateau boosted
    b.update({"fast": 1.0, "plateau": 5.0})  # plateau keeps stalling
    assert b.weights()["plateau"] > b.weights()["fast"]
    assert len(b.history) == 4  # trajectory recorded


def test_combine_modes():
    import torch as _t

    from looking_glass.cfm_training import _combine

    seqs = [_seq("c1"), _seq("c2", day="2025-01-02")]
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1)
    cfg = _cfg()
    T_ = {"next": _t.tensor(2.0), "dt": _t.tensor(4.0)}
    # dwa with weights = weighted sum
    v = float(_combine(model, T_, cfg, weights={"next": 1.0, "dt": 0.5}))
    assert v == pytest.approx(2.0 + 2.0)
    # dwa without history = equal sum
    assert float(_combine(model, T_, cfg)) == pytest.approx(6.0)
    # uncertainty mode = kendall formula (log_var zeros -> 0.5*L each)
    cfg_u = _cfg(weight_mode="uncertainty")
    assert float(_combine(model, T_, cfg_u)) == pytest.approx(3.0)
    # equal mode
    cfg_e = _cfg(weight_mode="equal")
    assert float(_combine(model, T_, cfg_e)) == pytest.approx(6.0)
    # objectives gate applies
    cfg_g = _cfg(objectives=("next",))
    assert float(_combine(model, T_, cfg_g, weights={"next": 1.0, "dt": 0.5})) == pytest.approx(2.0)


def test_train_uses_dwa_mode_by_default():
    assert CFMConfig().weight_mode == "dwa"
    assert CFMConfig().dwa_temp == 2.0


def test_variance_floor_hinge():
    import torch as _t

    from looking_glass.cfm_training import _task_losses

    seqs = [_seq("c1"), _seq("c2", day="2025-01-02"), _seq("c3", day="2025-01-03")]
    vocab = EventVocab.build(seqs)
    torch.manual_seed(0)
    model = CFM(vocab, dim=16, n_experts=1)
    cfg = _cfg()
    assert "variance" in cfg.objectives and "rank" in cfg.objectives
    assert CFMConfig().version == "v4.6.0"
    assert CFMConfig().n_experts == 2  # dual-velocity (DEC-025)

    # collapsed states (all identical rows) -> per-dim std 0 -> hinge = 1.0
    h = _t.zeros(4, 16)
    std = h.std(dim=0)
    assert torch.relu(1.0 - std).mean() == pytest.approx(1.0)
    # spread states (unit-ish std per column) -> hinge ~ 0
    g = _t.Generator().manual_seed(1)
    h2 = _t.randn(64, 16, generator=g)
    assert torch.relu(1.0 - h2.std(dim=0)).mean() < 0.1
    # trains: objective present, finite, gradients flow through proj -> trunk
    T_ = _task_losses(model, vocab, seqs, cfg)
    assert "variance" in T_ and torch.isfinite(T_["variance"])
    loss = _loss(model, vocab, seqs, cfg)
    loss.backward()
    assert model.proj.weight.grad is not None


def test_combine_scale_free_units():
    from looking_glass.cfm_training import _combine

    seqs = [_seq("c1"), _seq("c2", day="2025-01-02")]
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1)
    cfg = _cfg()
    T_ = {"next": torch.tensor(100.0), "dt": torch.tensor(1.0)}
    scales = {"next": 100.0, "dt": 1.0}
    w = {"next": 1.0, "dt": 1.0}
    # scale-free: both tasks contribute their NORMALIZED magnitude
    v = float(_combine(model, T_, cfg, weights=w, scales=scales))
    assert v == pytest.approx(2.0)
    # without scales (unit system not yet learned) -> raw sum
    v2 = float(_combine(model, T_, cfg, weights=w))
    assert v2 == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# DEC-019: cross-batch bank + closed-loop geometry governor
# ---------------------------------------------------------------------------
def test_geometry_bank_penalties_and_governor():
    import torch as _t

    from looking_glass.cfm_training import GeometryBank

    b = GeometryBank(dim=16, size=8192, target=0.32, alpha=0.25, lam_max=50.0, tau=0.05)
    # empty bank -> no penalties (fail safe)
    assert b.penalties()["redundancy_bank"] is None
    # collapsed bank: all rows identical -> rank 1/dim, huge gap -> lam ramps
    g = _t.Generator().manual_seed(0)
    collapsed = _t.randn(1, 16, generator=g).repeat(256, 1)
    b.push(collapsed)
    pen = b.penalties()
    assert (
        pen["redundancy_bank"].item() < 1e-8
    )  # within-batch: no decorrelation left to do? NO — bank is one state repeated -> corr off-diag = 1
    assert pen["bank_rank"] < 0.01  # perfectly collapsed: PR ~ 0
    assert b.lam > 1.0  # ramped (rank 0.0625 << target 0.32)
    # spread bank: unit-ish variance per dim, low correlation -> lam backs off
    spread = _t.randn(4096, 16, generator=g)
    b2 = GeometryBank(dim=16, size=8192, target=0.32, alpha=0.25, lam_max=50.0, tau=0.05)
    b2.push(spread)
    pen2 = b2.penalties()
    assert pen2["eigfloor"].item() < pen["eigfloor"].item()  # fewer collapsed eigs
    # closed loop: rank healthy -> lam decays toward 1
    b2.lam = 50.0
    b2.penalties()
    assert b2.lam < 50.0  # backed off


def test_bank_detaches_from_graph():
    import torch as _t

    from looking_glass.cfm_training import GeometryBank

    b = GeometryBank(dim=8, size=128, target=0.32, alpha=0.25, lam_max=50.0, tau=0.05)
    x = _t.randn(64, 8, requires_grad=True)
    b.push(x)  # must not hold graph references (memory leak)
    pen = b.penalties()
    assert pen["redundancy_bank"].requires_grad is False


def test_barrier_and_bank_joint():
    import torch as _t

    from looking_glass.cfm_training import GeometryBank

    b = GeometryBank(dim=16, size=8192, target=0.32, alpha=0.25, lam_max=50.0, tau=0.05)
    g = _t.Generator().manual_seed(0)
    collapsed = _t.randn(1, 16, generator=g).repeat(512, 1)
    b.push(collapsed)
    pen = b.penalties()
    # collapsed bank: barrier is LARGE (near-wall), spread bank: barrier small
    assert pen["barrier"].item() > 0
    b2 = GeometryBank(dim=16, size=8192, target=0.32, alpha=0.25, lam_max=50.0, tau=0.05)
    b2.push(_t.randn(4096, 16, generator=g))
    assert b2.penalties()["barrier"].item() < pen["barrier"].item()


def test_dual_velocity_config_and_ortho_loss():
    from looking_glass.cfm_training import _task_losses

    assert CFMConfig().n_experts == 2
    assert CFMConfig().version == "v4.6.0"
    from looking_glass.cfm_config import _expert_biases

    biases = _expert_biases(2, 7)
    assert biases[0] < biases[1]  # fast expert first, slow expert last
    seqs = [_seq("c1"), _seq("c2", day="2025-01-02"), _seq("c3", day="2025-01-03")]
    torch.manual_seed(0)
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=32, n_experts=2)  # dual-velocity
    cfg = _cfg()
    assert "ortho" in cfg.objectives
    T_ = _task_losses(model, vocab, seqs, cfg)
    assert "ortho" in T_ and torch.isfinite(T_["ortho"])
    loss = _loss(model, vocab, seqs, cfg)
    loss.backward()
    assert model.ssm.experts[0].delta_bias.grad is not None
    assert model.ssm.experts[1].delta_bias.grad is not None
