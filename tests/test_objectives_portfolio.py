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
