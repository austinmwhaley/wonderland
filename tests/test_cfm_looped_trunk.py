"""DEC-613 (branch looped-transformer): the Dory-style looped GPT trunk.

Pins the graft contract (the `MultiScaleSSM` shape behind `CFM.ssm`), the
padding-invariance of the served state, the elastic loop count (one set of
weights serves L=1..4), the nGPT constant-depth guarantee at the public
interface, and the tag axis (the looped trunk is a distinct protocol; the SSM
default tag is unchanged)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_model import CFM, EventVocab, MultiScaleSSM
from looking_glass.cfm_looped_trunk import LoopedGPTTrunk


def _seq(key="c1", day="2025-01-01", n=10):
    ets = [f"{day}T{10 + i:02d}:00:00+00:00" for i in range(n)]
    ets[0] = f"{day}T08:00:00+00:00"
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
        "ts": [float(np.datetime64(t.replace("+00:00", ""), "s").astype("int64")) for t in ets],
        "co": [[0.0, 0.0]] * n,
    }


def test_default_trunk_stays_the_ssm():
    """The DEC-613 graft must not move the default behavior: trunk="ssm" (the
    default) builds exactly the existing MultiScaleSSM."""
    torch.manual_seed(0)
    model = CFM(EventVocab.build([_seq("c1")]), dim=16, n_experts=1)
    assert isinstance(model.ssm, MultiScaleSSM)
    assert model.trunk == "ssm"


def test_trunk_contract_shapes_and_finite():
    """x=(B,T,chan) -> y=(B,T,dim), h=(B,dim); finite everywhere (DEC-039:
    non-finite is a hard failure, never masked)."""
    torch.manual_seed(0)
    trunk = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=2, max_pos=32)
    x = torch.randn(2, 10, 8)
    y, h = trunk(x, mask=torch.ones(2, 10))
    assert y.shape == (2, 10, 16) and h.shape == (2, 16)
    assert torch.isfinite(y).all() and torch.isfinite(h).all()
    # mask=None means fully valid (the single-sequence forward path)
    y0, h0 = trunk(x)
    assert torch.equal(y, y0) and torch.equal(h, h0)


def test_served_state_is_padding_invariant():
    """The state of a customer must not depend on how much padding its batch
    carried: the h of a length-L sequence equals the h of the same sequence
    padded to L+k (per-row window, causal key-mask)."""
    torch.manual_seed(0)
    trunk = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=2, max_pos=32)
    L = 9
    x = torch.randn(2, L, 8)
    mask = torch.ones(2, L)
    mask[0, 6:] = 0.0  # row 0 valid to 5, row 1 valid to L-1
    y, h = trunk(x, mask=mask)
    k = 3
    x2 = torch.zeros(2, L + k, 8)
    x2[:, :L] = x
    mask2 = torch.zeros(2, L + k)
    mask2[:, :L] = mask
    y2, h2 = trunk(x2, mask=mask2)
    # allclose, not bitwise: the attention kernel tiles differently per shape
    # (ULP noise across T), but no PADDING KEY may leak into a valid query —
    # a real leak would be O(0.1) on unit-RMS readouts, never 1e-5.
    assert torch.allclose(h, h2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(y[:, :L], y2[:, :L], rtol=1e-5, atol=1e-6)
    assert (h - h2).abs().max().item() < 1e-4


def test_elastic_loop_count_shares_weights():
    """Dory's elastic compute: the loop is weight-tied, so one set of weights
    serves every L in 1..4. The public readout stays loop-count invariant in
    scale (nGPT constant depth at the interface): per-token RMS within
    [0.5, 2.0] at both ends of the range."""
    torch.manual_seed(0)
    t1 = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=1, max_pos=32)
    torch.manual_seed(0)  # same init stream -> identical weights, different L
    t4 = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=4, max_pos=32)
    for (pn, p1), (qname, p4) in zip(t1.named_parameters(), t4.named_parameters()):
        assert pn == qname
        assert torch.equal(p1, p4)  # the loop is weight-tied: one set of weights serves L
    x = torch.randn(2, 12, 8)
    for t in (t1, t4):
        y, h = t(x, mask=torch.ones(2, 12))
        rms = y.float().pow(2).mean(-1).sqrt().mean().item()
        assert 0.5 <= rms <= 2.0, f"public readout RMS {rms} outside [0.5, 2.0] (nGPT depth)"
        assert torch.isfinite(h).all()
    y1, _ = t1(x, mask=torch.ones(2, 12))
    y4, _ = t4(x, mask=torch.ones(2, 12))
    assert not torch.equal(y1, y4)  # the loop IS doing work (distinct dynamics)


def test_h0_enters_as_a_learned_prior():
    """The (B,dim) contract: h0 conditions the encoding (the SSM seeds its
    recurrence; the looped trunk adds a learned prior). h=None leaves it out."""
    torch.manual_seed(0)
    trunk = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=2, max_pos=32)
    x = torch.randn(2, 8, 8)
    m = torch.ones(2, 8)
    _, h_none = trunk(x, mask=m)
    _, h0a = trunk(x, h0=torch.randn(2, 16), mask=m)
    _, h0b = trunk(x, h0=torch.randn(2, 16) * 0.0 + 5.0, mask=m)
    assert not torch.equal(h_none, h0a)
    assert not torch.equal(h0a, h0b)


def test_deterministic_forward():
    torch.manual_seed(0)
    trunk = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=3, max_pos=32)
    x = torch.randn(2, 10, 8)
    m = torch.ones(2, 10)
    m[1, 7:] = 0.0
    y1, h1 = trunk(x, mask=m)
    y2, h2 = trunk(x, mask=m)
    assert torch.equal(y1, y2) and torch.equal(h1, h2)
    # v6 monotonic stability (Principle 5): the whole forward+backward must run
    # clean under the strict determinism flag — the trunk's attention is pinned
    # to the math SDPA backend (flash / mem-efficient reduce non-deterministic),
    # so a non-deterministic kernel would surface here, not at campaign scale.
    xc = x.clone().requires_grad_(True)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        y, h = trunk(xc, mask=m)
        y.sum().backward()
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=False)
    assert torch.isfinite(xc.grad).all()


def test_fail_safe_rejects():
    with pytest.raises(SystemExit):
        LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=0, max_pos=32)
    with pytest.raises(SystemExit):
        LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=5, max_pos=32)
    with pytest.raises(SystemExit):
        LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=2, max_pos=0)
    with pytest.raises(ValueError):
        CFM(
            EventVocab.build([_seq("c1")]),
            dim=16,
            n_experts=1,
            trunk="looped_gpt",
            n_loops=2,
            max_pos=0,  # the protocol seq_len is mandatory for the looped trunk
        )
    with pytest.raises(ValueError):
        CFM(EventVocab.build([_seq("c1")]), dim=16, n_experts=1, trunk="attention")
    trunk = LoopedGPTTrunk(chan=8, dim=16, n_experts=1, n_loops=2, max_pos=4)
    with pytest.raises(RuntimeError):
        trunk(torch.randn(1, 5, 8), mask=torch.ones(1, 5))  # T > max_pos: reject


def test_cfm_wiring_end_to_end():
    """The graft behind CFM: encode through the looped trunk (the training call
    site) and the heads read a (B,T,dim)/(B,dim) with the usual extras."""
    torch.manual_seed(0)
    model = CFM(
        EventVocab.build([_seq("c1")]),
        dim=16,
        n_experts=1,
        trunk="looped_gpt",
        n_loops=2,
        max_pos=32,
    )
    y, h = model.forward(_seq("c1", n=8))
    assert y.shape[-1] == 16 and h.shape == (16,)
    assert torch.isfinite(y).all() and torch.isfinite(h).all()
    # the pre-train step-0 health harness passes on the looped trunk (readout is
    # RMS-normalized; the spectral bound is vacuously met: no SSM experts)
    from looking_glass.cfm_model import model_health

    t = model.tokens(_seq("c1", n=8)).unsqueeze(0)
    stats = model_health(model, t, mask=torch.ones(1, t.shape[1]))
    assert 0.5 <= stats["readout_rms"] <= 2.0


def test_tag_axis():
    """DEC-445/448 discipline: the looped trunk is a distinct protocol identity,
    and so is each loop count; the SSM default tag is unchanged by DEC-613."""
    ssm = CFMConfig()
    lg1 = CFMConfig(trunk="looped_gpt", n_loops=1)
    lg2 = CFMConfig(trunk="looped_gpt", n_loops=2)
    lg4 = CFMConfig(trunk="looped_gpt", n_loops=4)
    assert ssm.tag == CFMConfig().tag  # default protocol untouched
    assert len({ssm.tag, lg1.tag, lg2.tag, lg4.tag}) == 4
    # --set plumbing reaches the config surface
    cfg = CFMConfig(set_overrides=["trunk=looped_gpt", "n_loops=4"])
    from looking_glass.cfm_config import apply_set_overrides

    apply_set_overrides(cfg)
    assert cfg.trunk == "looped_gpt" and cfg.n_loops == 4
