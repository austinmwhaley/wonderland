from __future__ import annotations

import numpy as np
import torch

from looking_glass.cfm_model import CFM, EventVocab
from looking_glass.probes import _stronger_per_sample, _verdict_ci
from looking_glass.report_card import _rl_checks


def _seqs(n=6, L=6):
    out = []
    for c in range(n):
        out.append(
            {
                "event_type": ["view", "cart", "purchase", "view", "purchase", "view"][:L],
                "brand": [None] * L,
                "entity_type": ["product"] * L,
                "entity_id": [None] * L,
                "value": [0.0, 0.0, 5.0 * (c + 1), 0.0, 3.0 * (c + 1), 0.0][:L],
                "ts": [1_700_000_000.0 + i * 3600 for i in range(L)],
                "customer": f"c{c}",
            }
        )
    return out


def test_rl_checks_stream_agnostic_no_actions_does_not_crash():
    """A behavioral stream with NO declared actions and no conversion event must
    not crash the card — the action-conditioned checks are omitted, not one-class
    LogisticRegression'd (the ecommerce_2019 crash)."""
    torch.manual_seed(0)
    seqs = _seqs()
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1).eval()
    rows_none = _rl_checks(model, vocab, seqs, actions=(), order_event=None, seed=0)
    assert rows_none == []


def test_rl_checks_derives_from_declared_actions_and_order_event():
    torch.manual_seed(0)
    seqs = _seqs(n=8)
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=16, n_experts=1).eval()
    rows = _rl_checks(model, vocab, seqs, actions=("cart",), order_event="purchase", seed=0)
    names = [r[0] for r in rows]
    # outcome sufficiency is generic and must appear; no crash on action checks
    assert any("outcome sufficiency" in n for n in names)
    assert all(isinstance(r[2], str) for r in rows)
    assert np.all([r[2] in {"PASS", "FAIL", "UNCERTAIN"} for r in rows])


def test_stronger_baseline_is_not_a_per_sample_oracle():
    """Regression (false-FAIL bug): the strong baseline must be chosen by MEAN
    loss and applied per sample — never `min(Bnull, Bstrong)` per sample, which is
    an unbeatable best-of oracle. Here the gram baseline is stronger on average but
    worse on one sample; the per-sample oracle would pick the unigram there and
    manufacture a spuriously strong baseline."""
    Bnull = np.array([2.0, 2.0, 2.0, 2.0])  # mean 2.0 (the STRONGER one by mean)
    Bstrong = np.array([0.5, 0.5, 0.5, 9.0])  # mean 2.625 (worse on average)
    got = _stronger_per_sample(
        bnull=float(Bnull.mean()), bstrong=float(Bstrong.mean()), Bnull=Bnull, Bstrong=Bstrong
    )
    # Bnull is stronger by mean -> used for ALL samples (not the 0.5 oracle picks)
    assert np.allclose(got, Bnull)
    assert got[0] == 2.0  # NOT the per-sample oracle's 0.5


def test_verdict_ci_pass_fail_no_signal():
    # model better than the (strong) baseline -> PASS
    assert _verdict_ci(1.0, 0.5, 1.5, stronger_base=2.0, null_base=3.0, floor=0.1) == "PASS"
    # model worse -> FAIL when the baseline carries signal
    assert _verdict_ci(-1.0, -1.5, -0.5, stronger_base=2.0, null_base=3.0, floor=0.1) == "FAIL"
    # baseline at the trivial floor -> NO_SIGNAL
    assert _verdict_ci(0.01, -0.1, 0.1, stronger_base=3.0, null_base=3.0, floor=0.1) == "NO_SIGNAL"
