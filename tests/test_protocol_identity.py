"""Protocol identity (DEC-447/448/449): the name follows the protocol, the switch is switchable,
the served table IS the certified donor, and a card that mixes two encoders refuses itself.

Each test here exists because a real run once did the opposite: two protocols shipped under
one tag (DEC-445); an objective sat unreachable behind a rejected spelling (DEC-449); the
served table skipped the certified boundary (DEC-447).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from looking_glass.cfm_config import CFMConfig, apply_set_overrides  # noqa: E402


def _cfg(**kw):
    c = CFMConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---- tag: the name is a function of the protocol (DEC-448) --------------------------------


def test_tag_is_stable_and_protocol_qualified():
    t = _cfg(batch=20).tag
    assert _cfg(batch=20).tag == t  # deterministic, no process salt
    assert "b20" in t
    # batch changes what the governor sees and when training stops: it MUST name itself
    assert _cfg(batch=27).tag != t
    # the objectives + effect declaration name the weights too (DEC-449's axes)
    assert _cfg(batch=20, effect_actions=("offer",)).tag != t
    # the sample-size suffix survives alongside the protocol suffix
    assert "s400" in _cfg(batch=20, sample_a_customers=400).tag


def test_protocols_a_and_b_cannot_share_a_tag():
    """The exact collision DEC-445 measured: identical code, different batch -> same name."""
    a = _cfg(batch=27, effect_actions=())
    b = _cfg(batch=20, effect_actions=())
    assert a.version == b.version and a.revision == b.revision  # same code identity
    assert a.tag != b.tag  # different protocol identity


def test_effect_head_form_is_a_protocol_axis_only_when_the_head_exists():
    """DEC-453 (protocol D): the bilinear head trains different weights, so two trainings
    differing ONLY in `effect_head` must not share a tag. With the effect objective OFF the
    head is never built, so the field must not split a protocol — A/B's shared tag holds."""
    base = _cfg(batch=20, effect_actions=("offer",))
    bil = _cfg(batch=20, effect_actions=("offer",), effect_head="bilinear")
    assert base.tag != bil.tag
    assert _cfg(batch=20, effect_actions=("offer",), effect_head="mlp").tag == base.tag
    assert _cfg(batch=27).tag == _cfg(batch=27, effect_head="bilinear").tag


# ---- --set: the declaration channel accepts a declaration (DEC-449) -----------------------


def test_tuple_field_accepts_a_comma_separated_declaration():
    c = _cfg()
    c.set_overrides = ["effect_actions=offer"]
    apply_set_overrides(c)
    assert c.effect_actions == ("offer",)

    c2 = _cfg()
    c2.set_overrides = ["effect_actions=offer, freq"]
    apply_set_overrides(c2)
    assert c2.effect_actions == ("offer", "freq")


def test_tuple_field_still_rejects_a_declaration_of_nothing():
    c = _cfg()
    c.set_overrides = ["effect_actions=,,"]
    with pytest.raises(SystemExit):
        apply_set_overrides(c)


def test_existing_coercions_and_rejections_are_unchanged():
    c = _cfg()
    c.set_overrides = ["batch=20"]
    apply_set_overrides(c)
    assert c.batch == 20 and isinstance(c.batch, int)
    c2 = _cfg()
    c2.set_overrides = ["isometric_boundary=false"]
    apply_set_overrides(c2)
    assert c2.isometric_boundary is False
    c3 = _cfg()
    c3.set_overrides = ["batch=off"]
    with pytest.raises(SystemExit):
        apply_set_overrides(c3)
    c4 = _cfg()
    c4.set_overrides = ["isometric_boundary=maybe"]
    with pytest.raises(SystemExit):
        apply_set_overrides(c4)


# ---- served donor: state_embeddings IS donor(h) (DEC-447) ---------------------------------


def _model_with_boundary(dim=8):
    from tests.test_intrinsic import _seqs  # one sequence builder, not a copy

    from looking_glass.cfm_model import CFM, EventVocab

    torch.manual_seed(0)
    model = CFM(EventVocab.build(_seqs()), dim=dim, n_experts=1, isometric_boundary=True)
    model.eval()
    with torch.no_grad():
        # a real learned boundary: zero skew would make R=I and the test vacuous
        model.iso_skew.copy_(torch.randn(dim, dim) * 0.3)
    return model


def test_donor_states_serves_the_certified_boundary_readout():
    from looking_glass.cfm_state import donor_states

    model = _model_with_boundary()
    rng = np.random.default_rng(0)
    states = [rng.normal(size=8).astype(np.float32) for _ in range(5)]

    served = donor_states(model, states)
    h = torch.as_tensor(np.stack(states))
    with torch.no_grad():
        assert np.allclose(served, model.donor_batch(h).numpy(), atol=1e-6)
        # the boundary is NOT decoration: with a learned skew it differs from raw proj...
        assert not np.allclose(served, model.proj(h).numpy(), atol=1e-3)
        # ... and the batch publisher agrees with the per-state donor, row for row
        per_state = torch.stack([model.donor(s) for s in h])
        assert torch.allclose(model.donor_batch(h), per_state, atol=1e-5)


# ---- the reader refuses a card that mixes encoders (DEC-448) ------------------------------


def test_mixed_artifact_card_refuses_the_minority_rows(tmp_path):
    import report_card as RC
    import judge_spec as J

    ident_a = {"path": "/work/cfm_b", "tag": "t-b20", "registry_sha256": "aaaa"}
    ident_b = {"path": "/work/cfm_a", "tag": "t-b27", "registry_sha256": "bbbb"}

    def receipt(pid):
        task = {
            "name": "churn90",
            "brain": 0.78,
            "pathA": 0.70,
            "ci_lo": 0.02,
            "ci_hi": 0.14,
            "n": 6000,
            "random": 0.5,
            "destroyed": 0.51,
        }
        return {
            "_meta": {
                "generated_at": "2026-10-30T00:00:00+00:00",
                "git_commit": "deadbeef",
                "behavior_hash": "abc123",
            },
            "columns": {c: {"tasks": [task]} for c in J.COLUMNS},
            "products": pid,
        }

    sources = sorted({r.source for r in J.LEG_ROWS})
    for i, name in enumerate(sources):
        (tmp_path / name).write_text(
            json.dumps(receipt(ident_a if i < len(sources) - 1 else ident_b))
        )
    mix = RC._artifact_mismatch(tmp_path, None)
    assert mix, "a card mixing two artifacts must produce a mismatch note"
    # the last source is the lone protocol-A row: it is the minority and gets the note
    assert sources[-1] in mix
    assert "mixes artifacts" in mix[sources[-1]]
    assert all(s in mix for s in sources[:-1]) is False  # majority is untouched
    # a uniform card (all A) produces no notes at all
    for name in sources:
        (tmp_path / name).write_text(json.dumps(receipt(ident_a)))
    assert RC._artifact_mismatch(tmp_path, None) == {}
    # stage blocks participate too: an S4 from the other artifact is named
    mix2 = RC._artifact_mismatch(tmp_path, {"S4": {"working": True, "products": dict(ident_b)}})
    assert "stage::S4" in mix2
