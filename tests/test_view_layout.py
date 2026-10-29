"""The outlet schema must describe the encoder, not the author's memory of it.

S4's gate is per-view accessibility; the precondition is that the published view table
means what its name says. These tests pin the three rules `view_layout` claims: derived
dims, named parts, and declared-but-not-emitted outlets that stay visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from looking_glass.cfm_model import CFM, EventVocab  # noqa: E402
from looking_glass import view_layout as V  # noqa: E402


def _seqs():
    """A toy stream in the canonical shape the encoder consumes: event_type plus an
    attribute bag and a numeric field (values repeat so the derived vocabulary keeps them —
    a singleton is unlearnable and maps to OOV by law, not by accident)."""
    return [
        {
            "event_type": ["a", "b", "c", "a"] * 3,
            "attrs": {"channel": ["web", "mob", "web", "mob"] * 3},
            "numeric": {"value": [1.0, 2.0, 3.0, 4.0] * 3},
        },
        {
            "event_type": ["b", "c", "a", "d"] * 2,
            "attrs": {"channel": ["mob", "mob", "web", "web"] * 2},
            "numeric": {"value": [0.5, 1.5, 2.5, 3.5] * 2},
        },
    ]


def _model(dim=16):
    torch.manual_seed(0)
    return CFM(EventVocab.build(_seqs()), dim=dim, n_experts=1)


def _yh(model):
    with torch.no_grad():
        h = torch.randn(3, model.dim)
        y = torch.randn(3, model.dim)
    return y, h


def test_view_dims_are_derived_from_the_model_not_from_a_literal():
    """The receipt's numbers must move when the encoder does — otherwise the faceplate is a
    hand-written claim, which is exactly what 'generate, don't narrate' forbids."""
    model = _model(dim=16)
    d1 = V.describe(model)
    assert d1["views"]["shape"] == 16  # the public readout is the encoder width
    with torch.no_grad():  # widening a head must widen its view automatically
        model.head_psr = torch.nn.Linear(model.dim, 11)
        model.head_dt = torch.nn.Linear(model.dim, 4)
    d2 = V.describe(model)
    assert d2["views"]["forecast"] == d1["views"]["forecast"] + (11 - 6)
    assert d2["views"]["uncertainty"] == d1["views"]["uncertainty"] + (4 - 1)


def test_every_declared_outlet_is_either_emitted_or_advertised():
    """An approved outlet that is not built must be ADVERTISED (O5: advertise as
    uncertified), never quietly dropped from the schema."""
    d = V.describe(_model())
    assert set(d["emitted"]) <= set(V.DECLARED)
    for view in V.FULL_ORDER:
        assert view in d["emitted"], f"{view} is in the full bundle but publishes nothing"
    assert "response" in d["declared_not_emitted"]  # DEC-384: approved, not built
    assert "measure" in d["declared_not_emitted"]  # DEC-383: the ledger is a separate producer


def test_full_bundle_is_the_named_concat_and_excludes_the_ledger():
    """`full` is a bundle, not content (SPINE §5): its width is the sum of FULL_ORDER and
    `measure` can never be inside it."""
    model = _model()
    d = V.describe(model)
    assert d["full_dim"] == sum(d["views"][v] for v in V.FULL_ORDER)
    assert "measure" not in d["full_order"] and "measure" not in d["emitted"]


def test_parts_are_named_and_concatenating_them_rebuilds_the_view():
    """Attribution and the knob-turn receipt both need per-part access: zeroing one part
    must be expressible, which it is only if the parts are recorded separately."""
    model = _model()
    y, h = _yh(model)
    group, views = V.parts(model, y, h), V.emit(model, y, h)
    for view, group_parts in group.items():
        assert group_parts, f"{view} emitted an empty group"
        assert all(t.shape[0] == 3 for t in group_parts.values()), "batch dim must survive"
        rebuilt = torch.cat(list(group_parts.values()), dim=-1)
        assert torch.equal(rebuilt, views[view])


def test_a_missing_head_drops_that_part_and_says_so(capsys):
    """Fail safe, but never silently: an absent head shrinks the view AND is reported, so a
    narrow view is explainable rather than a mystery for the next reader."""
    model = _model()
    del model.head_psr
    y, h = _yh(model)
    group = V.parts(model, y, h)
    assert "psr" not in group["forecast"]
    assert "future_state" in group["forecast"]  # the rest of the view survives
    assert "head_psr" in capsys.readouterr().out


def test_faceplate_is_self_describing_enough_to_refuse():
    """Consumers self-validate against the artifact (DEC-405 rule 4), so the stamp must name
    the version, every view width, and what it does NOT serve."""
    from looking_glass.cfm_config import CFMConfig

    model = _model()
    fp = V.faceplate(model, CFMConfig())
    assert fp["kind"] == "view_faceplate" and isinstance(fp["version"], str) and fp["version"]
    assert set(fp["views"]) == set(fp["parts"])
    # a consumer must be able to tell WHY an outlet is not served, not just that it is absent
    assert "response" in fp["not_served"] and len(fp["not_served"]["response"]) > 10


def test_the_single_and_batched_public_readout_agree():
    """`shape` has one meaning. `CFM.donor` (single state) and `CFM.donor_batch` (state
    batch) are two code paths, and this repo has already paid for train/serve skew twice —
    so the published column pins them together instead of trusting the docstring."""
    model = _model()
    with torch.no_grad():
        h = torch.randn(model.dim)
        single = V.emit(model, h.clone(), h)["shape"]
        batched = V.emit(model, h.unsqueeze(0), h.unsqueeze(0))["shape"]
    assert torch.allclose(single.reshape(-1), batched[0], atol=1e-6), (single, batched)


def test_a_per_step_readout_is_rejected_not_published():
    """`CFM.forward` hands back y=[T,dim] and h=[dim]. Feeding THAT to the heads makes every
    published column depend on the customer's sequence length — the first build of DEC-424
    did exactly this and shipped 2500 ragged rows. The producer must refuse."""
    model = _model()
    with torch.no_grad():
        h = torch.randn(model.dim)
        y_per_step = torch.randn(7, model.dim)
        with pytest.raises(ValueError, match=r"y\[-1\]\.unsqueeze\(0\)"):
            V.parts(model, y_per_step, h)


# ---------------------------------------------------------------- the producer's own gate
# `verify_outlets` is what stops a bad artifact from being readable at all. It needs a real
# DuckDB store, so it is tested against one rather than asserted in a docstring.


def _store(tmp_path):
    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_state import StateStore

    return StateStore(tmp_path / "p.duckdb", CFMConfig(), None)


def _row(customer, view, part, width):
    return {
        "customer_key": customer,
        "as_of_epoch": 1.0,
        "version": "test",
        "view": view,
        "part": part,
        "dim": width,
        "embedding": [0.5] * width,
    }


def test_a_ragged_published_column_is_refused(tmp_path):
    """A column whose width varies by customer means the producer fed the heads the wrong
    readout. Shipping it is worse than not shipping: consumers validated against the stamp
    and then read something else."""
    store = _store(tmp_path)
    store.bulk_view_rows([_row("a", "shape", "donor", 4), _row("b", "shape", "donor", 5)])
    with pytest.raises(RuntimeError, match="ragged widths 4..5"):
        store.verify_outlets({"parts": {"shape": {"donor": 4}}})
    store.close()


def test_a_column_that_disagrees_with_its_own_stamp_is_refused(tmp_path):
    """Uniform-but-undocumented is the other failure mode: the artifact describes 4 and
    publishes 6. The fix belongs in the producer, so the message says so."""
    store = _store(tmp_path)
    store.bulk_view_rows([_row("a", "shape", "donor", 6), _row("b", "shape", "donor", 6)])
    with pytest.raises(RuntimeError, match="stored 6 != faceplate 4"):
        store.verify_outlets({"parts": {"shape": {"donor": 4}}})
    store.close()


def test_a_conforming_release_passes_the_gate(tmp_path):
    store = _store(tmp_path)
    store.bulk_view_rows(
        [
            _row("a", "shape", "donor", 4),
            _row("b", "shape", "donor", 4),
            _row("a", "heads", "next_event", 3),
        ]
    )
    store.verify_outlets({"parts": {"shape": {"donor": 4}, "heads": {"next_event": 3}}})
    store.close()


def test_the_published_part_order_is_the_contract():
    """The S4 receipt rebuilds a view from the published parts, so the order the producer
    concatenated in has to be readable WITHOUT a model — otherwise the consumer guesses."""
    assert V.PART_ORDER["learned_state"] == ["recurrence", "ledger_totals", "rfm_recon"]
    assert set(V.PART_ORDER) == set(V.FULL_ORDER)
