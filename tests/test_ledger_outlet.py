"""The ledger must be a LEDGER (DEC-383 / SPINE §5 G1–G5), not a tally that merely runs.

Five claims are asserted rather than described: the columns are exactly the recipe's layout,
the numbers are the exact tallies they claim to be, no event after the cut contributes, the
guardrails actually fire (a check that cannot fail is not a check), and the faceplate only
advertises the `measure` socket when the producer wrote it.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from looking_glass import ledger_state as L  # noqa: E402
from looking_glass.cfm_model import EventVocab  # noqa: E402
from looking_glass.measure_state import MeasureBuilder  # noqa: E402

ET = ("view", "click", "buy")
CATS = {"channel": ("email", "sms", "push"), "region": ("n", "s")}
NUMS = ("margin", "qty")


def _vocab():
    return EventVocab(
        ET,
        attrs={f: list(v) for f, v in CATS.items()},
        numeric_fields=NUMS,
        primary_numeric="margin",
    )


def _seqs(n=9, seed=0, ts0=1_700_000_000, step=3600):
    """Deterministic synthetic histories: raw event_type / attribute STRINGS, raw numerics."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_ev = 4 + int(rng.integers(0, 6))
        out.append(
            {
                "customer": f"c{i}",
                "group": "A" if i % 3 else "B",
                "anchor_epoch": None,
                "event_type": [str(rng.choice(ET)) for _ in range(n_ev)],
                "attrs": {
                    f: np.array([str(rng.choice(list(v))) for _ in range(n_ev)])
                    for f, v in CATS.items()
                },
                "numeric": {
                    c: (
                        rng.normal(10.0, 3.0, n_ev)
                        if c == "margin"
                        else rng.integers(0, 5, n_ev).astype(float)
                    )
                    for c in NUMS
                },
                "numeric_mask": {c: np.ones(n_ev, dtype=bool) for c in NUMS},
                "is_action": np.zeros(n_ev),
                "value": np.zeros(n_ev),
                "primary_numeric": "margin",
                "event_ts": [ts0 + step * j for j in range(n_ev)],
                "ts": [float(ts0 + step * j) for j in range(n_ev)],
                "co": [],
                "eff_act": None,
                "eff_rew": None,
            }
        )
    return out


class _Cfg:
    """The config surface the guardrails read: attribution receipt + declared target names."""

    def __init__(self, attribution=None):
        self._ledger_attribution = attribution
        self.exogenous_events = ()
        self.action_fields = ()


@pytest.fixture
def built():
    vocab = _vocab()
    seqs = _seqs()
    train = [s["customer"] for s in seqs if s["group"] == "A"]
    builder, receipt = L.fit_vocabulary(seqs, vocab, train)
    return vocab, seqs, builder, receipt


def test_the_published_columns_are_exactly_the_recipes_layout(built):
    """The faceplate's `measure` widths and the published rows must both come from
    `part_slices`, and their union must tile the builder's own dim. The ragged-`shape` bug of
    DEC-425 was a layout described in one place and built in another."""
    _, _, builder, _ = built
    runs = builder.part_slices()
    assert set(runs) == set(L.PARTS)
    spans = sorted((int(a), int(b)) for r in runs.values() for a, b in r)
    assert spans[0][0] == 0
    for (_a0, b0), (a1, _b1) in zip(spans, spans[1:]):
        assert a1 >= b0, "overlapping runs would publish one column under two names"
    assert sum(b - a for a, b in spans) == builder.dim


def test_the_numbers_are_the_exact_tallies_they_claim(built):
    """P8/P11: a column called "exact" is falsifiable against a hand count."""
    vocab, seqs, builder, _ = built
    rows = L.publish_rows(seqs, builder, vocab, version="t")
    by = {(r["customer_key"], r["part"]): r["embedding"] for r in rows}
    for s in seqs:
        k = s["customer"]
        assert by[(k, "event_mass")][: len(ET)] == pytest.approx(
            [sum(1 for e in s["event_type"] if e == et) for et in ET]
        )
        assert by[(k, "event_mass")][len(ET)] == 0.0  # nothing unseen in this synthetic stream
        assert by[(k, "total_events")] == [float(len(s["event_type"]))]
        x = np.asarray(s["numeric"]["margin"], dtype=np.float64)
        cnt, sm, mean, sd, mn, mx = by[(k, "numeric_stats")][:6]
        assert cnt == len(x) and sm == pytest.approx(x.sum())
        assert mean == pytest.approx(x.mean()) and sd == pytest.approx(x.std())
        assert mn == pytest.approx(x.min()) and mx == pytest.approx(x.max())
        # per field, the shares over the capped vocabulary (incl. the "other" bucket) sum to 1
        off = 0
        for keep in builder.keep:
            width = len(keep) + 1
            assert sum(by[(k, "category_share")][off : off + width]) == pytest.approx(1.0)
            off += width


def test_an_event_after_the_cut_cannot_contribute(built):
    """G2 as a property of the PUBLISHER, not only of the audit: the as-of cut uses the same
    timestamp rule as the encoder, and a row cut mid-history must equal the row for the
    truncated history — a forward peek cannot survive that equality."""
    vocab, seqs, builder, _ = built
    cut = seqs[0]["ts"][2]  # every customer keeps exactly their first three events
    rows = L.publish_rows(seqs, builder, vocab, version="t", as_of=cut)
    total = {(r["customer_key"], r["part"]): r["embedding"] for r in rows}
    for s in seqs:
        assert total[(s["customer"], "total_events")] == [3.0], s["customer"]


def test_the_guardrails_run_and_report_evidence(built):
    vocab, _, builder, _ = built
    seqs_a = [s for s in _seqs() if s["group"] == "A"]
    cut = float(np.median([s["ts"][-1] for s in seqs_a]))
    tuples_a = L.seqs_to_tuples(seqs_a, vocab, as_of=cut)
    r = L.guardrails(
        builder,
        vocab,
        tuples_a,
        _Cfg(),
        [s["customer"] for s in seqs_a],
        ("view", "buy"),
        as_of=cut,
    )
    for g in ("G1_mechanical", "G2_history_only", "G3_target_blind", "G4_not_a_brain_input"):
        assert r[g]["working"], (g, r[g]["evidence"])
        assert len(r[g]["evidence"]) > 20, g  # a check with nothing to show is not a receipt
    # G5 is a soft receipt: before the card has run, the gap is named rather than hidden
    assert not r["G5_attribution"]["working"]
    assert "controls=(none)" in r["G5_attribution"]["evidence"]


def test_a_forward_peek_is_caught_by_g2(built):
    """The point of G2: if the publisher forgets to apply the as-of cut, the ledger silently
    counts events that had not happened yet. The guardrail recounts independently and must see
    the difference — this is the defect class (a future leak published as history) that no
    amount of prose in a design doc has ever prevented."""
    vocab, _, builder, _ = built
    seqs_a = [s for s in _seqs() if s["group"] == "A"]
    cut = float(np.median([s["ts"][-1] for s in seqs_a]))
    uncut = L.seqs_to_tuples(seqs_a, vocab)  # the bug: nothing was cut
    r = L.guardrails(
        builder,
        vocab,
        uncut,
        _Cfg(),
        [s["customer"] for s in seqs_a],
        ("nothing_named",),
        as_of=cut,
    )
    assert not r["G2_history_only"]["working"], r["G2_history_only"]["evidence"]
    assert "disagree" in r["G2_history_only"]["evidence"]


def test_a_guardrail_that_cannot_fail_is_not_a_guardrail(built):
    """Each law is broken deliberately and its own check must catch it — this is the whole
    reason G1–G5 are code and not prose in SPINE §5."""
    vocab, _, builder, _ = built
    seqs_a = [s for s in _seqs() if s["group"] == "A"]
    tuples_a = L.seqs_to_tuples(seqs_a, vocab)
    keys = [s["customer"] for s in seqs_a]
    cfg_ok = _Cfg(L.attribution_receipt({"ledger_only": 0.5, "learned_only": 0.2}))

    # G3: a recipe whose own text names the target it must not know about is not target-blind
    class _LeakyRecipe(MeasureBuilder):
        def transform(self, seqs, keys):  # noqa: ANN001  — deliberately references "buy"
            return super().transform(seqs, keys)  # the target "buy" appears in this body

    leaky = _LeakyRecipe.__new__(_LeakyRecipe)
    leaky.__dict__.update(builder.__dict__)
    r = L.guardrails(leaky, vocab, tuples_a, cfg_ok, keys, ("buy", "view"))
    assert not r["G3_target_blind"]["working"], r["G3_target_blind"]["evidence"]
    assert "FOUND" in r["G3_target_blind"]["evidence"]

    # G2: a builder that invents an event nobody logged must be caught, not admired
    class _Peeking(MeasureBuilder):
        def transform(self, seqs, keys):
            out = super().transform(seqs, keys).copy()
            out[:, 0] += 1.0  # every row gains an event that is not in its history
            return out

    peek = _Peeking.__new__(_Peeking)
    peek.__dict__.update(builder.__dict__)
    r2 = L.guardrails(peek, vocab, tuples_a, cfg_ok, keys, ("nothing_named_here",))
    assert not r2["G2_history_only"]["working"], r2["G2_history_only"]["evidence"]

    # G4: the brain's own training path may not read the ledger. A module whose source names the
    # ledger is rejected — and so is one whose source cannot be read, which is not the same
    # thing as clean.
    import types

    bad = types.ModuleType("pretend_trainer")
    bad.__dict__["__source__"] = None  # placeholder; getsource falls back to the file
    src_backup = L._source_of
    L._source_of = lambda obj: (
        "import measure_state  # the brain reads the ledger"
        if getattr(obj, "__name__", "") == "pretend_trainer"
        else src_backup(obj)
    )
    try:
        r3 = L.guardrails(
            builder,
            vocab,
            tuples_a,
            cfg_ok,
            keys,
            ("nothing_named_here",),
            brain_modules=[bad],
        )
        assert not r3["G4_not_a_brain_input"]["working"], r3["G4_not_a_brain_input"]["evidence"]
    finally:
        L._source_of = src_backup

    # and the real brain modules must pass, which is the claim that actually matters
    r_real = L.guardrails(builder, vocab, tuples_a, cfg_ok, keys, ("nothing_named_here",))
    assert r_real["G4_not_a_brain_input"]["working"], r_real["G4_not_a_brain_input"]["evidence"]

    # G5: with the attribution receipt present, the row is closed
    assert r2["G5_attribution"]["working"]


def test_an_uninspectable_recipe_is_not_reported_as_a_pass(built):
    """ "I could not read the source" must never print as "the law holds"."""
    assert L._source_of(lambda: None)  # lambdas ARE inspectable
    import inspect as _inspect

    real = _inspect.getsource

    def boom(obj):
        raise OSError("source unavailable")

    _inspect.getsource = boom
    try:
        assert L._source_of(lambda: None) == "<<UNINSPECTABLE>>"
    finally:
        _inspect.getsource = real


def test_unseen_values_get_a_bucket_of_their_own_not_a_clip_onto_a_real_one(built):
    """The tokenizer sends an unseen value to `len(map)`. If the ledger sized its columns by
    the bare vocabulary, `np.clip` would fold every unseen value onto the LAST real value —
    a silent substitution that would make a rare category look like a common one. Missing and
    OOV are therefore first-class buckets."""
    vocab, _, builder, _ = built
    s = _seqs(1, seed=3)[0]
    s["event_type"] = ["view", "never_seen_event_type", "never_seen_event_type"]
    s["attrs"] = {f: np.array(["email", None, "telepathy"]) for f in CATS}
    s["numeric"] = {c: np.array([1.0, 0.0, 3.0]) for c in NUMS}
    s["numeric_mask"] = {c: np.array([True, False, True]) for c in NUMS}
    for k in ("event_ts", "ts"):
        s[k] = [float(1_700_000_000 + j) for j in range(3)]
    rows = L.publish_rows([s], builder, vocab, version="t")
    by = {r["part"]: r["embedding"] for r in rows}

    # the unseen EVENT is its own column, and the two real ones still count correctly
    assert by["event_mass"][: len(ET)] == [1.0, 0.0, 0.0]
    assert by["event_mass"][len(ET)] == 2.0, "the OOV event bucket is where unseen types land"
    assert by["total_events"] == [3.0]

    # a never-logged attribute is missing, and an unseen value is NOT folded onto a real one
    channel_missing = by["missing_rate"][0]
    assert channel_missing == pytest.approx(1.0 / 3.0)
    shares = by["category_share"][: len(CATS["channel"]) + 1]
    assert shares[0] == pytest.approx(1 / 3)  # email
    assert shares[-1] == pytest.approx(2 / 3)  # "other": the None and the unseen value

    # a value masked out by the stream's own null mask is not tallied as a zero
    assert by["numeric_stats"][0] == 2.0, "count must follow the null mask, not the float 0.0"


def test_a_hand_edited_vocabulary_fails_g1(built):
    """G1's whole purpose is "no human ever hand-picks columns". The check must therefore
    compare the published vocabulary against what the recipe formula produces FROM THE DATA —
    a field added or dropped by hand is not a wider ledger, it is PATH A smuggling in."""
    vocab, _, builder, _ = built
    seqs_a = [s for s in _seqs() if s["group"] == "A"]
    keys = [s["customer"] for s in seqs_a]
    tuples_a = L.seqs_to_tuples(seqs_a, vocab)
    tampered = copy.copy(builder)
    tampered.keep = tuple(
        builder.keep[:-1] + (np.array([], dtype=np.int64),)
    )  # drop a field's values
    r = L.guardrails(tampered, vocab, tuples_a, _Cfg(), keys, ("nothing_named",))
    assert not r["G1_mechanical"]["working"], r["G1_mechanical"]["evidence"]
    assert "hand-picked fields: [1]" in r["G1_mechanical"]["evidence"]


def test_a_vocabulary_needs_a_training_population(built):
    """Fail safe (P7): resolving the ledger's vocabulary on zero customers would publish a
    measure view of an undefined population."""
    vocab, seqs, _, _ = built
    with pytest.raises(LookupError, match="training-population"):
        L.fit_vocabulary(seqs, vocab, [])


def test_the_wall_advertises_measure_only_when_the_producer_wrote_it():
    """The faceplate is the contract (O4): `measure` moves from not_served to emitted only when
    the ledger's own layout is passed in, and the widths it advertises are the recipe's."""
    torch = pytest.importorskip("torch")
    from looking_glass import view_layout
    from looking_glass.cfm_model import CFM

    assert "measure" in view_layout.DECLARED
    assert view_layout.PART_ORDER["measure"] == list(L.PARTS)
    torch.manual_seed(0)
    model = CFM(_vocab(), dim=16, n_experts=1)
    d_without = view_layout.describe(model)
    d_with = view_layout.describe(
        model, ledger={"slices": {"event_mass": [(0, 3)], "richness": [(3, 5)]}}
    )
    assert "measure" not in d_without["views"] and "measure" in d_without["declared_not_emitted"]
    assert d_with["views"]["measure"] == 5
    assert d_with["parts"]["measure"] == {"event_mass": 3, "richness": 2}
    assert "measure" not in d_with["declared_not_emitted"]
