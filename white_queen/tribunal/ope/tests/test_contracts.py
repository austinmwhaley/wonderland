"""Estimator contract + decision-as-inference tests (fast, no torch).

Pins the macro invariants: estimand typing, source-disjoint corroboration,
out-of-range/under-budget rejection (never silent clamping).
"""

from white_queen.tribunal.ope import contracts as K
from white_queen.tribunal.ope import decide as D


def _row(**kw):
    base = {
        "fqe_dm": 40.0,
        "sharp_dm": 40.0,
        "sharp_info": {},
        "mb": {"mb": 40.0, "se": 1.0},
        "mb_se": 1.0,
        "mb_sharp": 40.0,
        "mb_sharp_se": 1.0,
        "efqe": {"mean": 40.0, "disagreement": 1.0},
        "dr": 40.0,
        "dr_ci": [30.0, 50.0],
        "wis": 40.0,
        "is": 40.0,
        "lstdq": 40.0,
        "level_est": 40.0,
        "ess_frac": 0.2,
        "fqe_info": {},
        "dr_step_adv": 0.0,
    }
    base.update(kw)
    return base


def test_estimands_and_sources_are_tagged():
    est = {e.name: e for e in K.estimates_from_row(_row())}
    assert est["fqe_soft"].estimand == K.V_START
    assert est["fqe_soft"].sources == frozenset({K.S_QNET})
    assert est["fqe_argmax"].sources == frozenset({K.S_QNET})
    assert est["mb_argmax"].sources == frozenset({K.S_DYNAMICS})
    # DR reuses the Q net -> its source set includes qnet (not independent of FQE)
    assert K.S_QNET in est["dr"].sources
    assert est["fqe_argmax"].deployable and not est["fqe_soft"].deployable


def test_same_source_cannot_corroborate():
    # fqe_soft and fqe_argmax both clear, but share the Q net -> one source.
    row = _row(
        fqe_dm=90.0,
        sharp_dm=90.0,
        mb={"mb": 10.0, "se": 1.0},
        mb_sharp=10.0,
        dr=10.0,
        dr_ci=[0.0, 20.0],
    )
    est = K.estimates_from_row(row, bar=60.0, min_ess=0.02)
    assert K.independent_pairs(est, 60.0) == []
    assert D.decide(row, 60.0, 0.02)["deploy"] is False


def test_disjoint_sources_corroborate():
    # FQE (qnet) and MB (dynamics) are disjoint and both clear -> deploy.
    row = _row(
        fqe_dm=90.0,
        sharp_dm=95.0,
        mb={"mb": 90.0, "se": 1.0},
        mb_sharp=95.0,
        mb_sharp_se=1.0,
        ess_frac=0.0,
    )
    est = K.estimates_from_row(row, bar=60.0, min_ess=0.02)
    pairs = K.independent_pairs(est, 60.0, require_deployable=True)
    assert pairs, [(a.name, b.name) for a, b in pairs]
    assert D.decide(row, 60.0, 0.02)["deploy"] is True


def test_out_of_range_is_rejected_not_clamped():
    # value_clamp marks FQE out of range: it must be EXCLUDED, not substituted.
    row = _row(fqe_dm=1e9, sharp_dm=95.0, value_clamp={"fqe": {"raw": 1e9}})
    est = {e.name: e for e in K.estimates_from_row(row, bar=60.0, min_ess=0.02)}
    assert est["fqe_soft"].reliable is False
    assert est["fqe_argmax"].reliable is False
    assert est["fqe_argmax"].clears(60.0) is False


def test_under_budget_fqe_cannot_gate():
    row = _row(
        fqe_dm=90.0,
        sharp_dm=95.0,
        mb={"mb": 90.0, "se": 1.0},
        mb_sharp=95.0,
        fqe_info={"under_budget": True},
    )
    est = {e.name: e for e in K.estimates_from_row(row, bar=60.0, min_ess=0.02)}
    assert est["fqe_argmax"].reliable is False
    # Only MB family remains -> one source -> HOLD (needs independent agreement)
    assert D.decide(row, 60.0, 0.02)["deploy"] is False


def test_diverged_fqe_excluded_but_reason_reported():
    row = _row(
        fqe_dm=None,
        sharp_dm=None,
        sharp_info={"diverged": True, "reason": "holdout Bellman 9>8"},
        mb={"mb": 90.0, "se": 1.0},
        mb_sharp=95.0,
    )
    dv = D.decide(row, 60.0, 0.02)
    assert dv["deploy"] is False
    assert "not trustworthy" in dv["rule"] or "HOLD" in dv["rule"]
