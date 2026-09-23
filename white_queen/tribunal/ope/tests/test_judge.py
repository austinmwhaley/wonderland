"""Judge tests: decision logic, truth-blindness, simulator operating point,
v9 anchor backtest. Numpy only, instant. Run: pytest .../tests/test_judge.py -q
"""
import numpy as np

from white_queen.tribunal.ope import judge as J

BEH, STD = 50.0, 20.0
GATE = {"rel_edge_std": 0.5, "min_ess_frac": 0.02}


def _row(truth=80.0, ess=0.1, dr=70.0, dr_lo=65.0, fqe=70.0, edis=1.0,
         wis=60.0):
    # NOTE: no dr_vals — mirrors real pre-v11 rows (CI-bound fallback path).
    # Tests needing the exact path set row["dr_vals"] explicitly.
    return {"fqe_dm": fqe, "sharp_dm": fqe,
            "dr": dr, "dr_ci": [dr_lo, dr + 5.0],
            "efqe": {"mean": fqe, "disagreement": edis}, "ess_frac": ess,
            "wis": wis, "is": 0.0, "wdr": 0.0, "magic": 0.0, "magic_w": [],
            "lstdq": {"dm": 0.0, "cond": 0.0}, "fve_dm": 0.0,
            "mb": {"mb": 0.0, "se": 0.0, "sims": 0}, "gdice_mis": 0.0,
            "anchor": 0.0, "slope_pick": "", "slope_val": 0.0,
            "below_anchor": [], "mis": 0.0, "mis_info": {}, "lambda_dr": 0.0,
            "temperature": 1.0, "rho_cap": 0.0, "truth": truth}


def test_witnesses():
    ok, _, _ = J.witness_dr({"dr_ci": [61.0, 70.0]}, 60.0)
    assert ok
    ok, _, _ = J.witness_dr({"dr_ci": [59.0, 90.0]}, 60.0)
    assert not ok  # lower bound rules, not the point
    ok, _, _ = J.witness_dr({"dr_ci": ["x", 1]}, 60.0)
    assert not ok  # malformed -> safe HOLD direction
    ok, _ = J.witness_fqe({"efqe": {"mean": 70.0, "disagreement": 1.0}}, 60.0)
    assert ok  # 68 > 60
    ok, _ = J.witness_fqe({"efqe": {"mean": 61.0, "disagreement": 1.0}}, 60.0)
    assert not ok


def test_bootstrap_p_and_holm():
    from white_queen.tribunal.ope.receipts import bootstrap_p, holm_reject
    rng = np.random.default_rng(0)
    assert bootstrap_p(rng.normal(100, 1, 200), 60.0) == 0.0  # all above
    assert bootstrap_p(rng.normal(0, 1, 200), 60.0) == 1.0  # all below
    p = bootstrap_p(rng.normal(60, 10, 500), 60.0)
    assert 0.3 < p < 0.7  # symmetric-ish straddles
    assert holm_reject({"a": 0.01, "b": 0.02, "c": 0.30}, 0.05) == {
        "a": True, "b": True, "c": False}  # .02 <= .05/2 rejects; .30 stops
    assert holm_reject({"a": 0.01, "b": 0.015, "c": 0.04}, 0.05) == {
        "a": True, "b": True, "c": True}  # .04 <= .05/1: full rejection


def test_exact_path_uses_holm():
    # All-clear evidence but one borderline p: Holm at α/3 must hold it back.
    good = _row(truth=90.0, dr=100.0, dr_lo=95.0, fqe=95.0)
    good["dr_vals"] = [100.0] * 60  # p=0 exactly
    mid = _row(truth=90.0, dr=100.0, dr_lo=95.0, fqe=95.0)
    mid["dr_vals"] = [100.0] * 58 + [0.0] * 2  # mean 96.7, some mass ≤ bar?
    rows = {"good": good, "mid": mid,
            "bad": _row(truth=10.0, dr=30.0, dr_lo=20.0, fqe=30.0)}
    rows["bad"]["dr_vals"] = [0.0] * 60
    rows["mid"]["dr_ci"] = [90.0, 110.0]
    # DR (Q-based) must be corroborated by the INDEPENDENT MB family.
    rows["good"]["mb"] = {"mb": 100.0, "se": 1.0}
    rows["mid"]["mb"] = {"mb": 100.0, "se": 1.0}
    v = J.judge_diet(rows, BEH, STD, GATE, 0.0)
    assert v["decisions"]["good"]["deploy"]  # p=0 rejects at any Holm step
    assert v["decisions"]["mid"]["deploy"]  # p≈0 rejects at α/2 step
    assert not v["decisions"]["bad"]["deploy"]
    # Exact path: Holm language present, no fallback note (all rows evidenced).
    assert "Holm" in v["decisions"]["good"]["reasons"][2]
    assert "pre-dr_vals" not in v["decisions"]["good"]["reasons"][2]
    # And the legacy path still annotates itself: drop one evidence array.
    rows2 = dict(rows)
    rows2["mid"] = dict(rows["mid"])
    del rows2["mid"]["dr_vals"]
    v2 = J.judge_diet(rows2, BEH, STD, GATE, 0.0)
    assert "pre-dr_vals" in v2["decisions"]["mid"]["reasons"][2]


def test_truth_blindness_and_determinism():
    rows = {"a": _row(truth=99.0), "b": _row(truth=10.0, dr=100.0,
                                             dr_lo=95.0, fqe=95.0)}
    v1 = J.judge_diet(rows, BEH, STD, GATE, 0.5)
    rows2 = {"a": _row(truth=-5.0), "b": _row(truth=500.0, dr=100.0,
                                              dr_lo=95.0, fqe=95.0)}
    v2 = J.judge_diet(rows2, BEH, STD, GATE, 0.5)
    assert v1["deployed"] == v2["deployed"]  # truth shuffled, same call
    assert J.judge_diet(rows, BEH, STD, GATE, 0.5)["deployed"] == v1["deployed"]


def test_risk_ladder_monotone():
    rows = {"a": _row(truth=90.0, dr=100.0, dr_lo=90.0, fqe=90.0),
            "b": _row(truth=85.0, dr=30.0, dr_lo=20.0, fqe=30.0)}
    rows["a"]["mb"] = {"mb": 95.0, "se": 1.0}  # independent MB corroboration
    d0 = J.judge_diet(rows, BEH, STD, GATE, 0.0)["deployed"]
    d5 = J.judge_diet(rows, BEH, STD, GATE, 0.5)["deployed"]
    d1 = J.judge_diet(rows, BEH, STD, GATE, 1.0)["deployed"]
    assert set(d1) <= set(d5) <= set(d0)  # stricter risk, fewer ships
    assert "a" in d0  # clear winner ships at aggressive


def test_simulator_operating_point():
    cards, meta = J.simulate_cards(rng=0, n_cards=2000)
    s0 = J.score_setting(cards, meta, 0.0)
    s5 = J.score_setting(cards, meta, 0.5)
    s1 = J.score_setting(cards, meta, 1.0)
    # Superiority doctrine: the operating point is the fixed rejection rule
    # (deploy when 'not-better' is rejected by a covered, corroborated test),
    # so the point no longer moves with risk; risk tunes only the DR test's
    # significance. Measured seed 0: precision .98, recall .39.
    assert s5["precision"] >= 0.90, s5
    assert s5["recall"] >= 0.25, s5
    assert s0["recall"] >= s5["recall"] >= s1["recall"], (s0, s5, s1)
    assert s0["recall"] >= 0.25
    # No single witness type may carry the decision: DR alone (with the value
    # witnesses disagreeing) must not be enough to ship.
    assert s5["precision"] >= 0.90
    print("\nsim operating:", {"agg": s0, "std": s5, "con": s1})


# v9 anchor (verdicts/v9/*.json): (diet, cand, truth, ess, dr, dr_ci, fqe,
# efqe_mean, efqe_dis, wis, mb, mb_se). Row shapes mirror REAL gate rows
# (mb flat + mb_se) so shape bugs like witness_mb-dict-only can't recur.
ANCHOR = [
    ('expert', 'iql', 98.9, 0.027, 53.0, [52.3, 54.0], 51.4, 53.2, 1.8, 15.0, 87.9, 0.74),
    ('expert', 'bc', 97.5, 0.02, 57.4, [55.7, 59.7], 52.4, 55.3, 1.8, 15.9, 99.0, 0.06),
    ('expert', 'cql', 98.9, 0.021, 66.5, [64.3, 69.3], 60.6, 61.4, 0.4, 15.3, 99.3, 0.0),
    ('mixed', 'iql', 98.6, 0.037, -1999.6, [-2883.1, -1223.7], 50.7, 47.4, 1.2, 23.4, 98.0, 0.18),
    ('mixed', 'bc', 96.6, 0.035, -260.7, [-920.3, 380.9], 38.7, 40.0, 0.2, 22.6, 98.2, 0.18),
    ('mixed', 'cql', 98.5, 0.031, 1108.3, [701.4, 1563.6], 40.5, 41.9, 1.9, 24.4, 96.9, 0.33),
    ('novice', 'iql', 81.1, 0.095, -3751.6, [-6217.7, -1538.9], 36.4, 34.2, 1.1, 23.2, 93.3, 0.0),
    ('novice', 'bc', 71.4, 0.102, 787.4, [-562.7, 2178.7], 16.3, 17.3, 0.4, 21.4, 93.3, 0.0),
    ('novice', 'cql', 77.8, 0.101, 1529.8, [-487.0, 3495.9], 23.7, 24.7, 0.2, 22.0, 93.3, 0.0),
]
ANCHOR_BEH = {"expert": (68.9, None), "mixed": (52.2, None), "novice": (24.9, None)}


# v9 resolved gate (all diets): rel_edge 0.2, min_ess 0.02. Behavior stds
# backed out of v9 bars (bar = beh + 0.2*std): expert 24.9, mixed 33.55,
# novice 25.65. Anchor rows carry NO dr_vals (like real pre-v11 rows), so the
# CI-bound fallback path is what's pinned here; the exact path is covered by
# test_exact_path_uses_holm above.
ANCHOR_STD = {"expert": 24.9, "mixed": 33.55, "novice": 25.65}
ANCHOR_GATE = {"rel_edge_std": 0.2, "min_ess_frac": 0.02}


def _anchor_rows(diet):
    rows = {}
    for d, c, t, e, dr, ci, fq, em, ed, w, mb, se in ANCHOR:
        if d == diet:
            rows[c] = _row(truth=t, ess=e, dr=dr, dr_lo=ci[0], fqe=fq,
                           wis=w)
            rows[c]["dr_ci"] = list(ci)
            rows[c]["efqe"] = {"mean": em, "disagreement": ed}
            rows[c]["mb"] = mb
            rows[c]["mb_se"] = se
    return rows


def test_anchor_backtest():
    # Bar math must be exact, and the certificate must never ship a candidate
    # whose true value fails to beat the bar (safety). Per-candidate deploy
    # expectations from the pre-certificate rule are no longer asserted: the
    # product is now the interval (lo > behavior), not a witness count.
    truth_by = {}
    for d, c, t, *_ in ANCHOR:
        truth_by[(d, c)] = t
    for diet in ("expert", "mixed", "novice"):
        beh = {"expert": 68.9, "mixed": 52.2, "novice": 24.9}[diet]
        rows = _anchor_rows(diet)
        v = J.judge_diet(rows, beh, ANCHOR_STD[diet], ANCHOR_GATE, 0.5)
        assert abs(v["bar"] - {"expert": 73.88, "mixed": 58.91,
                               "novice": 30.03}[diet]) < 0.05, v
        for cand in v["deployed"]:
            assert truth_by[(diet, cand)] > v["bar"], (diet, cand, v)
        # every certificate is well-formed
        for cand, d in v["decisions"].items():
            c = d["certificate"]
            assert "lo" in c and "hi" in c


def test_explain_is_deterministic_prose():
    rows = {"a": _row(truth=90.0, dr=100.0, dr_lo=95.0, fqe=95.0)}
    v = J.judge_diet(rows, BEH, STD, GATE, 0.5)
    t1 = J.explain_diet("d", BEH, v, rows)
    assert J.explain_diet("d", BEH, v, rows) == t1
    assert "HOLD" in t1 or "DEPLOY" in t1


def test_negative_controls_hold_at_all_risks():
    # Random-policy cards: truth BELOW behavior, every witness below bar.
    # The anchor has zero negatives (all 9 truths beat behavior) — these
    # synthetic negatives are the only false-alarm protection in CI.
    rows = {
        "rand": _row(truth=15.0, ess=0.40, dr=18.0, dr_lo=12.0, fqe=20.0,
                     wis=18.0),
        "worse": _row(truth=30.0, ess=0.30, dr=35.0, dr_lo=28.0, fqe=33.0,
                      wis=30.0),
    }
    rows["rand"]["efqe"] = {"mean": 20.0, "disagreement": 1.0}
    rows["rand"]["mb"] = {"mb": 22.0, "se": 1.0}
    rows["worse"]["efqe"] = {"mean": 33.0, "disagreement": 1.0}
    rows["worse"]["mb"] = {"mb": 30.0, "se": 1.0}
    for ra in (0.0, 0.5, 1.0):
        v = J.judge_diet(rows, BEH, STD, GATE, ra)
        assert v["deployed"] == [], (ra, v["deployed"])


def test_lone_dr_fluke_cannot_deploy():
    # mixed/cql v9 pattern: DR hallucinates +1108 (CI clears) while FQE reads
    # ~40 and MB would read ~99... here MB FAILS too, isolating the rule:
    # one loud witness must never carry a deploy at any risk level.
    rows = {
        "fluke": _row(truth=98.0, ess=0.05, dr=1108.0, dr_lo=701.0, fqe=40.0,
                      wis=24.0),
        "sane": _row(truth=60.0, ess=0.05, dr=20.0, dr_lo=10.0, fqe=30.0,
                     wis=20.0),
    }
    rows["fluke"]["efqe"] = {"mean": 40.0, "disagreement": 1.0}
    rows["fluke"]["mb"] = {"mb": 30.0, "se": 1.0}  # below bar 60
    rows["sane"]["efqe"] = {"mean": 30.0, "disagreement": 1.0}
    rows["sane"]["mb"] = {"mb": 30.0, "se": 1.0}
    # A lone loud DR witness is never sufficient at ANY risk setting: the
    # value witnesses disagree, so 'not-better' cannot be rejected (the old
    # single-witness aggressive path is gone under the superiority doctrine).
    for ra in (0.0, 0.5, 1.0):
        v = J.judge_diet(rows, BEH, STD, GATE, ra)
        assert "fluke" not in v["deployed"], (ra, v["deployed"])


def test_sharp_witness_shapes():
    from white_queen.tribunal.ope.judge import witness_sharp
    # v20: sharp = FQE of the argmax policy; no action-support requirement.
    ok, why = witness_sharp({"sharp_dm": 80.0, "sharp_info": {}}, 60.0, 0.02)
    assert ok and "80.0" in why
    ok, _ = witness_sharp({"sharp_dm": 50.0, "sharp_info": {}}, 60.0, 0.02)
    assert not ok  # below bar
    ok, why = witness_sharp({"sharp_dm": None,
                             "sharp_info": {"diverged": True, "raw": 88428.0,
                                            "bound": 150.0}}, 60.0, 0.02)
    assert not ok and "diverged" in why
    ok, why = witness_sharp({"sharp_dm": None, "sharp_info": {"skipped": "x"}}, 60.0, 0.02)
    assert not ok and "skipped" in why
    ok, _ = witness_sharp({}, 60.0, 0.02)  # absent sharp: fail safe (no deploy)
    assert not ok


def test_prescription_orders_collection():
    from white_queen.tribunal.ope.judge import prescription
    row = {"ess_frac": 0.01,
           "support": {"top_decile_obs_center": [0.2, -1.5, 3.0, 0.0]}}
    p = prescription(row, 0.02, 400)
    assert p is not None and "~800" in p and "collect:" in p
    assert prescription({"ess_frac": 0.5}, 0.02, 400) is None  # healthy: silent
    assert prescription({"ess_frac": 0.01}, 0.02, 400) is not None  # no support dict: count only


def test_support_stats_shapes():
    import numpy as np
    from white_queen.tribunal.ope.marginalized import support_stats
    rng = np.random.default_rng(0)
    N = 500
    d = {"obs": rng.normal(size=(N, 4)).astype(np.float32),
         "act": rng.integers(0, 2, N)}
    s = support_stats(d, lambda o, a: np.full((len(np.atleast_1d(a)),), 1.0))
    assert s["top_decile_share"] == 1.0  # flat: ties include every row
    assert len(s["top_decile_obs_center"]) == 4
    n = len(d["act"])
    s1 = support_stats(d, lambda o, a: np.linspace(0.01, 2.0, n))
    assert 0.1 < s1["top_decile_share"] < 0.3  # graded weights: sane share
    s2 = support_stats(d, lambda o, a: (np.asarray(a) == 0).astype(float) * 100.0 + 0.01)
    assert s2["top_decile_share"] > 0.5  # degenerate emphasis detected


def test_lone_candidate_requires_both_argmax_witnesses():
    # v18 forensics: mixed-uniform (truth 9.1) deployed solo on soft-policy
    # witnesses. Lone judging now requires BOTH independent methods on the
    # DEPLOYABLE (argmax) estimand: FQE-argmax and MB-argmax. A random policy
    # reads negative/absent on both -> HOLD. A good policy with both -> deploy.
    bad = _row(truth=9.1, ess=0.044, dr=124.0, dr_lo=53.9, fqe=18.7, wis=23.0)
    bad["dr_ci"] = [53.9, 202.6]
    bad["mb"] = {"mb": 99.3, "se": 0.0}
    bad["efqe"] = {"mean": 18.7, "disagreement": 0.1}
    bad["sharp_dm"] = -25.0
    bad["mb_sharp"] = 12.0
    bad["mb_sharp_se"] = 1.0
    v = J.judge_diet({"uniform": bad}, 52.2, 33.55,
                     {"rel_edge_std": 0.2, "min_ess_frac": 0.02}, 0.5)
    assert v["lone_candidate"] is True
    assert v["deployed"] == [], v
    # Good policy: both argmax witnesses clear the bar -> DEPLOY solo.
    good = _row(truth=94.0, ess=0.2, dr=90.0, dr_lo=80.0, fqe=55.0, wis=60.0)
    good["dr_ci"] = [80.0, 100.0]
    good["mb"] = {"mb": 55.0, "se": 1.0}
    good["efqe"] = {"mean": 55.0, "disagreement": 1.0}
    good["sharp_dm"] = 90.0
    good["mb_sharp"] = 92.0
    good["mb_sharp_se"] = 1.5
    v2 = J.judge_diet({"cand": good}, 52.2, 33.55,
                      {"rel_edge_std": 0.2, "min_ess_frac": 0.02}, 0.5)
    assert v2["deployed"] == ["cand"], v2
    assert any("single-candidate" in r for r in v2["decisions"]["cand"]["reasons"])


def test_same_method_family_cannot_double_count():
    # FQE-soft and FQE-argmax share one Q net, so they are ONE method. Two FQE
    # variants clearing the bar while the independent MB family disagrees must
    # NOT deploy. This is the MountainCar false positive: FQE read -36 while
    # truth was -86, and counting soft+argmax FQE as two witnesses shipped it.
    rows = {
        "fluke": _row(truth=-86.0, ess=0.07, dr=-60.0, dr_lo=-88.0, fqe=-30.0),
        "peer": _row(truth=-86.0, ess=0.07, dr=-60.0, dr_lo=-88.0, fqe=-70.0),
    }
    for r in rows.values():
        r["mb"] = {"mb": -86.0, "se": 1.0}      # MB family disagrees
        r["mb_se"] = 1.0
    rows["fluke"]["sharp_dm"] = -30.0           # FQE-argmax also optimistic
    rows["fluke"]["sharp_info"] = {}
    rows["peer"]["sharp_dm"] = -70.0
    rows["peer"]["sharp_info"] = {}
    # Behavior ~ -86.6, tiny std -> bar ~ -86.1.
    v = J.judge_diet(rows, -86.6, 1.0,
                     {"rel_edge_std": 0.5, "min_ess_frac": 0.02}, 0.5)
    assert v["decisions"]["fluke"]["deploy"] is False, v["decisions"]["fluke"]
    assert v["decisions"]["fluke"]["fqe_pass"] is True  # FQE-soft passed
    assert v["decisions"]["fluke"]["sharp_pass"] is True  # FQE-argmax passed
    assert v["decisions"]["fluke"]["mb_sharp_pass"] is False  # MB disagreed
