"""Ground-truth benchmark harness tests: FP/FN/precision/recall + intervals."""

from white_queen.tribunal.bench import Case, Cell, score_case, summarize, wilson_interval


def _case(name, rows):
    # rows: (cand, deploy, truth, anchor, bar, rank)
    return Case(
        name, [Cell(c, dep, truth, anchor, bar, rk) for c, dep, truth, anchor, bar, rk in rows]
    )


def test_score_counts_fp_fn():
    # bar=60: good (truth 90) deployed = TP; bad (truth 10) deployed = FP;
    # good held = FN; bad held = TN.
    case = _case(
        "toy",
        [
            ("good_dep", True, 90.0, 50.0, 60.0, 1),
            ("bad_dep", True, 10.0, 50.0, 60.0, 2),
            ("good_hold", False, 90.0, 50.0, 60.0, 3),
            ("bad_hold", False, 10.0, 50.0, 60.0, 4),
        ],
    )
    s = score_case(case)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1)
    assert s["false_positive"] == ["bad_dep"]
    assert s["false_negative"] == ["good_hold"]
    assert 0.0 < s["rank_rho"] < 1.0  # truth ranks align with rank order mostly


def test_summarize_precision_recall_and_ci():
    cases = [
        _case("a", [("x", True, 90.0, 50.0, 60.0, 1), ("y", False, 10.0, 50.0, 60.0, 2)]),
        _case("b", [("p", True, 90.0, 50.0, 60.0, 1), ("q", True, 10.0, 50.0, 60.0, 2)]),
    ]
    out = summarize(cases)
    ov = out["overall"]
    assert ov["tp"] == 2 and ov["fp"] == 1 and ov["fn"] == 0 and ov["tn"] == 1
    assert ov["precision"] == round(2 / 3, 3)
    assert ov["recall"] == 1.0
    lo, hi = ov["precision_ci"]
    assert 0.0 <= lo <= ov["precision"] <= hi <= 1.0
    assert len(out["per_case"]) == 2


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.5
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.5 < lo < 1.0
