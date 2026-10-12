from __future__ import annotations

from rabbit_hole.lab import REQUIRED, USE_CASES, coverage_report

_FAMILIES = {
    "binary",
    "count",
    "regression",
    "multiclass",
    "representation",
    "decision",
    "ssl",
}
_KINDS = {"clf", "reg", "multi", "unsup", "ope", "ssl"}


def test_ontology_is_well_formed():
    ids = [uc.id for uc in USE_CASES]
    assert len(ids) == len(set(ids)), "use-case ids must be unique"
    for uc in USE_CASES:
        assert uc.family in _FAMILIES, uc
        assert uc.kind in _KINDS, uc
        assert set(uc.card) <= {"B", "C"} and uc.card, uc
        assert uc.baseline and uc.oracle, uc


def test_ontology_covers_the_named_families():
    fam = {uc.family for uc in USE_CASES}
    # every family the plan names must be represented
    assert {
        "binary",
        "count",
        "regression",
        "multiclass",
        "representation",
        "decision",
        "ssl",
    } <= fam
    # the plan's decision (RL/OPE) use cases must exist
    ids = {uc.id for uc in USE_CASES}
    assert {"rl_ope_incremental_margin", "optimal_action"} <= ids


def test_coverage_report_is_well_formed():
    rep = coverage_report()
    assert rep["n_use_cases"] == len(USE_CASES)
    assert 0.0 <= rep["coverage"] <= 1.0
    for uid, row in rep["rows"].items():
        assert set(row["present"]) == set(REQUIRED)
        assert 0 <= row["n_present"] <= len(REQUIRED)


def test_every_use_case_has_a_positive_spec():
    from rabbit_hole.lab import USE_CASES
    from rabbit_hole.lab_oracles import POSITIVE

    assert set(POSITIVE) == {uc.id for uc in USE_CASES}


def test_positive_and_negative_fixtures(tmp_path):
    from rabbit_hole.lab import coverage_report
    from rabbit_hole.lab_oracles import build_negative, build_positive

    uc = "next_event"
    build_positive(uc, root=str(tmp_path), n_bc=200, n_b=200)
    build_negative(uc, root=str(tmp_path), n=200)
    rep = coverage_report(tmp_path)
    present = rep["rows"][uc]["present"]
    assert present["stream"] and present["truth"] and present["oracle"] and present["baseline"]
    assert present["negative"]
