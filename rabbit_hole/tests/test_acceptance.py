"""rabbit_hole acceptance: one unified stream from wide tables, realistic and
consumable, with looking_glass pointing at rabbit_hole (no duplication)."""

from rabbit_hole.acceptance import check, check_no_duplication


def test_all_acceptance_checks_pass():
    _rows, checks = check()
    checks = checks + check_no_duplication()
    failed = [c for c in checks if not c["ok"]]
    assert not failed, failed


def test_stream_is_realistic():
    _rows, checks = check()
    by = {c["check"]: c for c in checks}
    assert by["produced events"]["ok"]
    assert by["chronological per customer"]["ok"]
    assert by["orders preceded by add_to_cart"]["ok"]
    assert by["signup attr diversity: loyalty_tier"]["ok"]
