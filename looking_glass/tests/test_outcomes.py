from datetime import datetime, timedelta, timezone

import pytest

from looking_glass import LabelSpec, assert_no_leakage, build_outcomes

AS_OF = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _ev(cid, days_from_as_of, value=10.0):
    ts = (AS_OF + timedelta(days=days_from_as_of)).isoformat()
    return {"customer_id": cid, "event_ts": ts, "value": value}


def _spec(**kw):
    base = dict(
        id_field="customer_id",
        timestamp_field="event_ts",
        as_of=AS_OF,
        history_days=365,
        horizon_days=120,
        value_field="value",
        label_kind="value_sum",
        min_history_events=2,
    )
    base.update(kw)
    return LabelSpec(**base)


def test_churn_and_value_labels():
    events = [
        _ev("a", -100), _ev("a", -50), _ev("a", 30, value=25.0),   # has future -> churn 0
        _ev("b", -100), _ev("b", -20),                              # no future  -> churn 1
    ]
    frame = build_outcomes(events, _spec())
    by_id = {str(r["customer_id"]): r for r in frame.rows}
    assert by_id["a"]["churn_label"] == 0.0
    assert by_id["b"]["churn_label"] == 1.0
    assert by_id["a"]["value_label"] == 25.0
    assert by_id["b"]["value_label"] == 0.0
    # Features use past only: 'a' had 2 past events (future not counted).
    assert by_id["a"]["event_count"] == 2.0


def test_eligibility_filter_drops_thin_entities():
    events = [_ev("a", -100), _ev("a", -50), _ev("c", -10)]  # c has 1 past event
    frame = build_outcomes(events, _spec(min_history_events=2))
    ids = {str(r["customer_id"]) for r in frame.rows}
    assert "a" in ids and "c" not in ids


def test_events_outside_windows_ignored():
    events = [
        _ev("a", -400), _ev("a", -100), _ev("a", -50),  # -400 outside history window
        _ev("a", 200),                                   # outside horizon window
    ]
    frame = build_outcomes(events, _spec())
    row = frame.rows[0]
    assert row["event_count"] == 2.0  # only -100 and -50
    assert row["churn_label"] == 1.0  # 200 is outside horizon -> counts as no future


def test_assert_no_leakage_passes_for_built_frame():
    events = [_ev("a", -100), _ev("a", -50), _ev("a", 30)]
    spec = _spec()
    frame = build_outcomes(events, spec)
    assert_no_leakage(events, spec, frame)  # should not raise


def test_assert_no_leakage_detects_tampering():
    events = [_ev("a", -100), _ev("a", -50), _ev("a", 30)]
    spec = _spec()
    frame = build_outcomes(events, spec)
    frame.rows[0]["event_count"] = 99.0  # simulate a leaked/future-contaminated feature
    with pytest.raises(AssertionError):
        assert_no_leakage(events, spec, frame)


def test_value_sum_requires_value_field():
    with pytest.raises(ValueError):
        LabelSpec(id_field="c", timestamp_field="t", as_of=AS_OF, label_kind="value_sum")
