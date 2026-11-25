"""Unit tests for the served recency block (`build_bank_e.elapsed_features`).

The block is the DEC-667 pre-registration: per-type counts in the frozen 7 / 30 / 90-day
windows plus elapsed times, with types derived from the stream's own vocabulary and no fitted
parameter. The tests pin the properties the measurement depends on — inclusive window
boundaries, censoring at the stream's history window, a customer with no events getting the
censored value, frequency-derived types, data-derived width, causality, and determinism —
on a synthetic frame only. No real artifact is read.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_bank_e import ELAPSED_WINDOWS_DAYS, elapsed_features  # noqa: E402

AS_OF = np.datetime64("2025-01-01")
WINDOW_DAYS = 400.0
CENSORED = float(np.log1p(WINDOW_DAYS))
KEYS = ["customer_a", "customer_c", "customer_b", "customer_d"]


def _ts(days_before: float) -> str:
    return (dt.datetime(2025, 1, 1) - dt.timedelta(days=days_before)).isoformat(sep=" ")


def _frame(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_key": [r[0] for r in rows],
            "event_type": [r[1] for r in rows],
            "event_ts": [r[2] for r in rows],
        }
    )


def _synthetic() -> pl.DataFrame:
    """view (10 events) > purchase (3) > search (2) > email (1); one customer has none.

    customer_a carries events at exactly 0/7/30/90 days (inclusive boundaries) and just past
    them (7.5/30.5/90.5, excluded), one at 400 (the censoring boundary), one at 401 (censored
    elapsed, out of every window) and future events (must be invisible).
    """
    rows = [("customer_a", "view", _ts(d)) for d in (0, 7, 7.5, 30, 30.5, 90, 90.5, 400, 401)]
    rows += [("customer_a", "purchase", _ts(d)) for d in (3, 8, 400)]
    rows += [("customer_a", "search", _ts(2))]
    rows += [("customer_a", "purchase", _ts(-2))]  # future: ignored
    rows += [("customer_a", "view", _ts(-1))]  # future: ignored
    rows += [("customer_b", "search", _ts(1))]
    rows += [("customer_d", "view", _ts(401)), ("customer_d", "email", _ts(5))]
    return _frame(rows)


def _row(out: np.ndarray, key: str) -> np.ndarray:
    return out[KEYS.index(key)]


def test_counts_at_inclusive_boundaries() -> None:
    out, meta = elapsed_features(_synthetic(), KEYS, AS_OF, WINDOW_DAYS, k_types=2)
    assert meta["types"] == ["view", "purchase"]
    assert meta["windows"] == [7.0, 30.0, 90.0]
    assert meta["dim"] == out.shape[1] == 1 + 2 * (1 + 3)
    a = _row(out, "customer_a")
    # exactly 7/30/90 are IN; 7.5/30.5/90.5 are OUT; 401 is out of every window.
    assert a[3] == np.log1p(2)  # view: 0, 7
    assert a[4] == np.log1p(4)  # view: 0, 7, 7.5, 30
    assert a[5] == np.log1p(6)  # view: + 30.5, 90
    assert a[6] == np.log1p(1)  # purchase: 3
    assert a[7] == np.log1p(2)  # purchase: 3, 8
    assert a[8] == np.log1p(2)  # purchase: 400 is out of the 90-day window


def test_elapsed_and_censoring() -> None:
    out, meta = elapsed_features(_synthetic(), KEYS, AS_OF, WINDOW_DAYS, k_types=2)
    a = _row(out, "customer_a")
    assert a[0] == np.log1p(0)  # last event is at as_of
    assert a[1] == np.log1p(0)  # last view is at as_of
    assert a[2] == np.log1p(3)  # last purchase
    d = _row(out, "customer_d")
    assert d[0] == np.log1p(5)  # last event is the 5-day-old email (not a top type)
    # last view is 401 days old: elapsed is censored at the 400-day history window, and the
    # event contributes to no window.
    assert d[1] == CENSORED
    assert d[3] == d[4] == d[5] == np.log1p(0)
    assert meta["censored_days"] == WINDOW_DAYS


def test_customer_with_no_events_gets_censored_value() -> None:
    out, _ = elapsed_features(_synthetic(), KEYS, AS_OF, WINDOW_DAYS, k_types=2)
    c = _row(out, "customer_c")
    assert np.all(c[:3] == CENSORED)
    assert np.all(c[3:] == np.log1p(0))
    # customer_b has only a non-top-type event: last-event elapsed is known, type rows censored.
    b = _row(out, "customer_b")
    assert b[0] == np.log1p(1)
    assert np.all(b[1:3] == CENSORED)
    assert np.all(b[3:] == np.log1p(0))


def test_types_are_derived_by_frequency_and_width_follows_the_stream() -> None:
    frame = _synthetic()
    _, wide = elapsed_features(frame, KEYS, AS_OF, WINDOW_DAYS, k_types=5)
    # only four types exist; the width is the actual vocabulary, not the requested k.
    assert wide["types"] == ["view", "purchase", "search", "email"]
    assert wide["dim"] == 1 + 4 * (1 + 3)
    # ties are broken by name so the block is reproducible.
    tie = _frame([("k", "b_type", _ts(1)), ("k", "a_type", _ts(1))])
    _, tie_meta = elapsed_features(tie, ["k"], AS_OF, WINDOW_DAYS, k_types=5)
    assert tie_meta["types"] == ["a_type", "b_type"]


def test_windows_clip_to_a_shorter_history_window() -> None:
    out, meta = elapsed_features(_synthetic(), KEYS, AS_OF, 30.0, k_types=2)
    # 90 is clipped to 30 and de-duplicated: the block carries only observable windows.
    assert meta["windows"] == [7.0, 30.0]
    assert meta["windows_pre_registered"] == list(ELAPSED_WINDOWS_DAYS)
    assert meta["dim"] == out.shape[1] == 1 + 2 * (1 + 2)
    assert meta["censored_days"] == 30.0
    a = _row(out, "customer_a")
    assert a[3] == np.log1p(2)
    assert a[4] == np.log1p(4)
    d = _row(out, "customer_d")
    assert d[1] == np.log1p(30.0)


def test_output_is_deterministic_and_row_order_independent() -> None:
    frame = _synthetic()
    out1, meta1 = elapsed_features(frame, KEYS, AS_OF, WINDOW_DAYS, k_types=2)
    out2, meta2 = elapsed_features(frame.reverse(), KEYS, AS_OF, WINDOW_DAYS, k_types=2)
    assert meta1 == meta2
    np.testing.assert_array_equal(out1, out2)
    out3, _ = elapsed_features(frame, KEYS, AS_OF, WINDOW_DAYS, k_types=2)
    np.testing.assert_array_equal(out1, out3)
    assert out1.dtype == np.float64
