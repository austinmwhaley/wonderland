"""Equivalence gate for the opt-in sequence cache (DEC-669, engineering G(iii)).

The cache is admissible only if a hit returns the exact object a rebuild produces
(including the cfg-side derived state downstream reads), a changed input MISSES,
and the default path executes no cache machinery. These are the four acceptance
cases plus the default-path proof.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from looking_glass import cfm_data
from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_data import build_sequences

CUSTOMERS = ("c1", "c2", "c3", "c4")
SPLIT = {"c1": "A", "c2": "B", "c3": "A", "c4": "B"}


def _frame(n_events: int = 10) -> pl.DataFrame:
    """Tiny sorted synthetic stream: 4 customers, mixed event types/fields."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for ci, k in enumerate(CUSTOMERS):
        for i in range(n_events):
            ts = base + timedelta(days=3 * i, hours=ci)
            is_mail = i % 4 == 2
            is_buy = i % 4 == 1
            rows.append(
                {
                    "customer_key": k,
                    "event_ts": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "event_type": "buy" if is_buy else ("email" if is_mail else "view"),
                    "channel": "email" if is_mail else None,
                    "freq": float(1 + i % 3) if is_mail else None,
                    "value": 19.99 if is_buy else 0.0,
                }
            )
    return pl.DataFrame(rows).sort(["customer_key", "event_ts"])


def _cfg(**overrides) -> CFMConfig:
    cfg = CFMConfig(
        seq_len=4,
        n_anchors=4,
        split_seed=7,
        as_of="2025-02-01",
        numeric_fields=("value",),
        attr_fields=("channel",),
        action_fields=("freq",),
        effect_actions=("freq",),
        survival=True,
        action_profile=True,
        action_profile_fields=("freq",),
        primary_numeric="value",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _enable(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv(cfm_data._SEQ_CACHE_ENV, "1")
    monkeypatch.setenv(cfm_data._SEQ_CACHE_DIR_ENV, str(tmp_path / "cache"))
    return tmp_path / "cache"


def _assert_exact(a, b, path: str = "value") -> None:
    """Exact recursive equality: dtype, key order, every element (no allclose)."""
    assert type(a) is type(b), f"{path}: type {type(a)} != {type(b)}"
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        if a.dtype == object:
            assert a.tolist() == b.tolist(), path
        else:
            assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys/order {list(a)} != {list(b)}"
        for k in a:
            _assert_exact(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_exact(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_seqs_equal(a, b) -> None:
    assert len(a) == len(b), f"n_seqs {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        _assert_exact(x, y, f"seqs[{i}]")


def _assert_cfg_derived_equal(a: CFMConfig, b: CFMConfig) -> None:
    assert a._effect_levels == b._effect_levels
    assert a._effect_dim == b._effect_dim
    assert a._action_receipt == b._action_receipt
    assert getattr(a, "_survival_cache", None) == getattr(b, "_survival_cache", None)
    assert getattr(a, "_survival_horizons_s", None) == getattr(b, "_survival_horizons_s", None)
    assert getattr(a, "_profile_cache", None) == getattr(b, "_profile_cache", None)
    assert getattr(a, "_action_profile_levels", None) == getattr(b, "_action_profile_levels", None)
    assert getattr(a, "_action_profile_dims", None) == getattr(b, "_action_profile_dims", None)
    per_a, levels_a = a._effect_cache
    per_b, levels_b = b._effect_cache
    assert levels_a == levels_b
    if per_a is None:
        assert per_b is None
        return
    assert list(per_a) == list(per_b)
    for k in per_a:
        for x, y in zip(per_a[k], per_b[k]):
            assert np.array_equal(x, y)


# ---------------------------------------------------------------------------
# case 1 — baseline determinism, and the default path runs NO cache code
# ---------------------------------------------------------------------------
def test_default_path_deterministic_and_cache_free(tmp_path, monkeypatch):
    monkeypatch.delenv(cfm_data._SEQ_CACHE_ENV, raising=False)
    monkeypatch.setenv(cfm_data._SEQ_CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "system-temp"))
    monkeypatch.setattr(cfm_data, "_SEQ_CACHE_LAST", {})

    def _boom(*_a, **_k):  # any cache key computation on the default path is a bug
        raise AssertionError("cache key machinery ran with WONDERLAND_SEQ_CACHE unset")

    monkeypatch.setattr(cfm_data, "_seq_cache_key", _boom)

    df = _frame()
    a = build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    b = build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)

    assert a, "fixture produced no sequences"
    assert any(s["anchor_epoch"] is not None for s in a), "fixture produced no anchors"
    _assert_seqs_equal(a, b)
    assert cfm_data._SEQ_CACHE_LAST == {}, "the default path must not touch the cache"
    assert not (tmp_path / "cache").exists(), "the default path must not create a cache dir"
    assert not (tmp_path / "system-temp").exists(), "no cache dir was even considered"


# ---------------------------------------------------------------------------
# case 2 — miss then hit, and the hit is byte-identical to a fresh rebuild
# ---------------------------------------------------------------------------
def test_hit_is_exactly_a_rebuild(tmp_path, monkeypatch):
    monkeypatch.delenv(cfm_data._SEQ_CACHE_ENV, raising=False)
    df = _frame()

    baseline_cfg = _cfg()
    baseline = build_sequences(df, list(CUSTOMERS), baseline_cfg, SPLIT, with_anchors=True)

    cache_dir = _enable(tmp_path, monkeypatch)
    first_cfg = _cfg()
    first = build_sequences(df, list(CUSTOMERS), first_cfg, SPLIT, with_anchors=True)
    receipt = dict(cfm_data._SEQ_CACHE_LAST)
    assert receipt["status"] == "miss"
    assert receipt["stored"] is True
    miss_key = receipt["key"]
    assert list(cache_dir.glob("*.pkl")), "a miss must store an entry"
    _assert_seqs_equal(baseline, first)

    # same cfg object: its derived state equals the stored cold state -> a real hit
    again = build_sequences(df, list(CUSTOMERS), first_cfg, SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "hit"
    _assert_seqs_equal(baseline, again)

    second_cfg = _cfg()
    second = build_sequences(df, list(CUSTOMERS), second_cfg, SPLIT, with_anchors=True)
    receipt = dict(cfm_data._SEQ_CACHE_LAST)
    assert receipt["status"] == "hit"
    assert receipt["key"] == miss_key

    _assert_seqs_equal(baseline, second)
    _assert_cfg_derived_equal(baseline_cfg, second_cfg)


# ---------------------------------------------------------------------------
# case 3 — every changed input MISSES (no stale object is ever served)
# ---------------------------------------------------------------------------
def test_changed_input_changes_key_and_misses(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    df = _frame()
    build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    base_key = cfm_data._SEQ_CACHE_LAST["key"]
    base_anchors = {
        s["customer"]: s["anchor_epoch"]
        for s in build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
        if s["anchor_epoch"] is not None
    }
    assert base_anchors, "fixture produced no anchors"

    # (a) one more event on the stream
    extra = pl.DataFrame(
        {
            "customer_key": ["c2"],
            "event_ts": ["2025-02-01T00:00:00+00:00"],
            "event_type": ["buy"],
            "channel": [None],
            "freq": [None],
            "value": [5.0],
        }
    )
    df2 = pl.concat([df, extra]).sort(["customer_key", "event_ts"])
    build_sequences(df2, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "miss"
    assert cfm_data._SEQ_CACHE_LAST["key"] != base_key

    # (b) as_of changes (same frame -> construct-equivalent, but a new address)
    build_sequences(df, list(CUSTOMERS), _cfg(as_of="2025-03-01"), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "miss"
    assert cfm_data._SEQ_CACHE_LAST["key"] != base_key

    # (c) the anchor seed changes -> new key, and the anchors actually move
    seqs = build_sequences(df, list(CUSTOMERS), _cfg(split_seed=11), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "miss"
    assert cfm_data._SEQ_CACHE_LAST["key"] != base_key
    moved = {s["customer"]: s["anchor_epoch"] for s in seqs if s["anchor_epoch"] is not None}
    assert moved and moved != base_anchors


# ---------------------------------------------------------------------------
# case 4 — a corrupt entry falls back to a rebuild, never raises
# ---------------------------------------------------------------------------
def test_corrupt_entry_falls_back_to_rebuild(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    df = _frame()
    baseline = build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    path = Path(cfm_data._SEQ_CACHE_LAST["path"])
    assert path.exists()
    path.write_bytes(b"\x00not a pickle\xff")

    rebuilt = build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "miss_corrupt"
    assert cfm_data._SEQ_CACHE_LAST["stored"] is True
    _assert_seqs_equal(baseline, rebuilt)

    # the corrupt entry was atomically replaced by a valid one: the next call hits
    with open(path, "rb") as f:
        assert isinstance(pickle.load(f), dict)
    build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "hit"


# ---------------------------------------------------------------------------
# the effect period rows are canonical (this was NOT deterministic pre-fix)
# ---------------------------------------------------------------------------
def _effect_frame() -> pl.DataFrame:
    """One action event per period per customer, level == period index + 1."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for ci, k in enumerate(CUSTOMERS):
        for i in range(6):
            ts = base + timedelta(days=8 * i, hours=ci)
            rows.append(
                {
                    "customer_key": k,
                    "event_ts": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "event_type": "email",
                    "channel": "email",
                    "freq": float(i + 1),
                    "value": 1.0,
                }
            )
    return pl.DataFrame(rows).sort(["customer_key", "event_ts"])


def test_effect_period_order_is_canonical():
    from looking_glass.cfm_data import _effect_targets

    df = _effect_frame()
    per, levels = _effect_targets(df, _cfg(effect_actions=("freq",)))
    assert levels == [7]
    want = np.arange(1, 7, dtype=np.float32)
    for k in CUSTOMERS:
        act = per[k][0][:, 0]
        assert np.array_equal(act, want), f"{k}: effect periods are not period-sorted ({act})"


# ---------------------------------------------------------------------------
# the effect path is cacheable AND deterministic across calls
# ---------------------------------------------------------------------------
def test_effect_path_deterministic_and_cached(tmp_path, monkeypatch):
    cache_dir = _enable(tmp_path, monkeypatch)
    df = _frame()
    a = build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "miss"
    assert cfm_data._SEQ_CACHE_LAST["stored"] is True
    b = build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "hit"
    _assert_seqs_equal(a, b)
    assert any(s["eff_act"] is not None for s in b)
    assert list(cache_dir.glob("*.pkl"))


# ---------------------------------------------------------------------------
# the key carries every accepted construction input (a receipt, not a promise)
# ---------------------------------------------------------------------------
def test_key_fields_cover_the_construction_inputs(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    df = _frame()
    build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    fields = cfm_data._SEQ_CACHE_LAST["key_fields"]
    for name in (
        "schema",
        "source",
        "constants",
        "runtime",
        "data",
        "stream",
        "as_of",
        "keys",
        "n_keys",
        "split",
        "seq_len",
        "n_anchors",
        "seed",
        "with_anchors",
        "min_events",
        "primary_numeric",
        "attr_fields",
        "numeric_fields",
        "action_fields",
        "effect_actions",
        "exogenous_events",
        "survival",
        "action_profile",
        "action_profile_fields",
    ):
        assert name in fields, f"cache key is missing construction input {name!r}"


# ---------------------------------------------------------------------------
# the source stream's file identity is in the key (path/size/mtime)
# ---------------------------------------------------------------------------
def test_stream_file_identity_is_keyed(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    src = tmp_path / "stream.parquet"
    src.write_bytes(b"not actually parquet: only its identity is read here")
    df = _frame()

    build_sequences(df, list(CUSTOMERS), _cfg(db=str(src)), SPLIT, with_anchors=True)
    first = cfm_data._SEQ_CACHE_LAST["key"]
    build_sequences(df, list(CUSTOMERS), _cfg(db=str(src)), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "hit"

    # same frame, changed mtime -> a new address, so the entry is not served
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    build_sequences(df, list(CUSTOMERS), _cfg(db=str(src)), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] == "miss"
    assert cfm_data._SEQ_CACHE_LAST["key"] != first


# ---------------------------------------------------------------------------
# default location: a per-uid dir under the system temp
# ---------------------------------------------------------------------------
def test_default_cache_dir_is_per_uid(tmp_path, monkeypatch):
    monkeypatch.setenv(cfm_data._SEQ_CACHE_ENV, "1")
    monkeypatch.delenv(cfm_data._SEQ_CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    df = _frame()
    build_sequences(df, list(CUSTOMERS), _cfg(), SPLIT, with_anchors=True)
    assert cfm_data._SEQ_CACHE_LAST["status"] in ("miss", "hit")
    uid = str(os.getuid()) if hasattr(os, "getuid") else None
    expected = {
        p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("wonderland-seq-cache-")
    }
    assert expected, "no per-uid cache dir created under the system temp"
    if uid is not None:
        assert (tmp_path / f"wonderland-seq-cache-{uid}").is_dir()
