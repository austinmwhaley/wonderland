from datetime import datetime, timezone

import pytest

from looking_glass import InMemoryStateStore, StateRecord

T1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2024, 2, 1, tzinfo=timezone.utc)
T3 = datetime(2024, 3, 1, tzinfo=timezone.utc)


def _store():
    s = InMemoryStateStore(default_version="v1")
    s.write_states(
        [
            StateRecord("a", T1, [1.0, 1.0], "v1"),
            StateRecord("a", T2, [2.0, 2.0], "v1"),
            StateRecord("a", T3, [3.0, 3.0], "v1"),
        ]
    )
    return s


def test_as_of_returns_latest_at_or_before():
    s = _store()
    assert s.get_state_as_of("a", datetime(2024, 2, 15, tzinfo=timezone.utc)) == [2.0, 2.0]
    assert s.get_state_as_of("a", T2) == [2.0, 2.0]  # inclusive at boundary
    assert s.get_state_as_of("a", T3) == [3.0, 3.0]


def test_as_of_before_first_is_none():
    s = _store()
    assert s.get_state_as_of("a", datetime(2023, 12, 1, tzinfo=timezone.utc)) is None


def test_unknown_entity_is_none():
    s = _store()
    assert s.get_state_as_of("zzz", T3) is None


def test_version_isolation_and_invalidate():
    s = InMemoryStateStore()
    s.write_states([StateRecord("a", T1, [1.0], "v1"), StateRecord("a", T1, [9.0], "v2")])
    assert s.get_state_as_of("a", T2, backbone_version="v1") == [1.0]
    assert s.get_state_as_of("a", T2, backbone_version="v2") == [9.0]
    assert s.versions() == {"v1", "v2"}
    removed = s.invalidate("v1")
    assert removed == 1
    assert s.get_state_as_of("a", T2, backbone_version="v1") is None
    assert s.get_state_as_of("a", T2, backbone_version="v2") == [9.0]


def test_batch_lookup():
    s = _store()
    s.write_states([StateRecord("b", T1, [5.0, 5.0], "v1")])
    out = s.get_states_as_of(["a", "b", "c"], T3)
    assert out["a"] == [3.0, 3.0]
    assert out["b"] == [5.0, 5.0]
    assert out["c"] is None


def test_lancedb_store_roundtrip(tmp_path):
    pytest.importorskip("lancedb")
    from looking_glass import LanceDBStateStore

    s = LanceDBStateStore(tmp_path / "states", default_version="v1")
    s.write_states([StateRecord("a", T1, [1.0, 1.0], "v1"), StateRecord("a", T3, [3.0, 3.0], "v1")])
    assert s.get_state_as_of("a", T2) == [1.0, 1.0]
    assert s.get_state_as_of("a", T3) == [3.0, 3.0]
