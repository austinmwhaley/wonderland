"""Schema-derived field-role tests: no business field names may be consulted."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "demo"))
from stream_loader import derive_identity_fields  # noqa: E402


def _fixture():
    """3 entities x 4 events. Field 0 is constant per entity (identity);
    field 1 varies within every entity (behavioural)."""
    gcat = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ],
        dtype=np.int64,
    )
    group = np.repeat(np.arange(3, dtype=np.int64), 4)
    catspec = [{"field": "anonymous_a", "size": 4}, {"field": "anonymous_b", "size": 8}]
    return gcat, group, catspec


def test_identity_field_is_detected_by_behaviour_not_name():
    gcat, group, catspec = _fixture()
    idx = derive_identity_fields(gcat, group, 3, catspec)
    assert idx == [0]


def test_field_names_are_irrelevant():
    gcat, group, catspec = _fixture()
    renamed = [{"field": "segment", "size": 4}, {"field": "region", "size": 8}]
    # Names that *look* like identity fields must not change the answer: field 1
    # varies within entity, so it is behavioural regardless of its name.
    assert derive_identity_fields(gcat, group, 3, renamed) == [0]


def test_pathological_single_value_field_is_identity():
    gcat = np.ones((6, 1), dtype=np.int64)
    group = np.repeat(np.arange(2, dtype=np.int64), 3)
    catspec = [{"field": "anything", "size": 3}]
    assert derive_identity_fields(gcat, group, 2, catspec) == [0]


def test_all_missing_field_is_skipped():
    gcat = np.zeros((6, 1), dtype=np.int64)
    group = np.repeat(np.arange(2, dtype=np.int64), 3)
    catspec = [{"field": "empty", "size": 3}]
    assert derive_identity_fields(gcat, group, 2, catspec) == []
