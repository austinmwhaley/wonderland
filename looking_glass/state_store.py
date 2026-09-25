"""Point-in-time state store for reusable backbone representations.

The temporal core produces a customer/entity state that is only valid *as of*
the cutoff it was encoded at. To let many downstream tasks (each with its own
prediction date) reuse one backbone, those states must be stored time-indexed
and queried "as of date D" with point-in-time correctness: a query returns the
most recent state recorded at or before D, never a future one.

Every state is stamped with a ``backbone_version`` so retraining the backbone
does not silently mix incompatible representations (see :meth:`invalidate`).
Two backends share identical as-of semantics: an in-memory store (tests, small
use) and a LanceDB-backed store consistent with the project's other vectors.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StateRecord:
    """A single time-stamped entity state under one backbone version."""

    entity_id: str
    as_of_ts: datetime
    vector: list[float]
    backbone_version: str = "v0"


def _epoch(ts: datetime) -> float:
    return ts.timestamp()


def _table_names(db) -> list[str]:
    """Return table names as a plain list across lancedb versions.

    Newer lancedb returns a result object (``tables=[...] page_token=None``)
    rather than a list, so membership tests on the raw return value silently
    fail. This normalizes both shapes to ``list[str]``.
    """

    names = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    inner = getattr(names, "tables", None)
    if inner is not None:
        return [str(n) for n in inner]
    try:
        return [str(n) for n in names]
    except TypeError:  # pragma: no cover - defensive
        return []


class PointInTimeStateStore:
    """Base class defining as-of-correct query semantics over state records."""

    def __init__(self, default_version: str = "v0") -> None:
        self.default_version = default_version

    def write_states(self, records: list[StateRecord]) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def _all_for_entity(
        self, entity_id: str, backbone_version: str
    ) -> list[tuple[float, list[float]]]:  # pragma: no cover - interface
        raise NotImplementedError

    def versions(self) -> set[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def invalidate(self, backbone_version: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def get_state_as_of(
        self,
        entity_id: str,
        as_of: datetime,
        backbone_version: str | None = None,
    ) -> list[float] | None:
        """Return the latest state recorded at or before ``as_of``.

        Returns ``None`` if the entity has no state at or before the cutoff.
        """

        version = backbone_version or self.default_version
        rows = self._all_for_entity(str(entity_id), version)
        if not rows:
            return None
        rows.sort(key=lambda pair: pair[0])
        timestamps = [pair[0] for pair in rows]
        cutoff = _epoch(as_of)
        idx = bisect.bisect_right(timestamps, cutoff) - 1
        if idx < 0:
            return None
        return list(rows[idx][1])

    def get_states_as_of(
        self,
        entity_ids: list[str],
        as_of: datetime,
        backbone_version: str | None = None,
    ) -> dict[str, list[float] | None]:
        return {eid: self.get_state_as_of(eid, as_of, backbone_version) for eid in entity_ids}


class InMemoryStateStore(PointInTimeStateStore):
    """Dict-backed store: ``{version: {entity_id: [(epoch, vector), ...]}}``."""

    def __init__(self, default_version: str = "v0") -> None:
        super().__init__(default_version=default_version)
        self._data: dict[str, dict[str, list[tuple[float, list[float]]]]] = {}

    def write_states(self, records: list[StateRecord]) -> int:
        for rec in records:
            by_entity = self._data.setdefault(rec.backbone_version, {})
            by_entity.setdefault(rec.entity_id, []).append((_epoch(rec.as_of_ts), list(rec.vector)))
        return len(records)

    def _all_for_entity(
        self, entity_id: str, backbone_version: str
    ) -> list[tuple[float, list[float]]]:
        return list(self._data.get(backbone_version, {}).get(entity_id, []))

    def versions(self) -> set[str]:
        return set(self._data.keys())

    def invalidate(self, backbone_version: str) -> int:
        removed = sum(len(v) for v in self._data.get(backbone_version, {}).values())
        self._data.pop(backbone_version, None)
        return removed


class LanceDBStateStore(PointInTimeStateStore):
    """LanceDB-backed store, consistent with the project's embedding tables.

    Rows are ``{entity_id, as_of_epoch, vector, backbone_version}``. Selection
    of the as-of record is done in Python after loading the entity's rows; this
    keeps semantics identical to the in-memory store. Large-scale predicate
    pushdown is part of the streaming work item.
    """

    def __init__(
        self, lancedb_dir, table_name: str = "entity_states", default_version: str = "v0"
    ) -> None:
        super().__init__(default_version=default_version)
        from pathlib import Path

        self.lancedb_dir = Path(lancedb_dir)
        self.table_name = table_name

    def _connect(self):
        try:
            import lancedb
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "lancedb is not installed. Install with: pip install lancedb pyarrow"
            ) from exc
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self.lancedb_dir))

    def write_states(self, records: list[StateRecord]) -> int:
        if not records:
            return 0
        db = self._connect()
        rows = [
            {
                "entity_id": rec.entity_id,
                "as_of_epoch": _epoch(rec.as_of_ts),
                "vector": list(rec.vector),
                "backbone_version": rec.backbone_version,
            }
            for rec in records
        ]
        if self.table_name in _table_names(db):
            db.open_table(self.table_name).add(rows)
        else:
            db.create_table(self.table_name, data=rows)
        return len(rows)

    def _rows(self) -> list[dict]:
        db = self._connect()
        if self.table_name not in _table_names(db):
            return []
        table = db.open_table(self.table_name)
        return table.to_pylist() if hasattr(table, "to_pylist") else table.to_arrow().to_pylist()

    def _all_for_entity(
        self, entity_id: str, backbone_version: str
    ) -> list[tuple[float, list[float]]]:
        return [
            (float(r["as_of_epoch"]), list(r["vector"]))
            for r in self._rows()
            if str(r["entity_id"]) == entity_id and str(r["backbone_version"]) == backbone_version
        ]

    def versions(self) -> set[str]:
        return {str(r["backbone_version"]) for r in self._rows()}

    def invalidate(self, backbone_version: str) -> int:
        db = self._connect()
        if self.table_name not in _table_names(db):
            return 0
        table = db.open_table(self.table_name)
        before = table.count_rows()
        table.delete(f"backbone_version = '{backbone_version}'")
        return before - table.count_rows()
