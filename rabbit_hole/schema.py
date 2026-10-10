"""Canonical unified customer event stream schema (Layer A).

rabbit_hole ends at ONE canonical, append-only event stream. The canonical
fields are exactly the fields that **every** input stream is guaranteed to have
(so downstream consumers can rely on them across streams):

    customer_key, event_ts, event_type, source_table, event_attributes

Everything else a producer happens to carry — ``brand``, ``entity_type``,
``entity_id``, ``value``, ``category_id``, ``user_session``, ``event_id``, … —
is **not** universal, so it lives inside ``event_attributes`` (a schema-evolved
JSON map). ``source_table`` is canonical because every stream declares its
origin. ``brand`` is explicitly *not* canonical (many streams have no brand).

Incoming rows are canonicalized through :data:`COLUMN_ALIASES`; any
non-canonical column is folded into ``event_attributes``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

# Fields present in EVERY input stream. Keep this minimal.
CANONICAL_FIELDS = (
    "customer_key",
    "event_ts",
    "event_type",
    "source_table",
    "event_attributes",
)

# Well-known non-canonical fields (folded into event_attributes). Not exhaustive:
# canonicalize_row folds ANY non-canonical column, this list is documentation.
FOLDED_FIELDS = ("brand", "event_id", "entity_type", "entity_id", "value")

EVENT_STREAM_TABLE = "customer_events"
CANONICAL_VIEW = "customer_events_canonical"

# Accepted source column names -> canonical field.
COLUMN_ALIASES = {
    "customer_key": ("customer_key", "customer_id", "cust_id", "customer", "user_id"),
    "event_ts": ("event_ts", "timestamp", "ts", "event_time", "event_timestamp"),
    "event_type": ("event_type", "type", "event_name"),
    "source_table": ("source_table", "source", "stream", "stream_name"),
    "event_attributes": (
        "event_attributes",
        "event_payload_json",
        "payload_json",
        "payload",
        "attributes",
        "event_payload",
    ),
}


@dataclass(frozen=True)
class EventRecord:
    customer_key: Any
    event_ts: Any
    event_type: str
    source_table: Any
    event_attributes: Mapping[str, Any]

    def to_row(self) -> dict:
        attrs = self.event_attributes
        if not isinstance(attrs, str):
            attrs = json.dumps(attrs, separators=(",", ":"))
        return {
            "customer_key": self.customer_key,
            "event_ts": self.event_ts,
            "event_type": self.event_type,
            "source_table": self.source_table,
            "event_attributes": attrs,
        }


class SchemaError(ValueError):
    """Raised when a row cannot be canonicalized."""


def _detect(cols, canonical):
    for alias in COLUMN_ALIASES[canonical]:
        if alias in cols:
            return alias
    return None


def canonicalize_row(row: Mapping[str, Any], *, strict: bool = False) -> dict:
    """Map an arbitrary event row onto the canonical five fields.

    Missing ``source_table``/``event_attributes`` become ``None``/``{}``; a
    missing ``customer_key``/``event_ts`` raises when ``strict``. Every
    non-canonical column (brand, entity_id, value, category_id, …) is folded into
    ``event_attributes`` so no information is dropped.
    """
    cols = set(row.keys())
    out = {}
    consumed = set()
    for field in CANONICAL_FIELDS:
        src = _detect(cols, field)
        out[field] = row.get(src) if src is not None else None
        if src is not None:
            consumed.add(src)
    if out["customer_key"] in (None, "") or out["event_ts"] in (None, ""):
        if strict:
            raise SchemaError("missing customer_key or event_ts")
    # Fold EVERY non-canonical column into the attribute map.
    extras = {}
    for k in cols:
        if k in consumed or k in CANONICAL_FIELDS:
            continue
        v = row.get(k)
        if v is not None:
            extras[k] = v
    if extras:
        attrs = parse_attributes(out["event_attributes"])
        attrs.update(extras)
        out["event_attributes"] = json.dumps(attrs, separators=(",", ":"))
    if out["event_attributes"] is None:
        out["event_attributes"] = "{}"
    return out


def validate_event(row: Mapping[str, Any], *, strict: bool = False) -> dict:
    """Validate a canonical event row (keys already canonical)."""
    missing = [f for f in CANONICAL_FIELDS if f not in row]
    if missing:
        raise SchemaError(f"missing canonical fields: {missing}")
    if strict:
        for f in ("customer_key", "event_ts", "event_type"):
            if row.get(f) in (None, ""):
                raise SchemaError(f"null/empty required field: {f}")
    return dict(row)


def parse_attributes(raw: Any) -> dict:
    """Return event_attributes as a dict (accepts a JSON string or a mapping)."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw = raw.strip()
        return json.loads(raw) if raw else {}
    raise SchemaError(f"unsupported attributes type: {type(raw).__name__}")


def canonical_select(src: str, columns: tuple[str, ...] | list[str]) -> str:
    """A SELECT that yields the canonical five columns from a relation ``src``
    whose columns are ``columns``: every non-canonical column is folded into
    ``event_attributes`` (JSON merge; null patch values drop the key)."""
    folded = [c for c in columns if c not in CANONICAL_FIELDS and c != "event_attributes"]
    pairs = ", ".join(f"'{c}', {c}" for c in folded)
    merge = (
        "json_merge_patch(coalesce(try_cast(event_attributes AS JSON), json('{}')),"
        f" json_object({pairs}))"
        if pairs
        else "coalesce(try_cast(event_attributes AS JSON), json('{}'))"
    )
    st = "source_table" if "source_table" in columns else f"'{src}'"
    return (
        f"SELECT customer_key, event_ts, event_type, {st} AS source_table, "
        f"CAST({merge} AS VARCHAR) AS event_attributes FROM {src}"
    )


def canonical_view_sql(
    table: str = EVENT_STREAM_TABLE,
    view: str = CANONICAL_VIEW,
    columns: tuple[str, ...] | list[str] = FOLDED_FIELDS,
) -> str:
    """SQL to expose a WIDE materialized table as the canonical five-field form.

    ``columns`` = the table's actual columns; every non-canonical one is folded
    into ``event_attributes`` via a JSON patch (schema-on-read), so existing wide
    fixtures conform without a (space-prohibitive) rewrite.
    """
    return f"CREATE OR REPLACE VIEW {view} AS {canonical_select(table, columns)}"
