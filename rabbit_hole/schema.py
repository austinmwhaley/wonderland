"""Canonical unified customer event stream schema (Layer A).

rabbit_hole ends at ONE canonical, append-only event stream. Producers differ
in column naming, so the canonical field names below are fixed and incoming
rows are canonicalized through :data:`COLUMN_ALIASES`.

Canonical fields:

    customer_key, event_ts, brand, event_type, event_attributes

(raw input is `[customer_key, event_ts, brand, event_type, event_attributes]`,
per the system spec). `event_attributes` is a schema-evolved JSON/attribute map
for all remaining context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

CANONICAL_FIELDS = (
    "customer_key",
    "event_ts",
    "brand",
    "event_type",
    "event_attributes",
)

EVENT_STREAM_TABLE = "customer_events"

# Accepted source column names -> canonical field.
COLUMN_ALIASES = {
    "customer_key": ("customer_key", "customer_id", "cust_id", "customer"),
    "event_ts": ("event_ts", "timestamp", "ts", "event_time", "event_timestamp"),
    "brand": ("brand", "brand_id"),
    "event_type": ("event_type", "type", "event_name"),
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
    brand: Any
    event_type: str
    event_attributes: Mapping[str, Any]

    def to_row(self) -> dict:
        attrs = self.event_attributes
        if not isinstance(attrs, str):
            attrs = json.dumps(attrs, separators=(",", ":"))
        return {
            "customer_key": self.customer_key,
            "event_ts": self.event_ts,
            "brand": self.brand,
            "event_type": self.event_type,
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
    """Map an arbitrary event row's columns onto the canonical field names.

    Missing optional fields (brand, event_attributes) become empty/`{}`; a
    missing key/ts/type raises :class:`SchemaError`.
    """
    cols = set(row.keys())
    out = {}
    for field in CANONICAL_FIELDS:
        src = _detect(cols, field)
        out[field] = row.get(src) if src is not None else None
    if out["customer_key"] in (None, "") or out["event_ts"] in (None, ""):
        if strict:
            raise SchemaError("missing customer_key or event_ts")
    # Fold any first-class entity/source/value columns that a materializer may
    # emit into the attribute map, so the canonical row keeps that information.
    extras = {}
    for k in ("entity_type", "entity_id", "source_table", "value"):
        if k in cols and row.get(k) is not None:
            extras[k] = row[k]
    if extras:
        try:
            attrs = parse_attributes(out["event_attributes"])
        except Exception:
            attrs = {}
        attrs.update(extras)
        out["event_attributes"] = json.dumps(attrs)
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
