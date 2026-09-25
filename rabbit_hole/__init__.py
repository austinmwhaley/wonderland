"""rabbit_hole — the unified customer event stream (Layer A).

rabbit_hole *ends* at the customer event stream. It turns hundreds of wide
customer tables into one canonical, append-only stream:

    [event_id, customer_id, timestamp, event_type, event_payload_json]

and nothing downstream: tokenization, embeddings, and every model live in
``looking_glass``.

- schema.py  : canonical fields, EventRecord, validation
- stream.py  : read/write helpers for the customer event stream
- generators : synthetic omnichannel simulacrum + toy data generators
"""

from .schema import (
    CANONICAL_FIELDS,
    EVENT_STREAM_TABLE,
    EventRecord,
    SchemaError,
    validate_event,
    canonicalize_row,
    parse_attributes,
)
from .stream import read_events, write_events, CustomerEventStream, to_duckdb, to_parquet, engine_of

__all__ = [
    "CANONICAL_FIELDS",
    "EVENT_STREAM_TABLE",
    "EventRecord",
    "SchemaError",
    "validate_event",
    "canonicalize_row",
    "parse_attributes",
    "read_events",
    "write_events",
    "CustomerEventStream",
    "to_duckdb",
    "to_parquet",
    "engine_of",
]
