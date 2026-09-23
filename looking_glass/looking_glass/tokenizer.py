"""Event payload tokenizer for the unified event-stream schema.

Every event in the stream has exactly five top-level fields::

    {event_id, customer_id, timestamp, event_type, event_payload_json}

The payload is a heterogeneous JSON object whose keys vary by event type.
This module parses those keys into three signal sources that are
independently projected and summed element-wise by the temporal encoder:

    E_event = Σ E_cat + Σ E_num + Σ E_vector

Where:

* **categorical** — discrete values via embedding lookup.
* **numeric** — continuous values via linear projection.
* **vector** — entity IDs (product_id, store_id, campaign_id, ...) resolved
  against pre-computed static embedding tables and summed into a combined
  vector.  Every entity referenced by the event stream has its own
  embedding table — static ones are pre-computed offline, time-dependent
  ones (customers) emerge as byproducts of the temporal core.

Missing vector keys resolve to the zero vector (the identity element under
addition), so events without a product/store/campaign reference are
handled naturally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class PayloadSchema:
    """Declares how payload JSON keys map to tokenizer feature types.

    Attributes:
        categorical_fields: Payload keys treated as categorical features.
        numeric_fields: Payload keys treated as numeric features.
        vector_id_keys: Payload keys whose values are entity IDs that should
            be resolved against pre-computed static embedding tables.  Each
            key maps to a different table (e.g. ``"product_id"`` → product
            vectors, ``"store_id"`` → store vectors, ``"campaign_id"`` →
            campaign vectors).  All resolved vectors are summed
            element-wise.  Missing keys or unresolvable IDs contribute the
            zero vector.
    """

    categorical_fields: list[str] = field(default_factory=list)
    numeric_fields: list[str] = field(default_factory=list)
    vector_id_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedPayload:
    """Output of :func:`parse_event_payload` for a single event.

    Attributes:
        cat_values: ``{field_name: string_value}`` for categorical fields.
        num_values: ``{field_name: float_value}`` for numeric fields.
        combined_vector: Element-wise sum of all resolved static embedding
            vectors.  This is a ``list[float]`` (not a tensor) so it can be
            passed through the existing tensorize pipeline.  When no vector
            lookups match the result is a zero vector of the configured
            dimension.
    """

    cat_values: dict[str, str]
    num_values: dict[str, float]
    combined_vector: list[float]


def parse_event_payload(
    payload: dict[str, object],
    schema: PayloadSchema,
    vector_lookups: dict[str, Callable[[str], list[float] | None]] | None = None,
    default_vector_width: int = 0,
) -> ParsedPayload:
    """Parse one ``event_payload_json`` dict according to *schema*.

    Args:
        payload: The ``event_payload_json`` value from an event row.
        schema: Which payload keys map to which feature types.
        vector_lookups: ``{key: (id_value) -> vector}`` dict.  One entry
            per entry in ``schema.vector_id_keys``.  When ``None`` or when
            a key is missing the corresponding vector is all-zero.
        default_vector_width: Dimension of the zero vector returned when no
            lookups resolve.

    Returns:
        Parsed categorical values, numeric values, and the combined
        (element-wise summed, or zero) vector.
    """

    cat_values: dict[str, str] = {}
    for field in schema.categorical_fields:
        raw = payload.get(field)
        cat_values[field] = str(raw) if raw is not None else ""

    num_values: dict[str, float] = {}
    for field in schema.numeric_fields:
        raw = payload.get(field)
        try:
            num_values[field] = float(raw) if raw is not None else 0.0
        except (ValueError, TypeError):
            num_values[field] = 0.0

    # Sum all resolved entity vectors element-wise.
    combined_vec: list[float] = [0.0] * default_vector_width
    if vector_lookups is not None and default_vector_width > 0:
        import numpy as np  # local import — lightweight, already a dep
        vec_arr = np.zeros(default_vector_width, dtype=np.float64)
        for key in schema.vector_id_keys:
            lookup = vector_lookups.get(key)
            if lookup is None:
                continue
            entity_id = str(payload.get(key, "") or "")
            if not entity_id:
                continue
            resolved = lookup(entity_id)
            if resolved is not None and len(resolved) == default_vector_width:
                vec_arr += np.asarray(resolved, dtype=np.float64)
        combined_vec = vec_arr.tolist()

    return ParsedPayload(
        cat_values=cat_values,
        num_values=num_values,
        combined_vector=combined_vec,
    )


def collect_payload_vocabularies(
    event_rows: list[dict[str, object]],
    schema: PayloadSchema,
) -> dict[str, dict[str, int]]:
    """Scan the event stream and build deterministic per-categorical-field
    vocabularies from every payload key declared in *schema*.

    ``event_type`` is always added even when it is not in
    ``schema.categorical_fields``.
    """

    vocabs: dict[str, dict[str, int]] = {}
    for field in schema.categorical_fields:
        vocabs[field] = {}
    if "event_type" not in vocabs:
        vocabs["event_type"] = {}

    for row in event_rows:
        for field in schema.categorical_fields:
            raw = (row.get("event_payload_json") or {}).get(field) if isinstance(row.get("event_payload_json"), dict) else row.get(field)
            vocabs[field][str(raw) if raw is not None else ""] = 0
        et = str(row.get("event_type", "") or "")
        vocabs["event_type"][et] = 0

    return {
        field: {val: idx for idx, val in enumerate(sorted(values.keys()))}
        for field, values in vocabs.items()
    }
