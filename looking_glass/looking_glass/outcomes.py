"""Library-owned, leakage-safe outcome construction for supervised tasks.

This is the first-class replacement for per-script outcome builders. Given a
unified event stream and a :class:`LabelSpec`, it produces supervised outcome
rows where:

* **features** are aggregated strictly from events at or before ``as_of`` within
  the history window, and
* **labels** are derived strictly from events after ``as_of`` within the
  horizon window.

Because the two windows never overlap and feature aggregation never reads a
post-``as_of`` event, the construction is leakage-safe by design. The same
``LabelSpec`` shape works for any new task: only the target definition changes,
which is the whole point of the "add a target and you're done" workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


def _parse_ts(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


@dataclass(frozen=True)
class LabelSpec:
    """Definition of one supervised task over an event stream.

    Attributes:
        id_field: Entity/sequence id grouping events (e.g. ``customer_id``).
        timestamp_field: ISO-8601 event timestamp field.
        as_of: Prediction cutoff. Features use events <= as_of; labels use > as_of.
        history_days: Lookback window for feature aggregation.
        horizon_days: Forward window for label derivation.
        value_field: Optional numeric field aggregated for value/LTV targets.
        label_kind: ``"activity_churn"`` (1 if no future events) or
            ``"value_sum"`` (sum of future ``value_field``).
        min_history_events: Drop entities with fewer historical events.
        active_lookback_days: Window for the recent-activity eligibility filter.
        min_recent_events: Require this many events within ``active_lookback_days``.
        recent_window_days: Window for the ``recent_count`` / ``recent_value``
            recency features (default 30 days before ``as_of``).
    """

    id_field: str
    timestamp_field: str
    as_of: datetime
    history_days: int = 365
    horizon_days: int = 120
    value_field: str | None = None
    label_kind: str = "activity_churn"
    min_history_events: int = 1
    active_lookback_days: int = 120
    min_recent_events: int = 0
    recent_window_days: int = 30

    def __post_init__(self) -> None:
        if self.label_kind not in {"activity_churn", "value_sum"}:
            raise ValueError("label_kind must be 'activity_churn' or 'value_sum'")
        if self.label_kind == "value_sum" and not self.value_field:
            raise ValueError("value_sum label_kind requires value_field")
        if self.history_days <= 0 or self.horizon_days <= 0:
            raise ValueError("history_days and horizon_days must be positive")


@dataclass(frozen=True)
class OutcomeFrame:
    """Outcome rows plus metadata describing how they were built."""

    spec: LabelSpec
    rows: list[dict[str, object]]
    feature_fields: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def scores_and_labels(self, score_field: str, label_field: str = "churn_label") -> tuple[list[float], list[float]]:
        scores = [float(r.get(score_field, 0.0) or 0.0) for r in self.rows]
        labels = [float(r.get(label_field, 0.0) or 0.0) for r in self.rows]
        return scores, labels


# Generic per-entity feature names produced for every label spec.
FEATURE_FIELDS: list[str] = [
    "event_count",
    "total_value",
    "avg_value",
    "max_value",
    "recency_days",
    "recent_count",
    "recent_value",
    "active_count",
    "distinct_active_days",
]


def build_outcomes(events: list[dict[str, object]], spec: LabelSpec) -> OutcomeFrame:
    """Build leakage-safe outcome rows from a unified event stream.

    Each output row contains the entity id, generic history features, a binary
    ``churn_label`` and a continuous ``value_label`` (zero when no value field).
    """

    history_start = spec.as_of - timedelta(days=spec.history_days)
    horizon_end = spec.as_of + timedelta(days=spec.horizon_days)
    recent_start = spec.as_of - timedelta(days=spec.recent_window_days)
    active_start = spec.as_of - timedelta(days=spec.active_lookback_days)

    buckets: dict[str, dict[str, object]] = {}
    for row in events:
        ts = _parse_ts(row[spec.timestamp_field])
        if ts < history_start or ts > horizon_end:
            continue
        entity_id = str(row[spec.id_field])
        bucket = buckets.setdefault(
            entity_id,
            {
                "event_count": 0,
                "total_value": 0.0,
                "max_value": 0.0,
                "last_ts": 0.0,
                "recent_count": 0,
                "recent_value": 0.0,
                "active_count": 0,
                "active_days": set(),
                "future_count": 0,
                "future_value": 0.0,
            },
        )
        value = float(row.get(spec.value_field, 0.0) or 0.0) if spec.value_field else 0.0

        if ts <= spec.as_of:
            # PAST: contributes to features only.
            bucket["event_count"] = int(bucket["event_count"]) + 1
            bucket["total_value"] = float(bucket["total_value"]) + max(value, 0.0)
            bucket["max_value"] = max(float(bucket["max_value"]), max(value, 0.0))
            bucket["last_ts"] = max(float(bucket["last_ts"]), ts.timestamp())
            if ts >= recent_start:
                bucket["recent_count"] = int(bucket["recent_count"]) + 1
                bucket["recent_value"] = float(bucket["recent_value"]) + max(value, 0.0)
            if ts >= active_start:
                bucket["active_count"] = int(bucket["active_count"]) + 1
                bucket["active_days"].add(ts.date())  # type: ignore[union-attr]
        else:
            # FUTURE: contributes to labels only.
            bucket["future_count"] = int(bucket["future_count"]) + 1
            bucket["future_value"] = float(bucket["future_value"]) + max(value, 0.0)

    rows: list[dict[str, object]] = []
    for entity_id, bucket in buckets.items():
        event_count = int(bucket["event_count"])
        if event_count < spec.min_history_events:
            continue
        if int(bucket["active_count"]) < spec.min_recent_events:
            continue
        last_ts_unix = float(bucket["last_ts"])
        if last_ts_unix <= 0.0:
            continue

        last_ts = datetime.fromtimestamp(last_ts_unix, tz=spec.as_of.tzinfo)
        recency_days = float(max((spec.as_of - last_ts).days, 0))
        future_count = int(bucket["future_count"])
        future_value = float(bucket["future_value"])

        churn_label = 1.0 if future_count == 0 else 0.0
        # Honor label_kind: value_sum targets get the future spend; churn-only
        # specs emit a zero value label so it cannot be used as a target by
        # accident.
        value_label = future_value if spec.label_kind == "value_sum" else 0.0

        rows.append(
            {
                spec.id_field: entity_id,
                "event_count": float(event_count),
                "total_value": float(bucket["total_value"]),
                "avg_value": float(bucket["total_value"]) / max(event_count, 1),
                "max_value": float(bucket["max_value"]),
                "recency_days": recency_days,
                "recent_count": float(bucket["recent_count"]),
                "recent_value": float(bucket["recent_value"]),
                "active_count": float(bucket["active_count"]),
                "distinct_active_days": float(len(bucket["active_days"])),  # type: ignore[arg-type]
                "future_count": float(future_count),
                "churn_label": churn_label,
                "value_label": value_label,
            }
        )

    return OutcomeFrame(spec=spec, rows=rows, feature_fields=list(FEATURE_FIELDS))


def assert_no_leakage(events: list[dict[str, object]], spec: LabelSpec, frame: OutcomeFrame) -> None:
    """Independently re-derive features from past-only events and confirm a match.

    This is a defensive cross-check: it rebuilds ``event_count`` and
    ``total_value`` using a filter that physically excludes every event after
    ``as_of`` and asserts the produced frame agrees. A mismatch means a
    post-cutoff event influenced a feature, i.e. label leakage.
    """

    history_start = spec.as_of - timedelta(days=spec.history_days)
    recompute: dict[str, tuple[int, float]] = {}
    for row in events:
        ts = _parse_ts(row[spec.timestamp_field])
        if ts < history_start or ts > spec.as_of:  # strictly past-only
            continue
        entity_id = str(row[spec.id_field])
        value = float(row.get(spec.value_field, 0.0) or 0.0) if spec.value_field else 0.0
        count, total = recompute.get(entity_id, (0, 0.0))
        recompute[entity_id] = (count + 1, total + max(value, 0.0))

    for out_row in frame.rows:
        entity_id = str(out_row[spec.id_field])
        exp_count, exp_total = recompute.get(entity_id, (0, 0.0))
        if int(out_row["event_count"]) != exp_count:  # type: ignore[arg-type]
            raise AssertionError(
                f"leakage detected for {entity_id}: event_count {out_row['event_count']} != past-only {exp_count}"
            )
        if abs(float(out_row["total_value"]) - exp_total) > 1e-6:  # type: ignore[arg-type]
            raise AssertionError(
                f"leakage detected for {entity_id}: total_value {out_row['total_value']} != past-only {exp_total}"
            )


def attach_state_vectors(
    frame: OutcomeFrame,
    state_lookup,
    output_field: str = "core_as_of_vector",
) -> OutcomeFrame:
    """Attach each entity's point-in-time backbone state, evaluated at ``as_of``.

    ``state_lookup`` is any callable ``(entity_id, as_of) -> vector | None``;
    pass :meth:`PointInTimeStateStore.get_state_as_of` to wire in the store.
    Entities with no available state get an all-zero vector of matching width.
    """

    vectors: dict[str, list[float] | None] = {}
    width = 0
    for row in frame.rows:
        entity_id = str(row[frame.spec.id_field])
        vec = state_lookup(entity_id, frame.spec.as_of)
        vectors[entity_id] = vec
        if vec is not None:
            width = max(width, len(vec))

    zero = [0.0] * width
    for row in frame.rows:
        vec = vectors[str(row[frame.spec.id_field])]
        row[output_field] = list(vec) if vec is not None else list(zero)
    return frame
