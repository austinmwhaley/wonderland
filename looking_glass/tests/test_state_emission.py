from datetime import datetime, timedelta, timezone

from looking_glass import (
    InMemoryStateStore,
    create_temporal_core_model,
    to_state_records,
)

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _events():
    rows = []
    eid = 0
    for cid in ("a", "b"):
        for k in range(3):
            eid += 1
            rows.append(
                {
                    "customer_id": cid,
                    "event_id": f"e{eid}",
                    "event_ts": (BASE + timedelta(days=10 * (eid))).isoformat(),
                    "event_type": "click" if k % 2 == 0 else "order",
                    "value": float(k + 1),
                }
            )
    return rows


def test_temporal_core_emits_versioned_pit_states():
    model = create_temporal_core_model(
        sequence_id_field="customer_id",
        event_id_field="event_id",
        timestamp_field="event_ts",
        categorical_fields=["event_type"],
        numeric_fields=["value"],
        hidden_dim=32,
        epochs=1,
        device="cpu",
        sequence_backend="mamba2",
        backbone_version="test1",
        input_is_time_sorted=True,
    )
    outputs = model.fit_transform(_events())

    # Customer summaries are point-in-time stamped and versioned.
    assert len(outputs.customer_records) == 2
    for row in outputs.customer_records:
        assert row["backbone_version"] == "test1"
        assert row["core_as_of_ts"]  # non-empty ISO timestamp
        assert len(row["core_last_vector"]) == 32

    # Conversion + store roundtrip with as-of correctness and version isolation.
    records = to_state_records(outputs)
    assert len(records) == 2

    store = InMemoryStateStore(default_version="test1")
    store.write_states(records)
    future = BASE + timedelta(days=10_000)
    vec = store.get_state_as_of("a", future, backbone_version="test1")
    assert vec is not None and len(vec) == 32
    assert store.get_state_as_of("a", future, backbone_version="other") is None
    # Before any event there is no state.
    assert store.get_state_as_of("a", BASE - timedelta(days=1)) is None
