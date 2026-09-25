"""Core LR probe: 3 epochs each, watch next-event loss."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from looking_glass import create_temporal_core_model, PayloadSchema

DB = Path("scripts/data/toy/toy.db")
D = datetime(2025, 11, 15, tzinfo=timezone.utc)


def main():
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT customer_key, event_ts, brand, event_type, event_attributes FROM events"
    ).fetchall()
    conn.close()
    evs = []
    for i, r in enumerate(rows):
        try:
            p = json.loads(r[4] or "{}")
        except json.JSONDecodeError:
            p = {}
        if not isinstance(p, dict):
            p = {}
        v = p.get("margin_dollars", p.get("order_value", 0.0))
        try:
            v = float(v or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        evs.append(
            {
                "event_id": f"ev_{i:08d}",
                "customer_id": r[0],
                "event_ts": r[1],
                "event_type": r[3],
                "event_payload_json": p,
                "value": v,
            }
        )
    evs.sort(key=lambda e: (e["customer_id"], e["event_ts"], e["event_id"]))
    core_events = [e for e in evs if e["event_ts"] <= D.isoformat()]
    print(f"events={len(core_events)}")

    schema = PayloadSchema(
        categorical_fields=["device"], numeric_fields=["order_value", "margin_dollars"]
    )
    for lr in (1e-2, 3e-3, 1e-3):
        m = create_temporal_core_model(
            sequence_id_field="customer_id",
            event_id_field="event_id",
            timestamp_field="event_ts",
            categorical_fields=[],
            numeric_fields=[],
            vector_fields=[],
            payload_schema=schema,
            hidden_dim=128,
            epochs=3,
            seed=17,
            learning_rate=lr,
            device="cuda",
            sequence_backend="mamba2",
            train_batch_size=64,
            backbone_version="v1",
        )
        m.fit_transform(core_events)
        print(f"lr={lr:<8} loss={m.loss_:.4f}", flush=True)


if __name__ == "__main__":
    main()
