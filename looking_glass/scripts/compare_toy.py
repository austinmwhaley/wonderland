"""Compare per-D vs fresh encoders for 30d LTV plugin at D=2025-09-01.

A per-D (clean/stale): encoder <= D, vectors as-of D. No leak.
B fresh-leaky (WRONG demo): encoder <= TODAY, vectors as-of TODAY,
  plugin labels D->D+30. Input contains future. Shows inflation.
Proper fresh+recompute (weights TODAY, input <= D) needs a transform-only
path the lib lacks (prod daily job). B is the ceiling of that bias.

Usage:
  .venv/bin/python scripts/compare_toy.py
"""

from __future__ import annotations

import duckdb
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from looking_glass import (
    build_outcomes,
    create_supervised_model,
    create_temporal_core_model,
    LabelSpec,
    PayloadSchema,
)

DB = Path("scripts/data/toy/toy.db")
D = datetime(2025, 9, 1, tzinfo=timezone.utc)
HID = 64


def _iso(value):
    """Coerce timestamp values to ISO-8601 text for stable string comparisons."""

    return value.isoformat() if hasattr(value, "isoformat") else value


def load_events():
    conn = duckdb.connect(str(DB), read_only=True)
    cursor = conn.execute(
        "SELECT customer_key, event_ts, brand, event_type, event_attributes FROM events"
    )
    col_names = [desc[0] for desc in cursor.description]
    rows = [dict(zip(col_names, row)) for row in cursor.fetchall()]
    conn.close()
    evs = []
    for i, r in enumerate(rows):
        try:
            p = json.loads(r["event_attributes"] or "{}")
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
                "customer_id": r["customer_key"],
                "event_ts": _iso(r["event_ts"]),
                "event_type": r["event_type"],
                "event_payload_json": p,
                "value": v,
            }
        )
    evs.sort(key=lambda e: (e["customer_id"], e["event_ts"], e["event_id"]))
    return evs


def train_encoder(events, tag, epochs=4):
    schema = PayloadSchema(
        categorical_fields=["device"], numeric_fields=["order_value", "margin_dollars"]
    )
    m = create_temporal_core_model(
        sequence_id_field="customer_id",
        event_id_field="event_id",
        timestamp_field="event_ts",
        categorical_fields=[],
        numeric_fields=[],
        vector_fields=[],
        payload_schema=schema,
        hidden_dim=HID,
        epochs=epochs,
        device="cpu",
        sequence_backend="mamba2",
        train_batch_size=64,
        backbone_version=tag,
    )
    t0 = time.perf_counter()
    out = m.fit_transform(events)
    dt = time.perf_counter() - t0
    lookup = {str(r["customer_id"]): r for r in out.customer_records}
    return m, lookup, dt


def train_ltv(rows, lookup):
    for r in rows:
        r.update(lookup.get(str(r["customer_id"]), {}))
    rows = [r for r in rows if "core_last_vector" in r]
    agg = [
        f
        for f in rows[0].keys()
        if f
        in (
            "event_count",
            "total_value",
            "avg_value",
            "max_value",
            "recency_days",
            "recent_count",
            "recent_value",
            "active_count",
        )
    ]
    m = create_supervised_model(
        task="regression",
        id_field="customer_id",
        target_field="value_label",
        categorical_fields=[],
        numeric_fields=agg,
        vector_fields=["core_last_vector"],
        hidden_dim=HID,
        epochs=25,
        seed=17,
        validation_fraction=0.25,
        device="cpu",
        sequence_backend="mamba2",
    )
    res = m.fit_predict(rows)
    return res.report.metrics.get("r2", 0.0), res.report.metrics.get("rmse", 0.0), len(rows)


def main():
    events = load_events()
    d_iso = D.isoformat()
    per_d_events = [e for e in events if e["event_ts"] <= d_iso]
    print(f"total={len(events)} per-D(<=Sept1)={len(per_d_events)}")

    spec = LabelSpec(
        id_field="customer_id",
        timestamp_field="event_ts",
        as_of=D,
        history_days=180,
        horizon_days=30,
        value_field="value",
        label_kind="value_sum",
        min_history_events=2,
    )
    import copy

    base_rows = build_outcomes(events, spec).rows
    print(f"outcome rows at D={len(base_rows)}")

    print("\nA. per-D encoder (<=Sept1)...")
    _, look_a, dt_a = train_encoder(per_d_events, "v-perD")
    r2_a, rmse_a, n_a = train_ltv(copy.deepcopy(base_rows), look_a)
    print(f"   R2={r2_a:.3f} RMSE={rmse_a:.2f} n={n_a} time={dt_a:.0f}s")

    print("\nB. fresh encoder (ALL data, vectors as-of TODAY) -- LEAKY demo...")
    _, look_b, dt_b = train_encoder(events, "v-fresh")
    r2_b, rmse_b, n_b = train_ltv(copy.deepcopy(base_rows), look_b)
    print(f"   R2={r2_b:.3f} RMSE={rmse_b:.2f} n={n_b} time={dt_b:.0f}s")

    print("\n---")
    print(f"A per-D clean:  R2={r2_a:.3f} (safe, stale for 365d use)")
    print(f"B fresh-leaky:  R2={r2_b:.3f} (inflated: input saw the label window)")
    print("Proper fresh+recompute (weights TODAY, input <= D) sits between.")
    print("Lib lacks transform-only; prod needs daily-job recompute path.")
    print(f"Compute: per-D per window = N retrains ({dt_a:.0f}s each); fresh = 1 ({dt_b:.0f}s).")


if __name__ == "__main__":
    main()
