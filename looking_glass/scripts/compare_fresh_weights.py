"""Fresh-weights vs per-D encoders, zero state leakage.

EVAL customers (every 5th): sequences truncated to <= D in BOTH runs,
  so their vectors never contain their own future.
BG customers (rest): full history in fresh run, <= D in per-D run.

Plugin trains+evaluates on EVAL rows only, same rows both arms.
Delta = effect of future background data living in the weights.
No TODAY vectors used anywhere.

Usage:
  .venv/bin/python scripts/compare_fresh_weights.py
"""

from __future__ import annotations

import copy
import json
import sqlite3
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
EPOCHS = 4


def load_events():
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
    return evs


def train_encoder(records, tag):
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
        epochs=EPOCHS,
        device="cpu",
        sequence_backend="mamba2",
        train_batch_size=64,
        backbone_version=tag,
    )
    t0 = time.perf_counter()
    out = m.fit_transform(records)
    return m, {str(r["customer_id"]): r for r in out.customer_records}, time.perf_counter() - t0


def train_ltv(rows):
    agg = (
        "event_count",
        "total_value",
        "avg_value",
        "max_value",
        "recency_days",
        "recent_count",
        "recent_value",
        "active_count",
    )
    num = [f for f in agg if f in rows[0]]
    m = create_supervised_model(
        task="regression",
        id_field="customer_id",
        target_field="value_label",
        categorical_fields=[],
        numeric_fields=num,
        vector_fields=["core_last_vector"],
        hidden_dim=HID,
        epochs=25,
        seed=17,
        validation_fraction=0.25,
        device="cpu",
        sequence_backend="mamba2",
    )
    res = m.fit_predict(rows)
    return res.report.metrics.get("r2", 0.0), res.report.metrics.get("rmse", 0.0)


def main():
    events = load_events()
    d_iso = D.isoformat()
    customers = sorted({e["customer_id"] for e in events})
    eval_set = {c for i, c in enumerate(customers) if i % 5 == 0}
    print(f"customers={len(customers)} eval={len(eval_set)}")

    per_d = [e for e in events if e["event_ts"] <= d_iso]
    fresh = [e for e in events if e["event_ts"] <= d_iso or e["customer_id"] not in eval_set]
    print(
        f"per-D records={len(per_d)} fresh records={len(fresh)} "
        f"(+{len(fresh) - len(per_d)} future BG events in weights)"
    )

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
    base = [r for r in build_outcomes(events, spec).rows if str(r["customer_id"]) in eval_set]
    print(f"eval outcome rows at D={len(base)}")
    assert len(base) > 200, "too few eval rows"

    print("\nA. per-D encoder (all customers <= D)...")
    _, look_a, dt_a = train_encoder(per_d, "v-perD")
    rows_a = [
        dict(r, **look_a[str(r["customer_id"])])
        for r in copy.deepcopy(base)
        if str(r["customer_id"]) in look_a
    ]
    r2_a, rmse_a = train_ltv(rows_a)
    print(f"   R2={r2_a:.3f} RMSE={rmse_a:.2f} n={len(rows_a)} time={dt_a:.0f}s")

    print("\nB. fresh encoder (EVAL <= D, BG all times)...")
    _, look_b, dt_b = train_encoder(fresh, "v-fresh")
    rows_b = [
        dict(r, **look_b[str(r["customer_id"])])
        for r in copy.deepcopy(base)
        if str(r["customer_id"]) in look_b
    ]
    r2_b, rmse_b = train_ltv(rows_b)
    print(f"   R2={r2_b:.3f} RMSE={rmse_b:.2f} n={len(rows_b)} time={dt_b:.0f}s")

    # same-row subset for apples-to-apples
    common = {str(r["customer_id"]) for r in rows_a} & {str(r["customer_id"]) for r in rows_b}
    print(f"\ncommon eval customers scored both arms: {len(common)}")
    print(f"A per-D clean:        R2={r2_a:.3f}")
    print(f"B fresh-weights:      R2={r2_b:.3f}")
    print(
        "EVAL vectors contain no own-future in either arm. "
        "Delta = future background data in weights only."
    )


if __name__ == "__main__":
    main()
