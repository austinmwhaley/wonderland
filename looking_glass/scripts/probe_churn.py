"""Churn probe: is the frozen CFM good enough for churn?

Three-way ablation on same rows at D=2025-09-01, horizon 90d:
  vectors-only vs aggregates-only vs combined (+ xgboost baseline).
Answers: does the backbone carry churn signal beyond hand counts?

Usage:
  .venv/bin/python scripts/probe_churn.py
"""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from looking_glass import (
    build_outcomes,
    create_supervised_model,
    create_temporal_core_model,
    LabelSpec,
    PayloadSchema,
)

try:
    from looking_glass import GBTBaseline

    HAS_GBT = True
except Exception:
    HAS_GBT = False

DB = Path("scripts/data/toy/toy.db")
D = datetime(2025, 11, 15, tzinfo=timezone.utc)
HORIZON = 45  # toy is dense: 45d at Nov-15 gives ~36% churn; 90d gives ~0%
HID = 128
CORE_EPOCHS = 30
CORE_LR = 3e-3  # 1e-2 stalls backbone at ~2.9; 3e-3 hits ~1.7 in 3 epochs
HEAD_EPOCHS = 60
HEAD_LR = 1e-2  # default; auto-retry should heal collapse down to ~1e-3


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


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto uses CUDA if available, else CPU")
    args = ap.parse_args()

    events = load_events()
    d_iso = D.isoformat()
    core_events = [e for e in events if e["event_ts"] <= d_iso]
    print(f"encoder train: {len(core_events)} events <= D")

    schema = PayloadSchema(
        categorical_fields=["device"], numeric_fields=["order_value", "margin_dollars"]
    )
    core = create_temporal_core_model(
        sequence_id_field="customer_id",
        event_id_field="event_id",
        timestamp_field="event_ts",
        categorical_fields=[],
        numeric_fields=[],
        vector_fields=[],
        payload_schema=schema,
        hidden_dim=HID,
        epochs=CORE_EPOCHS,
        seed=17,
        learning_rate=CORE_LR,
        device=args.device,
        sequence_backend="mamba2",
        train_batch_size=64,
        backbone_version="v1",
    )
    out = core.fit_transform(core_events)
    print(f"core loss={core.loss_:.4f} customers={len(out.customer_records)}")
    lookup = {str(r["customer_id"]): r for r in out.customer_records}

    spec = LabelSpec(
        id_field="customer_id",
        timestamp_field="event_ts",
        as_of=D,
        history_days=180,
        horizon_days=HORIZON,
        value_field="value",
        label_kind="activity_churn",
        min_history_events=2,
    )
    frame = build_outcomes(events, spec)
    rows = frame.rows
    churn_rate = sum(float(r["churn_label"]) for r in rows) / max(len(rows), 1)
    print(f"outcome rows={len(rows)} churn_rate={churn_rate:.2f}")
    for r in rows:
        r.update(lookup.get(str(r["customer_id"]), {}))
    rows = [r for r in rows if "core_last_vector" in r]
    print(f"rows with vector={len(rows)}")
    agg = [
        f
        for f in (
            "event_count",
            "total_value",
            "avg_value",
            "max_value",
            "recency_days",
            "recent_count",
            "recent_value",
            "active_count",
        )
        if f in rows[0]
    ]

    def train_churn(name, num, vec):
        m = create_supervised_model(
            task="classification",
            id_field="customer_id",
            target_field="churn_label",
            categorical_fields=[],
            numeric_fields=num,
            vector_fields=vec,
            hidden_dim=HID,
            epochs=HEAD_EPOCHS,
            seed=17,
            learning_rate=HEAD_LR,
            validation_fraction=0.25,
            device=args.device,
            sequence_backend="mamba2",
        )
        res = m.fit_predict(copy.deepcopy(rows))
        met = res.report.metrics
        print(f"{name:16s} F1={met.get('f1', 0):.3f} AUC={met.get('roc_auc', 0):.3f}")
        return met

    print("\n--- churn ablation (same rows) ---")
    train_churn("vectors-only", [], ["core_last_vector"])
    train_churn("aggregates-only", agg, [])
    train_churn("combined", agg, ["core_last_vector"])

    if HAS_GBT:
        try:
            bl = GBTBaseline("classification", "customer_id", "churn_label", agg, seed=17)
            r = bl.fit_predict(copy.deepcopy(rows))
            print(
                f"{'xgb-baseline':16s} F1={r.metrics.get('f1', 0):.3f} AUC={r.metrics.get('roc_auc', 0):.3f}"
            )
        except Exception as e:
            print(f"xgb-baseline failed: {e}")
    else:
        print("xgb-baseline skipped (no xgboost)")

    print("\nRead: vectors-only vs aggregates-only tells you if the CFM")
    print("carries churn signal beyond hand counts. Combined should win.")


if __name__ == "__main__":
    main()
