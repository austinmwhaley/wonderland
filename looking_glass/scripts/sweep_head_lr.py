"""Head LR sweep on aggregates-only churn (no backbone needed, fast)."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from looking_glass import build_outcomes, create_supervised_model, LabelSpec

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
        evs.append(
            {
                "event_id": f"ev_{i:08d}",
                "customer_id": r[0],
                "event_ts": r[1],
                "event_type": r[3],
                "event_payload_json": p,
                "value": 0.0,
            }
        )
    evs.sort(key=lambda e: (e["customer_id"], e["event_ts"], e["event_id"]))

    spec = LabelSpec(
        id_field="customer_id",
        timestamp_field="event_ts",
        as_of=D,
        history_days=180,
        horizon_days=45,
        value_field="value",
        label_kind="activity_churn",
        min_history_events=2,
    )
    orows = build_outcomes(evs, spec).rows
    print(f"rows={len(orows)} churn={sum(float(r['churn_label']) for r in orows) / len(orows):.2f}")
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
        if f in orows[0]
    ]

    for lr in (1e-2, 3e-3, 1e-3, 3e-4):
        m = create_supervised_model(
            task="classification",
            id_field="customer_id",
            target_field="churn_label",
            categorical_fields=[],
            numeric_fields=agg,
            vector_fields=[],
            hidden_dim=128,
            epochs=60,
            seed=17,
            learning_rate=lr,
            validation_fraction=0.25,
            device="cuda",
            sequence_backend="mamba2",
        )
        res = m.fit_predict(copy.deepcopy(orows))
        met = res.report.metrics
        print(
            f"lr={lr:<8} F1={met.get('f1', 0):.3f} AUC={met.get('roc_auc', 0):.3f} "
            f"thr={met.get('threshold', 0):.3f}"
        )


if __name__ == "__main__":
    main()
