"""Stand-alone end-to-end demo of the frozen-core LoRA paradigm.

Generates a synthetic retail event stream in the canonical 5-field schema
({event_id, customer_id, timestamp, event_type, event_payload_json}), trains
a product embedding model, enriches events with product vectors at tokenize
time via the payload parser, pretrains a Mamba temporal core, then trains
downstream task heads (churn + LTV) on lightweight LoRA adapters over the
frozen backbone — with an xgboost-on-aggregates baseline for comparison.

The report includes a three-way ablation so you can see exactly what the
learned vectors contribute on top of (and instead of) hand-crafted features.

No external data. No special hardware. Runs start to finish in ~2 minutes on
a modern CPU; much faster on CUDA.

Usage:
    python scripts/example.py
"""

from __future__ import annotations

import logging
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the repository root is importable when this file is run directly.
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))


from looking_glass import (
    build_outcomes,
    create_embedding_model,
    create_supervised_model,
    create_temporal_core_model,
    GBTBaseline,
    LabelSpec,
    PayloadSchema,
    QDoRAConfig,
    save_embeddings_to_lancedb,
    wrap_core_with_qdora,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("example")

# ---------------------------------------------------------------------------
# 1.  Synthetic data — canonical event-stream schema.
#
# Every event row has exactly five top-level fields.  There are no nullable
# columns.  The payload carries whatever is relevant to that event type.
#
# Event type taxonomy:
#   profile     — customer_profile          (anchor events)
#   web         — web_page_view, web_product_view, web_search, web_click
#   commerce    — add_to_cart, online_purchase, retail_purchase,
#                 return_initiated, product_review
#   marketing   — email_send, email_open, email_click,
#                 sms_send, sms_click,
#                 app_push_send, app_push_open, app_push_click
#   account     — login, password_reset, address_update
#   support     — support_ticket_created, support_chat_started, support_resolved
# ---------------------------------------------------------------------------

NOW = datetime(2024, 12, 1, tzinfo=timezone.utc)
RNG = random.Random(42)

CATEGORIES = ["electronics", "clothing", "home", "sports", "books"]
BRANDS = ["alpha", "beta", "gamma", "delta", "epsilon"]
PAYMENT_METHODS = ["card", "paypal", "apple_pay", "google_pay"]
STORE_LOCATIONS = ["NYC", "SF", "CHI", "ATX", "SEA", None]  # None = online
COUNTRIES = ["US", "UK", "DE", "FR", "JP"]
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]
ACQUISITION_CHANNELS = ["organic", "paid_search", "social", "email", "referral"]
CAMPAIGN_IDS = ["spring_sale", "clearance", "new_arrivals", "loyalty_rewards", None]

N_PRODUCTS = 80
N_CUSTOMERS = 500
N_EVENTS = 12000

HIDDEN_DIM = 128
SEED = 17
OUTCOME_CUTOFF = NOW - timedelta(days=90)

AGG_FEATURE_FIELDS = [
    "event_count",
    "total_value",
    "avg_value",
    "max_value",
    "recency_days",
    "recent_count",
    "recent_value",
    "active_count",
]


def _gen_products() -> list[dict[str, object]]:
    return [
        {
            "product_id": f"p{i:04d}",
            "category": RNG.choice(CATEGORIES),
            "brand": RNG.choice(BRANDS),
            "base_price": round(RNG.uniform(5, 200), 2),
            "unit_cost": round(RNG.uniform(2, 100), 2),
        }
        for i in range(N_PRODUCTS)
    ]


def _make_event(
    eid: int,
    customer_id: str,
    ts: datetime,
    event_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "event_id": f"e{eid:06d}",
        "customer_id": customer_id,
        "event_ts": ts.isoformat(),
        "event_type": event_type,
        "event_payload_json": payload,
    }


def _gen_event_stream() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    eid = 0
    product_ids = [f"p{i:04d}" for i in range(N_PRODUCTS)]

    for cid in range(N_CUSTOMERS):
        customer_id = f"c{cid:05d}"
        signup_day = RNG.randint(0, 300)
        profile_ts = NOW - timedelta(days=300) + timedelta(days=signup_day)

        # Anchor event: customer profile enters the stream.
        eid += 1
        events.append(
            _make_event(
                eid,
                customer_id,
                profile_ts,
                "customer_profile",
                {
                    "country": RNG.choice(COUNTRIES),
                    "acquisition_channel": RNG.choice(ACQUISITION_CHANNELS),
                    "loyalty_tier": RNG.choice(LOYALTY_TIERS),
                },
            )
        )

        # Behavioural events across the customer's lifetime.
        n_events = RNG.randint(3, 25)
        base_ts = profile_ts + timedelta(days=RNG.randint(1, 30))
        for _ in range(n_events):
            ts = base_ts + timedelta(days=RNG.randint(0, 180))
            if ts > NOW:
                ts = NOW - timedelta(days=RNG.randint(1, 10))
            eid += 1
            pid = RNG.choice(product_ids)
            value = round(RNG.uniform(5, 200), 2)

            r = RNG.random()
            if r < 0.05:
                events.append(
                    _make_event(eid, customer_id, ts, "web_page_view", {"page_url": f"/shop/{pid}"})
                )
            elif r < 0.12:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "web_product_view",
                        {"product_id": pid, "page_url": f"/product/{pid}"},
                    )
                )
            elif r < 0.18:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "web_search",
                        {"query": RNG.choice(["shoes", "laptop", "jacket", "book"])},
                    )
                )
            elif r < 0.22:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "web_click",
                        {"element": RNG.choice(["banner", "recommendation", "nav"])},
                    )
                )
            elif r < 0.30:
                events.append(_make_event(eid, customer_id, ts, "add_to_cart", {"product_id": pid}))
            elif r < 0.42:
                loc = RNG.choice(STORE_LOCATIONS)
                payload: dict[str, object] = {
                    "product_id": pid,
                    "order_value": value,
                    "payment_method": RNG.choice(PAYMENT_METHODS),
                }
                if loc:
                    payload["store_location"] = loc
                    events.append(_make_event(eid, customer_id, ts, "retail_purchase", payload))
                else:
                    events.append(_make_event(eid, customer_id, ts, "online_purchase", payload))
            elif r < 0.48:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "return_initiated",
                        {
                            "product_id": pid,
                            "return_reason": RNG.choice(
                                ["wrong_size", "defective", "changed_mind"]
                            ),
                        },
                    )
                )
            elif r < 0.52:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "product_review",
                        {"product_id": pid, "rating": RNG.randint(1, 5)},
                    )
                )
            elif r < 0.58:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "email_send",
                        {"campaign_id": RNG.choice([c for c in CAMPAIGN_IDS if c])},
                    )
                )
            elif r < 0.62:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "email_open",
                        {"campaign_id": RNG.choice([c for c in CAMPAIGN_IDS if c])},
                    )
                )
            elif r < 0.66:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "email_click",
                        {"campaign_id": RNG.choice([c for c in CAMPAIGN_IDS if c])},
                    )
                )
            elif r < 0.70:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "sms_send",
                        {"campaign_id": RNG.choice([c for c in CAMPAIGN_IDS if c])},
                    )
                )
            elif r < 0.74:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "sms_click",
                        {"campaign_id": RNG.choice([c for c in CAMPAIGN_IDS if c])},
                    )
                )
            elif r < 0.78:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "app_push_send",
                        {"title": RNG.choice(["Flash Sale!", "New arrivals", "Back in stock"])},
                    )
                )
            elif r < 0.82:
                events.append(_make_event(eid, customer_id, ts, "app_push_open", {}))
            elif r < 0.86:
                events.append(_make_event(eid, customer_id, ts, "app_push_click", {}))
            elif r < 0.90:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "login",
                        {"device": RNG.choice(["ios", "android", "web"])},
                    )
                )
            elif r < 0.93:
                events.append(_make_event(eid, customer_id, ts, "address_update", {}))
            elif r < 0.96:
                events.append(
                    _make_event(
                        eid,
                        customer_id,
                        ts,
                        "support_ticket_created",
                        {"category": RNG.choice(["billing", "shipping", "technical"])},
                    )
                )
            else:
                events.append(_make_event(eid, customer_id, ts, "support_chat_started", {}))

    # Extract a top-level "value" column for outcome construction.
    for event in events:
        payload = event.get("event_payload_json", {})
        event["value"] = (
            float(payload.get("order_value", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        )

    events.sort(key=lambda r: (r["customer_id"], r["event_ts"], r["event_id"]))
    return events[:N_EVENTS] if len(events) > N_EVENTS else events


# ---------------------------------------------------------------------------
# 2.  Product embeddings (static, self-supervised reconstruction).
# ---------------------------------------------------------------------------

logger.info("=== Stage 1: product embeddings ===")
product_records = _gen_products()
prod_model = create_embedding_model(
    id_field="product_id",
    categorical_fields=["category", "brand"],
    numeric_fields=["base_price", "unit_cost"],
    hidden_dim=HIDDEN_DIM,
    epochs=8,
    device="cpu",
    sequence_backend="mamba2",
    show_progress=True,
    progress_label="Product embeds",
)
product_embs = prod_model.fit_transform(product_records)
prod_lancedb = Path(tempfile.mkdtemp(prefix="lg_example_prods_"))
save_embeddings_to_lancedb(
    prod_lancedb, "product_embeddings", product_records, "product_id", product_embs
)
logger.info(
    "  products=%d  vectors=%d  dim=%d  loss=%.4f",
    len(product_records),
    len(product_embs),
    HIDDEN_DIM,
    prod_model.loss_,
)

# ---------------------------------------------------------------------------
# 3.  Event stream generation.
# ---------------------------------------------------------------------------

logger.info("=== Stage 2: event stream (%d customers, %d target events) ===", N_CUSTOMERS, N_EVENTS)
events = _gen_event_stream()
event_types = sorted({e["event_type"] for e in events})
logger.info(
    "  events=%d  customers=%d  event_types=%d",
    len(events),
    len({e["customer_id"] for e in events}),
    len(event_types),
)
for et in event_types:
    logger.info("    %-30s %5d", et, sum(1 for e in events if e["event_type"] == et))

# ---------------------------------------------------------------------------
# 4.  Payload schema — declares how the tokenizer reads event_payload_json.
# ---------------------------------------------------------------------------

payload_schema = PayloadSchema(
    categorical_fields=["payment_method", "device"],
    numeric_fields=["order_value", "rating"],
    vector_id_keys=["product_id"],
)

product_lookup: dict[str, list[float]] = {k: list(v) for k, v in product_embs.vectors.items()}

# ---------------------------------------------------------------------------
# 5.  Temporal core pretraining with payload-schema tokenization.
#     Only events BEFORE the outcome cutoff are included, so the backbone
#     does not see future information that would leak into downstream heads.
# ---------------------------------------------------------------------------

logger.info("=== Stage 3: temporal core pretraining ===")
cutoff_iso = OUTCOME_CUTOFF.isoformat()
core_events = [e for e in events if str(e["event_ts"]) <= cutoff_iso]
logger.info("  events before cutoff: %d / %d", len(core_events), len(events))

core_model = create_temporal_core_model(
    sequence_id_field="customer_id",
    event_id_field="event_id",
    timestamp_field="event_ts",
    categorical_fields=[],
    numeric_fields=[],
    vector_fields=[],
    payload_schema=payload_schema,
    vector_lookups={"product_id": product_lookup},
    hidden_dim=HIDDEN_DIM,
    epochs=25,
    device="cpu",
    sequence_backend="mamba2",
    train_batch_size=64,
    show_progress=True,
    progress_label="Temporal core",
    backbone_version="v1",
)
core_outputs = core_model.fit_transform(core_events)
logger.info(
    "  loss=%.4f  customers=%d  mean_events=%.1f",
    core_model.loss_ or 0,
    len(core_outputs.customer_records),
    sum(float(r["core_event_count"]) for r in core_outputs.customer_records)  # type: ignore[operator]
    / max(len(core_outputs.customer_records), 1),
)

pretrained_core = core_model.trained_core
assert pretrained_core is not None
wrap_core_with_qdora(pretrained_core, QDoRAConfig(rank=4, quantize_base=False))
logger.info("  core retrofitted with QDoRA (rank=4)")

core_lookup = {
    str(r["customer_id"]): {  # type: ignore[index]
        "core_last_vector": r["core_last_vector"],
        "core_mean_vector": r["core_mean_vector"],
        "core_event_count": r["core_event_count"],
    }
    for r in core_outputs.customer_records
}

# ---------------------------------------------------------------------------
# 6.  Leakage-safe supervised outcomes.
# ---------------------------------------------------------------------------

logger.info("=== Stage 4: supervised outcomes ===")
outcome_spec = LabelSpec(
    id_field="customer_id",
    timestamp_field="event_ts",
    as_of=OUTCOME_CUTOFF,
    history_days=300,
    horizon_days=90,
    value_field="value",
    label_kind="value_sum",
    min_history_events=2,
)
outcome_frame = build_outcomes(events, outcome_spec)
logger.info(
    "  outcome rows=%d  churn_rate=%.2f  mean_ltv=%.2f",
    len(outcome_frame),
    sum(float(r["churn_label"]) for r in outcome_frame.rows) / max(len(outcome_frame), 1),
    sum(float(r["value_label"]) for r in outcome_frame.rows) / max(len(outcome_frame), 1),
)

for row in outcome_frame.rows:
    cid = str(row["customer_id"])
    row.update(core_lookup.get(cid, {}))
    row["customer_vector"] = [0.0] * HIDDEN_DIM

outcome_rows = outcome_frame.rows
agg_features = [f for f in outcome_frame.feature_fields if f != "distinct_active_days"]

# ---------------------------------------------------------------------------
# 7.  Churn + LTV heads — three-way ablation (vectors vs aggregates vs both)
#     against an xgboost-on-aggregates baseline.
# ---------------------------------------------------------------------------

HEAD_EPOCHS = 60


def _train_churn(variant_name, numeric_fields, vector_fields):
    """Train a classification head and return (f1, roc_auc)."""
    m = create_supervised_model(
        task="classification",
        id_field="customer_id",
        target_field="churn_label",
        categorical_fields=[],
        numeric_fields=numeric_fields,
        vector_fields=vector_fields,
        hidden_dim=HIDDEN_DIM,
        epochs=HEAD_EPOCHS,
        seed=SEED,
        learning_rate=1e-2,
        validation_fraction=0.25,
        device="cpu",
        sequence_backend="mamba2",
        pretrained_core=pretrained_core,
    )
    r = m.fit_predict(outcome_rows)
    mets = r.report.metrics
    auc = float(mets["roc_auc"])
    if auc < 0.5:
        logger.warning(
            "  *** churn AUC = %.4f (< 0.5) for variant %s — check for leakage or inverted labels",
            auc,
            variant_name,
        )
    return float(mets["f1"]), auc


def _train_ltv(variant_name, numeric_fields, vector_fields):
    """Train a regression head and return R²."""
    m = create_supervised_model(
        task="regression",
        id_field="customer_id",
        target_field="value_label",
        categorical_fields=[],
        numeric_fields=numeric_fields,
        vector_fields=vector_fields,
        hidden_dim=HIDDEN_DIM,
        epochs=HEAD_EPOCHS,
        seed=SEED,
        learning_rate=1e-2,
        validation_fraction=0.25,
        device="cpu",
        sequence_backend="mamba2",
        pretrained_core=pretrained_core,
    )
    r = m.fit_predict(outcome_rows)
    return float(r.report.metrics["r2"])


logger.info("=== Stage 5: churn ablation ===")
t0 = time.perf_counter()
churn_vec = _train_churn("vectors-only", [], ["core_last_vector", "customer_vector"])
churn_agg = _train_churn("aggregates-only", agg_features, [])
churn_both = _train_churn("combined", agg_features, ["core_last_vector", "customer_vector"])
churn_time = time.perf_counter() - t0

logger.info("=== Stage 6: LTV ablation ===")
t0 = time.perf_counter()
ltv_vec = _train_ltv("vectors-only", [], ["core_last_vector", "customer_vector"])
ltv_agg = _train_ltv("aggregates-only", agg_features, [])
ltv_both = _train_ltv("combined", agg_features, ["core_last_vector", "customer_vector"])
ltv_time = time.perf_counter() - t0

logger.info("=== Stage 7: baseline (xgboost) ===")
t0 = time.perf_counter()
churn_bl = GBTBaseline("classification", "customer_id", "churn_label", agg_features, seed=SEED)
churn_bl_r = churn_bl.fit_predict(outcome_rows)
ltv_bl = GBTBaseline("regression", "customer_id", "value_label", agg_features, seed=SEED)
ltv_bl_r = ltv_bl.fit_predict(outcome_rows)
bl_time = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 8.  Report.
# ---------------------------------------------------------------------------


def _delta(model_val, baseline_val, higher_is_better=True):
    try:
        d = float(model_val) - float(baseline_val)
    except (TypeError, ValueError):
        return "n/a"
    win = (d > 0 and higher_is_better) or (d < 0 and not higher_is_better)
    return f"{d:+.4f} {'*' if win else ''}"


print()
print(
    "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
)
print("║     looking_glass — frozen-core LoRA paradigm + ablation                            ║")
print(
    "╠══════════════════════════════════════════════════════════════════════════════════════════╣"
)
print(
    f"║  products        {len(product_records):>5d}   customers  {len(core_outputs.customer_records):>5d}   events  {len(events):>6d}   event types  {len(event_types):>2d}     ║"
)
print(
    f"║  outcome rows    {len(outcome_rows):>5d}   core loss  {core_model.loss_:>8.4f}                                          ║"
)
print(
    "╠══════════════════════════════════════════════════════════════════════════════════════════╣"
)
print(
    f"║  churn F1 (vectors)   {churn_vec[0]:>10.4f}   vs xgboost {churn_bl_r.metrics['f1']:>10.4f}   delta {_delta(churn_vec[0], churn_bl_r.metrics['f1']):>10s} ║"
)
print(
    f"║  churn F1 (aggrs)    {churn_agg[0]:>10.4f}   vs xgboost {churn_bl_r.metrics['f1']:>10.4f}   delta {_delta(churn_agg[0], churn_bl_r.metrics['f1']):>10s} ║"
)
print(
    f"║  churn F1 (combined) {churn_both[0]:>10.4f}   vs xgboost {churn_bl_r.metrics['f1']:>10.4f}   delta {_delta(churn_both[0], churn_bl_r.metrics['f1']):>10s} ║"
)
print(
    f"║  churn AUC (vectors) {churn_vec[1]:>10.4f}   vs xgboost {churn_bl_r.metrics['roc_auc']:>10.4f}   delta {_delta(churn_vec[1], churn_bl_r.metrics['roc_auc']):>10s} ║"
)
print(
    "╠══════════════════════════════════════════════════════════════════════════════════════════╣"
)
print(
    f"║  LTV R² (vectors)   {ltv_vec:>10.4f}   vs xgboost {ltv_bl_r.metrics['r2']:>10.4f}   delta {_delta(ltv_vec, ltv_bl_r.metrics['r2']):>10s} ║"
)
print(
    f"║  LTV R² (aggrs)     {ltv_agg:>10.4f}   vs xgboost {ltv_bl_r.metrics['r2']:>10.4f}   delta {_delta(ltv_agg, ltv_bl_r.metrics['r2']):>10s} ║"
)
print(
    f"║  LTV R² (combined)  {ltv_both:>10.4f}   vs xgboost {ltv_bl_r.metrics['r2']:>10.4f}   delta {_delta(ltv_both, ltv_bl_r.metrics['r2']):>10s} ║"
)
print(
    "╠══════════════════════════════════════════════════════════════════════════════════════════╣"
)
print(
    f"║  churn heads  {churn_time:>5.1f}s  |  LTV heads  {ltv_time:>5.1f}s  |  xgboost  {bl_time:>5.1f}s                                      ║"
)
print(
    "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
)
print()
print("* = model outperformed baseline on this metric")
