"""Full-pipeline benchmark against the omnichannel retail simulacrum.

Reads dimension tables + event stream from the SQLite database produced by
``generate_full.py``, runs the complete frozen-core LoRA paradigm, and
produces a side-by-side comparison with an xgboost baseline.

Usage:
    python scripts/run_full.py [--db scripts/data/full/simulacrum.db]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
    wrap_core_with_qdora,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HIDDEN_DIM = 128
SEED = 42

PAYLOAD_SCHEMA = PayloadSchema(
    categorical_fields=["payment_method", "device", "return_reason", "category"],
    numeric_fields=["order_value", "rating"],
    vector_id_keys=["product_id", "store_id", "campaign_id"],
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
OUTCOME_CUTOFF = NOW.replace(year=2025, month=9, day=1)  # ~4 months of labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_full")


def _load_table(db: Path, table: str) -> list[dict[str, object]]:
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        cols = [d[1] for d in conn.execute(f"PRAGMA table_info({table})")]
    return [dict(zip(cols, r)) for r in rows]


def _parse_payload(row: dict) -> dict[str, object]:
    raw = row.get("event_payload_json", "{}")
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return raw if isinstance(raw, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full frozen-core LoRA benchmark")
    parser.add_argument("--db", type=Path, default=Path("scripts/data/full/simulacrum.db"))
    parser.add_argument("--product-epochs", type=int, default=6)
    parser.add_argument("--core-epochs", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--core-batch-size", type=int, default=64,
                        help="Temporal core train batch size (lower for 8 GB GPUs)")
    parser.add_argument("--event-limit", type=int, default=0,
                        help="Cap events for fast smoke runs (0 = all)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    import torch as _torch
    _torch.backends.cudnn.benchmark = True

    t_total = time.perf_counter()

    # ── 1. Load data ────────────────────────────────────────────────────

    logger.info("=== loading data ===")
    products = _load_table(args.db, "products")
    customers = _load_table(args.db, "customers")
    stores = _load_table(args.db, "stores")
    campaigns = _load_table(args.db, "campaigns")
    events = _load_table(args.db, "events")
    logger.info("  products=%d  customers=%d  stores=%d  campaigns=%d  events=%d",
                len(products), len(customers), len(stores), len(campaigns), len(events))

    # Parse payload JSON strings back to dicts for the tokenizer.
    for e in events:
        e["event_payload_json"] = _parse_payload(e)

    if args.event_limit > 0 and len(events) > args.event_limit:
        events = events[:args.event_limit]
        logger.info("  capped to %d events", len(events))

    # ── 2. Product embeddings ──────────────────────────────────────────

    logger.info("=== product embeddings ===")
    t0 = time.perf_counter()
    prod_model = create_embedding_model(
        id_field="product_id",
        categorical_fields=["category", "subcategory", "brand", "inventory_tier"],
        numeric_fields=["base_price", "unit_cost", "margin_pct"],
        hidden_dim=HIDDEN_DIM,
        epochs=args.product_epochs,
        seed=SEED,
        device=args.device,
        sequence_backend="mamba2",
        show_progress=True,
        progress_label="Product embeds",
    )
    product_embs = prod_model.fit_transform(products)
    product_lookup_row = {k: list(v) for k, v in product_embs.vectors.items()}
    logger.info("  products=%d  loss=%.4f  time=%.1fs", len(product_embs), prod_model.loss_, time.perf_counter() - t0)
    del prod_model

    t0 = time.perf_counter()
    store_model = create_embedding_model(
        id_field="store_id",
        categorical_fields=["store_type", "region", "city"],
        numeric_fields=["square_footage", "staff_count", "opened_year"],
        hidden_dim=HIDDEN_DIM, epochs=args.product_epochs, seed=SEED,
        device=args.device, sequence_backend="mamba2",
        show_progress=True, progress_label="Store embeds",
    )
    store_embs = store_model.fit_transform(stores)
    store_lookup = {k: list(v) for k, v in store_embs.vectors.items()}
    logger.info("  stores=%d  loss=%.4f  time=%.1fs", len(store_embs), store_model.loss_, time.perf_counter() - t0)
    del store_model

    t0 = time.perf_counter()
    campaign_model = create_embedding_model(
        id_field="campaign_id",
        categorical_fields=["campaign_type", "channel", "target_segment"],
        numeric_fields=["discount_pct"],
        hidden_dim=HIDDEN_DIM, epochs=args.product_epochs, seed=SEED,
        device=args.device, sequence_backend="mamba2",
        show_progress=True, progress_label="Campaign embeds",
    )
    campaign_embs = campaign_model.fit_transform(campaigns)
    campaign_lookup = {k: list(v) for k, v in campaign_embs.vectors.items()}
    logger.info("  campaigns=%d  loss=%.4f  time=%.1fs", len(campaign_embs), campaign_model.loss_, time.perf_counter() - t0)
    del campaign_model

    vector_lookups: dict[str, dict[str, list[float]]] = {
        "product_id": product_lookup_row,
        "store_id": store_lookup,
        "campaign_id": campaign_lookup,
    }
    if str(args.device) != "cpu":
        __import__("torch").cuda.empty_cache()

    # ── 3. Temporal core pretraining ───────────────────────────────────
    # Only events on or before the outcome cutoff — otherwise the backbone
    # sees post-cutoff activity and leaks future information into
    # core_last_vector, inflating downstream head metrics.

    logger.info("=== temporal core pretraining ===")
    core_events = [e for e in events if datetime.fromisoformat(str(e["event_ts"]).replace("Z", "+00:00")) <= OUTCOME_CUTOFF]
    logger.info("  events before cutoff: %d / %d", len(core_events), len(events))

    t0 = time.perf_counter()
    core_model = create_temporal_core_model(
        sequence_id_field="customer_id",
        event_id_field="event_id",
        timestamp_field="event_ts",
        categorical_fields=[],
        numeric_fields=[],
        vector_fields=[],
        payload_schema=PAYLOAD_SCHEMA,
        vector_lookups=vector_lookups,
        hidden_dim=HIDDEN_DIM,
        epochs=args.core_epochs,
        seed=SEED,
        learning_rate=1e-3,
        device=args.device,
        sequence_backend="mamba2",
        train_batch_size=args.core_batch_size,
        show_progress=True,
        progress_label="Temporal core",
        backbone_version="v1",
    )
    core_outputs = core_model.fit_transform(core_events)
    core_loss = core_model.loss_ or 0.0
    logger.info("  loss=%.4f  customers=%d  mean_events=%.1f  time=%.1fs",
                core_loss, len(core_outputs.customer_records),
                sum(float(r["core_event_count"]) for r in core_outputs.customer_records) / max(len(core_outputs.customer_records), 1),
                time.perf_counter() - t0)

    # ── 4. Retrofit with QDoRA ─────────────────────────────────────────

    pretrained_core: object | None = core_model.trained_core
    assert pretrained_core is not None
    wrap_core_with_qdora(pretrained_core, QDoRAConfig(rank=8, quantize_base=False))

    core_lookup = {
        str(r["customer_id"]): {
            "core_last_vector": r["core_last_vector"],
            "core_mean_vector": r["core_mean_vector"],
            "core_event_count": r["core_event_count"],
        }
        for r in core_outputs.customer_records
    }
    n_customers = len(core_outputs.customer_records)

    # Free GPU memory held by the temporal core training artefacts.
    del core_model, core_outputs
    if str(args.device) != "cpu":
        import torch as _torch2
        _torch2.cuda.empty_cache()

    # ── 5. Supervised outcomes ─────────────────────────────────────────

    logger.info("=== supervised outcomes ===")
    # Extract order_value from payload as a top-level value for the outcome builder.
    for e in events:
        payload = e.get("event_payload_json", {})
        e["value"] = float(payload.get("order_value", 0.0) or 0.0) if isinstance(payload, dict) else 0.0

    outcome_spec = LabelSpec(
        id_field="customer_id",
        timestamp_field="event_ts",
        as_of=OUTCOME_CUTOFF,
        history_days=365,
        horizon_days=120,
        value_field="value",
        label_kind="value_sum",
        min_history_events=3,
        active_lookback_days=90,
        min_recent_events=1,
    )
    outcome_frame = build_outcomes(events, outcome_spec)
    logger.info("  outcome rows=%d  churn_rate=%.3f  mean_ltv=%.2f",
                len(outcome_frame),
                sum(float(r["churn_label"]) for r in outcome_frame.rows) / max(len(outcome_frame), 1),
                sum(float(r["value_label"]) for r in outcome_frame.rows) / max(len(outcome_frame), 1))

    for row in outcome_frame.rows:
        cid = str(row["customer_id"])
        row.update(core_lookup.get(cid, {}))
        row["customer_vector"] = [0.0] * HIDDEN_DIM  # no separate customer model

    outcome_rows = outcome_frame.rows
    agg_features = [f for f in outcome_frame.feature_fields if f != "distinct_active_days"]

    # ── 6. Ablation: churn and LTV heads — three variants each ──────────
    #
    # vectors-only: the backbone's learned representation alone.
    # aggregates-only: hand-crafted features (same input as the xgboost baseline).
    # combined: both vectors and hand-crafted features together.
    #
    # Each variant is trained independently with the same seed.
    # The xgboost baseline uses aggregates-only by construction.

    head_kwargs = dict(
        sequence_backend="mamba2",
        pretrained_core=pretrained_core,
        hidden_dim=HIDDEN_DIM,
        epochs=args.head_epochs,
        seed=SEED,
        learning_rate=1e-2,
        validation_fraction=0.25,
        device=args.device,
    )

    def _run_churn(name, numeric_fields, vector_fields):
        m = create_supervised_model(
            task="classification", id_field="customer_id", target_field="churn_label",
            categorical_fields=[], numeric_fields=numeric_fields,
            vector_fields=vector_fields, **head_kwargs,
        )
        r = m.fit_predict(outcome_rows)
        return r.report.metrics

    def _run_ltv(name, numeric_fields, vector_fields):
        m = create_supervised_model(
            task="regression", id_field="customer_id", target_field="value_label",
            categorical_fields=[], numeric_fields=numeric_fields,
            vector_fields=vector_fields, **head_kwargs,
        )
        r = m.fit_predict(outcome_rows)
        return r.report.metrics

    logger.info("=== churn ablation ===")
    t0 = time.perf_counter()
    churn_vec = _run_churn("vectors-only", [], ["customer_vector", "core_last_vector"])
    churn_agg = _run_churn("aggregates-only", agg_features, [])
    churn_combined = _run_churn("combined", agg_features, ["customer_vector", "core_last_vector"])
    churn_time = time.perf_counter() - t0

    logger.info("=== LTV ablation ===")
    t0 = time.perf_counter()
    ltv_vec = _run_ltv("vectors-only", [], ["customer_vector", "core_last_vector"])
    ltv_combined = _run_ltv("combined", agg_features, ["customer_vector", "core_last_vector"])
    ltv_time = time.perf_counter() - t0

    # ── 7. Baseline (xgboost on aggregates) ──────────────────────────────

    logger.info("=== baseline (xgboost) ===")
    t0 = time.perf_counter()
    churn_bl = GBTBaseline("classification", "customer_id", "churn_label", agg_features, seed=SEED).fit_predict(outcome_rows)
    ltv_bl = GBTBaseline("regression", "customer_id", "value_label", agg_features, seed=SEED).fit_predict(outcome_rows)
    bl_time = time.perf_counter() - t0

    # ── 8. Report ──────────────────────────────────────────────────────

    total_time = time.perf_counter() - t_total

    def _delta(m, b, higher_is_better=True):
        d = float(m) - float(b)
        win = (d > 0 and higher_is_better) or (d < 0 and not higher_is_better)
        return f"{d:+.4f} {'*' if win else ''}"

    def _warn_auc(auc_val, label):
        if auc_val < 0.5:
            print(f"║  *** WARNING: {label} AUC = {auc_val:.4f} (<0.5) — check for leakage/inversion ***  ║")

    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print("║     looking_glass — omnichannel retail simulacrum benchmark + ablation             ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  products          {len(products):>6d}   stores        {len(stores):>6d}   campaigns      {len(campaigns):>6d}      ║")
    print(f"║  customers         {n_customers:>6d}   core events    {len(core_events):>6d}   outcome rows    {len(outcome_rows):>6d}      ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  temporal core loss     {core_loss:>8.4f}                                                ║")
    print("╠══════════════════════════╦══════════════╦══════════════╦══════════════╗")
    print("║  churn variant           ║  F1          ║  AUC         ║  vs xgboost  ║")
    print("╠══════════════════════════╬══════════════╬══════════════╬══════════════╣")
    bl_f1 = float(churn_bl.metrics["f1"])
    bl_auc = float(churn_bl.metrics.get("roc_auc", 0.5))
    for label, met in [("vectors-only  ", churn_vec), ("aggregates-only", churn_agg), ("combined       ", churn_combined)]:
        f1v = float(met["f1"])
        aucv = float(met.get("roc_auc", 0.5))
        _warn_auc(aucv, label.strip())
        print(f"║  {label:<20s}    ║  {f1v:>10.4f}  ║  {aucv:>10.4f}  ║  {_delta(f1v, bl_f1):>12s}  ║")
    print(f"║  xgboost                 ║  {bl_f1:>10.4f}  ║  {bl_auc:>10.4f}  ║  (baseline)  ║")
    print("╠══════════════════════════╬══════════════╬══════════════╬══════════════╣")
    print("║  LTV variant             ║  R²          ║  RMSE        ║  vs xgboost  ║")
    print("╠══════════════════════════╬══════════════╬══════════════╬══════════════╣")
    bl_r2 = float(ltv_bl.metrics["r2"])
    bl_rmse = float(ltv_bl.metrics["rmse"])
    for label, met in [("vectors-only  ", ltv_vec), ("combined       ", ltv_combined)]:
        r2v = float(met["r2"])
        rmsev = float(met["rmse"])
        print(f"║  {label:<20s}    ║  {r2v:>10.4f}  ║  {rmsev:>10.4f}  ║  {_delta(r2v, bl_r2):>12s}  ║")
    print(f"║  xgboost                 ║  {bl_r2:>10.4f}  ║  {bl_rmse:>10.4f}  ║  (baseline)  ║")
    print("╠══════════════════════════╩══════════════╩══════════════╩══════════════╣")
    print(f"║  total time  {total_time:>5.0f}s   churn heads {churn_time:>5.1f}s   LTV heads {ltv_time:>5.1f}s   xgboost {bl_time:>5.1f}s        ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════════╝")
    print()
    print("* = model outperformed baseline on this metric")


if __name__ == "__main__":
    main()
