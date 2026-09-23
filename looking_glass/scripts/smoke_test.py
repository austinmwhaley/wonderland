"""End-to-end smoke pipeline for representation and outcome quality.

This script validates the full modeling stack on synthetic data:
1. Train product embeddings.
2. Train customer embeddings.
3. Enrich customer event stream with those vectors.
4. Train temporal core on event sequences.
5. Build leakage-safe supervised outcomes.
6. Train churn and LTV heads.
7. Enforce baseline gates and optionally enforce stricter "good quality" gates.

Design note:
The script favors explicit stages and verbose logging over compactness so it
is easy to audit failures and reason about data-quality regressions.
"""

# Standard library imports used for timestamp math and path bootstrapping.
from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
from tqdm import tqdm

# Ensure the repository root is importable when this file is run directly.
# This makes `python scripts/smoke_test.py` work without installing the package.
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_data import main as generate_data_main
from looking_glass import (
    attach_vector_feature,
    create_embedding_model,
    create_supervised_model,
    create_temporal_core_model,
    get_sequence_backend_name,
    get_sequence_implementation_name,
    load_vectors_from_lancedb,
    load_records_from_sqlite,
    QDoRAConfig,
    save_embeddings_to_lancedb,
    validate_classification_success,
    validate_embeddings,
    validate_regression_success,
    validate_prediction_report,
    wrap_core_with_qdora,
    GBTBaseline,
    ranking_metrics,
)

SQLITE_PATH = Path("scripts/data/sqlite/events.db")
LANCEDB_DIR = Path("scripts/data/lancedb")
HIDDEN_DIM = 128
SEED = 17
DEFAULT_PRODUCT_EPOCHS = 120
DEFAULT_CUSTOMER_EPOCHS = 120
DEFAULT_CORE_EPOCHS = 120
DEFAULT_OUTCOME_EPOCHS = 80


def _int_env(name: str, default: int) -> int:
    """Read positive integer env var with fallback to ``default``."""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _optional_int_env(name: str) -> int | None:
    """Read optional positive integer env var, returning ``None`` if unset/invalid."""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _float_env(name: str, default: float) -> float:
    """Read float env var with fallback to ``default`` on parse failure."""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_float_env(name: str) -> float | None:
    """Read optional float env var, returning ``None`` if unset/invalid."""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


PRODUCT_EPOCHS = _int_env("SMOKE_PRODUCT_EPOCHS", DEFAULT_PRODUCT_EPOCHS)
CUSTOMER_EPOCHS = _int_env("SMOKE_CUSTOMER_EPOCHS", DEFAULT_CUSTOMER_EPOCHS)
CORE_EPOCHS = _int_env("SMOKE_CORE_EPOCHS", DEFAULT_CORE_EPOCHS)
# Keep smoke defaults conservative for attention-heavy temporal training.
CORE_TRAIN_BATCH_SIZE = _int_env("SMOKE_CORE_BATCH_SIZE", 128)
OUTCOME_EPOCHS = _int_env("SMOKE_OUTCOME_EPOCHS", DEFAULT_OUTCOME_EPOCHS)
SMOKE_EVENT_LIMIT = _optional_int_env("SMOKE_EVENT_LIMIT")
OUTCOME_HISTORY_DAYS = _int_env("SMOKE_OUTCOME_HISTORY_DAYS", 365)
OUTCOME_LABEL_DAYS = _int_env("SMOKE_OUTCOME_LABEL_DAYS", 120)
OUTCOME_MIN_PAST_ORDERS = _int_env("SMOKE_OUTCOME_MIN_PAST_ORDERS", 2)
OUTCOME_ACTIVE_LOOKBACK_DAYS = _int_env("SMOKE_OUTCOME_ACTIVE_LOOKBACK_DAYS", 120)
OUTCOME_MIN_RECENT_ORDERS = _int_env("SMOKE_OUTCOME_MIN_RECENT_ORDERS", 1)
OUTCOME_TARGET_MODE = os.getenv("SMOKE_OUTCOME_TARGET_MODE", "auto").strip().lower()
if OUTCOME_TARGET_MODE not in {"event", "order", "auto"}:
    OUTCOME_TARGET_MODE = "auto"

OUTCOME_MIN_CHURN_POSITIVE_RATE = _float_env("SMOKE_OUTCOME_MIN_CHURN_POSITIVE_RATE", 0.02)
OUTCOME_MAX_CHURN_POSITIVE_RATE = _float_env("SMOKE_OUTCOME_MAX_CHURN_POSITIVE_RATE", 0.98)
OUTCOME_MIN_NONZERO_LTV_RATE = _float_env("SMOKE_OUTCOME_MIN_NONZERO_LTV_RATE", 0.02)

OUTCOME_MIN_CHURN_POSITIVE_RATE = min(max(OUTCOME_MIN_CHURN_POSITIVE_RATE, 0.0), 1.0)
OUTCOME_MAX_CHURN_POSITIVE_RATE = min(max(OUTCOME_MAX_CHURN_POSITIVE_RATE, 0.0), 1.0)
OUTCOME_MIN_NONZERO_LTV_RATE = min(max(OUTCOME_MIN_NONZERO_LTV_RATE, 0.0), 1.0)
if OUTCOME_MAX_CHURN_POSITIVE_RATE <= OUTCOME_MIN_CHURN_POSITIVE_RATE:
    OUTCOME_MIN_CHURN_POSITIVE_RATE = 0.02
    OUTCOME_MAX_CHURN_POSITIVE_RATE = 0.98

SMOKE_SEQUENCE_BACKEND = os.getenv("SMOKE_SEQUENCE_BACKEND", "samba").strip().lower()

PRODUCT_EMBED_MIN_NORM_STD = _float_env("SMOKE_PRODUCT_EMBED_MIN_NORM_STD", 0.01)
CUSTOMER_EMBED_MIN_NORM_STD = _float_env("SMOKE_CUSTOMER_EMBED_MIN_NORM_STD", 0.005)
EVENT_EMBED_MIN_NORM_STD = _float_env("SMOKE_EVENT_EMBED_MIN_NORM_STD", 0.005)
EMBED_MAX_MEAN_ABS_COSINE = _float_env("SMOKE_EMBED_MAX_MEAN_ABS_COSINE", 0.98)

CHURN_MIN_F1 = _float_env("SMOKE_CHURN_MIN_F1", 0.25)
CHURN_MIN_RECALL = _float_env("SMOKE_CHURN_MIN_RECALL", 0.25)
CHURN_MIN_PRECISION = _float_env("SMOKE_CHURN_MIN_PRECISION", 0.10)
LTV_MIN_R2 = _float_env("SMOKE_LTV_MIN_R2", 0.02)

# "Good" quality bar for component-level determination.
PRODUCT_GOOD_MIN_NORM_STD = _float_env("SMOKE_PRODUCT_GOOD_MIN_NORM_STD", 0.02)
PRODUCT_GOOD_MAX_MEAN_ABS_COSINE = _float_env("SMOKE_PRODUCT_GOOD_MAX_MEAN_ABS_COSINE", 0.85)

CUSTOMER_GOOD_MIN_NORM_STD = _float_env("SMOKE_CUSTOMER_GOOD_MIN_NORM_STD", 0.05)
CUSTOMER_GOOD_MAX_MEAN_ABS_COSINE = _float_env("SMOKE_CUSTOMER_GOOD_MAX_MEAN_ABS_COSINE", 0.35)

EVENT_GOOD_MIN_NORM_STD = _float_env("SMOKE_EVENT_GOOD_MIN_NORM_STD", 0.08)
EVENT_GOOD_MAX_MEAN_ABS_COSINE = _float_env("SMOKE_EVENT_GOOD_MAX_MEAN_ABS_COSINE", 0.65)

CORE_GOOD_MAX_LOSS = _float_env("SMOKE_CORE_GOOD_MAX_LOSS", 3.0)
CORE_GOOD_MIN_CUSTOMERS = _int_env("SMOKE_CORE_GOOD_MIN_CUSTOMERS", 1000)
CORE_GOOD_MIN_MEAN_EVENTS = _float_env("SMOKE_CORE_GOOD_MIN_MEAN_EVENTS", 5.0)

CHURN_GOOD_MIN_ACCURACY = _float_env("SMOKE_CHURN_GOOD_MIN_ACCURACY", 0.70)
CHURN_GOOD_MIN_PRECISION = _float_env("SMOKE_CHURN_GOOD_MIN_PRECISION", 0.60)
CHURN_GOOD_MIN_RECALL = _float_env("SMOKE_CHURN_GOOD_MIN_RECALL", 0.70)
CHURN_GOOD_MIN_F1 = _float_env("SMOKE_CHURN_GOOD_MIN_F1", 0.70)

LTV_GOOD_MIN_R2 = _float_env("SMOKE_LTV_GOOD_MIN_R2", 0.45)
LTV_GOOD_MAX_RMSE = _optional_float_env("SMOKE_LTV_GOOD_MAX_RMSE")
if LTV_GOOD_MAX_RMSE is not None and LTV_GOOD_MAX_RMSE <= 0.0:
    LTV_GOOD_MAX_RMSE = None

CUSTOMER_DEVICE = "auto"
CORE_DEVICE = "auto"
GOOD_QUALITY_MODE = os.getenv("SMOKE_ENFORCE_GOOD_QUALITY", "auto").strip().lower()
if GOOD_QUALITY_MODE not in {"auto", "0", "1", "false", "true", "no", "yes"}:
    GOOD_QUALITY_MODE = "auto"

REDUCED_SMOKE_BUDGET = (
    PRODUCT_EPOCHS < DEFAULT_PRODUCT_EPOCHS
    or CUSTOMER_EPOCHS < DEFAULT_CUSTOMER_EPOCHS
    or CORE_EPOCHS < DEFAULT_CORE_EPOCHS
    or OUTCOME_EPOCHS < DEFAULT_OUTCOME_EPOCHS
    or SMOKE_EVENT_LIMIT is not None
)
if GOOD_QUALITY_MODE == "auto":
    ENFORCE_GOOD_QUALITY = not REDUCED_SMOKE_BUDGET
else:
    ENFORCE_GOOD_QUALITY = GOOD_QUALITY_MODE in {"1", "true", "yes"}

# This is a smoke script, so defaults intentionally trade perfect accuracy
# for runtime stability and easy reproducibility on a laptop/workstation.


class _StageProgress:
    """Track high-level run progress with elapsed time per stage."""

    def __init__(self, total_stages: int) -> None:
        self._bar = tqdm(total=total_stages, desc="Smoke pipeline", unit="stage", file=sys.stderr)

    @contextmanager
    def stage(self, name: str):
        """Context manager that logs stage timing and advances progress bar."""

        _log = logging.getLogger("smoke_test")
        start = perf_counter()
        _log.info("[START] %s", name)
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            _log.info("[DONE]  %s  (%.2fs)", name, elapsed)
            self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


def _resolve_device_label(requested: str) -> str:
    """Resolve a user/device config string into a display label like CUDA/CPU."""
    value = requested.lower().strip()
    if value == "auto":
        if torch.cuda.is_available():
            return "CUDA"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS"
        return "CPU"
    if value.startswith("cuda"):
        return "CUDA"
    if value.startswith("mps"):
        return "MPS"
    return "CPU"


def _build_customer_outcome_records(
    order_records: list[dict[str, object]],
    cutoff_ts: datetime,
    history_days: int,
    label_days: int,
    min_past_orders: int,
    active_lookback_days: int,
    min_recent_orders: int,
) -> list[dict[str, object]]:
    """Build leakage-safe churn/LTV rows using fixed history and future windows."""
    if not order_records:
        return []

    history_start_ts = cutoff_ts - timedelta(days=history_days)
    label_end_ts = cutoff_ts + timedelta(days=label_days)
    recent_start_ts = cutoff_ts - timedelta(days=30)
    active_start_ts = cutoff_ts - timedelta(days=active_lookback_days)

    # Aggregate all events into a single per-customer bucket keyed by id.
    per_customer: dict[str, dict[str, float | int]] = {}
    for row in order_records:
        order_ts = datetime.fromisoformat(str(row["order_ts"]).replace("Z", "+00:00"))
        if order_ts < history_start_ts or order_ts > label_end_ts:
            continue

        customer_id = str(row["customer_id"])
        bucket = per_customer.setdefault(
            customer_id,
            {
                "past_order_count": 0,
                "past_completed_count": 0,
                "past_cancelled_count": 0,
                "past_refunded_count": 0,
                "past_return_count": 0,
                "past_total_spend": 0.0,
                "past_total_return": 0.0,
                "past_max_order_total": 0.0,
                "past_last_order_ts": 0.0,
                "past_recent_order_count_30d": 0,
                "past_recent_spend_30d": 0.0,
                "past_active_order_count": 0,
                "future_order_count": 0,
                "future_completed_count": 0,
                "future_total_spend": 0.0,
                "future_total_return": 0.0,
            },
        )

        order_total = float(row.get("order_total", 0.0) or 0.0)
        return_amount = float(row.get("return_amount", 0.0) or 0.0)
        status = str(row.get("order_status", ""))

        if order_ts <= cutoff_ts:
            # History window features available at prediction time.
            bucket["past_order_count"] = int(bucket["past_order_count"]) + 1
            bucket["past_total_spend"] = float(bucket["past_total_spend"]) + order_total
            bucket["past_total_return"] = float(bucket["past_total_return"]) + return_amount
            bucket["past_max_order_total"] = max(float(bucket["past_max_order_total"]), order_total)
            bucket["past_last_order_ts"] = max(float(bucket["past_last_order_ts"]), order_ts.timestamp())

            if order_ts >= recent_start_ts:
                bucket["past_recent_order_count_30d"] = int(bucket["past_recent_order_count_30d"]) + 1
                bucket["past_recent_spend_30d"] = float(bucket["past_recent_spend_30d"]) + order_total
            if order_ts >= active_start_ts:
                bucket["past_active_order_count"] = int(bucket["past_active_order_count"]) + 1

            if status == "completed":
                bucket["past_completed_count"] = int(bucket["past_completed_count"]) + 1
            elif status == "cancelled":
                bucket["past_cancelled_count"] = int(bucket["past_cancelled_count"]) + 1
            elif status == "refunded":
                bucket["past_refunded_count"] = int(bucket["past_refunded_count"]) + 1

            if int(row.get("return_flag", 0) or 0) == 1:
                bucket["past_return_count"] = int(bucket["past_return_count"]) + 1
        else:
            # Future window labels/targets for supervision only.
            bucket["future_order_count"] = int(bucket["future_order_count"]) + 1
            bucket["future_total_spend"] = float(bucket["future_total_spend"]) + order_total
            bucket["future_total_return"] = float(bucket["future_total_return"]) + return_amount
            if status == "completed":
                bucket["future_completed_count"] = int(bucket["future_completed_count"]) + 1

    results: list[dict[str, object]] = []
    for customer_id, bucket in per_customer.items():
        past_order_count = int(bucket["past_order_count"])
        if past_order_count < min_past_orders:
            continue
        if int(bucket["past_active_order_count"]) < min_recent_orders:
            continue

        last_order_unix = float(bucket["past_last_order_ts"])
        if last_order_unix <= 0.0:
            continue

        last_order_ts = datetime.fromtimestamp(last_order_unix, tz=cutoff_ts.tzinfo)
        recency_days = float(max((cutoff_ts - last_order_ts).days, 0))
        avg_order_total = float(bucket["past_total_spend"]) / max(past_order_count, 1)

        return_rate = float(bucket["past_return_count"]) / max(past_order_count, 1)
        cancel_rate = float(bucket["past_cancelled_count"]) / max(past_order_count, 1)
        complete_rate = float(bucket["past_completed_count"]) / max(past_order_count, 1)

        future_net_ltv = max(float(bucket["future_total_spend"]) - float(bucket["future_total_return"]), 0.0)
        churn_label = 1.0 if int(bucket["future_completed_count"]) == 0 else 0.0

        results.append(
            {
                "customer_id": customer_id,
                "order_count": float(past_order_count),
                "completed_count": float(bucket["past_completed_count"]),
                "cancelled_count": float(bucket["past_cancelled_count"]),
                "refunded_count": float(bucket["past_refunded_count"]),
                "return_count": float(bucket["past_return_count"]),
                "total_spend": float(bucket["past_total_spend"]),
                "avg_order_total": avg_order_total,
                "max_order_total": float(bucket["past_max_order_total"]),
                "recency_days": recency_days,
                "recent_order_count_30d": float(bucket["past_recent_order_count_30d"]),
                "recent_spend_30d": float(bucket["past_recent_spend_30d"]),
                "active_order_count": float(bucket["past_active_order_count"]),
                "return_rate": return_rate,
                "cancel_rate": cancel_rate,
                "complete_rate": complete_rate,
                "future_order_count": float(bucket["future_order_count"]),
                "churn_label": churn_label,
                "ltv_value": future_net_ltv,
            }
        )

    return results


def _build_event_outcome_records(
    event_records: list[dict[str, object]],
    cutoff_ts: datetime,
    history_days: int,
    label_days: int,
    min_past_events: int,
    active_lookback_days: int,
    min_recent_events: int,
) -> list[dict[str, object]]:
    """Build leakage-safe outcomes from customer event activity/value windows."""
    if not event_records:
        return []

    history_start_ts = cutoff_ts - timedelta(days=history_days)
    label_end_ts = cutoff_ts + timedelta(days=label_days)
    recent_start_ts = cutoff_ts - timedelta(days=30)
    active_start_ts = cutoff_ts - timedelta(days=active_lookback_days)

    # Keep event-based fallback targets available when order-only labels degenerate.
    per_customer: dict[str, dict[str, float | int]] = {}
    for row in event_records:
        event_ts = datetime.fromisoformat(str(row["event_ts"]).replace("Z", "+00:00"))
        if event_ts < history_start_ts or event_ts > label_end_ts:
            continue

        customer_id = str(row["customer_id"])
        bucket = per_customer.setdefault(
            customer_id,
            {
                "past_event_count": 0,
                "past_total_value": 0.0,
                "past_max_value": 0.0,
                "past_last_event_ts": 0.0,
                "past_recent_count_30d": 0,
                "past_recent_value_30d": 0.0,
                "past_active_count": 0,
                "past_order_event_count": 0,
                "past_product_event_count": 0,
                "future_event_count": 0,
                "future_total_value": 0.0,
            },
        )

        event_value = float(row.get("value", 0.0) or 0.0)
        source_table = str(row.get("source_table", ""))
        entity_type = str(row.get("entity_type", ""))

        if event_ts <= cutoff_ts:
            bucket["past_event_count"] = int(bucket["past_event_count"]) + 1
            bucket["past_total_value"] = float(bucket["past_total_value"]) + max(event_value, 0.0)
            bucket["past_max_value"] = max(float(bucket["past_max_value"]), max(event_value, 0.0))
            bucket["past_last_event_ts"] = max(float(bucket["past_last_event_ts"]), event_ts.timestamp())

            if event_ts >= recent_start_ts:
                bucket["past_recent_count_30d"] = int(bucket["past_recent_count_30d"]) + 1
                bucket["past_recent_value_30d"] = float(bucket["past_recent_value_30d"]) + max(event_value, 0.0)
            if event_ts >= active_start_ts:
                bucket["past_active_count"] = int(bucket["past_active_count"]) + 1

            if source_table == "orders":
                bucket["past_order_event_count"] = int(bucket["past_order_event_count"]) + 1
            if entity_type == "product":
                bucket["past_product_event_count"] = int(bucket["past_product_event_count"]) + 1
        else:
            bucket["future_event_count"] = int(bucket["future_event_count"]) + 1
            bucket["future_total_value"] = float(bucket["future_total_value"]) + max(event_value, 0.0)

    results: list[dict[str, object]] = []
    for customer_id, bucket in per_customer.items():
        past_event_count = int(bucket["past_event_count"])
        if past_event_count < min_past_events:
            continue
        if int(bucket["past_active_count"]) < min_recent_events:
            continue

        last_event_unix = float(bucket["past_last_event_ts"])
        if last_event_unix <= 0.0:
            continue

        last_event_ts = datetime.fromtimestamp(last_event_unix, tz=cutoff_ts.tzinfo)
        recency_days = float(max((cutoff_ts - last_event_ts).days, 0))

        avg_value = float(bucket["past_total_value"]) / max(past_event_count, 1)
        order_event_ratio = float(bucket["past_order_event_count"]) / max(past_event_count, 1)
        product_event_ratio = float(bucket["past_product_event_count"]) / max(past_event_count, 1)

        future_event_count = int(bucket["future_event_count"])
        future_total_value = float(bucket["future_total_value"])
        churn_label = 1.0 if future_event_count == 0 else 0.0

        results.append(
            {
                "customer_id": customer_id,
                "order_count": float(past_event_count),
                "completed_count": float(bucket["past_order_event_count"]),
                "cancelled_count": 0.0,
                "refunded_count": 0.0,
                "return_count": 0.0,
                "total_spend": float(bucket["past_total_value"]),
                "avg_order_total": avg_value,
                "max_order_total": float(bucket["past_max_value"]),
                "recency_days": recency_days,
                "recent_order_count_30d": float(bucket["past_recent_count_30d"]),
                "recent_spend_30d": float(bucket["past_recent_value_30d"]),
                "active_order_count": float(bucket["past_active_count"]),
                "return_rate": 0.0,
                "cancel_rate": 0.0,
                "complete_rate": order_event_ratio,
                "product_event_ratio": product_event_ratio,
                "future_order_count": float(future_event_count),
                "churn_label": churn_label,
                "ltv_value": future_total_value,
            }
        )

    return results


def _build_customer_dimension_records(
    customer_records: list[dict[str, object]],
    reference_ts: datetime,
) -> list[dict[str, object]]:
    """Project customer dimension rows into model-ready static features.

    The output is used to train the customer embedding model, which produces a
    reusable static representation for each customer.
    """
    results: list[dict[str, object]] = []
    for row in customer_records:
        signup_ts = datetime.fromisoformat(str(row["signup_ts"]).replace("Z", "+00:00"))
        results.append(
            {
                "customer_id": str(row["customer_id"]),
                "gender": str(row.get("gender", "")),
                "country": str(row.get("country", "")),
                "state_region": str(row.get("state_region", "")),
                "city": str(row.get("city", "")),
                "postal_code": str(row.get("postal_code", "")),
                "loyalty_tier": str(row.get("loyalty_tier", "")),
                "income_band": str(row.get("income_band", "")),
                "acquisition_channel": str(row.get("acquisition_channel", "")),
                "lifecycle_stage": str(row.get("lifecycle_stage", "")),
                # Numeric demographic/account signals.
                "birth_year": float(row.get("birth_year", 0) or 0),
                "cardholder_status": float(row.get("cardholder_status", 0) or 0),
                # Tenure gives a time-normalized customer-age feature.
                "tenure_days": float((reference_ts - signup_ts).days),
            }
        )
    return results


def _attach_customer_lookup_fields(
    records: list[dict[str, object]],
    lookup: dict[str, dict[str, object]],
    id_field: str,
) -> list[dict[str, object]]:
    """Attach precomputed feature blocks by id (left-join style)."""
    enriched: list[dict[str, object]] = []
    for row in records:
        payload = dict(row)
        payload.update(lookup.get(str(row[id_field]), {}))
        enriched.append(payload)
    return enriched


def _assert_ok(name: str, checks: dict[str, object], enforce: bool = True) -> None:
    """Fail fast on invalid checks, or warn when strict enforcement is disabled."""

    if bool(checks.get("ok", False)):
        return
    if enforce:
        raise RuntimeError(f"{name} validation failed: {checks}")
    logging.getLogger("smoke_test").warning(
        "%s validation failed but strict good-quality enforcement is disabled: %s",
        name,
        checks,
    )


def _embedding_good_quality(
    validation: dict[str, object],
    min_norm_std: float,
    max_mean_abs_cosine: float,
    min_nonzero_fraction: float = 0.99,
) -> dict[str, object]:
    """Apply stricter geometric quality gates on top of baseline embedding checks."""

    norm_std = float(validation.get("norm_std", 0.0) or 0.0)
    mean_abs_cosine = float(validation.get("mean_abs_cosine", 1.0) or 1.0)
    nonzero_fraction = float(validation.get("nonzero_fraction", 0.0) or 0.0)

    checks = {
        "base_ok": bool(validation.get("ok", False)),
        "norm_std_ok": norm_std >= float(min_norm_std),
        "nonzero_ok": nonzero_fraction >= float(min_nonzero_fraction),
        "cosine_ok": mean_abs_cosine <= float(max_mean_abs_cosine),
    }
    return {
        "ok": all(checks.values()),
        **checks,
        "norm_std": norm_std,
        "nonzero_fraction": nonzero_fraction,
        "mean_abs_cosine": mean_abs_cosine,
        "required_min_norm_std": float(min_norm_std),
        "required_min_nonzero_fraction": float(min_nonzero_fraction),
        "required_max_mean_abs_cosine": float(max_mean_abs_cosine),
    }


def _temporal_core_good_quality(
    core_validation: dict[str, object],
    core_loss: float,
    customer_records: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate temporal core against embedding, loss, and coverage constraints."""

    customer_count = len(customer_records)
    mean_events = 0.0
    if customer_count > 0:
        mean_events = (
            sum(float(row.get("core_event_count", 0.0) or 0.0) for row in customer_records)
            / customer_count
        )

    checks = {
        "embedding_ok": bool(core_validation.get("ok", False)),
        "loss_ok": float(core_loss) <= float(CORE_GOOD_MAX_LOSS),
        "customer_count_ok": customer_count >= int(CORE_GOOD_MIN_CUSTOMERS),
        "mean_events_ok": mean_events >= float(CORE_GOOD_MIN_MEAN_EVENTS),
    }
    return {
        "ok": all(checks.values()),
        **checks,
        "loss": float(core_loss),
        "customer_count": customer_count,
        "mean_events_per_customer": mean_events,
        "required_max_loss": float(CORE_GOOD_MAX_LOSS),
        "required_min_customers": int(CORE_GOOD_MIN_CUSTOMERS),
        "required_min_mean_events": float(CORE_GOOD_MIN_MEAN_EVENTS),
    }


# ---------------------------------------------------------------------------
# Single fixed log file — always written to the same path so you can `tail -f`
# it across runs. Each run overwrites cleanly (mode="w").
# ---------------------------------------------------------------------------
_LOG_PATH = Path("scripts/data/logs/smoke_test.log")


def _setup_logging() -> logging.Logger:
    """Configure root logger with a stream handler and a fixed-path file handler."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # capture epoch-level debug lines too
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)  # terminal only shows INFO+
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return logging.getLogger("smoke_test")


if __name__ == "__main__":
    logger = _setup_logging()
    sequence_backend = get_sequence_backend_name(SMOKE_SEQUENCE_BACKEND)
    sequence_implementation = get_sequence_implementation_name()
    logger.info("Log: %s", _LOG_PATH.resolve())
    logger.info(
        (
            "Smoke config: product_epochs=%d customer_epochs=%d core_epochs=%d "
            "outcome_epochs=%d core_batch=%d event_limit=%s "
            "history_days=%d label_days=%d min_past_orders=%d "
            "active_lookback_days=%d min_recent_orders=%d target_mode=%s "
            "good_quality_mode=%s good_quality_enforced=%s sequence_backend=%s sequence_impl=%s "
            "churn_rate_range=[%.2f, %.2f] min_nonzero_ltv_rate=%.2f "
            "good_gate: core_max_loss=%.2f core_min_customers=%d core_min_mean_events=%.2f "
            "churn_good(f1>=%.2f,precision>=%.2f,recall>=%.2f,accuracy>=%.2f) "
            "ltv_good(r2>=%.2f,rmse<=%s)"
        ),
        PRODUCT_EPOCHS,
        CUSTOMER_EPOCHS,
        CORE_EPOCHS,
        OUTCOME_EPOCHS,
        CORE_TRAIN_BATCH_SIZE,
        "all" if SMOKE_EVENT_LIMIT is None else str(SMOKE_EVENT_LIMIT),
        OUTCOME_HISTORY_DAYS,
        OUTCOME_LABEL_DAYS,
        OUTCOME_MIN_PAST_ORDERS,
        OUTCOME_ACTIVE_LOOKBACK_DAYS,
        OUTCOME_MIN_RECENT_ORDERS,
        OUTCOME_TARGET_MODE,
        GOOD_QUALITY_MODE,
        ENFORCE_GOOD_QUALITY,
        sequence_backend,
        sequence_implementation,
        OUTCOME_MIN_CHURN_POSITIVE_RATE,
        OUTCOME_MAX_CHURN_POSITIVE_RATE,
        OUTCOME_MIN_NONZERO_LTV_RATE,
        CORE_GOOD_MAX_LOSS,
        CORE_GOOD_MIN_CUSTOMERS,
        CORE_GOOD_MIN_MEAN_EVENTS,
        CHURN_GOOD_MIN_F1,
        CHURN_GOOD_MIN_PRECISION,
        CHURN_GOOD_MIN_RECALL,
        CHURN_GOOD_MIN_ACCURACY,
        LTV_GOOD_MIN_R2,
        "none" if LTV_GOOD_MAX_RMSE is None else f"{LTV_GOOD_MAX_RMSE:.2f}",
    )

    run_progress = _StageProgress(total_stages=8)
    # ---------------------------------------------------------------------
    # Stage 0: (Optional) synthetic data generation/bootstrap.
    # ---------------------------------------------------------------------
    # Optional bootstrap: regenerate source data if needed.
    # Keep this off for normal iterative runs because it's expensive.
    # generate_data_main()

    product_device_label = _resolve_device_label("auto")
    customer_device_label = _resolve_device_label(CUSTOMER_DEVICE)
    core_device_label = _resolve_device_label(CORE_DEVICE)

    try:
        # -----------------------------------------------------------------
        # Stage 1: Product embedding model (entity silo #1).
        # -----------------------------------------------------------------
        with run_progress.stage(f"product embedding model ({product_device_label})"):
            # 1) Load raw entity records from SQLite.
            records = load_records_from_sqlite(
                sqlite_path=SQLITE_PATH,
                source_table="products",
                columns=["product_id", "category", "base_price"],
            )

            # 2) Train product siloed embedding model.
            model = create_embedding_model(
                id_field="product_id",
                categorical_fields=["category"],
                numeric_fields=["base_price"],
                hidden_dim=HIDDEN_DIM,
                epochs=PRODUCT_EPOCHS,
                seed=SEED,
                sequence_backend=sequence_backend,
                show_progress=True,
                progress_label="Product embedding epochs",
            )
            embeddings = model.fit_transform(records)

            # 3) Persist vectors + selected metadata for retrieval/debugging.
            product_rows = embeddings.save(
                lancedb_dir=LANCEDB_DIR,
                output_table="product_embeddings",
                keep_fields=["category"],
            )
            product_validation = validate_embeddings(
                embeddings.vectors,
                min_norm_std=PRODUCT_EMBED_MIN_NORM_STD,
                min_nonzero_fraction=0.99,
                max_mean_abs_cosine=EMBED_MAX_MEAN_ABS_COSINE,
            )
            _assert_ok("product_embeddings", product_validation)
            product_quality = _embedding_good_quality(
                product_validation,
                min_norm_std=PRODUCT_GOOD_MIN_NORM_STD,
                max_mean_abs_cosine=PRODUCT_GOOD_MAX_MEAN_ABS_COSINE,
            )
            _assert_ok("product_embedding_quality", product_quality, enforce=ENFORCE_GOOD_QUALITY)

        # -----------------------------------------------------------------
        # Stage 2: Customer embedding model (entity silo #2).
        # -----------------------------------------------------------------
        with run_progress.stage(f"customer embedding model ({customer_device_label})"):
            # Use all customers (no row cap) so the customer silo sees full coverage.
            customer_dimension_rows = load_records_from_sqlite(
                sqlite_path=SQLITE_PATH,
                source_table="customers",
                columns=[
                    "customer_id",
                    "signup_ts",
                    "birth_year",
                    "gender",
                    "country",
                    "state_region",
                    "city",
                    "postal_code",
                    "loyalty_tier",
                    "cardholder_status",
                    "income_band",
                    "acquisition_channel",
                    "lifecycle_stage",
                ],
            )
            if not customer_dimension_rows:
                raise RuntimeError("No customer rows loaded for customer embedding run")

            # Anchor tenure to latest observed event time in the dataset.
            latest_event_row = load_records_from_sqlite(
                sqlite_path=SQLITE_PATH,
                source_table="customer_events",
                columns=["event_ts"],
                order_by="event_ts DESC",
                limit=1,
            )
            if not latest_event_row:
                raise RuntimeError("No customer events found for tenure reference timestamp")
            latest_event_ts = datetime.fromisoformat(
                str(latest_event_row[0]["event_ts"]).replace("Z", "+00:00")
            )

            customer_dimension_records = _build_customer_dimension_records(
                customer_records=customer_dimension_rows,
                reference_ts=latest_event_ts,
            )

            customer_model = create_embedding_model(
                id_field="customer_id",
                categorical_fields=[
                    "gender",
                    "country",
                    "state_region",
                    "city",
                    "postal_code",
                    "loyalty_tier",
                    "income_band",
                    "acquisition_channel",
                    "lifecycle_stage",
                ],
                numeric_fields=["birth_year", "cardholder_status", "tenure_days"],
                hidden_dim=HIDDEN_DIM,
                epochs=CUSTOMER_EPOCHS,
                seed=SEED,
                device=CUSTOMER_DEVICE,
                sequence_backend=sequence_backend,
                show_progress=True,
                progress_label="Customer embedding epochs",
            )
            customer_embeddings = customer_model.fit_transform(customer_dimension_records)
            customer_rows = customer_embeddings.save(
                lancedb_dir=LANCEDB_DIR,
                output_table="customer_embeddings",
                keep_fields=["country", "loyalty_tier", "lifecycle_stage"],
            )
            customer_validation = validate_embeddings(
                customer_embeddings.vectors,
                min_norm_std=CUSTOMER_EMBED_MIN_NORM_STD,
                min_nonzero_fraction=0.99,
                max_mean_abs_cosine=EMBED_MAX_MEAN_ABS_COSINE,
            )
            _assert_ok("customer_embeddings", customer_validation)
            customer_quality = _embedding_good_quality(
                customer_validation,
                min_norm_std=CUSTOMER_GOOD_MIN_NORM_STD,
                max_mean_abs_cosine=CUSTOMER_GOOD_MAX_MEAN_ABS_COSINE,
            )
            _assert_ok("customer_embedding_quality", customer_quality, enforce=ENFORCE_GOOD_QUALITY)

        # -----------------------------------------------------------------
        # Stage 3: Load event stream + attach siloed vectors.
        # -----------------------------------------------------------------
        with run_progress.stage("event enrichment"):
            latest_order_row = load_records_from_sqlite(
                sqlite_path=SQLITE_PATH,
                source_table="orders",
                columns=["order_ts"],
                order_by="order_ts DESC",
                limit=1,
            )
            if not latest_order_row:
                raise RuntimeError("No order rows found to derive leakage cutoff")

            latest_order_ts = datetime.fromisoformat(
                str(latest_order_row[0]["order_ts"]).replace("Z", "+00:00")
            )
            outcome_cutoff_ts = latest_order_ts - timedelta(days=OUTCOME_LABEL_DAYS)
            outcome_cutoff_iso = outcome_cutoff_ts.isoformat()

            logger.info("event enrichment: load customer events")
            customer_event_records = load_records_from_sqlite(
                sqlite_path=SQLITE_PATH,
                source_table="customer_events",
                where="event_ts <= ?",
                params=(outcome_cutoff_iso,),
                order_by="event_ts DESC" if SMOKE_EVENT_LIMIT is not None else "event_ts",
                limit=SMOKE_EVENT_LIMIT,
                columns=[
                    "event_id",
                    "customer_id",
                    "event_ts",
                    "event_type",
                    "entity_type",
                    "source_table",
                    "entity_id",
                    "value",
                ],
            )
            if SMOKE_EVENT_LIMIT is not None:
                customer_event_records.reverse()
                logger.info(
                    "event enrichment: using latest %d historical customer events before cutoff=%s",
                    len(customer_event_records),
                    outcome_cutoff_iso,
                )
            else:
                logger.info(
                    "event enrichment: using full historical event stream before cutoff=%s",
                    outcome_cutoff_iso,
                )
            if len(customer_event_records) < 2:
                raise RuntimeError("Not enough customer event rows for temporal core training")

            # Attach static customer vectors to every event row by customer_id.
            logger.info("event enrichment: attach customer vectors")
            customer_event_records = attach_vector_feature(
                records=customer_event_records,
                lookup_key="customer_id",
                vector_lookup=customer_embeddings.vectors,
                output_field="customer_vector",
                show_progress=True,
                progress_label="Enrichment: customer vectors",
            )
            customer_hits = sum(
                1
                for row in customer_event_records
                if str(row.get("customer_id", "")) in customer_embeddings.vectors
            )
            customer_hit_rate = customer_hits / max(len(customer_event_records), 1)
            logger.info(
                "event enrichment: customer vector coverage %.2f%%",
                100.0 * customer_hit_rate,
            )
            if customer_hit_rate < 0.99:
                raise RuntimeError(
                    f"Customer vector coverage too low: {customer_hit_rate:.3f}"
                )

            logger.info("event enrichment: load product vectors")
            product_vectors = load_vectors_from_lancedb(
                lancedb_dir=LANCEDB_DIR,
                table_name="product_embeddings",
                id_field="product_id",
            )

            # Attach product vectors to event rows by entity_id when applicable.
            logger.info("event enrichment: attach product vectors")
            customer_event_records = attach_vector_feature(
                records=customer_event_records,
                lookup_key="entity_id",
                vector_lookup=product_vectors,
                output_field="product_vector",
                show_progress=True,
                progress_label="Enrichment: product vectors",
            )
            product_entity_rows = [
                row
                for row in customer_event_records
                if str(row.get("entity_type", "")).lower() == "product"
            ]
            if product_entity_rows:
                product_hits = sum(
                    1 for row in product_entity_rows if str(row.get("entity_id", "")) in product_vectors
                )
                product_hit_rate = product_hits / max(len(product_entity_rows), 1)
                logger.info(
                    "event enrichment: product vector coverage %.2f%% on product events",
                    100.0 * product_hit_rate,
                )
                if product_hit_rate < 0.85:
                    raise RuntimeError(
                        f"Product vector coverage too low on product events: {product_hit_rate:.3f}"
                    )

        # -----------------------------------------------------------------
        # Stage 4: Temporal core model over event sequences.
        # -----------------------------------------------------------------
        with run_progress.stage(f"temporal core model ({core_device_label})"):
            core_model = create_temporal_core_model(
                sequence_id_field="customer_id",
                event_id_field="event_id",
                timestamp_field="event_ts",
                categorical_fields=["event_type", "entity_type", "source_table"],
                numeric_fields=["value"],
                vector_fields=["customer_vector", "product_vector"],
                hidden_dim=HIDDEN_DIM,
                epochs=CORE_EPOCHS,
                seed=SEED,
                learning_rate=1e-3,
                device=CORE_DEVICE,
                sequence_backend=sequence_backend,
                train_batch_size=CORE_TRAIN_BATCH_SIZE,
                input_is_time_sorted=True,
                show_progress=True,
                progress_label="Temporal core",
            )
            core_outputs = core_model.fit_transform(customer_event_records)
            core_validation = validate_embeddings(
                core_outputs.event_embeddings,
                min_norm_std=EVENT_EMBED_MIN_NORM_STD,
                min_nonzero_fraction=0.99,
                max_mean_abs_cosine=EMBED_MAX_MEAN_ABS_COSINE,
            )
            _assert_ok("event_embeddings", core_validation)
            core_embedding_quality = _embedding_good_quality(
                core_validation,
                min_norm_std=EVENT_GOOD_MIN_NORM_STD,
                max_mean_abs_cosine=EVENT_GOOD_MAX_MEAN_ABS_COSINE,
            )
            _assert_ok("event_embedding_quality", core_embedding_quality, enforce=ENFORCE_GOOD_QUALITY)
            core_quality = _temporal_core_good_quality(
                core_validation=core_validation,
                core_loss=float(core_model.loss_ or 0.0),
                customer_records=core_outputs.customer_records,
            )
            _assert_ok("temporal_core_quality", core_quality, enforce=ENFORCE_GOOD_QUALITY)
            event_rows = save_embeddings_to_lancedb(
                lancedb_dir=LANCEDB_DIR,
                output_table="event_embeddings",
                records=customer_event_records,
                id_field="event_id",
                embeddings=core_outputs.event_embeddings,
                keep_fields=["customer_id", "event_type", "entity_type", "source_table"],
            )

            # Retrofit the trained core with QDoRA adapters so downstream
            # task heads train only lightweight LoRA params on a frozen backbone.
            qdora_core = core_model.trained_core
            if qdora_core is not None:
                wrap_core_with_qdora(qdora_core, QDoRAConfig(rank=8, quantize_base=False))
                logger.info("temporal core retrofitted with QDoRA for frozen-core downstream heads")

        # -----------------------------------------------------------------
        # Stage 5: Supervised outcomes from transactional history.
        # -----------------------------------------------------------------
        with run_progress.stage("build supervised outcomes"):
            # Pull per-customer temporal summaries emitted by the core model.
            core_customer_lookup = {
                str(row["customer_id"]): {
                    "core_last_vector": row["core_last_vector"],
                    "core_mean_vector": row["core_mean_vector"],
                    "core_event_count": row["core_event_count"],
                }
                for row in core_outputs.customer_records
            }
            if not core_customer_lookup:
                raise RuntimeError("No temporal customer summaries available for supervised outcomes")

            def _build_outcomes_for_mode(mode: str) -> list[dict[str, object]]:
                if mode == "order":
                    order_records = load_records_from_sqlite(
                        sqlite_path=SQLITE_PATH,
                        source_table="orders",
                        columns=[
                            "customer_id",
                            "order_ts",
                            "order_status",
                            "order_total",
                            "return_flag",
                            "return_amount",
                        ],
                    )
                    return _build_customer_outcome_records(
                        order_records=order_records,
                        cutoff_ts=outcome_cutoff_ts,
                        history_days=OUTCOME_HISTORY_DAYS,
                        label_days=OUTCOME_LABEL_DAYS,
                        min_past_orders=OUTCOME_MIN_PAST_ORDERS,
                        active_lookback_days=OUTCOME_ACTIVE_LOOKBACK_DAYS,
                        min_recent_orders=OUTCOME_MIN_RECENT_ORDERS,
                    )

                event_value_records = load_records_from_sqlite(
                    sqlite_path=SQLITE_PATH,
                    source_table="customer_events",
                    columns=[
                        "customer_id",
                        "event_ts",
                        "source_table",
                        "entity_type",
                        "value",
                    ],
                )
                return _build_event_outcome_records(
                    event_records=event_value_records,
                    cutoff_ts=outcome_cutoff_ts,
                    history_days=OUTCOME_HISTORY_DAYS,
                    label_days=OUTCOME_LABEL_DAYS,
                    min_past_events=OUTCOME_MIN_PAST_ORDERS,
                    active_lookback_days=OUTCOME_ACTIVE_LOOKBACK_DAYS,
                    min_recent_events=OUTCOME_MIN_RECENT_ORDERS,
                )

            primary_mode = "order" if OUTCOME_TARGET_MODE == "auto" else OUTCOME_TARGET_MODE
            fallback_mode = "event" if primary_mode == "order" else "order"
            mode_candidates = [primary_mode, fallback_mode]
            selection_issues: list[str] = []
            selected_mode: str | None = None
            selected_records: list[dict[str, object]] | None = None

            for mode in mode_candidates:
                outcome_records = _build_outcomes_for_mode(mode)
                if len(outcome_records) < 2:
                    issue = f"mode={mode}: not enough raw outcome rows ({len(outcome_records)})"
                    logger.warning("build supervised outcomes: %s", issue)
                    selection_issues.append(issue)
                    continue

                pre_filter_positive_rate = (
                    sum(float(row.get("churn_label", 0.0) or 0.0) for row in outcome_records)
                    / len(outcome_records)
                )
                logger.info(
                    "build supervised outcomes: mode=%s pre-filter rows=%d churn_positive_rate=%.4f",
                    mode,
                    len(outcome_records),
                    pre_filter_positive_rate,
                )

                core_feature_coverage = (
                    sum(1 for row in outcome_records if str(row["customer_id"]) in core_customer_lookup)
                    / len(outcome_records)
                )
                logger.info(
                    "build supervised outcomes: mode=%s temporal core coverage %.2f%%",
                    mode,
                    100.0 * core_feature_coverage,
                )
                retained_records = [
                    row
                    for row in outcome_records
                    if str(row["customer_id"]) in core_customer_lookup
                ]
                if len(retained_records) < 2:
                    issue = (
                        f"mode={mode}: not enough rows after temporal-coverage filter "
                        f"({len(retained_records)})"
                    )
                    logger.warning("build supervised outcomes: %s", issue)
                    selection_issues.append(issue)
                    continue

                churn_positive_rate = (
                    sum(float(row.get("churn_label", 0.0) or 0.0) for row in retained_records)
                    / len(retained_records)
                )
                nonzero_ltv_rate = (
                    sum(1 for row in retained_records if float(row.get("ltv_value", 0.0) or 0.0) > 0.0)
                    / len(retained_records)
                )
                ltv_mean = (
                    sum(float(row.get("ltv_value", 0.0) or 0.0) for row in retained_records)
                    / len(retained_records)
                )
                logger.info(
                    (
                        "build supervised outcomes: mode=%s retained rows=%d "
                        "churn_positive_rate=%.4f nonzero_ltv_rate=%.4f ltv_mean=%.2f"
                    ),
                    mode,
                    len(retained_records),
                    churn_positive_rate,
                    nonzero_ltv_rate,
                    ltv_mean,
                )

                if (
                    churn_positive_rate <= OUTCOME_MIN_CHURN_POSITIVE_RATE
                    or churn_positive_rate >= OUTCOME_MAX_CHURN_POSITIVE_RATE
                ):
                    issue = (
                        f"mode={mode}: degenerate churn label balance "
                        f"({churn_positive_rate:.4f}) outside "
                        f"[{OUTCOME_MIN_CHURN_POSITIVE_RATE:.4f}, {OUTCOME_MAX_CHURN_POSITIVE_RATE:.4f}]"
                    )
                    logger.warning("build supervised outcomes: %s", issue)
                    selection_issues.append(issue)
                    continue

                if nonzero_ltv_rate < OUTCOME_MIN_NONZERO_LTV_RATE:
                    issue = (
                        f"mode={mode}: sparse nonzero LTV coverage "
                        f"({nonzero_ltv_rate:.4f}) below {OUTCOME_MIN_NONZERO_LTV_RATE:.4f}"
                    )
                    logger.warning("build supervised outcomes: %s", issue)
                    selection_issues.append(issue)
                    continue

                selected_mode = mode
                selected_records = retained_records
                break

            if selected_records is None or selected_mode is None:
                issue_summary = "; ".join(selection_issues) if selection_issues else "unknown"
                raise RuntimeError(
                    "Unable to build non-degenerate supervised outcomes in any mode: "
                    f"{issue_summary}"
                )

            if selected_mode != primary_mode:
                logger.warning(
                    (
                        "build supervised outcomes: switched target mode from %s to %s "
                        "due to label degeneracy checks"
                    ),
                    primary_mode,
                    selected_mode,
                )
            logger.info("build supervised outcomes: selected target mode=%s", selected_mode)

            outcome_records = selected_records

            outcome_records = _attach_customer_lookup_fields(
                records=outcome_records,
                lookup=core_customer_lookup,
                id_field="customer_id",
            )
            outcome_records = attach_vector_feature(
                records=outcome_records,
                lookup_key="customer_id",
                vector_lookup=customer_embeddings.vectors,
                output_field="customer_vector",
            )

        # -----------------------------------------------------------------
        # Stage 6: Churn classification head (75/25 train/validation split).
        # -----------------------------------------------------------------
        with run_progress.stage(f"churn head ({core_device_label})"):
            churn_model = create_supervised_model(
                task="classification",
                id_field="customer_id",
                target_field="churn_label",
                categorical_fields=[],
                numeric_fields=[
                    "order_count",
                    "completed_count",
                    "cancelled_count",
                    "refunded_count",
                    "return_count",
                    "total_spend",
                    "avg_order_total",
                    "max_order_total",
                    "recency_days",
                    "recent_order_count_30d",
                    "recent_spend_30d",
                    "active_order_count",
                    "return_rate",
                    "cancel_rate",
                    "complete_rate",
                    "product_event_ratio",
                    "core_event_count",
                ],
                vector_fields=["customer_vector", "core_last_vector"],
                hidden_dim=HIDDEN_DIM,
                epochs=OUTCOME_EPOCHS,
                seed=SEED,
                learning_rate=1e-3,
                validation_fraction=0.25,
                device=CORE_DEVICE,
                sequence_backend=sequence_backend,
                show_progress=True,
                progress_label="Churn head epochs",
                pretrained_core=qdora_core,
            )
            churn_results = churn_model.fit_predict(outcome_records)
            churn_validation = validate_prediction_report(churn_results.report)
            _assert_ok("churn_model", churn_validation)

            # Require churn to do more than predict the majority class.
            churn_quality = validate_classification_success(
                churn_results.report,
                min_f1=CHURN_MIN_F1,
                min_recall=CHURN_MIN_RECALL,
                min_precision=CHURN_MIN_PRECISION,
            )
            _assert_ok("churn_quality", churn_quality)
            churn_good_quality = validate_classification_success(
                churn_results.report,
                min_accuracy=CHURN_GOOD_MIN_ACCURACY,
                min_precision=CHURN_GOOD_MIN_PRECISION,
                min_recall=CHURN_GOOD_MIN_RECALL,
                min_f1=CHURN_GOOD_MIN_F1,
            )
            _assert_ok("churn_good_quality", churn_good_quality, enforce=ENFORCE_GOOD_QUALITY)

            churn_scores = [float(churn_results.predictions[str(r["customer_id"])]) for r in outcome_records if str(r["customer_id"]) in churn_results.predictions]
            churn_labels = [float(r.get("churn_label", 0.0) or 0.0) for r in outcome_records if str(r["customer_id"]) in churn_results.predictions]
            churn_rank = ranking_metrics(churn_scores, churn_labels)
            churn_results.report.metrics.update(churn_rank)
            logger.info("Churn ranking: %s", {k: round(float(v), 4) for k, v in churn_rank.items()})

            churn_baseline = GBTBaseline(
                task="classification",
                id_field="customer_id",
                target_field="churn_label",
                feature_fields=[
                    "order_count", "completed_count", "cancelled_count", "refunded_count",
                    "return_count", "total_spend", "avg_order_total", "max_order_total",
                    "recency_days", "recent_order_count_30d", "recent_spend_30d",
                    "active_order_count", "return_rate", "cancel_rate", "complete_rate",
                    "core_event_count",
                ],
                seed=SEED,
            )
            churn_baseline_result = churn_baseline.fit_predict(outcome_records)
            logger.info(
                "Churn baseline (%s): f1=%.4f auc=%.4f",
                churn_baseline_result.implementation,
                churn_baseline_result.metrics.get("f1", 0),
                churn_baseline_result.metrics.get("roc_auc", 0),
            )

        # -----------------------------------------------------------------
        # Stage 7: LTV regression head (75/25 train/validation split).
        # -----------------------------------------------------------------
        with run_progress.stage(f"ltv head ({core_device_label})"):
            ltv_model = create_supervised_model(
                task="regression",
                id_field="customer_id",
                target_field="ltv_value",
                categorical_fields=[],
                numeric_fields=[
                    "order_count",
                    "completed_count",
                    "cancelled_count",
                    "refunded_count",
                    "return_count",
                    "total_spend",
                    "avg_order_total",
                    "max_order_total",
                    "recency_days",
                    "recent_order_count_30d",
                    "recent_spend_30d",
                    "active_order_count",
                    "return_rate",
                    "cancel_rate",
                    "complete_rate",
                    "product_event_ratio",
                    "core_event_count",
                ],
                vector_fields=["customer_vector", "core_mean_vector"],
                hidden_dim=HIDDEN_DIM,
                epochs=OUTCOME_EPOCHS,
                seed=SEED,
                learning_rate=1e-3,
                validation_fraction=0.25,
                device=CORE_DEVICE,
                sequence_backend=sequence_backend,
                show_progress=True,
                progress_label="LTV head epochs",
                pretrained_core=qdora_core,
            )
            ltv_results = ltv_model.fit_predict(outcome_records)
            ltv_validation = validate_prediction_report(ltv_results.report)
            _assert_ok("ltv_model", ltv_validation)
            ltv_quality = validate_regression_success(
                ltv_results.report,
                min_r2=LTV_MIN_R2,
            )
            _assert_ok("ltv_quality", ltv_quality)
            ltv_good_quality = validate_regression_success(
                ltv_results.report,
                min_r2=LTV_GOOD_MIN_R2,
                max_rmse=LTV_GOOD_MAX_RMSE,
            )
            _assert_ok("ltv_good_quality", ltv_good_quality, enforce=ENFORCE_GOOD_QUALITY)

            ltv_baseline = GBTBaseline(
                task="regression",
                id_field="customer_id",
                target_field="ltv_value",
                feature_fields=[
                    "order_count", "completed_count", "total_spend", "avg_order_total",
                    "max_order_total", "recency_days", "recent_order_count_30d",
                    "active_order_count", "return_rate", "complete_rate", "core_event_count",
                ],
                seed=SEED,
            )
            ltv_baseline_result = ltv_baseline.fit_predict(outcome_records)
            logger.info(
                "LTV baseline (%s): r2=%.4f rmse=%.2f",
                ltv_baseline_result.implementation,
                ltv_baseline_result.metrics.get("r2", 0),
                ltv_baseline_result.metrics.get("rmse", float("inf")),
            )

        # -----------------------------------------------------------------
        # Stage 8: Final run report (human-readable smoke-test summary).
        # -----------------------------------------------------------------
        with run_progress.stage("final report"):
            logger.info("Product embeddings on %s | rows=%d | loss=%.6f | %s", model.device_, product_rows, model.loss_, product_validation)
            logger.info("Product quality determination: %s", product_quality)
            logger.info("Customer embeddings on %s | rows=%d | loss=%.6f | %s", customer_model.device_, customer_rows, customer_model.loss_, customer_validation)
            logger.info("Customer quality determination: %s", customer_quality)
            logger.info("Temporal core on %s | event rows=%d | loss=%.6f | %s", core_model.device_, event_rows, core_model.loss_, core_validation)
            logger.info("Temporal core embedding quality determination: %s", core_embedding_quality)
            logger.info("Temporal core quality determination: %s", core_quality)
            logger.info("Churn metrics: %s", churn_results.report.metrics)
            logger.info("Churn validation: %s | quality: %s", churn_validation, churn_quality)
            logger.info("Churn good-quality determination: %s", churn_good_quality)
            logger.info("LTV metrics: %s", ltv_results.report.metrics)
            logger.info("LTV validation: %s | quality: %s", ltv_validation, ltv_quality)
            logger.info("LTV good-quality determination: %s", ltv_good_quality)

            baseline_quality = {
                "ok": all(
                    [
                        bool(product_validation.get("ok", False)),
                        bool(customer_validation.get("ok", False)),
                        bool(core_validation.get("ok", False)),
                        bool(churn_validation.get("ok", False)),
                        bool(churn_quality.get("ok", False)),
                        bool(ltv_validation.get("ok", False)),
                        bool(ltv_quality.get("ok", False)),
                    ]
                ),
                "product_embeddings": bool(product_validation.get("ok", False)),
                "customer_embeddings": bool(customer_validation.get("ok", False)),
                "temporal_event_embeddings": bool(core_validation.get("ok", False)),
                "churn_head": bool(churn_validation.get("ok", False)) and bool(churn_quality.get("ok", False)),
                "ltv_head": bool(ltv_validation.get("ok", False)) and bool(ltv_quality.get("ok", False)),
            }
            logger.info("Baseline quality determination: %s", baseline_quality)

            good_quality = {
                "ok": all(
                    [
                        bool(product_quality.get("ok", False)),
                        bool(customer_quality.get("ok", False)),
                        bool(core_embedding_quality.get("ok", False)),
                        bool(core_quality.get("ok", False)),
                        bool(churn_good_quality.get("ok", False)),
                        bool(ltv_good_quality.get("ok", False)),
                    ]
                ),
                "product_embeddings": bool(product_quality.get("ok", False)),
                "customer_embeddings": bool(customer_quality.get("ok", False)),
                "temporal_event_embeddings": bool(core_embedding_quality.get("ok", False)),
                "temporal_core": bool(core_quality.get("ok", False)),
                "churn_head": bool(churn_good_quality.get("ok", False)),
                "ltv_head": bool(ltv_good_quality.get("ok", False)),
            }
            logger.info("Good-quality determination: %s", good_quality)

            overall_quality = {
                "ok": bool(baseline_quality["ok"]) and (
                    bool(good_quality["ok"]) if ENFORCE_GOOD_QUALITY else True
                ),
                "good_quality_enforced": ENFORCE_GOOD_QUALITY,
                "baseline_ok": bool(baseline_quality["ok"]),
                "good_quality_ok": bool(good_quality["ok"]),
                "product_embeddings": bool(good_quality.get("product_embeddings", False)),
                "customer_embeddings": bool(good_quality.get("customer_embeddings", False)),
                "temporal_event_embeddings": bool(good_quality.get("temporal_event_embeddings", False)),
                "temporal_core": bool(good_quality.get("temporal_core", False)),
                "churn_head": bool(good_quality.get("churn_head", False)),
                "ltv_head": bool(good_quality.get("ltv_head", False)),
            }
            logger.info("Overall quality determination: %s", overall_quality)

            logger.info(
                "Paradigm: frozen-core LoRA | core shared=%s | heads trained with LoRA adapters",
                qdora_core is not None,
            )
            logger.info(
                "Churn model vs baseline: f1=%.4f vs %.4f | auc=%.4f vs %.4f",
                churn_results.report.metrics.get("f1", 0),
                churn_baseline_result.metrics.get("f1", 0),
                churn_results.report.metrics.get("roc_auc", 0),
                churn_baseline_result.metrics.get("roc_auc", 0),
            )
            logger.info(
                "LTV model vs baseline: r2=%.4f vs %.4f | rmse=%.2f vs %.2f",
                ltv_results.report.metrics.get("r2", 0),
                ltv_baseline_result.metrics.get("r2", 0),
                ltv_results.report.metrics.get("rmse", float("inf")),
                ltv_baseline_result.metrics.get("rmse", float("inf")),
            )
    except Exception:
        logger.exception("Smoke test failed")
        raise
    finally:
        run_progress.close()
        logging.info("Log written to: %s", _LOG_PATH.resolve())
