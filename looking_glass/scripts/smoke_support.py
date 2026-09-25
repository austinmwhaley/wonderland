"""Smoke-test helpers: stage progress, record builders, and quality gates."""

from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from time import perf_counter

import torch
from tqdm import tqdm

from looking_glass.scripts.smoke_config import (
    CORE_GOOD_MAX_LOSS,
    CORE_GOOD_MIN_CUSTOMERS,
    CORE_GOOD_MIN_MEAN_EVENTS,
)


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
            bucket["past_last_order_ts"] = max(
                float(bucket["past_last_order_ts"]), order_ts.timestamp()
            )

            if order_ts >= recent_start_ts:
                bucket["past_recent_order_count_30d"] = (
                    int(bucket["past_recent_order_count_30d"]) + 1
                )
                bucket["past_recent_spend_30d"] = (
                    float(bucket["past_recent_spend_30d"]) + order_total
                )
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

        future_net_ltv = max(
            float(bucket["future_total_spend"]) - float(bucket["future_total_return"]), 0.0
        )
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
            bucket["past_last_event_ts"] = max(
                float(bucket["past_last_event_ts"]), event_ts.timestamp()
            )

            if event_ts >= recent_start_ts:
                bucket["past_recent_count_30d"] = int(bucket["past_recent_count_30d"]) + 1
                bucket["past_recent_value_30d"] = float(bucket["past_recent_value_30d"]) + max(
                    event_value, 0.0
                )
            if event_ts >= active_start_ts:
                bucket["past_active_count"] = int(bucket["past_active_count"]) + 1

            if source_table == "orders":
                bucket["past_order_event_count"] = int(bucket["past_order_event_count"]) + 1
            if entity_type == "product":
                bucket["past_product_event_count"] = int(bucket["past_product_event_count"]) + 1
        else:
            bucket["future_event_count"] = int(bucket["future_event_count"]) + 1
            bucket["future_total_value"] = float(bucket["future_total_value"]) + max(
                event_value, 0.0
            )

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
    file_handler.setLevel(logging.DEBUG)  # capture epoch-level debug lines too
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)  # terminal only shows INFO+
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return logging.getLogger("smoke_test")
