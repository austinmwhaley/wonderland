"""Smoke-test configuration: env-driven budgets, gates, and paths."""

import os
from pathlib import Path

DUCKDB_PATH = Path("scripts/data/events.duckdb")
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
