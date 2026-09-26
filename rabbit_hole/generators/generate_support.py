"""Shared generator primitives: progress reporting and vectorized sampling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timezone as _tz
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True)
class CustomerEventRow:
    """Normalized customer event row used for smoke testing."""

    event_id: int
    customer_key: str
    event_ts: str
    brand: str | None
    event_type: str
    event_attributes: str
    entity_type: str | None
    entity_id: str | None
    source_table: str | None
    value: float | None


_REFERENCE_NOW = datetime(2026, 1, 1, tzinfo=_tz.utc)  # fixed data window end (deterministic)

# Intraday activity prior (hours 0..23): quiet nights, midday/evening peaks.
_HOUR_WEIGHTS = np.array(
    [
        0.02,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.02,
        0.03,
        0.05,
        0.06,
        0.06,
        0.06,
        0.05,
        0.05,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.09,
        0.07,
        0.05,
        0.03,
        0.02,
    ],
    dtype=np.float64,
)


class ProgressReporter:
    """Minimal terminal progress bar with optional timestamped log file."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path
        self._current_label: str | None = None
        self._current_total: int = 0
        self._current_count: int = 0
        self._last_percent: int = -1
        self._last_render_time: float = 0.0
        self._log_handle = None

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        """Close active progress rendering and any open log handle."""

        if self._current_label is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._current_label = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def log(self, message: str) -> None:
        """Write a timestamped message to stdout and optional log file."""

        timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        if self._current_label is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        print(line, flush=True)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    def start(self, label: str, total: int) -> None:
        """Start a named progress scope with total work units."""

        self._current_label = label
        self._current_total = max(total, 1)
        self._current_count = 0
        self._last_percent = -1
        self._last_render_time = 0.0
        self.log(f"START {label} total={total}")
        self._render(force=True)

    def advance(self, count: int) -> None:
        """Advance current progress by ``count`` work units."""

        self._current_count = min(self._current_count + count, self._current_total)
        self._render()

    def finish(self, detail: str | None = None) -> None:
        """Finalize current progress scope and emit completion message."""

        if self._current_label is None:
            if detail:
                self.log(detail)
            return
        self._current_count = self._current_total
        label = self._current_label
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._current_label = None
        message = f"DONE {label}"
        if detail:
            message = f"{message} {detail}"
        self.log(message)

    def _render(self, force: bool = False) -> None:
        """Render progress bar with throttling to reduce terminal/log churn."""

        if self._current_label is None:
            return
        ratio = self._current_count / max(self._current_total, 1)
        percent = int(ratio * 100)
        now = time.monotonic()
        if not force and percent == self._last_percent and (now - self._last_render_time) < 1.0:
            return

        self._last_percent = percent
        self._last_render_time = now
        filled = int(ratio * 28)
        bar = "#" * filled + "-" * (28 - filled)
        line = (
            f"\r[{bar}] {percent:3d}% "
            f"{self._current_label} {self._current_count:,}/{self._current_total:,}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

        if self._log_handle is not None:
            timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            self._log_handle.write(
                f"[{timestamp}] PROGRESS {self._current_label} "
                f"{self._current_count}/{self._current_total} ({percent}%)\n"
            )
            self._log_handle.flush()


# ---------------------------------------------------------------------------
# vectorized sampling primitives (np.random.Generator; batch/columnar only)
# ---------------------------------------------------------------------------


def phase_rng(seed: int, phase: int) -> np.random.Generator:
    """Deterministic, mutually independent generator for one seeding phase."""

    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(phase,)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis`` (batch form of the scalar helper)."""

    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def sample_categorical(rng: np.random.Generator, weights, size: int | None = None) -> np.ndarray:
    """Inverse-CDF categorical draws.

    ``weights`` is ``(k,)`` (``size`` draws) or ``(n, k)`` (one draw per row).
    Matches the scalar ``random.choices``/``_weighted_choice`` semantics.
    """

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        cum = np.cumsum(w)
        t = rng.random(size) * cum[-1]
        return (cum < t[..., None]).sum(axis=-1)
    cum = np.cumsum(w, axis=-1)
    t = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum < t[..., None]).sum(axis=-1)


def sample_from_cum(rng: np.random.Generator, cum: np.ndarray, size: int) -> np.ndarray:
    """Draw indices from precomputed cumulative weights (bisect_left equivalent)."""

    t = rng.random(size) * cum[-1]
    return np.searchsorted(cum, t, side="left")


def sample_event_ts(rng: np.random.Generator, start_ts: datetime, total_days: int, size: int):
    """Sample event timestamps with recency and intra-day activity bias (vectorized)."""

    # numpy is (left, mode, right); matches random.triangular(0, days-1, 0.72*days)
    day = rng.triangular(0, total_days * 0.72, total_days - 1, size).astype(np.int64)
    hour = sample_categorical(rng, _HOUR_WEIGHTS, size)
    minute = rng.integers(0, 60, size)
    second = rng.integers(0, 60, size)
    start = np.datetime64(start_ts.astimezone(timezone.utc).replace(tzinfo=None), "ms")
    return (
        start
        + day.astype("timedelta64[D]").astype("timedelta64[ms]")
        + hour.astype("timedelta64[h]").astype("timedelta64[ms]")
        + minute.astype("timedelta64[m]").astype("timedelta64[ms]")
        + second.astype("timedelta64[s]").astype("timedelta64[ms]")
    )


def utc_naive(start_ts: datetime) -> np.datetime64:
    """Naive-UTC wall time of an aware timestamp (numpy epoch anchor)."""

    return np.datetime64(start_ts.astimezone(timezone.utc).replace(tzinfo=None), "us")


def day_of_year(dt64: np.ndarray) -> np.ndarray:
    """Day-of-year (1-based, tm_yday equivalent) for datetime64 arrays."""

    d = dt64.astype("datetime64[D]")
    return (d - d.astype("datetime64[Y]")).astype(np.int64) + 1


def seasonal_wave(position: np.ndarray) -> np.ndarray:
    """Seasonality helper with two harmonics for smoother demand variation."""

    # Position expected in [0, 1]. Two yearly demand peaks with different amplitudes.
    return (
        np.sin(position * 6.283185307179586 * 2.0) * 0.7
        + np.sin(position * 6.283185307179586 * 4.0) * 0.3
    )


def browse_probs(activity: np.ndarray) -> np.ndarray:
    """Map customer activity to browse-event type probabilities, shape (n, 4)."""

    page_view = np.maximum(0.44, 0.60 - activity * 0.10)
    search = np.maximum(0.08, 0.14 - activity * 0.02)
    product_view = np.minimum(0.30, 0.16 + activity * 0.07)
    add_to_cart = np.minimum(0.24, 0.08 + activity * 0.09)
    total = page_view + search + product_view + add_to_cart
    return np.stack([page_view, search, product_view, add_to_cart], axis=-1) / total[:, None]


def product_arrays(products_by_category, categories, product_price, product_cost):
    """Flat product ids + per-category offsets + price/cost vectors (columnar index)."""

    flat_ids: list[str] = []
    offsets = [0]
    prices: list[float] = []
    costs: list[float] = []
    for cat in categories:
        ids = products_by_category[cat]
        flat_ids.extend(ids)
        prices.extend(product_price[i] for i in ids)
        costs.extend(product_cost[i] for i in ids)
        offsets.append(len(flat_ids))
    return (
        np.array(flat_ids, dtype=object),
        np.array(offsets, dtype=np.int64),
        np.asarray(prices, dtype=np.float64),
        np.asarray(costs, dtype=np.float64),
    )


def customer_arrays(
    customer_ids, customer_activity, customer_pref_category, customer_country_state, categories
):
    """Columnar views over the per-customer dicts (activity, pref category, geo)."""

    n = len(customer_ids)
    activity = np.fromiter((customer_activity[c] for c in customer_ids), dtype=np.float64, count=n)
    cat_index = {cat: i for i, cat in enumerate(categories)}
    pref_idx = np.fromiter(
        (cat_index[customer_pref_category[c]] for c in customer_ids), dtype=np.int64, count=n
    )
    country = np.fromiter(
        (customer_country_state[c][0] for c in customer_ids), dtype=object, count=n
    )
    state = np.fromiter((customer_country_state[c][1] for c in customer_ids), dtype=object, count=n)
    return activity, pref_idx, country, state


ISO_S = "%Y-%m-%dT%H:%M:%S"
ISO_US = "%Y-%m-%dT%H:%M:%S%.6f"


def iso_expr(col: str, fmt: str = ISO_S):
    """Polars expression: datetime column -> ISO-8601 UTC string (``+00:00``)."""

    import polars as pl

    return pl.col(col).dt.strftime(fmt) + pl.lit("+00:00")
