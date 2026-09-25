"""Shared generator primitives: progress reporting, sampling, and row types."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import timezone as _tz
from pathlib import Path
import random
import sys
import time

import torch


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
                f"[{timestamp}] PROGRESS {self._current_label} {self._current_count}/{self._current_total} ({percent}%)\n"
            )
            self._log_handle.flush()


def _weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    """Sample a single item from ``items`` proportionally to ``weights``."""

    threshold = rng.random() * sum(weights)
    running = 0.0
    for item, weight in zip(items, weights):
        running += weight
        if running >= threshold:
            return item
    return items[-1]


def _sample_weighted_index(rng: random.Random, cumulative_weights: list[float]) -> int:
    """Sample an index from precomputed cumulative weights."""

    if not cumulative_weights:
        raise ValueError("cumulative_weights must not be empty")
    target = rng.random() * cumulative_weights[-1]
    return int(bisect.bisect_left(cumulative_weights, target))


def _seasonal_wave(position: float) -> float:
    """Seasonality helper with two harmonics for smoother demand variation."""

    # Position expected in [0, 1]. Two yearly demand peaks with different amplitudes.
    return (
        torch.sin(torch.tensor(position * 6.283185307179586 * 2.0)).item() * 0.7
        + torch.sin(torch.tensor(position * 6.283185307179586 * 4.0)).item() * 0.3
    )


def _browse_distribution(activity: float) -> list[float]:
    """Map customer activity to browse-event type probabilities."""

    page_view = max(0.44, 0.60 - activity * 0.10)
    search = max(0.08, 0.14 - activity * 0.02)
    product_view = min(0.30, 0.16 + activity * 0.07)
    add_to_cart = min(0.24, 0.08 + activity * 0.09)
    total = page_view + search + product_view + add_to_cart
    return [
        page_view / total,
        search / total,
        product_view / total,
        add_to_cart / total,
    ]


def _sample_event_ts(rng: random.Random, start_ts: datetime, total_days: int) -> datetime:
    """Sample a timestamp with recency and intra-day activity bias."""

    day_offset = int(rng.triangular(0, total_days - 1, total_days * 0.72))
    hour = _weighted_choice(
        rng,
        [str(h) for h in range(24)],
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
    )
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return start_ts + timedelta(days=day_offset, hours=int(hour), minutes=minute, seconds=second)
