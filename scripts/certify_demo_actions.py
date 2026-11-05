#!/usr/bin/env python3
"""Regenerate red_queen's action-composition certificate on the demo stream (B-006).

The old channel_certification.json was a stored artifact from the retired canonical
stream (white_queen OPE, estimated provenance). This producer reads the current
demo's randomized weekly offer panel and certifies the per-channel offer policy:

    fit the per-customer arm values on pre-as-of randomized send days
    score the argmax policy on post-as-of randomized send days
    DEPLOY iff the held-out policy margin > the held-out behavior margin and the
    paired customer-bootstrap 95% CI lower bound is > 0; otherwise HOLD.

The demo randomizes only `offer`; `discount` and `frequency` have no randomized
variation, so they are recorded as HOLD / not identified rather than silently
carried over from the old stream. The certificate is the red_queen gate input.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from red_queen.offer_panel import fit, panel as offer_panel, policy  # noqa: E402

DEMO = ROOT / "looking_glass/artifacts/demo_v2"
OUT = ROOT / "red_queen/certification/channel_certification.json"
AS_OF = date(2025, 1, 1)
SEED = 0
N_BOOT = 2000


def _channels(stream_path: Path, panel: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet(stream_path)
    df = df.filter(pl.col("event_type") == "email_send").with_columns(
        [
            pl.col("event_ts").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", time_zone="UTC").alias("ts"),
            pl.col("event_attributes").str.json_path_match("$.channel").alias("channel"),
            pl.col("event_attributes").str.json_path_match("$.offer").cast(pl.Int32).alias("offer"),
        ]
    )
    ch = df.select(
        [
            pl.col("customer_key"),
            pl.col("ts").dt.date().alias("day"),
            pl.col("offer"),
            pl.col("channel"),
        ]
    )
    return panel.join(ch, on=["customer_key", "day", "offer"], how="left")


def _paired_ci(delta: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = np.asarray(
        [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(N_BOOT)]
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    stream = DEMO / "customer_event_stream.parquet"
    rows = _channels(stream, offer_panel(stream))
    pre = rows.filter(pl.col("day") < AS_OF)
    post = rows.filter(pl.col("day") >= AS_OF)
    entries = []
    for channel in ("email", "sms", "push"):
        fit_rows = pre.filter(pl.col("channel") == channel)
        eval_rows = post.filter(pl.col("channel") == channel)
        V, _pooled = fit(fit_rows, outcome="margin", shrink=2.0)
        pol = policy(V, eval_rows, outcome="margin")
        delta = pol["policy_by_customer"] - pol["random_by_customer"]
        lo, hi = _paired_ci(delta)
        deploy = bool(pol["policy"] > pol["random_all"] and lo > 0)
        entries.append(
            {
                "channel": channel,
                "action": "offer",
                "levels": [0, 1, 2, 3],
                "rows": int(eval_rows.height),
                "n_customers": int(pol["n_customers"]),
                "nA": 4,
                "deploy": deploy,
                "behavior": round(pol["policy"], 6),
                "bar": round(pol["random_all"], 6),
                "metric": "held-out margin on the randomized offer day",
                "ci95_policy_minus_behavior": [round(lo, 6), round(hi, 6)],
                "provenance": "ipw_policy_pre_post",
                "fit_cells": int(fit_rows.height),
                "eval_cells": int(eval_rows.height),
            }
        )
    # The demo randomizes one factor (offer). Discount and frequency are not
    # identified; recording them as HOLD is the fail-safe, not a missing run.
    entries.append(
        {
            "channel": "email",
            "action": "discount",
            "levels": [],
            "rows": 0,
            "n_customers": 0,
            "nA": 0,
            "deploy": False,
            "behavior": None,
            "bar": None,
            "provenance": "not_identified",
            "reason": "the demo randomizes offer only; no randomized discount bucket exists",
        }
    )
    entries.append(
        {
            "channel": "email",
            "action": "frequency",
            "levels": [],
            "rows": 0,
            "n_customers": 0,
            "nA": 0,
            "deploy": False,
            "behavior": None,
            "bar": None,
            "provenance": "not_identified",
            "reason": "frequency is at the customer optimum; it is not randomized in the demo",
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(entries, indent=1) + "\n")
    print(json.dumps(entries, indent=1))
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
