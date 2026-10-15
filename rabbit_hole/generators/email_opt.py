"""Email-optimization stream (the real product use case).

Designs an event stream for the two "optimization" use cases the operator actually
needs, with a KNOWN dose-response so capability is falsifiable:

  * WEEKLY EMAIL FREQUENCY optimization — how many sends/week maximizes margin;
  * DAILY TIME-OF-DAY optimization     — which hour maximizes margin.

Each customer has a latent **optimal weekly frequency** `x0 ∈ {0,1,2,3}` and **optimal
hour** `h0 ∈ [0,24)`. Every week the platform assigns a **RANDOMIZED** frequency
(uniform over {0,1,2,3} — 0 = the no-send control) and, per send, a **RANDOMIZED** hour
(uniform). The weekly conversion probability is UNIMODAL in both:

    p(week) = PEAK · exp(-((f - x0) / σf)² ) · exp(-((h - h0) / σh)² )

so E[margin | do(freq)] peaks at the customer's `x0`, and E[margin | do(hour)] at `h0`.
Randomization makes both effects *identifiable* (the ground truth every OPE/dose-response
gate needs); the per-customer optima are the truth receipt.

Truth receipt: `email_opt_truth.json` with `x0` (optimal frequency) and `h0` (optimal hour)
per customer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def build_email_opt(n_customers: int = 8000, weeks: int = 52, seed: int = 17):
    rng = np.random.default_rng(seed)
    peak = 0.6  # max weekly conversion prob at (x0, h0)
    sigma_f, sigma_h = 1.0, 4.0  # unimodal widths (frequency units, hours)
    rows = {"customer_key": [], "event_ts": [], "event_type": [], "source_table": [], "attrs": []}
    truth = []

    def emit(key, day, etype, attrs):
        rows["customer_key"].append(key)
        rows["event_ts"].append((REF_START + timedelta(days=float(day))).isoformat())
        rows["event_type"].append(etype)
        rows["source_table"].append("email_opt")
        rows["attrs"].append(json.dumps(attrs, separators=(",", ":")))

    for c in range(n_customers):
        key = f"eo{c:07d}"
        x0 = int(rng.integers(0, 4))  # optimal weekly frequency (0..3)
        h0 = float(rng.uniform(0.0, 24.0))  # optimal hour
        truth.append({"customer": key, "x0": x0, "h0": round(h0, 2)})
        for w in range(weeks):
            day0 = w * 7.0
            f = int(rng.integers(0, 4))  # RANDOMIZED weekly frequency (0 = control)
            if f == 0:
                continue
            h_week = float(rng.uniform(0.0, 24.0))  # RANDOMIZED send hour
            for j in range(f):
                send_day = day0 + float(rng.uniform(0.0, 7.0))
                emit(
                    key,
                    send_day,
                    "email_send",
                    {"freq": f, "hour": round(h_week, 2), "week": w, "send_idx": j},
                )
            p = (
                peak
                * np.exp(-(((f - x0) / sigma_f) ** 2))
                * np.exp(-(((h_week - h0) / sigma_h) ** 2))
            )
            if rng.random() < p:
                emit(
                    key,
                    day0 + 7.0 + float(rng.random()),
                    "purchase",
                    {"value": round(float(np.exp(rng.normal(3.0, 0.4))), 2), "week": w},
                )
    df = (
        pl.DataFrame(rows)
        .select(["customer_key", "event_ts", "event_type", "source_table", "attrs"])
        .rename({"attrs": "event_attributes"})
    )
    meta = {
        "name": "email_opt",
        "n_customers": n_customers,
        "weeks": weeks,
        "seed": seed,
        "use_cases": ["weekly_email_frequency_optimization", "daily_time_of_day_optimization"],
        "peak": peak,
        "sigma_f": sigma_f,
        "sigma_h": sigma_h,
        "note": "randomized weekly frequency (0..3, 0=control) + randomized send hour; unimodal "
        "response; per-customer optima are the truth.",
    }
    return df, meta, truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=8000, dest="n_customers")
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="rabbit_hole/data/email_opt")
    a = ap.parse_args(argv)
    df, meta, truth = build_email_opt(a.n_customers, a.weeks, a.seed)
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / "email_opt_truth.json").write_text(
        json.dumps({"meta": meta, "customers": truth}, indent=1)
    )
    print(f"[email_opt] {df.height} events, {a.n_customers} customers -> {d}")
    print(f"[email_opt] use cases: {meta['use_cases']}")


if __name__ == "__main__":
    main()
