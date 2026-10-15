"""Magnitude-critical synthetic stream (controlled test for the MEASURE).

Purpose: a stream where the *future* is defined to depend on a customer's **cumulative
magnitude** (lifetime value), so a **windowed, normalizing sequence encoder cannot hold
it** (the bare `U` should be LOW) while a **full-history additive measure can** (the
pipeline `U` should be HIGH). This turns "the measure matters because some real stream is
hard" into a *controlled, causal* demonstration.

Construction: each customer has a latent "wealth" S ~ LogNormal(0, 1.5). Their event
RATE scales with S, and each event's VALUE scales with S. So:
  * cumulative value over the full history  ∝ S          (the measure can see S)
  * future LTV over the horizon             ∝ S²         (count ∝ S, value ∝ S)
  * a short recent WINDOW of *normalized* events loses the scale S -> bare encoder fails.

Run: python3 -m rabbit_hole.generators.magnitude --n-customers 5000 --out /tmp/opencode/mag
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def build_magnitude(n_customers: int = 5000, window_days: int = 365, seed: int = 17):
    rng = np.random.default_rng(seed)
    base_rate = 0.05  # events/day at S=1
    rows = {"customer_key": [], "event_ts": [], "event_type": [], "source_table": [], "attrs": []}
    for c in range(n_customers):
        key = f"mc{c:07d}"
        S = float(np.exp(rng.normal(0.0, 1.5)))  # latent wealth (wide)
        # activity is INDEPENDENT of wealth (so a windowed encoder cannot read S from
        # event COUNT/rate) — ONLY the VALUE SCALE reveals S, and a normalizing encoder
        # erases scale. Full-history cumulative VALUE (the measure) is the only way to S.
        rate = base_rate * float(np.exp(rng.normal(0.0, 0.3)))
        n = max(1, int(rng.poisson(rate * window_days)))
        times = np.sort(rng.random(n) * window_days)
        for t in times:
            # value scales with wealth -> cumulative value ∝ S; future LTV ∝ S^2
            val = round(float(S * np.exp(rng.normal(3.0, 0.4))), 2)
            rows["customer_key"].append(key)
            rows["event_ts"].append((REF_START + timedelta(days=float(t))).isoformat())
            rows["event_type"].append("purchase")
            rows["source_table"].append("magnitude")
            rows["attrs"].append(json.dumps({"value": val}, separators=(",", ":")))
    df = (
        pl.DataFrame(rows)
        .select(["customer_key", "event_ts", "event_type", "source_table", "attrs"])
        .rename({"attrs": "event_attributes"})
    )
    truth = {
        "name": "magnitude",
        "n_customers": n_customers,
        "window_days": window_days,
        "seed": seed,
        "note": "future LTV ∝ cumulative magnitude (S^2); a normalizing windowed encoder "
        "cannot hold it, a full-history additive measure can — the MEASURE's controlled test.",
    }
    return df, truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=5000, dest="n_customers")
    ap.add_argument("--window-days", type=int, default=365, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="/tmp/opencode/mag")
    a = ap.parse_args(argv)
    df, truth = build_magnitude(a.n_customers, a.window_days, a.seed)
    d = Path(a.out) / "magnitude"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / "magnitude_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"[magnitude] {df.height} events, {a.n_customers} customers -> {d}")


if __name__ == "__main__":
    main()
