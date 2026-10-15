"""Magnitude-critical synthetic stream (CONTROLLED test for the MEASURE).

Designed so that, by construction, a **windowed, normalizing sequence encoder MUST fail**
while a **full-history additive measure MUST succeed**. The three required conditions:

1. **histories ≫ the window**: ~3600 events/customer over 730 days, so a 256-event window
   is a tiny slice of the recent past (train with `--set seq_len=256`).
2. **activity INDEPENDENT of the latent**: the event rate is uniform across customers, so
   event COUNT/rate cannot reveal the latent wealth `S`.
3. **target ∝ full-history cumulative, driver in the distant past**: a customer's latent
   wealth `S` is written ONLY into an **early transient** (first `early_days`) and into the
   **future window**; the *recent* past (what a window sees) has value independent of `S`.
   The future LTV over the gate horizon is ∝ `S`.

So:
  * bare (windowed, normalized) sees only recent S-independent values -> cannot get S -> LOW U;
  * the full-history measure's **sum of values** is dominated by the early ∝S transient ->
    recovers S -> predicts the future -> HIGH U.

Run: python3 -m rabbit_hole.generators.magnitude --n-customers 3000 --out /tmp/opencode/mag2
Then: use --as-of <REF_START + as_of_day> and --set seq_len=256.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def build_magnitude(
    n_customers: int = 3000,
    window_days: int = 730,
    early_days: int = 45,
    as_of_day: int = 600,
    base_rate: float = 6.0,
    seed: int = 17,
):
    rng = np.random.default_rng(seed)
    rows = {"customer_key": [], "event_ts": [], "event_type": [], "source_table": [], "attrs": []}
    early_scale = 1.0e4  # early transient DOMINATES the cumulative -> full-history sum ∝ S
    for c in range(n_customers):
        key = f"mc{c:07d}"
        S = float(np.exp(rng.normal(0.0, 1.5)))  # latent wealth (wide)
        # (2) rate INDEPENDENT of S (uniform, tiny jitter) -> count cannot reveal S
        rate = base_rate * float(np.exp(rng.normal(0.0, 0.15)))
        n = max(1, int(rng.poisson(rate * window_days)))
        times = np.sort(rng.random(n) * window_days)
        for t in times:
            if t < early_days or t >= as_of_day:
                # (3) early transient AND the future window carry S (early dominates the sum)
                val = (
                    S * early_scale * np.exp(rng.normal(0.0, 0.3))
                    if t < early_days
                    else (S * np.exp(rng.normal(3.0, 0.4)))
                )
            else:
                # recent past: value INDEPENDENT of S -> the window cannot see S
                val = float(np.exp(rng.normal(3.0, 0.4)))
            rows["customer_key"].append(key)
            rows["event_ts"].append((REF_START + timedelta(days=float(t))).isoformat())
            rows["event_type"].append("purchase")
            rows["source_table"].append("magnitude")
            rows["attrs"].append(json.dumps({"value": round(float(val), 2)}, separators=(",", ":")))
    df = (
        pl.DataFrame(rows)
        .select(["customer_key", "event_ts", "event_type", "source_table", "attrs"])
        .rename({"attrs": "event_attributes"})
    )
    truth = {
        "name": "magnitude",
        "n_customers": n_customers,
        "window_days": window_days,
        "early_days": early_days,
        "as_of_day": as_of_day,
        "seed": seed,
        "note": "future LTV ∝ S (latent in an early transient + future); recent past is S-free; "
        "rate independent of S; histories ≫ window. Run the gate with --as-of (REF_START + "
        "as_of_day) and --set seq_len=256: bare MUST fail, the measure MUST succeed.",
    }
    return df, truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=3000, dest="n_customers")
    ap.add_argument("--window-days", type=int, default=730, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="/tmp/opencode/mag2")
    a = ap.parse_args(argv)
    df, truth = build_magnitude(a.n_customers, a.window_days, seed=a.seed)
    d = Path(a.out) / "magnitude"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / "magnitude_truth.json").write_text(json.dumps(truth, indent=1))
    as_of = (REF_START + timedelta(days=truth["as_of_day"])).date().isoformat()
    print(f"[magnitude] {df.height} events, {a.n_customers} customers -> {d}")
    print(f"[magnitude] use --as-of {as_of} --set seq_len=256")


if __name__ == "__main__":
    main()
