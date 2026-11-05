#!/usr/bin/env python3
"""red_queen's next-best-action plan on the demo stream (B-007).

The old V1.5 plan was synthesised from `reference_truth.offer_opt` and the response
view, so caterpillar explained a truth, not a decision. This producer builds the
plan red_queen actually owns:

  * fit the per-customer offer values on pre-as-of randomized send days;
  * keep only channels certified DEPLOY in `red_queen/certification`;
  * choose the value-maximising offer per customer under a weekly send budget and
    the per-customer cap (1 send/week);
  * fail safe: no send when the chosen value is not positive.

The plan uses no generator truth. It is the input to caterpillar's artifact-only
explanation (V1.5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from red_queen.offer_panel import fit, panel as offer_panel  # noqa: E402

DEMO = ROOT / "looking_glass/artifacts/demo_v2"
CERT = ROOT / "red_queen/certification/channel_certification.json"
OUT = ROOT / "red_queen/artifacts/nba_demo_plan.json"
AS_OF = date(2025, 1, 1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _channel_lookup(stream_path: Path) -> dict[str, str]:
    df = pl.read_parquet(stream_path).filter(pl.col("event_type") == "email_send")
    df = df.with_columns(
        pl.col("event_attributes").str.json_path_match("$.channel").alias("channel")
    )
    out = {}
    for key, g in df.group_by("customer_key"):
        out[str(key[0])] = str(g["channel"].mode().sort()[0])
    return out


def build_plan(budget: float | None = None, cap: float = 1.0) -> dict:
    stream = DEMO / "customer_event_stream.parquet"
    rows = offer_panel(stream)
    pre = rows.filter(pl.col("day") < AS_OF)
    V, pooled = fit(pre, outcome="margin", shrink=2.0)
    cert = json.loads(CERT.read_text())
    deployed = {e["channel"] for e in cert if e.get("action") == "offer" and e.get("deploy")}
    channels = _channel_lookup(stream)
    rows = []
    for key, vals in V.items():
        channel = channels.get(key)
        if channel not in deployed:
            continue
        arm = int(max(vals, key=vals.get))
        value = float(vals[arm])
        if value <= 0:
            continue
        rows.append((key, channel, arm, value))
    rows.sort(key=lambda r: (-r[3], r[0]))
    if budget is None:
        budget = float(len(rows))
    served = []
    used = 0.0
    for key, channel, arm, value in rows:
        if used + cap > budget + 1e-9:
            break
        served.append(
            {
                "customer_key": key,
                "channel": channel,
                "arm": arm,
                "weekly_sends": float(cap),
                "expected_gp": round(value, 6),
            }
        )
        used += cap
    # Caterpillar's first-class key order: keep a customer with an anchor first so the
    # artifact-only explanation can use the served anchor embeddings.
    served.sort(key=lambda r: (not r["customer_key"].startswith("v2000"), r["customer_key"]))
    plan = {
        "meta": {
            "kind": "red_queen_nba_plan",
            "stream": str(stream),
            "stream_sha256": _sha256(stream),
            "certificate": str(CERT),
            "certificate_sha256": _sha256(CERT),
            "as_of": AS_OF.isoformat(),
            "fit_cells": int(pre.height),
            "n_customers_fit": len(V),
            "shrink": 2.0,
            "budget": budget,
            "per_customer_cap": cap,
            "n_served": len(served),
            "sends_used": round(used, 6),
            "deployed_channels": sorted(deployed),
            "pooled_fit": {str(k): round(v, 6) for k, v in pooled.items()},
            "fail_safe": "no send when the chosen expected margin is not positive",
        },
        "plan": served,
        "customer_key": [r["customer_key"] for r in served],
        "arm": [r["arm"] for r in served],
        "weekly_sends": [r["weekly_sends"] for r in served],
        "expected_gp": [r["expected_gp"] for r in served],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--cap", type=float, default=1.0)
    a = ap.parse_args()
    plan = build_plan(a.budget, a.cap)
    print(json.dumps(plan["meta"], indent=1))
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
