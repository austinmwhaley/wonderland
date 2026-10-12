"""Complexity ladder of synthetic event streams (Layer A; SPEC §20.5 stream zoo).

One generator, many streams. Each rung adds exactly ONE structural dimension on
top of the previous, so a frozen encoder can be tested for *which* structure it
recovers and *where* it breaks — the capability-vs-complexity curve. Every rung
ships with a KNOWN-TRUTH receipt (the parameters that generated it) and there is
an explicit NEGATIVE CONTROL (L0) with no recoverable structure, so a capability
model must report NO_SIGNAL there.

Rungs (accumulating):
  L0_null    homogeneous rate, i.i.d. event types            (negative control)
  L1_rate    + per-customer activity (rate heterogeneity)
  L2_markov  + a fixed order-1 event-type transition matrix
  L3_intent  + per-customer intent sharpening the transitions
  L4_trend   + per-customer rate DRIFT over time (velocity)
  L5_entity  + per-customer product affinity (high-cardinality entity)
  L6_action  + exogenous actions with a latent unimodal response

Output per rung: a canonical Parquet stream (the 5 canonical fields) + a
`<rung>_truth.json` receipt. Deterministic for a given seed.

Run:
  python3 -m rabbit_hole.generators.ladder --all --n-customers 5000 \
      --out rabbit_hole/data/ladder
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

RUNGS = ("L0_null", "L1_rate", "L2_markov", "L3_intent", "L4_trend", "L5_entity", "L6_action")
TYPES = ("view", "cart", "purchase")  # the event vocabulary on every rung
ETYPES = ("page", "product", "order")  # entity-type vocabulary (L5+)
# fixed order-1 transition matrix (rows sum to 1); a KNOWN piece of structure
P_FIXED = np.array(
    [
        [0.60, 0.30, 0.10],  # view -> view/cart/purchase
        [0.20, 0.30, 0.50],  # cart -> ...
        [0.50, 0.20, 0.30],  # purchase -> ...
    ]
)
REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _rung_index(rung: str) -> int:
    return RUNGS.index(rung)


def _simulate_customer(level, n_events, rng, params):
    """Return (times_days, type_idx) for one customer at a complexity `level`."""
    base, sigma_a, sigma_i, sigma_t, K, p_aff = (
        params["base_rate"],
        params["sigma_activity"],
        params["sigma_intent"],
        params["sigma_trend"],
        params["n_entities"],
        params["p_affinity"],
    )
    a = np.exp(rng.normal(0.0, sigma_a))  # activity multiplier
    _ = base * (a if level >= 1 else 1.0)  # rate level (count drawn by caller)
    T = params["window_days"]
    # times: homogeneous Poisson; with a per-customer drift (velocity) at L4+
    if level >= 4:
        k = rng.normal(0.0, sigma_t)
        u = rng.random(n_events)
        if abs(k) < 1e-6:  # no drift -> uniform (guard the k->0 limit)
            frac = u
        else:
            frac = np.log1p(u * np.expm1(k)) / k
        times = np.sort(np.clip(frac, 0.0, 1.0) * T)
    else:
        times = np.sort(rng.random(n_events) * T)
    # types: i.i.d. marginal (L0/L1) or an order-1 Markov chain (L2+)
    if level >= 2:
        if level >= 3:
            # per-customer intent sharpens (or flattens) the fixed transition
            b = rng.normal(0.0, sigma_i)
            logits = np.log(P_FIXED) * np.exp(b)
            Pc = np.exp(logits)
            Pc = Pc / Pc.sum(1, keepdims=True)
        else:
            Pc = P_FIXED
        idx = np.empty(n_events, dtype=np.int64)
        idx[0] = rng.integers(0, len(TYPES))
        draws = rng.random(n_events)
        for t in range(1, n_events):
            idx[t] = int(np.searchsorted(np.cumsum(Pc[idx[t - 1]]), draws[t]))
    else:
        idx = rng.integers(0, len(TYPES), size=n_events)
    # entity affinity (L5+): purchases carry a preferred high-cardinality id
    if level >= 5:
        pref = int(rng.integers(0, K))
        is_pur = idx == TYPES.index("purchase")
        aff = rng.random(n_events) < p_aff
        eid = np.where(is_pur & aff, pref, rng.integers(0, K, size=n_events)).astype(np.int64)
    else:
        eid = np.full(n_events, -1, dtype=np.int64)
    return times, idx, eid


def build_rung(rung: str, n_customers: int, window_days: int, seed: int):
    level = _rung_index(rung)
    rng = np.random.default_rng(seed)
    params = {
        "base_rate": 0.08,  # events/day at a neutral customer
        "sigma_activity": 0.9,
        "sigma_intent": 0.6,
        "sigma_trend": 0.5,
        "n_entities": 64,
        "p_affinity": 0.6,
        "window_days": window_days,
        "resp_peak": 0.5,  # peak purchase prob at the optimal arm (L6+)
        "holdout_frac": 0.1,  # randomized no-send holdout (ground-truth control, L6+)
    }
    mean_events = params["base_rate"] * window_days * np.exp(0.5 * params["sigma_activity"] ** 2)
    rows = {"customer_key": [], "event_ts": [], "event_type": [], "source_table": [], "attrs": []}
    truth_customers = []  # noqa: F841

    for c in range(n_customers):
        n = int(rng.poisson(mean_events * (np.exp(rng.normal(0, 0.9)) if level >= 1 else 1.0)))
        n = max(n, 1)
        times, idx, eid = _simulate_customer(level, n, rng, params)
        key = f"lc{c:07d}"
        et_pref = int(rng.integers(0, len(ETYPES))) if level >= 5 else -1

        def emit(day, etype, attrs):
            rows["customer_key"].append(key)
            rows["event_ts"].append((REF_START + timedelta(days=float(day))).isoformat())
            rows["event_type"].append(etype)
            rows["source_table"].append("ladder")
            rows["attrs"].append(json.dumps(attrs, separators=(",", ":")))

        for t, ti, ei in zip(times, idx, eid, strict=True):
            attrs = {}
            if ti == TYPES.index("purchase"):
                attrs["value"] = round(float(np.exp(rng.normal(3.0, 0.5))), 2)
            if level >= 5:
                # per-customer entity TYPE affinity (so `entity` is not degenerate)
                eti = et_pref if rng.random() < 0.7 else int(rng.integers(0, len(ETYPES)))
                attrs["entity_type"] = ETYPES[eti]
                attrs["entity_id"] = (
                    f"p{int(ei)}" if ei >= 0 else f"p{int(rng.integers(0, params['n_entities']))}"
                )
            emit(t, TYPES[int(ti)], attrs)

        if level >= 6:
            # exogenous marketing ACTIONS (observed policy, confounded by activity,
            # NOT randomized) with a latent UNIMODAL response (per-customer optimal
            # arm x0) — the known-truth RL/OPE structure. A small random holdout.
            a = float(np.exp(rng.normal(0.0, params["sigma_activity"])))
            x0 = float(rng.uniform(0.0, 2.4))
            n_per = 12
            per_days = window_days / n_per
            for p in range(n_per):
                logits = np.array([0.0, 0.4, 0.8]) + 0.7 * (a - 1.0) * np.array([0, 1, 2])
                pr = np.exp(logits - logits.max())
                pr = pr / pr.sum()
                arm = int(rng.choice(3, p=pr))
                tday = p * per_days + float(rng.random()) * per_days
                if rng.random() >= params["holdout_frac"]:
                    emit(
                        tday,
                        "action_send",
                        {"arm": arm, "discount_pct": round(0.1 * arm, 2), "campaign_id": f"c{p}"},
                    )
                pp = params["resp_peak"] * np.exp(-(((arm - x0) / 1.0) ** 2))
                if rng.random() < pp:
                    emit(
                        tday + 0.05,
                        "purchase",
                        {
                            "value": round(float(np.exp(rng.normal(3.0, 0.5))), 2),
                            "converted_by": f"c{p}",
                        },
                    )
            truth_customers.append(
                {"customer": key, "optimal_arm": round(x0, 3), "activity": round(a, 3)}
            )
    df = (
        pl.DataFrame(rows)
        .select(["customer_key", "event_ts", "event_type", "source_table", "attrs"])
        .rename({"attrs": "event_attributes"})
    )
    truth = {
        "rung": rung,
        "level": level,
        "n_customers": n_customers,
        "window_days": window_days,
        "seed": seed,
        "params": params,
        "transitions": P_FIXED.tolist(),
        "dimensions": {
            "rate": level >= 1,
            "markov": level >= 2,
            "intent": level >= 3,
            "trend": level >= 4,
            "entity": level >= 5,
            "action": level >= 6,
        },
        "note": "L0_null is the NEGATIVE CONTROL; every capability must be NO_SIGNAL here.",
    }
    if level >= 6:
        truth["entity_types"] = list(ETYPES)
        truth["customer_truth"] = truth_customers  # optimal_arm (x0) + activity per customer
    return df, truth, truth_customers


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the synthetic complexity ladder")
    ap.add_argument("--rung", choices=RUNGS, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-customers", type=int, default=5000, dest="n_customers")
    ap.add_argument("--window-days", type=int, default=365, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="rabbit_hole/data/ladder")
    a = ap.parse_args(argv)
    rungs = RUNGS if a.all or a.rung is None else (a.rung,)
    out = Path(a.out)
    for rung in rungs:
        df, truth, _ = build_rung(rung, a.n_customers, a.window_days, a.seed)
        d = out / rung
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
        (d / f"{rung}_truth.json").write_text(json.dumps(truth, indent=1))
        print(f"[ladder] {rung}: {df.height} events, {a.n_customers} customers -> {d}")


if __name__ == "__main__":
    main()
