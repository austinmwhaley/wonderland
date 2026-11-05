"""red_king's per-customer effect estimator on a randomized send panel.

The demo stream's action family is a weekly randomized offer. The known effect is on
the send day's purchase intensity, so the honest unit is a (customer, send-day) cell,
not a 7-day window (which dilutes the effect). A per-customer arm mean over those
cells is the nonparametric value estimate; shrinking it toward the pooled arm mean
is the low-variance red_king variant used when a customer has few cells.

`fit` is deliberately causal: it uses only randomized outcomes before the caller's
cut. `evaluate` scores the resulting policy on later randomized cells, where the
assignment is still uniform, so the held-out mean of the chosen arm is an unbiased
policy value. No generator truth is used to fit; `offer_opt` is validation only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def panel(stream_path: Path) -> pl.DataFrame:
    """(customer_key, day, offer, count, margin) for every randomized offer send.

    A purchase is attributed to the send's calendar day (the generator's known-effect
    window), not to a trailing week; the day-level count/margin is the outcome.
    """
    df = pl.read_parquet(stream_path)
    df = df.with_columns(
        pl.col("event_ts").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", time_zone="UTC").alias("ts")
    )
    sends = (
        df.filter(pl.col("event_type") == "email_send")
        .with_columns(
            pl.col("event_attributes").str.json_path_match("$.offer").cast(pl.Int32).alias("offer"),
            pl.col("ts").dt.date().alias("day"),
        )
        .select(["customer_key", "day", "offer"])
        .drop_nulls("offer")
    )
    purch = (
        df.filter(pl.col("event_type") == "purchase")
        .with_columns(
            pl.col("event_attributes")
            .str.json_path_match("$.margin")
            .cast(pl.Float64)
            .alias("margin"),
            pl.col("ts").dt.date().alias("day"),
        )
        .select(["customer_key", "day", "margin"])
        .drop_nulls("margin")
    )
    agg = purch.group_by(["customer_key", "day"]).agg(
        [pl.col("margin").sum().alias("margin"), pl.len().alias("count")]
    )
    out = sends.join(agg, on=["customer_key", "day"], how="left").with_columns(
        [pl.col("margin").fill_null(0.0), pl.col("count").fill_null(0).cast(pl.Float64)]
    )
    return out.sort(["customer_key", "day"])


def fit(
    p: pl.DataFrame, outcome: str = "count", shrink: float = 2.0
) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """(V[customer][arm], pooled arm means) with empirical shrinkage.

    ``shrink`` is the number of pooled pseudo-cells added to each customer cell; it is
    a fixed capacity choice, not fitted on the evaluation window.
    """
    if outcome not in ("count", "margin"):
        raise ValueError("outcome must be 'count' or 'margin'")
    workers = int(p["offer"].max()) + 1
    pooled = {a: float(p.filter(pl.col("offer") == a)[outcome].mean()) for a in range(workers)}
    V: dict[str, dict[int, float]] = {}
    for key, g in p.group_by("customer_key", maintain_order=True):
        k = str(key[0])
        vals: dict[int, float] = {}
        for a in range(workers):
            sub = g.filter(pl.col("offer") == a)
            n = sub.height
            raw = float(sub[outcome].mean()) if n else pooled[a]
            vals[a] = (n * raw + shrink * pooled[a]) / (n + shrink) if n + shrink > 0 else pooled[a]
        V[k] = vals
    return V, pooled


def policy(V: dict[str, dict[int, float]], p: pl.DataFrame, outcome: str = "count") -> dict:
    """Held-out value of the argmax policy vs random and the best constant arm.

    Each customer contributes the mean outcome of their chosen arm over the evaluation
    cells where that arm was actually drawn (the randomized assignment is the
    propensity; no inverse weighting is needed for an arm-conditional mean). A
    customer with no evaluation cell at the chosen arm falls back to the pooled mean
    for that arm, so the policy is defined for every fitted customer.
    """
    workers = int(p["offer"].max()) + 1
    pooled = {a: float(p.filter(pl.col("offer") == a)[outcome].mean()) for a in range(workers)}
    best_const = max(pooled, key=pooled.get)
    by_key: dict[str, dict[int, float]] = {}
    for key, g in p.group_by("customer_key", maintain_order=True):
        k = str(key[0])
        by_key[k] = {
            a: float(g.filter(pl.col("offer") == a)[outcome].mean())
            if g.filter(pl.col("offer") == a).height
            else pooled[a]
            for a in range(workers)
        }
    keys = [k for k in V if k in by_key]
    chosen = {k: max(range(workers), key=lambda a: V[k][a]) for k in keys}
    vals = np.asarray([by_key[k][chosen[k]] for k in keys], dtype=np.float64)
    rnd = np.asarray(
        [float(np.mean([by_key[k][a] for a in range(workers)])) for k in keys],
        dtype=np.float64,
    )
    est = np.asarray([V[k][chosen[k]] for k in keys], dtype=np.float64)
    return {
        "n_customers": len(keys),
        "chosen": chosen,
        "policy": float(vals.mean()) if len(vals) else None,
        "random": float(rnd.mean()) if len(rnd) else None,
        "random_all": float(p[outcome].mean()),
        "best_constant": float(pooled[best_const]),
        "best_constant_arm": int(best_const),
        "estimated_value": float(est.mean()) if len(est) else None,
        "pooled": pooled,
        "policy_by_customer": vals,
        "random_by_customer": rnd,
    }


def known_effect(V: dict[str, dict[int, float]], truth_path: Path) -> dict:
    """Per-customer argmax vs the generator's declared offer optimum."""
    truth = {
        c["customer"]: int(c["offer_opt"])
        for c in json.loads(truth_path.read_text())["customers"]
        if c.get("offer_opt") is not None
    }
    keys = [k for k in V if k in truth]
    pred = np.asarray([max(V[k], key=V[k].get) for k in keys], dtype=np.int64)
    opt = np.asarray([truth[k] for k in keys], dtype=np.int64)
    hits = int((pred == opt).sum())
    n = len(keys)
    workers = len(next(iter(V.values())))
    chance = 1.0 / workers
    from collections import Counter

    majority = Counter(opt.tolist()).most_common(1)[0][0]
    majority_acc = float((np.full(n, majority) == opt).mean()) if n else None
    from scipy.stats import binomtest

    p = float(binomtest(hits, n, chance, alternative="greater").pvalue) if n else 1.0
    return {
        "n_customers": n,
        "hits": hits,
        "accuracy": round(hits / n, 6) if n else None,
        "chance": round(chance, 6),
        "majority_arm": int(majority),
        "majority_accuracy": round(majority_acc, 6) if majority_acc is not None else None,
        "binom_p": p,
        "predicted_distribution": {str(a): int((pred == a).sum()) for a in range(workers)},
    }
