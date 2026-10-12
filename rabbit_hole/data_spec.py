"""Parameterized Layer-A data spec (plan step 2).

ONE simulator. A stream is a point in a parameter space:
  * latent structure — rate heterogeneity, transition matrix, per-customer intent,
    trend, entity affinity, action/response;
  * observation model — value noise, (de)richment;
  * signal strength (SNR) — dials the structure toward the NULL (snr=0 => uniform /
    no structure; snr=1 => full). This is the knob for the high->low difficulty sweep.

Truth AND oracle are derived from the SAME spec (single source of truth). A fixture
is `write_fixture(spec, dir)` -> canonical stream + `<name>_truth.json` + oracle.json +
baseline.json, ready for the coverage gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

TYPES = ("view", "cart", "purchase")
P_FIXED = np.array([[0.60, 0.30, 0.10], [0.20, 0.30, 0.50], [0.50, 0.20, 0.30]])
REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class DataSpec:
    name: str
    n_customers: int = 2000
    window_days: int = 365
    seed: int = 17
    # ---- latent structure (0/None = absent) ----
    rate_heterogeneity: float = 0.0  # sigma of log per-customer rate
    transition: bool = False  # order-1 Markov event types (P_FIXED) vs i.i.d.
    intent: float = 0.0  # per-customer transition sharpening
    trend: float = 0.0  # per-customer rate drift sigma
    n_entities: int = 0  # >0 -> entity ids on purchases
    entity_affinity: float = 0.0
    action_peak: float = 0.0  # >0 -> exogenous actions + unimodal response
    # ---- observation ----
    value_log_sigma: float = 0.5
    base_rate: float = 0.08
    # ---- signal strength ----
    snr: float = 1.0  # 0 -> null, 1 -> full structure

    def effective(self) -> dict:
        s = self.snr
        return {
            "rate_heterogeneity": s * self.rate_heterogeneity,
            "transition_scale": s,  # 0 -> uniform (i.i.d.), 1 -> P_FIXED
            "intent": s * self.intent,
            "trend": s * self.trend,
            "entity_affinity": s * self.entity_affinity,
            "action_peak": s * self.action_peak,
        }


def _transition(eff):
    # interpolate P_FIXED toward the row-uniform marginal by snr
    u = np.full_like(P_FIXED, 1.0 / P_FIXED.shape[1])
    return (1.0 - eff["transition_scale"]) * u + eff["transition_scale"] * P_FIXED


def simulate(spec: DataSpec):
    """Return (canonical DataFrame, truth dict, per-customer latent summary)."""
    eff = spec.effective()
    rng = np.random.default_rng(spec.seed)
    P = _transition(eff) if spec.transition else np.full_like(P_FIXED, 1.0 / len(TYPES))
    # per-entity log-price: knowing WHICH product a customer buys predicts the next
    # order's value (beyond RFM) when affinity is high.
    ent_logprice = rng.normal(3.0, 0.8, spec.n_entities) if spec.n_entities else np.array([3.0])
    mean_events = spec.base_rate * spec.window_days * np.exp(0.5 * eff["rate_heterogeneity"] ** 2)
    rows = {"customer_key": [], "event_ts": [], "event_type": [], "source_table": [], "attrs": []}
    cust_summary = []

    def emit(key, day, etype, attrs):
        rows["customer_key"].append(key)
        rows["event_ts"].append((REF_START + timedelta(days=float(day))).isoformat())
        rows["event_type"].append(etype)
        rows["source_table"].append("lab")
        rows["attrs"].append(json.dumps(attrs, separators=(",", ":")))

    for c in range(spec.n_customers):
        key = f"lc{c:07d}"
        a = float(np.exp(rng.normal(0.0, eff["rate_heterogeneity"])))
        n = max(
            1, int(rng.poisson(mean_events * (a / np.exp(0.5 * eff["rate_heterogeneity"] ** 2))))
        )
        T = spec.window_days
        if eff["trend"] > 0:
            k = rng.normal(0.0, eff["trend"])
            u = rng.random(n)
            frac = u if abs(k) < 1e-6 else np.log1p(u * np.expm1(k)) / k
            times = np.sort(np.clip(frac, 0, 1) * T)
        else:
            times = np.sort(rng.random(n) * T)
        # transitions
        if spec.transition and eff["transition_scale"] > 0:
            Pc = P
            if eff["intent"] > 0:
                b = rng.normal(0.0, eff["intent"])
                L = np.exp(np.log(np.clip(P, 1e-6, 1)) * np.exp(b))
                Pc = L / L.sum(1, keepdims=True)
            idx = np.empty(n, int)
            idx[0] = rng.integers(0, len(TYPES))
            draws = rng.random(n)
            for t in range(1, n):
                idx[t] = int(np.searchsorted(np.cumsum(Pc[idx[t - 1]]), draws[t]))
        else:
            idx = rng.integers(0, len(TYPES), size=n)
        pref = int(rng.integers(0, spec.n_entities)) if spec.n_entities else -1
        cust_summary.append({"customer": key, "rate": round(a, 4)})
        for t in range(n):
            ti = int(idx[t])
            attrs = {}
            if ti == TYPES.index("purchase"):
                if spec.n_entities:
                    use_pref = rng.random() < eff["entity_affinity"]
                    eid = pref if use_pref else int(rng.integers(0, spec.n_entities))
                    attrs["entity_type"] = "product"
                    attrs["entity_id"] = f"p{eid}"
                    lp = ent_logprice[eid]
                else:
                    lp = 3.0
                attrs["value"] = round(float(np.exp(lp + rng.normal(0.0, spec.value_log_sigma))), 2)
            emit(key, times[t], TYPES[ti], attrs)
        if eff["action_peak"] > 0:
            x0 = float(rng.uniform(0.0, 2.4))
            cust_summary[-1]["optimal_arm"] = round(x0, 3)
            n_per = 12
            per_days = T / n_per
            for p in range(n_per):
                logits = np.array([0.0, 0.4, 0.8]) + 0.7 * (a - 1.0) * np.array([0, 1, 2])
                pr = np.exp(logits - logits.max())
                pr /= pr.sum()
                arm = int(rng.choice(3, p=pr))
                day = p * per_days + float(rng.random()) * per_days
                if rng.random() >= 0.1:
                    emit(
                        key,
                        day,
                        "action_send",
                        {"arm": arm, "discount_pct": round(0.1 * arm, 2), "campaign_id": f"c{p}"},
                    )
                if rng.random() < eff["action_peak"] * np.exp(-((arm - x0) ** 2)):
                    emit(
                        key,
                        day + 0.05,
                        "purchase",
                        {"value": round(float(np.exp(rng.normal(3.0, spec.value_log_sigma))), 2)},
                    )

    df = (
        pl.DataFrame(rows)
        .select(["customer_key", "event_ts", "event_type", "source_table", "attrs"])
        .rename({"attrs": "event_attributes"})
    )
    truth = {
        "name": spec.name,
        "spec": asdict(spec),
        "effective": eff,
        "transition_matrix": P.tolist(),
        "note": "truth + oracle are DERIVED from this spec (single source of truth).",
    }
    return df, truth, cust_summary


def transition_oracle(spec: DataSpec) -> dict:
    """Closed-form next-event ceiling vs the 1-gram baseline.

    baseline (1-gram) CE = E_prev[H(P[prev])] over the stationary distribution of prev;
    oracle CE = E_c[H(P_c)] (knowing the customer's true transition). The gap is the
    incremental structure beyond the 1-gram.
    """
    eff = spec.effective()
    P = _transition(eff) if spec.transition else np.full_like(P_FIXED, 1.0 / len(TYPES))
    pi = np.linalg.matrix_power(P, 50)[0]  # stationary

    def H(M):
        return float(-(M * np.log(np.clip(M, 1e-12, 1))).sum(1).dot(pi))

    base_ce = H(P)
    # oracle: if intent>0, per-customer sharpening reduces entropy by an ensemble average
    if eff["intent"] > 0:
        b = np.random.default_rng(spec.seed).normal(0.0, eff["intent"], 2000)
        ces = []
        for bi in b:
            L = np.exp(np.log(np.clip(P, 1e-6, 1)) * np.exp(bi))
            Pc = L / L.sum(1, keepdims=True)
            ces.append(H(Pc))
        oracle_ce = float(np.mean(ces))
    else:
        oracle_ce = base_ce
    return {
        "baseline_ce": round(base_ce, 4),
        "oracle_ce": round(oracle_ce, 4),
        "incremental": round(base_ce - oracle_ce, 4),
    }


def write_fixture(spec: DataSpec, root="rabbit_hole/data/lab"):
    """Write stream + truth + oracle + baseline receipts for a use case."""
    d = Path(root) / spec.name
    d.mkdir(parents=True, exist_ok=True)
    df, truth, _ = simulate(spec)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / f"{spec.name}_truth.json").write_text(json.dumps(truth, indent=1))
    orc = transition_oracle(spec)
    (d / "oracle.json").write_text(json.dumps({"target": "next_event", **orc}, indent=1))
    (d / "baseline.json").write_text(
        json.dumps({"target": "next_event", "ce": orc["baseline_ce"]}, indent=1)
    )
    return d
