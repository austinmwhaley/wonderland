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

Domains (DEC-611): the `DOMAINS` table makes each application domain a point in THIS
parameter space — its own event-type vocabulary, its own moderately-dominant base
transition matrix, and its own per-type observation model — not a second simulator.
`domain="ecommerce"` (the default) resolves to the original `TYPES`/`P_FIXED` and the
original observation model, so the pre-DEC-611 fixtures regenerate byte-identically
(the no-op proof; recorded in DEC-611's outcome addendum).
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


# ---------------------------------------------------------------------------
# per-domain observation models (attribute JSON per event; the rng draw ORDER is
# part of the byte-identity contract — the ecommerce model reproduces the
# pre-DEC-611 per-event draws exactly)
# ---------------------------------------------------------------------------
def _ecommerce_attr(spec: DataSpec, ti: int, rng, ctx) -> dict:
    attrs = {}
    if ti == 2:  # purchase
        if spec.n_entities:
            use_pref = rng.random() < ctx["entity_affinity"]
            eid = ctx["pref"] if use_pref else int(rng.integers(0, spec.n_entities))
            attrs["entity_type"] = "product"
            attrs["entity_id"] = f"p{eid}"
            lp = ctx["ent_logprice"][eid]
        else:
            lp = 3.0
        attrs["value"] = round(float(np.exp(lp + rng.normal(0.0, spec.value_log_sigma))), 2)
    return attrs


def _sensor_attr(spec: DataSpec, ti: int, rng, ctx) -> dict:
    if ti == 0:  # telemetry: state-dependent continuous reading (DEC-611)
        return {"reading": round(float(rng.normal(20.0 + 8.0 * ctx["load"], 0.3)), 3)}
    if ti == 1:  # alarm
        return {"severity": int(1 + 3.0 * rng.random())}
    return {"crew": "crew_" + "abcd"[int(rng.integers(0, 4))]}  # maintenance


def _click_attr(spec: DataSpec, ti: int, rng, ctx) -> dict:
    if ti == 0:  # page_view
        return {"page_id": "p%d" % int(rng.integers(0, 500))}
    if ti == 1:  # search
        return {"query_id": "q%d" % int(rng.integers(0, 2000))}
    if ti == 2:  # add_to_cart
        return {"item_id": "i%d" % int(rng.integers(0, 800))}
    if ti == 3:  # checkout (the rare target class)
        return {"order_id": "o%d" % int(rng.integers(0, 1_000_000))}
    return {}  # bounce


def _server_attr(spec: DataSpec, ti: int, rng, ctx) -> dict:
    if ti == 0:  # request: load-independent latency distractor
        return {"latency_ms": round(float(np.exp(rng.normal(3.5, 0.5))), 1)}
    if ti == 1:  # error
        return {"code": int(rng.choice((500, 502, 503, 504)))}
    if ti == 2:  # deploy
        return {"version": "v%d" % int(rng.integers(0, 50))}
    return {}  # restart


# DEC-611: the domain table — single source of truth for every non-ecommerce lab
# domain. `p_base` rows are moderately dominant (no near-deterministic rows: the
# frozen `exp(b)` character ensemble must LOWER entropy on average, or the oracle
# is not CARRIED — verified per domain by the closed-form sweep recorded in the
# DEC-611 outcome addendum).
DOMAINS: dict[str, dict] = {
    "ecommerce": {
        "types": TYPES,
        "p_base": P_FIXED,
        "attr": _ecommerce_attr,
    },
    "sensor_fleet": {
        "types": ("telemetry", "alarm", "maintenance"),
        "p_base": np.array(
            [
                [0.72, 0.18, 0.10],  # healthy: mostly telemetry, drift to the others
                [0.20, 0.50, 0.30],  # alarm regime persists, then maintenance
                [0.55, 0.15, 0.30],  # maintenance: back toward telemetry
            ]
        ),
        "attr": _sensor_attr,
    },
    "clickstream_sessions": {
        "types": ("page_view", "search", "add_to_cart", "checkout", "bounce"),
        "p_base": np.array(
            [
                [0.55, 0.25, 0.10, 0.02, 0.08],
                [0.60, 0.20, 0.12, 0.01, 0.07],
                [0.45, 0.05, 0.35, 0.10, 0.05],  # cart: checkout is the rare class
                [0.70, 0.00, 0.00, 0.10, 0.20],  # checkout -> next session
                [0.95, 0.03, 0.00, 0.00, 0.02],  # bounce ends the session
            ]
        ),
        "attr": _click_attr,
    },
    "server_logs": {
        "types": ("request", "error", "deploy", "restart"),
        "p_base": np.array(
            [
                [0.76, 0.14, 0.04, 0.06],
                [0.35, 0.45, 0.00, 0.20],  # error bursts escalate to restart
                [0.20, 0.50, 0.10, 0.20],  # deploy -> regressed error burst
                [0.80, 0.08, 0.00, 0.12],  # restart -> back to load
            ]
        ),
        "attr": _server_attr,
        "diurnal_peak": 13.0 / 24.0,  # fraction of the day; truth, not a model assumption
    },
}


def _dom(spec: DataSpec) -> dict:
    try:
        return DOMAINS[spec.domain]
    except KeyError:
        raise ValueError(f"unknown domain {spec.domain!r}; known: {sorted(DOMAINS)}") from None


@dataclass
class DataSpec:
    name: str
    n_customers: int = 2000
    window_days: int = 365
    seed: int = 17
    # ---- domain (DEC-611): a point in the parameter space, not a new simulator ----
    domain: str = "ecommerce"
    # ---- latent structure (0/None = absent) ----
    rate_heterogeneity: float = 0.0  # sigma of log per-customer rate
    transition: bool = False  # order-1 Markov event types (domain p_base) vs i.i.d.
    intent: float = 0.0  # per-customer transition sharpening
    trend: float = 0.0  # per-customer rate drift sigma
    n_entities: int = 0  # >0 -> entity ids on purchases
    entity_affinity: float = 0.0
    action_peak: float = 0.0  # >0 -> exogenous actions + unimodal response
    # ---- timing shape (off by default; snr-scaled, so the null stays null) ----
    session_burst: float = 0.0  # >0 -> bursty session gaps (DEC-611)
    diurnal: float = 0.0  # >0 -> diurnal rate modulation (DEC-611)
    # ---- observation ----
    value_log_sigma: float = 0.5
    base_rate: float = 0.08
    # ---- signal strength ----
    snr: float = 1.0  # 0 -> null, 1 -> full structure

    def effective(self) -> dict:
        s = self.snr
        return {
            "rate_heterogeneity": s * self.rate_heterogeneity,
            "transition_scale": s,  # 0 -> uniform (i.i.d.), 1 -> p_base
            "intent": s * self.intent,
            "trend": s * self.trend,
            "entity_affinity": s * self.entity_affinity,
            "action_peak": s * self.action_peak,
            "session_burst": s * self.session_burst,
            "diurnal": s * self.diurnal,
        }


def _transition(eff, p_base):
    # interpolate p_base toward the row-uniform marginal by snr
    u = np.full_like(p_base, 1.0 / p_base.shape[1])
    return (1.0 - eff["transition_scale"]) * u + eff["transition_scale"] * p_base


def _session_times(rng, n: int, T: float, burst: float) -> np.ndarray:
    """Bursty session timing (DEC-611): geometric sessions; the intra-session gap is
    (1+burst)^2 faster than the inter-session gap; both scales are derived from the
    window budget so the expected total is T. Off at burst=0 (the uniform path)."""
    if n <= 1:
        return np.sort(rng.random(n) * T)
    p = burst / (1.0 + burst)
    ratio = (1.0 + burst) ** 2
    cont = rng.binomial(1, p, n - 1)
    m_w = T / ((n - 1) * (p + (1.0 - p) * ratio))
    gaps = rng.exponential(m_w, n - 1) * np.where(cont, 1.0, ratio)
    return np.concatenate(([0.0], np.cumsum(gaps)))


def _diurnal_times(rng, n: int, T: float, amp: float, peak: float) -> np.ndarray:
    """Diurnal rate modulation (DEC-611): the daily load profile
    w(h) = 1 + amp/2 * (1 + cos(2*pi*(h - peak))) is sampled by rejection from the
    uniform clock; off at amp=0 (the uniform path). `peak` is a fraction of the day."""
    t = np.empty(n, float)
    left = n
    while left > 0:  # rejection passes; each pass is fully vectorized
        cand = rng.random(left) * T
        w = 1.0 + 0.5 * amp * (1.0 + np.cos(2.0 * np.pi * ((cand % 1.0) - peak)))
        keep = rng.random(left) <= w / (1.0 + amp)
        k = int(keep.sum())
        t[n - left : n - left + k] = cand[keep]
        left -= k
    return np.sort(t)


def simulate(spec: DataSpec):
    """Return (canonical DataFrame, truth dict, per-customer latent summary)."""
    dom = _dom(spec)
    types = dom["types"]
    eff = spec.effective()
    rng = np.random.default_rng(spec.seed)
    P = (
        _transition(eff, dom["p_base"])
        if spec.transition
        else np.full_like(dom["p_base"], 1.0 / len(types))
    )
    # per-entity log-price: knowing WHICH product a customer buys predicts the next
    # order's value (beyond RFM) when affinity is high.
    ent_logprice = rng.normal(3.0, 0.8, spec.n_entities) if spec.n_entities else np.array([3.0])
    mean_events = spec.base_rate * spec.window_days * np.exp(0.5 * eff["rate_heterogeneity"] ** 2)
    rows = {"customer_key": [], "event_ts": [], "event_type": [], "source_table": [], "attrs": []}
    cust_summary = []
    attr_fn = dom["attr"]

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
            if eff["session_burst"] > 0:
                times = _session_times(rng, n, T, eff["session_burst"])
            elif eff["diurnal"] > 0:
                times = _diurnal_times(rng, n, T, eff["diurnal"], dom.get("diurnal_peak", 0.5))
            else:
                times = np.sort(rng.random(n) * T)
        # transitions
        if spec.transition and eff["transition_scale"] > 0:
            Pc = P
            if eff["intent"] > 0:
                b = rng.normal(0.0, eff["intent"])
                # log-space stable softmax (DEC-586): subtract the row max before exp;
                # algebraically identical where the raw math is finite, defined where it
                # row-underflows to 0/0 at b >= ~5.8.
                logits = np.log(np.clip(P, 1e-6, 1)) * np.exp(b)
                logits = logits - logits.max(axis=1, keepdims=True)
                L = np.exp(logits)
                Pc = L / L.sum(1, keepdims=True)
            idx = np.empty(n, int)
            idx[0] = rng.integers(0, len(types))
            draws = rng.random(n)
            for t in range(1, n):
                idx[t] = int(np.searchsorted(np.cumsum(Pc[idx[t - 1]]), draws[t]))
        else:
            idx = rng.integers(0, len(types), size=n)
        pref = int(rng.integers(0, spec.n_entities)) if spec.n_entities else -1
        ctx = {
            "entity_affinity": eff["entity_affinity"],
            "pref": pref,
            "ent_logprice": ent_logprice,
        }
        if spec.domain == "sensor_fleet":
            # device load (latent, recorded in the per-customer summary): shifts the
            # telemetry level — the value-transform axis the character ensemble cannot
            # carry through the types alone.
            ctx["load"] = float(rng.normal(0.0, 1.0))
        cs = {"customer": key, "rate": round(a, 4)}
        if "load" in ctx:
            cs["load"] = round(ctx["load"], 4)
        cust_summary.append(cs)
        for t in range(n):
            ti = int(idx[t])
            attrs = attr_fn(spec, ti, rng, ctx)
            emit(key, times[t], types[ti], attrs)
        if eff["action_peak"] > 0 and spec.domain == "ecommerce":
            # the action/response mechanism is the ecommerce exogenous design (the
            # randomized arm is a Layer-C contract); other domains reject it rather
            # than emit a type their vocabulary does not contain.
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
    incremental structure beyond the 1-gram. Domain-generic (DEC-611): the base matrix
    and the vocabulary come from the spec's domain row.
    """
    dom = _dom(spec)
    types = dom["types"]
    eff = spec.effective()
    P = (
        _transition(eff, dom["p_base"])
        if spec.transition
        else np.full_like(dom["p_base"], 1.0 / len(types))
    )
    pi = np.linalg.matrix_power(P, 50)[0]  # stationary

    def H(M):
        return float(-(M * np.log(np.clip(M, 1e-12, 1))).sum(1).dot(pi))

    base_ce = H(P)
    # oracle: if intent>0, per-customer sharpening reduces entropy by an ensemble average
    if eff["intent"] > 0:
        b = np.random.default_rng(spec.seed).normal(0.0, eff["intent"], 2000)
        ces = []
        for bi in b:
            # log-space stable softmax (DEC-586), as in simulate()
            logits = np.log(np.clip(P, 1e-6, 1)) * np.exp(bi)
            logits = logits - logits.max(axis=1, keepdims=True)
            L = np.exp(logits)
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
