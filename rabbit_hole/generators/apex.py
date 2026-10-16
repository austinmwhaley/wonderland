"""rabbit_hole APEX synthetic stream — the composite, end-to-end controlled fixture.

The Layer-A laboratory (`rabbit_hole.lab`) is the per-dimension ONTOLOGY (one fixture
per use case, each isolating a single structural dimension). The **apex** is its
COMPOSITE: one synthetic stream that stacks *every* dimension the lab isolates at once,
on top of realistic multi-domain structure, so `looking_glass` can be certified
end-to-end from A → B (universality) and B + C (decision / OPE / white_queen).

Composed dimensions (each with a KNOWN-TRUTH receipt; all dialed by `snr`, so
`snr=0` is the NULL control in which a capability model must read NO_SIGNAL):

  * realistic multi-domain events (signup · view · cart · purchase · return · cancel ·
    email_send) over multiple brands and high-cardinality products with categorical
    attributes (category/aisle/department);
  * per-customer **rate heterogeneity**, **rate drift** (velocity), **order-1 Markov**
    transitions sharpened by **intent**, **entity affinity**, **segmentation**;
  * **seasonality** (time-of-day, day-of-week) and a **churn hazard** (customers leave);
  * exogenous **randomized ACTIONS**: a weekly email **frequency** (0..3) and a send
    **hour** (0..23) with a per-customer unimodal **response** (`x0`, `h0`);
  * a randomized **ARM** (discount 0/1/2) with a per-customer **incremental margin**
    effect — the ground truth every OPE / decision gate needs;
  * a log-normal **value** distribution per product.

Output: a canonical Parquet stream (the 5 canonical fields + entity/value/attrs) and a
`apex_truth.json` receipt (population params + every per-customer latent + known
effects). Deterministic for a given seed.

Run:
  python3 -m rabbit_hole.generators.apex --n-customers 8000 --window-days 540 \
      --out rabbit_hole/data/apex
  python3 -m rabbit_hole.generators.apex --n-customers 8000 --snr 0 \
      --out rabbit_hole/data/apex_null          # the NULL control
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

# order-1 transition over the shopping vocabulary (view -> cart -> purchase);
# `return`/`cancel` are terminal-ish side branches added separately.
TYPES = ("view", "cart", "purchase")
P_FIXED = np.array(
    [
        [0.62, 0.30, 0.08],  # view  -> view/cart/purchase
        [0.18, 0.34, 0.48],  # cart  -> ...
        [0.52, 0.23, 0.25],  # purchase -> ...
    ]
)
BRANDS = ("aurora", "bloom", "cobalt", "delta", "ember")
CHANNELS = ("organic", "paid_search", "social", "referral", "email")
SEGMENTS = ("browsers", "loyalists", "deal_hunters", "lapsed")
DEPARTMENTS = ("grocery", "home", "beauty", "electronics", "apparel")
REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class ApexSpec:
    n_customers: int = 8000
    window_days: int = 540
    seed: int = 17
    # latent structure (scaled by snr)
    rate_heterogeneity: float = 0.1
    transition: bool = True
    intent: float = 0.6
    trend: float = 0.7
    n_entities: int = 256
    entity_affinity: float = 0.7
    n_products_per_dept: int = 8  # categorical attribute cardinality
    action_peak: float = 0.6  # weekly-response peak (freq/hour)
    sigma_f: float = 1.0
    sigma_h: float = 4.0
    arm_gain: float = 4.0  # per-unit-arm multiplicative margin gain (strong, identifiable)
    seasonality: float = 0.5
    churn_hazard: float = 0.15  # fraction of customers who leave over the window
    treatment_day: int = 365  # the arm treatment KICKS IN after this day (as_of = this)
    base_rate: float = 0.25
    value_log_sigma: float = 0.2
    snr: float = 1.0

    def effective(self) -> dict:
        s = self.snr
        return {
            "rate_heterogeneity": s * self.rate_heterogeneity,
            "transition_scale": s,
            "intent": s * self.intent,
            "trend": s * self.trend,
            "entity_affinity": s * self.entity_affinity,
            "action_peak": s * self.action_peak,
            "arm_gain": s * self.arm_gain,
            "seasonality": s * self.seasonality,
            "churn_hazard": s * self.churn_hazard,
        }


def _transition(scale: float) -> np.ndarray:
    u = np.full_like(P_FIXED, 1.0 / P_FIXED.shape[1])
    return (1.0 - scale) * u + scale * P_FIXED


def build_apex(spec: ApexSpec):
    eff = spec.effective()
    rng = np.random.default_rng(spec.seed)
    P = _transition(eff["transition_scale"])
    T = spec.window_days
    # product catalogue: id -> (brand, department, log-price)
    n_prod = max(1, spec.n_entities)
    prod_brand = rng.integers(0, len(BRANDS), n_prod)
    prod_dept = rng.integers(0, len(DEPARTMENTS), n_prod)
    prod_aisle = (prod_dept * 10 + (np.arange(n_prod) % spec.n_products_per_dept)).astype(int)
    prod_logprice = rng.normal(3.0, 0.7, n_prod)
    # an "arm gain" catalogue per customer is drawn below; the population mean effect is
    # E[margin | arm=a] = exp(arm_gain * a) * base -> a KNOWN, monotone incremental lift.
    rows = {
        "customer_key": [],
        "event_ts": [],
        "event_type": [],
        "brand": [],
        "entity_type": [],
        "entity_id": [],
        "source_table": [],
        "value": [],
    }
    attrs: list[str] = []
    truth_customers = []

    def emit(key, day, etype, *, brand="", ent_type="", ent_id="", value=0.0, **a):
        rows["customer_key"].append(key)
        rows["event_ts"].append((REF_START + timedelta(days=float(day))).isoformat())
        rows["event_type"].append(etype)
        rows["brand"].append(brand)
        rows["entity_type"].append(ent_type)
        rows["entity_id"].append(ent_id)
        rows["source_table"].append("apex")
        rows["value"].append(float(value))
        # monetary value rides BOTH the canonical `value` column AND event_attributes
        # (`$.value`) — the OPE / decision gates read the attribute form.
        av = dict(a)
        if value:
            av["value"] = round(float(value), 2)
        attrs.append(json.dumps(av, separators=(",", ":")))

    mean_ev = spec.base_rate * T * np.exp(0.5 * eff["rate_heterogeneity"] ** 2)
    n_per = max(2, int(round(T / 7.0)))  # weekly action cadence
    per_days = T / n_per
    for c in range(spec.n_customers):
        key = f"ap{c:07d}"
        a_rate = float(np.exp(rng.normal(0.0, eff["rate_heterogeneity"])))
        seg = int(rng.integers(0, len(SEGMENTS)))
        channel = int(rng.integers(0, len(CHANNELS)))
        # churn: with prob hazard, the customer leaves at day U(0.3T, T)
        churn_day = float(rng.uniform(0.3 * T, T)) if rng.random() < eff["churn_hazard"] else T
        # activity count (Poisson heterogeneity), truncated at the churn day
        n = max(
            1, int(rng.poisson(mean_ev * (a_rate / np.exp(0.5 * eff["rate_heterogeneity"] ** 2))))
        )
        # rate DRIFT (velocity): a per-customer time-warp (trend), or decay
        if eff["trend"] > 0:
            k = rng.normal(0.0, eff["trend"])
            u = rng.random(n)
            frac = u if abs(k) < 1e-6 else np.log1p(u * np.expm1(k)) / k
            times = np.sort(np.clip(frac, 0, 1) * T)
        else:
            times = np.sort(rng.random(n) * T)
        times = times[times <= churn_day]
        n = times.size
        # transitions (order-1) with per-customer intent sharpening
        idx = np.empty(n, int)
        if n:
            idx[0] = int(rng.integers(0, len(TYPES)))
            Pc = P
            if eff["intent"] > 0:
                b = rng.normal(0.0, eff["intent"])
                L = np.exp(np.log(np.clip(P, 1e-6, 1)) * np.exp(b))
                Pc = L / L.sum(1, keepdims=True)
            draws = rng.random(n)
            for t in range(1, n):
                idx[t] = int(np.searchsorted(np.cumsum(Pc[idx[t - 1]]), draws[t]))
        pref = int(rng.integers(0, n_prod))
        # per-customer action optima + a RANDOMIZED per-customer treatment arm
        # (persistent across sends) with a personalizable, unimodal margin response:
        # the optimal arm is the customer's `arm_pref`. The rotated assignment is the
        # logged propensity every OPE/decision gate needs.
        x0 = float(rng.uniform(0.0, 3.0))
        h0 = float(rng.uniform(0.0, 24.0))
        arm_gain_c = eff["arm_gain"] * float(np.exp(rng.normal(0.0, 0.4)))
        arm_c = int(rng.integers(0, 3))  # randomized assigned arm (logged)
        # the true optimal arm is a function of the customer's ACTIVITY TIER (a latent `E`
        # encodes strongly from the history) — so `E` can PERSONALIZE the arm; the randomized
        # assignment is the logged propensity. The classic A/B OPE test white_queen certifies
        # (a policy from E beats uniform logging).
        arm_pref_c = int(np.clip(np.digitize(a_rate, [0.8, 1.3]), 0, 2))
        # the treatment scales the customer's margin by a unimodal response peaked at their
        # optimal arm — a strong, personalizable incremental margin lift on ALL purchases.
        lift_c = 1.0 + arm_gain_c * float(np.exp(-(((arm_c - arm_pref_c) / 0.8) ** 2)))
        # signup
        emit(
            key,
            0.0,
            "customer_signup",
            brand="",
            ent_type="customer",
            ent_id=key,
            acquisition_channel=CHANNELS[channel],
            segment=SEGMENTS[seg],
        )
        for t in range(n):
            ti = int(idx[t])
            et = TYPES[ti]
            hour = int(rng.integers(0, 24))
            dow = int((times[t] + REF_START.weekday()) % 7)
            # seasonality: view/purchase cluster in the evening & on weekends
            if eff["seasonality"] > 0 and rng.random() < eff["seasonality"]:
                hour = int(rng.integers(18, 23)) if et != "cart" else hour
                dow = 5 if rng.random() < 0.5 else dow
            if et == "purchase":
                use_pref = rng.random() < eff["entity_affinity"]
                pid = pref if use_pref else int(rng.integers(0, n_prod))
                lp = prod_logprice[pid]
                # the ARM treatment only affects purchases AFTER the treatment day, so the
                # history up to as_of is PRE-treatment (no leakage into E — a valid OPE).
                _lift = lift_c if times[t] > spec.treatment_day else 1.0
                val = float(np.exp(lp + rng.normal(0.0, spec.value_log_sigma)) * _lift)
                emit(
                    key,
                    times[t],
                    "purchase",
                    brand=BRANDS[int(prod_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid}",
                    value=val,
                    hour=hour,
                    dow=dow,
                    category_code=f"c{int(prod_dept[pid])}",
                    aisle=f"a{int(prod_aisle[pid])}",
                    department=DEPARTMENTS[int(prod_dept[pid])],
                )
                if rng.random() < 0.06:  # occasional return
                    emit(
                        key,
                        times[t] + float(rng.random()),
                        "return",
                        brand=BRANDS[int(prod_brand[pid])],
                        ent_type="product",
                        ent_id=f"p{pid}",
                        value=-round(val * float(rng.uniform(0.2, 1.0)), 2),
                    )
            else:
                pid = int(rng.integers(0, n_prod))
                emit(
                    key,
                    times[t],
                    et,
                    brand=BRANDS[int(prod_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid}",
                    hour=hour,
                    dow=dow,
                    category_code=f"c{int(prod_dept[pid])}",
                    aisle=f"a{int(prod_aisle[pid])}",
                    department=DEPARTMENTS[int(prod_dept[pid])],
                )
        # ---- randomized weekly ACTIONS (freq, hour) + ARM, with known response ----
        for p in range(n_per):
            day0 = p * per_days
            if day0 > churn_day:
                break
            f = int(rng.integers(0, 4))  # randomized weekly frequency (0 = no send)
            arm = arm_c  # the customer's randomized assigned (logged) arm — persistent
            if f == 0:
                continue
            h_week = int(rng.integers(0, 24))  # randomized send hour
            for j in range(f):
                emit(
                    key,
                    day0 + float(rng.uniform(0.0, per_days)),
                    "email_send",
                    brand="",
                    ent_type="customer",
                    ent_id=key,
                    freq=f,
                    hour=h_week,
                    arm=arm,
                    discount_pct=round(0.05 * arm, 2),
                    week=p,
                    send_idx=j,
                )
            p_conv = (
                eff["action_peak"]
                * np.exp(-(((f - x0) / spec.sigma_f) ** 2))
                * np.exp(-(((h_week - h0) / spec.sigma_h) ** 2))
            )
            if rng.random() < p_conv:
                pid = (
                    pref if rng.random() < eff["entity_affinity"] else int(rng.integers(0, n_prod))
                )
                # ARM: the customer's treatment lift (post-treatment-day only)
                _lift = lift_c if day0 > spec.treatment_day else 1.0
                val = float(
                    np.exp(prod_logprice[pid] + rng.normal(0.0, spec.value_log_sigma)) * _lift
                )
                emit(
                    key,
                    day0 + float(rng.uniform(0.0, per_days)),
                    "purchase",
                    brand=BRANDS[int(prod_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid}",
                    value=val,
                    arm=arm,
                    week=p,
                    category_code=f"c{int(prod_dept[pid])}",
                    aisle=f"a{int(prod_aisle[pid])}",
                    department=DEPARTMENTS[int(prod_dept[pid])],
                )
        # a cancel event marks the churn
        if churn_day < T:
            emit(key, churn_day, "cancel", brand="", ent_type="customer", ent_id=key)
        truth_customers.append(
            {
                "customer": key,
                "rate": round(a_rate, 4),
                "segment": SEGMENTS[seg],
                "channel": CHANNELS[channel],
                "entity_pref": int(pref),
                "x0": round(x0, 3),
                "h0": round(h0, 3),
                "arm_assign": int(arm_c),
                "arm_pref": int(arm_pref_c),
                "arm_gain": round(arm_gain_c, 4),
                "churn_day": round(churn_day, 2),
                "n_events": int(n),
            }
        )
    df = pl.DataFrame(rows).with_columns(pl.Series("event_attributes", attrs))
    meta = {
        "name": "apex",
        "spec": asdict(spec),
        "effective": eff,
        "transition_matrix": P.tolist(),
        "n_customers": spec.n_customers,
        "window_days": spec.window_days,
        "seed": spec.seed,
        "event_types": [
            "customer_signup",
            "view",
            "cart",
            "purchase",
            "return",
            "email_send",
            "cancel",
        ],
        "action_keys": ["freq", "hour"],
        "arm_key": "arm",
        "arm_levels": 3,
        "action": {
            "peak": spec.action_peak,
            "sigma_f": spec.sigma_f,
            "sigma_h": spec.sigma_h,
            "note": "weekly randomized freq (0..3, 0=no send) + randomized hour; unimodal "
            "per-customer response peaked at (x0,h0).",
        },
        "arm": {
            "levels": 3,
            "gain": spec.arm_gain,
            "note": "randomized discount arm 0/1/2; realized margin scales by "
            "exp(per-customer gain * arm) -> a KNOWN monotone incremental lift.",
        },
        "note": "APEX composite: every lab dimension at once + realistic multi-domain "
        "structure. Truth + oracle relationships are DERIVED from this spec.",
    }
    return df, {"meta": meta, "customers": truth_customers}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=8000, dest="n_customers")
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--out", default="rabbit_hole/data/apex")
    a = ap.parse_args(argv)
    spec = ApexSpec(n_customers=a.n_customers, window_days=a.window_days, seed=a.seed, snr=a.snr)
    df, truth = build_apex(spec)
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    name = "apex_null" if a.snr < 1.0 else "apex"
    (d / f"{name}_truth.json").write_text(json.dumps(truth, indent=1))
    print(
        f"[apex] {df.height} events, {a.n_customers} customers, snr={a.snr} -> {d} "
        f"({len(set(df['event_type'].to_list()))} event types)"
    )
    return d


if __name__ == "__main__":
    main()
