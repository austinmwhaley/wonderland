"""rabbit_hole APEX — the canonical, controlled, apparel omni-channel event stream.

The single fixture that must be able to train **any** Layer-C plugin off the frozen Layer-B
foundation model. It models an omni-channel **clothing retailer**: a product catalogue
(category · brand · colour · size · list price · season), marketing contacts across channels
(email/sms/push/direct_mail), purchases across touchpoints (web/app/store), returns with
reasons, loyalty, acquisition, seasonality and churn.

It ships a KNOWN-TRUTH receipt for EVERY latent dimension, so each plugin has a ground-truth
signal to be certified against, plus a NULL control (`snr=0`, no structure) and an
ADVERSARIAL control opportunity (a leaked attribute that echoes a target).

Dimensions a plugin can rely on (all present, all dialed by `snr`):
  * customer latents: activity · category/department affinity · item affinity · brand affinity ·
    price sensitivity · contact-channel preference · purchase-channel mix · body size ·
    return-proneness · churn hazard · lifecycle · segment;
  * randomized marketing ACTIONS per week: **channel** · **frequency** · **hour** · **day-of-week** ·
    **offer depth**, each with a per-customer OPTIMUM (so frequency/hour/dow/channel/offer decision
    plugins are identifiable);
  * outcomes: orders (item · size · price · discount · touchpoint · store), returns (with reason),
    store visits, loyalty transitions — from which churn / reactivation / LTV / next-purchase /
    next-order-value / return-propensity / discount-sensitivity / channel-of-next-purchase /
    category-affinity / item-affinity targets are all derivable.

Run:
  python3 -m rabbit_hole.generators.apex --n-customers 20000 --window-days 540 \
      --out rabbit_hole/data/apex
  python3 -m rabbit_hole.generators.apex --n-customers 20000 --snr 0 --out rabbit_hole/data/apex_null
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

# ---- vocabulary (realistic apparel omni-channel) -------------------------------------------
CATEGORIES = ("tops", "bottoms", "dresses", "outerwear", "shoes", "accessories")
BRANDS = ("aurora", "bloom", "cobalt", "delta", "ember", "fable", "grove", "harbor")
COLORS = ("black", "white", "blue", "red", "green", "beige", "charcoal")
SIZES = ("XS", "S", "M", "L", "XL")
ACQ = ("organic", "paid_search", "social", "referral", "store")
LOYALTY = ("none", "silver", "gold", "platinum")
INCOME = ("low", "mid", "high")
CONTACT = ("email", "sms", "push", "direct_mail")  # marketing-contact channels
TOUCH = ("web", "app", "store")  # where a purchase happens
RETURN_REASONS = ("fit_small", "fit_large", "color", "quality", "changed_mind")
TYPES = ("view", "cart", "purchase")  # organic shopping funnel
P_FIXED = np.array(
    [[0.62, 0.30, 0.08], [0.18, 0.34, 0.48], [0.52, 0.23, 0.25]]  # view/cart/purchase transitions
)
REF_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class ApexSpec:
    n_customers: int = 20000
    window_days: int = 540
    seed: int = 17
    # latent structure (scaled by snr)
    rate_heterogeneity: float = 0.7
    transition: bool = True
    intent: float = 0.5
    trend: float = 0.5
    churn_hazard: float = 0.2
    seasonality: float = 0.6
    n_items: int = 400
    # contacts / actions
    action_peak: float = 0.9  # weekly response peak
    sigma_f: float = 1.0
    sigma_h: float = 2.5
    sigma_d: float = 1.8
    channel_base_aff: float = 0.35  # non-preferred-channel relative response
    offer_gain: float = 0.6  # margin gain at the customer's optimal depth
    treatment_day: int = 180  # contacts only have an effect after this day (clean OPE history)
    # observation
    base_rate: float = 0.12
    value_log_sigma: float = 0.35
    return_base: float = 0.06
    snr: float = 1.0

    def eff(self) -> dict:
        s = self.snr
        return {
            "rate": s * self.rate_heterogeneity,
            "transition": s,
            "intent": s * self.intent,
            "trend": s * self.trend,
            "churn": s * self.churn_hazard,
            "seasonality": s * self.seasonality,
            "action_peak": s * self.action_peak,
            "offer_gain": s * self.offer_gain,
            "return": s * self.return_base,
        }


def _transition(scale: float) -> np.ndarray:
    u = np.full_like(P_FIXED, 1.0 / P_FIXED.shape[1])
    return (1.0 - scale) * u + scale * P_FIXED


def build_apex(spec: ApexSpec):
    eff = spec.eff()
    rng = np.random.default_rng(spec.seed)
    T = spec.window_days
    P = _transition(eff["transition"])
    # ---- product catalogue ----
    n_items = max(1, spec.n_items)
    item_cat = rng.integers(0, len(CATEGORIES), n_items)
    item_brand = rng.integers(0, len(BRANDS), n_items)
    item_color = rng.integers(0, len(COLORS), n_items)
    item_price = np.round(np.exp(rng.normal(3.4, 0.5, n_items)), 2)  # ~$30 median

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
    cust_truth = []

    def emit(key, day, etype, *, brand="", ent_type="", ent_id="", value=0.0, **a):
        rows["customer_key"].append(key)
        rows["event_ts"].append((REF_START + timedelta(days=float(day))).isoformat())
        rows["event_type"].append(etype)
        rows["brand"].append(brand)
        rows["entity_type"].append(ent_type)
        rows["entity_id"].append(ent_id)
        rows["source_table"].append("apex")
        rows["value"].append(float(value))
        av = dict(a)
        if value:
            av["value"] = round(float(value), 2)
        attrs.append(json.dumps(av, separators=(",", ":")))

    def _pick_item(dept_pref, item_pref, affinity):
        if rng.random() < affinity:
            return int(item_pref)
        if rng.random() < 0.5:
            pool = np.flatnonzero(item_cat == dept_pref)
            return int(rng.choice(pool)) if pool.size else int(rng.integers(0, n_items))
        return int(rng.integers(0, n_items))

    mean_ev = spec.base_rate * T * np.exp(0.5 * eff["rate"] ** 2)
    n_per = max(2, int(round(T / 7.0)))
    per_days = T / n_per
    for c in range(spec.n_customers):
        key = f"ap{c:07d}"
        a_rate = float(np.exp(rng.normal(0.0, eff["rate"])))
        dept_pref = int(rng.integers(0, len(CATEGORIES)))
        item_pref = int(
            rng.choice(np.flatnonzero(item_cat == dept_pref))
            if (item_cat == dept_pref).any()
            else [rng.integers(0, n_items)]
        )
        brand_pref = int(rng.integers(0, len(BRANDS)))
        price_sens = float(rng.beta(2.0, 2.0)) * eff["offer_gain"] / max(eff["offer_gain"], 1e-9)
        price_sens = min(max(price_sens, 0.0), 1.0)
        chan_pref = int(rng.integers(0, len(CONTACT)))
        touch_w = rng.dirichlet(np.array([2.0, 1.5, 1.0]))  # web/app/store
        body_size = int(rng.integers(0, len(SIZES)))
        return_prone = float(np.clip(rng.beta(1.5, 6.0) + rng.normal(0, 0.02), 0, 1))
        seg = int(rng.integers(0, 5))
        loyal = int(rng.integers(0, len(LOYALTY)))
        acq = int(rng.integers(0, len(ACQ)))
        income = int(rng.integers(0, len(INCOME)))
        churn_day = float(rng.uniform(0.35 * T, T)) if rng.random() < eff["churn"] else float(T)
        n = max(1, int(rng.poisson(mean_ev * (a_rate / np.exp(0.5 * eff["rate"] ** 2)))))
        if eff["trend"] > 0:
            k = rng.normal(0.0, eff["trend"])
            u = rng.random(n)
            frac = u if abs(k) < 1e-6 else np.log1p(u * np.expm1(k)) / k
            times = np.sort(np.clip(frac, 0, 1) * T)
        else:
            times = np.sort(rng.random(n) * T)
        times = times[times <= churn_day]
        n = times.size
        # per-customer action optima (the decision-plugin ground truth)
        x0 = float(rng.uniform(0.0, 3.0))  # optimal weekly frequency
        h0 = float(rng.uniform(0.0, 24.0))  # optimal hour
        d0 = float(rng.uniform(0.0, 7.0))  # optimal day-of-week
        offer_opt = int(rng.integers(0, 4))  # optimal discount depth
        # organic funnel (order-1 transitions, intent-sharpened)
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
        emit(
            key,
            0.0,
            "customer_signup",
            ent_type="customer",
            ent_id=key,
            acquisition_channel=ACQ[acq],
            loyalty_tier=LOYALTY[loyal],
            income_band=INCOME[income],
            device=rng.choice(["ios", "android", "web"]).item(),
        )
        for t in range(n):
            ti = int(idx[t])
            et = TYPES[ti]
            hour = int(rng.integers(0, 24))
            dow = int((times[t] + REF_START.weekday()) % 7)
            if eff["seasonality"] > 0 and rng.random() < eff["seasonality"]:
                hour = int(rng.integers(18, 23)) if rng.random() < 0.5 else hour
                dow = 5 if rng.random() < 0.5 else dow
            pid = _pick_item(dept_pref, item_pref, 0.6 if eff["transition"] > 0 else 0.0)
            common = dict(
                hour=hour,
                dow=dow,
                category_code=CATEGORIES[int(item_cat[pid])],
                brand_name=BRANDS[int(item_brand[pid])],
            )
            if et == "purchase":
                touch = int(rng.choice(len(TOUCH), p=touch_w))
                val = float(np.exp(np.log(item_price[pid]) + rng.normal(0.0, spec.value_log_sigma)))
                emit(
                    key,
                    times[t],
                    "purchase",
                    brand=BRANDS[int(item_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid}",
                    value=val,
                    size=SIZES[body_size],
                    color=COLORS[int(item_color[pid])],
                    touchpoint=TOUCH[touch],
                    store_id=(f"s{int(rng.integers(0, 40)):02d}" if touch == 2 else ""),
                    **common,
                )
                if rng.random() < eff["return"] + return_prone * 0.2:
                    reason = RETURN_REASONS[
                        int(
                            np.argmax(
                                [
                                    0.3 + (body_size == 0),
                                    0.3 + (body_size == len(SIZES) - 1),
                                    0.1,
                                    0.1,
                                    0.2,
                                ]
                            )
                        )
                    ]
                    emit(
                        key,
                        times[t] + float(rng.uniform(2, 20)),
                        "return",
                        brand=BRANDS[int(item_brand[pid])],
                        ent_type="product",
                        ent_id=f"p{pid}",
                        value=-round(val * float(rng.uniform(0.3, 1.0)), 2),
                        size=SIZES[body_size],
                        return_reason=reason,
                    )
            else:
                emit(
                    key,
                    times[t],
                    et,
                    brand=BRANDS[int(item_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid}",
                    size=SIZES[body_size] if et == "cart" else "",
                    **common,
                )
        # ---- randomized weekly marketing ACTIONS (channel, freq, hour, dow, offer) ----
        for p in range(n_per):
            day0 = p * per_days
            if day0 > churn_day:
                break
            chan = int(rng.integers(0, len(CONTACT)))  # randomized channel
            freq = int(rng.integers(0, 4))  # randomized weekly frequency (0 = no send)
            offer = int(rng.integers(0, 4))  # randomized discount depth
            if freq == 0:
                continue
            SOFT = 0.3
            best_t = -1.0  # the week responds to the BEST-timed of its sends
            win_h = 0
            win_d = 0
            for j in range(freq):
                hour_j = int(rng.integers(0, 24))  # per-send randomized timing
                dow_j = int(rng.integers(0, 7))
                emit(
                    key,
                    day0 + float(rng.uniform(0.0, per_days)),
                    f"{CONTACT[chan]}_send",
                    ent_type="customer",
                    ent_id=key,
                    channel=chan,
                    freq=freq,
                    hour=hour_j,
                    dow=dow_j,
                    offer=offer,
                    discount_pct=round(0.05 * offer, 2),
                    week=p,
                    send_idx=j,
                )
                h_q = SOFT + (1 - SOFT) * float(np.exp(-(((hour_j - h0) / spec.sigma_h) ** 2)))
                d_q = SOFT + (1 - SOFT) * float(np.exp(-(((dow_j - d0) / spec.sigma_d) ** 2)))
                if h_q * d_q > best_t:
                    best_t, win_h, win_d = h_q * d_q, hour_j, dow_j
            if day0 <= spec.treatment_day:
                continue  # clean OPE: contacts have NO effect before treatment_day
            # Each action factor is a soft bowl in [SOFT, 1] (identifiable but high-probability,
            # so every action's optimum is recoverable per customer). Discount-sensitive
            # customers respond to contacts more overall.
            f_q = SOFT + (1 - SOFT) * float(np.exp(-(((freq - x0) / spec.sigma_f) ** 2)))
            c_q = SOFT + (1 - SOFT) * (1.0 if chan == chan_pref else 0.0)
            o_q = SOFT + (1 - SOFT) * float(np.exp(-(((offer - offer_opt) / 1.0) ** 2)))
            p_conv = (
                eff["action_peak"] * (0.5 + price_sens) * f_q * c_q * o_q * max(best_t, SOFT * SOFT)
            )
            if rng.random() < min(p_conv, 0.95):
                pid = _pick_item(dept_pref, item_pref, 0.6 if eff["transition"] > 0 else 0.0)
                touch = int(np.argmax(touch_w))
                val = float(
                    np.exp(np.log(item_price[pid]) + rng.normal(0.0, spec.value_log_sigma))
                    * (1.0 - 0.05 * offer)
                )
                emit(
                    key,
                    day0 + float(rng.uniform(0.0, per_days)),
                    "purchase",
                    brand=BRANDS[int(item_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid}",
                    value=val,
                    channel=chan,
                    freq=freq,
                    hour=win_h,
                    dow=win_d,
                    offer=offer,
                    week=p,
                    touchpoint=TOUCH[touch],
                    size=SIZES[body_size],
                    color=COLORS[int(item_color[pid])],
                    category_code=CATEGORIES[int(item_cat[pid])],
                )
        if churn_day < T:
            emit(key, churn_day, "loyalty_change", ent_type="customer", ent_id=key, to_tier="none")
        cust_truth.append(
            {
                "customer": key,
                "rate": round(a_rate, 4),
                "segment": seg,
                "loyalty": LOYALTY[loyal],
                "acquisition": ACQ[acq],
                "income": INCOME[income],
                "dept_pref": CATEGORIES[dept_pref],
                "item_pref": f"p{item_pref}",
                "brand_pref": BRANDS[brand_pref],
                "price_sensitivity": round(price_sens, 4),
                "channel_pref": CONTACT[chan_pref],
                "touch_mix": [round(float(x), 4) for x in touch_w],
                "body_size": SIZES[body_size],
                "return_proneness": round(return_prone, 4),
                "churn_day": round(churn_day, 2),
                "x0": round(x0, 3),
                "h0": round(h0, 3),
                "d0": round(d0, 3),
                "offer_opt": offer_opt,
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
        "treatment_day": spec.treatment_day,
        "categories": list(CATEGORIES),
        "brands": list(BRANDS),
        "sizes": list(SIZES),
        "contact_channels": list(CONTACT),
        "touchpoints": list(TOUCH),
        "event_types": [
            "customer_signup",
            "view",
            "cart",
            "purchase",
            "return",
            "email_send",
            "sms_send",
            "push_send",
            "direct_mail_send",
            "loyalty_change",
        ],
        # per-period randomized action keys (decision plugins); channel/dow are int-encoded
        "action_keys": ["freq", "hour", "dow", "channel", "offer"],
        "arm_key": "channel",
        "action": {
            "peak": spec.action_peak,
            "sigma_f": spec.sigma_f,
            "sigma_h": spec.sigma_h,
            "sigma_d": spec.sigma_d,
            "note": "weekly randomized channel x freq(0..3) x hour(0..23) x dow(0..6) x offer(0..3); "
            "per-customer unimodal response in each -> frequency/hour/dow/channel/offer optima.",
        },
        "note": "APEX apparel omni-channel composite: every latent dimension a Layer-C plugin "
        "needs, with a known-truth receipt. snr=0 = the NULL control.",
    }
    return df, {"meta": meta, "customers": cust_truth}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=20000, dest="n_customers")
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
