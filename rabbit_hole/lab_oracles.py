"""Build Layer-A laboratory fixtures (plan step 3): for each use case, construct a
POSITIVE spec whose target is CARRIED beyond its strong model-free baseline, derive
the oracle/ceiling analytically, and write stream + truth + oracle + baseline.

Design rule: a use case is only defensible if the target depends on structure the
strong baseline CANNOT see. So each B+C head's spec adds the structure its baseline
misses (e.g. churn -> trend beyond RFM; next-value -> entity beyond RFM). A pure-rate
target is ¬CARRIED beyond RFM by construction and is therefore NO_SIGNAL, not a promo.
"""

from __future__ import annotations

import json
from pathlib import Path


from rabbit_hole.data_spec import DataSpec, simulate, transition_oracle

# use case -> positive spec (the structure the target needs, beyond the baseline)
POSITIVE: dict[str, DataSpec] = {
    "next_event": DataSpec("next_event", transition=True, intent=0.5, snr=1.0),
    "mask": DataSpec("mask", transition=True, intent=0.5, snr=1.0),
    "order": DataSpec("order", transition=True, intent=0.4, snr=1.0),
    "next_entity": DataSpec("next_entity", transition=True, n_entities=64, entity_affinity=0.7),
    "next": DataSpec("next", transition=True, intent=0.5, snr=1.0),
    # churn/retention depend on RATE DRIFT (velocity) — beyond RFM's level
    "churn": DataSpec("churn", rate_heterogeneity=0.8, trend=0.7, snr=1.0),
    "week_n_retention": DataSpec("week_n_retention", rate_heterogeneity=0.8, trend=0.7),
    "time_to_purchase": DataSpec("time_to_purchase", rate_heterogeneity=0.9, trend=0.6),
    # LTV / counts are rate-driven; add an entity/spend factor RFM can't see
    "ltv": DataSpec("ltv", rate_heterogeneity=1.0, n_entities=64, entity_affinity=0.6),
    "order_count": DataSpec(
        "order_count", rate_heterogeneity=1.0, n_entities=64, entity_affinity=0.6
    ),
    "basket_size": DataSpec(
        "basket_size", rate_heterogeneity=0.9, n_entities=64, entity_affinity=0.6
    ),
    # next-value depends on the product (entity) bought — beyond RFM
    "next_order_value": DataSpec("next_order_value", n_entities=64, entity_affinity=0.8),
    "price_sensitivity": DataSpec("price_sensitivity", n_entities=64, entity_affinity=0.7),
    # conversion/abandonment/propensity depend on the ACTION response — beyond RFM
    "purchase_propensity": DataSpec("purchase_propensity", action_peak=0.6),
    "funnel_conversion": DataSpec("funnel_conversion", transition=True, action_peak=0.5),
    "cart_abandonment": DataSpec("cart_abandonment", transition=True, rate_heterogeneity=0.8),
    "funnel_step": DataSpec("funnel_step", transition=True, rate_heterogeneity=0.8),
    "brand_affinity": DataSpec(
        "brand_affinity", transition=True, n_entities=32, entity_affinity=0.8
    ),
    "segmentation": DataSpec("segmentation", rate_heterogeneity=1.2, intent=0.6),
    "anomaly": DataSpec("anomaly", transition=True, intent=0.5, snr=1.0),
    "identity": DataSpec("identity", transition=True, rate_heterogeneity=0.8, snr=1.0),
    "rl_ope_incremental_margin": DataSpec("rl_ope_incremental_margin", action_peak=0.7),
    "optimal_action": DataSpec("optimal_action", action_peak=0.7),
    "dt": DataSpec("dt", rate_heterogeneity=0.9),
    "value": DataSpec("value", n_entities=64, entity_affinity=0.6),
    "agg": DataSpec("agg", rate_heterogeneity=1.0, transition=True),
    "sf": DataSpec("sf", transition=True, rate_heterogeneity=0.8),
    "jepa": DataSpec("jepa", transition=True, intent=0.5, rate_heterogeneity=0.5),
    "query": DataSpec("query", transition=True, rate_heterogeneity=0.9),
    "pacing": DataSpec("pacing", rate_heterogeneity=0.9, trend=0.7),
    "boundary": DataSpec("boundary", transition=True, intent=0.4),
    "uncertainty": DataSpec("uncertainty", transition=True, intent=0.5),
}

# oracle kind per family
_TRANS = {
    "next_event",
    "mask",
    "order",
    "next",
    "anomaly",
    "boundary",
    "uncertainty",
}


def oracle_for(uc: str, spec: DataSpec) -> dict:
    if uc in _TRANS:
        o = transition_oracle(spec)
        return {"kind": "transition", **o}
    if spec.action_peak > 0:
        # action-response oracle: the incremental margin is CARRIED by the arm (policy
        # confounded, so RFM/logging-policy can't recover it) — non-zero by construction.
        return {
            "kind": "action_response",
            "baseline_ce": None,
            "oracle_ce": None,
            "incremental": round(float(spec.action_peak), 4),
        }
    # rate/value oracle: the target is a function of the per-customer latent rate
    # (heterogeneity) which RFM proxies; the ADDED entity factor is what beats RFM.
    inc = round(float(spec.entity_affinity if spec.n_entities else spec.rate_heterogeneity), 4)
    return {"kind": "rate_or_entity", "baseline_ce": None, "oracle_ce": None, "incremental": inc}


def build_positive(uc: str, root="rabbit_hole/data/lab", n_bc=8000, n_b=4000):
    from rabbit_hole.lab import USE_CASES

    spec = POSITIVE[uc]
    spec.name = uc
    # size the fixture to its gate: B+C cases need enough held-out for the RFM+GBM
    # bootstrap (DEC-078/079); B-only cases can be smaller.
    card = next(u.card for u in USE_CASES if u.id == uc)
    spec.n_customers = n_bc if "C" in card else n_b
    d = Path(root) / uc
    d.mkdir(parents=True, exist_ok=True)
    df, truth, _ = simulate(spec)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / f"{uc}_truth.json").write_text(json.dumps(truth, indent=1))
    orc = oracle_for(uc, spec)
    (d / "oracle.json").write_text(json.dumps({"use_case": uc, **orc}, indent=1))
    (d / "baseline.json").write_text(
        json.dumps({"use_case": uc, "baseline": _baseline_name(uc)}, indent=1)
    )
    return d


def _baseline_name(uc: str) -> str:
    from rabbit_hole.lab import USE_CASES

    return next(u.baseline for u in USE_CASES if u.id == uc)


def build_negative(uc: str, root="rabbit_hole/data/lab", n=8000):
    """Negative control: the SAME use case with all structure off (snr=0) -> the
    target is NOT carried; the encoder must read NO_SIGNAL here."""
    import dataclasses

    spec = dataclasses.replace(POSITIVE[uc], name=uc, snr=0.0, n_customers=n)
    d = Path(root) / uc / "negative"
    d.mkdir(parents=True, exist_ok=True)
    df, truth, _ = simulate(spec)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / f"{uc}_negative_truth.json").write_text(json.dumps(truth, indent=1))
    return d


def build_adversarial(uc: str, root="rabbit_hole/data/lab", n=8000):
    """Adversarial control (leakage trap): the target is NOT carried (snr=0), but a
    never-masked attribute ECHOES the target, so a LEAKY gate would false-PASS while
    a correct (fully-redacting) gate must read NO_SIGNAL. Here the case is the
    transition family (the masked event's type is echoed in an attribute)."""
    import dataclasses

    import polars as pl

    spec = dataclasses.replace(POSITIVE[uc], name=uc, snr=0.0, n_customers=n)
    df, truth, _ = simulate(spec)
    df = df.sort(["customer_key", "event_ts"]).with_columns(
        pl.col("event_type").shift(-1).over("customer_key").fill_null("").alias("_leak")
    )
    # echo the NEXT event type into the JSON attributes (a spurious "tell")
    attrs = []
    for a, lk in zip(df["event_attributes"].to_list(), df["_leak"].to_list(), strict=True):
        obj = json.loads(a)
        obj["echo_next"] = lk
        attrs.append(json.dumps(obj, separators=(",", ":")))
    df = df.with_columns(pl.Series("event_attributes", attrs)).drop("_leak")
    d = Path(root) / uc / "adversarial"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    (d / f"{uc}_adversarial_truth.json").write_text(json.dumps(truth, indent=1))
    return d


def main(argv=None):  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rabbit_hole/data/lab")
    ap.add_argument("--use-case", default=None)
    a = ap.parse_args(argv)
    ucs = [a.use_case] if a.use_case else list(POSITIVE)
    for uc in ucs:
        d = build_positive(uc, a.out)
        build_negative(uc, a.out)
        build_adversarial(uc, a.out)
        print(f"[lab] {uc}: {d}")


if __name__ == "__main__":
    main()
