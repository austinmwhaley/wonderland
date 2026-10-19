"""Tests for the rabbit_hole REFERENCE apparel omni-channel stream (rabbit_hole/generators/reference.py).

The reference is the canonical controlled Layer-A fixture: it must carry a KNOWN-TRUTH signal for
every Layer-C plugin (supervised / unsupervised / decision). These tests pin determinism, the
schema, the action-key declaration, and the recoverability of the decision optima, the category
affinity, and the clean (pre/post treatment_day) OPE structure.
"""

from __future__ import annotations

import polars as pl

from rabbit_hole.generators.reference import ReferenceSpec, build_reference


def _json_path(df, col, path):
    return df.with_columns(pl.col("event_attributes").str.json_path_match(path).alias(col))


def test_reference_deterministic_and_schema():
    s = ReferenceSpec(n_customers=150, window_days=240, seed=3)
    df1, t1 = build_reference(s)
    df2, _ = build_reference(s)
    assert df1.equals(df2)
    assert {
        "customer_key",
        "event_ts",
        "event_type",
        "event_attributes",
        "brand",
        "entity_type",
        "entity_id",
        "value",
    }.issubset(set(df1.columns))
    et = set(df1["event_type"].unique().to_list())
    assert {"view", "cart", "purchase", "return", "email_send", "customer_signup"}.issubset(et)
    m = t1["meta"]
    assert m["action_keys"] == ["freq", "hour", "dow", "channel", "offer"]
    assert len(t1["customers"]) == 150
    # every latent a plugin needs is in the receipt
    c0 = t1["customers"][0]
    for k in (
        "x0",
        "h0",
        "d0",
        "offer_opt",
        "channel_pref",
        "dept_pref",
        "item_pref",
        "price_sensitivity",
        "return_proneness",
        "segment",
        "touch_mix",
    ):
        assert k in c0


def _weekly_actions(df):
    d = (
        _json_path(df, "wk", r"$.week")
        .with_columns(
            [
                pl.col("wk").cast(pl.Int64),
                pl.col("event_attributes")
                .str.json_path_match(r"$.freq")
                .cast(pl.Float64)
                .alias("f"),
                pl.col("event_attributes")
                .str.json_path_match(r"$.offer")
                .cast(pl.Float64)
                .alias("o"),
                pl.col("event_attributes")
                .str.json_path_match(r"$.channel")
                .cast(pl.Float64)
                .alias("ch"),
                pl.col("event_attributes").str.json_path_match(r"$.target_key").alias("tk"),
            ]
        )
        .filter(pl.col("f").is_not_null() & (pl.col("event_type") != "purchase"))
    )
    return d


def test_reference_decision_action_optima_recoverable():
    """One-factor-daily design: condition on the randomized key (target_key) for each
    action, exactly as the served response block does. The randomized freq is peaked at
    x0 per customer; offer at offer_opt."""
    s = ReferenceSpec(n_customers=4000, window_days=540, seed=5)
    df, truth = build_reference(s)
    C = {c["customer"]: c for c in truth["customers"]}
    d = _weekly_actions(df)
    rew = (
        df.filter(pl.col("event_type") == "purchase")
        .pipe(_json_path, "wk", r"$.week")
        .filter(pl.col("wk").is_not_null())
        .with_columns(pl.col("wk").cast(pl.Int64))
        .group_by(["customer_key", "wk"])
        .agg(pl.lit(1.0).alias("y"))
    )
    j = (
        d.filter(pl.col("tk").is_in(["freq", "offer"]))
        .group_by(["customer_key", "wk", "tk"])
        .agg(pl.col("f").last(), pl.col("o").last())
        .join(rew.select(["customer_key", "wk", "y"]), on=["customer_key", "wk"], how="left")
        .fill_null(0.0)
    )
    # per-(customer, action) MEAN reward = the empirical dose-response, then argmax
    agg_f = (
        j.filter(pl.col("tk") == "freq")
        .group_by(["customer_key", "f"])
        .agg(pl.col("y").mean().alias("r"))
    )
    agg_o = (
        j.filter(pl.col("tk") == "offer")
        .group_by(["customer_key", "o"])
        .agg(pl.col("y").mean().alias("r"))
    )
    ok_f = tot_f = ok_o = tot_o = 0
    for c, sub in agg_f.group_by("customer_key"):
        c = c[0]
        if c not in C:
            continue
        ok_f += int(sub.sort("r", "f").row(-1)[1]) == int(min(max(round(C[c]["x0"]), 1), 3))
        tot_f += 1
    for c, sub in agg_o.group_by("customer_key"):
        c = c[0]
        if c not in C:
            continue
        ok_o += int(sub.sort("r", "o").row(-1)[1]) == int(C[c]["offer_opt"])
        tot_o += 1
    assert ok_f / tot_f > 0.35  # vs ~0.25 chance
    assert ok_o / tot_o > 0.28  # vs ~0.25 chance


def test_reference_category_affinity_recoverable():
    s = ReferenceSpec(n_customers=500, window_days=400, seed=7)
    df, truth = build_reference(s)
    C = {c["customer"]: c for c in truth["customers"]}
    pur = df.filter(pl.col("event_type") == "purchase").pipe(_json_path, "cat", r"$.category_code")
    agg = pur.filter(pl.col("cat") != "").group_by(["customer_key", "cat"]).agg(pl.len().alias("n"))
    ok = tot = 0
    for c, sub in agg.group_by("customer_key"):
        c = c[0]
        if c not in C:
            continue
        ok += sub.sort("n").row(-1)[1] == C[c]["dept_pref"]
        tot += 1
    assert ok / tot > 0.5  # vs ~1/6 chance


def test_reference_contacts_have_no_effect_before_treatment_day():
    """Clean OPE: contact response only exists AFTER treatment_day, so `E` at a pre-treatment
    as_of carries no treatment. Pre-treatment contact weeks must have ~no incremental conversion."""
    s = ReferenceSpec(n_customers=1500, window_days=540, seed=9, treatment_day=180)
    df, _ = build_reference(s)
    from datetime import datetime, timezone

    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    d = _json_path(df, "wk", r"$.week").with_columns(
        [
            pl.col("event_attributes").str.json_path_match(r"$.freq").cast(pl.Float64).alias("f"),
            pl.col("event_ts").str.to_datetime(time_zone="UTC").dt.epoch("s").alias("t"),
        ]
    )
    sends = d.filter(pl.col("f").is_not_null())
    pre = sends.filter(pl.col("t") <= t0 + 180 * 86400).filter(pl.col("f") >= 1)
    post = sends.filter(pl.col("t") > t0 + 180 * 86400).filter(pl.col("f") >= 1)
    assert pre.height > 0 and post.height > 0
    # contact-driven purchases (tagged with `week`) before treatment_day should be absent
    apur = d.filter(pl.col("event_type") == "purchase").filter(pl.col("wk").is_not_null())
    n_pre = apur.filter(pl.col("t") <= t0 + 180 * 86400).height
    n_post = apur.filter(pl.col("t") > t0 + 180 * 86400).height
    assert n_pre == 0
    assert n_post > 0
