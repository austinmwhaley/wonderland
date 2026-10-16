"""Tests for the rabbit_hole APEX composite synthetic stream (rabbit_hole/generators/apex.py).

The apex composes every Layer-A lab dimension into one controlled fixture; these tests pin
that (a) it is deterministic, (b) it carries the KNOWN truth for the action response and the
randomized-arm treatment, and (c) the truth receipt exposes the declared action keys.
"""

from __future__ import annotations


import numpy as np
import polars as pl

from rabbit_hole.generators.apex import ApexSpec, build_apex


def _frames(spec):
    df, truth = build_apex(spec)
    return df, truth


def test_apex_deterministic_and_schema():
    s = ApexSpec(n_customers=120, window_days=180, seed=3)
    df1, t1 = _frames(s)
    df2, _ = _frames(s)
    assert df1.equals(df2)  # deterministic for a seed
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
    # multi-domain + randomized structure present
    et = set(df1["event_type"].unique().to_list())
    assert {"view", "cart", "purchase", "email_send"}.issubset(et)
    assert t1["meta"]["action_keys"] == ["freq", "hour"]
    assert t1["meta"]["arm_key"] == "arm"
    assert len(t1["customers"]) == 120


def test_apex_action_response_is_recoverable():
    """The randomized weekly freq has a known per-customer optimum x0 (truth)."""
    s = ApexSpec(n_customers=400, window_days=360, seed=5)
    df, truth = _frames(s)
    cust = {c["customer"]: c for c in truth["customers"]}
    d = df.with_columns(
        [
            pl.col("event_attributes").str.json_path_match(r"$.week").cast(pl.Int64).alias("wk"),
            pl.col("event_attributes").str.json_path_match(r"$.freq").cast(pl.Float64).alias("f"),
        ]
    )
    snd = (
        d.filter(pl.col("f").is_not_null()).group_by(["customer_key", "wk"]).agg(pl.col("f").last())
    )
    rew = (
        d.filter(pl.col("event_type") == "purchase")
        .group_by(["customer_key", "wk"])
        .agg(pl.lit(1.0).alias("y"))
    )
    j = snd.join(rew, on=["customer_key", "wk"], how="left").fill_null(0.0)
    agg = j.group_by(["customer_key", "f"]).agg(pl.col("y").mean().alias("r"))
    ok = tot = 0
    for c, sub in agg.group_by("customer_key"):
        c = c[0]
        if c not in cust:
            continue
        amax = int(sub.sort("r", "f").row(-1)[1])
        # the optimal OBSERVED action (freq 0 is the no-send control -> nearest is 1)
        orc = int(min(max(round(cust[c]["x0"]), 1), 3))
        tot += 1
        ok += amax == orc
    assert ok / tot > 0.55  # well above the ~1/3 chance


def test_apex_arm_treatment_only_after_treatment_day():
    """The randomized arm must NOT affect pre-treatment-day margin (clean OPE: E is
    pre-treatment), and MUST lift post-treatment margin toward the optimal arm."""
    from datetime import datetime, timezone

    s = ApexSpec(n_customers=800, window_days=540, seed=7, treatment_day=365)
    df, truth = _frames(s)
    cust = {c["customer"]: c for c in truth["customers"]}
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    d = df.with_columns(
        [
            pl.col("event_attributes").str.json_path_match(r"$.arm").cast(pl.Int64).alias("arm"),
            pl.col("value").cast(pl.Float64).alias("v"),
            pl.col("event_ts").str.to_datetime(time_zone="UTC").dt.epoch("s").alias("t"),
        ]
    ).filter(pl.col("v") > 0)
    pre = d.filter(pl.col("t") <= t0 + 365 * 86400)
    assert pre.height > 0  # pre-treatment purchases exist
    post = d.filter(pl.col("t") > t0 + 365 * 86400)
    pref = np.array([cust[c]["arm_pref"] for c in post["customer_key"].to_list()])
    arm = post["arm"].to_numpy()
    vv = post["v"].to_numpy()
    # post-treatment: the optimal arm yields a HIGHER margin than a mismatched arm
    assert vv[arm == pref].mean() > vv[arm != pref].mean()
    # pre-treatment: margin is ~arm-invariant (the history E sees carries no treatment)
    parm = pre["arm"].to_numpy()
    pvv = pre["v"].to_numpy()
    pp = np.array([cust[c]["arm_pref"] for c in pre["customer_key"].to_list()])
    if (parm == pp).any() and (parm != pp).any():
        ratio = pvv[parm == pp].mean() / max(pvv[parm != pp].mean(), 1e-9)
        assert ratio < 1.15  # no leakage pre-treatment
