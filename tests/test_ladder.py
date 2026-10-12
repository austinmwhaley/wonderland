from __future__ import annotations

import numpy as np

from rabbit_hole.generators.ladder import RUNGS, build_rung


def _next_gain(df):
    """CE(unigram) - CE(prev-event) on the next-event task (structure detector)."""
    df = df.sort(["customer_key", "event_ts"])
    e = df["event_type"].to_numpy()
    k = df["customer_key"].to_numpy()
    m = k[:-1] == k[1:]
    prev, nxt = e[:-1][m], e[1:][m]
    types = sorted(set(e))
    tmap = {t: i for i, t in enumerate(types)}
    pi = np.array([tmap[x] for x in prev])
    ni = np.array([tmap[x] for x in nxt])
    marg = np.bincount(ni, minlength=len(types)) / len(ni)
    ce_u = -np.mean(np.log(marg[ni] + 1e-12))
    tab = np.zeros((len(types), len(types)))
    np.add.at(tab, (pi, ni), 1)
    ce_g = -np.mean(np.log(tab[pi, ni] / np.maximum(tab.sum(1)[pi], 1) + 1e-12))
    return ce_u - ce_g


def test_ladder_canonical_schema_and_determinism():
    df, truth, _ = build_rung("L2_markov", n_customers=200, window_days=365, seed=17)
    assert list(df.columns) == [
        "customer_key",
        "event_ts",
        "event_type",
        "source_table",
        "event_attributes",
    ]
    assert df.height > 0
    df2, _, _ = build_rung("L2_markov", n_customers=200, window_days=365, seed=17)
    assert df.equals(df2)  # deterministic for a given seed
    assert truth["dimensions"]["markov"] is True


def test_ladder_null_has_no_structure_and_markov_does():
    l0, t0, _ = build_rung("L0_null", n_customers=500, window_days=365, seed=7)
    l2, t2, _ = build_rung("L2_markov", n_customers=500, window_days=365, seed=7)
    g0, g2 = _next_gain(l0), _next_gain(l2)
    # L0 is the NEGATIVE CONTROL: previous event carries no info about the next.
    assert abs(g0) < 0.02
    # L2 injects a real order-1 transition matrix: prev-event greatly helps.
    assert g2 > 0.05
    assert t0["dimensions"]["markov"] is False and t2["dimensions"]["markov"] is True


def test_ladder_all_rungs_build():
    for rung in RUNGS:
        df, truth, _ = build_rung(rung, n_customers=50, window_days=180, seed=3)
        assert df.height > 0
        assert truth["rung"] == rung


def test_capability_card_null_and_order1_are_not_carried():
    """The §20.4 ceiling: an i.i.d. stream (L0) and an order-1 Markov stream (L2)
    carry NO structure beyond the 1-gram, so the transition capability is
    NO_SIGNAL there (no model may be required to beat an optimal 1-gram)."""
    from looking_glass.capability_card import transition_carried

    l0, _, _ = build_rung("L0_null", n_customers=1500, window_days=365, seed=5)
    l2, _, _ = build_rung("L2_markov", n_customers=1500, window_days=365, seed=5)

    # build rows with the fields the card reads (et, co)
    def rows(df):
        out = []
        for (key,) in df.select("customer_key").unique().iter_rows():
            sub = df.filter(df["customer_key"] == key).sort("event_ts")
            et = [["view", "cart", "purchase"].index(x) for x in sub["event_type"].to_list()]
            out.append({"et": et, "co": [0.0] * len(et)})
        return out

    c0, _ = transition_carried(rows(l0), 3)
    c2, _ = transition_carried(rows(l2), 3)
    assert c0 is False
    assert c2 is False
