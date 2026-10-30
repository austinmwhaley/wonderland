"""PATH A provenance (DEC-446): the stamp is derived from construction, and construction cannot leak.

The audit's whole value is that "no feature reads the future" stops being a citation to one
filter and becomes a measurement. These tests grade the mechanism, not the string: the canary
is that perturbing ONLY post-`as_of` events leaves every feature byte-identical while the
labels move. A stamp that survives this test cannot be decoration.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rabbit_hole.generators.path_a_build import build_path_a, path_a_provenance  # noqa: E402

AS_OF = datetime(2025, 1, 1)
_ETYPES = ["purchase", "view", "return", "email_send", "search"]
_CATS = ["apparel", "footwear", "accessories", "beauty", "home", "outdoor"]


def _events(n_customers: int = 4, per_customer: int = 8) -> pl.DataFrame:
    """Deterministic stream: half of each customer's events before `AS_OF`, half after.

    Future purchases carry value=10.0 — the quantity the label canary moves on.
    """
    rows = []
    for c in range(n_customers):
        for e in range(per_customer):
            pre = e < per_customer // 2
            day = 1 + e * 9 if pre else 2 + (e - per_customer // 2) * 4  # post: Jan 2025
            month = 6 if pre else 1
            year = 2024 if pre else 2025
            et = _ETYPES[(c + e) % len(_ETYPES)]
            rows.append(
                {
                    "customer_key": f"c{c}",
                    "event_ts": f"{year}-{month:02d}-{day:02d} 12:00:00",
                    "event_type": et,
                    "value": 10.0 if (not pre and et == "purchase") else 1.0 * (e + 1),
                    "margin": 0.3 * (e + 1),
                    "product_id": f"p{(c + e) % 3}",
                    "channel": ["web", "app", "store"][(c + e) % 3],
                    "campaign_id": f"k{(c + e) % 2}",
                    "discount": 0.05 * (e % 4),
                    "qty": 1 + (e % 3),
                    "category_code": _CATS[(c + e) % len(_CATS)],
                }
            )
    return pl.DataFrame(rows)


def _customers(n: int = 4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_key": [f"c{i}" for i in range(n)],
            "signup_ts": ["2023-12-01 00:00:00"] * n,
            "segment": ["a", "b", "a", "b"][:n],
        }
    )


def _build(base=None):
    base = _events() if base is None else base
    return build_path_a(base, _customers(), AS_OF, engine=None)


def test_future_perturbation_moves_labels_but_never_features():
    """The canary. If a feature ever read the future, this test is where it surfaces."""
    f1, l1, _, _ = _build()
    # double the value of EVERY post-`as_of` purchase — the label's own source quantity
    base2 = _events().with_columns(
        pl.when((pl.col("event_ts") >= "2025-01-01") & (pl.col("event_type") == "purchase"))
        .then(pl.col("value") * 7.0)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    f2, l2, _, _ = _build(base2)
    assert f1.sort("customer_key").equals(f2.sort("customer_key")), (
        "a post-as_of perturbation moved a feature — the hist filter is no longer the only cut"
    )
    assert not l1.sort("customer_key").equals(l2.sort("customer_key")), (
        "the future perturbation did not reach the labels — the test built no future events"
    )


def test_stamp_covers_every_built_column():
    features, labels, _, state = _build()
    prov = state["provenance"]
    built = (set(features.columns) | set(labels.columns)) - {"customer_key"}
    assert built - set(prov) == set()
    assert {v["source"] for v in prov.values()} <= {"hist", "fut", "customers"}
    for c in labels.columns:
        if c != "customer_key":
            assert prov[c]["source"] == "fut", f"label {c} is not stamped fut"
    for c in features.columns:
        if c in ("customer_key", "segment", "signup_ts", "tenure_days"):
            continue  # static table attributes are stamped `customers`, not behavior
        assert prov[c]["source"] == "hist", f"feature {c} is not stamped hist"


def test_pure_helper_equals_the_builder_stamp():
    """The audit reads the pure map; the builder stamps its own copy — they may not drift."""
    _, _, _, state = _build()
    assert path_a_provenance(AS_OF, customer_cols=_customers().columns) == state["provenance"]


def test_unstamped_column_is_refused_not_stamped_clean(monkeypatch):
    """Principle 7: a column the builder cannot attribute to a cut is a hard error.

    `_collect` is patched to smuggle a column in OUTSIDE the expression lists — the only way
    a real future edit could create an unattributed column (the stamp itself is list-derived).
    """
    from rabbit_hole.generators import path_a_build as P

    orig = P._collect

    def smuggle(lf, engine):
        df = orig(lf, engine)
        if "n_events" in df.columns:  # only the per-customer (features) frame
            df = df.with_columns(pl.lit(1.0).alias("sneaky"))
        return df

    monkeypatch.setattr(P, "_collect", smuggle)
    with pytest.raises(ValueError, match="sneaky"):
        _build()


def test_comparator_drops_every_fut_column_it_is_given():
    """The audit's fourth fact, at unit scale: the drop rule from `table_features` itself
    removes exactly the fut-stamped columns and keeps only hist|customers."""
    from demo.auto_encoder import table_feature_columns

    _, _, wide, state = _build()
    prov = state["provenance"]
    comparator = set(table_feature_columns(wide))
    fut = {c for c, v in prov.items() if v["source"] == "fut"}
    assert not (comparator & fut), "the comparator reads a column the builder says is future"
    assert fut and fut <= set(wide.columns), "no fut columns were built — the fixture is wrong"
    assert all(prov[c]["source"] in ("hist", "customers") for c in comparator)
    assert "signup_ts" not in comparator and "customer_key" not in comparator
