"""Layer-A onboarding: arbitrary raw tables -> canonical stream + metadata receipt."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from looking_glass.onboard import CANONICAL_COLUMNS, onboard


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def test_onboard_raw_schema(tmp_path):
    raw = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u2"],
            "ts": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "2024-01-04T00:00:00Z",
            ],
            "evt": ["send", "purchase", "send", "purchase", "view"],
            "amount": [0.0, 12.5, 0.0, 7.0, 0.0],
            "brand_name": ["acme", "acme", "beta", "beta", "beta"],
            "attrs": [
                '{"frequency":1,"arm":0,"campaign":"c1"}',
                '{"order_id":"o1"}',
                '{"frequency":2,"arm":1,"campaign":"c2"}',
                '{"order_id":"o2"}',
                '{"page":"p"}',
            ],
        }
    )
    source = tmp_path / "raw_events.parquet"
    raw.write_parquet(source)

    out_dir = tmp_path / "onboarded"
    receipt = onboard(source, out_dir=out_dir)

    canonical_path = out_dir / "customer_event_stream.parquet"
    meta_path = out_dir / "onboard_meta.json"
    assert canonical_path.exists() and meta_path.exists()
    canonical = _read(canonical_path)
    assert canonical.columns == list(CANONICAL_COLUMNS)
    assert canonical.height == 5
    assert canonical["customer_key"].n_unique() == 2

    meta = json.loads(meta_path.read_text())
    assert meta == receipt
    assert meta["n_rows"] == 5 and meta["n_customers"] == 2
    assert meta["columns_mapped"]["customer_key"] == "user_id"
    assert meta["columns_mapped"]["event_ts"] == "ts"
    assert meta["columns_mapped"]["event_type"] == "evt"
    assert meta["columns_mapped"]["value"] == "amount"
    assert meta["columns_mapped"]["brand"] == "brand_name"
    assert meta["columns_mapped"]["event_attributes"] == ["attrs"]
    assert meta["value_col"] == "amount"
    assert meta["reward_event"] == "purchase"
    assert meta["action_keys"] == ["arm", "frequency"]
    assert meta["cadence"] == "week"

    first = canonical.row(0, named=True)
    assert first["customer_key"] == "u1"
    assert first["event_ts"] == "2024-01-01T00:00:00+00:00"
    assert first["event_type"] == "send"
    assert first["brand"] == "acme"
    attrs = json.loads(first["event_attributes"])
    assert attrs["frequency"] == 1 and attrs["arm"] == 0 and attrs["campaign"] == "c1"

    purchase = canonical.filter(pl.col("event_type") == "purchase")
    assert purchase["value"].sum() == 19.5
    assert purchase["entity_id"].to_list() == ["", ""]


def test_onboard_canonical_is_near_identity(tmp_path):
    raw = pl.DataFrame(
        {
            "customer_key": ["c1", "c1"],
            "event_ts": ["2024-02-01T00:00:00+00:00", "2024-02-02T00:00:00+00:00"],
            "event_type": ["send", "purchase"],
            "brand": ["acme", "acme"],
            "entity_type": ["email", "order"],
            "entity_id": ["e1", "o1"],
            "source_table": ["s", "s"],
            "value": [0.0, 9.0],
            "event_attributes": ['{"frequency":1}', "{}"],
        }
    )
    out_dir = tmp_path / "canonical"
    receipt = onboard(raw, out_dir=out_dir)

    canonical = _read(out_dir / "customer_event_stream.parquet")
    assert canonical.columns == list(CANONICAL_COLUMNS)
    assert canonical.equals(raw)
    assert receipt["columns_mapped"] == {
        "customer_key": "customer_key",
        "event_ts": "event_ts",
        "event_type": "event_type",
        "brand": "brand",
        "entity_type": "entity_type",
        "entity_id": "entity_id",
        "source_table": "source_table",
        "value": "value",
        "event_attributes": ["event_attributes"],
    }
    assert receipt["reward_event"] == "purchase"


def test_onboard_drops_null_customer_and_time(tmp_path):
    raw = pl.DataFrame(
        {
            "user_id": [None, "u1", "u2", "u3"],
            "ts": [
                "2024-03-01T00:00:00Z",
                None,
                "2024-03-01T00:00:00Z",
                "2024-03-02T00:00:00Z",
            ],
            "evt": ["send", "send", "send", "view"],
        }
    )
    out_dir = tmp_path / "dropped"
    receipt = onboard(raw, out_dir=out_dir)

    canonical = _read(out_dir / "customer_event_stream.parquet")
    assert canonical.height == 2
    assert canonical["customer_key"].to_list() == ["u2", "u3"]
    assert receipt["n_rows"] == 2 and receipt["n_customers"] == 2
    assert any("dropped 2 rows" in w for w in receipt["warnings"])
    assert json.loads((out_dir / "onboard_meta.json").read_text()) == receipt
