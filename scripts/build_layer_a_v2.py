#!/usr/bin/env python3
"""Build the Layer A v2 demo layout (stream + derivable PATH A + truth) in one command.

v2 is the stream whose forward VALUE depends on the current regime (DEC-522). This builder emits
the same layout the certification producers expect, so nothing downstream needs to change:

    <out>/path_b/path_b_customer_event_stream.parquet
    <out>/path_a/{customer_features,customer_labels,path_a_wide_table}.parquet
    <out>/reference_truth.json  <out>/relational_reference_truth.json  <out>/manifest.json

P8 by construction: the static customer columns (segment, loyalty_tier, signup_ts, region) are
read back OUT of the stream's own `customer_signup` events — PATH A cannot contain a fact the
stream does not carry.

    python3 scripts/build_layer_a_v2.py --out /tmp/opencode/demo_v2 --customers 5000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from layer_a_v2_prototype import build as build_v2  # noqa: E402
from rabbit_hole.generators.path_a_build import build_path_a  # noqa: E402


def _customers_from_stream(stream: Path) -> pl.DataFrame:
    """Static customer columns, derived from the stream's own signup events (P8)."""
    ev = pl.read_parquet(stream)
    signup = ev.filter(pl.col("event_type") == "customer_signup")
    rows = signup.select("customer_key", "event_attributes").to_dicts()
    decoded = [
        {
            "customer_key": r["customer_key"],
            "segment": json.loads(r["event_attributes"])["segment"],
            "loyalty_tier": json.loads(r["event_attributes"])["loyalty_tier"],
            "signup_ts": json.loads(r["event_attributes"])["signup_ts"],
            "region": json.loads(r["event_attributes"])["region"],
        }
        for r in rows
    ]
    return pl.DataFrame(decoded).unique(subset="customer_key", keep="first")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--customers", type=int, default=5000)
    ap.add_argument("--days", type=int, default=540)
    ap.add_argument("--as-of-day", type=int, default=365)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--engine", default="cpu")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    info = build_v2(out, a.customers, a.days, a.as_of_day, a.seed)
    stream = Path(info["stream"])
    print(f"[v2] {info['events']} events, {info['customers']} customers -> {stream}", flush=True)

    manifest = json.loads((out / "manifest.json").read_text())
    as_of = datetime.fromisoformat(manifest["as_of"])
    ev = pl.read_parquet(stream)

    def _js(field: str, dtype=pl.Utf8) -> pl.Expr:
        # polars 1.44: str.json_path_match returns Utf8 (or null when the path is absent)
        return (
            pl.col("event_attributes")
            .str.json_path_match(f"$.{field}")
            .cast(dtype, strict=False)
            .alias(field)
        )

    # PATH A's builders read these as top-level columns; v2 carries them in event_attributes,
    # so decode them WITHOUT dropping the canonical four-column stream that Layer B consumes.
    base = ev.with_columns(
        [
            _js("margin", pl.Float64),
            _js("qty", pl.Int64),
            _js("discount", pl.Float64),
            _js("product_id"),
            _js("category_code"),
            _js("channel"),
            _js("campaign_id"),
        ]
    )
    customers = _customers_from_stream(stream)
    print(f"[v2] customers table derived from the stream: {customers.shape}", flush=True)

    features, labels, wide, engine_state = build_path_a(base, customers, as_of, engine=a.engine)
    pa = out / "path_a"
    pa.mkdir(exist_ok=True)
    features.write_parquet(pa / "customer_features.parquet")
    labels.write_parquet(pa / "customer_labels.parquet")
    wide.write_parquet(pa / "path_a_wide_table.parquet")

    pb = out / "path_b"
    pb.mkdir(exist_ok=True)
    shutil.copy2(stream, pb / "path_b_customer_event_stream.parquet")

    # the response outlet reads the relational truth for actions; same schema as the truth
    truth = json.loads((out / "reference_truth.json").read_text())
    (out / "relational_reference_truth.json").write_text(json.dumps(truth, indent=1))

    manifest["paths"]["A"] = {
        "description": "traditional supervised wide table (derived from the v2 stream)",
        "features": "path_a/customer_features.parquet",
        "labels": "path_a/customer_labels.parquet",
        "wide": "path_a/path_a_wide_table.parquet",
        "rows": int(wide.height),
        "columns": int(wide.width),
        "engine": engine_state,
    }
    manifest["paths"]["B"]["stream"] = "path_b/path_b_customer_event_stream.parquet"
    manifest["truth"] = "reference_truth.json"
    manifest["counts"] = {"customers": int(customers.height), "events": int(ev.height)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(
        f"[v2] PATH A {wide.shape} (engine={engine_state['used']}) | "
        f"manifest + truth written to {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
