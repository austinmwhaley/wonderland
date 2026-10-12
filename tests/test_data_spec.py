from __future__ import annotations

import math

from rabbit_hole.data_spec import DataSpec, simulate, transition_oracle, write_fixture


def test_snr_dial_null_vs_full():
    # snr=0 => the transition is the uniform marginal: no structure beyond the 1-gram.
    null = DataSpec("null", n_customers=200, transition=True, intent=0.6, snr=0.0)
    o0 = transition_oracle(null)
    assert abs(o0["incremental"]) < 1e-6
    assert abs(o0["baseline_ce"] - math.log(3)) < 1e-3
    # snr=1 with per-customer intent => real incremental structure beyond the 1-gram.
    full = DataSpec("full", n_customers=200, transition=True, intent=0.6, snr=1.0)
    o1 = transition_oracle(full)
    assert o1["incremental"] > 0.01


def test_simulate_canonical_schema():
    df, truth, _ = simulate(DataSpec("s", n_customers=100, transition=True, snr=1.0))
    assert list(df.columns) == [
        "customer_key",
        "event_ts",
        "event_type",
        "source_table",
        "event_attributes",
    ]
    assert df.height > 0
    assert truth["name"] == "s"


def test_write_fixture_emits_pieces(tmp_path):
    d = write_fixture(DataSpec("uc", n_customers=100, transition=True, snr=1.0), root=str(tmp_path))
    assert (d / "customer_event_stream.parquet").exists()
    assert (d / "uc_truth.json").exists()
    assert (d / "oracle.json").exists()
    assert (d / "baseline.json").exists()
