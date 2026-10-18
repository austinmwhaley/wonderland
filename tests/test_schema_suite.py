"""Degenerate-schema suite: the evidence behind T1.schema_universal.

"If T0 PASS, the encoder can take ANY stream" is only meaningful if the pipeline
runs (or rejects loudly) on schemas far from the happy path. Each case is a
canonical 4-column stream (customer_key, event_ts, event_type, event_attributes)
and must survive read -> fields -> sequences -> vocab -> model -> losses.

Cases: minimal (no attributes), all-categorical, single-event customers,
high-cardinality ids, mixed/dirty types, null-heavy numeric, nested objects,
and a multi-field stream.
"""

from __future__ import annotations

import json

import polars as pl
import torch

from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_data import _read_stream, build_sequences
from looking_glass.cfm_model import CFM, EventVocab
from looking_glass.cfm_training import _collate, _task_losses


def _stream(tmp_path, name, attr_fn, n_cust=4, n_ev=6):
    rows = []
    for c in range(n_cust):
        for i in range(n_ev):
            rows.append(
                {
                    "customer_key": f"c{c}",
                    "event_ts": f"2024-01-{i + 1:02d}T00:00:00+00:00",
                    "event_type": "buy" if i % 2 else "view",
                    "event_attributes": json.dumps(
                        attr_fn(c, i), separators=(",", ":"), default=str
                    ),
                }
            )
    path = tmp_path / f"{name}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


CASES = {
    "minimal": (lambda c, i: {}, {"attr_fields": (), "numeric_fields": ()}),
    "all_categorical": (
        lambda c, i: {"color": ["red", "blue"][i % 2], "zone": f"z{i % 3}"},
        {"attr_fields": ("color", "zone"), "numeric_fields": ()},
    ),
    "high_cardinality": (
        lambda c, i: {"sku": f"s{c}_{i}"},
        {"attr_fields": ("sku",), "numeric_fields": ()},
    ),
    "mixed_dirty": (
        lambda c, i: {"code": ("1.5" if i % 2 else "abc")},
        {"attr_fields": ("code",), "numeric_fields": ()},
    ),
    "null_heavy": (
        lambda c, i: {"amount": 1.0 + i} if i % 3 == 0 else {"amount": None},
        {"attr_fields": (), "numeric_fields": ("amount",)},
    ),
    "nested_only": (
        lambda c, i: {"meta": {"nested": i, "ok": True}},
        {"attr_fields": (), "numeric_fields": ()},
    ),
    "multi_field": (
        lambda c, i: {
            "color": ["red", "blue"][i % 2],
            "zone": f"z{i % 3}",
            "amount": 2.0 + i,
            "value": 1.0 + (i % 4),
        },
        {"attr_fields": ("color", "zone"), "numeric_fields": ("amount", "value")},
    ),
}


def _run(tmp_path, name, attr_fn, expect):
    path = _stream(tmp_path, name, attr_fn)
    cfg = CFMConfig(db=str(path), max_attr_vocab=0)
    df = _read_stream(cfg)
    assert set(expect["attr_fields"]) <= set(cfg.attr_fields), name
    assert set(expect["numeric_fields"]) <= set(cfg.numeric_fields), name
    if name == "nested_only":
        assert "meta" in cfg._field_receipt["skipped_nested_fields"]
    if name == "mixed_dirty":
        assert cfg._field_receipt["fields"]["code"]["dirty"] is True

    keys = sorted(df["customer_key"].unique().to_list())
    split = {k: "A" for k in keys}
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
    assert seqs, f"{name}: no sequences"
    vocab = EventVocab.build(
        seqs,
        attr_fields=cfg.attr_fields,
        max_attr=cfg.max_attr_vocab,
        numeric_fields=cfg.numeric_fields,
        primary_numeric=cfg.primary_numeric,
    )
    model = CFM(vocab, dim=16, n_experts=1)
    t = _collate(seqs, vocab, torch.device("cpu"))
    with torch.no_grad():
        y, h = model.encode(t)
    assert torch.isfinite(y).all(), f"{name}: non-finite readout"
    cfg.agg_horizons_days = [1.0]
    T_ = _task_losses(model, vocab, seqs, cfg)
    assert T_, f"{name}: no task losses"
    for k, v in T_.items():
        assert torch.isfinite(v).all(), f"{name}: non-finite loss {k}"
    # axis receipts exist for every stream
    assert cfg._field_receipt["rule"].startswith("numeric iff")
    assert "n_events" in cfg._action_receipt


def test_degenerate_schema_suite(tmp_path):
    for name, (attr_fn, expect) in CASES.items():
        _run(tmp_path, name, attr_fn, expect)


def test_single_event_customers(tmp_path):
    path = _stream(
        tmp_path, "single_event", lambda c, i: {"color": "red", "amount": 1.0}, n_cust=3, n_ev=1
    )
    cfg = CFMConfig(db=str(path))
    df = _read_stream(cfg)
    keys = sorted(df["customer_key"].unique().to_list())
    split = {k: "A" for k in keys}
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
    assert len(seqs) == 3
    vocab = EventVocab.build(
        seqs,
        attr_fields=cfg.attr_fields,
        max_attr=cfg.max_attr_vocab,
        numeric_fields=cfg.numeric_fields,
        primary_numeric=cfg.primary_numeric,
    )
    model = CFM(vocab, dim=16)
    t = _collate(seqs, vocab, torch.device("cpu"))
    with torch.no_grad():
        y, _ = model.encode(t)
    assert torch.isfinite(y).all()
