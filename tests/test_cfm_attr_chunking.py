"""No-op proof for the row-chunked per-field attr CE draw (DEC-622).

The unbounded `head_attr` logits draw (rows x max-vocab) OOMs at ~646K-class
fields — the DEC-615 wall. The chunked draw must produce the SAME loss values as
the unbounded path (up to float reduction order: chunked sums vs one full sum),
and the gradients must still reach the head. Both are pinned here.
"""

from __future__ import annotations

import json

import polars as pl
import torch
import torch.nn.functional as F

from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_data import _read_stream, build_sequences
from looking_glass.cfm_model import CFM, EventVocab
from looking_glass.cfm_training import _collate, _target_valid_mask, _task_losses


def _high_cardinality_stream():
    """40 customers x 40 events, one categorical field with 60 unique values.

    rows = 40 * 39 = 1560 > 1024 (the CPU chunk bound), so the chunk loop engages.
    """
    from datetime import datetime, timedelta, timezone

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    skus = [f"sku{i:03d}" for i in range(60)]
    for c in range(40):
        for i in range(40):
            attrs = {
                "value": 1.0 + (i % 7),
                "sku": skus[(c * 5 + i) % len(skus)],
            }
            ts = base + timedelta(days=i)
            rows.append(
                {
                    "customer_key": f"c{c}",
                    "event_ts": ts.isoformat(),
                    "event_type": "buy",
                    "event_attributes": json.dumps(attrs, separators=(",", ":")),
                }
            )
    return pl.DataFrame(rows)


def _build(tmp_path):
    path = tmp_path / "stream.parquet"
    _high_cardinality_stream().write_parquet(path)
    cfg = CFMConfig(db=str(path), max_attr_vocab=64)
    df = _read_stream(cfg)
    keys = sorted(df["customer_key"].unique().to_list())
    split = {k: "A" for k in keys}
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
    vocab = EventVocab.build(seqs, attr_fields=cfg.attr_fields, max_attr=cfg.max_attr_vocab)
    model = CFM(vocab, dim=16, n_experts=1)
    cfg.agg_horizons_days = [1.0, 7.0]
    return model, vocab, seqs, cfg, df


def test_chunked_attr_loss_matches_unbounded_reference(tmp_path):
    model, vocab, seqs, cfg, df = _build(tmp_path)
    assert model.head_attr is not None
    assert cfg.attr_fields  # the field is present

    T_, _ = _task_losses(model, vocab, seqs, cfg)

    # reference: the unbounded path, reimplemented inline (the pre-DEC-622 code)
    t = _collate(seqs, vocab, torch.device("cpu"))
    y, _ = model.encode(t)
    valid_t = _target_valid_mask(t, vocab)
    valid = valid_t.reshape(-1)
    nv = valid.sum().clamp(min=1)
    for f in vocab.attrs:
        j = model._attr_index[f]
        logits = model.head_attr(y[:, :-1] + model.attr_field_emb.weight[j])
        ref = (
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), t[f][:, 1:].reshape(-1), reduction="none"
            )
            * valid
        ).sum() / nv
        assert torch.allclose(T_[f"attr_next:{f}"], ref, atol=1e-5, rtol=1e-5)


def test_chunked_attr_gradients_reach_the_head(tmp_path):
    model, vocab, seqs, cfg, df = _build(tmp_path)
    T_, _ = _task_losses(model, vocab, seqs, cfg)
    loss = sum(v for k, v in T_.items() if k.startswith("attr_next:"))
    loss.backward()
    assert model.head_attr.weight.grad is not None
    assert torch.isfinite(model.head_attr.weight.grad).all()
    assert float(model.head_attr.weight.grad.abs().sum()) > 0.0
