"""Generic attribute discovery/encoding + exogenous actions as input-only tokens.

No field name is hardcoded by the implementation: the canonical stream's JSON
keys are discovered, typed (numeric vs categorical) from a seeded sample, and
each gets a tokenizer channel. Action events stay INPUT tokens, but no loss
predicts them.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
import torch

from looking_glass.cfm_config import CFMConfig
from looking_glass.cfm_data import _read_stream, build_sequences
from looking_glass.cfm_model import CFM, EventVocab
from looking_glass.cfm_training import _collate, _target_valid_mask, _task_losses


def _canonical_stream():
    rows = []
    skus = [f"sku{i:03d}" for i in range(40)]
    for c in range(6):
        for i in range(12):
            is_action = i % 4 == 0
            attrs = (
                {"discount_pct": 0.05 * (i % 3)}
                if is_action
                else {
                    "value": 1.0 + (i % 7),
                    "amount": 2.0 + i,
                    "color": ["red", "blue"][i % 2],
                    "sku": skus[(c * 5 + i) % len(skus)],
                }
            )
            attrs["meta"] = {"nested": True}  # nested -> skipped in v1
            rows.append(
                {
                    "customer_key": f"c{c}",
                    "event_ts": f"2024-01-{i + 1:02d}T00:00:00+00:00",
                    "event_type": "send" if is_action else "buy",
                    "event_attributes": json.dumps(attrs, separators=(",", ":")),
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize("suffix", [".parquet", ".feather"])
def test_heterogeneous_attributes_all_encoded(tmp_path, suffix):
    df = _canonical_stream()
    path = tmp_path / f"stream{suffix}"
    if suffix == ".parquet":
        df.write_parquet(path)
    else:
        df.write_ipc(path)
    cfg = CFMConfig(
        db=str(path),
        action_fields=("discount_pct",),
        max_attr_vocab=4,
    )
    out = _read_stream(cfg)
    # every top-level key becomes a column; nested object fields are skipped
    assert {"amount", "value", "color", "sku", "discount_pct"} <= set(out.columns)
    assert {"amount", "value", "discount_pct"} <= set(cfg.numeric_fields)
    assert {"color", "sku"} <= set(cfg.attr_fields)
    assert "meta" in cfg._field_receipt["skipped_nested_fields"]
    assert cfg._field_receipt["rule"].startswith("numeric iff")
    assert cfg._field_receipt["fields"]["amount"]["kind"] == "numeric"

    keys = sorted(out["customer_key"].unique().to_list())
    split = {k: "A" for k in keys}
    seqs = build_sequences(out, keys, cfg, split, with_anchors=False, min_events=1)
    vocab = EventVocab.build(
        seqs,
        attr_fields=cfg.attr_fields,
        max_attr=cfg.max_attr_vocab,
        numeric_fields=cfg.numeric_fields,
    )
    # generic vocab: high-cardinality `sku` keeps a frequency-capped prefix
    assert 0 < vocab.n_attr["sku"] <= cfg.max_attr_vocab
    assert {"color", "sku"} <= set(vocab.attrs)
    assert {"amount", "value", "discount_pct"} <= set(vocab.numeric_stats)

    model = CFM(
        vocab,
        dim=16,
        n_experts=1,
        numeric_fields=vocab.numeric_fields,
        action_fields=cfg.action_fields,
    )
    for f in vocab.attrs:
        assert f in model.emb_attr
    assert model.head_attr is not None
    for f in vocab.numeric_fields:
        assert f in model.w_num
    assert model.head_num is not None

    t = _collate(seqs, vocab, torch.device("cpu"))
    # the declared action field is present as an INPUT (identity not dropped)
    assert bool((t["discount_pct"] != 0).any())
    assert bool(t["is_action"].any())
    y, h = model.encode(t)
    assert torch.isfinite(y).all()

    cfg.agg_horizons_days = [1.0, 7.0]
    T_ = _task_losses(model, vocab, seqs, cfg)
    for f in vocab.attrs:
        assert f"attr_next:{f}" in T_ and torch.isfinite(T_[f"attr_next:{f}"])
    for f in vocab.numeric_fields:
        assert f"num_next:{f}" in T_ and torch.isfinite(T_[f"num_next:{f}"])

    # action-handling receipt (#7): detected from data, with partial-presence flagged
    assert cfg._action_receipt["n_actions"] > 0
    assert cfg._action_receipt["action_fields"] == ["discount_pct"]

    # every valid target is a BEHAVIOUR event: no action target position is valid
    valid = _target_valid_mask(t, vocab)
    next_action = t["is_action"][:, 1:]
    assert not bool(((valid > 0) & (next_action > 0)).any())
    # pad targets stay masked
    next_pad = t["et"][:, 1:] == vocab.n_et
    assert not bool(((valid > 0) & next_pad).any())

    # gradients reach the per-field heads and (via tokens) the tokenizer
    loss = sum(v for k, v in T_.items() if k.startswith(("attr_next:", "num_next:")))
    loss.backward()
    assert model.head_attr.weight.grad is not None
    assert model.head_num.weight.grad is not None
    assert model.w_num["discount_pct"].weight.grad is not None


def _independent_seqs(n_cust=12, n_ev=24, seed=7):
    import numpy as np

    rng = np.random.default_rng(seed)
    seqs = []
    for c in range(n_cust):
        color = rng.integers(0, 3, n_ev)
        zone = rng.integers(0, 5, n_ev)
        amount = rng.normal(0.0, 1.0, n_ev)
        et = rng.integers(0, 4, n_ev)
        seqs.append(
            {
                "customer": f"c{c}",
                "event_type": [f"e{x}" for x in et],
                "attrs": {
                    "color": [f"c{x}" for x in color],
                    "zone": [f"z{x}" for x in zone],
                },
                "numeric": {"amount": amount.tolist()},
                "numeric_mask": {"amount": np.ones(n_ev, dtype=bool)},
                "value": amount.tolist(),
                "event_ts": [f"2024-01-{i + 1:02d}T00:00:00+00:00" for i in range(n_ev)],
                "ts": [1704067200.0 + i * 86400.0 for i in range(n_ev)],
                "co": [[0.0, 0.0]] * n_ev,
                "is_action": np.zeros(n_ev, dtype=bool),
            }
        )
    return seqs


def test_simple_probe_recovers_every_field():
    """#8 (Accessible): after removing the named-key path, each discovered field is
    still recoverable by a SIMPLE linear probe from the shared tokenizer — presence
    is not enough; a downstream consumer can actually read it."""
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression

    seqs = _independent_seqs()
    vocab = EventVocab.build(seqs)
    model = CFM(vocab, dim=32).eval()
    t = _collate(seqs, vocab, torch.device("cpu"))
    tok = model.tokens_batch(t).detach().numpy()
    mask = t["mask"].numpy().astype(bool)
    X = tok[mask]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2 :]

    for f in vocab.attrs:
        Y = t[f].numpy()[mask]
        clf = LogisticRegression(max_iter=500).fit(X[tr], Y[tr])
        acc = float((clf.predict(X[te]) == Y[te]).mean())
        chance = 1.0 / max(vocab.n_attr[f], 1)
        assert acc > chance + 0.25, f"{f}: probe acc {acc:.3f} vs chance {chance:.3f}"

    for f in vocab.numeric_fields:
        Y = t[f].numpy()[mask].astype(float)
        lin = LinearRegression().fit(X[tr], Y[tr])
        r2 = float(lin.score(X[te], Y[te]))
        assert r2 > 0.5, f"{f}: probe R2 {r2:.3f}"


def test_shared_prediction_heads_are_capacity_bounded():
    """#5 (Efficient): prediction capacity does NOT grow with schema breadth —
    one field-conditioned head, not one head per field."""
    import copy

    seqs = _independent_seqs()
    v2 = EventVocab.build(seqs)
    m2 = CFM(v2, dim=32)
    seqs10 = copy.deepcopy(seqs)
    for c in range(3, 12):
        for s in seqs10:
            s["attrs"][f"f{c}"] = s["attrs"]["color"]
    v10 = EventVocab.build(seqs10)
    m10 = CFM(v10, dim=32)
    assert len(v10.attrs) >= 10
    assert m10.head_attr.weight.numel() == m2.head_attr.weight.numel()
    assert m10.head_num.weight.numel() == m2.head_num.weight.numel()
