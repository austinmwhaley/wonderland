"""DEC-659: the vectorised `_collate` must be byte-identical to the historical one.

The optimisation staged each field on the host and transferred once; it is only admissible
because it changes no value. This test keeps a faithful copy of the OLD implementation as the
oracle and compares every tensor of the batch dict, exactly (not approximately): the same
vocab fallbacks, pad fills, float32 roundings, dt/tod arithmetic, masks and cut points.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from looking_glass.cfm_training import _collate, _f, _tod, _to_epoch


class _V:
    """Minimal stand-in for EventVocab (only the attributes _collate reads)."""

    def __init__(self):
        self.n_et = 3
        self.et = {"a": 0, "b": 1}
        self.attrs = {"brand": {"x": 0}, "tier": {"gold": 1}}
        self.n_attr = {"brand": 2, "tier": 3}
        self.numeric_fields = ("price", "qty")


def _old_collate(seqs, vocab, device):
    """The historical implementation, verbatim in behaviour (the oracle)."""
    B = len(seqs)
    T = max(len(s["event_type"]) for s in seqs)
    et = torch.full((B, T), vocab.n_et, dtype=torch.long, device=device)
    attr_t = {
        f: torch.full((B, T), vocab.n_attr[f], dtype=torch.long, device=device)
        for f in (getattr(vocab, "attrs", {}) or {})
    }
    num_t = {
        c: torch.zeros(B, T, device=device) for c in (getattr(vocab, "numeric_fields", ()) or ())
    }
    num_mask_t = {
        c: torch.zeros(B, T, device=device) for c in (getattr(vocab, "numeric_fields", ()) or ())
    }
    is_action = torch.zeros(B, T, device=device)
    val = torch.zeros(B, T, 1, device=device)
    dt = torch.zeros(B, T, 1, device=device)
    tod = torch.zeros(B, T, 4, device=device)
    cov = torch.zeros(B, T, 2, device=device)
    mask = torch.zeros(B, T, device=device)
    cut = torch.ones(B, dtype=torch.long, device=device)
    for i, sq in enumerate(seqs):
        L = len(sq["event_type"])
        et[i, :L] = torch.tensor(
            [vocab.et.get(str(x), vocab.n_et) for x in sq["event_type"]], device=device
        )
        for f, ten in attr_t.items():
            vals = (sq.get("attrs") or {}).get(f)
            if vals is not None:
                ten[i, :L] = torch.tensor(
                    [
                        vocab.attrs[f].get(str(x) if x is not None else "none", vocab.n_attr[f])
                        for x in vals
                    ],
                    device=device,
                )
        for c, ten in num_t.items():
            vals = (sq.get("numeric") or {}).get(c)
            if vals is not None and len(vals) == L:
                ten[i, :L] = torch.tensor(
                    [float(v) for v in vals], dtype=torch.float32, device=device
                )
            nm = (sq.get("numeric_mask") or {}).get(c)
            if nm is not None and len(nm) == L:
                num_mask_t[c][i, :L] = torch.tensor(np.asarray(nm, dtype=np.float32), device=device)
        val[i, :L, 0] = torch.tensor(
            [_f(v) for v in sq.get("value", [0.0] * L)], dtype=torch.float32, device=device
        )
        _act = sq.get("is_action")
        if _act is not None:
            is_action[i, :L] = torch.tensor(np.asarray(_act, dtype=np.float32), device=device)
        ts = np.asarray(sq.get("ts") or [_to_epoch(x) for x in sq["event_ts"]], dtype=np.float64)
        d = np.zeros(L)
        d[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
        dt[i, :L, 0] = torch.tensor(np.log1p(d), dtype=torch.float32, device=device)
        tod[i, :L] = torch.tensor(_tod(ts), dtype=torch.float32, device=device)
        if sq.get("co"):
            cov[i, :L] = torch.tensor(sq["co"], dtype=torch.float32, device=device)
        mask[i, :L] = 1.0
        cut[i] = min(L - 1, max(1, int(0.6 * L)))
    return {
        "et": et,
        **attr_t,
        **num_t,
        "num_mask": num_mask_t,
        "is_action": is_action,
        "val": val,
        "dt": dt,
        "tod": tod,
        "co": cov,
        "mask": mask,
        "cut": cut,
    }


def _seq(n, t0=1_700_000_000, with_numeric=True, with_attrs=True):
    """A length-exact synthetic sequence (every field has exactly n entries)."""
    ts = [t0 + 3600 * i for i in range(n)]
    types = ["a", "b", "zzz"] * (n // 3 + 1)
    sq = {
        "event_type": types[:n],
        "event_ts": [str(int(x)) for x in ts],
        "ts": ts,
        "value": [0.1] * n,
        "is_action": [float(i % 2) for i in range(n)],
        "co": [[0.0, 1.0]] * n,
    }
    if with_attrs:
        # include the None and unknown paths, length-exact
        brand = ["x", None, "unknown"] * (n // 3 + 1)
        sq["attrs"] = {"brand": brand[:n], "tier": ["gold"] * n}
    if with_numeric:
        sq["numeric"] = {"price": [1.0] * n, "qty": [float(i) for i in range(n)]}
        sq["numeric_mask"] = {
            "price": [1.0] * n,
            "qty": [1.0] * max(n - 1, 0) + ([0.0] if n else []),
        }
    return sq


@pytest.mark.parametrize("lengths", [(1,), (3, 1), (5, 4, 2), (7, 7)])
def test_collate_is_bit_identical_to_the_historical_implementation(lengths):
    vocab = _V()
    seqs = [_seq(n) for n in lengths]
    got = _collate(seqs, vocab, torch.device("cpu"))
    want = _old_collate(seqs, vocab, torch.device("cpu"))
    assert set(got) == set(want), set(got) ^ set(want)
    for k in want:
        if k == "num_mask":
            assert set(got[k]) == set(want[k])
            for f in want[k]:
                assert torch.equal(got[k][f], want[k][f]), f"num_mask[{f}] differs"
            continue
        assert torch.equal(got[k], want[k]), f"{k} differs"


def test_collate_matches_without_optional_fields():
    vocab = _V()
    seqs = [
        _seq(4, with_numeric=False, with_attrs=False),
        _seq(2, with_numeric=False, with_attrs=False),
    ]
    got = _collate(seqs, vocab, torch.device("cpu"))
    want = _old_collate(seqs, vocab, torch.device("cpu"))
    for k in want:
        if k == "num_mask":
            for f in want[k]:
                assert torch.equal(got[k][f], want[k][f])
            continue
        assert torch.equal(got[k], want[k]), f"{k} differs"
