"""DEC-473: the ledger's recency block is exact, history-only, and cut at the encoder's as_of."""

import numpy as np

from looking_glass.measure_state import MeasureBuilder


def _tuples(ts):
    n = len(ts)
    et = np.zeros(n, dtype=np.int64)
    empty_cat = np.zeros((n, 0), dtype=np.int64)
    empty_num = np.zeros((n, 0), dtype=np.float64)
    return (et, empty_cat, empty_num, np.asarray(ts, np.float64), None, None, empty_num)


def test_recency_block_is_the_exact_history_facts():
    day = 86400.0
    as_of = 100 * day
    seqs = {
        "a": _tuples([90 * day, 95 * day, 99 * day]),  # silence 1d, tenure 10d, span 9d, n=3
        "b": _tuples([60 * day]),  # silence 40d, tenure 40d, span 0, n=1
    }
    vs = {"et": 1, "gcat": [], "gnum": 0}
    b = MeasureBuilder(seqs, ["a", "b"], vs, coverage=0.99)
    b.as_of = as_of
    m = b.transform(seqs, ["a", "b"])
    lo, hi = b.part_slices()["recency"][0]
    rec = m[:, lo:hi]
    assert rec.shape == (2, 4)
    # silence, tenure, span, log1p(count)
    assert np.allclose(rec[0], [np.log1p(1), np.log1p(10), np.log1p(9), np.log1p(3)])
    assert np.allclose(rec[1], [np.log1p(40), np.log1p(40), 0.0, np.log1p(1)])
    assert m.shape[1] == b.dim  # the layout contract holds with the new part


def test_recency_spec_roundtrips_and_old_specs_stay_valid():
    day = 86400.0
    seqs = {"a": _tuples([day])}
    vs = {"et": 1, "gcat": [], "gnum": 0}
    b = MeasureBuilder(seqs, ["a"], vs, coverage=0.99)
    from looking_glass.measure_state import measure_from_dict, measure_to_dict

    d = measure_to_dict(b)
    assert d["n_recency"] == 4
    b2 = measure_from_dict(vs, d)
    assert b2.dim == b.dim
    # a pre-DEC-473 persisted spec (no n_recency) reconstructs with the old width
    old = {k: v for k, v in d.items() if k != "n_recency"}
    b3 = measure_from_dict(vs, old)
    assert b3.dim == b.dim - 4
