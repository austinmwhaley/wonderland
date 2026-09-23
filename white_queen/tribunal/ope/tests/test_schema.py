"""Declared source-schema adapter tests."""
import numpy as np

from white_queen.tribunal.ope import schema as S


def _colony_cols(n=8):
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(4):
        cols[f"o{i}"] = rng.normal(size=n).astype(np.float32)
        cols[f"n{i}"] = rng.normal(size=n).astype(np.float32)
    cols["action"] = rng.integers(0, 2, n)
    cols["reward"] = rng.normal(size=n).astype(np.float32)
    cols["done"] = np.zeros(n, dtype=np.float32)
    cols["greedy_action"] = rng.integers(0, 2, n)
    cols["eps"] = np.full(n, 0.2, dtype=np.float32)
    cols["prob_taken"] = np.full(n, 0.8, dtype=np.float32)
    return cols


def test_colony_adapter_declares_roles():
    roles = S.detect_schema(_colony_cols())
    assert roles is not None
    assert roles["context"] == ["o0", "o1", "o2", "o3"]
    assert roles["next_context"] == ["n0", "n1", "n2", "n3"]
    assert "greedy_action" in roles["exclude"]


def test_explicit_columns_suppress_detection():
    assert S.detect_schema(_colony_cols(), explicit=["o0", "o1"]) is None


def test_unknown_layout_returns_none():
    cols = {"a": np.zeros(3), "b": np.zeros(3)}
    assert S.detect_schema(cols) is None


def test_register_custom_schema():
    ad = S.SchemaAdapter("toy", lambda c: "f0" in c,
                         lambda c: ["f0", "f1"], None, exclude=["meta"])
    S.register_schema(ad, overwrite=True)
    roles = S.detect_schema({"f0": [1], "f1": [2], "meta": [3]})
    assert roles["context"] == ["f0", "f1"]
