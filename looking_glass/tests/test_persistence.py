"""Persistence roundtrip tests for EmbeddingModel and SupervisedModel."""

import tempfile
from pathlib import Path

import torch

from looking_glass import (
    EmbeddingModelConfig,
    EmbeddingModel,
    SupervisedModel,
    SupervisedModelConfig,
)


def _dummy_records(n=20):
    return [
        {"pid": f"p{i%4}", "cat": f"c{i%3}", "price": float(i * 10)}
        for i in range(n)
    ]


def test_embedding_save_load_transform():
    torch.manual_seed(42)
    records = _dummy_records(20)
    emb = EmbeddingModel(
        id_field="pid", categorical_fields=["cat"], numeric_fields=["price"],
        config=EmbeddingModelConfig(hidden_dim=16, epochs=3, seed=42, device="cpu",
                                    sequence_backend="mamba2"),
    )
    orig = emb.fit_transform(records)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "emb.pt"
        emb.save_pretrained(path)
        loaded = EmbeddingModel.load_pretrained(path, device="cpu")

    restored = loaded.transform(records)
    assert len(restored) == len(orig)
    for pid in orig.vectors:
        orig_vec = torch.tensor(orig.vectors[pid], dtype=torch.float32)
        rest_vec = torch.tensor(restored.vectors[pid], dtype=torch.float32)
        assert torch.allclose(orig_vec, rest_vec, atol=1e-5), f"mismatch for {pid}"


def _classification_records(n=40):
    return [
        {"cid": f"c{i}", "x": float(i), "label": 1.0 if i < n // 2 else 0.0}
        for i in range(n)
    ]


def test_supervised_classification_save_load_predict():
    torch.manual_seed(42)
    records = _classification_records(40)
    model = SupervisedModel(
        task="classification", id_field="cid", target_field="label",
        categorical_fields=[], numeric_fields=["x"],
        config=SupervisedModelConfig(hidden_dim=16, epochs=5, seed=42, device="cpu",
                                      sequence_backend="mamba2", validation_fraction=0.25),
    )
    result = model.fit_predict(records)
    orig_preds = result.predictions

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cls.pt"
        model.save_pretrained(path)
        loaded = SupervisedModel.load_pretrained(path, device="cpu")

    new_preds = loaded.predict(records)
    assert set(new_preds.keys()) == set(orig_preds.keys())

    for cid in orig_preds:
        o = orig_preds[cid]
        n = new_preds[cid]
        assert abs(o - n) < 0.05, f"predictions diverged for {cid}: {o} vs {n}"

    assert 0.5 <= result.report.metrics["roc_auc"] <= 1.0  # AUC in report
    assert 0.0 <= result.report.metrics["pr_auc"] <= 1.0   # PR-AUC in report


def test_supervised_regression_save_load_predict():
    torch.manual_seed(42)
    records = [
        {"cid": f"c{i}", "x": float(i), "label": 2.0 * i}
        for i in range(20)
    ]
    model = SupervisedModel(
        task="regression", id_field="cid", target_field="label",
        categorical_fields=[], numeric_fields=["x"],
        config=SupervisedModelConfig(hidden_dim=16, epochs=5, seed=42, device="cpu",
                                      sequence_backend="mamba2", validation_fraction=0.25),
    )
    result = model.fit_predict(records)
    assert "r2" in result.report.metrics

    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "reg.pt"
        model.save_pretrained(reg_path)
        loaded = SupervisedModel.load_pretrained(reg_path, device="cpu")
    new_preds = loaded.predict(records)
    assert len(new_preds) == len(result.predictions)


def test_predict_before_fit_raises():
    model = SupervisedModel("classification", "cid", "label", [], [])
    import pytest
    with pytest.raises(RuntimeError):
        model.predict([])
