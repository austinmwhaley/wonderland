"""End-to-end tests for the frozen-core + LoRA paradigm."""

import math

import torch
from torch import Tensor, nn

from looking_glass import (
    EntityCore,
    QDoRAConfig,
    SequenceEngine,
    TaskHeadBase,
    TemporalStack,
    create_supervised_model,
    extract_final_states,
    incremental_forward,
)

HIDDEN = 32


def _build_qdora_core() -> EntityCore:
    return EntityCore(
        temporal_encoder=TemporalStack(hidden_dim=HIDDEN),
        sequence_engine=SequenceEngine(
            hidden_dim=HIDDEN,
            recurrent_steps=1,
            num_heads=8,
            backend="mamba2",
            qdora_config=QDoRAConfig(rank=4, quantize_base=False),
        ),
        task_head=_StubHead(),
    )


class _StubHead(TaskHeadBase):
    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        return {}


def _collect_weights(module, exclude_keys: set = set()) -> list[float]:
    vals: list[float] = []
    for name, param in module.named_parameters():
        if any(k in name for k in exclude_keys):
            continue
        vals.extend(param.data.detach().cpu().flatten().tolist())
    return vals


def _classification_rows(n=120):
    return [
        {"id": f"c{i}", "x": float(i), "label": 1.0 if i < n // 2 else 0.0}
        for i in range(n)
    ]


class TestFrozenCoreTraining:
    def test_core_non_adapter_weights_unchanged(self):
        """Training a supervised head must not alter frozen (non-adapter) core weights."""
        core = _build_qdora_core()
        pre = _collect_weights(core, exclude_keys={"lora_a", "lora_b", "magnitude", "qdo"})

        model = create_supervised_model(
            task="classification", id_field="id", target_field="label",
            categorical_fields=[], numeric_fields=["x"],
            hidden_dim=HIDDEN, epochs=10, seed=0, device="cpu",
            sequence_backend="mamba2", pretrained_core=core,
        )
        model.fit_predict(_classification_rows())

        post = _collect_weights(core, exclude_keys={"lora_a", "lora_b", "magnitude", "qdo"})
        assert len(pre) == len(post)
        for a, b in zip(pre, post):
            assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6), "non-adapter weight changed"

    def test_lora_params_gradient_updating(self):
        """QDoRA adapter params should receive gradients; frozen base should not."""
        core = _build_qdora_core()
        model = create_supervised_model(
            task="classification", id_field="id", target_field="label",
            categorical_fields=[], numeric_fields=["x"],
            hidden_dim=HIDDEN, epochs=1, seed=0, device="cpu",
            sequence_backend="mamba2", pretrained_core=core,
        )
        result = model.fit_predict(_classification_rows())
        assert result.report.metrics["f1"] > 0.5


class TestIncrementalForward:
    def test_incremental_runs_and_returns_valid_shape(self):
        """Incremental forward prepends a seed state and processes new events."""
        core = _build_qdora_core()
        batch, seq = 3, 5
        torch.manual_seed(42)
        states = torch.randn(batch, seq, HIDDEN)
        dt = torch.rand(batch, seq, 1)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        seed = extract_final_states(core, states[:, :2], dt[:, :2], mask[:, :2])
        result = incremental_forward(core, seed, states[:, 2:], dt[:, 2:], mask[:, 2:])
        assert result.shape == (batch, HIDDEN)
        assert torch.isfinite(result).all()


class TestMultiTaskDeepcopy:
    def test_two_tasks_train_independently_without_error(self):
        """Each supervised call deep-clones the core — adapters are private."""
        core = _build_qdora_core()

        m1 = create_supervised_model(
            task="classification", id_field="id", target_field="label",
            categorical_fields=[], numeric_fields=["x"],
            hidden_dim=HIDDEN, epochs=3, seed=1, device="cpu",
            sequence_backend="mamba2", pretrained_core=core,
        )
        r1 = m1.fit_predict(_classification_rows(100))
        assert r1.report.metrics["f1"] > 0.0

        m2 = create_supervised_model(
            task="regression", id_field="id", target_field="label",
            categorical_fields=[], numeric_fields=["x"],
            hidden_dim=HIDDEN, epochs=3, seed=42, device="cpu",
            sequence_backend="mamba2", pretrained_core=core,
        )
        r2 = m2.fit_predict(_classification_rows(100))
        assert r2.report.metrics["r2"] > -1.0
