"""Test core checkpoint roundtrips and the persistence methods on each model class."""

import tempfile
from pathlib import Path

import torch

from looking_glass import (
    EntityCore,
    QDoRAConfig,
    SequenceEngine,
    TemporalCoreConfig,
    TemporalCoreModel,
    TemporalStack,
    derive_backbone_version,
    load_core_checkpoint,
    save_core_checkpoint,
    wrap_core_with_qdora,
)
from looking_glass.interfaces import TaskHeadBase

HIDDEN = 16


class _NoopHead(TaskHeadBase):
    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


def _build_core(backend="mamba2"):
    return EntityCore(
        temporal_encoder=TemporalStack(hidden_dim=HIDDEN),
        sequence_engine=SequenceEngine(
            hidden_dim=HIDDEN,
            recurrent_steps=2,
            num_heads=4,
            backend=backend,
            qdora_config=QDoRAConfig(rank=4, quantize_base=False),
        ),
        task_head=_NoopHead(),
    )


def test_core_checkpoint_roundtrip():
    core = _build_core()
    core.eval()
    x = torch.randn(2, 3, HIDDEN)
    dt = torch.randn(2, 3, 1).abs()
    with torch.no_grad():
        out_before = core(x, dt).encoded_states

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.pt"
        version = save_core_checkpoint(core, path, backbone_version="test1")
        assert version == "test1"
        loaded_core, ver, extra = load_core_checkpoint(path)
        assert ver == "test1"

    loaded_core.eval()
    with torch.no_grad():
        out_after = loaded_core(x, dt).encoded_states
    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_core_checkpoint_with_qdora_roundtrip():
    core = _build_core("samba")
    wrap_core_with_qdora(core, QDoRAConfig(rank=4, quantize_base=False))
    core.eval()
    x = torch.randn(2, 3, HIDDEN)
    dt = torch.randn(2, 3, 1).abs()
    with torch.no_grad():
        out_before = core(x, dt).encoded_states

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core_qdora.pt"
        save_core_checkpoint(core, path)
        loaded_core, _, _ = load_core_checkpoint(path)

    loaded_core.eval()
    with torch.no_grad():
        out_after = loaded_core(x, dt).encoded_states
    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_derive_backbone_version_is_deterministic():
    core = _build_core()
    v1 = derive_backbone_version(core)
    v2 = derive_backbone_version(core)
    assert v1 == v2


def test_derive_backbone_version_changes_with_weights():
    core = _build_core()
    v1 = derive_backbone_version(core)
    for p in core.parameters():
        p.data.add_(1e-3)
    v2 = derive_backbone_version(core)
    assert v1 != v2


def test_temporal_core_save_load():
    torch.manual_seed(42)
    events = _dummy_events()
    model = TemporalCoreModel(
        sequence_id_field="cid", event_id_field="eid", timestamp_field="ts",
        categorical_fields=["etype"], numeric_fields=["val"],
        config=TemporalCoreConfig(hidden_dim=HIDDEN, epochs=1, device="cpu",
                                  sequence_backend="mamba2", backbone_version="save-test"),
    )
    orig_out = model.fit_transform(events)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "temporal.pt"
        version = model.save_pretrained(path)
        loaded = TemporalCoreModel.load_pretrained(path, device="cpu")
        assert loaded.trained_core is not None
        assert version == "save-test"  # explicit non-default version is preserved

    # Encode same data through the reloaded core.
    core = loaded.trained_core
    assert core is not None
    core.eval()
    hidden = torch.randn(2, 3, HIDDEN)
    dt = torch.randn(2, 3, 1).abs()
    with torch.no_grad():
        out = core(hidden, dt).encoded_states
    assert out.shape == (2, 3, HIDDEN)
    assert torch.isfinite(out).all()


def _dummy_events():
    return [
        {"cid": "a", "eid": "e1", "ts": "2024-01-01T00:00:00", "etype": "x", "val": 1.0},
        {"cid": "a", "eid": "e2", "ts": "2024-01-02T00:00:00", "etype": "y", "val": 2.0},
        {"cid": "b", "eid": "e3", "ts": "2024-01-01T00:00:00", "etype": "z", "val": 3.0},
        {"cid": "b", "eid": "e4", "ts": "2024-01-03T00:00:00", "etype": "x", "val": 4.0},
    ]
