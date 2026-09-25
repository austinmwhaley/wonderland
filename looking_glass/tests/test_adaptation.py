import torch

from looking_glass import (
    AdapterFusion,
    ClassificationHead,
    QDoRAConfig,
    QDoRALinear,
    RegressionHead,
)


def test_qdora_zero_init_equals_base():
    lin = QDoRALinear(8, 4, QDoRAConfig(rank=2, quantize_base=False), bias=False)
    base = lin.base
    with torch.no_grad():
        base.weight.copy_(torch.randn_like(base.weight))
    x = torch.randn(3, 8)
    base_out = base(x)
    qdora_out = lin(x)
    assert torch.allclose(base_out, qdora_out, atol=1e-5)


def test_base_frozen():
    lin = QDoRALinear(8, 4, QDoRAConfig(rank=2))
    for p in lin.base.parameters():
        assert p.requires_grad is False
    assert lin.lora_a.weight.requires_grad is True
    assert lin.lora_b.weight.requires_grad is True
    assert lin.magnitude.requires_grad is True


def test_magnitude_scales():
    lin = QDoRALinear(4, 4, QDoRAConfig(rank=2, quantize_base=False), bias=False)
    with torch.no_grad():
        lin.lora_a.weight.copy_(torch.randn(2, 4))
        lin.lora_b.weight.copy_(torch.randn(4, 2))
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    original = lin(x).clone()
    with torch.no_grad():
        lin.magnitude.copy_(lin.magnitude * 2)
    assert not torch.allclose(original, lin(x), atol=1e-4)


def test_adapter_fusion_smoke():
    fusion = AdapterFusion(hidden_dim=8, num_adapters=2)
    x = torch.randn(3, 5, 8)
    out = fusion([x, x + 1])
    assert out.shape == (3, 5, 8)
    assert torch.isfinite(out).all()


def test_classification_head_output_shape():
    head = ClassificationHead(hidden_dim=16, num_classes=5)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 5)


def test_classification_head_binary():
    head = ClassificationHead(hidden_dim=16, num_classes=1)
    out = head(torch.randn(4, 16))
    assert out.shape == (4,)


def test_regression_head():
    head = RegressionHead(hidden_dim=16, output_dim=3)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 3)


def test_regression_head_scalar():
    head = RegressionHead(hidden_dim=16, output_dim=1)
    out = head(torch.randn(4, 16))
    assert out.shape == (4,)
