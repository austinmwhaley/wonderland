"""Tests for the S1 anchored additive counter (looking_glass/cfm_model.py)."""

from __future__ import annotations

import torch

from looking_glass.cfm_model import CounterBlock, signed_log1p


def test_counter_is_exact_monoid_and_holds_on_pad():
    torch.manual_seed(0)
    blk = CounterBlock(6, 5)
    x = torch.randn(2, 5, 6)
    raw = torch.rand(2, 5, 5)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    cre, M = blk(x, raw, mask=mask)
    drive = torch.sigmoid(blk.W_g(x)) * raw * mask.unsqueeze(-1)
    assert torch.allclose(M[:, -1], drive.sum(1), atol=1e-5)  # exact sum
    assert torch.allclose(M[0, 3], M[0, 2], atol=1e-6)  # pads HOLD
    assert torch.allclose(cre, signed_log1p(M))  # magnitude-preserving readout


def test_counter_gate_open_at_init():
    blk = CounterBlock(6, 5)
    assert float(torch.sigmoid(blk.W_g.bias.detach()).min()) > 0.9  # starts ~exact


def test_signed_log1p_monotone():
    x = torch.tensor([1.0, 10.0, 100.0, 1000.0])
    y = signed_log1p(x)
    assert torch.all(y[1:] > y[:-1])
    assert torch.allclose(signed_log1p(-x), -y)


def test_counter_registers_exact_and_route_sums_to_one():
    torch.manual_seed(0)
    blk = CounterBlock(6, 5, n_registers=4)
    x = torch.randn(2, 5, 6)
    raw = torch.rand(2, 5, 5)
    mask = torch.ones(2, 5)
    cre, M = blk(x, raw, mask=mask)
    assert M.shape == (2, 5, 20)  # 4 registers x 5 raw
    assert torch.allclose(signed_log1p(M), cre)
    # softmax routing sums to 1 -> total over registers == gated raw sum (exact)
    g = torch.sigmoid(blk.W_g(x))
    assert torch.allclose(M.reshape(2, 5, 4, 5).sum(2)[:, -1], (g * raw).sum(1), atol=1e-5)
