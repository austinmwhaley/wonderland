"""B4: equivalence of the chunked SSD scan (`ssd_scan`) with a naive loop
reference and with the existing `looking_glass.cfm_model._scan` affine scan.

CPU-only. Asserts `torch.allclose` at fp32 atol 1e-5 for T in {1,7,64,65,130},
d in {1,8,64}, with and without batch dims and an initial state h0.
"""

import os

os.environ.setdefault("LG_SCAN", "eager")  # deterministic reference path

import pytest  # noqa: E402
import torch  # noqa: E402

from looking_glass.cfm_model import _scan, _scan_eager  # noqa: E402
from looking_glass.ssd_scan import ssd_scan  # noqa: E402


def _rand(shape_lead, T, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    decay = torch.rand(*shape_lead, T, 1, generator=g) * 0.98 + 0.01
    drive = torch.randn(*shape_lead, T, d, generator=g)
    return decay, drive


def _naive(decay, drive, h0=None):
    """Straight Python-loop reference: h_t = decay_t h_{t-1} + drive_t."""
    lead = tuple(drive.shape[:-2])
    T, d = drive.shape[-2], drive.shape[-1]
    dec = decay.reshape(*lead, T)
    h = torch.zeros(*lead, d) if h0 is None else h0
    outs = []
    for t in range(T):
        h = dec[..., t].unsqueeze(-1) * h + drive[..., t, :]
        outs.append(h)
    return torch.stack(outs, dim=-2)


@pytest.mark.parametrize("T", [1, 7, 64, 65, 130])
@pytest.mark.parametrize("d", [1, 8, 64])
@pytest.mark.parametrize("chunk", [1, 32, 64, 128])
def test_ssd_matches_naive(T, d, chunk):
    decay, drive = _rand((), T, d, seed=T * 7 + d + chunk)
    h0 = torch.randn(d)
    ref = _naive(decay, drive, h0)
    got = ssd_scan(decay, drive, chunk=chunk, h0=h0)
    assert got.shape == (T, d)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


@pytest.mark.parametrize("T", [1, 7, 64, 65, 130])
@pytest.mark.parametrize("d", [1, 8, 64])
def test_ssd_matches_naive_batched(T, d):
    decay, drive = _rand((2, 3), T, d, seed=101)
    h0 = torch.randn(2, 3, d)
    ref = _naive(decay, drive, h0)
    got = ssd_scan(decay, drive, chunk=64, h0=h0)
    assert got.shape == (2, 3, T, d)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


@pytest.mark.parametrize("T", [1, 7, 64, 65, 130])
@pytest.mark.parametrize("d", [1, 8, 64])
def test_ssd_matches_existing_scan(T, d):
    # `_scan` is 3D-only (B, T, ...); compare on that shape, eager reference.
    B = 4
    decay, drive = _rand((B,), T, d, seed=202)
    h0 = torch.randn(B, d)
    De, H = _scan_eager(decay, drive)
    ref = H + De * h0.unsqueeze(1)
    # dispatcher path (compiled or eager) must agree as well
    Dd, Hd = _scan(decay, drive)
    ref_disp = Hd + Dd * h0.unsqueeze(1)
    got = ssd_scan(decay, drive, chunk=64, h0=h0)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    assert torch.allclose(got, ref_disp, atol=1e-5), (got - ref_disp).abs().max()


def test_ssd_no_h0_matches_scan_zero_init():
    B, T, d = 5, 129, 16
    decay, drive = _rand((B,), T, d, seed=303)
    _, H = _scan_eager(decay, drive)
    got = ssd_scan(decay, drive, chunk=64)
    assert torch.allclose(got, H, atol=1e-5), (got - H).abs().max()
