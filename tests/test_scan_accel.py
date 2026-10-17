"""A2: numerical equivalence of the accelerated `_scan` path vs the eager scan.

The accelerated path is a module-cached `torch.compile` of the SAME affine
recurrence (`LG_SCAN=auto|compiled`); these tests force it on, run it on CPU
against the eager implementation, and assert `torch.allclose` (atol 1e-5)."""

import pytest
import torch

from looking_glass import cfm_model
from looking_glass.cfm_model import _scan_eager


def setup_module(module):
    # Force the accelerated path on for this module (default `auto` may already
    # select it; this makes the test independent of the ambient LG_SCAN value).
    cfm_model._scan_disabled = False
    cfm_model._scan_compiled = None


def _accelerated(d, b):
    out = cfm_model._scan(d, b)
    assert cfm_model._scan_compiled is not None, "accelerated path did not engage"
    assert not cfm_model._scan_disabled, "accelerated path fell back to eager"
    return out


def _rand_inputs(B, T, D, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    decay = torch.rand(B, T, 1, generator=g, dtype=dtype) * 0.999 + 1e-3
    drive = torch.randn(B, T, D, generator=g, dtype=dtype)
    return decay, drive


@pytest.mark.parametrize("T", [1, 7, 130])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scan_accel_equivalent(T, dtype):
    decay, drive = _rand_inputs(3, T, 16, dtype)
    De, He = _scan_eager(decay, drive)
    Da, Ha = _accelerated(decay, drive)
    assert torch.allclose(De, Da, atol=1e-5)
    assert torch.allclose(He, Ha, atol=1e-5)
    # the dispatcher's default path agrees too
    Dd, Hd = cfm_model._scan(decay, drive)
    assert torch.allclose(He, Hd, atol=1e-5) and torch.allclose(De, Dd, atol=1e-5)


def test_scan_accel_batch_equivalent():
    B, T, D = 64, 33, 24
    decay, drive = _rand_inputs(B, T, D, torch.float32, seed=7)
    _, He = _scan_eager(decay, drive)
    _, Ha = _accelerated(decay, drive)
    assert Ha.shape == (B, T, D)
    assert torch.allclose(He, Ha, atol=1e-5)


def test_scan_accel_preserves_initial_state():
    # The `+ D*h0` initial-state term is applied by callers, outside `_scan`;
    # both paths must compose with it identically.
    B, T, D = 5, 41, 12
    decay, drive = _rand_inputs(B, T, D, torch.float32, seed=11)
    h0 = torch.randn(B, D)
    De, He = _scan_eager(decay, drive)
    Da, Ha = _accelerated(decay, drive)
    assert torch.allclose(He + De * h0.unsqueeze(1), Ha + Da * h0.unsqueeze(1), atol=1e-5)
