import pytest
import torch

from looking_glass import RejectionSampler, SamplingReport


def _batches():
    return {
        "x": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "y": torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
    }


def test_and_logic():
    v1 = lambda b: b["x"].squeeze(-1) > 1.5
    v2 = lambda b: b["y"].squeeze(-1) == 1.0
    sampler = RejectionSampler([v1, v2])
    filtered, report = sampler.filter_batch(_batches())
    assert report.total == 4
    assert report.kept == 2                # rows 2 and 3 (x>1.5 & y==1)
    assert report.rejected == 2
    assert filtered["x"].tolist() == [[3.0], [4.0]]


def test_keep_zero():
    v = lambda b: b["x"].squeeze(-1) < 0
    sampler = RejectionSampler([v])
    filtered, report = sampler.filter_batch(_batches())
    assert report.kept == 0
    assert report.keep_ratio == 0.0


def test_keep_all():
    v = lambda b: torch.ones(b["x"].size(0), dtype=torch.bool)
    sampler = RejectionSampler([v])
    filtered, report = sampler.filter_batch(_batches())
    assert report.kept == 4


def test_empty_batch_raises():
    sampler = RejectionSampler([lambda b: torch.ones(0, dtype=torch.bool)])
    with pytest.raises(ValueError):
        sampler.filter_batch({})


def test_no_validators_raises():
    with pytest.raises(ValueError):
        RejectionSampler([])
