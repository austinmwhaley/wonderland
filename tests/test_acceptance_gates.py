"""Acceptance-gate wrappers: the layer gates run as pytest, not just scripts.

- rabbit_hole has its own suite (rabbit_hole/tests/test_acceptance.py).
- plugins acceptance runs end-to-end when the CFM products db exists,
  otherwise skips (fresh clone / CI without generated data).
- the white_queen bench *contract* is validated cheaply here; the full
  bench run is exercised via `python -m white_queen.tribunal.bench.scorecard`.
"""

from __future__ import annotations

import pytest


def test_plugins_layer_acceptance():
    from plugins.base import CFM_PRODUCTS

    if not CFM_PRODUCTS.exists():
        pytest.skip("CFM products db not generated (run rabbit_hole + looking_glass first)")
    from plugins import acceptance

    assert acceptance.main([]) == 0


def test_bench_contract_integrity():
    from white_queen.tribunal.bench.acceptance import CONTRACT, completion, render

    suite_families = set(CONTRACT["suite"])
    assert set(CONTRACT["families"]) == suite_families
    assert CONTRACT["seeds_min"] <= len(CONTRACT["seeds"])
    th = CONTRACT["thresholds"]
    assert th["false_deploys"] == 0
    assert 0 < th["coverage"] <= 1
    assert 0 <= th["precision"] <= 1 and 0 <= th["recall"] <= 1
    assert 0 <= th["rank_rho"] <= 1

    # render/completion survive a degenerate (all-failing) row set
    rows = [("Coverage", "truth in reported CI", 0.9, None, False)]
    assert completion(rows) == 0.0
    out = render(rows)
    assert "FAIL" in out and "completion: 0%" in out
