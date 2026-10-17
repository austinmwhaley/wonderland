"""Tests for leave-one-out objective-set minimization (C6 / minimality)."""

from __future__ import annotations

import json

import numpy as np

from looking_glass.objective_pruning import load_ablation, prune_objectives


def test_leave_one_out_split_and_receipt():
    base = [1.0] * 5
    without = {
        "redundant_a": [1.0] * 5,
        "redundant_b": [1.0] * 5,
        "redundant_c": [1.0] * 5,
        "loadbearing_a": [0.5] * 5,
        "loadbearing_b": [0.5] * 5,
    }
    kept, pruned, receipt = prune_objectives(base, without, full_seeds=list(range(5)))

    assert set(pruned) == {"redundant_a", "redundant_b", "redundant_c"}
    assert set(kept) == {"loadbearing_a", "loadbearing_b"}

    assert receipt["k_se"] == 1.0
    assert receipt["n_seeds"] == 5
    assert receipt["full_seeds"] == [0, 1, 2, 3, 4]
    assert receipt["base"] == {"mean": 1.0, "se": 0.0}
    assert receipt["noise_floor"] == 0.0

    assert receipt["per_objective"]["redundant_a"]["mean"] == 1.0
    assert receipt["per_objective"]["redundant_a"]["se"] == 0.0
    assert receipt["per_objective"]["redundant_a"]["dominated"] is True
    assert receipt["per_objective"]["loadbearing_a"]["mean"] == 0.5
    assert receipt["per_objective"]["loadbearing_a"]["dominated"] is False

    assert receipt["kept"] == ["loadbearing_a", "loadbearing_b"]
    assert receipt["pruned"] == ["redundant_a", "redundant_b", "redundant_c"]


def test_noise_floor_is_data_derived():
    base = [1.0, 1.1, 0.9, 1.0, 1.0]
    without = {
        # removal clearly hurts: below mean - 1 SE -> kept
        "hurts": [0.4, 0.5, 0.3, 0.45, 0.35],
        # removal is a no-op -> pruned
        "flat": list(base),
    }
    kept, pruned, receipt = prune_objectives(base, without, full_seeds=5, k_se=1.0)

    assert set(kept) == {"hurts"}
    assert set(pruned) == {"flat"}
    assert receipt["base"]["se"] > 0.0
    assert receipt["noise_floor"] == receipt["base"]["se"]


def test_greedy_prunes_jointly_redundant_pair():
    base = [1.0] * 5
    without = {"a": [0.5] * 5, "b": [0.5] * 5}

    def evaluate(removed: tuple[str, ...]):
        return [1.0] * 5 if set(removed) == {"a", "b"} else [0.5] * 5

    kept, pruned, receipt = prune_objectives(
        base, without, [0, 1, 2, 3, 4], k_se=1.0, evaluate=evaluate
    )

    # Neither alone survives the noise floor, so neither is single-prunable...
    assert set(pruned) == set()
    assert set(kept) == {"a", "b"}
    # ...but the joint ablation is within tolerance, so greedy drops both.
    assert set(receipt["greedy_removed"]) == {"a", "b"}
    assert "joint ablation" in receipt["greedy_reason"]


def test_greedy_additive_fallback_without_evaluate():
    base = [1.0] * 5
    without = {"redundant": [1.0] * 5, "hurts": [0.5] * 5}
    _, _, receipt = prune_objectives(base, without, 5)
    assert receipt["greedy_removed"] == ["redundant"]


def test_load_ablation_roundtrip(tmp_path):
    payload = {"base": [1.0, 1.0], "without": {"x": [1.0, 1.0], "y": [0.2, 0.3]}}
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(payload))

    base, without = load_ablation(path)
    assert np.allclose(base, [1.0, 1.0])
    assert set(without) == {"x", "y"}
    assert np.allclose(without["y"], [0.2, 0.3])

    _, pruned, _ = prune_objectives(base, without, 2)
    assert pruned == ("x",)
