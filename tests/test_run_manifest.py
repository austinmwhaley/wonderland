"""Provenance guards: a card must not inherit config from an unrelated run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import frozen  # noqa: E402


def _write_manifest(tmp_path, axes, **extra):
    base = tmp_path / "cache"
    man = {"axes": axes, "seed": 0, "data": "/tmp/x", "source": "build_frozen", **extra}
    frozen.manifest_path(base).write_text(json.dumps(man))
    return base


def test_assert_axes_accepts_matching_config(tmp_path):
    base = _write_manifest(tmp_path, ["measure"])
    man = frozen.assert_axes(base, ["measure"])
    assert man["source"] == "build_frozen"


def test_assert_axes_rejects_mismatched_config(tmp_path):
    base = _write_manifest(tmp_path, ["noaction"])
    with pytest.raises(RuntimeError, match="refusing to mix runs"):
        frozen.assert_axes(base, [])


def test_assert_axes_is_order_insensitive(tmp_path):
    base = _write_manifest(tmp_path, ["measure", "query"])
    assert frozen.assert_axes(base, ["query", "measure"])["axes"] == ["measure", "query"]


def test_manifest_path_is_adjacent_to_cache(tmp_path):
    p = frozen.manifest_path(tmp_path / "frozen_6k")
    assert p.name == "frozen_6k.manifest.json"
