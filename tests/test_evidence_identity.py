"""Evidence-identity regressions (ROADMAP Phase 0): receipts must survive,
be attributable to code, and never silently shrink what they claim to cover."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ladder rung dirs are per-as_of (overwrites destroyed Nov's receipt)
# ---------------------------------------------------------------------------
def test_ladder_rung_dirs_are_as_of_scoped():
    import plugins.ladder_sample_a as L

    a = L._rung_dir("2025-11-01", 250)
    b = L._rung_dir("2025-12-01", 250)
    assert a != b
    assert a.parts[-2:] == ("2025_11_01", "r250")
    assert a.name == "r250" and L.LADDER_DIR in a.parents


# ---------------------------------------------------------------------------
# per-run artifact archive (monthly reruns overwrite the tag path)
# ---------------------------------------------------------------------------
def test_save_artifact_archives_each_run(tmp_path):
    from plugins.base import PluginSpec, save_artifact

    spec = PluginSpec(name="t", kind="supervised", target="x", window_days=30)
    payload = {"as_of": "2025-11-01", "verdict": True}
    save_artifact(spec, payload, out=tmp_path)
    assert (tmp_path / f"{spec.tag}.json").exists()
    arch = tmp_path / "archives" / f"{spec.tag}_2025-11-01.json"
    assert arch.exists()
    assert json.loads(arch.read_text())["payload"]["verdict"] is True
    # no as_of -> no archive clutter (legacy payload path unchanged)
    save_artifact(spec, {"verdict": False}, out=tmp_path)
    assert len(list((tmp_path / "archives").glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# a mixed-version products table must fail at TRAIN time, not at inference
# ---------------------------------------------------------------------------
def test_run_target_rejects_null_encoder_pin(monkeypatch):
    import plugins.base as base
    import plugins.head_template as ht
    from plugins.targets import REGISTRY

    fake = SimpleNamespace(meta={"encoder_version": None, "feature_table": "donor_embeddings"})
    monkeypatch.setattr(base, "load_dataset", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="multiple encoder versions"):
        ht.run_target(REGISTRY["supervised_purchase_propensity_30d"], as_of="2025-11-01")


# ---------------------------------------------------------------------------
# rehearsal: midweek is derived per cycle (the literal dropped cycle 2's day)
# ---------------------------------------------------------------------------
def test_rehearsal_midweek_derived_per_cycle():
    from plugins.rehearsal import _inference_days

    nov = _inference_days(date(2025, 11, 1), 8)
    dec = _inference_days(date(2025, 12, 1), 8)
    assert date(2025, 11, 5) in nov  # Wednesday after a Saturday start
    assert date(2025, 12, 3) in dec  # derived for cycle 2 (was: literal Nov 5)
    assert date(2025, 11, 5) not in dec
    # explicit midweek still wins
    assert date(2025, 12, 4) in _inference_days(date(2025, 12, 1), 8, date(2025, 12, 4))
    # training day always present
    assert nov[0] == date(2025, 11, 1)


def test_rehearsal_receipt_carries_git_stamp():
    from plugins.rehearsal import _git_sha

    sha = _git_sha()
    assert isinstance(sha, str) and sha and sha != "unknown"  # in a git checkout


# ---------------------------------------------------------------------------
# battery: skipped targets recorded; empty portfolio fails safe
# ---------------------------------------------------------------------------
def _battery_inputs(n=120):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    groups = np.array([f"c{i % 20}" for i in range(n)])
    y = X[:, 0] * 2 + rng.normal(size=n)
    E, R = X.copy(), rng.normal(size=(n, 3))
    full = {"orders_90": y}
    sparse = {"tiny": np.where(np.arange(n) < 50, y, np.nan)}  # 50 labeled < 100
    return E, R, groups, full, sparse


def test_battery_records_skipped_targets(monkeypatch):
    import looking_glass.sufficiency_battery as B

    E, R, groups, full, sparse = _battery_inputs()
    monkeypatch.setattr(B, "_load", lambda *a, **k: (None, None, None, None, "vTest"))
    monkeypatch.setattr(B, "build_all", lambda *a: (E, R, groups, {**full, **sparse}))
    res = B.run(0.6)
    assert [s["target"] for s in res["skipped"]] == ["tiny"]
    assert [r["target"] for r in res["rows"]] == ["orders_90"]
    assert res["seed"] == 0
    assert set(res) >= {"version", "rows", "skipped", "signal_cov", "success"}


def test_battery_empty_portfolio_fails_safe(monkeypatch):
    import looking_glass.sufficiency_battery as B

    E, R, groups, _full, _sparse = _battery_inputs()
    all_nan = {f"t{i}": np.full(len(groups), np.nan) for i in range(3)}
    monkeypatch.setattr(B, "_load", lambda *a, **k: (None, None, None, None, "vTest"))
    monkeypatch.setattr(B, "build_all", lambda *a: (E, R, groups, all_nan))
    with pytest.raises(SystemExit, match="nothing to gate"):
        B.run(0.6)


# ---------------------------------------------------------------------------
# battery receipt is persisted by main() (per-run stamped file)
# ---------------------------------------------------------------------------
def test_battery_main_writes_receipt(tmp_path, monkeypatch):
    import looking_glass.sufficiency_battery as B

    res = {
        "version": "v2.1.0r1",
        "rows": [{"target": "t", "n": 100}],
        "skipped": [{"target": "tiny", "n_labeled": 50}],
        "seed": 0,
        "n": 100,
        "signal_cov": 1.0,
        "standalone_cov": 1.0,
        "unique_cov": 1.0,
        "threshold": 0.6,
        "success": True,
    }
    monkeypatch.setattr(B, "run", lambda *a, **k: dict(res))
    monkeypatch.setattr(B, "show", lambda r: True)
    rc = B.main([], out_dir=tmp_path)
    assert rc == 0
    files = list(tmp_path.glob("battery_v2_1_0r1_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["skipped"][0]["target"] == "tiny"
    assert "ran_at" in saved and saved["success"] is True
    # failing battery exits 1
    monkeypatch.setattr(B, "run", lambda *a, **k: {**res, "success": False})
    assert B.main([], out_dir=tmp_path / "b2") == 1
