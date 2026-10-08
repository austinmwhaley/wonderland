"""Real-data fix regressions (backported bug classes from the vendored fork).

Each test pins one failure that only shows up beyond the dense synthetic
streams rabbit_hole produces:

  1. half-life derived from a MERGED timeline was pinned at the 1-hour floor
  2. resolve_cfm recorded explicit overrides, then returned derived values
  3. govern returned an older/worse state and counted patience before the
     noise floor was measurable
  4. blanket min_events=3 hid 1-2 event customers from the state store
  5. training-anchor readouts skipped the fade serving applies (train/serve skew)
  6. warm-start ignored behavior-version / architecture mismatches
  7. ladder tolerance (unpaired 2*max SE) too wide; unconverged rungs selectable
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from looking_glass.autotune import derive_half_life, govern, resolve_cfm
from looking_glass.cfm_config import CFMConfig, apply_set_overrides
from looking_glass.cfm_data import _read_stream, build_sequences


# ---------------------------------------------------------------------------
# 1. half-life: within-customer gaps (p95), never the merged timeline
# ---------------------------------------------------------------------------
def test_half_life_from_within_customer_gaps_not_merged_timeline():
    # 10 customers, events 10 days apart WITHIN a customer, but interleaved
    # 1 second apart in the merged timeline -> the old np.sort(diffs) median
    # lands on ~1s and clips to the 1h floor; the fix must see the 10d gaps.
    rows_k, rows_t = [], []
    for c in range(10):
        for d in range(6):
            dt = datetime(2025, 1, 1) + timedelta(days=10 * d, seconds=c)
            rows_k.append(f"c{c}")
            rows_t.append(dt.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00")
    df = pl.DataFrame({"customer_key": rows_k, "event_ts": rows_t})
    hl, receipt = derive_half_life(df)
    assert receipt["state_half_life_days"].startswith("p95 within-customer")
    assert hl == pytest.approx(10.0, abs=0.01)  # merged-timeline bug -> 1/24


# ---------------------------------------------------------------------------
# 2. resolve: explicit overrides are APPLIED, not just recorded
# ---------------------------------------------------------------------------
def test_resolve_cfm_applies_explicit_overrides():
    df = pl.DataFrame(
        {
            "customer_key": ["c1", "c1", "c2"],
            "event_ts": ["2025-01-01T00:00:00+00:00"] * 3,
        }
    )
    base = CFMConfig(seq_len=96)  # != default 128 -> explicit override
    res = resolve_cfm(base, df, ["c1", "c2"], (2, 1, 1))
    assert res.seq_len == 96  # was: recorded in overrides but derived value returned
    assert res.receipt["overrides"]["seq_len"] == 96
    res_def = resolve_cfm(CFMConfig(), df, ["c1", "c2"], (2, 1, 1))
    assert "seq_len" not in res_def.receipt["overrides"]


def test_set_overrides_applied_after_derivation_and_fail_safe():
    cfg = CFMConfig()
    cfg.set_overrides = ["state_half_life_days=7.5", "dim=128"]
    receipt: dict = {}
    apply_set_overrides(cfg, receipt)
    assert cfg.state_half_life_days == 7.5
    assert cfg.dim == 128
    assert receipt["overrides"]["dim"] == 128
    bad = CFMConfig()
    bad.set_overrides = ["no_such_field=1"]
    with pytest.raises(SystemExit):
        apply_set_overrides(bad)


# ---------------------------------------------------------------------------
# 3. govern: strict lowest-loss; patience gated on a measurable noise floor
# ---------------------------------------------------------------------------
def test_govern_patience_waits_for_measurable_noise(capsys):
    vals = iter([4.0] * 10)  # flat from the very first eval

    def val_metric():
        v = next(vals)
        return v, v

    gov, best = govern(lambda n: None, val_metric, 10_000, 10, patience=3)
    # earliest stop = 3 evals (floor measurable) + patience-1 -> 5 evals
    assert gov["steps"] == 50 and gov["evals"] == 5
    assert gov["stopped"] == "converged"
    assert best == 4.0
    assert "[govern]" in capsys.readouterr().out  # progress prints


def test_govern_returns_the_true_lowest_loss_state():
    # global min arrives inside the noise band of the running best — the old
    # `v < best - tol` record rule skipped it and returned an older, worse state
    vals = iter([5.0, 4.0, 3.0, 2.95, 2.95, 2.95, 2.95])

    def val_metric():
        v = next(vals)
        return v, v

    gov, best = govern(lambda n: None, val_metric, 10_000, 10, patience=3)
    assert gov["best"] == pytest.approx(2.95)
    assert best == pytest.approx(2.95)  # old rule kept best=3.0 here
    assert gov["stopped"] == "converged"


# ---------------------------------------------------------------------------
# 4. min_events: 3 for training sequences, 1 for state rows
# ---------------------------------------------------------------------------
def _events(pairs):
    return pl.DataFrame(
        {
            "customer_key": [k for k, _ in pairs],
            "event_ts": [t for _, t in pairs],
            "event_type": ["view"] * len(pairs),
            "brand": [None] * len(pairs),
            "entity_type": ["customer"] * len(pairs),
            "entity_id": [k for k, _ in pairs],
            "value": [0.0] * len(pairs),
        }
    ).sort(["customer_key", "event_ts"])


def test_min_events_training_keeps_3_states_accept_1():
    df = _events(
        [("c1", "2025-01-01T00:00:00+00:00"), ("c1", "2025-01-02T00:00:00+00:00")]
        + [("c2", f"2025-01-0{i}T00:00:00+00:00") for i in range(1, 5)]
    )
    cfg = CFMConfig()
    split = {"c1": "A", "c2": "A"}
    train = build_sequences(df, ["c1", "c2"], cfg, split, with_anchors=False)
    assert [s["customer"] for s in train] == ["c2"]  # 2-event c1 filtered
    states = build_sequences(df, ["c1", "c2"], cfg, split, with_anchors=False, min_events=1)
    assert sorted(s["customer"] for s in states) == ["c1", "c2"]


# ---------------------------------------------------------------------------
# 5. readout fade: independent closed-form oracle + serving path parity
# ---------------------------------------------------------------------------
def test_fade_closed_form_and_semigroup_oracle():
    from looking_glass.cfm_state import fade

    h = torch.tensor([1.0, 2.0, -3.0])
    hl = 7.0
    dt = 3 * 86400.0
    expected = h * (0.5 ** (dt / (hl * 86400.0)))  # independent formula
    assert torch.allclose(fade(h, dt, hl), expected, atol=1e-6)
    half = fade(h, 1.5 * 86400, hl)
    assert torch.allclose(fade(half, 1.5 * 86400, hl), expected, atol=1e-6)
    assert torch.equal(fade(h, 0.0, hl), h)
    assert torch.equal(fade(h, -100.0, hl), h)


def test_fade_idle_serving_path_matches_closed_form(tmp_path):
    from looking_glass.cfm_model import CFM, EventVocab
    from looking_glass.cfm_state import StateStore

    torch.manual_seed(0)
    seq = {
        "event_type": np.array(["view", "view", "order"], dtype=object),
        "brand": np.array([None, None, None], dtype=object),
        "entity_type": np.array(["customer"] * 3, dtype=object),
    }
    model = CFM(EventVocab.build([seq]), dim=16, n_experts=1)
    cfg = CFMConfig(state_half_life_days=7.0)
    store = StateStore(tmp_path / "s.duckdb", cfg, model)
    h = torch.zeros(16)
    h[0], h[5] = 1.0, 2.0
    t0 = 1_700_000_000.0
    store.upsert("c1", h, torch.zeros(16), t0, None)
    store.fade_idle(t0 + 3 * 86400.0)
    got = torch.tensor(
        store.con.execute("SELECT state FROM customer_state").fetchone()[0],
        dtype=torch.float32,
    )
    expected = h * (0.5 ** (3.0 / 7.0))  # 3 days at a 7-day half-life
    assert torch.allclose(got, expected, atol=1e-6)
    store.close()


# ---------------------------------------------------------------------------
# 6. warm-start: behavior-version + resolved-architecture compatibility
# ---------------------------------------------------------------------------
def test_warm_pick_requires_version_and_architecture_match(tmp_path):
    import json as _json

    from looking_glass.cfm_training import _pick_warm_checkpoint

    def reg(name, version, seq_len):
        (tmp_path / name).write_text(
            _json.dumps(
                {
                    "tag": f"{version}r1",
                    "as_of": "2025-11-01",
                    "version": version,
                    "config": {"seq_len": seq_len, "dim": 64, "state_half_life_days": 30.0},
                }
            )
        )

    cfg = CFMConfig(version="v2.1.0", seq_len=128, dim=64, state_half_life_days=30.0)
    # old behavior version -> incompatible
    reg("registry_old.json", "v2.0.0", 128)
    assert _pick_warm_checkpoint(tmp_path, "2025-12-01", cfg) == (None, None)
    # same version but different resolved architecture -> incompatible
    reg("registry_baddim.json", "v2.1.0", 64)
    assert _pick_warm_checkpoint(tmp_path, "2025-12-01", cfg) == (None, None)
    # matching -> eligible
    reg("registry_ok.json", "v2.1.0", 128)
    tag, _ = _pick_warm_checkpoint(tmp_path, "2025-12-01", cfg)
    assert tag == "v2.1.0r1"


# ---------------------------------------------------------------------------
# 7. ladder: paired noise vs best; unconverged rungs excluded; fallback
# ---------------------------------------------------------------------------
def _ladder_rows():
    return [
        {
            "rung": 250,
            "auc": 0.700,
            "auc_se": 0.020,
            "governor_stopped": "converged",
            "n_train_sequences": 250,
        },
        {
            "rung": 500,
            "auc": 0.730,
            "auc_se": 0.010,
            "governor_stopped": "converged",
            "n_train_sequences": 500,
        },
        {
            "rung": 1000,
            "auc": 0.735,
            "auc_se": 0.010,
            "governor_stopped": "budget",
            "n_train_sequences": 1000,
        },
    ]


def _ladder_preds(seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    keys = np.array([f"c{i % 50}" for i in range(n)])
    y = rng.integers(0, 2, n).astype(np.int64)
    base = 0.2 + 0.6 * y + rng.normal(0, 0.25, n)
    return {
        250: {"keys": keys, "y": y, "pred": base + rng.normal(0, 0.02, n)},
        500: {"keys": keys, "y": y, "pred": base},
    }


def test_ladder_choose_paired_noise_and_excludes_unconverged():
    from plugins.ladder_sample_a import _choose

    chosen, best, tol, mode, excluded = _choose(_ladder_rows(), _ladder_preds())
    assert excluded == [1000]  # governor hit the budget -> not selectable
    assert best["rung"] == 500  # the unconverged 0.735 never wins
    assert mode == "paired"
    assert chosen["rung"] == 500  # gap 0.03 > paired tol; unpaired 0.04 picked 250


def test_ladder_unpaired_fallback_for_legacy_receipts():
    from plugins.ladder_sample_a import _choose

    chosen, _best, tol, mode, _excluded = _choose(_ladder_rows(), {})
    assert mode == "unpaired_fallback"
    assert chosen["rung"] == 250  # legacy rule: 2*max(SE)=0.04 >= gap 0.03
    assert tol == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# as_of pushdown: SQL cut == polars cut (leak-free load, bounded memory)
# ---------------------------------------------------------------------------
def test_read_stream_pushes_as_of_cut_into_sql(tmp_path):
    import duckdb

    db = tmp_path / "s.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE customer_events AS SELECT * FROM (VALUES "
        "('c1','2025-01-01T00:00:00+00:00','b','view','{}','cust','e1',0.0), "
        "('c1','2025-06-01T00:00:00+00:00','b','view','{}','cust','e2',0.0)"
        ") AS t(customer_key, event_ts, brand, event_type, event_attributes, "
        "entity_type, entity_id, value)"
    )
    con.close()
    df = _read_stream(CFMConfig(db=str(db), as_of="2025-03-01"))
    assert df.height == 1
    assert df["event_ts"][0].startswith("2025-01-01")
