"""Observational-log guarantees: the system must NOT require a holdout or A/B.

Contracts:
  * causal-claim paths REJECT with NotIdentifiableError (clear, actionable)
    when the stream has no randomized control / logged propensity — never a
    raw duckdb CatalogException, never NaN garbage;
  * with a usable control present, identification still works (positive path);
  * nba_engine's WHO step falls back to empty targeting WITH a receipt;
  * white_queen's observational path (estimate_propensity=True -> provenance
    "estimated") is covered in white_queen/tribunal/ope/tests/test_data.py.

All inputs are synthetic duckdb files under tmp_path; no repo artifacts read.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

import red_queen.engine as engine
import red_queen.evaluate_plan as evaluate_plan
import red_queen.incrementality as incrementality
import red_queen.nba_engine as nba_engine
import red_queen.response_model as response_model
from red_queen.identifiability import (
    NotIdentifiableError,
    require_holdout,
    require_propensity,
    require_stream_view,
)


def _mk_stream(path, *, holdout_rows=None, holdout_table=True, n_customers=10, n_periods=10):
    """Minimal observational stream: events + orders (+ optional holdout)."""
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE customer_events (customer_id VARCHAR, event_ts TIMESTAMP)")
    con.execute(
        "CREATE TABLE orders (customer_id VARCHAR, order_ts TIMESTAMP, gross_margin DOUBLE)"
    )
    # two distinct timestamps so period width > 0
    con.execute(
        "INSERT INTO customer_events VALUES "
        "('c0', TIMESTAMP '2026-01-01 00:00:00'), ('c0', TIMESTAMP '2026-04-01 00:00:00'), "
        "('c1', TIMESTAMP '2026-01-01 00:00:00'), ('c1', TIMESTAMP '2026-04-01 00:00:00')"
    )
    if holdout_rows is None:
        rows = [
            (f"c{c}", p, 1 if p % 10 < 3 else 0)
            for c in range(n_customers)
            for p in range(n_periods)
        ]
        holdout_rows = rows
    if holdout_table:
        con.execute(
            "CREATE TABLE email_holdout (customer_id VARCHAR, period INTEGER, holdout INTEGER)"
        )
        con.executemany("INSERT INTO email_holdout VALUES (?, ?, ?)", holdout_rows)
    con.close()
    return path


def _mk_cfm(path, n_customers=10, dim=4):
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE anchor_embeddings (customer_key VARCHAR, anchor_epoch BIGINT, embedding DOUBLE[])"
    )
    rng = np.random.default_rng(0)
    rows = [(f"c{c}", 1, rng.normal(size=dim).tolist()) for c in range(n_customers)]
    con.executemany("INSERT INTO anchor_embeddings VALUES (?, ?, ?)", rows)
    con.close()
    return path


# --------------------------------------------------------------------------
# rejection paths (no randomized control)
# --------------------------------------------------------------------------
def test_incrementality_rejects_missing_holdout_view(tmp_path, monkeypatch):
    p = _mk_stream(tmp_path / "s.duckdb", holdout_table=False)
    monkeypatch.setattr(incrementality, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="email_holdout"):
        incrementality.load()


def test_incrementality_rejects_all_zero_holdout(tmp_path, monkeypatch):
    rows = [(f"c{c}", p, 0) for c in range(5) for p in range(10)]
    p = _mk_stream(tmp_path / "s.duckdb", holdout_rows=rows)
    monkeypatch.setattr(incrementality, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="no usable randomized holdout"):
        incrementality.load()


def test_incrementality_rejects_tiny_holdout_instead_of_nan_ci(tmp_path, monkeypatch):
    # exactly 1 held row -> bootstrap CI degenerates to NaN; must reject
    rows = [("c0", 0, 1)] + [(f"c{c}", p, 0) for c in range(1, 5) for p in range(1)]
    rows = rows[:4]
    rows[0] = ("c0", 0, 1)
    p = _mk_stream(tmp_path / "s.duckdb", holdout_rows=rows)
    monkeypatch.setattr(incrementality, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="non-finite"):
        incrementality.run(seed=0)


def test_response_model_rejects_missing_holdout_view(tmp_path, monkeypatch):
    p = _mk_stream(tmp_path / "s.duckdb", holdout_table=False)
    monkeypatch.setattr(response_model, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="email_holdout"):
        response_model.build()


def test_fit_uplift_rejects_missing_holdout_view(tmp_path, monkeypatch):
    p = _mk_stream(tmp_path / "s.duckdb", holdout_table=False)
    monkeypatch.setattr(response_model, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="email_holdout"):
        response_model.fit_uplift()


def test_engine_arm_effects_rejects_missing_view(tmp_path, monkeypatch):
    p = tmp_path / "s.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE customer_events (customer_id VARCHAR, event_ts TIMESTAMP)")
    con.close()
    monkeypatch.setattr(engine, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="email_arm"):
        engine._validated_arm_effects()


def test_engine_arm_effects_rejects_missing_propensity_column(tmp_path, monkeypatch):
    p = tmp_path / "s.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE email_arm (customer_id VARCHAR, arm INTEGER)")
    con.close()
    monkeypatch.setattr(engine, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="propensity"):
        engine._validated_arm_effects()


def test_evaluate_plan_rejects_missing_view(tmp_path, monkeypatch):
    p = tmp_path / "s.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE customer_events (customer_id VARCHAR, event_ts TIMESTAMP)")
    con.close()
    monkeypatch.setattr(evaluate_plan, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="email_arm"):
        evaluate_plan._load()


def test_nba_ipw_rejects_missing_propensity(tmp_path, monkeypatch):
    p = tmp_path / "s.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE contact_sends (channel VARCHAR, arm INTEGER, discount_pct DOUBLE)")
    con.close()
    monkeypatch.setattr(nba_engine, "STREAM", p)
    with pytest.raises(NotIdentifiableError, match="propensity"):
        nba_engine._ipw_effects("email", "arm")


# --------------------------------------------------------------------------
# WHO fallback with receipt (fail-safe, not silent)
# --------------------------------------------------------------------------
def test_nba_who_falls_back_with_receipt(monkeypatch, tmp_path, capsys):
    from red_queen.identifiability import OBSERVATIONAL_HINT

    monkeypatch.setattr(nba_engine, "_best_joint", lambda ch: ((1, 5.0), 0.0))
    monkeypatch.setattr(nba_engine, "_ipw_effects", lambda ch, act: {0: 0.1, 1: 0.2})
    monkeypatch.setattr(nba_engine, "_best_hour", lambda ch: 10)
    monkeypatch.setattr(nba_engine, "OUT", tmp_path / "plan.json")

    def _no_uplift(seed=0):
        raise NotIdentifiableError(f"no control. {OBSERVATIONAL_HINT}")

    monkeypatch.setattr(response_model, "fit_uplift", _no_uplift)
    report = nba_engine.build_plan("weekly", budget=None, seed=0)
    out = capsys.readouterr().out
    assert "WHO receipt" in out and "uplift not identifiable" in out
    assert report["responders"] == 0
    assert report["budget"] == 0
    assert report["targeted"] == 0


# --------------------------------------------------------------------------
# positive path: identification still works when a control exists
# --------------------------------------------------------------------------
def test_incrementality_works_with_usable_holdout(tmp_path, monkeypatch):
    p = _mk_stream(tmp_path / "s.duckdb")
    monkeypatch.setattr(incrementality, "STREAM", p)
    monkeypatch.setattr(incrementality, "CFM", _mk_cfm(tmp_path / "cfm.duckdb"))
    res = incrementality.run(seed=0)
    assert res["heldout"] == 30 and res["treated"] == 70
    assert np.isfinite(res["pop_ci95"][0]) and np.isfinite(res["pop_ci95"][1])
    assert res["pop_ci95"][0] <= res["pop_ci95"][1]


def test_response_model_uplift_fits_with_usable_holdout(tmp_path, monkeypatch):
    p = _mk_stream(tmp_path / "s.duckdb")
    monkeypatch.setattr(response_model, "STREAM", p)
    monkeypatch.setattr(response_model, "CFM", _mk_cfm(tmp_path / "cfm.duckdb"))
    # fit_uplift derives best_arm from engine._validated_arm_effects (real
    # stream on a provisioned machine) — stub it so this test stays hermetic
    monkeypatch.setattr(engine, "_validated_arm_effects", lambda: np.array([0.1, 0.9]))
    grid = response_model.build()
    assert len(grid) > 0 and set(np.unique(grid["holdout"].to_numpy())) == {0, 1}
    keys, up, best_arm = response_model.fit_uplift(seed=0)
    assert len(keys) == len(up)
    assert np.all(np.isfinite(up))
    assert best_arm == 1


# --------------------------------------------------------------------------
# guard unit contracts
# --------------------------------------------------------------------------
def test_require_holdout_and_propensity_contracts():
    import polars as pl

    with pytest.raises(NotIdentifiableError):
        require_holdout(pl.DataFrame({"nope": [1, 2]}))
    with pytest.raises(NotIdentifiableError):
        require_holdout(pl.DataFrame({"holdout": [0, 0, 0]}))
    with pytest.raises(NotIdentifiableError):
        require_holdout(pl.DataFrame({"holdout": [1, 1, 1]}))
    require_holdout(pl.DataFrame({"holdout": [0, 1, 0]}))  # both arms: OK

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t (a INTEGER)")
    with pytest.raises(NotIdentifiableError, match="propensity"):
        require_propensity(con, "t")
    with pytest.raises(NotIdentifiableError, match="missing_view"):
        require_stream_view(con, "missing_view", "unit test")
    con.close()


# --------------------------------------------------------------------------
# decision-path purity (post-A/B removal of red_king — see red_king/ab_witness.py)
# --------------------------------------------------------------------------
def test_red_queen_decision_path_never_imports_red_king():
    """The decisive A/B (25 candidates x 5 cells, known truth) changed ZERO
    certified decisions with red_king as an MB witness -> pre-committed rule:
    red_king is analyst-tool only. Lock it: no red_queen module may import it."""
    import re
    from pathlib import Path

    rq = Path(__file__).resolve().parents[1] / "red_queen"
    offenders = []
    for f in sorted(rq.glob("*.py")):
        src = f.read_text()
        # strip comments/strings conservatively: match only import statements
        if re.search(r"^\s*(from red_king\b|import red_king\b)", src, re.M):
            offenders.append(f.name)
    assert not offenders, f"red_king leaked into red_queen decision code: {offenders}"


def test_engine_run_has_no_red_king_switch():
    import inspect

    import red_queen.engine as engine

    sig = inspect.signature(engine.run)
    assert "use_red_king" not in sig.parameters
