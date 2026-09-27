"""Supervised purchase-propensity plugin: target, template, cutoffs, inference.

Hermetic (synthetic duckdb/polars only). Locks the contracts that make the
monthly/weekly loop safe: leak-free cutoffs, binary labels, monthly split
rotation, state-consistent features, head persistence/predict, pin rejection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb
import numpy as np
import pytest

from plugins import inference as inference_mod
from plugins.base import Dataset, PluginSpec, load_dataset
from plugins.head_template import HeadTemplate
from plugins.targets import PURCHASE_PROPENSITY_30D, Target, clv_target
from looking_glass.cfm_config import CFMConfig, monthly_split_seed
from looking_glass.cfm_data import _cut_as_of
from looking_glass.cfm_state import new_event_sequences


def _ep(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# P2: monthly split rotation + point-in-time cutoff
# ---------------------------------------------------------------------------
def test_monthly_split_seed_deterministic_and_rotating():
    a = monthly_split_seed(7, "2025-11-01")
    assert a == monthly_split_seed(7, "2025-11-15")  # same month -> same split
    b = monthly_split_seed(7, "2025-12-01")
    assert a != b  # next month -> new split
    assert monthly_split_seed(7, "2025-12-31") == b


def test_cut_as_of_drops_future_events():
    import polars as pl

    df = pl.DataFrame(
        {
            "customer_key": ["c1", "c1", "c2"],
            "event_ts": [
                "2025-10-15T00:00:00+00:00",
                "2025-11-10T00:00:00+00:00",
                "2025-11-20T00:00:00+00:00",
            ],
        }
    )
    cfg = CFMConfig()
    assert _cut_as_of(df, cfg).height == 3  # no as_of -> untouched
    cfg.as_of = "2025-11-01"
    out = _cut_as_of(df, cfg)
    assert out.height == 1
    assert out["event_ts"][0].startswith("2025-10-15")
    with pytest.raises(ValueError):
        CFMConfig(as_of="not-a-date") and _cut_as_of(df, CFMConfig(as_of="not-a-date"))


# ---------------------------------------------------------------------------
# P3: binary labels + closed-window eligibility (no peeking)
# ---------------------------------------------------------------------------
@pytest.fixture
def mini_dbs(tmp_path):
    prod = tmp_path / "cfm_products.duckdb"
    stream = tmp_path / "stream.duckdb"
    pc = duckdb.connect(str(prod))
    for tbl in ("donor_embeddings", "anchor_embeddings"):
        pc.execute(
            f"CREATE TABLE {tbl} (customer_key TEXT, anchor_epoch DOUBLE, "
            "version TEXT, dim INT, embedding FLOAT[])"
        )
        # 09-10 window (09-10,10-10] closed, no order -> y=0
        # 10-01 window (10-01,10-31] closed, order 10-15 -> y=1
        # 10-20 window (10-20,11-19] NOT closed by the cutoff -> dropped
        for e in (_ep(2025, 9, 10), _ep(2025, 10, 1), _ep(2025, 10, 20)):
            pc.execute(f"INSERT INTO {tbl} VALUES ('c1', ?, 'vT', 2, [1.0, 0.0])", [e])
    pc.close()
    sc = duckdb.connect(str(stream))
    sc.execute("CREATE TABLE customer_events (customer_id TEXT, event_ts VARCHAR)")
    sc.execute("CREATE TABLE orders (customer_id TEXT, order_ts VARCHAR, gross_margin DOUBLE)")
    # an event AFTER the cutoff (must not widen eligibility), plus in-window data
    sc.execute("INSERT INTO customer_events VALUES ('c1','2025-11-20T00:00:00+00:00')")
    sc.execute("INSERT INTO orders VALUES ('c1','2025-10-15T00:00:00+00:00', 5.0)")
    sc.execute("INSERT INTO orders VALUES ('c1','2025-11-05T00:00:00+00:00', 9.0)")
    sc.close()
    return prod, stream


def test_load_dataset_binary_labels_closed_window(mini_dbs):
    prod, stream = mini_dbs
    ds = load_dataset(
        30, cfm_products=prod, stream_db=stream, target=PURCHASE_PROPENSITY_30D, as_of="2025-11-01"
    )
    # anchor 10-20 (+30d = 11-19) is NOT closed by the cutoff -> dropped,
    # so the post-cutoff order 11-05 can never leak into a label.
    assert len(ds.y) == 2
    # 09-10 window excludes the 10-15 order (y=0); 10-01 window contains it (y=1)
    assert sorted(ds.y.tolist()) == [0, 1]
    assert ds.meta["kind"] == "binary"
    assert ds.meta["feature_table"] == "donor_embeddings"
    assert ds.meta["encoder_version"] == "vT"
    assert ds.meta["as_of"] == "2025-11-01"


def test_load_dataset_legacy_path_unchanged(mini_dbs):
    """No target / no as_of keeps the original continuous contract."""
    prod, stream = mini_dbs
    ds = load_dataset(30, cfm_products=prod, stream_db=stream)
    assert ds.meta["feature_table"] == "anchor_embeddings"  # default (legacy)
    assert ds.meta["kind"] == "continuous"
    assert ds.y.dtype == np.float64


# ---------------------------------------------------------------------------
# P1: target contract + head template persist/predict
# ---------------------------------------------------------------------------
def test_target_contracts_and_state_consistent_feature_table():
    assert PURCHASE_PROPENSITY_30D.kind == "binary"
    assert PURCHASE_PROPENSITY_30D.window_days == 30
    assert PURCHASE_PROPENSITY_30D.feature_table == "donor_embeddings"  # state-consistent
    assert PURCHASE_PROPENSITY_30D.tag == "supervised_purchase_propensity_30d_v1.0.0r1"
    assert clv_target(365).feature_table == "anchor_embeddings"
    assert PURCHASE_PROPENSITY_30D.spec().kind == "supervised"


def _binary_ds(n=120, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    logit = 1.4 * X[:, 0] - 0.7 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return Dataset(
        spec=PluginSpec(name="t", kind="supervised", target="purchase_in_30d", window_days=30),
        keys=np.repeat(np.arange(n // 3), 3)[:n],
        anchor_epoch=np.zeros(n),
        X=X,
        y=y,
        x_base=rng.integers(0, 2, size=n).astype(np.int64),
        meta={"encoder_version": "vT", "feature_table": "donor_embeddings"},
    )


def test_head_template_persist_predict_roundtrip(tmp_path):
    ds = _binary_ds()
    tpl = HeadTemplate(PURCHASE_PROPENSITY_30D)
    heads, fitted = tpl.fit(ds, seed=0)
    # model bake-off: all families + baseline on one shared split
    assert {h["name"] for h in heads} == {"logistic", "mlp", "hgb", "trailing_baseline"}
    fam_auc = {h["name"]: h["metrics"]["auc"] for h in heads if "fitted" in h}
    assert tpl.primary_head == max(fam_auc, key=fam_auc.get)  # winner = best AUC
    assert fitted["family"] == tpl.primary_head
    # ladder probe narrows the roster to one family
    heads1, _ = tpl.fit(ds, seed=0, families=("logistic",))
    assert {h["name"] for h in heads1} == {"logistic", "trailing_baseline"}
    heads, fitted = tpl.fit(ds, seed=0)  # back to full roster for the checks below
    rows = tpl.gate_rows(heads)
    assert all("check" in r and "ok" in r for r in rows)

    p = tpl.persist(fitted, tmp_path / "head.joblib")
    loaded = HeadTemplate.load(p)
    a = HeadTemplate.predict(loaded, ds.X[:5])
    b = HeadTemplate.predict(fitted, ds.X[:5])
    np.testing.assert_allclose(a, b)
    assert a.shape == (5,)
    assert ((a >= 0) & (a <= 1)).all()


# ---------------------------------------------------------------------------
# P5: live-advance sequence window + pin rejection
# ---------------------------------------------------------------------------
def test_new_event_sequences_only_strictly_new_events():
    import polars as pl

    df = pl.DataFrame(
        {
            "customer_key": ["c1"] * 4,
            "event_ts": [
                "2025-11-01T00:00:00+00:00",
                "2025-11-03T00:00:00+00:00",
                "2025-11-07T00:00:00+00:00",
                "2025-11-09T00:00:00+00:00",
            ],
            "brand": ["b"] * 4,
            "event_type": ["view", "email_send", "order", "view"],
            "event_attributes": ["{}"] * 4,
            "entity_type": ["customer"] * 4,
            "entity_id": ["c1"] * 4,
            "value": [1.0, 0.0, 2.0, 1.0],
        }
    )
    as_of = {"c1": _ep(2025, 11, 3)}
    seqs = new_event_sequences(df, as_of, _ep(2025, 11, 8), ("email_send",), seq_len=128)
    assert len(seqs) == 1
    types = list(seqs[0]["event_type"])
    assert "email_send" not in types  # company action is a covariate, not a token
    assert types == ["order"]  # only 11-07 (strictly after as_of, <= upto)
    assert len(seqs[0]["ts"]) == 1


def test_inference_rejects_encoder_pin_mismatch(tmp_path, monkeypatch):
    import joblib

    # a head manifest pinned to vHEAD, products carrying vPROD
    out = tmp_path / "artifacts"
    (out / "heads").mkdir(parents=True)
    head = {"kind": "binary", "scaler": None, "model": None}
    p = out / "heads" / f"{PURCHASE_PROPENSITY_30D.tag}.joblib"
    joblib.dump(head, p)
    (out / f"{PURCHASE_PROPENSITY_30D.tag}.json").write_text(
        json.dumps(
            {
                "spec": {},
                "tag": PURCHASE_PROPENSITY_30D.tag,
                "payload": {"encoder_version": "vHEAD", "head_path": str(p)},
            }
        )
    )
    products = tmp_path / "products.duckdb"
    con = duckdb.connect(str(products))
    con.execute(
        "CREATE TABLE donor_embeddings (customer_key TEXT, anchor_epoch DOUBLE, "
        "version TEXT, dim INT, embedding FLOAT[])"
    )
    con.execute("INSERT INTO donor_embeddings VALUES ('c1', 0.0, 'vPROD', 2, [1.0, 0.0])")
    con.close()
    monkeypatch.setattr(inference_mod, "OUT", out)
    with pytest.raises(ValueError, match="pin mismatch"):
        inference_mod.score_as_of(PURCHASE_PROPENSITY_30D, "2025-11-08", products=products)


def test_inference_rejects_state_inconsistent_feature_table(tmp_path, monkeypatch):
    import joblib

    out = tmp_path / "artifacts"
    (out / "heads").mkdir(parents=True)
    legacy = Target(
        name="clv_supervised_30d",
        kind="continuous",
        window_days=30,
        spec_target="gross_margin",
        primary_head="two_part",
        feature_table="anchor_embeddings",
    )
    p = out / "heads" / f"{legacy.tag}.joblib"
    joblib.dump({"kind": "continuous_two_part"}, p)
    (out / f"{legacy.tag}.json").write_text(
        json.dumps(
            {
                "spec": {},
                "tag": legacy.tag,
                "payload": {"encoder_version": "vX", "head_path": str(p)},
            }
        )
    )
    monkeypatch.setattr(inference_mod, "OUT", out)
    with pytest.raises(NotImplementedError, match="state-consistent"):
        inference_mod.score_as_of(legacy, "2025-11-08", products=tmp_path / "nope.duckdb")


# ---------------------------------------------------------------------------
# the B->C split: Layer B materializes daily embeddings; (3) only reads them
# ---------------------------------------------------------------------------
def test_state_store_materializes_inference_embeddings(tmp_path):
    import torch

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_state import StateStore

    class StubModel(torch.nn.Module):
        """Minimal device-aware stand-in: the materializer only needs .proj."""

        def __init__(self, dim=4):
            super().__init__()
            self.proj = torch.nn.Linear(dim, dim)

    cfg = CFMConfig()
    model = StubModel(4)
    store = StateStore(tmp_path / "p.duckdb", cfg, model)
    store.upsert("c1", torch.zeros(4), torch.zeros(4), 100.0, None)
    store.upsert("c2", torch.ones(4), torch.ones(4), 200.0, None)

    n = store.materialize_state_embeddings(model, 300.0)
    assert n == 2
    rows = store.con.execute(
        "SELECT customer_key, version FROM state_embeddings WHERE as_of_epoch = 300.0 "
        "ORDER BY customer_key"
    ).fetchall()
    assert [r[0] for r in rows] == ["c1", "c2"]
    assert rows[0][1] == cfg.tag
    # idempotent per day: re-running the daily job rewrites, never duplicates
    store.materialize_state_embeddings(model, 300.0)
    assert (
        store.con.execute(
            "SELECT count(*) FROM state_embeddings WHERE as_of_epoch = 300.0"
        ).fetchone()[0]
        == 2
    )
    store.close()


def test_score_rejects_missing_daily_state_job(tmp_path, monkeypatch):
    import joblib

    out = tmp_path / "artifacts"
    (out / "heads").mkdir(parents=True)
    p = out / "heads" / f"{PURCHASE_PROPENSITY_30D.tag}.joblib"
    joblib.dump({"kind": "binary", "scaler": None, "model": None}, p)
    (out / f"{PURCHASE_PROPENSITY_30D.tag}.json").write_text(
        json.dumps(
            {
                "spec": {},
                "tag": PURCHASE_PROPENSITY_30D.tag,
                "payload": {"encoder_version": "vX", "head_path": str(p)},
            }
        )
    )
    products = tmp_path / "products.duckdb"
    con = duckdb.connect(str(products))
    con.execute(
        "CREATE TABLE donor_embeddings (customer_key TEXT, anchor_epoch DOUBLE, "
        "version TEXT, dim INT, embedding FLOAT[])"
    )
    con.execute("INSERT INTO donor_embeddings VALUES ('c0', 0.0, 'vX', 2, [1.0, 0.0])")
    con.close()
    monkeypatch.setattr(inference_mod, "OUT", out)
    # no state_embeddings table -> Layer-C must reject and name the daily job,
    # never advance state itself
    with pytest.raises(ValueError, match="daily state job"):
        inference_mod.score_as_of(
            PURCHASE_PROPENSITY_30D,
            "2025-11-08",
            products=products,
            scores_db=tmp_path / "scores.duckdb",
        )


def test_score_reads_materialized_embeddings(tmp_path, monkeypatch):
    from looking_glass.cfm_config import _to_epoch

    ds = _binary_ds(n=90, dim=8)
    tpl = HeadTemplate(PURCHASE_PROPENSITY_30D)
    _heads, fitted = tpl.fit(ds, seed=0)
    out = tmp_path / "artifacts"
    (out / "heads").mkdir(parents=True)
    hp = out / "heads" / f"{PURCHASE_PROPENSITY_30D.tag}.joblib"
    tpl.persist(fitted, hp)
    (out / f"{PURCHASE_PROPENSITY_30D.tag}.json").write_text(
        json.dumps(
            {
                "spec": {},
                "tag": PURCHASE_PROPENSITY_30D.tag,
                "payload": {"encoder_version": "vX", "head_path": str(hp)},
            }
        )
    )

    day = "2025-11-08"
    ep = _to_epoch(day)
    products = tmp_path / "products.duckdb"
    con = duckdb.connect(str(products))
    con.execute(
        "CREATE TABLE donor_embeddings (customer_key TEXT, anchor_epoch DOUBLE, "
        "version TEXT, dim INT, embedding FLOAT[])"
    )
    con.execute(
        "CREATE TABLE customer_state (customer_key TEXT, as_of_epoch DOUBLE, "
        "version TEXT, dim INT, state FLOAT[], embedding FLOAT[], last_event_ts VARCHAR)"
    )
    con.execute(
        "CREATE TABLE state_embeddings (customer_key TEXT, as_of_epoch DOUBLE, "
        "version TEXT, dim INT, embedding FLOAT[])"
    )
    con.execute("INSERT INTO donor_embeddings VALUES ('c0', 0.0, 'vX', 8, [0,0,0,0,0,0,0,0])")
    states = [("c0", [0.0] * 4), ("c1", [1.0] * 4)]
    for k, st in states:
        con.execute(f"INSERT INTO customer_state VALUES ('{k}', {ep}, 'vX', 4, {st}, {st}, NULL)")
    emb = {"c0": [0.1] * 8, "c1": [0.9] * 8}
    for k, e in emb.items():
        con.execute(f"INSERT INTO state_embeddings VALUES ('{k}', {ep}, 'vX', 8, {e})")
    con.close()
    monkeypatch.setattr(inference_mod, "OUT", out)

    rec = inference_mod.score_as_of(
        PURCHASE_PROPENSITY_30D, day, products=products, scores_db=tmp_path / "scores.duckdb"
    )
    assert rec["n_scored"] == 2
    assert "read-only" in rec["source"]
    assert rec["encoder_version"] == "vX"
    sc = duckdb.connect(str(tmp_path / "scores.duckdb"), read_only=True)
    assert sc.execute("SELECT count(*) FROM plugin_scores").fetchone()[0] == 2
    assert sc.execute("SELECT count(*) FROM inference_receipts").fetchone()[0] == 1
    sc.close()


def test_daily_state_job_refuses_to_move_backwards(tmp_path):
    from looking_glass.daily_states import assert_forward_only

    products = tmp_path / "p.duckdb"
    con = duckdb.connect(str(products))
    con.execute(
        "CREATE TABLE customer_state (customer_key TEXT, as_of_epoch DOUBLE, "
        "version TEXT, dim INT, state FLOAT[], embedding FLOAT[], last_event_ts VARCHAR)"
    )
    con.execute(
        "INSERT INTO customer_state VALUES ('c1', 2000000000.0, 'vX', 2, [1.0], [1.0], NULL)"
    )
    con.close()
    # forward / same day: allowed (2e9 ~ 2033)
    assert_forward_only(products, 2000000000.0)
    assert_forward_only(products, 2000008640.0)
    # past day: rejected — states would be relabeled backwards
    with pytest.raises(ValueError, match="forward only"):
        assert_forward_only(products, 1999900000.0)


# ---------------------------------------------------------------------------
# population vs sample (compute/signal knobs) + model-family pin
# ---------------------------------------------------------------------------
def test_populations_disjoint_and_samples_deterministic():
    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_data import assign_split, draw_sample

    cfg = CFMConfig(sample_customers=None)
    keys = [f"c{i}" for i in range(1000)]
    split = assign_split(keys, cfg)  # populations: one draw, two pools
    popA = {k for k in keys if split[k] == "A"}
    popB = {k for k in keys if split[k] == "B"}
    assert popA.isdisjoint(popB) and popA | popB == set(keys)

    sA = draw_sample(keys, split, "A", 80, cfg.split_seed)
    sB = draw_sample(keys, split, "B", 50, cfg.split_seed)
    # samples respect their sizes and can never cross populations
    assert len(sA) == 80 and len(sB) == 50
    assert set(sA) <= popA and set(sB) <= popB
    assert set(sA).isdisjoint(sB)
    # deterministic for a given month; n=None/oversized -> whole population
    assert draw_sample(keys, split, "A", 80, cfg.split_seed) == sA
    assert len(draw_sample(keys, split, "A", None, cfg.split_seed)) == len(popA)
    assert len(draw_sample(keys, split, "A", 10**6, cfg.split_seed)) == len(popA)


def test_target_family_pin_contract():
    # production default: bake-off picks the winner by evidence each run
    assert PURCHASE_PROPENSITY_30D.family == ""
    # pinning is just a Target field (e.g. "mlp" once evidence favors it)
    pinned = Target(name="t", kind="binary", window_days=30, spec_target="x", family="mlp")
    assert pinned.family == "mlp"
