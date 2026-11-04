"""The products build must SERVE what it claims to serve (DEC-429).

`state_embeddings` is the table plugins are told to read. It shipped **empty** for six decisions
because the day-1 close was guarded on `if state_rows:` — a build buffer that the memory fix a
few lines earlier had already cleared — and no print or gate ever read that table's count. The
build exited 0 and looked healthy.

Three properties pin the defect class, and none of them is the number itself:
  * clearing a buffer must not silently disable a DOWNSTREAM CONSUMER — so the guard asks the
    stored table, and a real `build_products` must leave a non-empty served table;
  * a count nobody prints is a count nobody checks — so the release prints both counts and
    REFUSES when they disagree, rather than letting an empty table become a consumer's problem;
  * non-empty is not the same as correct — the served column must actually be a readout of the
    state (and a stable one: the same past produces the same donor).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from looking_glass.cfm_config import CFMConfig  # noqa: E402
from looking_glass.cfm_data import build_sequences  # noqa: E402
from looking_glass.cfm_model import CFM, EventVocab  # noqa: E402
from looking_glass.cfm_state import StateStore, build_products  # noqa: E402


def _stream(n_customers=6, n_events=5):
    """A tiny stream in the canonical shape the loader consumes.

    Each customer sees a DIFFERENT window of the event-type cycle, so their histories are
    genuinely distinguishable — a test that demanded "all embeddings differ" over identical
    histories would be demanding a defect (see `test_identical_histories_land_on_the_same_point`).
    """
    n_types = n_customers + 2
    keys = [f"c{i}" for i in range(n_customers) for _ in range(n_events)]
    ets = [
        f"2025-01-{1 + j:02d}T00:00:00+00:00" for _ in range(n_customers) for j in range(n_events)
    ]
    types = [f"e{(i + j) % n_types}" for i in range(n_customers) for j in range(n_events)]
    return pl.DataFrame(
        {
            "customer_key": keys,
            "event_ts": ets,
            "event_type": types,
            "brand": [None] * len(keys),
            "entity_type": ["customer"] * len(keys),
            "entity_id": keys,
            "value": [1.0] * len(keys),
        }
    ).sort(["customer_key", "event_ts"])


def _build(df, tmp_path, *, split=None, dim=16):
    """One real `build_products` over `df`, then reopen the artifact for reading."""
    cfg = CFMConfig()
    cfg.out_dir = str(tmp_path)
    keys = sorted({k for k in df["customer_key"].to_list()})
    split = split or {k: ("A" if i % 2 == 0 else "B") for i, k in enumerate(keys)}
    torch.manual_seed(0)
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
    model = CFM(EventVocab.build(seqs), dim=dim, n_experts=1)
    n_states, n_anchors = build_products(cfg, model, model.vocab, df, keys, split)
    con = duckdb.connect(str(Path(cfg.out_dir) / "cfm_products.duckdb"), read_only=True)
    return con, cfg, n_states, n_anchors


@pytest.fixture
def built(tmp_path):
    con, _cfg, n_states, n_anchors = _build(_stream(), tmp_path)
    try:
        yield con, n_states, n_anchors
    finally:
        con.close()  # the fixture closes what the fixture opened


def test_a_products_build_leaves_a_non_empty_served_state_table(built):
    """The regression itself: `customer_state` populated while `state_embeddings` is empty meant
    every plugin reading the served donor got nothing, and the build still exited 0."""
    con, n_states, _n_anchors = built
    served = con.execute("SELECT count(*) FROM state_embeddings").fetchone()[0]
    assert n_states == 6, n_states
    assert served == n_states, (
        f"customer_state={n_states} but state_embeddings={served}: the day-1 close did not run "
        "(it must be guarded on the stored table, never on a build buffer)"
    )
    # and the served rows are the SAME customers, not an arbitrary subset
    missing = con.execute(
        "SELECT count(*) FROM customer_state c LEFT JOIN state_embeddings e USING (customer_key) "
        "WHERE e.customer_key IS NULL"
    ).fetchone()[0]
    assert missing == 0, f"{missing} customers have a state but no served embedding"


def test_served_embeddings_are_a_real_readout_of_the_state(built):
    """Non-empty is not enough — the column must be a fixed-width, finite readout that actually
    varies with the customer. A constant column would pass every count check above."""
    import numpy as np

    con, _n, _a = built
    rows = con.execute(
        "SELECT customer_key, dim, embedding FROM state_embeddings ORDER BY customer_key"
    ).fetchall()
    assert rows
    assert {r[1] for r in rows} == {rows[0][1]}, "ragged served widths"
    E = np.asarray([r[2] for r in rows], dtype=np.float64)
    assert np.isfinite(E).all()
    # Explicit statistic blocks may legitimately be constant on a tiny fixture (e.g. no
    # negative events -> n_neg is all zero). The readout must still vary overall, and the
    # unique-row assertion below is what rules out a collapsed donor.
    assert E.std(0).max() > 0, "the served readout is constant in every dimension"
    assert len({tuple(row) for row in E}) == len(E), (
        "the readout collapsed every customer to one point"
    )


def test_identical_histories_land_on_the_same_point(tmp_path):
    """The other side of that coin, so the assertion above cannot be satisfied by noise: the same
    past must produce the same donor. A failure here means the readout depends on something other
    than the stream (an identity leak, or states bleeding across customers in the batched path)."""
    df = _stream(n_customers=4)
    twin = df.filter(pl.col("customer_key") == "c0").with_columns(
        pl.lit("twin").alias("customer_key")
    )
    con, _cfg, _s, _a = _build(pl.concat([df, twin]), tmp_path)
    try:
        got = dict(
            con.execute(
                "SELECT customer_key, embedding FROM state_embeddings "
                "WHERE customer_key IN ('c0', 'twin')"
            ).fetchall()
        )
        assert set(got) == {"c0", "twin"}, f"the twin customer was not served: {sorted(got)}"
        assert got["c0"] == pytest.approx(got["twin"]), (
            "the same past produced two different donors"
        )
    finally:
        con.close()


def test_a_release_that_serves_nothing_is_refused_not_printed(tmp_path):
    """Fail safe (P7): the two counts can disagree only through a defect, so the release refuses.
    Before this, the mismatch was simply invisible — no print, no gate, no nonzero exit."""
    store = StateStore(tmp_path / "v.duckdb", CFMConfig(), None)
    try:
        with pytest.raises(RuntimeError, match="EMPTY"):
            store.verify_state_serving(n_states=5, n_embeddings=0, close_at=1.0)
        with pytest.raises(RuntimeError, match="lost customers"):
            store.verify_state_serving(n_states=5, n_embeddings=4, close_at=1.0)
        # a build with no states at all is not a failure — there was nothing to serve
        store.verify_state_serving(n_states=0, n_embeddings=0, close_at=1.0)
    finally:
        store.close()


def test_the_served_counts_are_printed_every_build(tmp_path, capsys):
    """Observability (P11): the reason this survived six decisions is that the number was never
    printed. A receipt that lives only in a table nobody queries is not a receipt."""
    store = StateStore(tmp_path / "p.duckdb", CFMConfig(), None)
    try:
        store.verify_state_serving(n_states=7, n_embeddings=7, close_at=123.0)
    finally:
        store.close()
    out = capsys.readouterr().out
    assert "customer_state=7" in out and "state_embeddings=7" in out
