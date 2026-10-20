"""CFM state operations (fade/absorb) and the DuckDB state store."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from looking_glass.cfm_config import CFMConfig, LN2, _to_epoch, as_of_epoch
from looking_glass.cfm_data import build_sequences
from looking_glass.cfm_model import CFM


# ---------------------------------------------------------------------------
# state operations (fade / absorb)
# ---------------------------------------------------------------------------
def fade(h: torch.Tensor, dt_seconds: float, half_life_days: float) -> torch.Tensor:
    """Decay the state when time passes with NO events."""
    if dt_seconds <= 0:
        return h
    decay = math.exp(-LN2 * dt_seconds / max(half_life_days * 86400.0, 1.0))
    return h * decay


def absorb(model: CFM, seq, h0: torch.Tensor | None = None, as_of_epoch: float | None = None):
    """Push new events through the recurrence from an existing state.

    Fades h0 from as_of to the first new event, then runs only the new events.
    Returns (new_h, embedding).
    """
    with torch.no_grad():
        h = h0
        if h is not None:
            h = h.to(next(model.parameters()).device)  # store rows are CPU
        if h is not None and as_of_epoch is not None and len(seq["event_ts"]):
            h = fade(
                h,
                _to_epoch(seq["event_ts"][0]) - as_of_epoch,
                getattr(model, "half_life_days", 30.0),
            )
        y, h = model(seq, h0=h)
        return h, model.embed(h)


def new_event_sequences(
    df,
    as_of_by_key,
    upto_epoch,
    company,
    seq_len,
    action_fields=(),
    attr_fields=(),
    numeric_fields=(),
):
    """Sequences of ONLY the events in ``(as_of_by_key[k], upto_epoch]``.

    The live-advance primitive: the stored state already covers each customer's
    history up to its ``as_of_epoch``, so only that new tail may be absorbed —
    feeding full history would double-count it. ALL events are tokens (actions
    included as INPUT channels), matching training's build_sequences.
    """
    import numpy as np
    import polars as pl

    from looking_glass.cfm_data import _action_flags, _covariates, _primary_numeric_field

    if not as_of_by_key:
        return []
    company = set(map(str, company))
    d = (
        df.with_columns(
            pl.col("event_ts")
            .str.to_datetime(time_zone="UTC", strict=False)
            .dt.epoch("s")
            .alias("_ts")
        )
        .filter(pl.col("_ts") <= float(upto_epoch))
        .join(
            pl.DataFrame(
                {
                    "customer_key": list(as_of_by_key),
                    "as_of_epoch": [float(v) for v in as_of_by_key.values()],
                }
            ),
            on="customer_key",
            how="inner",
        )
        .filter(pl.col("_ts") > pl.col("as_of_epoch"))
        .sort(["customer_key", "_ts"])
    )

    _F = SimpleNamespace(exogenous_events=tuple(company), action_fields=tuple(action_fields))

    value_col = _primary_numeric_field(d, None)
    seqs = []
    for g in d.partition_by("customer_key", maintain_order=True):
        et_arr = g["event_type"].to_numpy()
        ts_all = g["_ts"].to_numpy()
        is_action = _action_flags(g, _F)
        co = _covariates(et_arr, ts_all, company, is_action=is_action)
        n = et_arr.size
        if n < 1:
            continue
        ki = np.arange(max(0, n - int(seq_len)), n)
        L = ki.size

        def _col(name, default):
            return (
                g[name].to_numpy()
                if name in g.columns
                else np.full(g.height, default, dtype=object)
            )

        if value_col is not None and value_col in g.columns:
            val = g[value_col].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        else:
            val = np.zeros(g.height)
        seqs.append(
            {
                "customer": g["customer_key"][0],
                "group": "B",
                "anchor_epoch": None,
                "event_type": et_arr[ki],
                "attrs": {c: _col(c, "none")[ki] for c in (attr_fields or ())},
                "numeric": {
                    c: (
                        g[c].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()[ki]
                        if c in g.columns
                        else np.zeros(L)
                    )
                    for c in (numeric_fields or ())
                },
                "numeric_mask": {
                    c: (
                        g[c].is_not_null().to_numpy()[ki]
                        if c in g.columns
                        else np.zeros(L, dtype=bool)
                    )
                    for c in (numeric_fields or ())
                },
                "is_action": is_action[ki],
                "value": val[ki],
                "primary_numeric": value_col,
                "event_ts": g["event_ts"].to_numpy()[ki],
                "ts": ts_all[ki].tolist(),
                "co": co[ki].tolist(),
            }
        )
    return seqs


def advance_to_date(store, cfg, df, upto_epoch):
    """Advance every stored state to ``upto_epoch`` (the daily-inference step).

    Customers with new events get fade->absorb; the remainder are faded
    (decay-only). Returns ``(n_with_events, n_idle)``.
    """
    rows = store.con.execute("SELECT customer_key, as_of_epoch FROM customer_state").fetchall()
    as_of_by_key = {k: a for k, a in rows}
    seqs = new_event_sequences(
        df,
        as_of_by_key,
        upto_epoch,
        getattr(cfg, "exogenous_events", ()),
        cfg.seq_len,
        action_fields=tuple(getattr(cfg, "action_fields", ()) or ()),
        attr_fields=tuple(getattr(cfg, "attr_fields", ()) or ()),
        numeric_fields=tuple(getattr(cfg, "numeric_fields", ()) or ()),
    )
    if seqs:
        store.advance(seqs, incremental=True)
    with_events = {s["customer"] for s in seqs}
    store.fade_idle(float(upto_epoch))
    return len(with_events), len(as_of_by_key) - len(with_events)


def donor_states(model: CFM, states):
    """Batched ``donor(h) = proj(h)`` for live states (the exact readout written
    into ``donor_embeddings`` at training time, so heads transfer). Device-safe:
    follows wherever the model lives (cuda when the job moved it there)."""
    with torch.no_grad():
        t = torch.as_tensor(states, dtype=torch.float32)
        return model.proj(t.to(next(model.parameters()).device)).cpu().numpy()


def load_frozen_encoder(tag: str, cfm_dir):
    """Load a frozen CFM checkpoint + its registry-resolved config (Layer B)."""
    import json
    from pathlib import Path as _P

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_model import CFM, EventVocab

    cfm_dir = _P(cfm_dir)
    ckpt = cfm_dir / f"cfm_{tag.replace('.', '_')}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"encoder checkpoint for pin {tag} not found: {ckpt}")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    _v = blob["vocab"]
    vocab = EventVocab(
        _v["et"],
        _v.get("attrs"),
        numeric_fields=tuple(_v.get("numeric_fields", ()) or ()),
        numeric_stats=_v.get("numeric_stats"),
        primary_numeric=_v.get("primary_numeric") or blob.get("primary_numeric"),
    )
    model = CFM(
        vocab,
        blob["dim"],
        n_experts=blob.get("n_experts", 1),
        sf_mode=blob.get("sf_mode", "event_types"),
        slow_intent=bool(blob.get("slow_intent", False)),
        unified=bool(blob.get("unified", False)),
        zca=bool(blob.get("zca", False)),
        isometric_boundary=bool(blob.get("isometric_boundary", False)),
        bilinear=bool(blob.get("bilinear_recurrence", False)),
        readout_norm=bool(blob.get("readout_norm", False)),
        input_norm=bool(blob.get("input_norm", False)),
        readout_skip=bool(blob.get("readout_skip", False)),
        event_residual=bool(blob.get("event_residual", False)),
        readout_scale=bool(blob.get("readout_scale", False)),
        stats_pathway=bool(blob.get("stats_pathway", False)),
        continuous_time=bool(blob.get("continuous_time", False)),
        counter=bool(blob.get("counter", False)),
        counter_registers=int(blob.get("counter_registers", 1)),
        counter_compact=bool(blob.get("counter_compact", False)),
        counter_event_types=bool(blob.get("counter_event_types", False)),
        numeric_fields=tuple(blob.get("numeric_fields", ()) or ()),
        action_fields=tuple(blob.get("action_fields", ()) or ()),
        effect_dim=int(blob.get("effect_dim", 0) or 0),
    )
    # tolerate checkpoints from earlier revisions (e.g. a head added later):
    # load what matches, report what did not. The encoder trunk must match.
    missing, unexpected = model.load_state_dict(blob["state"], strict=False)
    if missing or unexpected:
        print(f"[load] {tag}: missing={len(missing)} unexpected={len(unexpected)} keys", flush=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # The served `E` = shape (donor) || learned-compressed measure. The in-net ACCUMULATOR is
    # the hand-RFM BASELINE (used only by the B card's comparison), NOT part of the pipeline —
    # keeping it ON bloated E to ~2141 dims (1803 were accumulator) and slowed every gate. The
    # B card toggles it explicitly for its "hand features" row.
    model.accumulator_on = False
    if blob.get("whiten_W") is not None:
        model.set_whitening(blob["whiten_mean"], blob["whiten_W"])  # DEC-022
    cfg = CFMConfig()
    cfg.sf_mode = blob.get("sf_mode", "event_types")  # checkpoint truth (phi shape)
    # RL leg: the declared action keys must be restored so the served measure rebuilds with
    # the same action-conditioned response block the Layer-B compression was fit on.
    cfg.effect_actions = tuple(blob.get("effect_actions", ()) or ())
    cfg.action_fields = tuple(blob.get("action_fields", ()) or ())
    cfg.numeric_fields = tuple(vocab.numeric_fields)
    cfg.attr_fields = tuple(vocab.attrs)
    reg = cfm_dir / f"registry_{tag.replace('.', '_')}.json"
    if reg.exists():
        meta = json.loads(reg.read_text())
        cfg.version = meta.get("version", cfg.version)
        cfg.revision = int(meta.get("revision", cfg.revision))
        # per-stream geometry capacity (v4.4, DEC-034): the stream's own
        # achievable whitened participation ratio, used to calibrate the geometry
        # floor instead of a cross-stream constant.
        wh = meta.get("whiten") or {}
        if wh.get("pr_after") is not None:
            cfg._whiten_pr_after = float(wh["pr_after"])
        # training stability (v6): a divergent run is NOT certifiable.
        cfg._stable = not bool((meta.get("governor") or {}).get("diverged", False))
        resolved = meta.get("resolved", {})
        cfg.seq_len = int(resolved.get("seq_len", cfg.seq_len))
        cfg.state_half_life_days = float(resolved.get("half_life_days", cfg.state_half_life_days))
        db = meta.get("db")
        if db and _P(db).exists():
            cfg.db = db
        else:
            from looking_glass.cfm_config import STREAM_TABLE  # noqa: F401

            cfg.db = str(
                _P(__file__).resolve().parents[2]
                / "rabbit_hole"
                / "data"
                / "duckdb"
                / "customer_event_stream.parquet"
            )
    else:
        cfg.db = str(
            _P(__file__).resolve().parents[2]
            / "rabbit_hole"
            / "data"
            / "duckdb"
            / "customer_event_stream.parquet"
        )
    model.half_life_days = cfg.state_half_life_days
    # Layer-B MEASURE pipeline: load the learned measure compression if it exists.
    _lb_path = cfm_dir / f"layer_b_{tag.replace('.', '_')}.pkl"
    if _lb_path.exists():
        from looking_glass.layer_b import LayerBEmbedder

        model.layer_b = LayerBEmbedder.load(_lb_path)
        cfg.use_measure = True
        # Restore the cadence the response block was fit on (train/serve parity).
        _aps = getattr(model.layer_b, "action_period_seconds", None)
        if _aps is not None:
            cfg.action_period_seconds = float(_aps)
    # CERTIFIED MINIMAL SERVE PROJECTION (DEC-213): if fitted, the consumed E is
    # the frozen projection to its minimal sufficient rank.
    _proj = cfm_dir / f"serve_proj_{tag.replace('.', '_')}.npz"
    if _proj.exists():
        import numpy as _np

        _z = _np.load(_proj)
        if str(_z.get("mode", "pca")) == "blocks":
            d = int(_z["shape_dim"])
            model.serve_total_dim = int(_z.get("total_dim", d))
            model.set_serve_blocks(d, _z["mean"], _z["scale"], _z["W"])
            print(
                f"[load] block-wise serve: shape->{int(_z['W'].shape[0])} + measure "
                f"-> {int(model.serve_total_dim) - d} dims",
                flush=True,
            )
        else:
            model.set_serve_projection(_z["mean"], _z["scale"], _z["W"])
            print(f"[load] serve projection: E -> {int(_z['W'].shape[0])} dims", flush=True)
    return model, cfg


# ---------------------------------------------------------------------------
# state store (DuckDB): customer_state + anchor_embeddings + donor_embeddings
# ---------------------------------------------------------------------------
class StateStore:
    def __init__(self, path, cfg: CFMConfig, model: CFM):
        import duckdb

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.cfg = cfg
        self.model = model
        self.con = duckdb.connect(self.path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS customer_state ("
            "customer_key TEXT, as_of_epoch DOUBLE, version TEXT, dim INT, "
            "state FLOAT[], embedding FLOAT[], last_event_ts TEXT)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS anchor_embeddings ("
            "customer_key TEXT, anchor_epoch DOUBLE, version TEXT, dim INT, embedding FLOAT[])"
        )
        # state-consistent readout: donor(h) = proj(h) (no entity pooling, no
        # normalization). It is a PURE FUNCTION OF THE STATE, so a head trained
        # on this table can be scored against live advanced states.
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS donor_embeddings ("
            "customer_key TEXT, anchor_epoch DOUBLE, version TEXT, dim INT, embedding FLOAT[])"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS encoder_samples ("
            "customer_key TEXT, split TEXT, version TEXT)"
        )
        # DAILY INFERENCE EMBEDDINGS: materialized by the Layer-B daily state
        # job from customer_state (donor(h)). Plugin inference (Layer C) is a
        # pure READ of this table — it never writes state or embeddings.
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS state_embeddings ("
            "customer_key TEXT, as_of_epoch DOUBLE, version TEXT, dim INT, embedding FLOAT[])"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS state_job_receipts ("
            "as_of_epoch DOUBLE, version TEXT, n_absorbed BIGINT, n_idle BIGINT, "
            "n_embeddings BIGINT, device TEXT, wall_seconds DOUBLE, ran_at TIMESTAMP)"
        )

    def record_receipt(self, epoch, n_absorbed, n_idle, n_embeddings, device, wall_seconds):
        """One insert path for state_job_receipts (daily job AND the encoder's
        day-1 close both land here — single source of truth for the schema)."""
        from datetime import datetime, timezone

        self.con.execute(
            "INSERT INTO state_job_receipts VALUES (?,?,?,?,?,?,?,?)",
            [
                float(epoch),
                self.cfg.tag,
                int(n_absorbed),
                int(n_idle),
                int(n_embeddings),
                device,
                round(float(wall_seconds), 2),
                datetime.now(timezone.utc),
            ],
        )

    def write_splits(self, keys, split):
        """Persist the sample-A/B assignment (strict disjoint) as a reusable table."""
        import polars as pl

        self.con.execute(
            "CREATE OR REPLACE TABLE encoder_samples (customer_key TEXT, split TEXT, version TEXT)"
        )
        df = pl.DataFrame(
            {
                "customer_key": list(keys),
                "split": [split[k] for k in keys],
                "version": [self.cfg.tag] * len(keys),
            }
        )
        self.con.register("_splits", df)
        try:
            self.con.execute(
                "INSERT INTO encoder_samples SELECT customer_key, split, version FROM _splits"
            )
        finally:
            self.con.unregister("_splits")

    def get_state(self, key):
        row = self.con.execute(
            "SELECT as_of_epoch, state FROM customer_state WHERE customer_key=? "
            "ORDER BY as_of_epoch DESC LIMIT 1",
            [key],
        ).fetchone()
        if not row or row[1] is None:
            return None, None
        return torch.tensor(row[1], dtype=torch.float32), float(row[0])

    def upsert(self, key, h, emb, as_of_epoch, last_event_ts):
        self.con.execute("DELETE FROM customer_state WHERE customer_key=?", [key])
        self.con.execute(
            "INSERT INTO customer_state VALUES (?,?,?,?,?,?,?)",
            [key, as_of_epoch, self.cfg.tag, len(h), h.tolist(), emb.tolist(), last_event_ts],
        )

    def advance(self, seqs, incremental=True):
        """fade to each customer's first new event, absorb the events, persist.

        BATCHED: sequences are grouped into same-length buckets (zero padding)
        and each bucket runs ONE padded forward — the daily job is built for
        this scale. Upserts are bulk DELETE+INSERT. Same math as the per-customer
        fade->absorb path (covered by an equivalence test).
        """
        from looking_glass.cfm_training import forward_states

        if not seqs:
            return
        # ONE query for all stored states (no per-customer round-trips)
        state_by_key = {}
        if incremental:
            keys = [s["customer"] for s in seqs]
            for k0 in range(0, len(keys), 4000):
                chunk = keys[k0 : k0 + 4000]
                ph = ",".join("?" * len(chunk))
                for ck, as_of, st in self.con.execute(
                    f"SELECT customer_key, as_of_epoch, state FROM customer_state "
                    f"WHERE customer_key IN ({ph})",
                    chunk,
                ).fetchall():
                    state_by_key[ck] = (
                        torch.tensor(st, dtype=torch.float32) if st is not None else None,
                        float(as_of) if as_of is not None else None,
                    )
        prepared = []  # (key, seq, h0_or_None, as_of_epoch, last_event_ts)
        for seq in seqs:
            if incremental:
                h0, as_of = state_by_key.get(seq["customer"], (None, None))
            else:
                h0, as_of = None, None
            if h0 is not None:
                # absorb()'s pre-fade: decay h0 to the first new event
                h0 = fade(h0, _to_epoch(seq["event_ts"][0]) - as_of, self.cfg.state_half_life_days)
            prepared.append(
                (
                    seq["customer"],
                    seq,
                    h0,
                    _to_epoch(seq["event_ts"][-1]),
                    seq["event_ts"][-1],
                )
            )
        # same-length buckets -> no padding waste; one forward per bucket
        prepared.sort(key=lambda x: len(x[1]["event_type"]))
        rows = []
        i = 0
        while i < len(prepared):
            j = i
            L = len(prepared[i][1]["event_type"])
            while j < len(prepared) and len(prepared[j][1]["event_type"]) == L:
                j += 1
            bucket = prepared[i:j]
            with_state = [b for b in bucket if b[2] is not None]
            without = [b for b in bucket if b[2] is None]
            for part, use_h0 in ((with_state, True), (without, False)):
                if not part:
                    continue
                H = forward_states(
                    self.model,
                    [b[1] for b in part],
                    [b[2] for b in part] if use_h0 else None,
                )
                with torch.no_grad():
                    E = torch.nn.functional.normalize(self.model.proj(H), dim=1).cpu()
                for k, (key, _seq, _h0, as_of_e, last_ts) in enumerate(part):
                    h = H[k].cpu()
                    rows.append(
                        {
                            "customer_key": key,
                            "as_of_epoch": as_of_e,
                            "version": self.cfg.tag,
                            "dim": int(h.shape[0]),
                            "state": h.tolist(),
                            "embedding": E[k].tolist(),
                            "last_event_ts": str(last_ts),
                        }
                    )
            i = j
        self.replace_states(rows)

    def replace_states(self, rows) -> None:
        """Bulk upsert: DELETE the touched keys, INSERT the new rows."""
        if not rows:
            return
        import polars as pl

        keys = sorted({r["customer_key"] for r in rows})
        for k0 in range(0, len(keys), 500):
            chunk = keys[k0 : k0 + 500]
            ph = ",".join("?" * len(chunk))
            self.con.execute(f"DELETE FROM customer_state WHERE customer_key IN ({ph})", chunk)
        df = pl.DataFrame(rows)
        self.con.register("_rs", df)
        try:
            self.con.execute("INSERT INTO customer_state SELECT * FROM _rs")
        finally:
            self.con.unregister("_rs")

    def add_training(self, key, anchor_epoch, emb):
        self.con.execute(
            "INSERT INTO anchor_embeddings VALUES (?,?,?,?,?)",
            [key, anchor_epoch, self.cfg.tag, len(emb), emb.tolist()],
        )

    def add_donor(self, key, anchor_epoch, emb):
        self.con.execute(
            "INSERT INTO donor_embeddings VALUES (?,?,?,?,?)",
            [key, anchor_epoch, self.cfg.tag, len(emb), emb.tolist()],
        )

    def fade_idle(self, now_epoch):
        """Advance every idle state to now_epoch (decay only; no events).

        Batched: one projection over all states + a bulk rewrite, so a 25k-customer
        store advances in seconds rather than per-row round-trips.
        """
        import polars as pl

        rows = self.con.execute(
            "SELECT customer_key, as_of_epoch, state FROM customer_state"
        ).fetchall()
        if not rows:
            return
        keys = [r[0] for r in rows]
        H = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        dt = torch.tensor(
            [float(now_epoch) - float(r[1]) for r in rows], dtype=torch.float32
        ).unsqueeze(1)
        decay = torch.exp(-LN2 * dt / max(self.cfg.state_half_life_days * 86400.0, 1.0)).clamp(
            max=1.0
        )  # dt <= 0 -> identity, matching fade()
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            H = H * decay
            E = torch.nn.functional.normalize(self.model.proj(H.to(dev)), dim=1).cpu()
        df = pl.DataFrame(
            {
                "customer_key": keys,
                "as_of_epoch": [float(now_epoch)] * len(keys),
                "version": [self.cfg.tag] * len(keys),
                "dim": [int(H.shape[1])] * len(keys),
                "state": H.tolist(),
                "embedding": E.tolist(),
                "last_event_ts": [None] * len(keys),
            }
        )
        self.con.execute("DELETE FROM customer_state")
        self.con.register("_idle", df)
        try:
            self.con.execute("INSERT INTO customer_state SELECT * FROM _idle")
        finally:
            self.con.unregister("_idle")

    def materialize_state_embeddings(self, model, as_of_epoch):
        """Project every live state to its inference embedding for this day.

        Idempotent per day (delete + rewrite), so the daily job is re-runnable.
        Returns the number of rows written.
        """
        import polars as pl

        rows = self.con.execute("SELECT customer_key, state FROM customer_state").fetchall()
        if not rows:
            return 0
        keys = [r[0] for r in rows]
        E = donor_states(model, [r[1] for r in rows])
        self.con.execute("DELETE FROM state_embeddings WHERE as_of_epoch = ?", [float(as_of_epoch)])
        df = pl.DataFrame(
            {
                "customer_key": keys,
                "as_of_epoch": [float(as_of_epoch)] * len(keys),
                "version": [self.cfg.tag] * len(keys),
                "dim": [int(E.shape[1])] * len(keys),
                "embedding": E.tolist(),
            }
        )
        self.con.register("_se", df)
        try:
            self.con.execute("INSERT INTO state_embeddings SELECT * FROM _se")
        finally:
            self.con.unregister("_se")
        return len(keys)

    def bulk_states(self, rows) -> None:
        """Initial-build fast path: write many customer_state rows in ONE insert
        (single-row commits dominated the build: ~15-20min -> seconds)."""
        import polars as pl

        if not rows:
            return
        df = pl.DataFrame(rows)
        self.con.register("_bs", df)
        try:
            self.con.execute("INSERT INTO customer_state SELECT * FROM _bs")
        finally:
            self.con.unregister("_bs")

    def bulk_anchor_rows(self, anchor_rows, donor_rows) -> None:
        """Initial-build fast path for anchor_embeddings + donor_embeddings."""
        import polars as pl

        for table, rows in (("anchor_embeddings", anchor_rows), ("donor_embeddings", donor_rows)):
            if not rows:
                continue
            df = pl.DataFrame(rows)
            self.con.register("_br", df)
            try:
                self.con.execute(f"INSERT INTO {table} SELECT * FROM _br")
            finally:
                self.con.unregister("_br")

    def count(self):
        s = self.con.execute("SELECT count(*) FROM customer_state").fetchone()[0]
        t = self.con.execute("SELECT count(*) FROM anchor_embeddings").fetchone()[0]
        return s, t

    def close(self):
        self.con.close()


def build_products(cfg, model, vocab, df, keys, split):
    # Rebuild fresh each release so embedding dims never mix across runs.
    pdb = Path(cfg.out_dir) / "cfm_products.duckdb"
    if pdb.exists():
        pdb.unlink()
    store = StateStore(pdb, cfg, model)
    store.write_splits(keys, split)  # persist the POPULATIONS (the truth)
    # inference states for every customer (full history, as_of = last event):
    # run the recurrence per customer, WRITE IN BULK (row-at-a-time commits
    # dominated this phase; identical rows, one insert)
    state_rows = []
    # min_events=1: states exist for every customer with at least one customer
    # event (the blanket 3 silently starved long-tail customers of states).
    for seq in build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1):
        with torch.no_grad():
            _y, h = model(seq)
            emb = model.embed(h)
        last_ts = seq["event_ts"][-1]
        state_rows.append(
            {
                "customer_key": seq["customer"],
                "as_of_epoch": float(_to_epoch(last_ts)),
                "version": cfg.tag,
                "dim": int(h.shape[0]),
                "state": h.tolist(),
                "embedding": emb.tolist(),
                "last_event_ts": str(last_ts),
            }
        )
    store.bulk_states(state_rows)
    # (1) also CLOSES day 1: fade every state to the as_of boundary (UTC day
    # start = all of yesterday absorbed, today excluded) and materialize that
    # day's inference embeddings. The daily job therefore only handles days >= 2
    # — no redundant day-1 absorb/fade/materialize work.
    if state_rows:
        import time as _time

        t_close = _time.perf_counter()
        close_at = (
            as_of_epoch(cfg.as_of)
            if getattr(cfg, "as_of", None)
            else float(
                store.con.execute("SELECT max(as_of_epoch) FROM customer_state").fetchone()[0]
            )
        )
        store.fade_idle(close_at)
        n_emb = store.materialize_state_embeddings(model, close_at)
        # no absorbs (states were just built from full history); every customer
        # was faded idly to the boundary, so idle == embeddings
        store.record_receipt(close_at, 0, n_emb, n_emb, "encoder", _time.perf_counter() - t_close)
    # training embeddings: Population B SAMPLE at anchors (states stay for everyone)
    anchor_split = split
    if cfg.sample_b_customers is not None:
        from looking_glass.cfm_data import draw_sample

        b_sample = set(draw_sample(keys, split, "B", cfg.sample_b_customers, cfg.split_seed))
        anchor_split = {k: ("B" if (split[k] == "B" and k in b_sample) else "A") for k in keys}
        print(f"[sample] plugins train on {len(b_sample)} of population B", flush=True)
    anchor_rows, donor_rows = [], []
    for seq in build_sequences(df, keys, cfg, anchor_split, with_anchors=True):
        if seq["group"] == "B" and seq["anchor_epoch"] is not None:
            with torch.no_grad():
                y, h = model(seq)
                # train/serve parity — ONE readout rule: serving fades the
                # stored state from its last event to the scoring boundary, so
                # the training-anchor readout must fade the same way to the
                # anchor. (Un-faded-in-training / faded-in-serving is a silent
                # train/serve skew; on real data it cost served AUC.)
                gap = float(seq["anchor_epoch"]) - float(seq["ts"][-1])
                h_read = fade(h, gap, cfg.state_half_life_days)
                dsq = model.donor_seq(y, h_read, seq)  # legacy entity-pooled readout
                dnr = model.donor(h_read)  # state-consistent readout (inference)
            anchor_rows.append(
                {
                    "customer_key": seq["customer"],
                    "anchor_epoch": float(seq["anchor_epoch"]),
                    "version": cfg.tag,
                    "dim": int(dsq.shape[0]),
                    "embedding": dsq.tolist(),
                }
            )
            donor_rows.append(
                {
                    "customer_key": seq["customer"],
                    "anchor_epoch": float(seq["anchor_epoch"]),
                    "version": cfg.tag,
                    "dim": int(dnr.shape[0]),
                    "embedding": dnr.tolist(),
                }
            )
    store.bulk_anchor_rows(anchor_rows, donor_rows)
    s, t = store.count()
    store.close()
    return s, t
