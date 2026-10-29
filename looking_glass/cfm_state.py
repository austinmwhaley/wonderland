"""CFM state operations (fade/absorb) and the DuckDB state store."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from looking_glass.cfm_config import CFMConfig, LN2, TIME_UNIT_SECONDS, _to_epoch, as_of_epoch
from looking_glass.cfm_data import build_sequences
from looking_glass.cfm_model import CFM
from looking_glass import view_layout


# ---------------------------------------------------------------------------
# state operations (fade / absorb)
# ---------------------------------------------------------------------------
def fade(h: torch.Tensor, dt_seconds, half_life_days: float) -> torch.Tensor:
    """Decay the state when time passes with NO events.

    `dt_seconds` is a scalar OR a tensor broadcastable against `h` (the `response` outlet fades a
    whole padded batch in one op, DEC-426). One law, one expression: a tensor path written
    separately is a second definition of the same law, and the two drift.
    """
    dt = torch.as_tensor(dt_seconds, dtype=h.dtype, device=h.device)
    decay = torch.exp(-(LN2 * dt) / max(half_life_days * TIME_UNIT_SECONDS, 1.0))
    return h * torch.where(dt > 0, decay, torch.ones_like(decay))


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
        counter_type_values=bool(blob.get("counter_type_values", False)),
        schema_summary=bool(blob.get("schema_summary", False)),
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
        if str(_z.get("mode", "pca")) == "explicit":
            d = int(_z["shape_dim"])
            model.serve_total_dim = int(_z.get("total_dim", d))
            model.set_serve_explicit(
                d, int(_z["donor_dim"]), _z["mean_d"], _z["scale_d"], _z["W_d"]
            )
            print(
                f"[load] explicit serve: donor->{int(_z['W_d'].shape[0])} + explicit "
                f"{int(model.serve_total_dim) - int(_z['donor_dim'])} dims",
                flush=True,
            )
        elif str(_z.get("mode", "pca")) == "split":
            d = int(_z["shape_dim"])
            model.serve_total_dim = int(_z.get("total_dim", d))
            model.set_serve_split(
                d,
                int(_z["counter_dim"]),
                _z["mean_d"],
                _z["scale_d"],
                _z["W_d"],
                _z["mean_m"],
                _z["scale_m"],
                _z["W_m"],
            )
            print(
                f"[load] split serve: shape->{int(_z['W_d'].shape[0])}+{int(_z['W_m'].shape[0])} "
                f"+ measure -> {int(model.serve_total_dim) - d} dims",
                flush=True,
            )
        elif str(_z.get("mode", "pca")) == "blocks":
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
        # PUBLISHED OUTLETS (SPINE §5, DEC-424): the named views the frozen artifact owes
        # its consumers. Long format, one row per customer x view x PART — a card row can
        # attribute a score to a part and the conformance kit can zero or permute exactly
        # one of them (DEC-405 rule 6). `full` is deliberately NOT stored: it is the
        # documented concatenation of `view_layout.FULL_ORDER`, a bundle and not content.
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS view_embeddings ("
            "customer_key TEXT, as_of_epoch DOUBLE, version TEXT, view TEXT, part TEXT, "
            "dim INT, embedding FLOAT[])"
        )
        # The faceplate (Outlet Standard O4): the artifact describes itself so thousands of
        # consumers can self-validate and locally refuse a stale or short artifact.
        self.con.execute("CREATE TABLE IF NOT EXISTS view_faceplate (version TEXT, stamp JSON)")
        # The fitted `response` head (DEC-426) with its calibration receipt: the outlet is
        # reproducible from the artifact alone, and a consumer can see the coverage its
        # intervals actually achieved. Rewritten per release, like the faceplate.
        self.con.execute("CREATE TABLE IF NOT EXISTS response_head (version TEXT, stamp JSON)")

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
        decay = torch.exp(
            -LN2 * dt / max(self.cfg.state_half_life_days * TIME_UNIT_SECONDS, 1.0)
        ).clamp(max=1.0)  # dt <= 0 -> identity, matching fade()
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

    def bulk_view_rows(self, view_rows) -> int:
        """Insert one bounded buffer of outlet rows (the caller owns the buffer budget —
        see `build_products`, which flushes on a size derived from the pool).

        Chunking matters: polars materialises the whole list of dicts before DuckDB sees any
        of it, and on a pool shared with an LLM server an unbounded build buffer is what puts
        `MemAvailable` under the system floor (measured 2026-10-29: the products phase reached
        6.7 GB against the 8.0 floor and the watchdog killed the run).
        """
        import polars as pl

        if not view_rows:
            return 0
        df = pl.DataFrame(view_rows)
        self.con.register("_vr", df)
        try:
            self.con.execute("INSERT INTO view_embeddings SELECT * FROM _vr")
        finally:
            self.con.unregister("_vr")
        return len(view_rows)

    def record_response_head(self, head: dict) -> None:
        """Publish the fitted response head + its calibration receipt (one row per release)."""
        import json

        self.con.execute("DELETE FROM response_head")
        self.con.execute(
            "INSERT INTO response_head VALUES (?, ?)",
            [str(head.get("version", "")), json.dumps(head)],
        )

    def verify_outlets(self, stamp: dict) -> None:
        """Refuse a release whose published columns disagree with the artifact's own stamp.

        This is the invariant the first build of DEC-424 violated: the heads were fed the
        PER-STEP readout, so a view's width varied with each customer's sequence length and
        the table silently published ragged vectors. A consumer validating against the
        faceplate would then have read something the stamp never described (O4), so this is
        checked in SQL against the stored arrays and RAISES — a bad artifact must not be
        readable at all (Principle 7: reject, do not clamp).
        """
        bad = []
        for view, part, n_distinct, lo, hi in self.con.execute(
            "SELECT view, part, count(DISTINCT array_length(embedding)), "
            "min(array_length(embedding)), max(array_length(embedding)) "
            "FROM view_embeddings GROUP BY view, part ORDER BY view, part"
        ).fetchall():
            declared = ((stamp.get("parts") or {}).get(view) or {}).get(part)
            if n_distinct != 1:
                bad.append(f"{view}.{part}: ragged widths {lo}..{hi}")
            elif declared is not None and hi != int(declared):
                bad.append(f"{view}.{part}: stored {hi} != faceplate {declared}")
        if bad:
            raise RuntimeError(
                "published outlets contradict the faceplate: "
                + "; ".join(bad)
                + " — the producer is feeding the heads the wrong readout; fix the producer, "
                "do not restamp the artifact"
            )

    def write_faceplate(self, stamp: dict) -> None:
        """Stamp the artifact with its own column inventory (O4). Rewritten per release, so
        the faceplate can never be newer or older than the tables it describes."""
        import json

        self.con.execute("DELETE FROM view_faceplate")
        self.con.execute(
            "INSERT INTO view_faceplate VALUES (?, ?)", [str(stamp["version"]), json.dumps(stamp)]
        )

    def verify_state_serving(self, n_states: int, n_embeddings: int, close_at: float) -> None:
        """Refuse a release that serves an empty `state_embeddings` next to a full `customer_state`.

        Plugins are told to read `state_embeddings`; `customer_state` is what it is derived from.
        The two counts can disagree only through a defect — the day-1 close was once guarded on a
        buffer that had already been freed, so the served table shipped EMPTY while the build
        printed nothing alarming — and a consumer reading an empty table gets a failure it cannot
        diagnose. So the counts are PRINTED every build (O11) and RAISED on disagreement
        (Principle 7: reject, do not let a silent absence become someone else's zero).
        """
        print(
            f"[states] served: customer_state={n_states} state_embeddings={n_embeddings} "
            f"at boundary {close_at:.0f}",
            flush=True,
        )
        if n_states and n_embeddings == 0:
            raise RuntimeError(
                f"release refused: customer_state has {n_states} rows but state_embeddings is "
                f"EMPTY — every plugin reading the served table would get nothing. The day-1 "
                f"close (fade_idle + materialize_state_embeddings) did not run; check that it is "
                f"not guarded on a build buffer that was already freed."
            )
        if n_embeddings != n_states:
            raise RuntimeError(
                f"release refused: state_embeddings={n_embeddings} != customer_state={n_states} "
                f"at boundary {close_at:.0f} — the odometer lost customers between the state and "
                f"its served embedding"
            )

    def count(self):
        s = self.con.execute("SELECT count(*) FROM customer_state").fetchone()[0]
        t = self.con.execute("SELECT count(*) FROM anchor_embeddings").fetchone()[0]
        return s, t

    def close(self):
        self.con.close()


def _publish_response_outlet(cfg, model, df, keys, split, store) -> dict | None:
    """Fit and publish the `response` outlet (DEC-384 / DEC-426); returns the head for the stamp.

    Fitted on population A only and calibrated on B, AFTER the encoder is frozen. Returns None
    (→ the faceplate advertises `response` as not served) when the stream has no logged actions
    or too few rows to calibrate honestly — a thin outlet is worse than an advertised gap,
    because a consumer would read its interval as real.
    """
    import gc

    import numpy as np
    import polars as pl

    from looking_glass import gpu_budget, response_outlet

    # γ is derived from the stream's own cadence, in one columnar pass (no Python loop over
    # events): the horizon the outcome is measured over is one typical inter-event interval.
    gaps_s = (
        df.sort(["customer_key", "event_ts"])
        .with_columns(
            pl.col("event_ts")
            .str.to_datetime(time_zone="UTC", strict=False)
            .dt.epoch("s")
            .diff()
            .over("customer_key")
            .alias("_gap_s")
        )["_gap_s"]
        .drop_nulls()
    )
    gamma, gamma_receipt = response_outlet.derive_gamma(gaps_s.to_numpy() / TIME_UNIT_SECONDS)
    # The action family comes from the stream's own declaration (the generator's truth file),
    # never from an event name typed here. No declaration → no outlet, with the reason printed.
    design = response_outlet.declare_action_design(cfg.db)

    seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=2)
    rows = response_outlet.collect(model, cfg, seqs, gamma=gamma, action_design=design)
    if rows is None or len(rows["target"]) == 0:
        print(
            "[response] no logged action events in the stream — outlet stays not_served", flush=True
        )
        return None
    grp = {str(k): split.get(k, "A") for k in keys}
    is_a = np.asarray([grp.get(c, "A") == "A" for c in rows["customer"]], dtype=bool)

    def pick(mask):
        """Rows for one population — the same action-event rows, split by A/B. A customer's
        rows can only land on one side (the split is by customer), so fit and calibration
        never share a customer."""
        return {
            k: (v[mask] if isinstance(v, np.ndarray) else np.asarray(v, dtype=object)[mask])
            for k, v in rows.items()
            if k != "n_actions"
        }

    rows_a, rows_b = pick(is_a), pick(~is_a)
    if len(rows_a["target"]) < 30 or len(rows_b["target"]) < 30:
        print(
            f"[response] too few action rows to fit AND calibrate honestly "
            f"(A={len(rows_a['target'])}, B={len(rows_b['target'])}) — not_served rather than "
            f"publishing an interval nobody measured",
            flush=True,
        )
        return None
    head = response_outlet.fit(rows_a, rows_b)
    head["gamma_derivation"] = gamma_receipt
    head["action_design"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in design.items()}
    head["version"] = str(cfg.tag)
    pub = response_outlet.publish_rows(rows, head, version=cfg.tag)
    # Stamped at the customer's own last event, the same convention as every other view: the
    # response summarizes the actions that actually happened in that customer's history.
    last = {str(s.get("customer")): float(_to_epoch(s["event_ts"][-1])) for s in seqs}
    for r in pub:
        r["as_of_epoch"] = last.get(r["customer_key"])
    pub = [r for r in pub if r["as_of_epoch"] is not None]
    n = store.bulk_view_rows(pub)
    store.record_response_head(head)
    # Release the build buffers before the next phase (the pool is shared; freeing them late is
    # how a products phase reaches the system floor). Assigning None rather than `del` keeps
    # the names bound — the row slices above close over `rows`, and deleting it out from under
    # a closure is a bug that only appears at the worst moment.
    seqs = rows = rows_a = rows_b = pub = None
    gc.collect()
    gpu_budget.release(cfg, "products_response")
    print(
        f"[response] published: fit on {head['n_fit']} A-rows, calibrated on {head['n_calibrate']} "
        f"B-rows; τ={gamma_receipt['tau_days']}d γ={gamma_receipt['gamma']}; holdout R²="
        f"{head['target_r2_holdout']} (leaky ceiling {head['target_r2_ceiling_leaky']}); "
        f"coverage {head['coverage_empirical']} (nominal {head['coverage_nominal']}) at measured "
        f"σ {head['sigma_randomized']} rand / {head['sigma_unrandomized']} unrand, mean interval "
        f"width {head['mean_interval_width']}; randomized fraction "
        f"{head['randomized_frac']}; {n} rows",
        flush=True,
    )
    return head


def _publish_ledger_outlet(cfg, model, vocab, df, keys, split, store) -> dict | None:
    """Build and publish the `measure` outlet (SPINE §5 / DEC-383): the ledger, outside the net.

    The recipe already existed (`measure_state.MeasureBuilder`) but was never written into a
    release, so the wall advertised `NOT SERVED: measure` while the numbers were one pass away.
    This publishes them AND runs the five guardrails: G1–G4 are HARD (a ledger whose recipe was
    hand-picked, peeked forward, read the targets, or fed the brain is not a ledger — it is PATH
    A wearing a bucket name, so publishing it would be laundering), and the release refuses it
    rather than shipping that. G5 is a soft receipt: the attribution row is produced by the card,
    so before the card has run it is an honest gap, not a defect in the data.

    Returns the stamp payload (vocabulary + guardrail receipts) or None to advertise not_served.
    """
    import gc

    from looking_glass import gpu_budget, ledger_state

    # The `products` command calls build_products with vocab=None (the frozen artifact carries
    # its own), so the model is the authority here rather than a second source to disagree with.
    vocab = vocab if vocab is not None else getattr(model, "vocab", None)
    if vocab is None:
        raise LookupError(
            "ledger: neither a vocabulary argument nor model.vocab — nothing to tally against"
        )
    # full history, every customer with at least one event — the ledger is an odometer over
    # everything visible before the cut, NOT the encoder's seq_len window (calling a windowed
    # tally "exact" would be a false claim of the first order)
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
    if not seqs:
        raise LookupError("ledger: no customer has a single event, so there is nothing to tally")
    train_keys = [k for k in keys if split.get(k, "A") == "A"]
    cut = as_of_epoch(cfg.as_of) if getattr(cfg, "as_of", None) else None
    builder, vocab_receipt = ledger_state.fit_vocabulary(
        seqs, vocab, train_keys, coverage=getattr(cfg, "ledger_coverage", None), as_of=cut
    )
    rows = ledger_state.publish_rows(seqs, builder, vocab, version=cfg.tag, as_of=cut)
    # The audit sees the SAME adapter output the publisher used, so it is auditing the artifact
    # and not a second, more convenient reading of the stream — and it runs BEFORE the write,
    # because a ledger that fails its own audit must never be readable at all.
    tuples_a = ledger_state.seqs_to_tuples(
        [s for s in seqs if split.get(s["customer"], "A") == "A"], vocab, as_of=cut
    )
    receipts = ledger_state.guardrails(
        builder,
        vocab,
        tuples_a,
        cfg,
        train_keys,
        target_names=tuple(getattr(cfg, "exogenous_events", ()))
        + tuple(getattr(cfg, "action_fields", ())),
        as_of=cut,
    )
    hard_names = ("G1_mechanical", "G2_history_only", "G3_target_blind", "G4_not_a_brain_input")
    hard = [g for g in hard_names if not receipts[g]["working"]]
    if hard:
        raise RuntimeError(
            "ledger refused: "
            + "; ".join(f"{g} — {receipts[g]['evidence']}" for g in hard)
            + " (the guardrails are what make this a ledger rather than a hand-built table)"
        )
    n = store.bulk_view_rows(rows)
    payload = {
        "kind": "ledger",
        "version": str(cfg.tag),
        "vocabulary": vocab_receipt,
        "slices": {p: [list(r) for r in runs] for p, runs in builder.part_slices().items()},
        "n_rows": n,
        "guardrails": receipts,
        # G1–G4 are hard (the release refused above if any failed). G5 is produced LATER, by the
        # card, so it is reported as pending rather than folded into one green/red.
        "working": all(receipts[g]["working"] for g in hard_names),
        "pending": sorted(g for g, r in receipts.items() if r.get("pending")),
    }
    seqs = rows = tuples_a = None
    gc.collect()
    gpu_budget.release(cfg, "products_ledger")
    print(
        f"[ledger] published {payload['vocabulary']['dim']}-d measure view over "
        f"{len(view_layout.PART_ORDER['measure'])} parts ({n} rows); "
        f"{payload['vocabulary']['n_fields_vocabularied']}"
        f"/{payload['vocabulary']['n_fields_discovered']} fields vocabularied at coverage "
        f"{payload['vocabulary']['coverage']}; guardrails "
        + " ".join(
            f"{g.split('_')[0]}="
            + ("ok" if r["working"] else "pending" if r.get("pending") else "FAIL")
            for g, r in receipts.items()
        ),
        flush=True,
    )
    return payload


def build_products(cfg, model, vocab, df, keys, split):
    # The `products` command runs this WITHOUT train_cfm, so the co-tenant cap must be
    # applied here too (idempotent when train_cfm already did it). DEC-423.
    from looking_glass import gpu_budget

    gpu_budget.enforce(cfg)
    gpu_budget.release(cfg, "pre_products")
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
    # PUBLISHED OUTLETS (SPINE §5 / DEC-424): the named views are taken from the SAME
    # forward that produced the state rows. Paying for the encoder once and publishing
    # several things is the cost doctrine (Principle 15); a second pass would be a second
    # chance for the views to disagree with the state they claim to summarize.
    view_rows = []
    n_view = 0
    _flush_at = None  # derived once, from the first customer's own row width
    # principle-ok: an I/O buffer budget, derived from the pool the run actually has (2% of
    # MemAvailable, 32 MB floor) — the size is a resource decision, not a modeling constant.
    _buffer_bytes = max(32 << 20, int(0.02 * gpu_budget.available_gb() * 2**30))
    # min_events=1: states exist for every customer with at least one customer
    # event (the blanket 3 silently starved long-tail customers of states).
    for seq in build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1):
        with torch.no_grad():
            _y, h = model(seq)
            emb = model.embed(h)
            # `y` is the PER-STEP readout [T, dim]; an outlet is one vector per customer,
            # so the heads read the readout at the customer's LAST event — the same tick
            # `h` is summarized at. Feeding the whole [T, dim] here would (and did) publish
            # a column whose width varies with sequence length, which is not a view.
            # Batched (1, dim) so the batched readout path is the one that ships.
            group = view_layout.parts(model, _y[-1].unsqueeze(0), h.unsqueeze(0))
        last_ts = seq["event_ts"][-1]
        epoch = float(_to_epoch(last_ts))
        state_rows.append(
            {
                "customer_key": seq["customer"],
                "as_of_epoch": epoch,
                "version": cfg.tag,
                "dim": int(h.shape[0]),
                "state": h.tolist(),
                "embedding": emb.tolist(),
                "last_event_ts": str(last_ts),
            }
        )
        for view, group_parts in group.items():
            for part, tensor in group_parts.items():
                view_rows.append(
                    {
                        "customer_key": seq["customer"],
                        # The outlet is stamped at the customer's OWN last event, not at
                        # as_of: the heads read the readout at that tick, and fading a `y`
                        # is not defined (only the state fades). The daily job owns
                        # boundary-faded `shape`; conflating the two would be a silent
                        # train/serve skew wearing a bucket name.
                        "as_of_epoch": epoch,
                        "version": cfg.tag,
                        "view": view,
                        "part": part,
                        "dim": int(tensor.shape[-1]),
                        "embedding": tensor.reshape(-1).tolist(),
                    }
                )
        if _flush_at is None and view_rows:
            # one customer's worth of rows (all parts) sizes the buffer for the rest
            _width = sum(len(r["embedding"]) for r in view_rows)
            _flush_at = max(len(view_rows), int(_buffer_bytes / max(8 * _width, 1)))
        if _flush_at and len(view_rows) >= _flush_at:
            n_view += store.bulk_view_rows(view_rows)
            view_rows.clear()
    store.bulk_states(state_rows)
    state_rows.clear()
    # Phase receipt (P11) at the point where this phase historically breached the system
    # floor: the build buffers here, not the step loop, are what put MemAvailable at 6.7 GB
    # against an 8.0 GB floor on 2026-10-29. Naming the phase is what makes the fix a
    # measurement rather than a guess.
    gpu_budget.release(cfg, "products_states")
    # ---------------------------------------------------------------- the `response` outlet
    # SPINE §5 / DEC-384, built by DEC-426: conditional outcome given the LOGGED action, with
    # measured uncertainty. The encoder is READ, never trained — the head is fitted on
    # population A after freezing, and the release still succeeds without it (the faceplate
    # then advertises `not_served: response`, which is an absence the card can see, not one it
    # has to trust).
    head = None
    try:
        head = _publish_response_outlet(cfg, model, df, keys, split, store)
    except Exception as exc:  # an additive outlet must not hold the release hostage
        # Print WHERE it died. Dropping the traceback here is the DEC-425 failure mode again:
        # the operator sees "not published" and a one-line message, and has to reproduce the
        # crash to find it. A failure you cannot attribute is a failure you cannot fix.
        import traceback

        print(
            f"[response] outlet NOT published: {type(exc).__name__}: {exc} — the faceplate "
            f"will advertise it as not_served\n{traceback.format_exc()}",
            flush=True,
        )
        head = None
    # ------------------------------------------------------------------- the `measure` outlet
    # SPINE §5 / DEC-383: the ledger as a named companion. Additive for the same reason the
    # response outlet is — a release that cannot satisfy the guardrails ships without it and
    # says so, rather than shipping a tally nobody audited.
    ledger = None
    try:
        ledger = _publish_ledger_outlet(cfg, model, vocab, df, keys, split, store)
    except Exception as exc:
        import traceback

        print(
            f"[ledger] outlet NOT published: {type(exc).__name__}: {exc} — the faceplate "
            f"will advertise it as not_served\n{traceback.format_exc()}",
            flush=True,
        )
        ledger = None
    _fp = view_layout.faceplate(model, cfg, head=head, ledger=ledger)
    n_view += store.bulk_view_rows(view_rows)
    store.write_faceplate(_fp)
    store.verify_outlets(_fp)
    # The outlet receipt (O4): what the wall actually has sockets for, printed at the moment
    # it is written. An outlet that is approved but not served is printed too — a gap that
    # is not named is a gap that gets narrated away.
    print(
        f"[outlets] published {_fp['full_dim']}-d full bundle over {len(_fp['emitted'])} views: "
        + ", ".join(f"{v}={_fp['views'][v]}" for v in _fp["emitted"])
        + (f" | NOT SERVED: {', '.join(_fp['not_served'])}" if _fp["not_served"] else "")
        + f" ({n_view} rows)",
        flush=True,
    )
    # (1) also CLOSES day 1: fade every state to the as_of boundary (UTC day
    # start = all of yesterday absorbed, today excluded) and materialize that
    # day's inference embeddings. The daily job therefore only handles days >= 2
    # — no redundant day-1 absorb/fade/materialize work.
    #
    # ASK THE TABLE, NOT THE BUFFER. This guard used to read `if state_rows:`, but the buffer is
    # emptied a few lines above (DEC-425's memory fix), so the condition was false on every build
    # since: the day-1 fade + materialize never ran, `state_embeddings` shipped EMPTY, and every
    # plugin read an empty served table. Nothing printed a count and no gate read that table, so
    # it stayed invisible for six decisions. A guard that consults an in-memory buffer to decide
    # whether a STORED table was written is asking the wrong source of truth — the same class of
    # defect as the ragged-`shape` column (layout described in one place, built in another).
    n_states = int(store.con.execute("SELECT count(*) FROM customer_state").fetchone()[0])
    if n_states:
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
        store.verify_state_serving(n_states, n_emb, close_at)
    else:
        # Named, not skipped: a build with no states is a fact a reader can see, and the
        # `state_embeddings` table it leaves behind is empty for a REASON stated in the log.
        print("[states] customer_state is empty — no day-1 close to perform", flush=True)
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
                # The legacy entity-pooled readout (`donor_seq`) died with the role-free
                # Layer B (DEC-210/211) and this caller was never updated, so EVERY train
                # run aborted here and published ZERO rows in both tables — plugins read an
                # empty artifact. Restoring entity pooling would resurrect deleted role
                # machinery, and deleting the table would break every consumer's SQL, so
                # the two tables are now honestly the same object: the state-consistent
                # readout. Recorded in DECISIONS.md; collapsing them is a named follow-up.
                dnr = model.donor(h_read)  # state-consistent readout (inference)
                dsq = dnr
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
