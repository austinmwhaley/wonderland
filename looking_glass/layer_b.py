"""Layer B: the realizable universal embedding.

The sufficiency result (DEC-142/148) needs the **generic full-history measure** (all
fields), which the window-limited encoder cannot produce (DEC-153). This module makes
the architecture real and reusable:

    embedding(customer, t) = [ encoder(shape)  ‖  compress( generic full-history measure ) ]

* `generic_measure` computes the measure per customer over the FULL history (point-in-
  time, generic across schemas — every field aggregated by its natural type).
* `LayerBEmbedder` fits a **learned compression** (standardize + PCA) of the measure on
  training data, persists it, and composes it with the encoder embedding.

This is the object the loop builds toward: a universal foundation embedding whose
sufficiency is certified by `looking_glass.universality`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from looking_glass.cfm_config import as_of_epoch


def generic_measure(cfg, cust, only_keys=None):
    """Schema-derived, EXHAUSTIVE full-history measure (the Layer-B measure + the oracle's input).

    GENERIC, not hand-picked: discover EVERY field in the canonical stream and aggregate it by
    its natural type — every categorical field -> distinct-count; every numeric field ->
    count/sum/mean/std/min/max; plus generic timing. `only_keys` restricts the JSON keys (for
    ablations). Returns (X:(N,D) float32, names) aligned to `cust` order."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    keys = [str(c) for c in list(cust)]
    if not keys:
        return np.zeros((0, 1), dtype=np.float32), []
    con.register("_cust", pl.DataFrame({"customer_key": keys}))
    con.execute(
        f"CREATE TEMP TABLE base AS SELECT e.customer_key AS ck, "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) AS t, e.event_type AS et, "
        f"e.event_attributes AS attrs FROM {src} e "
        f"JOIN _cust c ON e.customer_key = c.customer_key WHERE "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) <= {tc}"
    )
    jkeys = [
        str(r[0])
        for r in con.execute(
            "SELECT DISTINCT unnest(json_keys(attrs)) AS k FROM base "
            "WHERE attrs IS NOT NULL ORDER BY k"
        ).fetchall()
    ]
    if only_keys is not None:
        allow = {str(k) for k in only_keys}
        jkeys = [k for k in jkeys if k in allow]
    numeric = []
    for k in jkeys:
        esc = k.replace("'", "''")
        cast = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
        nn, tot = con.execute(
            f"SELECT count(*) FILTER (WHERE {cast} IS NOT NULL), "
            f"count(json_extract_string(attrs,'$.{esc}')) FROM base"
        ).fetchone()
        if tot and nn / tot > 0.9:
            numeric.append(k)
    aggs = [
        "count(*) AS n_events",
        "count(DISTINCT et) AS n_etype",
        "max(t) - min(t) AS tspan",
        "stddev(t) AS tspread",
        "max(t) AS tlast",
    ]
    cols = ["n_events", "n_etype", "tspan", "tspread", "tlast"]
    for k in jkeys:
        esc = k.replace("'", "''")
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        aggs.append(f"count(DISTINCT json_extract_string(attrs,'$.{esc}')) AS cat_{safe}_nuniq")
        cols.append(f"cat_{safe}_nuniq")
    for k in numeric:
        esc = k.replace("'", "''")
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        c = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
        for fn, s in (
            ("cnt", f"count({c})"),
            ("sum", f"sum({c})"),
            ("mean", f"avg({c})"),
            ("std", f"stddev({c})"),
            ("min", f"min({c})"),
            ("max", f"max({c})"),
        ):
            aggs.append(f"{s} AS num_{safe}_{fn}")
            cols.append(f"num_{safe}_{fn}")
    df = con.execute(f"SELECT ck, {', '.join(aggs)} FROM base GROUP BY ck").pl()
    # ---- action-conditioned RESPONSE measure (RL leg, DEC-175): the empirical dose-response
    # exposed BY CONSTRUCTION (the same principle that solved the mass axis). For each DECLARED
    # randomized action key, per customer: mean reward at each action level over the stream's
    # action periods. Generic (keys declared; levels/cadence from the data; reward = the
    # stream's own value-bearing event). Makes `E` decision-sufficient for that action family.
    resp_blocks, resp_names = [], []
    acts = [str(k) for k in (getattr(cfg, "effect_actions", ()) or ())]
    if acts:
        from looking_glass.cfm_data import ACTION_PERIOD_S

        _t0 = con.execute("SELECT min(t) FROM base").fetchone()[0]
        if _t0 is not None:
            t0 = float(_t0)
            val = "TRY_CAST(json_extract_string(attrs,'$.value') AS DOUBLE)"
            cidx = {c: i for i, c in enumerate(keys)}
            for k in acts:
                esc = k.replace("'", "''")
                a = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
                mxr = con.execute(
                    f"SELECT max(TRY_CAST(json_extract_string(attrs,'$.{esc}') AS BIGINT)) "
                    f"FROM base WHERE {a} IS NOT NULL"
                ).fetchone()[0]
                if mxr is None:
                    continue
                Q = int(mxr) + 1
                dfk = con.execute(
                    f"WITH per AS (SELECT ck, "
                    f"CAST(floor((t - {t0})/{ACTION_PERIOD_S}) AS BIGINT) AS per, "
                    f"max(TRY_CAST(json_extract_string(attrs,'$.{esc}') AS BIGINT)) AS act "
                    f"FROM base WHERE {a} IS NOT NULL GROUP BY 1,2), "
                    f"rew AS (SELECT ck, "
                    f"CAST(floor((t - {t0})/{ACTION_PERIOD_S}) AS BIGINT) AS per, 1.0 AS y "
                    f"FROM base WHERE {val} IS NOT NULL AND {val} != 0 GROUP BY 1,2) "
                    f"SELECT per.ck AS ck, per.act AS act, avg(COALESCE(rew.y,0.0)) AS r "
                    f"FROM per LEFT JOIN rew ON per.ck=rew.ck AND per.per=rew.per GROUP BY 1,2"
                ).pl()
                M = np.zeros((len(keys), Q), np.float32)
                for c, aa, r in zip(dfk["ck"].to_list(), dfk["act"].to_list(), dfk["r"].to_list()):
                    j = cidx.get(c)
                    if j is not None and 0 <= int(aa) < Q:
                        M[j, int(aa)] = float(r)
                safe = "".join(ch if ch.isalnum() else "_" for ch in k)
                resp_blocks.append(M)
                resp_names += [f"resp_{safe}_{i}" for i in range(Q)]
    con.close()
    full = pl.DataFrame({"ck": keys}).join(df, on="ck", how="left").fill_null(0.0)
    full = full.with_columns((pl.lit(tc) - pl.col("tlast")).alias("recency"))
    cols = cols + ["recency"]
    X = np.column_stack([full[c].to_numpy().astype(np.float32) for c in cols])
    if resp_blocks:
        X = np.column_stack([X, *resp_blocks])
        cols = cols + resp_names
    return X.astype(np.float32), cols


def canonical_measure(cfg, cust, buckets: int = 256):
    """Schema-agnostic CANONICAL measure — FIXED dimension for ANY stream (universality).

    Every discovered field is **hashed** into `buckets` slots and aggregated per slot:
    categorical fields -> event COUNTS per `hash(field=value)`; numeric fields -> SUM and
    SUM-OF-SQUARES per `hash(field)`. Dimension = 2*buckets regardless of schema, so ONE
    measure model can serve many streams. Generic (keys discovered + hashed), not hand-picked."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    keys = [str(c) for c in list(cust)]
    names = [f"cnt_{i}" for i in range(int(buckets))] + [f"num_{i}" for i in range(int(buckets))]
    if not keys:
        return np.zeros((0, 2 * int(buckets)), dtype=np.float32), names
    con.register("_cust", pl.DataFrame({"customer_key": keys}))
    con.execute(
        f"CREATE TEMP TABLE base AS SELECT e.customer_key AS ck, "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) AS t, e.event_attributes AS attrs FROM {src} e "
        f"JOIN _cust c ON e.customer_key = c.customer_key WHERE "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) <= {tc}"
    )
    jkeys = [
        str(r[0])
        for r in con.execute(
            "SELECT DISTINCT unnest(json_keys(attrs)) AS k FROM base "
            "WHERE attrs IS NOT NULL ORDER BY k"
        ).fetchall()
    ]
    B = int(buckets)
    parts = []
    for k in jkeys:
        esc = k.replace("'", "''")
        cast = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
        nn, tot = con.execute(
            f"SELECT count(*) FILTER (WHERE {cast} IS NOT NULL), "
            f"count(json_extract_string(attrs,'$.{esc}')) FROM base"
        ).fetchone()
        if tot and nn / tot > 0.9:  # numeric -> sums per field-hash
            parts.append(
                f"SELECT ck, hash('{esc}')%{B} AS b, 0.0 AS cnt, "
                f"coalesce(sum({cast}),0.0) AS sx, coalesce(sum({cast}*{cast}),0.0) AS sxx "
                f"FROM base GROUP BY 1,2"
            )
        else:  # categorical -> value counts per (field=value)-hash
            parts.append(
                f"SELECT ck, hash(concat('{esc}','=', json_extract_string(attrs,'$.{esc}')))%{B} AS b, "
                f"count(*) AS cnt, 0.0 AS sx, 0.0 AS sxx FROM base "
                f"WHERE json_extract_string(attrs,'$.{esc}') IS NOT NULL GROUP BY 1,2"
            )
    df = con.execute(
        f"SELECT ck, b, sum(cnt) AS cnt, sum(sx) AS sx, sum(sxx) AS sxx "
        f"FROM ({' UNION ALL '.join(parts)}) GROUP BY 1,2"
    ).pl()
    con.close()
    N = len(keys)
    Xc = np.zeros((N, B), dtype=np.float32)
    Xs = np.zeros((N, B), dtype=np.float32)
    ki = {k: i for i, k in enumerate(keys)}
    for ck, b, cnt, sx, sxx in zip(df["ck"], df["b"], df["cnt"], df["sx"], df["sxx"]):
        i = ki.get(str(ck))
        if i is None:
            continue
        Xc[i, int(b)] += float(cnt)
        Xs[i, int(b)] += float(sx) + float(sxx)
    X = np.hstack([np.log1p(np.abs(Xc)), np.log1p(np.abs(Xs))]).astype(np.float32)
    return X, names


def field_set_measure(cfg, cust, only_keys=None):
    """Per-field aggregate SET — the schema-agnostic measure representation.

    Returns X:(N, F, 6) and field names: one 6-vector per field (numeric ->
    [count,sum,mean,std,min,max]; categorical -> [count,distinct,0,0,0,0]). A
    permutation-invariant (field-set) encoder over this works on ANY schema/field
    count -> ONE measure model can serve many streams (universality)."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    keys = [str(c) for c in list(cust)]
    if not keys:
        return np.zeros((0, 0, 6), dtype=np.float32), []
    con.register("_cust", pl.DataFrame({"customer_key": keys}))
    con.execute(
        f"CREATE TEMP TABLE base AS SELECT e.customer_key AS ck, "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) AS t, e.event_attributes AS attrs FROM {src} e "
        f"JOIN _cust c ON e.customer_key = c.customer_key WHERE "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) <= {tc}"
    )
    jkeys = [
        str(r[0])
        for r in con.execute(
            "SELECT DISTINCT unnest(json_keys(attrs)) AS k FROM base "
            "WHERE attrs IS NOT NULL ORDER BY k"
        ).fetchall()
    ]
    if only_keys is not None:
        allow = {str(k) for k in only_keys}
        jkeys = [k for k in jkeys if k in allow]
    aggs = []
    for k in jkeys:
        esc = k.replace("'", "''")
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        cast = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
        nn, tot = con.execute(
            f"SELECT count(*) FILTER (WHERE {cast} IS NOT NULL), "
            f"count(json_extract_string(attrs,'$.{esc}')) FROM base"
        ).fetchone()
        if tot and nn / tot > 0.9:  # numeric: 6 moments
            aggs += [
                f"count({cast}) AS {safe}_0",
                f"coalesce(sum({cast}),0.0) AS {safe}_1",
                f"coalesce(avg({cast}),0.0) AS {safe}_2",
                f"coalesce(stddev({cast}),0.0) AS {safe}_3",
                f"coalesce(min({cast}),0.0) AS {safe}_4",
                f"coalesce(max({cast}),0.0) AS {safe}_5",
            ]
        else:  # categorical: count + distinct
            aggs += [
                f"count(json_extract_string(attrs,'$.{esc}')) AS {safe}_0",
                f"count(DISTINCT json_extract_string(attrs,'$.{esc}')) AS {safe}_1",
                f"0.0 AS {safe}_2",
                f"0.0 AS {safe}_3",
                f"0.0 AS {safe}_4",
                f"0.0 AS {safe}_5",
            ]
    df = con.execute(
        f"SELECT ck{'' if not aggs else ', ' + ', '.join(aggs)} FROM base GROUP BY ck"
    ).pl()
    con.close()
    full = pl.DataFrame({"ck": keys}).join(df, on="ck", how="left").fill_null(0.0)
    F = len(jkeys)
    X = np.zeros((len(keys), F, 6), dtype=np.float32)
    for fi, k in enumerate(jkeys):
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        for m in range(6):
            X[:, fi, m] = full[f"{safe}_{m}"].to_numpy().astype(np.float32)
    return X, jkeys


class FieldSetMeasure:
    """Permutation-invariant (field-set) measure encoder: per-field 6-vectors ->
    shared MLP -> sum-pool -> d-dim. Field-agnostic (any F) -> cross-stream usable."""

    def __init__(self, d=64, hidden=64, epochs=200, seed=0):
        self.d = int(d)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.net = None
        self.scaler = None

    def fit(self, X):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        N, F, M = X.shape
        sc = StandardScaler().fit(X.reshape(-1, M))
        Xs = sc.transform(X.reshape(-1, M)).reshape(N, F, M).astype(np.float32)
        torch.manual_seed(self.seed)
        net = nn.Sequential(nn.Linear(M, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.d))
        x = torch.tensor(Xs)
        head = nn.Linear(self.d, M)
        params = list(net.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        for _ in range(self.epochs):
            opt.zero_grad()
            per = net(x)  # (N,F,d)
            pooled = per.mean(1)  # (N,d)
            rec = head(pooled)[:, None, :].expand_as(x)  # broadcast decode
            loss = ((rec - x) ** 2).mean()
            loss.backward()
            opt.step()
        self.net, self.scaler, self.head = net.eval(), sc, head.eval()
        return self

    def embed(self, X) -> np.ndarray:
        import torch

        N, F, M = X.shape
        Xs = self.scaler.transform(X.reshape(-1, M)).reshape(N, F, M).astype(np.float32)
        with torch.no_grad():
            per = self.net(torch.tensor(Xs))
            pooled = per.mean(1)
        return pooled.numpy().astype(np.float32)


class LayerBEmbedder:
    """Compose encoder embedding with a LEARNED compression of the generic measure.

    `fit(measure)` learns the compression (standardize + PCA) on TRAINING measures (no
    leakage); `compress(measure)` applies it; `embed(model, seqs, cfg)` returns
    `[encoder ‖ compressed measure]` for the sequences' customers."""

    def __init__(self, n_components: int = 64):
        self.n_components = int(n_components)
        self.scaler = None
        self.pca = None
        self.names: list = []

    def fit(self, measure):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(measure)
        m = self.scaler.transform(measure)
        k = min(self.n_components, m.shape[1], max(1, m.shape[0] - 1))
        self.pca = PCA(n_components=k, random_state=0).fit(m)
        return self

    def compress(self, measure) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("LayerBEmbedder not fit")
        return self.pca.transform(self.scaler.transform(measure)).astype(np.float32)

    def save(self, path) -> None:
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "pca": self.pca,
                    "names": self.names,
                    "n_components": self.n_components,
                },
                f,
            )

    @classmethod
    def load(cls, path) -> "LayerBEmbedder":
        import pickle

        with open(path, "rb") as f:
            b = pickle.load(f)
        e = cls(b.get("n_components", 64))
        e.scaler, e.pca, e.names = b["scaler"], b["pca"], b.get("names", [])
        return e

    def embed(self, model, cfg, seqs, only_keys=None) -> np.ndarray:
        """E = [encoder embedding ‖ compressed generic measure] for `seqs`."""
        import torch

        from looking_glass.cfm_training import forward_embedding

        with torch.no_grad():
            Z = forward_embedding(model, seqs)
        Z = Z.detach().float().cpu().numpy() if hasattr(Z, "detach") else np.asarray(Z)
        keys = [s["customer"] for s in seqs]
        m, self.names = generic_measure(cfg, keys, only_keys=only_keys)
        return np.hstack([Z, self.compress(m)]).astype(np.float32)
