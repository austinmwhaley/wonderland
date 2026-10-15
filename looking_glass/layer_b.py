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
            "SELECT DISTINCT unnest(json_keys(attrs)) FROM base WHERE attrs IS NOT NULL"
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
    con.close()
    full = pl.DataFrame({"ck": keys}).join(df, on="ck", how="left").fill_null(0.0)
    full = full.with_columns((pl.lit(tc) - pl.col("tlast")).alias("recency"))
    cols = cols + ["recency"]
    X = np.column_stack([full[c].to_numpy().astype(np.float32) for c in cols])
    return X.astype(np.float32), cols


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
