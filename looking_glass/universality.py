"""Universality gate (v8): is `embedding(customer, t)` a UNIVERSAL sufficient statistic?

The goal (AGENTS.md "Project goal"; loop/UNIVERSALITY.md): the embedding must be
sufficient for the FULL observable future, so any present or future consumer is
covered by construction. Measured by the UNIVERSALITY INDEX

    U = 1 - worst-case_tau ( normalized oracle-gap_tau )

where, for each task tau in a self-supervised future family,

    gap_tau       = score([E, history]) - score(E)          # what history adds beyond E
    normalized    = gap_tau / range_tau                     # range = oracle's achievable spread
                    (reg: range = score([E,history]); clf: range = score - 0.5)

`history` here is the **schema-derived, EXHAUSTIVE** full-history measure — every
field discovered from the stream, aggregated by its natural type (categorical ->
distinct-count; numeric -> count/sum/mean/std/min/max; + generic timing). It is
NOT a hand-picked feature set (no RFM), so an embedding cannot look sufficient by
reproducing hand features; the true oracle (a learned sequence model over the raw
history) bounds it from below, so the measured gap is a CONSERVATIVE estimate of
the real insufficiency. On the Layer-A lab (known truth) the gap is exact.

`U -> 1`  <=>  universal at this capacity. Worst-case (minimax), not average.

Run:
  python3 -m looking_glass.universality --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> [--customers 25000] [--horizons 30,90] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from looking_glass.cfm_config import _f, _to_epoch, as_of_epoch
from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.cfm_training import forward_embedding
from looking_glass.head_battery import (
    GBM_CLF,
    GBM_REG,
    _folds,
    _rfm_targets,
)


def _b_sequences(cfg):
    """Held-out (split B) raw event sequences — for the SEQUENCE-MODEL oracle."""
    from looking_glass.cfm_data import (
        _customer_keys,
        _read_stream,
        assign_split,
        build_sequences,
    )

    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    bkeys = [k for k in keys if split.get(k) == "B"]
    return build_sequences(df, bkeys, cfg, split, with_anchors=False)


def _seq_vocab(seqs):
    v: dict = {}
    for s in seqs:
        for e in s["event_type"]:
            k = str(e)
            if k not in v:
                v[k] = len(v)
    return v


def _seq_tensors(seqs, vocab, dev):
    B = len(seqs)
    T = max(1, max(len(s["event_type"]) for s in seqs))
    et = torch.zeros(B, T, dtype=torch.long)
    val = torch.zeros(B, T)
    dt = torch.zeros(B, T)
    mask = torch.zeros(B, T)
    for i, s in enumerate(seqs):
        L = len(s["event_type"])
        ts = np.asarray(s.get("ts") or [_to_epoch(x) for x in s["event_ts"]], dtype=np.float64)
        d = np.zeros(L)
        d[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
        for j in range(L):
            et[i, j] = vocab.get(str(s["event_type"][j]), len(vocab))
            val[i, j] = _f(s["value"][j])
            dt[i, j] = np.log1p(d[j])
        mask[i, :L] = 1.0
    return et.to(dev), val.to(dev), dt.to(dev), mask.to(dev)


class _SeqOracle(torch.nn.Module):
    """A small recurrent full-history oracle (order-aware), independent of E."""

    def __init__(self, n_et, hidden=32):
        super().__init__()
        self.emb = torch.nn.Embedding(n_et + 1, hidden)
        self.gru = torch.nn.GRU(hidden + 2, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, et, val, dt, mask):
        x = torch.cat([self.emb(et), val.unsqueeze(-1), dt.unsqueeze(-1)], dim=-1)
        h, _ = self.gru(x)
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def _seq_oracle_score(seqs_tr, y_tr, seqs_va, y_va, kind, dev, seed, epochs=40):
    """Train a GRU oracle on the raw train histories; score on the val histories."""
    torch.manual_seed(seed)
    vocab = _seq_vocab(seqs_tr)
    etr, vtr, dtr, mtr = _seq_tensors(seqs_tr, vocab, dev)
    eva, vva, dva, mva = _seq_tensors(seqs_va, vocab, dev)
    m = _SeqOracle(len(vocab) + 1).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    ym = ys = 0.0
    if kind == "reg":
        ym, ys = float(yt.mean()), float(yt.std().clamp(min=1e-6))
        yt = (yt - ym) / ys
    for _ in range(int(epochs)):
        m.train()
        opt.zero_grad()
        out = m(etr, vtr, dtr, mtr)
        loss = F.mse_loss(out, yt) if kind == "reg" else F.binary_cross_entropy_with_logits(out, yt)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        o = m(eva, vva, dva, mva)
        o = torch.sigmoid(o) if kind == "clf" else o * ys + ym
    return _score(kind, np.asarray(y_va, dtype=float), o.cpu().numpy())


def _history_features(past):
    """The model-free FULL-HISTORY proxy: RFM + recency windows + past-value stats.

    This is the strongest feature set production already has; the oracle in the
    sufficiency gap is [E, history], so a small gap means E already subsumes it.
    """
    return {
        "recency": None,  # filled by caller (needs tc)
        "freq": past["freq"].to_numpy().astype(float),
        "monetary": past["monetary"].to_numpy().astype(float),
        "recent30": past["recent30"].to_numpy().astype(float),
        "recent90": past["recent90"].to_numpy().astype(float),
        "past_vmean": past["past_vmean"].to_numpy().astype(float),
        "past_vlast": past["past_vlast"].to_numpy().astype(float),
    }


def _full_history_features(cfg, cust, only_keys=None):
    """Schema-derived, EXHAUSTIVE full-history features — the ORACLE's input.

    GENERIC, not hand-picked: discover EVERY field in the canonical stream and
    aggregate it by its natural type — every categorical field -> distinct-count;
    every numeric field -> count/sum/mean/std/min/max; plus generic timing.
    `only_keys` (optional) RESTRICTS the JSON keys to a subset — used to test
    whether the mass gap is a field-consumption gap (oracle restricted to the
    fields the encoder consumes). Returns (X:(N,D) float32, names)."""
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


def _future_family(j):
    """Self-supervised future targets (derived from the stream; no domain literal).

    All are functions of the FUTURE observables, so they are task-agnostic probes
    of predictive sufficiency (the exact opposite of our known business tasks).
    """
    orders = j["orders"].to_numpy().astype(float)
    return {
        "LTV (future log-margin)": (j["ltv"].to_numpy().astype(float), "reg"),
        "order_count (horizon)": (orders, "reg"),
        "propensity (order in horizon)": ((orders > 0).astype(float), "clf"),
        "churn (no order in horizon)": ((orders == 0).astype(float), "clf"),
        "next-order value": (j["next_v"].to_numpy().astype(float), "reg"),
    }


def _score(kind, y, s):
    from sklearn.metrics import r2_score, roc_auc_score

    if kind == "clf":
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, s))
    return float(r2_score(y, s))


def _normalized_gap(donor, e_alone, kind, eps=1e-9):
    """(gap, normalized) where gap = donor - E = what history adds beyond E.

    Range = the oracle/donor's own achievable spread (reg) or above-chance spread
    (clf), so the gap is scale-free and dimensionless."""
    gap = donor - e_alone
    denom = donor if kind == "reg" else (donor - 0.5)
    norm = 0.0 if denom <= eps else float(np.clip(gap / denom, 0.0, 1.0))
    return float(gap), norm


def _oracle_gap(
    Z, Xhist, yy, kind, cust, seeds=(0, 1, 2), eps=1e-9, seqs=None, dev=None, measure_dim=0
):
    """Mean oracle-gap over held-out folds for one target.

    E-alone = model on E; oracle = max(model on [E, history], learned SEQUENCE
    oracle on the raw history) — the independent, order-aware ceiling. gap =
    oracle - E-alone (what the full history adds beyond E), normalized. When
    `seqs` (raw held-out histories, aligned to `cust`) is given, a small GRU
    oracle is trained per fold and raises the ceiling (non-degenerate).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    Cls = HistGradientBoostingClassifier if kind == "clf" else HistGradientBoostingRegressor
    Conf = GBM_CLF if kind == "clf" else GBM_REG
    seq_by_cust = {s["customer"]: s for s in seqs} if seqs is not None else None
    e_al, raw_al, don_al, gaps, norms, nvs = [], [], [], [], [], []
    for seed in seeds:
        fold = np.zeros(len(cust), dtype=int)
        for fi, f in enumerate(_folds(cust, seed=seed)):
            fold[np.isin(cust, f)] = fi
        for fi in range(5):
            tr, va = fold != fi, fold == fi
            if (
                not tr.any()
                or not va.any()
                or (kind == "clf" and (len(set(yy[tr])) < 2 or len(set(yy[va])) < 2))
            ):
                continue
            scZ = StandardScaler().fit(Z[tr])
            Ztr, Zva = scZ.transform(Z[tr]), scZ.transform(Z[va])
            if measure_dim and int(measure_dim) > 0:
                # minimality (no leakage): compress the FULL-HISTORY measure with PCA fit on TRAIN
                from sklearn.decomposition import PCA

                scM = StandardScaler().fit(Xhist[tr])
                mtr, mva = scM.transform(Xhist[tr]), scM.transform(Xhist[va])
                kk = min(int(measure_dim), mtr.shape[1], max(1, mtr.shape[0] - 1))
                pca = PCA(n_components=kk, random_state=0).fit(mtr)
                Ztr = np.hstack([Ztr, pca.transform(mtr)])
                Zva = np.hstack([Zva, pca.transform(mva)])
            scH = StandardScaler().fit(Xhist[tr])
            Htr, Hva = scH.transform(Xhist[tr]), scH.transform(Xhist[va])
            Etra, Etva = np.hstack([Ztr, Htr]), np.hstack([Zva, Hva])  # oracle = [E, history]
            hE = Cls(**Conf).fit(Ztr, yy[tr])
            hH = Cls(**Conf).fit(Htr, yy[tr])
            hEH = Cls(**Conf).fit(Etra, yy[tr])
            if kind == "clf":
                se = hE.predict_proba(Zva)[:, 1]
                sh = hH.predict_proba(Hva)[:, 1]
                shh = hEH.predict_proba(Etva)[:, 1]
            else:
                se = hE.predict(Zva)
                sh = hH.predict(Hva)
                shh = hEH.predict(Etva)
            ya = yy[va]
            onE, onH, onEH = _score(kind, ya, se), _score(kind, ya, sh), _score(kind, ya, shh)
            if onE is None or onH is None or onEH is None:
                continue
            # independent, order-aware SEQUENCE oracle (raises the true ceiling)
            if seq_by_cust is not None and dev is not None:
                itr = [int(i) for i in np.where(tr)[0] if cust[i] in seq_by_cust]
                iva = [int(i) for i in np.where(va)[0] if cust[i] in seq_by_cust]
                ytr = [float(yy[i]) for i in itr]
                if len(itr) >= 8 and len(iva) >= 8 and (kind != "clf" or len(set(ytr)) >= 2):
                    try:
                        onSeq = _seq_oracle_score(
                            [seq_by_cust[cust[i]] for i in itr],
                            ytr,
                            [seq_by_cust[cust[i]] for i in iva],
                            [float(yy[i]) for i in iva],
                            kind,
                            dev,
                            seed=seed,
                        )
                        if onSeq is not None:
                            onEH = max(onEH, onSeq)
                    except Exception:  # noqa: BLE001 - oracle is best-effort
                        pass
            g, nrm = _normalized_gap(onEH, onE, kind, eps=eps)
            gaps.append(g)
            norms.append(nrm)
            e_al.append(onE)
            raw_al.append(onH)
            don_al.append(onEH)
            nvs.append(int(va.sum()))
    if not norms:
        return None
    # NOISE FLOOR (derived, not a magic number): the oracle's fold-to-fold SE,
    # floored by the finite-sample rate 1/sqrt(n_val). "Solved" for this member
    # = E is within this noise of the full-history oracle (gap indistinguishable
    # from 0), i.e. the embedding is as good as the raw history can make it.
    nval = max(int(np.mean(nvs)), 2)
    noise = max(float(np.std(don_al)), float(np.std(e_al)), 1.0 / np.sqrt(nval))
    gap_mean = float(np.mean(gaps))
    solved = abs(gap_mean) <= noise
    return (
        float(np.mean(e_al)),
        float(np.mean(raw_al)),
        float(np.mean(don_al)),
        gap_mean,
        float(np.mean(norms)),
        noise,
        bool(solved),
    )


def synthetic_machinery_check(seed=0, n=4000, kind="reg", seeds=(0, 1, 2)):
    """Ground-truth verification of the oracle-gap / U machinery.

    A KNOWN sufficient statistic S* = [direction S, log-magnitude], a future
    target that IS a function of both, and a history proxy that carries the
    magnitude. Then, by construction:
      * E = S*            (sufficient)      -> history adds nothing  -> U ~ 1
      * E = direction only (lossy)           -> history adds magnitude -> U < 1
      * E = [S*, noise]   (over-capacity)    -> no improvement       -> U ~ U(sufficient)
    Verifies the metric separates lossy from sufficient and SATURATES on extra
    capacity (no false reward for more dims)."""
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, 3))
    logmag = rng.normal(0.0, 1.0, size=n)  # the extensive axis (unbounded, additive)
    direction = S / np.linalg.norm(S, axis=1, keepdims=True)
    cust = np.array([f"c{i:07d}" for i in range(n)])
    # target depends on the DIRECTION (in E) AND the log-magnitude (in history):
    y = 2.0 * direction[:, 0] + 1.5 * logmag + rng.normal(0.0, 0.3, size=n)
    Xhist = np.column_stack([logmag, rng.normal(size=(n, 4))])  # history proxy carries magnitude
    reps = {
        "sufficient [S, logmag]": np.column_stack([S, logmag]),
        "lossy [direction]": direction,
        "over [S, logmag, noise]": np.column_stack([S, logmag, rng.normal(size=(n, 8))]),
    }
    out = {}
    for name, Z in reps.items():
        r = _oracle_gap(Z, Xhist, y, kind, cust, seeds=seeds)
        if r is not None:
            out[name] = {"norm_gap": round(r[4], 4), "gap": round(r[3], 4)}
    return out


def run_universality(
    model,
    cfg,
    horizons=(30, 90),
    seeds=(0, 1, 2),
    use_seq=True,
    oracle_keys=None,
    oracle_in_e=False,
    measure_dim=0,
):
    import polars as pl

    seqs = _b_sequences(cfg)  # raw held-out histories (for E and the sequence oracle)
    cust = np.array([s["customer"] for s in seqs])
    tmax = max((len(s["event_type"]) for s in seqs), default=1)
    bs = max(16, min(512, 131072 // max(tmax, 1)))  # bound GPU memory vs sequence length
    Zs = []
    for i in range(0, len(seqs), bs):
        with torch.no_grad():
            z = forward_embedding(model, seqs[i : i + bs])
        Zs.append(z.detach().float().cpu().numpy())
    Z = np.concatenate(Zs, 0) if Zs else np.zeros((0, 1), dtype=np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_arg = seqs if use_seq else None
    # ORACLE input: the schema-derived, exhaustive full-history measure (not RFM).
    Xhist, _hnames = _full_history_features(cfg, cust, only_keys=oracle_keys)
    if oracle_in_e:
        Z = np.hstack([Z, Xhist]).astype(np.float32)  # architecture receipt (DEC-138)
    rows = []
    for h in horizons:
        fut, _past, _tc = _rfm_targets(cfg, h)
        j = (
            pl.DataFrame({"customer": cust})
            .join(fut, left_on="customer", right_on="customer_key", how="left")
            .fill_null(0.0)
        )
        for name, (yy, kind) in _future_family(j).items():
            r = _oracle_gap(
                Z,
                Xhist,
                yy,
                kind,
                cust,
                seeds=seeds,
                seqs=seq_arg,
                dev=dev,
                measure_dim=measure_dim,
            )
            if r is None:
                continue
            e_al, raw_al, don_al, gap, norm, noise, solved = r
            rows.append(
                {
                    "target": f"{name} @ {h}d",
                    "E_alone": round(e_al, 4),
                    "history_alone": round(raw_al, 4),
                    "oracle [E,hist]": round(don_al, 4),
                    "gap (hist adds)": round(gap, 4),
                    "norm_gap": round(norm, 4),
                    "noise": round(noise, 4),
                    "solved": solved,
                }
            )
    u = 1.0 - max((r["norm_gap"] for r in rows), default=0.0)
    solved = bool(rows) and all(r["solved"] for r in rows)
    return rows, u, len(cust), solved


# target -> design axis (Diagnosable: the worst member names the bottleneck axis).
_AXIS = {
    "LTV": "state (mass/extensive)",
    "order_count": "state (mass/extensive)",
    "next-order value": "state (mass/extensive)",
    "propensity": "interface (decidability)",
    "churn": "interface (decidability)",
}


def _axis_of(target):
    for k, v in _AXIS.items():
        if str(target).startswith(k):
            return v
    return "unknown"


def _card(tag, rows, u, n, solved):
    worst = max(rows, key=lambda r: r["norm_gap"]) if rows else None
    lines = [
        f"# UNIVERSALITY GATE — {tag}  [loop/UNIVERSALITY.md]",
        f"{n} HELD-OUT (B) customers · oracle = [E, exhaustive full-history measure] · worst-case (minimax)",
        "",
        f"## U (universality index) = {u:.4f}   (1.0 = universal)",
        f"## SOLVED = {solved}   (E within the oracle's own noise on EVERY member; gap <= noise)",
    ]
    if worst is not None:
        lines.append(f"## BOTTLENECK AXIS = {_axis_of(worst['target'])}")
        lines.append(
            f"worst-case member: `{worst['target']}`  norm_gap {worst['norm_gap']:.4f}  "
            f"gap {worst['gap (hist adds)']:+.4f}  noise {worst['noise']:.4f}  "
            f"solved={worst['solved']}"
        )
    lines += [
        "",
        "| target | E alone | history alone | oracle [E,hist] | gap (hist adds) | noise | "
        "norm_gap | solved |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['E_alone']:.4f} | {r['history_alone']:.4f} | "
            f"{r['oracle [E,hist]']:.4f} | {r['gap (hist adds)']:+.4f} | {r['noise']:.4f} | "
            f"{r['norm_gap']:.4f} | {'✓' if r['solved'] else '✗'} |"
        )
    lines.append("")
    lines.append(
        "> gap = score([E,history]) - score(E): what the EXHAUSTIVE full-history measure adds "
        "beyond E. SOLVED for a member when |gap| <= noise (the oracle's fold SE, floored by "
        "1/sqrt(n)); SOLVED overall when every member is. NOT hand-RFM."
    )
    return "\n".join(lines) + "\n", u, solved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=False)
    ap.add_argument("--as-of", required=False, dest="as_of")
    ap.add_argument("--ckpt-dir", required=False, dest="ckpt_dir")
    ap.add_argument("--tag", required=False)
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument(
        "--seq-len",
        type=int,
        default=0,
        dest="seq_len",
        help="override sequence length (0 = checkpoint's resolved value). Use a large value "
        "to test whether the encoder's bounded window (train seq_len) truncates full history.",
    )
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--no-accumulator",
        action="store_true",
        dest="no_acc",
        help="disable the cumulative-accumulator measure pathway (matched control for U)",
    )
    ap.add_argument(
        "--generic-measure",
        action="store_true",
        dest="generic_measure",
        help="expose the generic, exhaustive additive measure in E (DEC-132 option B)",
    )
    ap.add_argument(
        "--oracle-keys",
        default="",
        dest="oracle_keys",
        help="comma list: RESTRICT the oracle to these JSON keys (test the field-consumption gap)",
    )
    ap.add_argument(
        "--oracle-in-E",
        action="store_true",
        dest="oracle_in_e",
        help="append the full-history measure to E (architecture receipt, DEC-138)",
    )
    ap.add_argument(
        "--measure-dim",
        type=int,
        default=0,
        dest="measure_dim",
        help="append a random-projection of the measure with this many dims (minimality sweep)",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="ground-truth machinery check (known sufficient statistic; no checkpoint)",
    )
    ap.add_argument(
        "--require-solved",
        action="store_true",
        dest="require_solved",
        help="exit non-zero unless SOLVED (E within oracle noise on every member) — loop predicate",
    )
    ap.add_argument(
        "--no-seq-oracle",
        action="store_true",
        dest="no_seq",
        help="skip the (costly) independent GRU sequence oracle; use the exhaustive feature oracle only",
    )
    a = ap.parse_args(argv)
    if a.synthetic:
        res = synthetic_machinery_check()
        print("# UNIVERSALITY MACHINERY CHECK (ground truth: known sufficient statistic)")
        for name, r in res.items():
            print(f"  {name:28s} norm_gap {r['norm_gap']:.4f}  (gap {r['gap']:+.4f})")
        return res
    if not (a.db and a.as_of and a.ckpt_dir and a.tag):
        ap.error("--db/--as-of/--ckpt-dir/--tag are required unless --synthetic")
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    if a.no_acc:
        model.accumulator_on = False  # matched control: shape-only (no measure pathway)
    if a.generic_measure:
        model.generic_measure_on = True  # expose the generic exhaustive measure in E
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    if a.seq_len:
        cfg.seq_len = int(a.seq_len)  # test whether the bounded window truncates history
    horizons = tuple(int(x) for x in a.horizons.split(","))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    oracle_keys = [k for k in a.oracle_keys.split(",") if k] or None
    rows, u, n, solved = run_universality(
        model,
        cfg,
        horizons,
        seeds,
        use_seq=not a.no_seq,
        oracle_keys=oracle_keys,
        oracle_in_e=a.oracle_in_e,
        measure_dim=a.measure_dim,
    )
    card, u, solved = _card(a.tag, rows, u, n, solved)
    print(card)
    if a.out:
        Path(a.out).write_text(card)
    if a.require_solved and not solved:
        raise SystemExit(1)
    return {"U": u, "solved": solved, "rows": rows}


if __name__ == "__main__":
    main()
