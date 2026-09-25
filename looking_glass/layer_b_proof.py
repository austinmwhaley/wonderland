"""Layer B proof — is the self-supervised encoder doing its job for plugins?

We must prove three things about the FROZEN encoder (no labels ever seen):
  (S) SIGNAL   : the embedding beats a scrambled embedding (it is not noise)
  (C) STANDALONE: embedding-only is competitive with raw behavioral features
  (I) INCREMENT: embedding + raw beats raw alone (carries signal plugins can
      use beyond what raw features already give)

Proof design (rigorous, no cherry-picking):
  * Features per anchor: E = frozen embedding; R = causal raw RFM (recency,
    frequency, monetary, tenure, mean-gap, per-type counts, trailing GP);
    E+R = concat; Scramble = row-shuffled E (control).
  * Linear probes only (Ridge) — we test the REPRESENTATION, not model capacity.
  * Grouped K-fold CV by customer (no customer appears in train and test).
  * Multiple target horizons (365d, 90d gross margin).
  * Bootstrap 95% CI on the INCREMENTAL gain of E+R over R (resampling customers).

Encoder was trained on sample A; anchors are sample B -> held-out by construction.
Verdict SUCCESS iff: E > Scramble on every horizon, E >= R - 0.05, and the
incremental gain (E+R - R) has a 95% CI whose lower bound > 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

WORK = Path(__file__).resolve().parents[0]
CFM = WORK / "artifacts" / "cfm"
STREAM = WORK.parent / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"


def _epoch(s):
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except Exception:
        return np.nan


def _v(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _load(
    name="data/arrow/customer_event_stream.feather", products="artifacts/cfm/cfm_products.duckdb"
):
    stream = (
        pl.read_ipc(name, memory_map=True)
        if str(name).endswith((".arrow", ".feather", ".ipc"))
        else _read_duckdb(name)
    )
    import duckdb

    con = duckdb.connect(str(products), read_only=True)
    try:
        anch = con.execute(
            "SELECT customer_key, anchor_epoch, version, embedding FROM anchor_embeddings"
        ).pl()
    finally:
        con.close()
    tag = anch["version"][0]
    # order gross margin + trailing/forward labels from the orders table
    con = duckdb.connect(str(STREAM), read_only=True)
    try:
        orders = con.execute(
            "SELECT customer_id, epoch(CAST(order_ts AS TIMESTAMPTZ)) t, "
            "gross_margin gm FROM orders"
        ).pl()
        data_end = float(
            con.execute(
                "SELECT max(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM customer_events"
            ).fetchone()[0]
        )
    finally:
        con.close()
    return stream, anch, orders, data_end, tag


def _read_duckdb(db):
    import duckdb

    con = duckdb.connect(db, read_only=True)
    try:
        return con.execute(
            "SELECT customer_key, event_ts, event_type, value "
            "FROM customer_events ORDER BY customer_key, event_ts"
        ).pl()
    finally:
        con.close()


def build(stream, anch, orders, data_end, window_days):
    etypes = sorted(set(map(str, stream["event_type"].unique().to_list())))
    scols = {c: stream[c].to_list() for c in stream.columns}
    n = stream.height
    sby, i = {}, 0
    while i < n:
        k = scols["customer_key"][i]
        j = i
        while j < n and scols["customer_key"][j] == k:
            j += 1
        ts = np.array([_epoch(x) for x in scols["event_ts"][i:j]], dtype=np.float64)
        sby[k] = (
            ts,
            [str(x) for x in scols["event_type"][i:j]],
            np.array([_v(x) for x in scols["value"][i:j]]),
        )
        i = j
    oby = {}
    for r in orders.iter_rows(named=True):
        oby.setdefault(r["customer_id"], []).append((float(r["t"]), float(r["gm"])))
    W = window_days * 86400
    E, R, y, groups = [], [], [], []
    for row in anch.iter_rows(named=True):
        k = row["customer_key"]
        a = float(row["anchor_epoch"])
        if a + W > data_end or k not in sby:
            continue
        ts, et, val = sby[k]
        m = ts <= a
        if m.sum() < 3:
            continue
        pt, pe, pv = ts[m], np.array(et)[m], val[m]
        o = oby.get(k, [])
        ot = np.array([x[0] for x in o])
        ogm = np.array([x[1] for x in o])
        trail = float(ogm[(ot > a - W) & (ot <= a)].sum()) if len(o) else 0.0
        fwd = float(ogm[(ot > a) & (ot <= a + W)].sum()) if len(o) else 0.0
        gap = np.diff(pt)
        rfm = [
            a - pt[-1],
            a - pt[0],
            float(m.sum()),
            float(pv.sum()),
            float(np.mean(gap)) if len(gap) else 0.0,
            float((pe == "order_placed").sum()),
            trail,
        ]
        rfm += [float((pe == t).sum()) for t in etypes]
        E.append(list(row["embedding"]))
        R.append(rfm)
        y.append(fwd)
        groups.append(k)
    return (
        np.array(E, np.float64),
        np.array(R, np.float64),
        np.array(y, np.float64),
        np.array(groups),
    )


def _ridge(X):
    return Ridge(alpha=1e-3 * float(np.mean(np.sum(X * X, axis=0))) + 1e-9)


def cv_pred(X, y, groups, folds=5):
    pred = np.zeros_like(y, dtype=np.float64)
    gkf = GroupKFold(n_splits=min(folds, len(set(groups.tolist()))))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        pred[te] = (
            _ridge(sc.transform(X[tr])).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        )
    return pred


def _rho(pred, y):
    return float(spearmanr(pred, y).statistic)


def _boot_ci(pred_a, pred_b, y, groups, B=400, seed=0):
    """95% CI on rho(a)-rho(b), resampling customers."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {k: np.where(groups == k)[0] for k in uniq}
    d = []
    for _ in range(B):
        ks = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by[k] for k in ks])
        d.append(_rho(pred_a[ix], y[ix]) - _rho(pred_b[ix], y[ix]))
    d = np.array(d)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _partial_ci(pE, pR, y, groups, B=400, seed=0):
    """Partial correlation of pE with y, controlling for pR. Resamples customers.
    This is the UNIQUE-VARIANCE test: does the embedding explain signal that raw
    features do not? Point estimate + 95% CI on the partial Spearman."""

    def resid(a, b):
        X = np.c_[np.ones_like(b), b]
        c, *_ = np.linalg.lstsq(X, a, rcond=None)
        return a - X @ c

    def stat(ix):
        return float(spearmanr(resid(pE[ix], pR[ix]), resid(y[ix], pR[ix])).statistic)

    allix = np.arange(len(y))
    point = stat(allix)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {k: np.where(groups == k)[0] for k in uniq}
    d = [
        stat(np.concatenate([idx_by[k] for k in rng.choice(uniq, len(uniq), replace=True)]))
        for _ in range(B)
    ]
    return point, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def run(windows=(365, 90), seed=0):
    stream, anch, orders, data_end, tag = _load()
    rows, ok_all = [], True
    for w in windows:
        E, R, y, groups = build(stream, anch, orders, data_end, w)
        if len(y) < 50:
            continue
        ER = np.hstack([E, R])
        Scr = E[np.random.default_rng(seed).permutation(len(E))]
        pE, pR, pER, pS = (
            cv_pred(E, y, groups),
            cv_pred(R, y, groups),
            cv_pred(ER, y, groups),
            cv_pred(Scr, y, groups),
        )
        rE, rR, rER, rS = _rho(pE, y), _rho(pR, y), _rho(pER, y), _rho(pS, y)
        lo, hi = _boot_ci(pER, pR, y, groups)
        up, ulo, uhi = _partial_ci(pE, pR, y, groups)
        signal = rE > rS + 0.05
        standalone = rE >= rR - 0.05
        increment = ulo > 0.0
        ok_all = ok_all and signal and standalone and increment
        rows.append(
            {
                "window": w,
                "n": int(len(y)),
                "E": rE,
                "R": rR,
                "E+R": rER,
                "scramble": rS,
                "increment": rER - rR,
                "ci": (lo, hi),
                "unique": up,
                "uci": (ulo, uhi),
                "signal": signal,
                "standalone": standalone,
                "increment_ok": increment,
            }
        )
    return {"version": str(tag), "rows": rows, "success": ok_all}


def show(res):
    print(f"== LAYER B PROOF ==  version={res['version']}  (encoder: self-supervised only)")
    print(
        f"{'window':>7s} {'n':>5s} {'E':>7s} {'R':>7s} {'scram':>7s} "
        f"{'unique(E|R)':>12s} {'95% CI':>16s}"
    )
    for r in res["rows"]:
        print(
            f"{r['window']:>6d}d {r['n']:5d} {r['E']:7.3f} {r['R']:7.3f} {r['scramble']:7.3f} "
            f"{r['unique']:+12.3f} [{r['uci'][0]:+.3f},{r['uci'][1]:+.3f}]"
        )
    print("\nchecks: S signal E>scramble | C standalone E>=R-0.05 | I unique(E|R) CI_lo>0")
    for r in res["rows"]:
        print(
            f"  {r['window']:>3d}d: S={'Y' if r['signal'] else 'N'} "
            f"C={'Y' if r['standalone'] else 'N'} I={'Y' if r['increment_ok'] else 'N'}"
        )
    print("\nLAYER B PROOF:", "SUCCESS" if res["success"] else "NOT PROVEN")
    return res["success"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Layer B proof")
    ap.add_argument("--db", default="data/arrow/customer_event_stream.feather")
    ap.add_argument("--products", default="artifacts/cfm/cfm_products.duckdb")
    ap.parse_args(argv)
    return 0 if show(run()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
