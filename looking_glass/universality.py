"""Universality gate (v8): is `embedding(customer, t)` a UNIVERSAL sufficient statistic?

The goal (AGENTS.md "Project goal"; loop/UNIVERSALITY.md): the embedding must be
sufficient for the FULL observable future, so any present or future consumer is
covered by construction. Measured by the UNIVERSALITY INDEX

    U = 1 - worst-case_tau ( normalized oracle-gap_tau )

where, for each task tau in a self-supervised future family,

    gap_tau       = score([E, history]) - score(E)          # what history adds beyond E
    normalized    = gap_tau / range_tau                     # range = oracle's achievable spread
                    (reg: range = score([E,history]); clf: range = score - 0.5)

`history` here is the model-free full-history proxy (RFM + clock + past-value
stats = what production has anyway); the TRUE oracle (full sequence) is bounded
below by this, so the measured gap is a CONSERVATIVE estimate of the real
insufficiency. On the Layer-A lab (known truth) the gap is exact.

`U -> 1`  <=>  universal at this capacity. Worst-case (minimax), not average.

Run:
  python3 -m looking_glass.universality --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> [--customers 25000] [--horizons 30,90] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import (
    GBM_CLF,
    GBM_REG,
    _b_states,
    _folds,
    _rfm_targets,
)


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


def _oracle_gap(Z, Xhist, yy, kind, cust, seeds=(0, 1, 2), eps=1e-9):
    """Mean oracle-gap over held-out folds for one target.

    E-alone = model on E; history-alone = model on the history proxy; oracle
    [E,history] = model on both. gap = oracle - E-alone (what history adds beyond
    E), normalized to the oracle's own range. Returns (e_alone, hist_alone,
    oracle, gap, norm) means, or None if no usable fold.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    Cls = HistGradientBoostingClassifier if kind == "clf" else HistGradientBoostingRegressor
    Conf = GBM_CLF if kind == "clf" else GBM_REG
    e_al, raw_al, don_al, gaps, norms = [], [], [], [], []
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
            Ztr, Zva = StandardScaler().fit(Z[tr]).transform(Z[tr]), None
            scZ = StandardScaler().fit(Z[tr])
            Ztr, Zva = scZ.transform(Z[tr]), scZ.transform(Z[va])
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
            g, nrm = _normalized_gap(onEH, onE, kind, eps=eps)
            gaps.append(g)
            norms.append(nrm)
            e_al.append(onE)
            raw_al.append(onH)
            don_al.append(onEH)
    if not norms:
        return None
    return (
        float(np.mean(e_al)),
        float(np.mean(raw_al)),
        float(np.mean(don_al)),
        float(np.mean(gaps)),
        float(np.mean(norms)),
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


def run_universality(model, cfg, horizons=(30, 90), seeds=(0, 1, 2)):
    import polars as pl

    cust, Z = _b_states(model, cfg)
    rows = []
    for h in horizons:
        fut, past, tc = _rfm_targets(cfg, h)
        j = (
            pl.DataFrame({"customer": cust})
            .join(fut, left_on="customer", right_on="customer_key", how="left")
            .join(past, left_on="customer", right_on="customer_key", how="left")
            .fill_null(0.0)
        )
        H = _history_features(j)
        recency = (tc - j["last_t"].to_numpy().astype(float)) / 86400.0
        Xhist = np.column_stack([recency] + [H[k] for k in H if k != "recency"]).astype(float)

        for name, (yy, kind) in _future_family(j).items():
            r = _oracle_gap(Z, Xhist, yy, kind, cust, seeds=seeds)
            if r is None:
                continue
            e_al, raw_al, don_al, gap, norm = r
            rows.append(
                {
                    "target": f"{name} @ {h}d",
                    "E_alone": round(e_al, 4),
                    "history_alone": round(raw_al, 4),
                    "oracle [E,hist]": round(don_al, 4),
                    "gap (hist adds)": round(gap, 4),
                    "norm_gap": round(norm, 4),
                }
            )
    u = 1.0 - max((r["norm_gap"] for r in rows), default=0.0)
    return rows, u, len(cust)


def _card(tag, rows, u, n):
    worst = max(rows, key=lambda r: r["norm_gap"]) if rows else None
    lines = [
        f"# UNIVERSALITY GATE — {tag}  [loop/UNIVERSALITY.md]",
        f"{n} HELD-OUT (B) customers · oracle = [E, full-history proxy] · worst-case (minimax)",
        "",
        f"## U (universality index) = {u:.4f}   (1.0 = universal)",
    ]
    if worst is not None:
        lines.append(
            f"worst-case member: `{worst['target']}`  norm_gap {worst['norm_gap']:.4f}"
            f"  (history adds {worst['gap (hist adds)']:+.4f} over E)"
        )
    lines += [
        "",
        "| target | E alone | history alone | oracle [E,hist] | gap (hist adds) | norm_gap |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['E_alone']:.4f} | {r['history_alone']:.4f} | "
            f"{r['oracle [E,hist]']:.4f} | {r['gap (hist adds)']:+.4f} | {r['norm_gap']:.4f} |"
        )
    lines.append("")
    lines.append(
        "> gap = score([E,history]) - score(E): what the full-history proxy adds beyond E "
        "(0 => E sufficient w.r.t. it). Conservative (proxy oracle; true sequence oracle is higher)."
    )
    return "\n".join(lines) + "\n", u


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=False)
    ap.add_argument("--as-of", required=False, dest="as_of")
    ap.add_argument("--ckpt-dir", required=False, dest="ckpt_dir")
    ap.add_argument("--tag", required=False)
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="ground-truth machinery check (known sufficient statistic; no checkpoint)",
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
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    horizons = tuple(int(x) for x in a.horizons.split(","))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    rows, u, n = run_universality(model, cfg, horizons, seeds)
    card, u = _card(a.tag, rows, u, n)
    print(card)
    if a.out:
        Path(a.out).write_text(card)
    return {"U": u, "rows": rows}


if __name__ == "__main__":
    main()
