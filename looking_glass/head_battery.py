"""Layer-C downstream head battery — SPEC 0.9.5 §9.6 rigorous B+C check.

Requirements (all enforced here):
  1. Standardized head inputs (per-dim z-score).
  2. Donor grading: head reads [E, raw]; baseline reads raw -> PASS iff E adds
     incremental signal (SPEC C9). E-alone reported for the C9 ablation.
  3. HELD-OUT customers only (the encoder's B split) -- no train-customer leakage.
  4. STRONG raw baseline: RFM (recency/frequency/monetary) + gradient boosting.
  5. Effect size + bootstrap 95% CI over held-out customers, across >=3 seeds.
  6. Pre-registered gates + reproducibility receipt.

Verdict per head (SPEC §1.2): PASS / FAIL / NO_SIGNAL / UNCERTAIN (CI excludes 0).

Run: python3 -m looking_glass.head_battery --db <stream> --ckpt-dir <dir> --tag <t> \
        --as-of <date> --horizon-days 90 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from looking_glass.cfm_config import CFMConfig, as_of_epoch
from looking_glass.cfm_data import (
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
)
from looking_glass.cfm_training import forward_states
from looking_glass.probes import freeze_encoder

# ---- PRE-REGISTERED GATE (fixed before evaluation) --------------------------
GATE = {
    "pass": "bootstrap 95% CI of Δ(donor − raw) excludes 0 AND mean Δ > 0",
    "no_signal": "raw baseline itself at chance (AUROC<=0.5, or R2<=0)",
    "uncertain": "CI straddles 0 and raw carries signal",
    "bootstrap": 2000,
    "seeds": 3,
    "folds": 5,
}
N_FOLDS = 5
GBM_REG = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
    "max_depth": 4,
}
GBM_CLF = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
    "max_depth": 4,
}


def _folds(customers, n=N_FOLDS, seed=1):
    cs = sorted(set(np.asarray(customers).tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cs)
    return np.array_split(np.asarray(cs, dtype=object), n)


def _b_states(model, cfg, batch=512):
    """Frozen donor states for HELD-OUT customers (split B) only."""
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    bkeys = [k for k in keys if split.get(k) == "B"]
    seqs = build_sequences(df, bkeys, cfg, split, with_anchors=False)
    Z = []
    for i in range(0, len(seqs), batch):
        with torch.no_grad():
            z = model.donor_batch(forward_states(model, seqs[i : i + batch]))
        Z.append(z.detach().float().cpu().numpy())
    return np.array([s["customer"] for s in seqs]), np.concatenate(Z, 0)


def _rfm_targets(cfg, horizon_days, lookback_days=365):
    """Per-customer RFM (past) + future targets (SQL, canonical stream)."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    hi, lo = tc + horizon_days * 86400.0, tc - lookback_days * 86400.0
    base = (
        f"SELECT customer_key, epoch(CAST(event_ts AS TIMESTAMPTZ)) t,"
        f" CAST(json_extract_string(event_attributes,'$.value') AS DOUBLE) v"
        f" FROM {src} WHERE event_type='order_placed'"
    )
    fut = con.execute(
        f"SELECT customer_key, sum(ln(1+abs(v))) ltv, count(*) orders,"
        f" arg_min(v, t) next_v, min(t) first_t"
        f" FROM ({base}) WHERE t>{tc} AND t<={hi} GROUP BY 1"
    ).pl()
    past = con.execute(
        f"SELECT customer_key, count(*) freq, sum(ln(1+abs(v))) monetary,"
        f" max(t) last_t"
        f" FROM ({base}) WHERE t>{lo} AND t<={tc} GROUP BY 1"
    ).pl()
    con.close()
    return fut, past, tc


def _metrics(kind, y, s_donor, s_base, w=None):
    w = np.ones(len(y)) if w is None else w
    if kind == "clf":
        if len(set(y)) < 2:
            return None, None
        return roc_auc_score(y, s_donor, sample_weight=w), roc_auc_score(y, s_base, sample_weight=w)
    ss = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2d = 1 - np.sum(w * (y - s_donor) ** 2) / max(ss, 1e-12)
    r2b = 1 - np.sum(w * (y - s_base) ** 2) / max(ss, 1e-12)
    return r2d, r2b


def run_head_battery(model, cfg, horizon_days=90, seeds=(0, 1, 2)):
    freeze_encoder(model)
    cust, Z = _b_states(model, cfg)
    fut, past, tc = _rfm_targets(cfg, horizon_days)
    import polars as pl

    j = (
        pl.DataFrame({"customer": cust})
        .join(fut, left_on="customer", right_on="customer_key", how="left")
        .join(past, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    # RFM features (recency small = recently active)
    recency = (tc - j["last_t"].to_numpy().astype(float)) / 86400.0
    Xraw = np.column_stack(
        [recency, j["freq"].to_numpy().astype(float), j["monetary"].to_numpy().astype(float)]
    ).astype(float)
    yall = {
        "LTV (future log-margin)": (j["ltv"].to_numpy().astype(float), "reg"),
        "churn (no order in horizon)": (
            (j["orders"].to_numpy().astype(float) == 0).astype(float),
            "clf",
        ),
        "purchase propensity (order in horizon)": (
            (j["orders"].to_numpy().astype(float) > 0).astype(float),
            "clf",
        ),
        "order count (horizon)": (j["orders"].to_numpy().astype(float), "reg"),
        "next-order value": (j["next_v"].to_numpy().astype(float), "reg"),
    }
    from sklearn.preprocessing import StandardScaler

    rows = []
    for name, (yy, kind) in yall.items():
        if kind == "clf" and len(set(yy)) < 2:
            rows.append(
                {
                    "head": name,
                    "n": len(yy),
                    "delta": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "e_alone": 0.0,
                    "raw": 0.0,
                    "verdict": "NO_SIGNAL",
                }
            )
            continue
        deltas = []  # bootstrap Δ samples pooled across folds/seeds
        raw_scores, e_scores = [], []
        for seed in seeds:
            fold = np.zeros(len(cust), dtype=int)
            for fi, f in enumerate(_folds(cust, seed=seed)):
                fold[np.isin(cust, f)] = fi
            for fi in range(N_FOLDS):
                tr, va = fold != fi, fold == fi
                if (
                    not tr.any()
                    or not va.any()
                    or (kind == "clf" and (len(set(yy[tr])) < 2 or len(set(yy[va])) < 2))
                ):
                    continue
                sc = StandardScaler().fit(Z[tr])
                Ztr, Zva = sc.transform(Z[tr]), sc.transform(Z[va])
                sc2 = StandardScaler().fit(Xraw[tr])
                Rtr, Rva = sc2.transform(Xraw[tr]), sc2.transform(Xraw[va])
                Etva = np.hstack([Zva, Rva])
                Etra = np.hstack([Ztr, Rtr])
                if kind == "clf":
                    hd = HistGradientBoostingClassifier(**GBM_CLF).fit(Etra, yy[tr])
                    hb = HistGradientBoostingClassifier(**GBM_CLF).fit(Rtr, yy[tr])
                    he = HistGradientBoostingClassifier(**GBM_CLF).fit(Ztr, yy[tr])
                    sd, sb, se_ = (
                        hd.predict_proba(Etva)[:, 1],
                        hb.predict_proba(Rva)[:, 1],
                        he.predict_proba(Zva)[:, 1],
                    )
                else:
                    hd = HistGradientBoostingRegressor(**GBM_REG).fit(Etra, yy[tr])
                    hb = HistGradientBoostingRegressor(**GBM_REG).fit(Rtr, yy[tr])
                    he = HistGradientBoostingRegressor(**GBM_REG).fit(Ztr, yy[tr])
                    sd, sb, se_ = hd.predict(Etva), hb.predict(Rva), he.predict(Zva)
                ya = yy[va]
                d, b = _metrics(kind, ya, sd, sb)
                e, _ = _metrics(kind, ya, se_, sb)
                if d is None:
                    continue
                raw_scores.append(b)
                e_scores.append(e)
                rng = np.random.default_rng(seed * 100 + fi)
                for _ in range(GATE["bootstrap"] // (len(seeds) * N_FOLDS)):
                    idx = rng.integers(0, len(ya), len(ya))
                    dd, _bb = _metrics(kind, ya[idx], sd[idx], sb[idx])
                    if dd is not None:
                        deltas.append(dd - _bb)
        if not deltas:
            rows.append(
                {
                    "head": name,
                    "n": len(cust),
                    "delta": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "e_alone": 0.0,
                    "raw": 0.0,
                    "verdict": "NO_SIGNAL",
                }
            )
            continue
        deltas = np.array(deltas)
        mean_d = float(deltas.mean())
        lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
        rawsig = float(np.mean(raw_scores)) if raw_scores else 0.5
        ealone = float(np.mean(e_scores)) if e_scores else rawsig
        carried = (rawsig > 0.5 + 1e-6) if kind == "clf" else (rawsig > 0.0)
        if not carried:
            verdict = "NO_SIGNAL"
        elif lo > 0:
            verdict = "PASS"
        elif hi < 0:
            verdict = "FAIL"
        else:
            verdict = "UNCERTAIN"
        rows.append(
            {
                "head": name,
                "n": len(cust),
                "delta": mean_d,
                "lo": lo,
                "hi": hi,
                "e_alone": ealone,
                "raw": rawsig,
                "verdict": verdict,
            }
        )
    # unsupervised
    from sklearn.cluster import KMeans

    yv = yall["LTV (future log-margin)"][0]
    km = KMeans(n_clusters=4, n_init=5, random_state=0).fit(Z)
    eta2 = 1.0 - np.mean(
        [yv[km.labels_ == c].var() for c in range(4) if (km.labels_ == c).sum() > 1]
    ) / max(yv.var(), 1e-9)
    rows.append(
        {
            "head": "unsupervised segmentation (eta^2 on LTV)",
            "n": len(cust),
            "delta": float(eta2),
            "lo": float(eta2),
            "hi": float(eta2),
            "e_alone": float(eta2),
            "raw": float("nan"),
            "verdict": "PASS" if eta2 > 0.02 else "NO_SIGNAL",
        }
    )
    receipt = {
        "stream": None,
        "counterfactual_guard": "held-out split B only",
        "n_heldout": int(len(cust)),
        "horizon_days": horizon_days,
        "seeds": list(seeds),
        "gate": GATE,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    return rows, receipt


def _card(stream, tag, rows, receipt):
    fails = [r["head"] for r in rows if r["verdict"] == "FAIL"]
    unc = [r["head"] for r in rows if r["verdict"] == "UNCERTAIN"]
    green = not fails and not unc
    lines = [
        f"# Layer-C (B+C) Head Battery — {stream}  [SPEC 0.9.5 §9.6]",
        f"frozen `{tag}` · {receipt['n_heldout']} HELD-OUT (B) customers · horizon {receipt['horizon_days']}d · "
        f"donor head `[E, RFM]` vs RFM+GBM baseline · bootstrap 95% CI · seeds {receipt['seeds']}",
        f"gate: {GATE['pass']}   git `{receipt['git_sha']}`",
        "",
        f"## VERDICT: {'GREEN ✅' if green else 'NOT GREEN ❌'}",
    ]
    if fails:
        lines.append(f"BLOCKERS (FAIL): {fails}")
    if unc:
        lines.append(f"UNCERTAIN: {unc}")
    lines += [
        "",
        "| head | Δ (donor−raw) 95% CI | E alone | raw | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
        lines.append(
            f"| {r['head']} | {ci} | {r['e_alone']:.4f} | {r['raw']:.4f} | {r['verdict']} |"
        )
    return "\n".join(lines) + "\n", green


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from looking_glass.cfm_state import load_frozen_encoder

    model, _ = load_frozen_encoder(args.tag, args.ckpt_dir)
    cfg = CFMConfig()
    cfg.db, cfg.as_of = args.db, args.as_of
    cfg.sample_customers = args.customers
    seeds = tuple(int(x) for x in args.seeds.split(","))
    rows, receipt = run_head_battery(model, cfg, args.horizon_days, seeds)
    receipt["stream"] = args.stream or Path(args.db).parent.name
    card, green = _card(receipt["stream"], args.tag, rows, receipt)
    print(card)
    if args.out:
        Path(args.out).write_text(card)
        Path(args.out).with_suffix(".json").write_text(
            json.dumps({"receipt": receipt, "rows": rows}, indent=1, default=float)
        )
    return {"green": green, "rows": rows}


if __name__ == "__main__":
    main()
