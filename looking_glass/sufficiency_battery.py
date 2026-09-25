"""Sufficiency battery — is the frozen encoder a UNIVERSAL donor for plugins?

The battery defines "universal donor" measurably: for a PORTFOLIO of downstream
targets spanning horizons, outcome types, and decision-relevant signals, does the
frozen embedding E add UNIQUE signal beyond raw RFM features R?

For every target (linear probe, grouped CV by customer, bootstrap CI):
  S signal    : E > scrambled E
  C standalone: E >= R - 0.05
  I unique    : partial corr(E, y | R) has 95% CI lower bound > 0

A universal donor must cover the portfolio, not just one target:
  UNIQUE COVERAGE = fraction of targets where I holds
  verdict UNIVERSAL iff unique_coverage >= threshold and S/C coverage high.

Encoder stays 100% self-supervised; labels only touch the frozen-embedding probes.
"""

from __future__ import annotations

import argparse

import numpy as np

from looking_glass.layer_b_proof import _load, cv_pred, _rho, _partial_ci


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


def build_all(stream, anch, orders, data_end):
    etypes = sorted(set(map(str, stream["event_type"].unique().to_list())))
    sc = {c: stream[c].to_list() for c in stream.columns}
    n, i = stream.height, 0
    sby = {}
    while i < n:
        k = sc["customer_key"][i]
        j = i
        while j < n and sc["customer_key"][j] == k:
            j += 1
        sby[k] = (
            np.array([_epoch(x) for x in sc["event_ts"][i:j]], dtype=np.float64),
            [str(x) for x in sc["event_type"][i:j]],
            np.array([_v(x) for x in sc["value"][i:j]]),
        )
        i = j
    oby = {}
    for r in orders.iter_rows(named=True):
        oby.setdefault(r["customer_id"], []).append((float(r["t"]), float(r["gm"])))
    E, R, groups = [], [], []
    T = {
        k: []
        for k in (
            "gp_365",
            "gp_90",
            "gp_30",
            "reorder_90",
            "reorder_30",
            "orders_90",
            "days_to_next",
            "next_order_value",
        )
    }
    D = 86400.0
    for row in anch.iter_rows(named=True):
        k = row["customer_key"]
        a = float(row["anchor_epoch"])
        if a + 365 * D > data_end or k not in sby:
            continue
        ts, et, val = sby[k]
        m = ts <= a
        if m.sum() < 3:
            continue
        pt, pe, pv = ts[m], np.array(et)[m], val[m]
        o = oby.get(k, [])
        ot = np.array([x[0] for x in o])
        og = np.array([x[1] for x in o])
        trail = float(og[(ot > a - 365 * D) & (ot <= a)].sum()) if len(o) else 0.0
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
        groups.append(k)
        fut = ot > a
        T["gp_365"].append(float(og[fut & (ot <= a + 365 * D)].sum()) if len(o) else 0.0)
        T["gp_90"].append(float(og[fut & (ot <= a + 90 * D)].sum()) if len(o) else 0.0)
        T["gp_30"].append(float(og[fut & (ot <= a + 30 * D)].sum()) if len(o) else 0.0)
        T["reorder_90"].append(1.0 if (fut & (ot <= a + 90 * D)).any() else 0.0)
        T["reorder_30"].append(1.0 if (fut & (ot <= a + 30 * D)).any() else 0.0)
        T["orders_90"].append(float((fut & (ot <= a + 90 * D)).sum()))
        T["days_to_next"].append(float(np.log1p((ot[fut].min() - a))) if fut.any() else np.nan)
        T["next_order_value"].append(float(og[fut][0]) if fut.any() else np.nan)
    return (
        np.array(E, np.float64),
        np.array(R, np.float64),
        np.array(groups),
        {k: np.array(v, np.float64) for k, v in T.items()},
    )


def run(threshold=0.6, seed=0):
    stream, anch, orders, data_end, tag = _load()
    E, R, groups, targets = build_all(stream, anch, orders, data_end)
    n = len(E)
    Scr = E[np.random.default_rng(seed).permutation(n)]
    rows = []
    for name, y in targets.items():
        mask = np.isfinite(y)
        if mask.sum() < 100:
            continue
        Xe, Xr, Xs, yy, gg = E[mask], R[mask], Scr[mask], y[mask], groups[mask]
        rE, rR, rS = (
            _rho(cv_pred(Xe, yy, gg), yy),
            _rho(cv_pred(Xr, yy, gg), yy),
            _rho(cv_pred(Xs, yy, gg), yy),
        )
        up, ulo, uhi = _partial_ci(cv_pred(Xe, yy, gg), cv_pred(Xr, yy, gg), yy, gg)
        rows.append(
            {
                "target": name,
                "n": int(mask.sum()),
                "E": rE,
                "R": rR,
                "scram": rS,
                "unique": up,
                "uci": (ulo, uhi),
                "S": rE > rS + 0.05,
                "C": rE >= rR - 0.05,
                "I": ulo > 0.0,
            }
        )
    sc = np.mean([r["S"] for r in rows])
    cc = np.mean([r["C"] for r in rows])
    ic = np.mean([r["I"] for r in rows])
    success = ic >= threshold and sc >= 0.8 and cc >= 0.8
    return {
        "version": str(tag),
        "rows": rows,
        "n": n,
        "signal_cov": sc,
        "standalone_cov": cc,
        "unique_cov": ic,
        "threshold": threshold,
        "success": bool(success),
    }


def show(res):
    print(
        f"== SUFFICIENCY BATTERY ==  version={res['version']}  anchors={res['n']}  "
        f"(encoder self-supervised only)"
    )
    print(
        f"{'target':18s} {'n':>5s} {'E':>7s} {'R':>7s} {'scram':>7s} "
        f"{'unique(E|R)':>12s} {'95% CI':>16s}  S C I"
    )
    for r in res["rows"]:
        print(
            f"{r['target']:18s} {r['n']:5d} {r['E']:7.3f} {r['R']:7.3f} {r['scram']:7.3f} "
            f"{r['unique']:+12.3f} [{r['uci'][0]:+.3f},{r['uci'][1]:+.3f}]  "
            f"{'Y' if r['S'] else '.'} {'Y' if r['C'] else '.'} {'Y' if r['I'] else '.'}"
        )
    print(
        f"\ncoverage: signal {res['signal_cov']:.0%} | standalone {res['standalone_cov']:.0%} "
        f"| UNIQUE {res['unique_cov']:.0%} (threshold {res['threshold']:.0%})"
    )
    print("SUFFICIENCY:", "UNIVERSAL DONOR" if res["success"] else "NOT YET")
    return res["success"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sufficiency battery for the CFM")
    ap.add_argument("--threshold", type=float, default=0.6)
    a = ap.parse_args(argv)
    return 0 if show(run(a.threshold)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
