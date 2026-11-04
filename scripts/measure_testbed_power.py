#!/usr/bin/env python3
"""Testbed power — can this stream distinguish a sufficient statistic from an aggregate? (DEC-520)

Read-only. It changes no gate and no generator.

The question P2 asks: if the future is generated from a *latent state*, can a table of aggregate
summaries (counts, sums, recency — what PATH A encodes) recover it as well as the latent truth
itself? If the aggregate matches the oracle, the stream cannot test the learned-encoder bet and
PATH A can only lose by luck. If the oracle strictly beats the aggregate, the stream has power and
the gap is the headroom a learned representation must close.

Pre-registered decision rule (DEC-520):
  * ANY pre-registered task with (oracle - aggregate) paired-CI lower bound > 0 -> HAS POWER.
  * EVERY task within noise -> INSUFFICIENT -> rebuild Layer A.

    python3 scripts/measure_testbed_power.py --demo /home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import _paired_ci, _ridge_oos  # noqa: E402
from looking_glass.cfm_config import CFMConfig  # noqa: E402
from looking_glass.layer_b import generic_measure  # noqa: E402

SECONDS_PER_DAY = 86400.0  # unit conversion, not a cadence assumption

#: pre-registered targets. Each is a FORWARD-window outcome computed from the stream after as_of.
TARGETS = {
    "fwd_value_90d": "sum(value) on purchase events in (as_of, as_of+90d]",
    "fwd_count_90d": "count(*) on purchase events in (as_of, as_of+90d]",
    "fwd_return_90d": "returns / purchases in the forward window",
}

#: the generator's STATIC latent truth — the sufficient-statistic proxy.
#: `churn_day` and `regime_runs` are deliberately excluded: they are realised FUTURE paths, so
#: using them as features would be clairvoyance, not an oracle.
LATENT_NUM = ("x0", "h0", "d0", "price_sensitivity", "return_proneness", "segment")
#: when the generator emits its Bayes conditional mean, that IS the sufficient statistic.
BAYES = ("bayes_fwd_value_90d", "bayes_fwd_count_90d")
LATENT_CAT = ("offer_opt", "channel_pref", "category_pref", "regime0")


def _latent_matrix(cust, truth) -> tuple[np.ndarray, list[str]]:
    """Static latent truth, numeric + one-hot. No future-realised fields."""
    rows = [truth[k] for k in cust]
    extra = tuple(k for k in BAYES if k in (rows[0] if rows else {}))
    cols = LATENT_NUM + extra
    num = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)
    names = list(cols)
    cats: list[str] = []
    for c in LATENT_CAT:
        vals = sorted({str(r.get(c)) for r in rows})
        cats += [f"{c}={v}" for v in vals]
        num = np.hstack(
            [
                num,
                np.array([[1.0 if str(r.get(c)) == v else 0.0 for v in vals] for r in rows]),
            ]
        )
    tp = np.array([list(map(float, r.get("touch_pref") or [])) for r in rows], dtype=np.float64)
    if tp.ndim == 2 and tp.shape[1]:
        num = np.hstack([num, tp])
        names += [f"touch_pref_{i}" for i in range(tp.shape[1])]
    return num, names + cats


def _forward_targets(db: Path, as_of: str, horizon_days: int) -> dict:
    """Forward-window outcomes in the stream, keyed by customer. One DuckDB pass per target."""
    import duckdb
    from datetime import datetime, timezone

    t0 = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc).timestamp()
    t1 = t0 + horizon_days * SECONDS_PER_DAY
    con = duckdb.connect()
    src = f"read_parquet('{db}')" if str(db).endswith((".parquet", ".pq")) else "customer_events"
    base = (
        f"SELECT customer_key AS ck, epoch(CAST(event_ts AS TIMESTAMPTZ)) AS t, "
        f"event_type AS et, CAST(json_extract(event_attributes,'$.value') AS DOUBLE) AS v "
        f"FROM {src} WHERE epoch(CAST(event_ts AS TIMESTAMPTZ)) > {t0} "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {t1}"
    )
    q = f"""
        WITH f AS ({base})
        SELECT ck,
            sum(CASE WHEN et = 'purchase' THEN coalesce(v,0) ELSE 0 END) AS fwd_value_90d,
            count(*) FILTER (WHERE et = 'purchase') AS fwd_count_90d,
            count(*) FILTER (WHERE et = 'return') AS _ret
        FROM f GROUP BY 1
    """
    rows = con.execute(q).fetchall()
    con.close()
    out = {}
    for ck, val, cnt, ret in rows:
        out[ck] = {
            "fwd_value_90d": float(val or 0.0),
            "fwd_count_90d": float(cnt or 0.0),
            "fwd_return_90d": float(ret / cnt) if cnt else 0.0,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--as-of", default="2025-01-01")
    ap.add_argument("--horizon-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs/report_cards/experiments/testbed_power.json"))
    a = ap.parse_args(argv)

    demo = Path(a.demo)
    manifest = json.loads((demo / "manifest.json").read_text())
    stream = demo / manifest["paths"]["B"]["stream"]
    truth_doc = json.loads((demo / "reference_truth.json").read_text())
    truth = {c["customer"]: c for c in truth_doc["customers"]}

    fwd = _forward_targets(stream, a.as_of, a.horizon_days)
    cust = sorted(set(truth) & set(fwd))
    if len(cust) < 200:
        print(f"only {len(cust)} customers with truth+future — too few", file=sys.stderr)
        return 2

    cfg = CFMConfig()
    cfg.db = str(stream)
    cfg.as_of = a.as_of
    agg, agg_names = generic_measure(cfg, cust)
    agg = np.asarray(agg, dtype=np.float64)
    lat, lat_names = _latent_matrix(cust, truth)

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(cust))
    n_a = int(0.7 * len(cust))
    A, B = idx[:n_a], idx[n_a:]

    results = {}
    for tname in TARGETS:
        y = np.array([fwd[k][tname] for k in cust], dtype=np.float64)
        if float(np.std(y)) < 1e-12:
            results[tname] = {"degenerate": True}
            continue
        p_agg, r2_agg = _ridge_oos(agg[A], y[A], agg[B], y[B])
        p_lat, r2_lat = _ridge_oos(lat[A], y[A], lat[B], y[B])
        p_both, r2_both = _ridge_oos(np.hstack([agg, lat])[A], y[A], np.hstack([agg, lat])[B], y[B])
        if p_agg is None or p_lat is None:
            results[tname] = {"degenerate": True}
            continue
        # per-customer squared-error advantage of the oracle over the aggregate
        delta = (p_agg - y[B]) ** 2 - (p_lat - y[B]) ** 2
        mean, lo, hi = _paired_ci(delta)
        results[tname] = {
            "n": int(len(B)),
            "r2_aggregate": round(float(r2_agg), 4),
            "r2_oracle": round(float(r2_lat), 4),
            "r2_both": round(float(r2_both), 4),
            "oracle_minus_aggregate_mse": round(float(mean), 6),
            "ci_lo": round(float(lo), 6),
            "ci_hi": round(float(hi), 6),
            "oracle_strictly_better": bool(lo > 0),
        }

    live = {k: v for k, v in results.items() if not v.get("degenerate")}
    has_power = any(v["oracle_strictly_better"] for v in live.values())
    body = {
        "_meta": {
            "producer": Path(__file__).name,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join([Path(__file__).name] + sys.argv[1:]),
            "read_only": True,
            "demo": str(demo),
            "stream": str(stream),
            "stream_sha256": hashlib.sha256(stream.read_bytes()).hexdigest()[:16],
            "as_of": a.as_of,
            "horizon_days": int(a.horizon_days),
            "n_customers": int(len(cust)),
            "n_fit": int(len(A)),
            "n_score": int(len(B)),
            "n_aggregate_columns": int(agg.shape[1]),
            "n_latent_columns": int(lat.shape[1]),
            "targets": TARGETS,
            "oracle_features": "static generator latents only; churn_day/regime_runs excluded "
            "(realised future paths would be clairvoyance)",
        },
        "tasks": results,
        "verdict": {
            "has_power": bool(has_power),
            "rule": "ANY task with oracle-minus-aggregate paired-CI lower bound > 0 -> HAS POWER; "
            "EVERY task within noise -> INSUFFICIENT -> rebuild Layer A (DEC-520)",
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n")

    print(f"\ntestbed power -> {out}")
    print(f"  n={len(cust)} (fit {len(A)} / score {len(B)})  as_of={a.as_of}")
    print(f"  aggregate measure {agg.shape[1]} cols | latent oracle {lat.shape[1]} cols")
    for k, v in live.items():
        print(
            f"  {k:16s} R2 agg {v['r2_aggregate']:+.4f}  oracle {v['r2_oracle']:+.4f}"
            f"  both {v['r2_both']:+.4f}  oracle-agg MSE gain {v['oracle_minus_aggregate_mse']:+.5f}"
            f"  CI [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}]"
            f"  {'ORACLE WINS' if v['oracle_strictly_better'] else 'tie'}"
        )
    print(f"  VERDICT: {'HAS POWER' if has_power else 'INSUFFICIENT — rebuild Layer A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
