"""Point-in-time decision gate using the E's RESPONSE BLOCK + DiD null control.

For each action key: choose the argmax response level per customer from the
consumed E, compare the response at the chosen vs the logged FUTURE action, and
require real_lift_lo - null_lift_hi > 0 (a strict, null-calibrated lift).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = ("freq", "hour", "dow", "channel", "offer")


def _run(tag, ckpt, db, as_of, customers, horizon, period):
    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.use_case_gate import run_use_case_response

    m, cfg = load_frozen_encoder(tag, ckpt)
    cfg.attr_fields = tuple(getattr(m, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(m, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = db, as_of, customers
    cfg.action_period_seconds = float(period)
    out = {}
    for k in KEYS:
        try:
            out[k] = run_use_case_response(m, cfg, k, horizon_days=horizon)
        except Exception as e:  # noqa: BLE001
            out[k] = {
                "action_key": k,
                "error": str(e),
                "lift": None,
                "lo": None,
                "hi": None,
                "n": 0,
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-db", required=True, dest="real_db")
    ap.add_argument("--real-ckpt", required=True, dest="real_ckpt")
    ap.add_argument("--real-tag", required=True, dest="real_tag")
    ap.add_argument("--null-db", required=True, dest="null_db")
    ap.add_argument("--null-ckpt", required=True, dest="null_ckpt")
    ap.add_argument("--null-tag", required=True, dest="null_tag")
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--horizon-days", type=int, default=180, dest="horizon_days")
    ap.add_argument("--period-seconds", type=float, default=86400.0, dest="period_seconds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    real = _run(
        a.real_tag, a.real_ckpt, a.real_db, a.as_of, a.customers, a.horizon_days, a.period_seconds
    )
    null = _run(
        a.null_tag, a.null_ckpt, a.null_db, a.as_of, a.customers, a.horizon_days, a.period_seconds
    )
    rows = {}
    for k in KEYS:
        r, n = real.get(k, {}), null.get(k, {})
        rl, rh, nl, nh = r.get("lo"), r.get("hi"), n.get("lo"), n.get("hi")
        diff = (rl - nh) if (rl is not None and nh is not None) else None
        final = "DEPLOY" if (diff is not None and diff > 0 and (r.get("lift") or 0) > 0) else "HOLD"
        rows[k] = {
            "real_lift": r.get("lift"),
            "real_lo": rl,
            "real_hi": rh,
            "null_lift": n.get("lift"),
            "null_lo": nl,
            "null_hi": nh,
            "diff_lo": diff,
            "n": r.get("n"),
            "levels": r.get("levels"),
            "final": final,
            "error": r.get("error") or n.get("error"),
        }
        print(
            f"{k:8s} real={r.get('lift')} CI[{rl},{rh}] null={n.get('lift')} "
            f"CI[{nl},{nh}] diff_lo={diff} -> {final}",
            flush=True,
        )
    uc = {
        "deploy": sum(1 for v in rows.values() if v["final"] == "DEPLOY"),
        "hold": sum(1 for v in rows.values() if v["final"] == "HOLD"),
        "deploy_uncalibrated": sum(
            1 for v in rows.values() if v["final"] == "DEPLOY" and (v["diff_lo"] or 0) <= 0
        ),
        "nulls_hold": sum(1 for v in rows.values() if (v["null_lo"] or 0.0) <= 0.0),
        "nulls_total": len(KEYS),
        "rule": "PIT response-block policy; DEPLOY iff real_lift>0 AND real_lo-null_hi>0",
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(uc, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in uc.items() if k != "rows"}, indent=1, default=str))
    return uc


if __name__ == "__main__":
    main()
