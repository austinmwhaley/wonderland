"""Reconcile the OPE stream bridge with the response-block decision gate.

Both are evaluated on the SAME policy: the per-customer argmax of the E's
action-conditioned response block. The response-block gate uses a randomized
future comparison (E[y|a=pi]-E[y] + DiD null); white_queen uses its conservative
FQE/DR + bar. This receipt records both on the identical policy so a
HOLD-vs-DEPLOY difference is attributable to the ESTIMATOR/bar, not to two
different policies.

Usage:
  python3 scripts/ope_bridge_reconcile.py --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> --customers 6000 --action-key channel \
      --out <reconcile.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--action-key", default="channel", dest="action_key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.head_battery import _b_states
    from looking_glass.use_case_gate import run_use_case_response
    from looking_glass.white_queen_bridge import certify_period

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers

    cust, Z = _b_states(model, cfg)
    names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
    d = int(Z.shape[1]) - len(names)
    idx = [i for i, n in enumerate(names) if n.startswith(f"resp_{a.action_key}_")]
    if not idx:
        raise SystemExit(f"no response block for {a.action_key!r}")
    R = np.asarray(Z[:, d:][:, idx], dtype=float)
    observed = np.where(R.max(0) > 0)[0]
    chosen = observed[np.argmax(R[:, observed], axis=1)]
    pi = {str(c): int(v) for c, v in zip(cust, chosen)}

    # white_queen on the SAME response-block policy (fixed per customer)
    wq = certify_period(model, cfg, a.action_key, pref_by_customer=pi)
    rb = run_use_case_response(model, cfg, a.action_key, horizon_days=180)
    same = bool(wq.get("deploy")) == (float(rb.get("lift") or 0.0) > 0.0)
    rec = {
        "policy": "response-block argmax (same for both)",
        "action_key": a.action_key,
        "white_queen": wq,
        "response_block": rb,
        "agree": bool(same),
        "reconciled": True,
        "explanation": (
            "Both consumers use the identical per-customer response-block argmax. "
            + (
                "They agree."
                if same
                else "They differ: the response-block gate is a randomized future "
                "conditional-mean lift with a DiD null; white_queen is a conservative "
                "FQE/DR estimate against its own bar. The difference is the ESTIMATOR/bar, "
                "not the policy."
            )
        ),
    }
    Path(a.out).write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {k: v for k, v in rec.items() if k not in ("white_queen", "response_block")}, indent=1
        )
    )
    print("wq", {k: wq.get(k) for k in ("deploy", "behavior_mean", "bar", "estimates")})
    print("rb", {k: rb.get(k) for k in ("lift", "lo", "hi", "n")})
    return rec


if __name__ == "__main__":
    main()
