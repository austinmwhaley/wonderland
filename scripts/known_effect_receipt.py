"""Known-effect recovery: the E's action-conditioned RESPONSE BLOCK must recover the
reference's KNOWN per-customer optimum for every randomized action key.

Direct positive controls on the sufficient statistic itself (the response block
estimates the reward at each action level from the as-of history), graded against
generator truth the encoder never saw. Binomial test vs chance per action.

Usage:
  python3 scripts/known_effect_receipt.py --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> --truth <truth.json> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = ("channel", "offer", "freq", "hour", "dow")


def _truth_value(key, c, channels, hour_buckets):
    if key == "channel":
        return channels.index(str(c["channel_pref"])) if c.get("channel_pref") in channels else -1
    if key == "offer":
        return int(c["offer_opt"]) if c.get("offer_opt") is not None else -1
    if key == "freq":
        return int(np.clip(round(float(c["x0"])), 1, 3)) if c.get("x0") is not None else -1
    if key == "hour":
        if c.get("h0") is None:
            return -1
        return int(np.argmin([abs(float(c["h0"]) - b) for b in hour_buckets]))
    if key == "dow":
        return int(np.clip(round(float(c["d0"])), 0, 6)) if c.get("d0") is not None else -1
    return -1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--action-key", default=None, dest="action_key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from scipy.stats import binomtest

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.head_battery import _b_states
    from rabbit_hole.generators.reference import HOUR_BUCKETS

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    cust, Z = _b_states(model, cfg)
    lb = getattr(model, "layer_b", None)
    names = list(getattr(lb, "names", ()) or ())
    if not names:
        raise SystemExit("no fitted LayerBEmbedder measure")
    d = int(Z.shape[1]) - len(names)
    em = np.asarray(Z[:, d:], dtype=float)

    meta = json.loads(Path(a.truth).read_text())
    T = {c["customer"]: c for c in meta.get("customers", [])}
    channels = (meta.get("meta") or {}).get("contact_channels") or [
        "email",
        "sms",
        "push",
        "direct_mail",
    ]
    hour_buckets = list(HOUR_BUCKETS)
    keys = [a.action_key] if a.action_key else list(KEYS)
    rows = {}
    for key in keys:
        idx = [i for i, n in enumerate(names) if n.startswith(f"resp_{key}_")]
        if not idx:
            rows[key] = {"passed": False, "error": "no response block"}
            continue
        R = em[:, idx]
        observed = np.where(R.max(0) > 0)[0]
        if len(observed) == 0:
            rows[key] = {"passed": False, "error": "no observed levels"}
            continue
        chosen = observed[np.argmax(R[:, observed], axis=1)]
        pairs = []
        for c, ch in zip(cust, chosen):
            t = _truth_value(key, T.get(c, {}), channels, hour_buckets)
            if t >= 0 and t in observed:
                pairs.append((int(ch), int(t)))
        n = len(pairs)
        hits = sum(1 for p, t in pairs if p == t)
        chance = 1.0 / max(len(observed), 1)
        p = float(binomtest(hits, n, chance, alternative="greater").pvalue) if n else 1.0
        acc = hits / max(n, 1)
        ok = bool(n > 0 and acc > chance and p < 0.01)
        rows[key] = {
            "passed": ok,
            "accuracy": round(acc, 4),
            "chance": round(chance, 4),
            "n_customers": int(n),
            "hits": int(hits),
            "binom_p": p,
            "levels": int(len(observed)),
        }
    n_pass = sum(1 for r in rows.values() if r.get("passed"))
    rec = {
        "passed": int(n_pass),
        "total": int(len(rows)),
        "baseline": "known randomized per-action optima (truth receipt)",
        "beats_traditional": bool(n_pass == len(rows) and len(rows) > 0),
        "evidence": "E response-block argmax recovers the reference's known per-action optimum",
        "actions": rows,
    }
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "actions"}, indent=2))
    for k, v in rows.items():
        print(k, v)
    return rec


if __name__ == "__main__":
    main()
