"""Known-effect recovery: the E's action-conditioned RESPONSE BLOCK must recover the
reference's KNOWN per-customer channel optimum (`channel_pref`) above chance.

This is a direct positive control on the sufficient statistic itself (the response
block estimates the reward at each action level from the as-of history), graded
against generator truth that the encoder never saw. Binomial test vs chance.
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
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--action-key", default="channel", dest="action_key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from scipy.stats import binomtest

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.head_battery import _b_states

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
    idx = [i for i, n in enumerate(names) if n.startswith(f"resp_{a.action_key}_")]
    if not idx:
        raise SystemExit(f"no response block for {a.action_key!r}")
    R = np.asarray(Z[:, d:][:, idx], dtype=float)
    observed = np.where(R.max(0) > 0)[0]
    if len(observed) == 0:
        chosen = np.zeros(len(cust), dtype=int)
    else:
        chosen = observed[np.argmax(R[:, observed], axis=1)]

    meta = json.loads(Path(a.truth).read_text())
    T = {c["customer"]: c for c in meta.get("customers", [])}
    channels = (meta.get("meta") or {}).get("contact_channels") or [
        "email",
        "sms",
        "push",
        "direct_mail",
    ]
    cmap = {str(ch): i for i, ch in enumerate(channels)}
    truth = {
        c: cmap.get(str(v.get("channel_pref")), -1)
        for c, v in T.items()
        if v.get("channel_pref") is not None
    }
    pairs = [(int(ch), truth[c]) for c, ch in zip(cust, chosen) if c in truth]
    pairs = [(p, t) for p, t in pairs if 0 <= t < len(channels)]
    n = len(pairs)
    hits = sum(1 for p, t in pairs if p == t)
    chance = 1.0 / max(len(observed), 1)
    acc = hits / max(n, 1)
    p = float(binomtest(hits, n, chance, alternative="greater").pvalue) if n else 1.0
    passed = bool(n > 0 and acc > chance and p < 0.01)
    rec = {
        "passed": int(passed),
        "total": 1,
        "baseline": f"known randomized effect ({a.action_key}_pref)",
        "beats_traditional": passed,
        "evidence": (
            f"E response-block argmax recovers the reference's known {a.action_key} optimum"
        ),
        "action_key": a.action_key,
        "accuracy": round(acc, 4),
        "chance": round(chance, 4),
        "n_customers": int(n),
        "hits": int(hits),
        "binom_p": p,
        "levels": int(len(observed)),
    }
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    main()
