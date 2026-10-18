"""Known-effect recovery: the E-policy must recover the reference's KNOWN per-customer
channel optimum (`channel_pref`) above chance (binomial test). This is a direct
positive control, independent of the (stub) white_queen bridge."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import polars as pl
    from scipy.stats import binomtest
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.head_battery import _b_states
    from looking_glass.use_case_gate import _period_rows

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    cust, Z = _b_states(model, cfg)
    j = _period_rows(cfg, cust, "channel")
    ci = {c: i for i, c in enumerate(cust)}
    rows = j.filter(pl.col("customer_key").is_in(list(ci)))
    cu = rows["customer_key"].to_numpy()
    A = rows["action"].to_numpy().astype(int)
    y = (rows["y"].to_numpy().astype(float) > 0).astype(float)
    Q = int(A.max()) + 1
    N = len(A)
    Zs = StandardScaler().fit_transform(np.stack([Z[ci[c]] for c in cu]))

    # cross-half V(E,a) -> argmax policy (never scores a row with a model that saw it)
    half = np.array([zlib.crc32(str(c).encode()) % 2 for c in cu])
    cand = np.empty(N, dtype=int)
    logged = np.unique(A)
    for h in (0, 1):
        tr = np.where(half != h)[0]
        va = np.where(half == h)[0]
        m = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=50).fit(
            np.column_stack([Zs[tr], A[tr].astype(float)]), y[tr]
        )
        Vh = np.column_stack(
            [m.predict(np.column_stack([Zs[va], np.full(len(va), float(k))])) for k in range(Q)]
        )
        cand[va] = logged[np.argmax(Vh[:, logged], axis=1)]

    # per-customer mode of the chosen action
    by_cust: dict = {}
    for c, act in zip(cu, cand):
        by_cust.setdefault(c, []).append(int(act))
    chosen = {c: int(np.bincount(v).argmax()) for c, v in by_cust.items()}
    T = {c["customer"]: c for c in json.loads(Path(a.truth).read_text())["customers"]}
    ch = ("email", "sms", "push", "direct_mail")
    truth = {c: ch.index(v["channel_pref"]) for c, v in T.items() if "channel_pref" in v}
    common = [c for c in chosen if c in truth]
    hits = sum(1 for c in common if chosen[c] == truth[c])
    n = len(common)
    chance = 1.0 / Q
    acc = hits / max(n, 1)
    p = float(binomtest(hits, n, chance, alternative="greater").pvalue) if n else 1.0
    passed = bool(n > 0 and acc > chance and p < 0.01)
    rec = {
        "passed": int(passed),
        "total": 1,
        "baseline": "known randomized effect (channel_pref)",
        "beats_traditional": passed,
        "evidence": "E-policy argmax recovers the reference's known channel optimum",
        "accuracy": round(acc, 4),
        "chance": round(chance, 4),
        "n_customers": int(n),
        "binom_p": p,
    }
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    main()
