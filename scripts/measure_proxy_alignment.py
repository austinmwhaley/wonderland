"""Proxy-alignment diagnostic for the supervised leg (DEC-462 discipline, CPU-only).

Protocols E and F both trained objectives whose own targets fit, yet the judged linear probe
did not move. Before another protocol, measure WHAT THE SERVED STATE CARRIES about the trait's
generating structure:

  * the frequency response curve — per-customer conversion rate at each randomized `freq`
    arm level (built by the SAME `_effect_targets` machinery the `effect` objective uses,
    so the curve is the objective's own target, not a lookalike);
  * `x0` — the generator's optimal weekly frequency = the curve's peak.

Three readings:
  1. ridge(E -> x0)            — the judge's number (reference: 0.7909 on protocol D);
  2. ridge(E -> curve level)   — can the served state read the response surface at all?
  3. ridge(curve -> x0)        — is the curve SUFFICIENT for the trait (the ceiling of any
     state that carries the curve perfectly)?

If (3) is high and (2) is high, a protocol that makes the curve linearly readable must work;
if (2) is low, the trunk does not carry the response surface and the lever is capacity/memory,
not another head. This gates nothing; it names the next lever (the loop's Diagnosable rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_judge_law import _arms, _ridge_oos  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/report_cards/experiments/proxy_alignment.json"),
    )
    a = ap.parse_args()
    demo, prod = Path(a.demo), Path(a.products)

    import polars as pl

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_data import _effect_targets, read_stream_window

    arms = _arms(str(prod / "cfm_products.duckdb"), demo)
    akeys, bkeys = list(arms["akeys"]), list(arms["bkeys"])
    brain = arms["columns"]["brain"]
    truth_by = arms["truth"]

    # the canonical resolved stream (event_attributes flattened to columns) — the same
    # frame build_sequences/`_effect_targets` see during training, not the raw parquet.
    manifest = json.loads((demo / "manifest.json").read_text())
    cfg = CFMConfig(
        db=str(demo / "path_b" / "path_b_customer_event_stream.parquet"),
        as_of=str(manifest["as_of"]),
        effect_actions=("freq",),
        sample_customers=2500,  # match the trained population (A ∪ B = 2500)
    )
    cfg.primary_numeric = "value"
    df = read_stream_window(cfg, None, None)
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    stream = df

    cfg = CFMConfig(effect_actions=("freq",))
    df = stream if isinstance(stream, pl.DataFrame) else pl.from_pandas(stream)
    per, levels = _effect_targets(df, cfg)
    Q = int(levels[0]) if levels else 0
    if per is None or Q < 2:
        print(json.dumps({"working": False, "note": "no randomized freq arm in the stream"}))
        return 2

    def curve(keys):
        rows, miss = [], 0
        for k in keys:
            p = per.get(k)
            if p is None:
                # no randomized freq periods: curve unknown -> zeros + a missing receipt
                rows.append(np.zeros(Q, dtype=np.float64))
                miss += 1
                continue
            act, rew = p
            r = np.zeros(Q)
            for q in range(Q):
                m = act[:, 0].astype(int) == q
                r[q] = rew[m].mean() if m.any() else 0.0
            rows.append(r)
        return np.vstack(rows), miss

    Ca, miss_a = curve(akeys)
    Cb, miss_b = curve(bkeys)
    EA = np.vstack([brain["A"][k] for k in akeys])
    EB = np.vstack([brain["B"][k] for k in bkeys])
    xa = np.array([float(truth_by[k]["x0"]) for k in akeys])
    xb = np.array([float(truth_by[k]["x0"]) for k in bkeys])

    def score(mat_a, mat_b):
        p, r2 = _ridge_oos(mat_a, xa, mat_b, xb)
        return {"x0_r2": round(float(r2), 4) if np.isfinite(r2) else None}

    rep = {
        "working": True,
        "arm": "brain (state_embeddings)",
        "n_fit": len(akeys),
        "n_score": len(bkeys),
        "freq_levels": Q,
        "curve_missing_A": miss_a,
        "curve_missing_B": miss_b,
        "x0_from_E": score(EA, EB),
        "x0_from_curve": score(Ca, Cb),
        "curve_from_E": {},
    }
    # per-level readability of the curve from E (each level its own regression)
    for q in range(Q):
        _p, r2 = _ridge_oos(EA, Ca[:, q], EB, Cb[:, q])
        rep["curve_from_E"][f"level_{q}"] = round(float(r2), 4) if np.isfinite(r2) else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"proxy_alignment working=True -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
