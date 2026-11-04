"""B6.3 transfer law: the donor adds signal on a family broader than the eight judge legs.

The family is fixed and published here, not selected after the score:

  persona/geometry (10): x0, h0, d0, price_sensitivity, return_proneness,
      bayes_fwd_value_90d, bayes_fwd_count_90d, churn_day, regime_switches,
      final_regime_days  (the generator's own latent/persona truth)
  30-day outcomes (5): y_orders_30, y_value_30, y_margin_30, y_next_days_30, y_churn_30
      (the judge legs use the 90-day outcomes, so these are held-out consumers)

For each target: the shipping standardized ridge is fit on split A and scored on split B;
E = the stored served embedding (read, not re-encoded), table = PATH A's wide table under
the same head. The paired CI of (E - table) is computed by resampling B customers. random
and destroyed E are the matched controls (fit and score sides broken).

BAR (DEC-596, frozen before the run): at least 2/3 of the non-degenerate targets have
E >= table, the family-mean (E - table) is > 0, and the family-mean E beats the random and
destroyed controls. The family and all per-target numbers are published; a degenerate
target is excluded and reported.

Writes docs/report_cards/experiments/encoder_transfer_law.json (stamped).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cfm_products import _db_path, resolve_served_tag  # noqa: E402
from measure_battery_law import SEED, _load_all, _ridge_head, controls_matched  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import _paired_stat_ci  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_transfer_law.json"
DEFAULT_PRODUCTS = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_c32rvd"
DEFAULT_DEMO = ROOT / "looking_glass/artifacts/demo_v2"

PERSONA = (
    "x0",
    "h0",
    "d0",
    "price_sensitivity",
    "return_proneness",
    "bayes_fwd_value_90d",
    "bayes_fwd_count_90d",
    "churn_day",
    "regime_switches",
    "final_regime_days",
)
LABELS_30 = ("y_orders_30", "y_value_30", "y_margin_30", "y_next_days_30", "y_churn_30")
MIN_WIN_SHARE = 2.0 / 3.0


def _metric(y, p, kind: str) -> float:
    if kind == "r2":
        ss = float(np.sum((y - y.mean()) ** 2))
        return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--demo", default=str(DEFAULT_DEMO))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from demo.auto_encoder import table_features

    t0 = time.perf_counter()
    demo = Path(a.demo)
    tag, _reg = resolve_served_tag(a.products)
    _r, _truth, emb, _raw, _stream, bkeys, akeys, _as_of = _load_all(
        str(_db_path(a.products)), demo
    )
    all_keys = list(akeys) + list(bkeys)
    n_a, n_b = len(akeys), len(bkeys)
    truth = {
        c["customer"]: c
        for c in json.loads((demo / "reference_truth.json").read_text())["customers"]
    }
    lab = {
        r["customer_key"]: r
        for r in pl.read_parquet(demo / "path_a" / "customer_labels.parquet").iter_rows(named=True)
    }
    targets: dict[str, tuple[np.ndarray, str, str]] = {}
    for f in PERSONA:
        targets[f] = (
            np.asarray([float(truth[k][f]) for k in all_keys], dtype=float),
            "r2",
            "persona/geometry truth",
        )
    for f in LABELS_30:
        targets[f] = (
            np.asarray([float(lab[k][f]) for k in all_keys], dtype=float),
            "auc" if "churn" in f else "r2",
            "30-day held-out outcome",
        )

    wide = pl.read_parquet(demo / "path_a" / "path_a_wide_table.parquet")
    Xf, _l, _s = table_features(wide, all_keys)
    Xa, Xb = Xf[:n_a], Xf[n_a:]
    E = np.vstack([emb[k] for k in all_keys])
    Ea, Eb = E[:n_a], E[n_a:]
    rng = np.random.default_rng(SEED)
    rand_b, dest_b = controls_matched(rng, Ea, Eb)
    rand_a, dest_a = controls_matched(rng, Ea, Ea)

    rows = []
    for name, (y, kind, family) in targets.items():
        ya, yb = y[:n_a], y[n_a:]
        if float(np.std(ya)) < 1e-12 or float(np.std(yb)) < 1e-12:
            rows.append(
                {
                    "target": name,
                    "family": family,
                    "metric": kind,
                    "degenerate": True,
                    "reason": "target has no variance on A or B",
                }
            )
            continue
        pT = _ridge_head().fit(Xa, ya).predict(Xb)
        pE = _ridge_head().fit(Ea, ya).predict(Eb)
        pR = _ridge_head().fit(rand_a, ya).predict(rand_b)
        pD = _ridge_head().fit(dest_a, ya).predict(dest_b)
        sT, sE = _metric(yb, pT, kind), _metric(yb, pE, kind)

        def stat_fn(idx, pE=pE, pT=pT):
            return _metric(yb[idx], pE[idx], kind), _metric(yb[idx], pT[idx], kind)

        lo, hi = _paired_stat_ci(stat_fn, len(bkeys))
        rows.append(
            {
                "target": name,
                "family": family,
                "metric": kind,
                "degenerate": False,
                "E": sE,
                "table": sT,
                "diff": sE - sT,
                "ci_lo": lo,
                "ci_hi": hi,
                "random": _metric(yb, pR, kind),
                "destroyed": _metric(yb, pD, kind),
            }
        )

    live = [r for r in rows if not r.get("degenerate")]
    if not live:
        raise SystemExit("every transfer target is degenerate")
    wins = sum(1 for r in live if r["E"] >= r["table"])
    mean_diff = float(np.mean([r["diff"] for r in live]))
    mean_e = float(np.mean([r["E"] for r in live]))
    mean_t = float(np.mean([r["table"] for r in live]))
    mean_rand = float(np.mean([r["random"] for r in live]))
    mean_dest = float(np.mean([r["destroyed"] for r in live]))
    passed = bool(
        wins / len(live) >= MIN_WIN_SHARE
        and mean_diff > 0
        and mean_e > mean_rand
        and mean_e > mean_dest
    )
    body = {
        "complete": True,
        "family": {
            "persona": list(PERSONA),
            "labels_30": list(LABELS_30),
            "n_targets": len(live),
            "n_degenerate": len(rows) - len(live),
            "why": (
                "persona/geometry truth from the generator plus 30-day outcomes; the judge "
                "legs score the 90-day outcomes, so this is a broader held-out consumer family"
            ),
        },
        "bar": (
            f"E >= table on >= {MIN_WIN_SHARE:.3f} of non-degenerate targets; family-mean "
            "difference > 0; family-mean E beats random and destroyed (DEC-596)"
        ),
        "wins": int(wins),
        "n_targets": len(live),
        "win_share": round(wins / len(live), 4),
        "mean_E": round(mean_e, 6),
        "mean_table": round(mean_t, 6),
        "mean_diff": round(mean_diff, 6),
        "mean_random": round(mean_rand, 6),
        "mean_destroyed": round(mean_dest, 6),
        "passed": passed,
        "targets": rows,
        "products": products_identity(a.products),
        "tag": tag,
        "n_A": int(n_a),
        "n_B": int(n_b),
        "device": "cpu",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(json.dumps({k: v for k, v in body.items() if k != "targets"}, indent=1, default=str))
    print("WROTE", a.out)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
