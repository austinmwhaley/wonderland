"""B6.2 donor battery on the cfm_v2k products (DEC-585).

Targets: y_value_90 / y_orders_90 (R2) + y_churn_90 (AUC from the ridge score).
Split: the judge's A/B populations via the battery's _load_all (same seed the build used).
Arms: raw = customer_features.parquet (40 -> 38 via table_features), fair =
path_a_wide_table.parquet (50 -> 38; on cfm_v2k the two 38-column matrices are
value-IDENTICAL — recorded, not assumed), E = stored state_embeddings (read, not
re-encoded; the compact served object includes the 16-dim level block and the 4-dim
response block), table_E = the ENSEMBLE mean of the table ridge and the E ridge (v2;
DEC-594), with the v1 hstack(fair, E) arm recorded alongside. random/destroyed E =
controls_matched, broken on BOTH fit and score sides. ONE encoding over A + B per table
arm; head = the shipping standardized ridge, fit on A, scored on B.

BAR (card, per non-degenerate target): E >= raw and E >= fair (point);
CI (table_E - fair) lower bound > 0 (paired, statistic recomputed in each replicate,
N_BOOT=2000); random < E and destroyed < E (point). Degenerate targets are excluded
and reported.

CPU producer (no forward): re-stamps docs/report_cards/experiments/encoder_donor.json;
the legacy numbers are quarantined in DEC-585.
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

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_donor.json"

TARGETS = ("y_value_90", "y_orders_90", "y_churn_90")


def _metric(y, p, metric: str) -> float:
    if metric == "r2":
        ss = float(np.sum((y - y.mean()) ** 2))
        return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _score(Xa, ya, Xb, yb, metric: str):
    m = _ridge_head()
    m.fit(Xa, ya)
    return m, m.predict(Xb)


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k",
    )
    ap.add_argument(
        "--demo",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2",
    )
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

    labels = pl.read_parquet(demo / "path_a" / "customer_labels.parquet")
    lab = {r["customer_key"]: r for r in labels.iter_rows(named=True)}
    y_all = {c: np.asarray([lab[k][c] for k in all_keys], dtype=float) for c in TARGETS}
    ya_all = {c: y_all[c][:n_a] for c in TARGETS}
    yb_all = {c: y_all[c][n_a:] for c in TARGETS}

    # one encoding over A + B per table arm (splitting the encoding is the trap)
    raw_df = pl.read_parquet(demo / "path_a" / "customer_features.parquet")
    wide_df = pl.read_parquet(demo / "path_a" / "path_a_wide_table.parquet")
    Xr_all, _lr, _sr = table_features(raw_df, all_keys)
    Xf_all, _lf, _sf = table_features(wide_df, all_keys)
    raw_equals_fair = bool(np.array_equal(Xr_all, Xf_all))
    Xr_a, Xr_b = Xr_all[:n_a], Xr_all[n_a:]
    Xf_a, Xf_b = Xf_all[:n_a], Xf_all[n_a:]

    Ea = np.vstack([emb[k] for k in akeys])
    Eb = np.vstack([emb[k] for k in bkeys])
    rng = np.random.default_rng(SEED)
    rand_b, dest_b = controls_matched(rng, Ea, Eb)
    rand_a, dest_a = controls_matched(rng, Ea, Ea)

    targets: dict[str, dict] = {}
    for c in TARGETS:
        metric = "auc" if c == "y_churn_90" else "r2"
        ya, yb = ya_all[c], yb_all[c]
        deg_reason = None
        if metric == "r2" and (float(np.std(ya)) < 1e-12 or float(np.std(yb)) < 1e-12):
            deg_reason = "target has no variance on A or B"
        elif metric == "auc" and (len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2):
            deg_reason = "target has <2 classes on A or B"
        if deg_reason:
            targets[c] = {"metric": metric, "degenerate": True, "reason": deg_reason}
            continue
        arms = {
            "E": (Ea, Eb),
            "raw": (Xr_a, Xr_b),
            "fair": (Xf_a, Xf_b),
            "hstack": (np.hstack([Xf_a, Ea]), np.hstack([Xf_b, Eb])),
            "random": (rand_a, rand_b),
            "destroyed": (dest_a, dest_b),
        }
        scores, preds_b = {}, {}
        for name, (Xa, Xb) in arms.items():
            _m, p = _score(Xa, ya, Xb, yb, metric)
            scores[name] = _metric(yb, p, metric)
            preds_b[name] = p
        # v2 (DEC-594): the table+E arm is the ENSEMBLE mean of the two fitted heads.
        # The v1 hstack arm is recorded too; neither is tuned on B.
        preds_b["ensemble"] = 0.5 * (preds_b["fair"] + preds_b["E"])
        scores["ensemble"] = _metric(yb, preds_b["ensemble"], metric)

        def _ci(name):
            P_te, P_f, y2 = preds_b[name], preds_b["fair"], yb

            def stat_fn(idx):
                return _metric(y2[idx], P_te[idx], metric), _metric(y2[idx], P_f[idx], metric)

            return _paired_stat_ci(stat_fn, len(bkeys))

        lo, hi = _ci("ensemble")
        h_lo, h_hi = _ci("hstack")
        targets[c] = {
            "metric": metric,
            "degenerate": False,
            "E": scores["E"],
            "raw": scores["raw"],
            "fair": scores["fair"],
            "table": scores["fair"],
            "table_E": scores["ensemble"],
            "table_E_hstack": scores["hstack"],
            "table_E_ensemble": scores["ensemble"],
            "hstack_ci_lo": h_lo,
            "hstack_ci_hi": h_hi,
            "random": scores["random"],
            "destroyed": scores["destroyed"],
            "ci_lo": lo,
            "ci_hi": hi,
        }

    deg = [c for c in TARGETS if targets.get(c, {}).get("degenerate")]
    body = {
        "complete": True,
        "targets": targets,
        "n_A": int(n_a),
        "n_B": int(n_b),
        "raw_equals_fair": raw_equals_fair,
        "combined_protocol": (
            "v2 (DEC-594): table_E = mean(table ridge, E ridge) — a fixed ensemble of the "
            "two fitted heads, no weights fit on B; the v1 hstack(fair,E) arm is recorded "
            "alongside as table_E_hstack"
        ),
        "note": (
            "raw and fair 38-column matrices "
            f"{'are value-identical on this stream (recorded, not assumed)' if raw_equals_fair else 'DIFFER on this stream'}; "
            "controls broken on fit+score sides (controls_matched, SEED); head = the shipping "
            "standardized ridge, one encoding over A+B, fit on A scored on B; E = stored "
            "state_embeddings (read, not re-encoded; the compact served object includes the "
            "level block and the response block)"
        ),
        "excluded": deg,
        "products": products_identity(a.products),
        "device": "cpu",
        "tag": tag,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(json.dumps({k: v for k, v in body.items() if k != "note"}, indent=1, default=str))
    print("WROTE", a.out)
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
