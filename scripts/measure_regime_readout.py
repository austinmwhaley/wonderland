#!/usr/bin/env python3
"""Regime readout — does the STATE carry the latent regime better than aggregates? (DEC-524)

Read-only, and the attribution half of the regime-extraction bundle. Aggregate summaries cannot
identify the regime (balanced accuracy 0.474 vs 0.333 chance on demo_v2). If the frozen state also
cannot, the encoder is not extracting it and the bundle is aimed correctly. If the state *can*, the
readout path is destroying information that the encoder already holds, and the fix is downstream.

Same probe, same folds, three representations:

    aggregates (generic_measure) | state (state_embeddings) | both

    python3 scripts/measure_regime_readout.py \
        --products /home/austin_whaley/wonderland_work/cfm_v2 \
        --demo /home/austin_whaley/wonderland_work/demo_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from looking_glass.cfm_config import CFMConfig  # noqa: E402
from looking_glass.layer_b import generic_measure  # noqa: E402


def _cv_balanced(X, y, seed: int, folds: int = 5) -> dict:
    """Cross-validated balanced accuracy (the metric that cannot be won by majority class)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
    )
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    bal = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy")
    acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    return {
        "balanced_accuracy_mean": round(float(bal.mean()), 4),
        "balanced_accuracy_sd": round(float(bal.std()), 4),
        "accuracy_mean": round(float(acc.mean()), 4),
        "folds": folds,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--as-of", default="2025-01-01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=str(ROOT / "docs/report_cards/experiments/regime_readout.json")
    )
    a = ap.parse_args(argv)

    import duckdb

    prod = Path(a.products)
    db = prod if prod.is_file() else prod / "cfm_products.duckdb"
    demo = Path(a.demo)
    truth = {
        c["customer"]: c
        for c in json.loads((demo / "reference_truth.json").read_text())["customers"]
    }

    con = duckdb.connect(str(db), read_only=True)
    try:
        keys = [
            r[0]
            for r in con.execute(
                "select customer_key from encoder_samples where split='B'"
            ).fetchall()
        ]
        emb = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
        served = [r[0] for r in con.execute("select distinct version from state_embeddings")]
    finally:
        con.close()

    keys = [k for k in keys if k in emb and k in truth]
    if len(keys) < 300:
        print(f"only {len(keys)} held-out states with truth — too few", file=sys.stderr)
        return 2
    state = np.vstack([emb[k] for k in keys])

    stream = demo / "path_b" / "path_b_customer_event_stream.parquet"
    cfg = CFMConfig()
    cfg.db = str(stream)
    cfg.as_of = a.as_of
    agg, _ = generic_measure(cfg, keys)
    agg = np.asarray(agg, dtype=np.float64)

    y = np.array([str(truth[k]["regime0"]) for k in keys])
    classes = sorted(set(y))
    chance = 1.0 / len(classes)

    blocks = {
        "aggregates": _cv_balanced(agg, y, a.seed),
        "state": _cv_balanced(state, y, a.seed),
        "both": _cv_balanced(np.hstack([agg, state]), y, a.seed),
    }
    body = {
        "_meta": {
            "producer": Path(__file__).name,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "read_only": True,
            "products": str(prod),
            "demo": str(demo),
            "served_versions": served,
            "n": int(len(keys)),
            "regime_classes": classes,
            "chance_balanced_accuracy": round(chance, 4),
            "as_of": a.as_of,
        },
        "readout": blocks,
        "interpretation": {
            "state_beats_aggregates": bool(
                blocks["state"]["balanced_accuracy_mean"]
                > blocks["aggregates"]["balanced_accuracy_mean"]
            ),
            "state_to_chance_ratio": round(
                blocks["state"]["balanced_accuracy_mean"] / max(chance, 1e-9), 3
            ),
            "aggregate_to_chance_ratio": round(
                blocks["aggregates"]["balanced_accuracy_mean"] / max(chance, 1e-9), 3
            ),
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n")

    print(f"\nregime readout -> {out}")
    print(f"  n={len(keys)}  classes={classes}  chance={chance:.3f}")
    for k, v in blocks.items():
        print(
            f"  {k:12s} balanced acc {v['balanced_accuracy_mean']:.3f}"
            f" ±{v['balanced_accuracy_sd']:.3f}  (raw {v['accuracy_mean']:.3f})"
        )
    i = body["interpretation"]
    print(
        f"  state/aggregate over chance: {i['state_to_chance_ratio']:.2f}x vs "
        f"{i['aggregate_to_chance_ratio']:.2f}x  -> "
        f"{'STATE carries regime' if i['state_beats_aggregates'] else 'STATE does not beat aggregates'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
