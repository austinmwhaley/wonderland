"""E-vs-R-vs-E+R ablation under the PRODUCTION bake-off head (ROADMAP P1-4).

The sufficiency battery proves donor-vs-raw with Spearman-of-Ridge linear
probes on `anchor_embeddings`; production ships the bake-off winner on
`donor_embeddings` judged by held-out AUC. This command measures the claim the
gate actually uses, on the table the gate actually reads:

  * fit the full family roster (logistic/mlp/hgb) on E, on R, and on [E|R]
    with the SAME split/seed (identical held-out rows),
  * every difference is reported with a paired customer-cluster bootstrap SE,
  * receipt -> plugins/artifacts/ablation_<tag>_<as_of>.json.

R (raw) is the battery-convention RFM vector built by
`looking_glass.sufficiency_battery.rfm_vector` — one definition of "raw" for
both measurements.

  python3 -m plugins.ablation supervised_purchase_propensity_30d --as-of 2025-12-01
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .base import CFM_PRODUCTS, OUT, STREAM_DB, Dataset, load_dataset
from .head_template import HeadTemplate
from .ladder_sample_a import _paired_auc_se
from .targets import REGISTRY, Target

WORK = Path(__file__).resolve().parents[1]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):  # NULL values (e.g. unknown-basket orders)
        return 0.0


def _epoch(s):
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except Exception:
        return float("nan")


def _customer_index_ts(stream):
    """customer_key -> (epochs, event_types, values), battery-compatible parse."""
    sc = {c: stream[c].to_list() for c in stream.columns}
    n, i = stream.height, 0
    sby = {}
    while i < n:
        k = sc["customer_key"][i]
        j = i
        while j < n and sc["customer_key"][j] == k:
            j += 1
        sby[k] = (
            np.array([_epoch(x) for x in sc["event_ts"][i:j]], dtype=np.float64),
            [str(x) for x in sc["event_type"][i:j]],
            np.array([_num(x) for x in sc["value"][i:j]], dtype=np.float64),
        )
        i = j
    return sby


def _order_index(orders):
    oby: dict = {}
    for r in orders.iter_rows(named=True):
        oby.setdefault(r["customer_id"], []).append((float(r["t"]), float(r["gm"])))
    return oby


def raw_matrix(ds: Dataset, stream, orders) -> np.ndarray:
    """Raw feature rows ALIGNED to ds (same customers/anchors, same order).

    Alignment is the contract: any anchor the raw index cannot serve fails
    safe (SystemExit) instead of silently dropping rows — an ablation on a
    different row set would compare nothing.
    """
    from looking_glass.sufficiency_battery import rfm_vector

    sby = _customer_index_ts(stream)
    oby = _order_index(orders)
    etypes = sorted(set(map(str, stream["event_type"].unique().to_list())))
    rows, missing = [], []
    for k, a in zip(ds.keys, ds.anchor_epoch):
        key = str(k)
        a = float(a)
        s = sby.get(key)
        if s is None:
            missing.append(key)
            rows.append(None)
            continue
        ts, ets, vals = s
        if int((ts <= a).sum()) < 3:
            missing.append(f"{key}@{a}")
            rows.append(None)
            continue
        rows.append(rfm_vector(ts, ets, vals, a, oby.get(key, []), etypes))
    if any(r is None for r in rows):
        raise SystemExit(
            f"E-vs-R alignment failed: {len(missing)}/{len(rows)} anchors lack a raw "
            f"vector (first: {missing[:3]}) — same rows are required for a paired "
            "comparison"
        )
    return np.asarray(rows, dtype=np.float64)


def _fit_view(target: Target, ds: Dataset, seed: int) -> tuple[dict, list[dict]]:
    tpl = HeadTemplate(target)
    heads, _fitted = tpl.fit(ds, seed=seed)
    winner = next(h for h in heads if h["name"] == tpl._winner)
    return winner, heads


def _view_summary(heads: list[dict]) -> dict:
    return {
        h["name"]: {
            "auc": h["metrics"]["auc"],
            "lift": h["metrics"]["top_decile_lift"],
            "brier": h["metrics"]["brier"],
            "ece": h["metrics"].get("ece"),
        }
        for h in heads
    }


def run(
    target: Target,
    as_of: str,
    seed: int = 0,
    cfm_products=None,
    stream_db=None,
    out_dir=None,
) -> dict:
    from looking_glass.layer_b_proof import _load

    t0 = time.perf_counter()
    products = cfm_products or CFM_PRODUCTS
    stream_path = stream_db or STREAM_DB
    out = Path(out_dir) if out_dir else OUT

    ds = load_dataset(
        target.window_days,
        target=target,
        as_of=as_of,
        cfm_products=products,
        stream_db=stream_path,
    )
    if ds.meta.get("encoder_version") is None:
        raise ValueError("products table holds multiple encoder versions; rebuild products")

    stream, _anch, orders, _data_end, _tag = _load(str(stream_path), str(products))
    R = raw_matrix(ds, stream, orders)
    E = np.asarray(ds.X, dtype=np.float64)

    views = {
        "donor": E,
        "raw": R,
        "both": np.hstack([E, R]),
    }
    winners, summaries = {}, {}
    for name, X in views.items():
        ds_v = dataclasses.replace(ds, X=X.astype(np.float32))
        w, heads = _fit_view(target, ds_v, seed)
        winners[name] = w
        summaries[name] = _view_summary(heads)
        print(
            f"  [{name:>5s}] winner={w['name']} auc={w['metrics']['auc']:.4f} "
            f"lift={w['metrics']['top_decile_lift']:.3f} "
            f"brier={w['metrics']['brier']:.4f}",
            flush=True,
        )

    def _diff(a: str, b: str) -> dict:
        wa, wb = winners[a], winners[b]
        se = _paired_auc_se(
            {"keys": wa["keys"], "y": wa["y_te"], "pred": wa["pred"]},
            {"keys": wb["keys"], "y": wb["y_te"], "pred": wb["pred"]},
        )
        d = float(wa["metrics"]["auc"] - wb["metrics"]["auc"])
        return {
            "delta_auc": d,
            "se_paired": se,
            "beyond_noise": bool(se is not None and d > 2.0 * se),
        }

    d_donor_raw = _diff("donor", "raw")
    d_both_donor = _diff("both", "donor")
    d_raw_donor = _diff("raw", "donor")

    receipt = {
        "as_of": as_of,
        "target": target.name,
        "tag": target.tag,
        "seed": seed,
        "encoder_version": ds.meta.get("encoder_version"),
        "feature_table": ds.meta.get("feature_table"),
        "n_rows": int(len(ds.y)),
        "n_customers": int(len(set(map(str, ds.keys.tolist())))),
        "base_rate": float(np.mean(ds.y)),
        "views": summaries,
        "winner": {k: v["name"] for k, v in winners.items()},
        "donor_vs_raw": d_donor_raw,
        "both_vs_donor": d_both_donor,
        "raw_vs_donor": d_raw_donor,
        "verdicts": {
            # the production claim: donor beats raw BEYOND paired noise
            "donor_beats_raw": bool(d_donor_raw["beyond_noise"]),
            # does raw add anything the donor lacks?
            "raw_adds_beyond_donor": bool(d_both_donor["beyond_noise"]),
            # is raw competitive (within noise of donor)?
            "raw_competitive": bool(
                d_raw_donor["delta_auc"] > -2.0 * (d_raw_donor["se_paired"] or 0.0)
            ),
        },
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    stamp = as_of.replace("-", "")
    path = out / f"ablation_{target.tag}_{stamp}.json"
    path.write_text(json.dumps(receipt, indent=1, default=float))

    print("\n== E-vs-R ABLATION (production head, paired noise) ==")
    print(
        f"  donor - raw : {d_donor_raw['delta_auc']:+.4f} "
        f"(2*SE {2 * (d_donor_raw['se_paired'] or 0.0):.4f}) "
        f"-> donor beyond noise: {receipt['verdicts']['donor_beats_raw']}"
    )
    print(
        f"  both  - donor: {d_both_donor['delta_auc']:+.4f} "
        f"(2*SE {2 * (d_both_donor['se_paired'] or 0.0):.4f}) "
        f"-> raw adds: {receipt['verdicts']['raw_adds_beyond_donor']}"
    )
    print(
        f"  raw   - donor: {d_raw_donor['delta_auc']:+.4f} "
        f"-> raw competitive: {receipt['verdicts']['raw_competitive']}"
    )
    print(f"receipt -> {path}")
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Donor-vs-raw ablation under the production head")
    ap.add_argument("target", choices=sorted(REGISTRY))
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--products", default=None)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    run(
        REGISTRY[a.target],
        as_of=a.as_of,
        seed=a.seed,
        cfm_products=a.products,
        stream_db=a.stream,
        out_dir=a.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
