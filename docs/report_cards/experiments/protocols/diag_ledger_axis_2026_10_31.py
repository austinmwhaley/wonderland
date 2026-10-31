"""DEC-493 diagnostic: the S5 ledger_reconstruction failure is a target-transform artifact.

Cheap pre-protocol diagnostic (DEC-462/464 rule), NOT a protocol and NOT a gate change.
The frozen S5 check fits a linear head on the served donor (`state_embeddings` = proj(h))
to four exact past totals, with `log1p` applied to the targets, and compares against an
event-type bag + recency. spend and margin lose under log1p. This measures the same
representation/probe under BOTH target scalings:

  raw    the algebra SPINE mandates for extensive mass (additive, linear in raw units)
  log1p  the frozen check's transform

If raw extensive mass is linearly recoverable and beats the count bag with a paired CI
excluding zero, the S5 failure names the evaluation transform, not the encoder. Changing
the check is an operator ruling (stop-and-ask); this receipt only measures.

Same population/split/head as the frozen judge. Read-only, CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from measure_battery_law import _load_all, _paired_ci, _ridge_oos  # noqa: E402
from measure_views_law import _load_views  # noqa: E402

PROD = Path("/home/austin_whaley/wonderland_work/cfm_m2_clean")
DEMO = Path("/home/austin_whaley/wonderland_work/demo")

reg, truth, emb, raw_state, stream, bkeys, akeys, as_of = _load_all(
    str(PROD / "cfm_products.duckdb"), DEMO
)
views, _stamp = _load_views(PROD / "cfm_products.duckdb")
FULL_ORDER = ("shape", "learned_state", "forecast", "uncertainty", "heads")
if all(views.get(v) for v in FULL_ORDER):
    shared = set.intersection(*(set(views[v]) for v in FULL_ORDER))
    views["full"] = {k: np.concatenate([views[v][k] for v in FULL_ORDER]) for k in shared}

con = duckdb.connect()
con.register("s", stream)
led = con.execute(
    """
    select customer_key,
           coalesce(sum(try_cast(json_extract_string(event_attributes,'$.price') as double)
                        * try_cast(json_extract_string(event_attributes,'$.qty') as double)), 0.0)::double as spend,
           coalesce(sum(try_cast(json_extract_string(event_attributes,'$.margin') as double)), 0.0)::double as margin,
           sum(case when event_type='purchase' then 1 else 0 end)::bigint as n_purch,
           sum(case when event_type='return' then 1 else 0 end)::bigint as n_ret,
           count(*)::bigint as n_events,
           date_diff('day', max(cast(event_ts as date)), cast(? as date))::bigint as days_since_last
    from s group by 1
    """,
    [as_of.astype(str)],
).pl()
con.close()

# The check's own frequency comparator: log1p counts of every event type (minus the
# target-defining type) + recency. Encoded once over all customers.
bag = (
    stream.group_by(["customer_key", "event_type"])
    .len()
    .pivot(values="len", index="customer_key", on="event_type", aggregate_function="sum")
    .fill_null(0)
    .sort("customer_key")
)
bag_keys = bag["customer_key"].to_numpy()
bag_types = [c for c in bag.columns if c != "customer_key"]
bag_mat = bag.select(bag_types).to_numpy().astype(np.float64)
fmap = {r["customer_key"]: r for r in led.to_dicts()}
rec = np.array([float((fmap.get(k) or {}).get("days_since_last") or 0.0) for k in bag_keys])
bag_mat = np.hstack([np.log1p(bag_mat), rec[:, None]])


def _y(keys, col, log):
    v = np.array([float((fmap.get(k) or {}).get(col) or 0.0) for k in keys])
    return np.log1p(np.maximum(v, 0.0)) if log else v


def _mat(source, keys, exclude=None):
    if source == "naive":
        cols = [i for i, t in enumerate(bag_types) if t != exclude]
        base = bag_mat[:, cols + [len(bag_types)]]
        return base[np.searchsorted(bag_keys, np.asarray(keys))]
    if source == "raw_state":
        return np.vstack([raw_state[k] for k in keys])
    if source == "state_embeddings":
        return np.vstack([emb[k] for k in keys])
    return np.vstack([views[source][k] for k in keys])


REPS = ["state_embeddings", "raw_state", "shape", "learned_state", "full", "measure"]
TRANSFORMS = (("raw", False), ("log1p", True))
out: dict = {}
for col, excl in (
    ("spend", None),
    ("margin", None),
    ("n_purch", "purchase"),
    ("n_ret", "return"),
):
    out[col] = {}
    for tname, log in TRANSFORMS:
        ya, yb = _y(akeys, col, log), _y(bkeys, col, log)
        pb, pn = {}, {}
        for src in ["naive", *REPS]:
            xa, xb = _mat(src, akeys, excl), _mat(src, bkeys, excl)
            p, r2 = _ridge_oos(xa, ya, xb, yb)
            if p is not None:
                pb[src], pn[src] = p, r2
        naive_p = pb.get("naive")
        row = {}
        for src in REPS:
            if src not in pb:
                continue
            d, lo, hi = _paired_ci((yb - naive_p) ** 2 - (yb - pb[src]) ** 2)
            row[src] = {
                "r2": round(float(pn[src]), 4),
                "r2_naive": round(float(pn["naive"]), 4),
                "delta_vs_naive": round(float(pn[src] - pn["naive"]), 4),
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
            }
        out[col][tname] = {"n_fit_A": len(akeys), "n_score_B": len(bkeys), "arms": row}
        print(f"\n== {col} / {tname} (naive R2 {pn['naive']:.4f}) ==")
        for src, v in row.items():
            star = " CI-EXCLUDES-0" if v["ci_lo"] > 0 else ""
            print(
                f"  {src:18s} R2={v['r2']:.4f} delta={v['delta_vs_naive']:+.4f} "
                f"CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]{star}"
            )

receipt = Path(__file__).with_suffix(".json")
receipt.write_text(
    json.dumps(
        {
            "diagnostic": "S5 ledger_reconstruction: extensive-mass target transform (diagnostic only)",
            "artifact": "0.11.0r103115-b20d256e3lr3e-04o1e051d9d",
            "as_of": str(as_of),
            "splits": {"n_fit_A": len(akeys), "n_score_B": len(bkeys)},
            "transforms": [t for t, _ in TRANSFORMS],
            "results": out,
        },
        indent=2,
    )
    + "\n"
)
print(f"\n-> {receipt}")
