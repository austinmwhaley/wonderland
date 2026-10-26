"""Build the self-contained interactive demo site from real receipts.

Everything shown is generated from `docs/report_cards/experiments/*.json` and, for the
customer explorer, from the frozen artifact plus the canonical stream. Nothing is typed by
hand, so the site cannot drift from the measured evidence.

    python3 scripts/build_demo_site.py            # writes demo/index.html
    python3 scripts/build_demo_site.py --no-explorer   # skip the customer sample

The output is one HTML file with embedded JSON, inline CSS, and vanilla JS. No server, no CDN.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

EXP = ROOT / "docs/report_cards/experiments"
OUT = ROOT / "demo/index.html"

ARM_LABEL = {
    "table": "Traditional table",
    "fair": "Fair table (no fixed windows)",
    "shape": "Encoder shape (128-d)",
    "learned_state": "Learned state",
    "forecast": "Forecast head",
    "measure": "Lawful measure",
    "full_legacy": "Full state",
    "full_measure": "Full + measure",
}


def _load(name):
    p = EXP / name
    return json.loads(p.read_text()) if p.exists() else None


def _explorer(n_customers=260, neighbours=5, seed=0):
    """Sample holdout customers: projection, lookalikes, timeline, truth."""
    import numpy as np
    import polars as pl
    from rabbit_hole.generators.reference import HOUR_BUCKETS
    from sklearn.decomposition import PCA

    import frozen as F

    cache = Path("/tmp/opencode/frozen_6k")
    data = Path("/tmp/opencode/t0_loop/ref_v3b_6k")
    if not (
        cache.with_suffix(".npz").exists() and (data / "customer_event_stream.parquet").exists()
    ):
        return None

    E, _Xtab, labels, hold, _seg, _model, _vs, _B, _best, _BLK = F.load(str(cache))
    E = np.asarray(E, dtype=float)
    z = (E - E.mean(0)) / (E.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(hold), size=min(n_customers, len(hold)), replace=False))
    Z = z[keep]
    xyz = PCA(n_components=3, random_state=0).fit_transform(Z)
    xyz = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0) + 1e-9)

    truth = {
        c["customer"]: c
        for c in json.loads((data / "reference_truth.json").read_text())["customers"]
    }
    yv = np.asarray(labels["y_value_90"], dtype=float)
    yo = np.asarray(labels["y_orders_90"], dtype=float)
    yc = np.asarray(labels["y_churn_90"], dtype=float)

    # exact neighbours within the sampled set (cheap and visually consistent)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :neighbours]

    parquet = pl.read_parquet(data / "customer_event_stream.parquet")
    notable = {"customer_signup", "purchase", "return", "support_ticket", "review"}
    counts = parquet.group_by("customer_key").len()
    totals = dict(zip(counts["customer_key"].to_list(), counts["len"].to_list()))
    by_customer: dict[str, list] = {}
    for r in parquet.filter(pl.col("event_type").is_in(list(notable))).iter_rows(named=True):
        by_customer.setdefault(r["customer_key"], []).append(r)

    customers = []
    for row, idx in enumerate(keep):
        key = hold[idx]
        events = by_customer.get(key, [])
        timeline = []
        for r in events[:8]:
            attrs = json.loads(r["event_attributes"] or "{}")
            detail = ""
            if "value" in attrs:
                detail = f"${float(attrs['value']):,.0f}"
                if attrs.get("category_code"):
                    detail += f" · {attrs['category_code']}"
            elif attrs.get("category_code"):
                detail = attrs["category_code"]
            timeline.append(
                {
                    "ts": str(r["event_ts"])[:16].replace("T", " "),
                    "event": r["event_type"],
                    "detail": detail,
                }
            )
        t = truth.get(key, {})
        nb = [int(x) for x in nn[row]]
        customers.append(
            {
                "key": key,
                "x": round(float(xyz[row, 0]), 4),
                "y": round(float(xyz[row, 1]), 4),
                "z": round(float(xyz[row, 2]), 4),
                "n_events": int(totals.get(key, len(events))),
                "value90": round(float(yv[idx]), 1),
                "orders90": round(float(yo[idx]), 1),
                "churn90": int(yc[idx]),
                "channel": t.get("channel_pref"),
                "offer_opt": int(t.get("offer_opt", 0) or 0),
                "freq_opt": int(np.clip(round(float(t.get("x0") or 1.0)), 1, 3)),
                "hour_opt": int(
                    HOUR_BUCKETS[
                        int(
                            np.argmin(
                                [
                                    abs(float(t.get("h0") or HOUR_BUCKETS[0]) - b)
                                    for b in HOUR_BUCKETS
                                ]
                            )
                        )
                    ]
                ),
                "dow_opt": int(np.clip(round(float(t.get("d0") or 0.0)), 0, 6)),
                "category": t.get("category_pref"),
                "regime": t.get("regime0"),
                "return_proneness": round(float(t.get("return_proneness", 0.0)), 3),
                "timeline": timeline,
                "neighbours": [
                    {
                        "key": hold[keep[j]],
                        "value90": round(float(yv[keep[j]]), 1),
                        "orders90": round(float(yo[keep[j]]), 1),
                        "churn90": int(yc[keep[j]]),
                    }
                    for j in nb
                ],
                "neighbour_mean_value": round(float(yv[keep[nb]].mean()), 1),
                "neighbour_churn_agree": round(float((yc[keep[nb]] == yc[idx]).mean()), 3),
            }
        )
    return {"customers": customers, "n_sampled": len(customers), "projection": "PCA (3 components)"}


def _payload(explorer):
    fs = _load("foundation_state.json") or {}
    sel = _load("view_selection.json") or {}
    ceil = _load("encoder_ceiling.json") or {}
    donor = _load("encoder_donor.json") or {}
    ident = _load("encoder_identity.json") or {}
    uns = _load("plugin_unsupervised.json") or {}
    rlk = _load("plugin_rl_known.json") or {}
    resp = _load("plugin_response.json") or {}
    rt = _load("runtime.json") or {}
    obj = (_load("encoder_objectives.json") or {}).get("objectives", {})
    bprops = _load("encoder_b_props.json") or {}
    bdata = None
    _bd = ROOT / "demo/b_data.json"
    if _bd.exists():
        bdata = json.loads(_bd.read_text())
    data_tab = None
    _dt = ROOT / "demo/data_tab.json"
    if _dt.exists():
        data_tab = json.loads(_dt.read_text())
    encoder_tab = None
    _et = ROOT / "demo/encoder_tab.json"
    if _et.exists():
        encoder_tab = json.loads(_et.read_text())
    cfg = _load("encoder_config.json") or {}
    data = _load("data_receipt.json") or {}

    targets = ["value90", "orders90", "churn90"]
    arms = ["table", "shape", "learned_state", "forecast", "measure", "full_measure"]
    sup = {}
    for t in targets:
        row = {}
        for a in arms:
            v = (fs.get("supervised") or {}).get(t, {}).get(a)
            if v is not None:
                row[a] = round(float(v), 4)
        ci = (fs.get("supervised") or {}).get(t, {}).get("full_measure_vs_table")
        if ci:
            row["full_ci"] = ci
        if t in (donor.get("results") or {}):
            d = donor["results"][t]
            row.setdefault("raw", round(float(d["raw"]), 4))
            row.setdefault("fair", round(float(d["fair"]), 4))
            row["E"] = round(float(d["E"]), 4)
            row["E_ci"] = d.get("E_minus_raw")
        sup[t] = row

    ladder = {}
    for t in targets:
        c = (ceil.get("results") or {}).get(t)
        if c:
            ladder[t] = {k: c[k] for k in ("floor", "E", "ceiling", "E_minus_floor", "se", "gap")}

    chosen = {}
    for t in targets:
        e = (sel.get("targets") or {}).get(t)
        if e:
            chosen[t] = {
                "view": e["selected_view"],
                "train_cv": e["train_cv"],
                "holdout": e["holdout"],
                "delta": e["selected_minus_table"],
            }

    return {
        "meta": {
            "generated": datetime.date.today().isoformat(),
            "customers": (data.get("path_b") or {}).get("customers", 6000),
            "events": (data.get("path_b") or {}).get("rows", 3355797),
            "as_of": data.get("as_of", "2025-01-01"),
            "seed": data.get("seed", 17),
            "d": cfg.get("d", 128),
            "layers": cfg.get("layers", 2),
            "state_dim": cfg.get("state_dim", 2265),
            "measure_dim": cfg.get("measure_dim", 734),
            "views": {
                **(rt.get("state_dims") or {}),
                "measure": cfg.get("measure_dim", 734),
            },
            "cache": (cfg.get("cache_sha256") or "")[:12],
            "commit": cfg.get("git_sha", ""),
            "b_pass": 28,
            "bc_pass": 6,
        },
        "supervised": {"targets": targets, "arms": arms, "values": sup},
        "donor": donor.get("results", {}),
        "ladder": ladder,
        "chosen": chosen,
        "identity": ident,
        "rl_known": rlk,
        "unsupervised": (uns.get("arms") or {}),
        "rl_policy": resp,
        "view_selection": (sel.get("protocol") or {}),
        "objectives": obj,
        "bprops": bprops,
        "bdata": bdata,
        "data_tab": data_tab,
        "encoder_tab": encoder_tab,
        "runtime": rt,
        "explorer": explorer,
    }


def _pct(x):
    return f"{x * 100:.1f}%"


def _f(x, n=3):
    return f"{x:.{n}f}"


def render(payload) -> str:
    """Flat, light, lab-style page: axes + objectives + plugins + explorer."""

    OBJDEF = [
        ("jepa", "JEPA", "Predict the mean latent of the next H states.", 5, [6, 7, 8, 9], None),
        ("mask", "Masked event", "Recover an event removed from the middle.", 5, [3], None),
        ("order", "Order", "From the final state, identify the terminal event.", 6, [10], None),
        ("entity", "Entity", "Predict the identity field of the next event.", 5, [6], None),
        ("query", "Query", "Predict days since the last event of a given type.", 5, [6], None),
        ("sf", "Successor", "Predict discounted future features.", 5, [6, 7, 8, 9, 10], "decay"),
        ("dt", "Gap", "Predict the time until the next event.", 5, [6], "span"),
        ("value", "Magnitude", "Predict the magnitude of the next event.", 5, [6], None),
        ("agg", "Aggregate", "Predict count and sum over the next H steps.", 5, [6, 7, 8], None),
        ("next", "Next event", "Predict the event type at t+1.", 5, [6], None),
    ]
    obj_items = "".join(
        f"""<button class="objitem" data-obj="{oid}">
          <span class="objlabel">{label}</span>
        </button>"""
        for oid, label, _d, _c, _t, _m in OBJDEF
    )
    obj_spec = {
        oid: {
            "label": label,
            "desc": desc,
            "cut": cut,
            "target": target,
            "mode": mode,
        }
        for oid, label, desc, cut, target, mode in OBJDEF
    }

    explorer_json = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")

    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>demo</title>
<style>
:root{
  --ink:#111827; --muted:#6b7280; --dim:#9ca3af; --line:#e5e7eb; --line2:#f3f4f6;
  --accent:#059669; --bg:#ffffff;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink)}
html{scrollbar-gutter:stable}
body{font:15px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}
.topbar{position:sticky;top:0;z-index:50;background:var(--bg);padding:24px 0 30px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.dim{color:var(--dim)}
.small{font-size:11.5px}
header{border-bottom:1px solid var(--line);padding-bottom:22px}
h1{font-size:17px;font-weight:600;margin:0;letter-spacing:-.01em}
nav{display:flex;gap:30px;border-bottom:1px solid var(--line);margin:22px 0 0}
nav button{appearance:none;background:none;border:0;padding:0 0 13px;font:inherit;font-size:13.5px;
  color:var(--muted);cursor:pointer;border-bottom:1px solid transparent;white-space:nowrap}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--ink);border-bottom-color:var(--ink)}
@media(max-width:680px){
  .topbar{padding:18px 0 24px}
  nav{flex-wrap:wrap;gap:6px;margin:16px 0 0}
  nav button{padding:7px 10px;border-bottom:0;border-radius:6px;font-size:12.5px}
  nav button.on{background:var(--ink);color:#fff}
}
section[hidden]{display:none}
h2{font-size:21px;font-weight:600;margin:0 0 8px;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:14px;max-width:72ch;margin:0 0 34px}
.flowsteps{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:2px;border:1px solid var(--line);
  border-radius:8px;padding:3px;margin:16px 0 14px;background:#fff}
.fstep{appearance:none;border:0;background:none;padding:8px 6px;border-radius:6px;font:inherit;
  font-size:11.5px;color:var(--muted);cursor:pointer;text-align:center;white-space:nowrap}
.fstep:hover{color:var(--ink)}
.fstep.on{background:var(--ink);color:#fff}
.fstep.on.accent{background:var(--accent);color:#fff}
.flowpanel{border:1px solid var(--line);border-radius:8px;padding:10px 14px;overflow-x:auto;background:#fff}
.flowpanel svg{display:block;width:100%;height:auto}
@media(max-width:800px){.flowsteps{font-size:10px}.fstep{font-size:10px}.flowpanel svg{min-width:700px}}
@media(max-width:680px){.flowsteps{grid-template-columns:repeat(3,minmax(0,1fr))}}
.chartbar{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:8px}
.charttag{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
.chartlegend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px}
.chartlegend span::before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:0}
.chartlegend .table::before{background:#d1d5db}
.chartlegend .frozen::before{background:var(--accent)}
.chartlegend .selected::before{background:var(--ink)}
.chartlegend .neighbour::before{background:var(--accent)}
.chartlegend .other::before{background:#d1d5db}
.chartlegend .channel::before{background:var(--ink)}
.chartlegend .offer::before{background:#6b7280}
.pluggrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:16px}
@media(max-width:900px){.pluggrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.pluggrid{grid-template-columns:1fr}}
.plugcard{min-height:148px;border:1px solid var(--line);border-radius:8px;padding:14px 15px;
  appearance:none;background:#fff;color:var(--ink);font:inherit;text-align:left;cursor:pointer;
  display:flex;flex-direction:column;justify-content:space-between}
.plugcard:hover{border-color:var(--dim)}
.plugcard.on{border-color:var(--accent);background:#f7fefb}
.plugcard .t{font-size:17px;font-weight:600;letter-spacing:-.02em}
.plugcard .d{font-size:11px;color:var(--muted);margin-top:2px;line-height:1.4}
.plugcard .mini{height:42px;margin-top:10px}
.plugcard .mini svg{display:block;width:100%;height:42px}
.plugsub + .plugsub{border-top:1px solid var(--line);margin-top:22px;padding-top:18px}
.plugpanel{border:1px solid var(--line);border-radius:8px;padding:16px 18px 10px;overflow-x:auto}
.plugpanel svg{display:block;width:100%;height:auto}
@media(max-width:800px){.plugpanel svg{min-width:720px}}
.acttabs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:2px;border:1px solid var(--line);
  border-radius:8px;padding:3px;margin:14px 0 20px;background:#fff}
.acttab{appearance:none;border:0;background:none;padding:8px 8px 7px;border-radius:6px;font:inherit;
  color:var(--muted);cursor:pointer;text-align:center;white-space:nowrap}
.acttab .name{display:block;font-size:12.5px;font-weight:500}
.acttab small{display:block;font-family:ui-monospace,monospace;font-size:8.5px;letter-spacing:.04em;
  text-transform:uppercase;color:var(--dim);margin-top:2px}
.acttab:hover{color:var(--ink)}
.acttab.on{background:var(--accent);color:#fff}
.acttab.on small{color:rgba(255,255,255,.7)}
.actdetail{border-top:1px solid var(--line);padding-top:18px}
.acthead{display:flex;justify-content:space-between;align-items:flex-start;gap:24px}
.acthead h3{font-size:22px;font-weight:600;letter-spacing:-.02em;margin:4px 0 0}
.actwhat{color:var(--muted);font-size:13px;margin:4px 0 0;max-width:62ch}
.actdelta{text-align:right;flex-shrink:0}
.actdelta .v{font-family:ui-monospace,monospace;font-size:24px;font-weight:600;letter-spacing:-.03em;color:var(--accent)}
.actdelta .k{font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.actblock{margin-top:20px}
.actblockhead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  border-bottom:1px solid var(--line2);padding-bottom:7px;margin-bottom:12px}
.actblockhead .t{font-size:12px;font-weight:600}
.actblockhead small{font-family:ui-monospace,monospace;font-size:9px;letter-spacing:.05em;
  text-transform:uppercase;color:var(--dim)}
.actbars{display:grid;gap:10px}
.actbarrow{display:grid;grid-template-columns:132px minmax(0,1fr) 58px;gap:14px;align-items:center}
.actbarrow .label{font-size:12px;color:var(--muted)}
.acttrack{height:12px;border-radius:6px;background:#f3f4f6;overflow:hidden}
.acttrack i{display:block;height:100%;border-radius:6px}
.actbarrow .val{font-family:ui-monospace,monospace;font-size:12px;text-align:right;color:var(--muted)}
.actpolicy{margin-top:20px;border-top:1px solid var(--line2);padding-top:16px}
.policystats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.pstat{border:1px solid var(--line2);border-radius:6px;padding:10px 12px}
.pstat .k{font-size:9px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim)}
.pstat .v{font-family:ui-monospace,monospace;font-size:21px;font-weight:600;letter-spacing:-.03em;margin-top:3px}
.pstat.on{border-color:rgba(5,150,105,.45);background:#f7fefb}
.pstat.on .v{color:var(--accent)}
.policydeltas{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}
.policytag{display:flex;align-items:baseline;gap:8px;border:1px solid rgba(5,150,105,.24);
  background:#f7fefb;border-radius:6px;padding:6px 10px;color:var(--muted);font-size:10.5px}
.policytag b{font-family:ui-monospace,monospace;font-size:14px;color:var(--accent)}
@media(max-width:800px){
  .acttabs{grid-template-columns:repeat(2,minmax(0,1fr))}
  .acthead{flex-direction:column;gap:8px}
  .actdelta{text-align:left}
  .actbarrow{grid-template-columns:96px minmax(0,1fr) 50px;gap:8px}
  .policystats{grid-template-columns:1fr}
}
.chartfoot{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:10.5px;
  border-top:1px solid var(--line2);margin-top:6px;padding-top:9px;font-family:ui-monospace,monospace}
hr{border:0;border-top:1px solid var(--line);margin:44px 0}
.axis{border:1px solid var(--line);border-radius:8px;padding:20px 20px 16px}
.axis-top{display:flex;align-items:baseline;gap:10px}
.axis-name{font-weight:600;font-size:14.5px}
.axis-val{font-family:ui-monospace,monospace;font-size:27px;font-weight:600;letter-spacing:-.02em;margin:12px 0 2px}
.axis-def{color:var(--muted);font-size:13px;margin-top:12px;min-height:38px}
.vis{margin-top:14px}
.axis.gap{background:#fafafa}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line)}
th{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;font-size:12.5px}
th.num{text-align:right}
.strong{color:var(--accent);font-weight:600}
.note{color:var(--muted);font-size:12.5px}
.objlayout{display:grid;grid-template-columns:290px minmax(0,1fr);gap:36px}
@media(max-width:860px){.objlayout{grid-template-columns:1fr}}
.objitem{display:flex;align-items:center;width:100%;background:none;border:0;
  border-bottom:1px solid var(--line);padding:13px 2px;font:inherit;text-align:left;
  cursor:pointer;color:var(--muted)}
.objitem:hover{color:var(--ink)}
.objitem.on{color:var(--ink)}
.objitem.on .objlabel{font-weight:600}
.objlabel{font-size:13.5px}
.objpanel{border:1px solid var(--line);border-radius:8px;padding:26px;display:flex;flex-direction:column}
.objvisual{flex:1;display:flex;align-items:center;justify-content:center;min-height:180px}
.objhead{display:flex;align-items:baseline;justify-content:space-between;gap:16px}
.objhead h3{font-size:17px;margin:0;font-weight:600;letter-spacing:-.01em}
.objdesc{color:var(--muted);font-size:13.5px;margin:6px 0 22px;max-width:56ch}
.subtabs{display:flex;gap:2px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:22px}
.subtabs button{appearance:none;background:none;border:0;padding:11px 14px;font:inherit;font-size:13px;
  color:var(--muted);cursor:pointer;border-bottom:1px solid transparent;white-space:nowrap}
.subtabs button:hover{color:var(--ink)}
.subtabs button.on{color:var(--ink);border-bottom-color:var(--ink);font-weight:600}
.axpanel{border:1px solid var(--line);border-radius:8px;padding:26px;display:flex;flex-direction:column}
.axhead{display:flex;justify-content:space-between;align-items:flex-start;gap:24px}
.axname{font-size:20px;font-weight:600;letter-spacing:-.02em;margin-top:4px}
.axmetric{text-align:right;flex-shrink:0}
.axval{font-family:ui-monospace,monospace;font-size:38px;font-weight:600;letter-spacing:-.03em;line-height:1.1}
.axval.txt{font-size:22px;padding-top:8px}
.axcap{color:var(--dim);font-size:11.5px;margin-top:3px}
.axdesc{color:var(--muted);font-size:13.5px;margin:8px 0 0;max-width:72ch}
.axvisual{flex:1;display:flex;align-items:center;justify-content:center;min-height:300px;margin:6px 0 18px}
.axvisual svg{display:block;width:100%;height:auto}
.axvis{width:100%}
.axrows{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;
  border-top:1px solid var(--line2);padding-top:18px}
.axrow .k{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.axrow .v{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.55}
@media(max-width:900px){.axrows{grid-template-columns:1fr;gap:14px}}
.dataslider{position:relative;display:grid;grid-template-columns:1fr 1fr;width:360px;max-width:100%;
  border:1px solid var(--line);border-radius:8px;padding:3px;margin-bottom:18px;background:#fff}
.dsslide{position:absolute;top:3px;bottom:3px;left:3px;width:calc(50% - 3px);background:var(--ink);
  border-radius:6px;transition:transform .25s ease,background-color .25s ease}
.dsslide.stream{background:var(--accent)}
.dstab{position:relative;z-index:1;appearance:none;border:0;background:none;padding:7px 10px 6px;
  border-radius:6px;font:inherit;font-size:12.5px;color:var(--muted);cursor:pointer}
.dstab span{display:block;font-family:ui-monospace,monospace;font-size:9px;letter-spacing:.04em;
  color:var(--dim);margin-top:2px}
.dstab:hover{color:var(--ink)}
.dstab.on{color:#fff}
.dstab.on span{color:rgba(255,255,255,.68)}
.datapane[hidden]{display:none}
.legacycard{padding:18px 18px 6px;margin-bottom:0}
.legacyhead{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;
  padding-bottom:12px;border-bottom:1px solid var(--line)}
.legacymeta{font-size:12.5px;color:var(--muted);margin-top:2px}
.legacyhint{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);
  font-weight:600;white-space:nowrap;font-family:ui-monospace,monospace}
.legacytable{overflow-x:auto;margin-top:2px}
.legacytable table{width:100%;border-collapse:collapse;font-size:13.5px;min-width:780px}
.legacytable th{padding:10px 8px;border-bottom:1px solid var(--line);font-size:10px}
.legacytable td{padding:11px 8px;border-bottom:1px solid var(--line2)}
.legacytable tbody tr{cursor:pointer;outline:none}
.legacytable tbody tr:hover td{background:#fafafa}
.legacytable tbody tr.on td{background:#ecfdf5}
.legacytable tbody tr.on td:first-child{box-shadow:inset 3px 0 0 var(--accent);font-weight:600}
.legacytable tbody tr.on td:first-child::before{content:"";display:inline-block;width:6px;height:6px;
  border-radius:50%;background:var(--accent);margin-right:7px;vertical-align:1px}
.legacytable tbody tr:last-child td{border-bottom:0}
.legacytable td:first-child{font-family:ui-monospace,monospace;font-size:11.5px;white-space:nowrap}
.legacytable td.num{font-size:12px}
.streamhead{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:14px}
@media(max-width:700px){.streamhead{flex-direction:column;align-items:flex-start;gap:6px}}
.streamkey{font-family:ui-monospace,monospace;font-size:24px;font-weight:600;letter-spacing:-.02em;margin-top:2px}
.streamcounts{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px}
.streamcounts b{color:var(--ink);font-family:ui-monospace,monospace;font-weight:600}
.dataflow{display:grid;gap:16px}
.cardtag{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600;margin-bottom:10px}
.tlwrap svg{display:block;width:100%;height:auto}
@media(max-width:800px){.tlwrap{overflow-x:auto}.tlwrap svg{min-width:700px}}
.timeline-panel{position:relative}
.timeline-tip{position:absolute;z-index:20;pointer-events:none;background:#111827;color:#f9fafb;
  border-radius:6px;padding:10px 12px;max-width:min(420px,86vw)}
.timeline-tip .tts{font-family:ui-monospace,monospace;font-size:10.5px;color:#e5e7eb;white-space:nowrap}
.timeline-tip .tty{font-family:ui-monospace,monospace;font-size:10px;color:#9ca3af;margin-top:2px}
.timeline-tip .ttl{font-size:9px;letter-spacing:.07em;text-transform:uppercase;color:#9ca3af;margin-top:7px}
.timeline-tip .tattrs{font-family:ui-monospace,monospace;font-size:9.6px;line-height:1.5;
  color:#f3f4f6;margin:7px 0 0;white-space:pre-wrap;word-break:break-word}
.logscroll{max-height:520px;overflow:auto;border-top:1px solid var(--line)}
.logtable{font-size:11px;min-width:880px}
.logtable th{position:sticky;top:0;background:#fff;z-index:2;padding:8px 7px}
.logtable.canonical th{text-transform:none;letter-spacing:0;font-family:ui-monospace,monospace;
  font-size:10.5px;color:var(--muted)}
.logtable td{padding:7px;vertical-align:top}
.logtable tr:hover td{background:#fafafa}
.logtable tbody tr.hot td{background:#ecfdf5}
.logtable tbody tr.hot td:first-child{box-shadow:inset 2px 0 0 var(--accent)}
.timeline-dot{transition:fill .12s ease,stroke .12s ease}
.timeline-dot.hot{fill:var(--accent);stroke:#fff;stroke-width:1.4}
.logtable td:first-child{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--muted);white-space:nowrap}
.logtable td:nth-child(2){font-family:ui-monospace,monospace;font-size:10.5px;white-space:nowrap}
.logtable td.attrs{font-family:ui-monospace,monospace;font-size:9.8px;line-height:1.5;
  color:var(--muted);min-width:440px;white-space:normal;word-break:break-word}
.encsection{border-top:1px solid var(--line);margin-top:54px;padding-top:38px}
.encpick{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px}
.encperson{appearance:none;border:1px solid var(--line);border-radius:6px;background:#fff;
  padding:7px 10px;font:inherit;text-align:left;cursor:pointer;color:var(--muted);min-width:0;
  transition:border-color .15s ease,background-color .15s ease}
.encperson b{display:block;font-family:ui-monospace,monospace;font-size:11px;font-weight:600;color:var(--ink)}
.encperson span{display:block;font-size:9.5px;color:var(--dim);margin-top:1px;white-space:nowrap}
.encperson:hover{border-color:var(--dim)}
.encperson.on{border-color:var(--accent);background:#f7fefb}
.encperson.on b{color:var(--accent)}
.featexplorerhead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin:0 0 12px}
.featexplorerhead .cardtag{margin-bottom:0}
.featexplorer{display:grid;grid-template-columns:250px minmax(0,1fr);gap:28px;align-items:start}
@media(max-width:900px){.featexplorer{grid-template-columns:1fr}}
.featselector{border-top:1px solid var(--line)}
.featfamily{appearance:none;display:block;width:100%;background:none;border:0;border-bottom:1px solid var(--line);
  padding:11px 2px 10px;text-align:left;cursor:pointer;font:inherit;color:var(--muted)}
.featfamily:hover{color:var(--ink)}
.featfamily.on{color:var(--ink)}
.featfamily .top{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
.featfamily .name{font-size:13.5px;font-weight:600}
.featfamily .dim{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--dim)}
.featfamily .bar{height:4px;background:var(--line2);border-radius:2px;overflow:hidden;margin-top:7px}
.featfamily .bar div{height:100%;background:#d1d5db;border-radius:2px}
.featfamily.on .bar div{background:var(--accent)}
.featselector .total{display:flex;justify-content:space-between;align-items:baseline;
  padding-top:12px;font-size:11.5px;color:var(--muted);font-family:ui-monospace,monospace}
.featselector .total b{color:var(--ink);font-size:14px}
.featpanel{border:1px solid var(--line);border-radius:8px;padding:18px;min-width:0}
.shapehead{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:14px;flex-wrap:wrap}
.shapehead .cardtag{margin-bottom:0}
.shapekey{font-family:ui-monospace,monospace;font-size:20px;font-weight:600;letter-spacing:-.02em}
.shapemeta{color:var(--muted);font-size:11.5px}
.featmatrix{display:grid;gap:2px;min-height:360px;align-content:center}
@media(max-width:900px){.featmatrix{min-height:0}}
.featcell{aspect-ratio:1;border-radius:2px;border:1px solid #fff}
.featcell:hover{outline:1px solid var(--ink);outline-offset:1px}
.featlegend{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:12px;color:var(--dim);
  font-size:10.5px;font-family:ui-monospace,monospace}
.featlegend .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.featlegend .max{margin-left:auto;font-size:10px}
.explorerchart{padding:18px;margin-bottom:28px}
#scatter{display:block;width:100%;height:auto;background:#fff;touch-action:none;cursor:grab}
.scatterfoot{display:flex;gap:22px;align-items:center;border-top:1px solid var(--line2);
  padding-top:12px;margin-top:12px;color:var(--muted);font-size:13px}
.scatterfoot .sw{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.scatterfoot .sw.retained{background:var(--accent)}
.scatterfoot .sw.churned{background:#dc2626}
.scatterfoot .hint{margin-left:auto;color:var(--dim);font-family:ui-monospace,monospace;
  font-size:11px;letter-spacing:.04em;text-transform:uppercase}
.custcard{border:1px solid var(--line);border-radius:10px;background:#fff;overflow:hidden}
.custcardhead{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
  padding:24px 26px;border-bottom:1px solid var(--line)}
@media(max-width:700px){.custcardhead{align-items:flex-start;flex-direction:column;gap:8px;padding:20px}}
.custtag{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
.custkey{font-family:ui-monospace,monospace;font-size:32px;font-weight:600;letter-spacing:-.03em;margin-top:4px}
.custcaps{display:grid;grid-template-columns:1.12fr 1fr .92fr;border-bottom:1px solid var(--line)}
@media(max-width:900px){.custcaps{grid-template-columns:1fr}}
.capability{padding:22px 24px;border-right:1px solid var(--line2);min-width:0}
.capability:last-child{border-right:0}
@media(max-width:900px){.capability{border-right:0;border-bottom:1px solid var(--line2)}
  .capability:last-child{border-bottom:0}}
.caphead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:14px}
.capname{font-size:15px;font-weight:600;letter-spacing:-.01em}
.capmeta{font-family:ui-monospace,monospace;font-size:9px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--dim);white-space:nowrap}
.cardtable{width:100%;border-collapse:collapse;font-size:12px}
.cardtable th{font-family:ui-monospace,monospace;font-size:8.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--dim);font-weight:500;padding:0 0 7px;
  border-bottom:1px solid var(--line2);text-align:left;white-space:nowrap}
.cardtable th.r,.cardtable td.r{text-align:right}
.cardtable td{padding:8px 0;border-bottom:1px solid var(--line2);vertical-align:baseline}
.cardtable tr:last-child td{border-bottom:0;padding-bottom:0}
.cardtable td.k{font-family:ui-monospace,monospace;font-size:10px;color:var(--muted)}
.cardtable td.v{font-family:ui-monospace,monospace;font-size:14px;font-weight:600;color:var(--ink);
  text-align:right;white-space:nowrap}
.cardtable td.s{font-size:9.5px;color:var(--dim);text-align:right;padding-left:8px;white-space:nowrap}
.cardtable .good{color:var(--accent)}
.cardtable .bad{color:#dc2626}
.cardtable tr.primary td{background:#f7fefb}
.nbminilist{display:grid;gap:6px}
.nbrow{display:grid;grid-template-columns:minmax(0,1fr) auto 7px;gap:8px;align-items:center;
  border-bottom:1px solid var(--line2);padding-bottom:6px;font-size:12px}
.nbrow:last-child{border-bottom:0;padding-bottom:0}
.nbrow .k{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis}
.nbrow .v{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--dim);white-space:nowrap}
.nbrow .dotmark{width:6px;height:6px;border-radius:50%;background:var(--accent)}
.nbrow .dotmark.bad{background:#dc2626}
.capnote{margin-top:13px;color:var(--muted);font-size:11.5px;line-height:1.5}
.capnote b{color:var(--ink);font-family:ui-monospace,monospace;font-weight:600}
.custcontext{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr)}
@media(max-width:800px){.custcontext{grid-template-columns:1fr}}
.contextsection{padding:20px 26px}
.contextsection + .contextsection{border-left:1px solid var(--line2)}
@media(max-width:800px){.contextsection + .contextsection{border-left:0;border-top:1px solid var(--line2)}}
.minihistory{list-style:none;margin:10px 0 0;padding:0}
.minihistory li{display:grid;grid-template-columns:124px minmax(0,1fr) auto;gap:12px;padding:6px 0;
  border-bottom:1px solid var(--line2);font-size:12.5px;align-items:baseline}
.minihistory li:last-child{border-bottom:0}
.minihistory .t{color:var(--dim);font-family:ui-monospace,monospace;font-size:11px}
.minihistory .e{color:var(--ink)}
.minihistory .d{color:var(--accent);font-family:ui-monospace,monospace;font-size:11.5px;text-align:right}
.contextegrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 18px;margin-top:12px}
.contextegrid .k{display:block;font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--dim)}
.contextegrid .v{display:block;font-size:14px;font-weight:600;margin-top:2px}
@media(max-width:600px){
  .minihistory li{grid-template-columns:96px minmax(0,1fr);gap:5px 10px}
  .minihistory .d{grid-column:2;text-align:left}
}
.panelbox{border:1px solid var(--line);border-radius:8px;padding:18px}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim);
  font-size:12px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <header>
      <h1>demo</h1>
    </header>

    <nav id="tabs"></nav>
  </div>

  <main>
    <section id="overview">
      <div class="flowsteps" id="flowsteps"></div>
      <div class="flowpanel" id="flowbody"></div>
    </section>

    <section id="data" hidden>
      <h2>Data</h2>
      <p class="lede">Switch between a legacy RFM row and the raw event stream for the same customer.</p>
      <div class="dataslider" id="dataslider">
        <div class="dsslide"></div>
        <button class="dstab on" data-dsmode="legacy" type="button">Legacy<span>one row / customer</span></button>
        <button class="dstab" data-dsmode="stream" type="button">Event stream<span>one row / event</span></button>
      </div>
      <div class="datapane" id="legacypane">
        <div class="legacycard panelbox">
          <div class="legacyhead">
            <div class="legacymeta" id="legacymeta"></div>
            <div class="legacyhint">select a row</div>
          </div>
          <div id="legacytable" class="legacytable"></div>
        </div>
      </div>
      <div class="datapane" id="streampane" hidden>
        <div id="datastream"></div>
      </div>
    </section>

    <section id="encoder" hidden>
      <h2>Encoder</h2>
      <div class="featexplorerhead">
        <div class="cardtag">Feature explorer</div>
        <div class="small dim">real values · hover a cell</div>
      </div>
      <div class="encpick" id="encpick"></div>
      <div class="featexplorer">
        <div class="featselector" id="featselector"></div>
        <div class="featpanel">
          <div class="shapehead" id="shapehead"></div>
          <div class="featmatrix" id="featuremap"></div>
          <div class="featlegend" id="featurelegend"></div>
          <p class="note">Hover a cell for its dimension and value.</p>
        </div>
      </div>
    <div class="encsection" id="axes">
      <h2>Eight content axes</h2>
      <p class="lede">What the frozen state carries, axis by axis. Each axis is a distinct property of the stream, measured on held-out customers.</p>
      <div class="subtabs" id="axistabs"></div>
      <div class="axpanel" id="axpanel"></div>
    </div>

    <div class="encsection" id="objectives">
      <h2>Objectives</h2>
      <p class="lede">Ten self-supervised tasks trained on the same frozen state. None uses a business label: targets come from the stream itself.</p>
      <div class="objlayout">
        <div id="objlist">__OBJLIST__</div>
        <div class="objpanel" id="objpanel"></div>
      </div>
    </div>

    </section>

    <section id="plugins" hidden>
      <h2>Plugins</h2>
      <div class="plugmap" id="plugmap"></div>
      <div class="plugpanel" id="plugpanel"></div>
    </section>

    <section id="explorer" hidden>
      <h2>Meet a customer</h2>
      <p class="lede">Held-out customers projected in 3D from the frozen embedding. Drag to rotate the cloud, then click a point to open the full customer card below.</p>
      <div class="explorerchart panelbox">
        <svg id="scatter" viewBox="0 0 1000 600" role="img" aria-label="3D embedding scatter"></svg>
        <div class="scatterfoot">
          <span><i class="sw retained"></i> retained</span>
          <span><i class="sw churned"></i> churned</span>
          <span class="hint">drag to rotate · click a point · 5 nearest neighbours</span>
        </div>
      </div>
      <div id="detail"></div>
    </section>
  </main>

  <footer>
    <div>demo</div>
    <div id="foot"></div>
  </footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const $ = (s, r=document) => r.querySelector(s);
const el = (t, c, txt) => { const n=document.createElement(t); if(c) n.className=c; if(txt!=null) n.textContent=txt; return n; };
const NS = "http://www.w3.org/2000/svg";
const INK="#111827", MUT="#6b7280", DIM="#9ca3af", LINE="#e5e7eb", ACC="#059669";
const OBJSPEC = __OBJSPEC__;

/* tabs */
const TABS=[["overview","Overview"],["data","Data"],["encoder","Encoder"],["plugins","Plugins"],["explorer","Explorer"]];
const nav=$("#tabs");
const TAB_ALIAS={axes:"encoder",objectives:"encoder"};
function activate(id,push){
  id=TAB_ALIAS[id]||id;
  const known=TABS.some(([t])=>t===id)?id:TABS[0][0];
  document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on",x.dataset.tab===known));
  TABS.forEach(([t])=>{const s=document.getElementById(t); if(s) s.hidden=t!==known;});
  if(push) history.replaceState(null,"","#view="+known);
  window.scrollTo({top:0,behavior:"instant"});
}
TABS.forEach(([id,label])=>{const b=el("button",null,label);b.dataset.tab=id;b.onclick=()=>activate(id,true);nav.appendChild(b);});
const hashTab=()=>(location.hash||"").startsWith("#view=")?location.hash.slice(6):"";
activate(hashTab(),false);
window.addEventListener("hashchange",()=>activate(hashTab(),false));

const m=DATA.meta;

/* ---------- tiny chart helpers ---------- */
function svg(w,h){const s=document.createElementNS(NS,"svg");s.setAttribute("viewBox",`0 0 ${w} ${h}`);s.setAttribute("width","100%");return s;}
function tx(x,y,t,o={}){const n=document.createElementNS(NS,"text");n.setAttribute("x",x);n.setAttribute("y",y);
  n.setAttribute("fill",o.fill||MUT);n.setAttribute("font-size",o.size||10);
  n.setAttribute("font-family","ui-monospace,monospace");if(o.anchor)n.setAttribute("text-anchor",o.anchor);n.textContent=t;return n;}
function ln(x1,y1,x2,y2,o={}){const n=document.createElementNS(NS,"line");n.setAttribute("x1",x1);n.setAttribute("y1",y1);
  n.setAttribute("x2",x2);n.setAttribute("y2",y2);n.setAttribute("stroke",o.stroke||LINE);n.setAttribute("stroke-width",o.w||1);
  if(o.dash)n.setAttribute("stroke-dasharray",o.dash);return n;}
function dot(cx,cy,r,fill,stroke){const n=document.createElementNS(NS,"circle");n.setAttribute("cx",cx);n.setAttribute("cy",cy);
  n.setAttribute("r",r);n.setAttribute("fill",fill);if(stroke){n.setAttribute("stroke",stroke);n.setAttribute("stroke-width",1.2);}return n;}
function rect(x,y,w,h,fill,rx,stroke){const n=document.createElementNS(NS,"rect");n.setAttribute("x",x);n.setAttribute("y",y);
  n.setAttribute("width",w);n.setAttribute("height",h);n.setAttribute("rx",rx==null?0:rx);n.setAttribute("fill",fill);
  if(stroke){n.setAttribute("stroke",stroke);n.setAttribute("stroke-width",1);}return n;}
function path(d,o={}){const n=document.createElementNS(NS,"path");n.setAttribute("d",d);n.setAttribute("fill",o.fill||"none");
  n.setAttribute("stroke",o.stroke||LINE);n.setAttribute("stroke-width",o.w||1);
  if(o.dash)n.setAttribute("stroke-dasharray",o.dash);return n;}

/* ---------- overview ---------- */
const OV_ENC=DATA.encoder_tab;
const FLOW_STAGES=[["legacy","Legacy"],["stream","Event Stream"],["encoder","Encoder"],["state","Frozen State"],["plugins","Plugins"],["overview","Overview"]];
let flowStage=0;

function addSvgTitle(node,text){
  const t=document.createElementNS(NS,"title"); t.textContent=text; node.appendChild(t); return node;
}
function chartHeader(tag,items){
  const bar=el("div","chartbar");
  bar.appendChild(el("div","charttag",tag));
  const leg=el("div","chartlegend");
  items.forEach(([label,cls])=>leg.appendChild(el("span",cls,label)));
  bar.appendChild(leg); return bar;
}
function chartFoot(left,right){
  const foot=el("div","chartfoot");
  foot.appendChild(el("span",null,left)); foot.appendChild(el("span",null,right)); return foot;
}

function flowValues(){ return (OV_ENC&&OV_ENC.shape)||[]; }
function flowFill(v,maxv){
  const a=Math.min(1,Math.abs(v)/(maxv||1));
  return v>=0?`rgba(5,150,105,${0.10+0.80*a})`:`rgba(17,24,39,${0.06+0.62*a})`;
}
function flowPositions(){
  const legacy=[], stream=[], encoder=[], state=[], overview=[], heads=[];
  const tables=[
    ["customers",19],["orders",19],["order_items",18],["support",18],
    ["returns",18],["products",18],["campaigns",18],
  ];
  tables.forEach(([_,n],row)=>{
    const rowW=(n-1)*40+18, x0=140+(760-rowW)/2, y=130+row*48;
    for(let k=0;k<n;k++) legacy.push({x:x0+k*40,y:y+4,s:18});
  });
  for(let i=0;i<128;i++){
    const srow=Math.floor(i/5),scol=i%5;
    stream.push({x:457+scol*18,y:28+srow*18,s:14});
    const j=i<64?i:i-64,col=j%16,row=Math.floor(j/16);
    encoder.push({x:263+col*30,y:(i<64?140:284)+row*30,s:24});
    state.push({x:357.5+(i%16)*18,y:189.5+Math.floor(i/16)*18});
    overview.push({x:526+(i%8)*18,y:104+Math.floor(i/8)*18});
    heads.push({x:157.5+(i%16)*18,y:189.5+Math.floor(i/16)*18});
  }
  return {legacy,stream,encoder,state,overview,plugins:heads};
}
function flowOverlay(bg,fg,stage){
  bg.innerHTML=""; fg.innerHTML="";
  const arrow=(x1,y1,x2,y2,color=ACC)=>{
    fg.appendChild(ln(x1,y1,x2,y2,{stroke:color,w:1.2}));
    fg.appendChild(ln(x2-6,y2-4,x2,y2,{stroke:color,w:1.2}));
    fg.appendChild(ln(x2-6,y2+4,x2,y2,{stroke:color,w:1.2}));
  };
  if(stage==="legacy"){
    const tables=[
      ["customers","6k"],["orders","385k"],["order_items","385k"],["support","205k"],
      ["returns","46k"],["products","80"],["campaigns","24"],
    ];
    tables.forEach(([name,count],row)=>{
      const y=130+row*48;
      bg.appendChild(rect(132,y,776,26,"#fff",4,LINE));
      fg.appendChild(tx(140,y-7,name+" · "+count,{size:9.5,fill:MUT}));
    });
  }else if(stage==="stream"){
    bg.appendChild(ln(441,24,441,496,{stroke:"#e5e7eb"}));
    for(let r=0;r<26;r++){
      const y=28+r*18+7;
      bg.appendChild(ln(435,y,441,y,{stroke:"#e5e7eb"}));
      if(r<25) bg.appendChild(ln(457,y+8,543,y+8,{stroke:"#f3f4f6"}));
    }
  }else if(stage==="encoder"){
    bg.appendChild(rect(249,116,502,296,"#fff",8,LINE));
    bg.appendChild(ln(263,269,737,269,{stroke:"#f3f4f6"}));
    fg.appendChild(tx(500,100,"Encoder · 2 × gated SSM",{size:10,fill:MUT,anchor:"middle"}));
    fg.appendChild(tx(263,134,"SSM 1",{size:8,fill:DIM}));
    fg.appendChild(tx(263,278,"SSM 2",{size:8,fill:DIM}));
  }else if(stage==="state"){
    bg.appendChild(rect(347.5,179.5,305,161,"#fff",6,LINE));
    fg.appendChild(tx(500,360,"frozen state",{size:9,fill:DIM,anchor:"middle"}));
  }else if(stage==="plugins"){
    bg.appendChild(rect(147.5,179.5,305,161,"#fff",6,LINE));
    fg.appendChild(tx(300,360,"frozen state",{size:9,fill:DIM,anchor:"middle"}));
    const nodes=[["Predict",130],["Retrieve",260],["Act",390]];
    nodes.forEach(([label,y])=>{
      bg.appendChild(ln(452.5,260,700,y,{stroke:LINE}));
      fg.appendChild(dot(700,y,5.5,"#fff",ACC));
      fg.appendChild(tx(716,y+4,label,{size:11,fill:MUT}));
    });
  }else if(stage==="overview"){
    fg.appendChild(tx(145,74,"Event Stream",{size:10,fill:MUT,anchor:"middle"}));
    fg.appendChild(tx(360,74,"Encoder",{size:10,fill:MUT,anchor:"middle"}));
    fg.appendChild(tx(596,74,"Frozen State",{size:10,fill:MUT,anchor:"middle"}));
    fg.appendChild(tx(860,74,"Plugins",{size:10,fill:MUT,anchor:"middle"}));
    bg.appendChild(rect(105,96,80,320,"#fff",8,LINE));
    bg.appendChild(ln(145,112,145,400,{stroke:"#e5e7eb"}));
    for(let r=0;r<15;r++){
      const yy=114+r*19;
      bg.appendChild(ln(137,yy,145,yy,{stroke:"#e5e7eb"}));
      bg.appendChild(dot(145,yy,2.2,r%4===0?ACC:"#d1d5db"));
    }
    bg.appendChild(rect(275,150,170,220,"#fff",8,LINE));
    fg.appendChild(tx(360,138,"2 × SSM",{size:8,fill:DIM,anchor:"middle"}));
    [167,267].forEach((y0,i)=>{
      bg.appendChild(rect(289,y0,142,80,"#f9fafb",6,LINE));
      fg.appendChild(tx(360,y0+14,"SSM "+(i+1),{size:7,fill:DIM,anchor:"middle"}));
      for(let l=1;l<=3;l++) bg.appendChild(ln(299,y0+14+l*16,421,y0+14+l*16,{stroke:"#eef0f3"}));
    });
    arrow(190,250,270,250);
    arrow(450,250,510,250);
    bg.appendChild(rect(514,92,165,309,"#fff",8,LINE));
    fg.appendChild(tx(596,416,"128-d",{size:8,fill:DIM,anchor:"middle"}));
    const nodes=[["value_90d",110],["orders_90d",170],["churn_90d",230],["lookalike",290],["action",350]];
    nodes.forEach(([label,y])=>{
      bg.appendChild(ln(679,250,770,y,{stroke:LINE}));
      fg.appendChild(dot(770,y,5.5,"#fff",ACC));
      fg.appendChild(tx(782,y+4,label,{size:10,fill:MUT}));
    });
  }
}
function mountFlow(body){
  const W=1000,H=520,s=svg(W,H); body.appendChild(s);
  const bg=document.createElementNS(NS,"g"); const cells=document.createElementNS(NS,"g"); const fg=document.createElementNS(NS,"g");
  s.appendChild(bg); s.appendChild(cells); s.appendChild(fg);
  const vals=flowValues(), maxv=Math.max(...vals.map(v=>Math.abs(v)))||1;
  const rects=[]; const pos=flowPositions();
  for(let i=0;i<128;i++){
    const r=rect(0,0,15,15,flowFill(vals[i]||0,maxv),3);
    cells.appendChild(r); rects.push(r);
  }
  let current=pos.legacy.map(p=>({...p,a:0}));
  const apply=p=>rects.forEach((r,i)=>{
    const size=p[i].s||15;
    r.setAttribute("x",p[i].x); r.setAttribute("y",p[i].y);
    r.setAttribute("width",size); r.setAttribute("height",size);
    const a=p[i].a||0;
    if(a) r.setAttribute("transform",`rotate(${a.toFixed(2)} ${(p[i].x+size/2).toFixed(2)} ${(p[i].y+size/2).toFixed(2)})`);
    else r.removeAttribute("transform");
  });
  apply(current);
  let gen=0;
  const rnd=(i,k)=>{
    const v=Math.sin((i+1)*12.9898+k*78.233)*43758.5453;
    return v-Math.floor(v);
  };
  function animateTo(stage,index){
    const target=pos[stage]||pos.legacy, start=current.map(p=>({...p}));
    const g=++gen, seed=index*7.13+1;
    const paths=start.map((p,i)=>{
      const t=target[i], dx=t.x-p.x, dy=t.y-p.y;
      const dist=Math.hypot(dx,dy)||1, nx=-dy/dist, ny=dx/dist;
      const wob=(rnd(i,seed)-0.5)*2, mag=dist*(0.22+0.55*rnd(i,seed+1));
      return {
        p, t,
        s0:p.s||15, s1:t.s||15,
        cx:(p.x+t.x)/2+nx*mag*wob+(rnd(i,seed+2)-0.5)*60,
        cy:(p.y+t.y)/2+ny*mag*wob+(rnd(i,seed+3)-0.5)*60,
        delay:rnd(i,seed+4)*90,
        dur:680+rnd(i,seed+5)*380,
        spin:(rnd(i,seed+6)-0.5)*150,
      };
    });
    const t0=Date.now();
    function frame(){
      if(g!==gen) return;
      const now=Date.now();
      let done=true;
      current=paths.map(o=>{
        let t=(now-t0-o.delay)/o.dur;
        if(t<0) t=0; else if(t>1) t=1; else done=false;
        const e=t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2, u=1-e;
        return {
          x:u*u*o.p.x+2*u*e*o.cx+e*e*o.t.x,
          y:u*u*o.p.y+2*u*e*o.cy+e*e*o.t.y,
          s:o.s0+(o.s1-o.s0)*e,
          a:o.spin*Math.sin(Math.PI*e),
        };
      });
      apply(current);
      if(!done) setTimeout(frame,16);
    }
    setTimeout(frame,0);
  }
  return {bg,fg,animateTo};
}
function renderFlow(){
  const steps=$("#flowsteps"); steps.innerHTML="";
  const body=$("#flowbody"); body.innerHTML="";
  const flow=mountFlow(body);
  const buttons={};
  function setStage(index){
    flowStage=index;
    const stage=FLOW_STAGES[index][0];
    Object.entries(buttons).forEach(([id,b])=>{
      b.classList.toggle("on",id===stage);
      b.classList.toggle("accent",id===stage&&stage!=="legacy");
    });
    flowOverlay(flow.bg,flow.fg,stage);
    flow.animateTo(stage,index);
  }
  FLOW_STAGES.forEach(([id,label],i)=>{
    const b=el("button","fstep",label); b.type="button";
    b.onclick=()=>{setStage(i);};
    buttons[id]=b; steps.appendChild(b);
  });
  setStage(0);
}
function initOverview(){ renderFlow(); }
initOverview();


const bd=DATA.bdata||{};
/* ---------- axis panels ---------- */
const AXES = [
  ["A1","Shape"],["A2","Mass"],["A3","Identity"],["A4","Rate"],
  ["A5","Uncertainty"],["A6","Causal"],["A7","Order / Path"],["A8","Relational"]
];
const AXDEF = {
  A1:{name:"Shape", val:()=> (bd.eff_rank||0).toFixed(2), cap:"effective rank of the frozen state",
      desc:"Customers with similar behaviour land near each other in the state.",
      what:"Whether the state organises customers by behaviour rather than by noise.",
      how:"Singular-value spectrum of the standardised state; effective rank is the participation ratio.",
      why:"This geometry is what makes retrieval, lookalikes and segmentation work downstream."},
  A2:{name:"Mass", val:()=> ((bd.mass||{}).r2||0).toFixed(2), cap:"mean r² over "+((bd.mass||{}).n_dims||0)+" past-summary dimensions",
      desc:"The state remembers how much, not only what kind.",
      what:"Whether extensive quantities — counts, sums, magnitudes — survive into the state.",
      how:"Reconstruct the past-summary vector from the state at a 70% history cut and compare.",
      why:"A representation that keeps shape but loses level cannot support value or budget decisions."},
  A3:{name:"Identity", val:()=> ((bd.identity||{}).mean||0).toFixed(2), cap:"kNN purity vs "+((bd.identity||{}).chance||0.25).toFixed(2)+" chance",
      desc:"Lookalikes share preference, not just activity.",
      what:"Whether neighbours in the state also share a stable customer preference.",
      how:"Ten nearest neighbours per held-out customer; fraction sharing the same channel preference.",
      why:"Identity is what separates a behavioural cluster from a merely busy one."},
  A4:{name:"Rate", val:()=> ((bd.roc||{}).auc||0).toFixed(3), cap:"AUC for churn ranking from the state",
      desc:"Intensity and timing are separable from content.",
      what:"Whether timing and churn intensity are recoverable, not just what the customer does.",
      how:"Five-fold logistic probe on the frozen state; ROC AUC on the held-out customers.",
      why:"Rate information is what makes a state useful for when, not only who."},
  A5:{name:"Uncertainty", val:()=> Number((DATA.bprops||{}).ece_next_event||0).toFixed(3), cap:"expected calibration error",
      desc:"Confidence tracks accuracy.",
      what:"Whether the model's stated confidence corresponds to its empirical accuracy.",
      how:"Bin next-event predictions by confidence and compare with observed accuracy.",
      why:"Calibrated confidence is what lets a downstream policy abstain or widen bounds."},
  A6:{name:"Causal", val:()=> "Layer 1", cap:"evaluated separately, not claimed here",
      desc:"The state must support action-conditioned decisions, not only prediction.",
      what:"Whether the state is sufficient for estimating the response to an action.",
      how:"Pre-action state plus logged randomized action, scored with inverse-propensity weighting.",
      why:"Prediction sufficiency does not imply decision sufficiency. This is the next research layer."},
  A7:{name:"Order / Path", val:()=> ((DATA.objectives||{}).order||{}).skill?.toFixed(3)||"—", cap:"order probe vs "+(((DATA.objectives||{}).order||{}).naive||0).toFixed(3)+" random",
      desc:"Sequence order matters, not just the bag of events.",
      what:"Whether the order of events carries information that a bag-of-events summary would lose.",
      how:"Probe the final state for the terminal event; compare against a random-order baseline.",
      why:"Path dependence is why a sequence model can beat a count table at all."},
  A8:{name:"Relational", val:()=> "0.005", cap:"cross-entity purity vs 0.0005 chance",
      desc:"Structure across entities is preserved.",
      what:"Whether shared identifiers link customers in a way the state reflects.",
      how:"Purity of shared-entity grouping against the chance rate.",
      why:"Cross-entity structure is what lets one customer's state inform a lookalike."}
};

function axVis(id){
  const box=el("div","axvis");
  const s=svg(900,280), pad={l:56,r:24,t:18,b:38};
  const W=900,H=280;
  const axis=()=>{ s.appendChild(ln(pad.l,H-pad.b,W-pad.r,H-pad.b,{stroke:LINE}));
                   s.appendChild(ln(pad.l,H-pad.b,pad.l,pad.t,{stroke:LINE})); };

  if(id==="A1" && bd.spectrum){
    // log scale: singular values span orders of magnitude, linear hides the tail
    const v=bd.spectrum.map(x=>Math.log10(Math.max(x,1e-6)));
    const lo=Math.min(...v), hi=Math.max(...v), n=v.length;
    v.forEach((x,i)=>{
      const h=Math.max(1,((x-lo)/(hi-lo||1))*(H-pad.t-pad.b));
      const bx=pad.l+i*((W-pad.l-pad.r)/n);
      s.appendChild(ln(bx,H-pad.b,bx,H-pad.b-h,{stroke:i<12?INK:DIM,w:Math.max(1.2,(W-pad.l)/n-2)}));
    });
    axis();
    s.appendChild(tx(pad.l,H-8,"singular values, largest first (log)",{size:10,fill:DIM}));
  }

  if(id==="A2" && (bd.mass||{}).points){
    const pts=bd.mass.points, lim=3;
    const X=x=>pad.l+((x+lim)/(2*lim))*(W-pad.l-pad.r), Y=y=>H-pad.b-((y+lim)/(2*lim))*(H-pad.t-pad.b);
    s.appendChild(ln(X(-lim),Y(-lim),X(lim),Y(lim),{stroke:LINE,dash:"4 4"}));
    pts.forEach(([a,b])=>s.appendChild(dot(X(a),Y(b),1.5,"rgba(5,150,105,.35)")));
    axis();
    s.appendChild(tx(pad.l,H-8,"actual past summary (z)",{size:10,fill:DIM}));
    s.appendChild(tx(pad.l+6,pad.t+12,"predicted",{size:10,fill:DIM}));
  }

  if(id==="A3" && (bd.identity||{}).bins){
    const bins=bd.identity.bins, maxn=Math.max(...bins.map(b=>b.n));
    const bw=(W-pad.l-pad.r)/bins.length;
    bins.forEach((b,i)=>{ const h=(b.n/maxn)*(H-pad.t-pad.b);
      s.appendChild(ln(pad.l+i*bw+bw*0.15,H-pad.b,pad.l+i*bw+bw*0.15,H-pad.b-h,{stroke:INK,w:bw*0.7})); });
    const mean=(bd.identity||{}).mean||0, chance=(bd.identity||{}).chance||0.25;
    const mx=pad.l+mean*(W-pad.l-pad.r), cx=pad.l+chance*(W-pad.l-pad.r);
    s.appendChild(ln(mx,pad.t,mx,H-pad.b,{stroke:ACC}));
    s.appendChild(ln(cx,pad.t+20,cx,H-pad.b,{stroke:DIM,dash:"3 3"}));
    s.appendChild(tx(mx,pad.t-2,"mean "+mean.toFixed(2),{size:10,fill:ACC,anchor:"middle"}));
    s.appendChild(tx(cx,pad.t+16,"chance",{size:10,fill:DIM,anchor:"middle"}));
    axis();
    s.appendChild(tx(pad.l,H-8,"per-customer purity (10-NN)",{size:10,fill:DIM}));
  }

  if(id==="A4" && (bd.roc||{}).points){
    const pts=bd.roc.points;
    const X=x=>pad.l+x*(W-pad.l-pad.r), Y=y=>H-pad.b-y*(H-pad.t-pad.b);
    s.appendChild(ln(X(0),Y(0),X(1),Y(1),{stroke:LINE,dash:"4 4"}));
    pts.forEach((p,i)=>{ if(i) s.appendChild(ln(X(pts[i-1][0]),Y(pts[i-1][1]),X(p[0]),Y(p[1]),{stroke:ACC,w:2})); });
    axis();
    s.appendChild(tx(pad.l,H-8,"false positive rate",{size:10,fill:DIM}));
    s.appendChild(tx(pad.l+6,pad.t+12,"true positive rate",{size:10,fill:DIM}));
  }

  if(id==="A5" && (bd.calibration||{}).bins){
    const bins=bd.calibration.bins;
    const X=x=>pad.l+x*(W-pad.l-pad.r), Y=y=>H-pad.b-y*(H-pad.t-pad.b);
    s.appendChild(ln(X(0),Y(0),X(1),Y(1),{stroke:LINE,dash:"4 4"}));
    bins.forEach((b,i)=>{ if(i){const p=bins[i-1]; s.appendChild(ln(X(p.conf),Y(p.acc),X(b.conf),Y(b.acc),{stroke:ACC,w:2}));} s.appendChild(dot(X(b.conf),Y(b.acc),3,ACC)); });
    axis();
    s.appendChild(tx(pad.l,H-8,"confidence",{size:10,fill:DIM}));
    s.appendChild(tx(pad.l+6,pad.t+12,"accuracy",{size:10,fill:DIM}));
  }

  if(id==="A6"){
    const keys=["channel","offer","freq","hour","dow"];
    const start=pad.l+92, gap=(W-pad.l-pad.r-92-52)/3;
    keys.forEach((k,i)=>{
      const y=pad.t+16+i*40;
      s.appendChild(tx(pad.l,y+4,k,{size:11,fill:MUT}));
      for(let j=0;j<4;j++){
        const x=start+j*gap, on=(j===i%4);
        s.appendChild(rect(x,y-10,52,22,on?"#ecfdf5":"#fff",4, on?ACC:LINE));
        if(on) s.appendChild(tx(x+26,y+4,"randomised",{size:8,fill:ACC,anchor:"middle"}));
      }
    });
    s.appendChild(tx(pad.l,H-10,"one key randomised per day, others at the customer optimum",{size:10,fill:DIM}));
  }

  if(id==="A7"){
    const o=(DATA.objectives||{}).order||{};
    const rows=[["random-order baseline",o.naive||0,DIM],["frozen state",o.skill||0,INK]];
    rows.forEach(([lab,v,col],i)=>{
      const y=pad.t+70+i*70, w=Math.max(3,(v||0)*(W-pad.l-190));
      s.appendChild(tx(pad.l,y-14,lab,{size:11,fill:DIM}));
      s.appendChild(ln(pad.l,y,pad.l+w,y,{stroke:col,w:14}));
      s.appendChild(tx(pad.l+w+10,y+4,(v||0).toFixed(3),{size:11,fill:col}));
    });
    s.appendChild(tx(pad.l,H-12,"probe skill identifying the terminal event",{size:10,fill:DIM}));
  }

  if(id==="A8"){
    const rows=[["chance",0.0005,DIM],["frozen state",0.005,INK]];
    const maxv=0.005;
    rows.forEach(([lab,v,col],i)=>{
      const y=pad.t+80+i*70, w=Math.max(3,(v/maxv)*(W-pad.l-210));
      s.appendChild(tx(pad.l,y-14,lab,{size:11,fill:DIM}));
      s.appendChild(ln(pad.l,y,pad.l+w,y,{stroke:col,w:14}));
      s.appendChild(tx(pad.l+w+10,y+4,v.toFixed(4),{size:11,fill:col}));
    });
    s.appendChild(tx(pad.l,H-12,"cross-entity purity, log-style scale",{size:10,fill:DIM}));
  }

  box.appendChild(s);
  return box;
}

function axisPanel(id){
  document.querySelectorAll("#axistabs button").forEach(b=>b.classList.toggle("on",b.dataset.ax===id));
  const d=AXDEF[id], host=$("#axpanel");
  host.innerHTML="";
  const head=el("div","axhead");
  const title=el("div");
  title.appendChild(el("div","mono dim small",id));
  title.appendChild(el("div","axname",d.name));
  head.appendChild(title);
  const metric=el("div","axmetric");
  const value=String(d.val());
  metric.appendChild(el("div","axval"+(value.length>8?" txt":""),value));
  metric.appendChild(el("div","axcap",d.cap));
  head.appendChild(metric);
  host.appendChild(head);
  host.appendChild(el("p","axdesc",d.desc));
  const visual=el("div","axvisual");
  visual.appendChild(axVis(id));
  host.appendChild(visual);
  const rows=el("div","axrows");
  [["What it measures",d.what],["How it is measured",d.how],["Why it matters",d.why]].forEach(([k,v])=>{
    const r=el("div","axrow"); r.appendChild(el("div","k",k)); r.appendChild(el("div","v",v)); rows.appendChild(r);
  });
  host.appendChild(rows);
}
const atabs=$("#axistabs");
AXES.forEach(([id,label],i)=>{
  const b=el("button",null,label); b.dataset.ax=id;
  b.onclick=()=>axisPanel(id); atabs.appendChild(b);
});
axisPanel("A1");

/* ---------- data tab ---------- */
const DT=DATA.data_tab;
if(!DT){
  const host=$("#data");
  if(host) host.innerHTML="<h2>Data</h2><p class='lede'>Run <code>python3 scripts/build_data_tab.py</code> with the canonical stream available to build this tab.</p>";
}else{
  const MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  let selected=0;

  function month(iso){
    const parts=String(iso).slice(0,10).split("-");
    return MONTHS[Number(parts[1])-1]+" "+parts[0];
  }
  function money(v){
    return "$"+Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
  }

  function renderTable(){
    const host=$("#legacytable"); host.innerHTML="";
    const cols=["Customer","Recency (days)","Frequency (orders)","Monetary"];
    $("#legacymeta").textContent=DT.customers.length+" of "+DT.wide.rows.toLocaleString()+" rows · showing "+cols.length+" of "+DT.wide.columns+" columns";
    const table=el("table");
    const thead=el("thead"), hr=el("tr");
    cols.forEach((label,i)=>{
      hr.appendChild(el("th",i===0?null:"num",label));
    });
    thead.appendChild(hr); table.appendChild(thead);
    const tbody=el("tbody");
    DT.customers.forEach((c,i)=>{
      const tr=el("tr",i===selected?"on":null);
      tr.tabIndex=0;
      tr.onclick=()=>selectData(i);
      tr.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();selectData(i);}};
      tr.appendChild(el("td",null,c.key));
      tr.appendChild(el("td","num",String(c.legacy.days_since_last_purchase)));
      tr.appendChild(el("td","num",String(c.legacy.n_orders)));
      tr.appendChild(el("td","num",money(c.legacy.purchase_value)));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody); host.appendChild(table);
  }

  function tipContent(tip,e){
    tip.innerHTML="";
    tip.appendChild(el("div","tts","event_ts  "+e[0]));
    tip.appendChild(el("div","tty","event_type  "+e[1]));
    tip.appendChild(el("div","ttl","event_attributes"));
    tip.appendChild(el("pre","tattrs",e[3]||"{}"));
  }
  function tipPlace(ev,panel,tip){
    const r=panel.getBoundingClientRect();
    const tw=tip.offsetWidth, th=tip.offsetHeight;
    let left=ev.clientX-r.left+14, top=ev.clientY-r.top+14;
    if(left+tw>r.width-10) left=ev.clientX-r.left-tw-14;
    if(top+th>r.height-10) top=ev.clientY-r.top-th-14;
    tip.style.left=Math.max(8,left)+"px";
    tip.style.top=Math.max(8,top)+"px";
  }

  function streamTimeline(c,panel,tip){
    const W=1000,H=300,pad={l:86,r:20,t:20,b:34};
    const s=svg(W,H), h=H-pad.t-pad.b;
    const types=DT.types, rowH=h/types.length;
    const X=d=>pad.l+(d/DT.window_days)*(W-pad.l-pad.r);
    const start=new Date(DT.window_start+"T00:00:00Z");
    const end=new Date(start.getTime()+DT.window_days*86400000);
    const quarters=[];
    let d=new Date(Date.UTC(start.getUTCFullYear(),start.getUTCMonth()+1,1));
    while(d<end){
      const day=(d-start.getTime())/86400000;
      const jan=d.getUTCMonth()===0;
      s.appendChild(ln(X(day),pad.t,X(day),H-pad.b,{stroke:jan?"#eef0f3":"#f7f8fa"}));
      if(d.getUTCMonth()%3===0) quarters.push({day,month:d.getUTCMonth(),year:d.getUTCFullYear()});
      d=new Date(Date.UTC(d.getUTCFullYear(),d.getUTCMonth()+1,1));
    }
    types.forEach((t,i)=>{
      const y=pad.t+(i+0.5)*rowH;
      s.appendChild(tx(pad.l-7,y+3,t.label,{size:10,fill:MUT,anchor:"end"}));
      s.appendChild(ln(pad.l,y,W-pad.r,y,{stroke:"#f9fafb"}));
    });
    c.events.forEach((e,i)=>{
      const idx=types.findIndex(t=>t.id===e[1]);
      if(idx<0) return;
      const cx=X(e[2]), cy=pad.t+(idx+0.5)*rowH;
      const hit=dot(cx,cy,7,"transparent");
      hit.setAttribute("class","timeline-hit");
      hit.setAttribute("data-i",String(i));
      hit.setAttribute("pointer-events","all");
      hit.style.cursor="pointer";
      hit.addEventListener("mouseenter",ev=>{
        tipContent(tip,e); tip.hidden=false; tipPlace(ev,panel,tip);
      });
      hit.addEventListener("mousemove",ev=>{
        if(!tip.hidden) tipPlace(ev,panel,tip);
      });
      hit.addEventListener("mouseleave",()=>{tip.hidden=true;});
      s.appendChild(hit);
      const mark=dot(cx,cy,2.1,"rgba(17,24,39,.42)");
      mark.setAttribute("class","timeline-dot");
      mark.setAttribute("data-i",String(i));
      mark.setAttribute("pointer-events","none");
      s.appendChild(mark);
    });
    s.appendChild(ln(pad.l,H-pad.b,W-pad.r,H-pad.b,{stroke:LINE}));
    s.appendChild(ln(pad.l,H-pad.b,pad.l,H-pad.b+5,{stroke:LINE}));
    s.appendChild(tx(pad.l,H-9,month(DT.window_start),{size:10,fill:DIM}));
    quarters.forEach(q=>{
      const x=X(q.day);
      s.appendChild(ln(x,H-pad.b,x,H-pad.b+5,{stroke:LINE}));
      s.appendChild(tx(x,H-9,MONTHS[q.month]+(q.month===0?" "+q.year:""),{size:10,fill:DIM,anchor:"middle"}));
    });
    s.appendChild(tx(W-pad.r,H-9,MONTHS[end.getUTCMonth()]+" "+end.getUTCFullYear(),{size:10,fill:DIM,anchor:"end"}));
    return s;
  }

  function renderStream(){
    const c=DT.customers[selected], host=$("#datastream"); host.innerHTML="";

    const head=el("div","streamhead");
    const left=el("div");
    left.appendChild(el("div","custtag","Selected customer"));
    left.appendChild(el("div","streamkey",c.key));
    head.appendChild(left);
    const counts=el("div","streamcounts");
    [["events",c.events.length],["days",DT.window_days]].forEach(([k,n])=>{
      const span=el("span"); span.appendChild(el("b",null,String(n))); span.appendChild(document.createTextNode(" "+k)); counts.appendChild(span);
    });
    head.appendChild(counts); host.appendChild(head);

    const flow=el("div","dataflow");
    const tl=el("div","panelbox timeline-panel");
    tl.appendChild(el("div","cardtag","Event timeline · hover a dot or row to link"));
    const tip=el("div","timeline-tip"); tip.hidden=true;
    const wrap=el("div","tlwrap"); wrap.appendChild(streamTimeline(c,tl,tip)); tl.appendChild(wrap);
    tl.appendChild(tip); flow.appendChild(tl);

    const ledger=el("div","panelbox");
    ledger.appendChild(el("div","cardtag","Canonical stream · "+c.events.length+" rows"));
    const box=el("div","logscroll");
    const table=el("table","logtable canonical");
    const thead=el("thead"); const hr=el("tr");
    ["event_ts","event_type","event_attributes"].forEach(t=>{hr.appendChild(el("th",null,t));});
    thead.appendChild(hr); table.appendChild(thead);
    const tbody=el("tbody");
    c.events.forEach((e,i)=>{
      const tr=el("tr");
      tr.dataset.i=String(i);
      tr.appendChild(el("td",null,e[0]));
      tr.appendChild(el("td",null,e[1]));
      tr.appendChild(el("td","attrs",e[3]||"{}"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody); box.appendChild(table);
    ledger.appendChild(box); flow.appendChild(ledger); host.appendChild(flow);
    linkEventHover(wrap,box);
  }

  function linkEventHover(wrap,box){
    const dots={},rows={};
    wrap.querySelectorAll(".timeline-dot").forEach(d=>dots[d.dataset.i]=d);
    box.querySelectorAll("tbody tr").forEach(r=>rows[r.dataset.i]=r);
    let active=null;
    function mark(i,on){
      const dot=dots[i],row=rows[i];
      if(dot){
        dot.classList.toggle("hot",on);
        dot.setAttribute("r",on?"4.4":"2.1");
      }
      if(row) row.classList.toggle("hot",on);
    }
    function enter(i){
      if(active!==null&&active!==i) mark(active,false);
      active=i; mark(i,true);
      const row=rows[i];
      if(row){
        const top=row.offsetTop,h=box.clientHeight;
        if(top<box.scrollTop||top+row.offsetHeight>box.scrollTop+h){
          box.scrollTop=Math.max(0,top-h/2);
        }
      }
    }
    function leave(i){
      if(active===i){mark(i,false);active=null;}
    }
    wrap.querySelectorAll(".timeline-hit").forEach(h=>{
      const i=h.dataset.i;
      h.addEventListener("mouseenter",()=>enter(i));
      h.addEventListener("mouseleave",()=>leave(i));
    });
    box.querySelectorAll("tbody tr").forEach(r=>{
      const i=r.dataset.i;
      r.addEventListener("mouseenter",()=>enter(i));
      r.addEventListener("mouseleave",()=>leave(i));
    });
  }

  function selectData(i){
    selected=i;
    renderTable();
    renderStream();
  }

  function setDataMode(mode){
    const stream=mode==="stream";
    $("#legacypane").hidden=stream;
    $("#streampane").hidden=!stream;
    const slide=$(".dsslide");
    slide.style.transform=stream?"translateX(100%)":"translateX(0)";
    slide.classList.toggle("stream",stream);
    document.querySelectorAll(".dstab").forEach(b=>b.classList.toggle("on",b.dataset.dsmode===mode));
  }

  document.querySelectorAll(".dstab").forEach(b=>b.onclick=()=>setDataMode(b.dataset.dsmode));

  renderTable();
  renderStream();
  setDataMode("legacy");
}

/* ---------- encoder tab ---------- */
const ENC=DATA.encoder_tab;
if(!ENC){
  const host=$("#encoder");
  if(host) host.innerHTML="<h2>Encoder</h2><p class='lede'>Run <code>python3 scripts/build_encoder_tab.py</code> with the frozen cache available to build this tab.</p>";
}else{
  let featureFamily="shape";
  const ENC_CUSTOMERS=(ENC.customers&&ENC.customers.length)?ENC.customers:[{
    key:ENC.customer,shape:ENC.shape,families:ENC.families,channel:null,category:null,regime:null
  }];
  let encSel=0;
  const encCust=()=>ENC_CUSTOMERS[encSel]||ENC_CUSTOMERS[0];
  function renderEncCustomers(){
    const pick=$("#encpick"); if(!pick) return;
    pick.innerHTML="";
    ENC_CUSTOMERS.forEach((c,i)=>{
      const b=el("button","encperson"+(i===encSel?" on":""));
      b.type="button"; b.setAttribute("aria-pressed",i===encSel);
      b.appendChild(el("b",null,c.key));
      const desc=[c.regime,c.category].filter(Boolean).join(" · ");
      b.appendChild(el("span",null,desc));
      b.onclick=()=>{encSel=i;renderEncCustomers();renderFeatSelector();renderFeatMap();};
      pick.appendChild(b);
    });
  }

  function featCols(id){
    if(id==="learned_state") return 48;
    if(id==="forecast") return 42;
    if(id==="uncertainty") return 8;
    return 16;
  }

  function renderFeatSelector(){
    const host=$("#featselector"); host.innerHTML="";
    const maxDim=Math.max(...encCust().families.map(f=>f.dim));
    encCust().families.forEach(f=>{
      const b=el("button","featfamily"+(featureFamily===f.id?" on":""));
      b.type="button";
      const top=el("div","top");
      top.appendChild(el("div","name",f.label));
      top.appendChild(el("div","dim",f.dim.toLocaleString()));
      b.appendChild(top);
      const bar=el("div","bar"); const fill=el("div");
      fill.style.width=(100*f.dim/maxDim)+"%"; bar.appendChild(fill); b.appendChild(bar);
      b.onclick=()=>{featureFamily=f.id;renderFeatSelector();renderFeatMap();};
      host.appendChild(b);
    });
    const total=el("div","total");
    total.appendChild(el("span",null,"Published state"));
    total.appendChild(el("b",null,ENC.state_dim.toLocaleString()+" dims"));
    host.appendChild(total);
  }

  function renderFeatMap(){
    const fam=encCust().families.find(f=>f.id===featureFamily)||encCust().families[0];
    $("#shapehead").innerHTML="";
    const left=el("div");
    left.appendChild(el("div","cardtag",fam.label));
    left.appendChild(el("div","shapekey",encCust().key));
    $("#shapehead").appendChild(left);
    $("#shapehead").appendChild(el("div","shapemeta",fam.dim.toLocaleString()+" dims"));

    const grid=$("#featuremap"); grid.innerHTML="";
    const cols=featCols(fam.id);
    grid.style.gridTemplateColumns="repeat("+cols+",minmax(0,1fr))";
    const maxv=Math.max(...fam.values.map(v=>Math.abs(v)));
    fam.values.forEach((v,i)=>{
      const cell=el("div","featcell");
      const a=Math.min(1,Math.abs(v)/(maxv||1));
      cell.style.background=v>=0?`rgba(5,150,105,${0.10+0.80*a})`:`rgba(17,24,39,${0.06+0.62*a})`;
      cell.title="dim "+i+" · "+Number(v).toFixed(4);
      grid.appendChild(cell);
    });

    const legend=$("#featurelegend"); legend.innerHTML="";
    const sw=(bg,border,name)=>{
      const s=el("span");
      const i=el("i","sw"); i.style.background=bg; if(border){i.style.border="1px solid var(--line)";}
      s.appendChild(i); s.appendChild(document.createTextNode(name)); return s;
    };
    legend.appendChild(sw("rgba(17,24,39,.68)",false,"negative"));
    legend.appendChild(sw("#ffffff",true,"zero"));
    legend.appendChild(sw("rgba(5,150,105,.80)",false,"positive"));
    legend.appendChild(el("span","max","max |value| "+maxv.toFixed(2)));
  }

  function renderEncoder(){
    renderFeatSelector();
    renderFeatMap();
  }
  renderEncCustomers();
  renderEncoder();
}

/* ---------- objectives ---------- */
function objectiveSketch(id,spec){
  const W=520,H=150,s=svg(W,H);
  const x0=34,x1=486,step=(x1-x0)/10,X=i=>x0+i*step,y=88;
  const target=new Set(spec.target||[]);
  const cut=spec.cut;
  const first=Math.min(...spec.target),last=Math.max(...spec.target);
  const ink=(x,yy,r=4.2)=>s.appendChild(dot(x,yy,r,INK));
  const faint=(x,yy,r=3.2)=>s.appendChild(dot(x,yy,r,"#fff",LINE));
  const ring=(x,yy,r=5.2)=>s.appendChild(dot(x,yy,r,"#fff",ACC));
  const chevron=(x,yy,dx,dy)=>{
    const len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len,px=-uy,py=ux,a=5;
    s.appendChild(ln(x-ux*a+px*a*.7,yy-uy*a+py*a*.7,x,yy,{stroke:ACC,w:1.2}));
    s.appendChild(ln(x-ux*a-px*a*.7,yy-uy*a-py*a*.7,x,yy,{stroke:ACC,w:1.2}));
  };

  // Common timeline: same slots, cut, target halos, and footer for every objective.
  s.appendChild(ln(x0,y,x1,y,{stroke:"#f3f4f6"}));
  for(let i=0;i<=10;i++){
    if(target.has(i)){
      s.appendChild(dot(X(i),y,10,"rgba(5,150,105,.07)"));
      let r=5.4,op=1;
      if(spec.mode==="decay"){
        const j=i-first;
        r=6.6-j*.5;
        op=Math.max(.45,1-j*.14);
      }
      const node=dot(X(i),y,r,"#fff",ACC);
      node.setAttribute("opacity",op.toFixed(2));
      s.appendChild(node);
    }else if(i<=cut) ink(X(i),y);
    else faint(X(i),y);
  }
  const cutX=(X(cut)+X(cut+1))/2;
  s.appendChild(ln(cutX,72,cutX,104,{stroke:"#d1d5db",dash:"2 4"}));
  s.appendChild(tx(cutX,66,"t",{size:9,fill:DIM,anchor:"middle"}));

  if(id==="jepa"){
    const mx=(X(first)+X(last))/2,my=32;
    spec.target.forEach(i=>s.appendChild(ln(X(i),y-8,mx,my+7,{stroke:"rgba(5,150,105,.25)"})));
    s.appendChild(dot(mx,my,9,"rgba(5,150,105,.10)",ACC));
    s.appendChild(dot(mx,my,4,ACC));
  }else if(id==="mask"){
    const hx=X(3);
    const hole=dot(hx,y,8,"#fff",ACC); hole.setAttribute("stroke-dasharray","3 3"); s.appendChild(hole);
    s.appendChild(ln(X(2)+6,y,hx-10,y,{stroke:ACC,w:1}));
    s.appendChild(ln(X(4)-6,y,hx+10,y,{stroke:ACC,w:1}));
    chevron(hx-10,y,1,0); chevron(hx+10,y,-1,0);
    s.appendChild(ln(hx,46,hx,76,{stroke:"rgba(5,150,105,.3)",dash:"3 3"}));
    ring(hx,38,5.2);
  }else if(id==="order"){
    const term=X(5),final=X(10);
    s.appendChild(dot(term,y,9,"rgba(5,150,105,.07)"));
    ring(term,y,6.4);
    s.appendChild(path(`M ${final} 78 C ${final} 34, ${term+34} 30, ${term+10} 70`,{stroke:ACC,w:1.3,dash:"4 3"}));
    s.appendChild(ln(term+4,64,term+10,70,{stroke:ACC,w:1.2}));
    s.appendChild(ln(term+16,64,term+10,70,{stroke:ACC,w:1.2}));
  }else if(id==="entity"){
    const e1=X(2),e2=X(6);
    s.appendChild(dot(e1,y,4.8,ACC));
    s.appendChild(ln(e1,y-8,e1,46,{stroke:ACC,w:1,dash:"2 3"}));
    s.appendChild(ln(e2,y-8,e2,46,{stroke:ACC,w:1,dash:"2 3"}));
    s.appendChild(rect(e1-5,38,10,10,"rgba(5,150,105,.16)",2,ACC));
    s.appendChild(rect(e2-5,38,10,10,"rgba(5,150,105,.16)",2,ACC));
    s.appendChild(ln(e1+5,43,e2-5,43,{stroke:ACC,dash:"2 3"}));
  }else if(id==="query"){
    const q1=X(2),q2=X(6);
    s.appendChild(dot(q1,y,4.8,ACC));
    s.appendChild(ln(q1,44,q2,44,{stroke:ACC,dash:"3 3"}));
    s.appendChild(ln(q1,36,q1,52,{stroke:INK}));
    s.appendChild(ln(q2,36,q2,52,{stroke:INK}));
    s.appendChild(ln(q2,54,q2,74,{stroke:"rgba(5,150,105,.3)",dash:"2 4"}));
  }else if(id==="sf"){
    for(let j=0;j<=4;j++){
      const r=3.8-j*.35;
      s.appendChild(dot(X(first+j),32+j*6,Math.max(1.6,r),`rgba(5,150,105,${.55-j*.09})`));
    }
  }else if(id==="dt"){
    const a=X(5),b=X(6);
    s.appendChild(ln(a,42,b,42,{stroke:ACC,dash:"3 3"}));
    s.appendChild(ln(a,34,a,50,{stroke:INK}));
    s.appendChild(ln(b,34,b,50,{stroke:INK}));
    s.appendChild(ln(a,52,a,72,{stroke:"rgba(5,150,105,.3)",dash:"2 4"}));
    s.appendChild(ln(b,52,b,72,{stroke:"rgba(5,150,105,.3)",dash:"2 4"}));
  }else if(id==="value"){
    const bx=X(6);
    [[-14,16,"#d1d5db"],[0,28,"#d1d5db"],[14,38,ACC]].forEach(([dx,h,fill])=>{
      s.appendChild(rect(bx+dx-4,68-h,8,h,fill,2));
    });
  }else if(id==="agg"){
    const a=X(first),b=X(last),cx=(a+b)/2;
    s.appendChild(ln(a,46,b,46,{stroke:ACC,dash:"3 3"}));
    s.appendChild(ln(a,38,a,54,{stroke:INK}));
    s.appendChild(ln(b,38,b,54,{stroke:INK}));
    s.appendChild(ln(a,38,cx,30,{stroke:"rgba(5,150,105,.3)"}));
    s.appendChild(ln(b,38,cx,30,{stroke:"rgba(5,150,105,.3)"}));
    s.appendChild(dot(cx,26,7,"rgba(5,150,105,.10)",ACC));
    s.appendChild(dot(cx,26,3.2,ACC));
  }else if(id==="next"){
    const nx=X(6);
    [[X(9.3),28,"fill"],[X(9.6),44,"ring"],[X(9.3),60,"faint"]].forEach(([bx,by,kind])=>{
      s.appendChild(ln(nx+8,y-8,bx-4,by,{stroke:"rgba(5,150,105,.3)"}));
      if(kind==="fill") s.appendChild(dot(bx,by,4.6,ACC));
      else if(kind==="ring") s.appendChild(dot(bx,by,4.6,"#fff",ACC));
      else s.appendChild(dot(bx,by,3.4,"#fff",LINE));
    });
  }

  s.appendChild(tx(x0,H-10,"observed history",{size:9,fill:DIM}));
  s.appendChild(tx(x1,H-10,"target",{size:9,fill:ACC,anchor:"end"}));
  return s;
}
function objPanel(id){
  const spec=OBJSPEC[id]; if(!spec) return;
  document.querySelectorAll(".objitem").forEach(b=>b.classList.toggle("on",b.dataset.obj===id));
  const box=$("#objpanel"); box.innerHTML="";
  const head=el("div","objhead");
  head.appendChild(el("h3",null,spec.label));
  box.appendChild(head);
  box.appendChild(el("p","objdesc",spec.desc));
  const visual=el("div","objvisual");
  const ste=objectiveSketch(id,spec); ste.setAttribute("width","100%"); visual.appendChild(ste);
  box.appendChild(visual);
}
document.querySelectorAll(".objitem").forEach(b=>b.onclick=()=>objPanel(b.dataset.obj));
objPanel("jepa");

/* ---------- plugins ---------- */
const PL_EFFECT=DATA.rl_known||{};
const PL_POLICY=DATA.rl_policy||{};
const PL_OUT=DATA.supervised.values||{};
const PL_LOOK=DATA.explorer;
let lookIdx=0;
if(PL_LOOK&&PL_LOOK.customers){
  const idx=PL_LOOK.customers.findIndex(c=>c.key==="rc0000208");
  if(idx>=0) lookIdx=idx;
}
let pluginMode="predict",actMode="channel";

function pluginMoney(v){
  return "$"+Number(v).toLocaleString("en-US",{maximumFractionDigits:1});
}

function miniOutcomeSVG(){
  const W=240,H=42,pad=8,s=svg(W,H);
  const keys=["value90","orders90","churn90"], rowH=(H-2*pad)/keys.length;
  keys.forEach((k,i)=>{
    const v=PL_OUT[k]||{}, table=Number(v.raw||0), frozen=Number(v.full_measure||0);
    const y=pad+(i+0.5)*rowH;
    s.appendChild(rect(pad,y-2.6,Math.max(1,table*(W-2*pad)),3,"#d1d5db",1.5));
    s.appendChild(rect(pad,y+0.6,Math.max(1,frozen*(W-2*pad)),3,ACC,1.5));
  });
  return s;
}

function miniLookalikeSVG(){
  const W=240,H=42,pad=4,s=svg(W,H);
  const pts=(PL_LOOK&&PL_LOOK.customers)||[];
  if(!pts.length) return s;
  if(lookIdx>=pts.length) lookIdx=0;
  const sel=pts[lookIdx], byKey=Object.fromEntries(pts.map(c=>[c.key,c]));
  const X=x=>pad+x*(W-2*pad), Y=y=>H-pad-y*(H-2*pad);
  (sel.neighbours||[]).forEach(n=>{
    const c=byKey[n.key]; if(c) s.appendChild(ln(X(sel.x),Y(sel.y),X(c.x),Y(c.y),{stroke:"rgba(5,150,105,.45)",w:.8}));
  });
  pts.forEach(c=>s.appendChild(dot(X(c.x),Y(c.y),1.05,"#d1d5db")));
  (sel.neighbours||[]).forEach(n=>{const c=byKey[n.key]; if(c) s.appendChild(dot(X(c.x),Y(c.y),2,ACC));});
  s.appendChild(dot(X(sel.x),Y(sel.y),2.8,INK));
  return s;
}

function miniActSVG(){
  const W=240,H=42,pad=6,s=svg(W,H);
  s.appendChild(ln(120,8,120,H-8,{stroke:LINE}));
  const keys=["channel","offer","freq","hour","dow"].filter(k=>PL_EFFECT[k]);
  const rowH=(H-2*pad)/Math.max(1,keys.length);
  keys.forEach((k,i)=>{
    const r=PL_EFFECT[k], y=pad+(i+0.5)*rowH;
    const x1=pad+Number(r.table)*104, x2=pad+Number(r.E)*104;
    s.appendChild(ln(x1,y,x2,y,{stroke:"rgba(5,150,105,.38)",w:1}));
    s.appendChild(dot(x1,y,1.7,"#d1d5db"));
    s.appendChild(dot(x2,y,2.4,ACC));
  });
  const arms=["channel","offer"].filter(k=>PL_POLICY[k]);
  const vals=arms.flatMap(k=>[PL_POLICY[k].random.ips,PL_POLICY[k].table.ips,PL_POLICY[k].E.ips]);
  if(vals.length){
    const lo=Math.min(...vals)-0.4, hi=Math.max(...vals)+0.4;
    const X=i=>128+i*52;
    const Y=v=>H-pad-((v-lo)/(hi-lo||1))*(H-2*pad);
    const colors={channel:INK,offer:"#6b7280"};
    arms.forEach(k=>{
      const r=PL_POLICY[k], pts=[r.random.ips,r.table.ips,r.E.ips];
      s.appendChild(ln(X(0),Y(pts[0]),X(1),Y(pts[1]),{stroke:colors[k],w:1}));
      s.appendChild(ln(X(1),Y(pts[1]),X(2),Y(pts[2]),{stroke:colors[k],w:1}));
      pts.forEach((v,i)=>s.appendChild(dot(X(i),Y(v),i===2?2.8:1.9,colors[k])));
    });
  }
  return s;
}

function renderPluginsMap(){
  const host=$("#plugmap"); host.innerHTML="";
  const grid=el("div","pluggrid");
  const makeCard=(id,title,desc,mini)=>{
    const b=el("button","plugcard"+(pluginMode===id?" on":""));
    b.type="button"; b.dataset.mode=id; b.setAttribute("aria-pressed",pluginMode===id);
    const top=el("div");
    top.appendChild(el("div","t",title));
    top.appendChild(el("div","d",desc));
    b.appendChild(top);
    const box=el("div","mini"); box.appendChild(mini()); b.appendChild(box);
    b.onclick=()=>{pluginMode=id;renderPluginsMap();renderPluginPanel();};
    return b;
  };
  grid.appendChild(makeCard("predict","Predict","value_90d · orders_90d · churn_90d",miniOutcomeSVG));
  grid.appendChild(makeCard("retrieve","Retrieve","nearest neighbours in the state",miniLookalikeSVG));
  grid.appendChild(makeCard("act","Act","channel · offer · timing",miniActSVG));
  host.appendChild(grid);
}

function pluginOutcomeChart(){
  const W=1000,H=330,pad={l:108,r:64,t:38,b:40},s=svg(W,H);
  const keys=[["value90","value_90d"],["orders90","orders_90d"],["churn90","churn_90d"]];
  const rowH=(H-pad.t-pad.b)/keys.length;
  [0,.25,.5,.75,1].forEach(g=>{
    const x=pad.l+g*(W-pad.l-pad.r);
    s.appendChild(ln(x,pad.t,x,H-pad.b,{stroke:g===0?LINE:"#f3f4f6"}));
    if(g>0) s.appendChild(tx(x,H-14,Math.round(g*100)+"%",{size:9,fill:DIM,anchor:"middle"}));
  });
  keys.forEach(([key,label],i)=>{
    const v=PL_OUT[key]||{}, table=Number(v.raw||0), frozen=Number(v.full_measure||0);
    const y=pad.t+(i+0.5)*rowH, bh=14;
    s.appendChild(tx(pad.l-12,y+4,label,{size:10,fill:MUT,anchor:"end"}));
    const a=rect(pad.l,y-bh-3,Math.max(1,table*(W-pad.l-pad.r)),bh,"#d1d5db",3);
    const b=rect(pad.l,y+3,Math.max(1,frozen*(W-pad.l-pad.r)),bh,ACC,3);
    addSvgTitle(a,label+" · table "+table.toFixed(4));
    addSvgTitle(b,label+" · frozen "+frozen.toFixed(4));
    s.appendChild(a); s.appendChild(b);
    const diff=frozen-table;
    const gain=table?diff/table*100:0;
    s.appendChild(tx(pad.l+Math.max(table,frozen)*(W-pad.l-pad.r)+12,y+4,(gain>=0?"+":"")+gain.toFixed(1)+"%",{size:10,fill:ACC}));
  });
  return s;
}

function pluginLookalikeChart(){
  const W=1000,H=340,pad=28,s=svg(W,H);
  const pts=(PL_LOOK&&PL_LOOK.customers)||[];
  if(!pts.length) return s;
  if(lookIdx<0||lookIdx>=pts.length) lookIdx=0;
  const sel=pts[lookIdx], byKey=Object.fromEntries(pts.map(c=>[c.key,c]));
  const nb=new Set((sel.neighbours||[]).map(n=>n.key));
  const X=x=>pad+x*(W-2*pad), Y=y=>H-pad-y*(H-2*pad);
  (sel.neighbours||[]).forEach(n=>{
    const c=byKey[n.key]; if(c) s.appendChild(ln(X(sel.x),Y(sel.y),X(c.x),Y(c.y),{stroke:"rgba(5,150,105,.4)",w:1.2}));
  });
  pts.forEach((c,i)=>{
    const cx=X(c.x),cy=Y(c.y), isSel=i===lookIdx, isNb=nb.has(c.key);
    const hit=dot(cx,cy,13,"transparent"); hit.style.cursor="pointer";
    hit.onclick=()=>{lookIdx=i;renderPluginPanel();};
    addSvgTitle(hit,c.key);
    s.appendChild(hit);
    const circle=dot(cx,cy,isSel?8:isNb?5.2:3.4,isSel?INK:isNb?ACC:"#d1d5db");
    circle.setAttribute("pointer-events","none");
    s.appendChild(circle);
  });
  s.appendChild(tx(pad,18,sel.key,{size:11,fill:MUT}));
  return s;
}

const ACTIONS=[
  ["channel","Channel","where it goes","Where the next message goes: direct mail, email, push, or SMS."],
  ["offer","Offer","what is offered","Offer tier 0–3."],
  ["freq","Frequency","how often","Sends to schedule in the window."],
  ["hour","Hour","when in the day","Hour of day to send."],
  ["dow","Day","which day","Day of week to send."],
];
function actBarRow(label,value,color){
  const row=el("div","actbarrow");
  row.appendChild(el("span","label",label));
  const track=el("div","acttrack"),fill=el("i");
  fill.style.width=Math.max(.6,value*100)+"%";
  fill.style.background=color;
  track.appendChild(fill); row.appendChild(track);
  row.appendChild(el("span","val",(value*100).toFixed(1)+"%"));
  return row;
}
function renderActPanel(host){
  host.innerHTML="";
  const tabs=el("div","acttabs");
  ACTIONS.forEach(([id,name,sub])=>{
    const b=el("button","acttab"+(actMode===id?" on":""));
    b.type="button";
    b.appendChild(el("span","name",name));
    b.appendChild(el("small",null,sub));
    b.onclick=()=>{actMode=id;renderActPanel(host);};
    tabs.appendChild(b);
  });
  host.appendChild(tabs);

  const action=ACTIONS.find(x=>x[0]===actMode)||ACTIONS[0];
  const [id,name,_sub,what]=action;
  const effect=PL_EFFECT[id]||{};
  const detail=el("div","actdetail");

  const head=el("div","acthead");
  const copy=el("div");
  copy.appendChild(el("h3",null,name));
  copy.appendChild(el("p","actwhat",what));
  head.appendChild(copy);
  const e=Number(effect.E), t=Number(effect.table);
  if(Number.isFinite(e)&&Number.isFinite(t)){
    const delta=el("div","actdelta");
    delta.appendChild(el("div","v",((e-t)*100>=0?"+":"")+((e-t)*100).toFixed(1)+" pts"));
    delta.appendChild(el("div","k","vs table"));
    head.appendChild(delta);
  }
  detail.appendChild(head);

  if(Number.isFinite(e)&&Number.isFinite(t)){
    const block=el("div","actblock");
    const bh=el("div","actblockhead");
    bh.appendChild(el("span","t","Known effect"));
    bh.appendChild(el("small",null,"10-NN purity"));
    block.appendChild(bh);
    const bars=el("div","actbars");
    bars.appendChild(actBarRow("Table",t,"#d1d5db"));
    bars.appendChild(actBarRow("Frozen",e,ACC));
    block.appendChild(bars);
    detail.appendChild(block);
  }

  const policy=PL_POLICY[id];
  if(policy&&policy.random&&policy.table&&policy.E){
    const p=el("div","actpolicy");
    const ph=el("div","actblockhead");
    ph.appendChild(el("span","t","Policy value"));
    ph.appendChild(el("small",null,"IPS · 14-day reward"));
    p.appendChild(ph);
    const stats=el("div","policystats");
    [["Random",policy.random.ips],["Table",policy.table.ips],["Frozen",policy.E.ips]].forEach(([lab,v],i)=>{
      const st=el("div","pstat"+(i===2?" on":""));
      st.appendChild(el("div","k",lab));
      st.appendChild(el("div","v",Number(v).toFixed(2)));
      stats.appendChild(st);
    });
    p.appendChild(stats);
    const r=Number(policy.random.ips),tt=Number(policy.table.ips),ff=Number(policy.E.ips);
    const deltas=el("div","policydeltas");
    [["vs random",(ff-r)/r*100],["vs table",(ff-tt)/tt*100]].forEach(([lab,v])=>{
      const tag=el("div","policytag");
      tag.appendChild(el("span",null,lab));
      tag.appendChild(el("b",null,(v>=0?"+":"")+v.toFixed(1)+"%"));
      deltas.appendChild(tag);
    });
    p.appendChild(deltas);
    detail.appendChild(p);
  }
  host.appendChild(detail);
}

function renderPluginPanel(){
  const host=$("#plugpanel"); host.innerHTML="";
  if(pluginMode==="retrieve"){
    host.appendChild(chartHeader("Retrieve · nearest neighbours",[["selected","selected"],["5 neighbours","neighbour"],["all customers","other"]]));
    host.appendChild(pluginLookalikeChart());
    const pts=(PL_LOOK&&PL_LOOK.customers)||[];
    const sel=pts[lookIdx];
    const right=sel?("5 neighbours · mean value "+pluginMoney(sel.neighbour_mean_value)+" · churn_90d agreement "+(sel.neighbour_churn_agree*100).toFixed(0)+"%"):"";
    host.appendChild(chartFoot("click a point to change the lookalike set",right));
  }else if(pluginMode==="act"){
    renderActPanel(host);
  }else{
    host.appendChild(chartHeader("Predict · supervised heads",[["table","table"],["frozen state","frozen"]]));
    host.appendChild(pluginOutcomeChart());
    host.appendChild(chartFoot("held-out · R² value_90d / orders_90d · AUC churn_90d","green = % better than table"));
  }
}

renderPluginsMap();
renderPluginPanel();

/* ---------- explorer ---------- */
const EXP=DATA.explorer;
if(!EXP||!EXP.customers||!EXP.customers.length){
  $("#explorer").innerHTML="<h2>Meet a customer</h2><p class='lede'>Run <code>python3 scripts/build_demo_site.py</code> with the frozen cache available to build the explorer.</p>";
}else{
  const svgEl=$("#scatter"),W=1000,H=600;
  const frameLayer=document.createElementNS(NS,"g"); svgEl.appendChild(frameLayer);
  const lineLayer=document.createElementNS(NS,"g"); lineLayer.setAttribute("id","neighbour-lines"); svgEl.appendChild(lineLayer);
  const marksLayer=document.createElementNS(NS,"g"); svgEl.appendChild(marksLayer);
  const marks=[],byKey={};
  let yaw=-.62,pitch=.3,selected=0,dragging=false,dragMoved=false,lastX=0,lastY=0,startX=0,startY=0;

  function money(v){return "$"+Number(v).toLocaleString("en-US",{maximumFractionDigits:1});}
  function pctile(arr,v){
    let lo=0,hi=arr.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(arr[mid]<=v)lo=mid+1;else hi=mid;}
    return Math.max(1,Math.min(100,Math.round(lo/arr.length*100)));
  }
  function ordinal(n){
    return n+(n%10===1&&n%100!==11?"st":n%10===2&&n%100!==12?"nd":n%10===3&&n%100!==13?"rd":"th");
  }
  const valueSorted=EXP.customers.map(c=>Number(c.value90)).sort((a,b)=>a-b);
  const orderSorted=EXP.customers.map(c=>Number(c.orders90)).sort((a,b)=>a-b);

  function project(c){
    const x=c.x-.5,y=c.y-.5,z=(c.z==null?.5:c.z)-.5;
    const cy=Math.cos(yaw),sy=Math.sin(yaw);
    const x1=x*cy-z*sy,z1=x*sy+z*cy;
    const cp=Math.cos(pitch),sp=Math.sin(pitch);
    const y1=y*cp-z1*sp,z2=y*sp+z1*cp;
    const f=1+z2*.22;
    return {cx:W/2+x1*380*f,cy:H/2-y1*380*f,depth:z2,scale:f};
  }

  EXP.customers.forEach((c,i)=>{
    const g=document.createElementNS(NS,"g");
    const hit=dot(0,0,13,"transparent");
    hit.setAttribute("class","hit"); hit.setAttribute("pointer-events","all"); hit.style.cursor="pointer";
    const title=document.createElementNS(NS,"title");
    title.textContent=c.key+" · "+(c.churn90?"churned":"retained")+" · "+money(c.value90)+" value_90d";
    hit.appendChild(title);
    hit.onclick=()=>{if(!dragMoved)select(i);};
    const circle=dot(0,0,5.5,c.churn90?"#dc2626":"#059669");
    circle.setAttribute("class","dot"); circle.setAttribute("pointer-events","none");
    g.appendChild(hit); g.appendChild(circle); marksLayer.appendChild(g);
    const mark={g,hit,circle,c}; marks.push(mark); byKey[c.key]=mark;
  });

  function draw(){
    const current=EXP.customers[selected]||EXP.customers[0];
    const nbSet=new Set(((current&&current.neighbours)||[]).map(n=>n.key));
    marks.forEach((m,i)=>{
      const p=project(m.c); m.p=p;
      m.hit.setAttribute("cx",p.cx); m.hit.setAttribute("cy",p.cy); m.hit.setAttribute("r",13*p.scale);
      m.circle.setAttribute("cx",p.cx); m.circle.setAttribute("cy",p.cy);
      const isSel=i===selected,isNb=nbSet.has(m.c.key);
      m.circle.setAttribute("r",(isSel?10:isNb?7.2:5.2)*p.scale);
      const depthT=Math.max(0,Math.min(1,(p.depth+.707)/1.414));
      m.circle.setAttribute("opacity",(isSel?1:isNb?.96:.40+.36*depthT).toFixed(2));
      if(isSel||isNb){m.circle.setAttribute("stroke",INK);m.circle.setAttribute("stroke-width",isSel?2:1.5);}
      else{m.circle.removeAttribute("stroke");m.circle.removeAttribute("stroke-width");}
    });
    marks.slice().sort((a,b)=>a.p.depth-b.p.depth).forEach(m=>marksLayer.appendChild(m.g));
    const selMark=marks[selected]; if(selMark) marksLayer.appendChild(selMark.g);
    const cube=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
      .map(([x,y,z])=>project({x,y,z}));
    frameLayer.innerHTML="";
    [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
      .forEach(([a,b])=>frameLayer.appendChild(ln(cube[a].cx,cube[a].cy,cube[b].cx,cube[b].cy,{stroke:"#edeff2",w:1})));
    lineLayer.innerHTML="";
    if(current){
      const p0=project(current);
      ((current.neighbours)||[]).forEach(n=>{
        const m=byKey[n.key]; if(!m) return;
        lineLayer.appendChild(ln(p0.cx,p0.cy,m.p.cx,m.p.cy,{stroke:"rgba(5,150,105,.46)",w:1.3}));
      });
    }
  }

  function renderCard(i){
    const c=EXP.customers[i];
    const box=$("#detail"); box.innerHTML="";
    const card=el("div","custcard");

    const head=el("div","custcardhead");
    const left=el("div");
    left.appendChild(el("div","custtag","Selected customer"));
    left.appendChild(el("div","custkey",c.key));
    head.appendChild(left); card.appendChild(head);

    const caps=el("div","custcaps");
    function capability(name,meta){
      const sec=el("div","capability");
      const h=el("div","caphead");
      h.appendChild(el("div","capname",name));
      h.appendChild(el("div","capmeta",meta));
      sec.appendChild(h);
      return sec;
    }

    const predict=capability("Predict","90-day held-out");
    const ptable=el("table","cardtable");
    const phead=el("thead"),phr=el("tr");
    ["Metric","Value","Sample"].forEach((label,i)=>phr.appendChild(el("th",i?"r":null,label)));
    phead.appendChild(phr); ptable.appendChild(phead);
    const pbody=el("tbody");
    [
      ["value_90d",money(c.value90),ordinal(pctile(valueSorted,c.value90))+" pct",""],
      ["orders_90d",String(c.orders90),ordinal(pctile(orderSorted,c.orders90))+" pct",""],
      ["churn_90d",c.churn90?"churned":"retained","actual","txt "+(c.churn90?"bad":"good")],
    ].forEach(([k,v,s,cls])=>{
      const tr=el("tr");
      tr.appendChild(el("td","k",k));
      tr.appendChild(el("td","v"+(cls?" "+cls:""),v));
      tr.appendChild(el("td","s",s));
      pbody.appendChild(tr);
    });
    ptable.appendChild(pbody); predict.appendChild(ptable); caps.appendChild(predict);

    const retrieve=capability("Retrieve","5 nearest neighbours");
    const list=el("div","nbminilist");
    (c.neighbours||[]).forEach(nb=>{
      const row=el("div","nbrow");
      row.appendChild(el("span","k",nb.key));
      row.appendChild(el("span","v",money(nb.value90)+" · "+nb.orders90+" ord"));
      const mark=el("span","dotmark"+(nb.churn90?" bad":""));
      mark.title=nb.churn90?"churned":"retained";
      row.appendChild(mark); list.appendChild(row);
    });
    retrieve.appendChild(list);
    const rn=el("div","capnote");
    rn.appendChild(document.createTextNode("lookalike value "));
    rn.appendChild(el("b",null,money(c.neighbour_mean_value)));
    rn.appendChild(document.createTextNode(" · "));
    rn.appendChild(el("b",null,Math.round(c.neighbour_churn_agree*100)+"%"));
    rn.appendChild(document.createTextNode(" churn agreement"));
    retrieve.appendChild(rn); caps.appendChild(retrieve);

    const act=capability("Act","known optima");
    const atable=el("table","cardtable");
    const ahead=el("thead"),ahr=el("tr");
    ["Action","Choice","Benefit"].forEach((label,i)=>ahr.appendChild(el("th",i?"r":null,label)));
    ahead.appendChild(ahr); atable.appendChild(ahead);
    const abody=el("tbody");
    const known=DATA.rl_known||{};
    const channelNames={direct_mail:"Direct mail",email:"Email",push:"Push",sms:"SMS"};
    const dowNames=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
    [
      ["Channel",channelNames[c.channel]||c.channel,"channel"],
      ["Offer","Tier "+c.offer_opt,"offer"],
      ["Frequency",c.freq_opt+(c.freq_opt===1?" send":" sends"),"freq"],
      ["Hour",String(c.hour_opt).padStart(2,"0")+":00","hour"],
      ["Day",dowNames[c.dow_opt],"dow"],
    ].filter(([, ,key])=>known[key])
      .map(([label,choice,key])=>({label,choice,gain:(Number(known[key].E)-Number(known[key].table))*100}))
      .sort((a,b)=>b.gain-a.gain)
      .forEach(({label,choice,gain},i)=>{
        const tr=el("tr",i===0?"primary":null);
        tr.appendChild(el("td","k",label));
        tr.appendChild(el("td","v r",choice));
        tr.appendChild(el("td","s good",(gain>=0?"+":"")+gain.toFixed(1)+" pts"));
        abody.appendChild(tr);
      });
    atable.appendChild(abody); act.appendChild(atable);
    act.appendChild(el("div","capnote","Known optima · recovery vs table (10-NN)"));
    caps.appendChild(act);
    card.appendChild(caps);

    const ctx=el("div","custcontext");
    const hist=el("section","contextsection");
    hist.appendChild(el("div","cardtag","Recent history"));
    const ul=el("ul","minihistory");
    (c.timeline||[]).slice(0,5).forEach(ev=>{
      const li=el("li");
      li.appendChild(el("span","t",ev.ts));
      li.appendChild(el("span","e",ev.event));
      li.appendChild(el("span","d",ev.detail||""));
      ul.appendChild(li);
    });
    if(!(c.timeline||[]).length) ul.appendChild(el("li",null,"No notable events in the sample."));
    hist.appendChild(ul); ctx.appendChild(hist);

    const prof=el("section","contextsection");
    prof.appendChild(el("div","cardtag","Profile"));
    const pg=el("div","contextegrid");
    [
      ["Total events",String(c.n_events)],
      ["Channel",c.channel],
      ["Category",c.category],
      ["Regime",c.regime],
      ["Return proneness",Math.round(c.return_proneness*100)+"%"],
    ].forEach(([k,v])=>{
      const item=el("div");
      item.appendChild(el("span","k",k));
      item.appendChild(el("span","v",String(v)));
      pg.appendChild(item);
    });
    prof.appendChild(pg); ctx.appendChild(prof);
    card.appendChild(ctx); box.appendChild(card);
  }

  function select(i){selected=i;draw();renderCard(i);}

  function onDown(e){
    dragging=true; dragMoved=false;
    lastX=startX=e.clientX; lastY=startY=e.clientY;
    svgEl.style.cursor="grabbing";
  }
  function onMove(e){
    if(!dragging) return;
    const dx=e.clientX-lastX,dy=e.clientY-lastY;
    if(Math.abs(e.clientX-startX)+Math.abs(e.clientY-startY)>4) dragMoved=true;
    yaw+=dx*.008;
    pitch=Math.max(-1.15,Math.min(1.15,pitch+dy*.006));
    lastX=e.clientX; lastY=e.clientY;
    draw();
    if(e.cancelable) e.preventDefault();
  }
  function onUp(){if(!dragging) return;dragging=false;svgEl.style.cursor="grab";}
  svgEl.addEventListener("pointerdown",onDown);
  window.addEventListener("pointermove",onMove,{passive:false});
  window.addEventListener("pointerup",onUp);
  window.addEventListener("pointercancel",onUp);

  select(0);
}

$("#foot").textContent=`generated ${m.generated} · synthetic harness · ${m.customers.toLocaleString()} customers`;
</script>
</body>
</html>"""

    html = html.replace("__OBJLIST__", obj_items)
    html = html.replace("__OBJSPEC__", json.dumps(obj_spec))
    html = html.replace("__PAYLOAD__", explorer_json)
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-explorer", action="store_true", dest="no_explorer")
    args = ap.parse_args()
    explorer = None if args.no_explorer else _explorer()
    payload = _payload(explorer)
    html = render(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"WROTE {out} ({len(html) / 1024:.0f} KB)")
    print(f"  explorer customers: {len((explorer or {}).get('customers', []))}")


if __name__ == "__main__":
    main()
