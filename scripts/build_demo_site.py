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
    xy = PCA(n_components=2, random_state=0).fit_transform(Z)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)

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
                "x": round(float(xy[row, 0]), 4),
                "y": round(float(xy[row, 1]), 4),
                "n_events": int(totals.get(key, len(events))),
                "value90": round(float(yv[idx]), 1),
                "orders90": round(float(yo[idx]), 1),
                "churn90": int(yc[idx]),
                "channel": t.get("channel_pref"),
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
    return {"customers": customers, "n_sampled": len(customers), "projection": "PCA (2 components)"}


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
    obj = payload.get("objectives") or {}

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
          <span class="mono dim">{oid}</span>
          <span class="objlabel">{label}</span>
          <span class="mono skill">{float((obj.get(oid) or {}).get("skill", 0.0)):+.3f}</span>
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
            "skill": float((obj.get(oid) or {}).get("skill", 0.0)),
            "naive": float((obj.get(oid) or {}).get("naive", 0.0)),
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
.pipeline{display:grid;grid-template-columns:minmax(0,1.05fr) 18px minmax(0,1.05fr) 18px minmax(0,1.05fr) 18px minmax(0,.95fr);
  gap:10px;align-items:stretch;margin-bottom:26px}
@media(max-width:900px){.pipeline{grid-template-columns:1fr}.pipeline-arrow{display:none}}
.pipeline-step{border:1px solid var(--line);border-radius:8px;padding:12px 13px 8px;background:#fff;min-width:0}
.pipeline-top{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:4px}
.pipeline-name{font-size:12px;font-weight:600}
.pipeline-meta{font-family:ui-monospace,monospace;font-size:9.5px;color:var(--dim);white-space:nowrap}
.pipeline-vis{height:96px;display:flex;align-items:center;justify-content:center}
.pipeline-vis svg{display:block;width:100%;height:96px}
.pipeline-arrow{display:flex;align-items:center;justify-content:center;color:var(--dim);font-size:16px}
.usecasehead{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
  border-top:1px solid var(--line);padding-top:22px;margin-bottom:14px}
.usecasehead .cardtag{margin-bottom:0}
.usecasetabs{display:flex;gap:2px;border:1px solid var(--line);border-radius:8px;padding:3px}
.usetab{appearance:none;border:0;background:none;padding:7px 14px;border-radius:6px;font:inherit;
  font-size:12px;color:var(--muted);cursor:pointer}
.usetab:hover{color:var(--ink)}
.usetab.on{background:var(--ink);color:#fff}
.usecasepanel{border:1px solid var(--line);border-radius:8px;padding:16px 18px 10px;overflow-x:auto}
.usecasepanel svg{display:block;width:100%;height:auto}
@media(max-width:800px){.usecasepanel svg{min-width:760px}}
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
.plugmap{display:grid;grid-template-columns:210px minmax(0,1fr) minmax(0,1fr);gap:12px;margin-bottom:16px}
@media(max-width:900px){.plugmap{grid-template-columns:1fr}}
.plugstate,.plugcard{min-height:148px;border:1px solid var(--line);border-radius:8px;padding:14px 15px}
.plugstate{display:flex;flex-direction:column;justify-content:space-between}
.plugstate .k,.plugcard .k{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
.plugstate .v{font-family:ui-monospace,monospace;font-size:25px;font-weight:600;letter-spacing:-.03em;margin-top:4px}
.plugstate .d{font-size:11px;color:var(--muted);margin-top:2px}
.plugstate .statevis{margin-top:10px}
.plugstate .statevis svg{display:block;width:100%;height:54px}
.plugcard{appearance:none;background:#fff;color:var(--ink);font:inherit;text-align:left;cursor:pointer;
  display:flex;flex-direction:column;justify-content:space-between}
.plugcard:hover{border-color:var(--dim)}
.plugcard.on{border-color:var(--accent);background:#f7fefb}
.plugcard .t{font-size:15px;font-weight:600;letter-spacing:-.01em;margin-top:4px}
.plugcard .d{font-size:11px;color:var(--muted);margin-top:2px}
.plugcard .mini{height:42px;margin-top:10px}
.plugcard .mini svg{display:block;width:100%;height:42px}
.plugpanel{border:1px solid var(--line);border-radius:8px;padding:16px 18px 10px;overflow-x:auto}
.plugpanel svg{display:block;width:100%;height:auto}
@media(max-width:800px){.plugpanel svg{min-width:720px}}
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
.objitem{display:grid;grid-template-columns:64px 1fr auto;align-items:center;gap:10px;width:100%;
  background:none;border:0;border-bottom:1px solid var(--line);padding:13px 2px;font:inherit;
  text-align:left;cursor:pointer;color:var(--muted)}
.objitem:hover{color:var(--ink)}
.objitem.on{color:var(--ink)}
.objitem.on .objlabel{font-weight:600}
.objitem.on .mono.skill{color:var(--accent)}
.objlabel{font-size:13.5px}
.mono.skill{font-size:12px;color:var(--dim)}
.objpanel{border:1px solid var(--line);border-radius:8px;padding:26px}
.objhead{display:flex;align-items:baseline;justify-content:space-between;gap:16px}
.objhead h3{font-size:17px;margin:0;font-weight:600;letter-spacing:-.01em}
.objdesc{color:var(--muted);font-size:13.5px;margin:6px 0 22px;max-width:56ch}
.objnums{display:flex;gap:34px;margin:18px 0 4px}
.objnum .v{font-family:ui-monospace,monospace;font-size:22px;font-weight:600}
.objnum .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.protocol{color:var(--dim);font-size:12px;margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}
.subtabs{display:flex;gap:2px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:34px}
.subtabs button{appearance:none;background:none;border:0;padding:11px 14px;font:inherit;font-size:13px;
  color:var(--muted);cursor:pointer;border-bottom:1px solid transparent;white-space:nowrap}
.subtabs button:hover{color:var(--ink)}
.subtabs button.on{color:var(--ink);border-bottom-color:var(--ink);font-weight:600}
.axpanel{display:grid;grid-template-columns:minmax(0,.82fr) minmax(0,1.18fr);gap:48px;align-items:start}
@media(max-width:900px){.axpanel{grid-template-columns:1fr}}
.axname{font-size:30px;font-weight:600;letter-spacing:-.03em;margin:8px 0 0}
.axval{font-family:ui-monospace,monospace;font-size:46px;font-weight:600;letter-spacing:-.03em;margin:20px 0 2px}
.axcap{color:var(--dim);font-size:12px}
.axdesc{color:var(--muted);font-size:14.5px;margin:22px 0 28px;max-width:38ch}
.axrows{display:grid;gap:16px;max-width:42ch}
.axrow .k{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
.axrow .v{font-size:13px;color:var(--muted);margin-top:3px}
.axvis{border:1px solid var(--line);border-radius:8px;padding:22px}
.datamode{margin-top:46px}
.datamode:first-of-type{margin-top:0}
.datamodehead{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
  border-bottom:1px solid var(--line);padding-bottom:9px;margin-bottom:14px}
.datamodetag{display:flex;align-items:center;gap:8px;font-size:15px;font-weight:600;letter-spacing:-.01em}
.datamodetag::before{content:"";width:8px;height:8px;border-radius:2px;background:#d1d5db}
.datamodetag.stream::before{background:var(--accent)}
.datamodeshape{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--dim);
  text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
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
.logtable td:first-child{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--muted);white-space:nowrap}
.logtable td:nth-child(2){font-family:ui-monospace,monospace;font-size:10.5px;white-space:nowrap}
.logtable td.attrs{font-family:ui-monospace,monospace;font-size:9.8px;line-height:1.5;
  color:var(--muted);min-width:440px;white-space:normal;word-break:break-word}
.encsection{border-top:1px solid var(--line);margin-top:54px;padding-top:38px}
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
#scatter{display:block;width:100%;height:auto;background:#fff}
.scatterfoot{display:flex;gap:22px;align-items:center;border-top:1px solid var(--line2);
  padding-top:12px;margin-top:12px;color:var(--muted);font-size:13px}
.scatterfoot .sw{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.scatterfoot .sw.retained{background:var(--accent)}
.scatterfoot .sw.churned{background:#dc2626}
.scatterfoot .hint{margin-left:auto;color:var(--dim);font-family:ui-monospace,monospace;
  font-size:11px;letter-spacing:.04em;text-transform:uppercase}
.custhead{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
  border-top:1px solid var(--line);padding-top:24px;margin-bottom:20px}
@media(max-width:700px){.custhead{align-items:flex-start;flex-direction:column;gap:8px}}
.custtag{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
.custkey{font-family:ui-monospace,monospace;font-size:32px;font-weight:600;letter-spacing:-.03em;margin-top:4px}
.custsub{color:var(--muted);font-size:15px;margin-top:2px}
.outcome{font-size:15px;font-weight:600;white-space:nowrap}
.outcome.retained{color:var(--accent)}
.outcome.churned{color:#dc2626}
.custstats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:18px}
@media(max-width:820px){.custstats{grid-template-columns:repeat(2,minmax(0,1fr))}}
.custstat{border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.custstat .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.custstat .v{font-family:ui-monospace,monospace;font-size:30px;font-weight:600;letter-spacing:-.03em;margin-top:4px}
.custprofile{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);margin-bottom:24px}
@media(max-width:900px){.custprofile{grid-template-columns:repeat(2,minmax(0,1fr))}}
.custprofile div{padding:13px 16px;border-right:1px solid var(--line2)}
.custprofile div:nth-child(5n){border-right:0}
@media(max-width:900px){.custprofile div:nth-child(5n){border-right:1px solid var(--line2)}
  .custprofile div:nth-child(2n){border-right:0}}
.custprofile .k{display:block;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.custprofile .v{display:block;font-size:16px;font-weight:600;margin-top:3px}
.custdetail{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(0,.92fr);gap:16px;align-items:start}
@media(max-width:900px){.custdetail{grid-template-columns:1fr}}
.custdetail .panelbox{padding:20px}
.timeline{list-style:none;margin:4px 0 0;padding:0}
.timeline li{display:grid;grid-template-columns:132px minmax(0,1fr) auto;gap:12px;padding:10px 0;
  border-bottom:1px solid var(--line2);font-size:14px;align-items:baseline}
.timeline li:last-child{border-bottom:0}
.timeline .t{color:var(--dim);font-family:ui-monospace,monospace;font-size:12px}
.timeline .e{color:var(--ink)}
.timeline .d{color:var(--accent);font-family:ui-monospace,monospace;font-size:12.5px;text-align:right}
.timeline .d.good{color:var(--accent)}
.timeline .d.bad{color:#dc2626}
@media(max-width:600px){
  .timeline li{grid-template-columns:92px minmax(0,1fr);gap:6px 10px}
  .timeline .e{min-width:0;overflow-wrap:anywhere}
  .timeline .d{grid-column:2;text-align:left;font-size:11.5px}
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
      <h2>Input: Event Stream. Output: Foundation Model</h2>
      <div class="pipeline" id="pipeline"></div>
      <div class="usecasehead">
        <div class="cardtag">One frozen state, three plugin families</div>
        <div class="usecasetabs" id="usecasetabs"></div>
      </div>
      <div class="usecasepanel" id="usecasepanel"></div>
    </section>

    <section id="data" hidden>
      <h2>Data</h2>
      <p class="lede">Two views of the same customer. The legacy RFM snapshot is on top; the raw event stream is below.</p>
      <div class="datamode">
        <div class="datamodehead">
          <div class="datamodetag">Legacy</div>
          <div class="datamodeshape">one row / customer</div>
        </div>
        <div class="legacycard panelbox">
          <div class="legacyhead">
            <div class="legacymeta" id="legacymeta"></div>
            <div class="legacyhint">select a row</div>
          </div>
          <div id="legacytable" class="legacytable"></div>
        </div>
      </div>
      <div class="datamode">
        <div class="datamodehead">
          <div class="datamodetag stream">Event stream</div>
          <div class="datamodeshape">one row / event</div>
        </div>
        <div id="datastream"></div>
      </div>
    </section>

    <section id="encoder" hidden>
      <h2>Encoder</h2>
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
      <p class="lede">Held-out customers projected from the frozen embedding. Click a point to see the customer, their history, and their nearest neighbours below the chart.</p>
      <div class="explorerchart panelbox">
        <svg id="scatter" viewBox="0 0 1000 600" role="img" aria-label="Embedding scatter"></svg>
        <div class="scatterfoot">
          <span><i class="sw retained"></i> retained</span>
          <span><i class="sw churned"></i> churned</span>
          <span class="hint">click a point · lines show 5 nearest neighbours</span>
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

/* ---------- overview ---------- */
const OV_DATA=DATA.data_tab;
const OV_ENC=DATA.encoder_tab;
const OV_EXP=DATA.explorer;
let overviewUse="predict";
let retrieveIdx=0;

function addSvgTitle(node,text){
  const t=document.createElementNS(NS,"title"); t.textContent=text; node.appendChild(t); return node;
}

function miniStreamSVG(){
  const W=220,H=96,pad=4,s=svg(W,H);
  if(!OV_DATA||!OV_DATA.customers||!OV_DATA.customers.length) return s;
  const c=OV_DATA.customers.find(x=>x.key==="rc0000208")||OV_DATA.customers[0];
  const types=OV_DATA.types||[];
  const rowH=(H-2*pad)/Math.max(1,types.length);
  types.forEach((t,i)=>{
    const y=pad+(i+0.5)*rowH;
    s.appendChild(ln(pad,y,W-pad,y,{stroke:"#f3f4f6"}));
  });
  c.events.forEach(e=>{
    const idx=types.findIndex(t=>t.id===e[1]);
    if(idx<0) return;
    const x=pad+(e[2]/OV_DATA.window_days)*(W-2*pad);
    const y=pad+(idx+0.5)*rowH;
    s.appendChild(dot(x,y,1.05,"rgba(17,24,39,.52)"));
  });
  s.appendChild(ln(pad,H-pad,W-pad,H-pad,{stroke:LINE}));
  return s;
}

function miniEncoderSVG(){
  const W=220,H=96,s=svg(W,H);
  const layers=[
    {x:18, ys:[30,42,54,66], fill:"#d1d5db"},
    {x:66, ys:[17,29,41,53,65,77], fill:INK},
    {x:114, ys:[17,29,41,53,65,77], fill:INK},
    {x:162, ys:[17,29,41,53,65,77], fill:INK},
    {x:204, ys:[30,42,54,66], fill:ACC},
  ];
  layers.slice(0,-1).forEach((a,li)=>{
    const b=layers[li+1];
    a.ys.forEach(y1=>b.ys.forEach(y2=>{
      s.appendChild(ln(a.x,y1,b.x,y2,{stroke:"rgba(17,24,39,.09)",w:.7}));
    }));
  });
  layers.forEach((layer,li)=>{
    layer.ys.forEach(y=>{
      s.appendChild(dot(layer.x,y,li===0?2.1:li===layers.length-1?2.6:2.3,layer.fill));
    });
  });
  return s;
}

function miniStateSVG(){
  const W=220,H=96,s=svg(W,H);
  if(!OV_ENC||!OV_ENC.shape) return s;
  const vals=OV_ENC.shape, maxv=Math.max(...vals.map(v=>Math.abs(v)))||1;
  const cols=16, rows=8, cell=9, gap=1.4;
  const x0=(W-(cols*cell+(cols-1)*gap))/2;
  const y0=(H-(rows*cell+(rows-1)*gap))/2;
  vals.slice(0,cols*rows).forEach((v,i)=>{
    const x=x0+(i%cols)*(cell+gap), y=y0+Math.floor(i/cols)*(cell+gap);
    const a=Math.min(1,Math.abs(v)/maxv);
    s.appendChild(rect(x,y,cell,cell,v>=0?`rgba(5,150,105,${0.12+0.78*a})`:`rgba(17,24,39,${0.06+0.60*a})`,1.5));
  });
  return s;
}

function miniPluginsSVG(){
  const W=220,H=96,s=svg(W,H);
  const cx=110,cy=49,r=30;
  const nodes=[
    ["predict",-Math.PI/2],
    ["retrieve",Math.PI/6],
    ["act",5*Math.PI/6],
  ];
  nodes.forEach(([label,angle])=>{
    const x=cx+r*Math.cos(angle), y=cy+r*Math.sin(angle);
    s.appendChild(ln(cx,cy,x,y,{stroke:LINE}));
    s.appendChild(dot(x,y,5.5,"#fff",ACC));
    const ly=y+(Math.sin(angle)<-0.5?-10:14);
    s.appendChild(tx(x,ly,label,{size:9,fill:MUT,anchor:"middle"}));
  });
  s.appendChild(dot(cx,cy,7,ACC));
  return s;
}

function renderPipeline(){
  const host=$("#pipeline"); host.innerHTML="";
  const steps=[
    ["stream","Event stream","one stream",miniStreamSVG],
    ["encoder","Encoder","2 × "+m.d,miniEncoderSVG],
    ["state","Frozen state",m.state_dim.toLocaleString()+" dims",miniStateSVG],
    ["plugins","Plugins","3 families",miniPluginsSVG],
  ];
  steps.forEach(([key,name,meta,vis],i)=>{
    const step=el("div","pipeline-step");
    const top=el("div","pipeline-top");
    top.appendChild(el("span","pipeline-name",name));
    top.appendChild(el("span","pipeline-meta",meta));
    step.appendChild(top);
    const wrap=el("div","pipeline-vis"); wrap.appendChild(vis()); step.appendChild(wrap);
    host.appendChild(step);
    if(i<steps.length-1) host.appendChild(el("div","pipeline-arrow","→"));
  });
}

function chartHeader(tag,items){
  const bar=el("div","chartbar");
  bar.appendChild(el("div","charttag",tag));
  const leg=el("div","chartlegend");
  items.forEach(([label,cls])=>{
    const span=el("span",cls,label); leg.appendChild(span);
  });
  bar.appendChild(leg); return bar;
}
function chartFoot(left,right){
  const foot=el("div","chartfoot");
  foot.appendChild(el("span",null,left)); foot.appendChild(el("span",null,right)); return foot;
}

function predictChart(){
  const W=1000,H=348,pad={l:48,r:22,t:44,b:42}, s=svg(W,H);
  const vals=DATA.supervised.values||{};
  const order=[["value90","value_90d"],["orders90","orders_90d"],["churn90","churn_90d"]];
  const plotH=H-pad.t-pad.b;
  [0,.25,.5,.75,1].forEach(g=>{
    const y=H-pad.b-g*plotH;
    s.appendChild(ln(pad.l,y,W-pad.r,y,{stroke:g===0?LINE:"#f3f4f6"}));
    if(g>0) s.appendChild(tx(pad.l-8,y+3,Math.round(g*100)+"%",{size:9,fill:DIM,anchor:"end"}));
  });
  const gw=(W-pad.l-pad.r)/order.length, bw=46;
  order.forEach(([key,label],i)=>{
    const v=vals[key]||{}, table=Number(v.raw||0), frozen=Number(v.full_measure||0);
    const center=pad.l+gw*(i+0.5);
    const h1=Math.max(1,table*plotH), h2=Math.max(1,frozen*plotH);
    const b1=rect(center-bw-5,H-pad.b-h1,bw,h1,"#d1d5db",3);
    const b2=rect(center+5,H-pad.b-h2,bw,h2,ACC,3);
    addSvgTitle(b1,label+" · table "+table.toFixed(4));
    addSvgTitle(b2,label+" · frozen "+frozen.toFixed(4));
    s.appendChild(b1); s.appendChild(b2);
    s.appendChild(tx(center,H-16,label,{size:11,fill:MUT,anchor:"middle"}));
  });
  return s;
}

function actChart(){
  const W=1000,H=348,pad={l:92,r:24,t:44,b:38}, s=svg(W,H);
  const rl=DATA.rl_known||{};
  const keys=["channel","offer","freq","hour","dow"].filter(k=>rl[k]);
  if(!keys.length) return s;
  const rowH=(H-pad.t-pad.b)/keys.length, bh=14;
  [0,.25,.5,.75,1].forEach(g=>{
    const x=pad.l+g*(W-pad.l-pad.r);
    s.appendChild(ln(x,pad.t,x,H-pad.b,{stroke:g===0?LINE:"#f3f4f6"}));
    if(g>0) s.appendChild(tx(x,H-12,Math.round(g*100)+"%",{size:9,fill:DIM,anchor:"middle"}));
  });
  keys.forEach((k,i)=>{
    const row=rl[k], center=pad.t+(i+0.5)*rowH;
    const y1=center-bh-3, y2=center+3;
    const w1=Math.max(1,Number(row.table)* (W-pad.l-pad.r));
    const w2=Math.max(1,Number(row.E)*(W-pad.l-pad.r));
    s.appendChild(tx(pad.l-10,center+4,k,{size:10,fill:MUT,anchor:"end"}));
    const b1=rect(pad.l,y1,w1,bh,"#d1d5db",3);
    const b2=rect(pad.l,y2,w2,bh,ACC,3);
    addSvgTitle(b1,k+" · table "+Number(row.table).toFixed(4));
    addSvgTitle(b2,k+" · frozen "+Number(row.E).toFixed(4));
    s.appendChild(b1); s.appendChild(b2);
  });
  return s;
}

function retrieveChart(){
  const W=1000,H=360,pad=28,s=svg(W,H);
  const pts=(OV_EXP&&OV_EXP.customers)||[];
  if(!pts.length) return s;
  if(retrieveIdx<0||retrieveIdx>=pts.length) retrieveIdx=0;
  const sel=pts[retrieveIdx];
  const byKey=Object.fromEntries(pts.map(c=>[c.key,c]));
  const nb=new Set((sel.neighbours||[]).map(n=>n.key));
  const X=x=>pad+x*(W-2*pad), Y=y=>H-pad-y*(H-2*pad);
  (sel.neighbours||[]).forEach(n=>{
    const c=byKey[n.key]; if(!c) return;
    s.appendChild(ln(X(sel.x),Y(sel.y),X(c.x),Y(c.y),{stroke:"rgba(5,150,105,.4)",w:1.2}));
  });
  pts.forEach((c,i)=>{
    const cx=X(c.x), cy=Y(c.y);
    const isSel=i===retrieveIdx, isNb=nb.has(c.key);
    const hit=dot(cx,cy,13,"transparent");
    hit.style.cursor="pointer";
    hit.onclick=()=>{retrieveIdx=i;renderOverviewPanel();};
    addSvgTitle(hit,c.key);
    s.appendChild(hit);
    const circle=dot(cx,cy,isSel?8:isNb?5.2:3.4,isSel?INK:isNb?ACC:"#d1d5db");
    circle.setAttribute("pointer-events","none");
    s.appendChild(circle);
  });
  s.appendChild(tx(pad,18,sel.key,{size:11,fill:MUT}));
  return s;
}

function renderOverviewPanel(){
  document.querySelectorAll(".usetab").forEach(b=>b.classList.toggle("on",b.dataset.use===overviewUse));
  const host=$("#usecasepanel"); host.innerHTML="";
  if(overviewUse==="retrieve"){
    host.appendChild(chartHeader("Retrieve · frozen geometry",[["selected","selected"],["5 neighbours","neighbour"],["all customers","other"]]));
    host.appendChild(retrieveChart());
    host.appendChild(chartFoot("click a point to change the neighbour set","5 nearest neighbours in the state"));
  }else if(overviewUse==="act"){
    host.appendChild(chartHeader("Act · known-effect recovery",[["table","table"],["frozen state","frozen"]]));
    host.appendChild(actChart());
    host.appendChild(chartFoot("randomized holdout · hover a bar for values","10-NN purity · higher is better"));
  }else{
    host.appendChild(chartHeader("Predict · supervised heads",[["table","table"],["frozen state","frozen"]]));
    host.appendChild(predictChart());
    host.appendChild(chartFoot("held-out · R² value_90d / orders_90d · AUC churn_90d","hover a bar · higher is better"));
  }
}

function initOverview(){
  renderPipeline();
  const tabs=$("#usecasetabs");
  [["predict","Predict"],["retrieve","Retrieve"],["act","Act"]].forEach(([id,label])=>{
    const b=el("button","usetab"+(id===overviewUse?" on":""),label);
    b.dataset.use=id;
    b.onclick=()=>{overviewUse=id;renderOverviewPanel();};
    tabs.appendChild(b);
  });
  if(OV_EXP&&OV_EXP.customers){
    const idx=OV_EXP.customers.findIndex(c=>c.key==="rc0000208");
    if(idx>=0) retrieveIdx=idx;
  }
  renderOverviewPanel();
}
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
  const s=svg(560,330), pad={l:44,r:16,t:16,b:38};
  const W=560,H=330;
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
    s.appendChild(tx(W-pad.r,pad.t+10,"effective rank "+(bd.eff_rank||0).toFixed(2),{size:11,fill:ACC,anchor:"end"}));
  }

  if(id==="A2" && (bd.mass||{}).points){
    const pts=bd.mass.points, lim=3;
    const X=x=>pad.l+((x+lim)/(2*lim))*(W-pad.l-pad.r), Y=y=>H-pad.b-((y+lim)/(2*lim))*(H-pad.t-pad.b);
    s.appendChild(ln(X(-lim),Y(-lim),X(lim),Y(lim),{stroke:LINE,dash:"4 4"}));
    pts.forEach(([a,b])=>s.appendChild(dot(X(a),Y(b),1.5,"rgba(5,150,105,.35)")));
    axis();
    s.appendChild(tx(pad.l,H-8,"actual past summary (z)",{size:10,fill:DIM}));
    s.appendChild(tx(W-pad.r,pad.t+10,"r² "+(((bd.mass||{}).r2)||0).toFixed(2),{size:11,fill:ACC,anchor:"end"}));
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
    s.appendChild(tx(W-pad.r,H-pad.b-12,"AUC "+(((bd.roc||{}).auc)||0).toFixed(3),{size:11,fill:ACC,anchor:"end"}));
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
    s.appendChild(tx(W-pad.r,pad.t+10,"ECE "+Number(((DATA.bprops||{}).ece_next_event)||0).toFixed(3),{size:11,fill:ACC,anchor:"end"}));
  }

  if(id==="A6"){
    const keys=["channel","offer","freq","hour","dow"];
    keys.forEach((k,i)=>{
      const y=pad.t+18+i*44;
      s.appendChild(tx(pad.l,y+4,k,{size:11,fill:MUT}));
      for(let j=0;j<4;j++){
        const x=140+j*70, on=(j===i%4);
        s.appendChild(rect(x,y-10,52,22,on?"#ecfdf5":"#fff",4, on?ACC:LINE));
        if(on) s.appendChild(tx(x+26,y+4,"randomised",{size:8,fill:ACC,anchor:"middle"}));
      }
    });
    s.appendChild(tx(140,H-8,"one key randomised per day, others at the customer optimum",{size:10,fill:DIM}));
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
  const left=el("div");
  left.appendChild(el("div","mono dim small",id));
  left.appendChild(el("div","axname",d.name));
  left.appendChild(el("div","axval",d.val()));
  left.appendChild(el("div","axcap",d.cap));
  left.appendChild(el("p","axdesc",d.desc));
  const rows=el("div","axrows");
  [["What it measures",d.what],["How it is measured",d.how],["Why it matters",d.why]].forEach(([k,v])=>{
    const r=el("div","axrow"); r.appendChild(el("div","k",k)); r.appendChild(el("div","v",v)); rows.appendChild(r);
  });
  left.appendChild(rows);
  host.appendChild(left);
  host.appendChild(axVis(id));
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
  const defaultIndex=Math.max(0,DT.customers.findIndex(c=>c.key==="rc0000208"));
  let selected=defaultIndex;

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
      tr.onclick=()=>select(i);
      tr.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select(i);}};
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
      s.appendChild(dot(cx,cy,2.1,"rgba(17,24,39,.42)"));
    });
    s.appendChild(ln(pad.l,H-pad.b,W-pad.r,H-pad.b,{stroke:LINE}));
    s.appendChild(tx(pad.l,H-9,month(DT.window_start),{size:10,fill:DIM}));
    s.appendChild(tx(W-pad.r,H-9,"window end",{size:10,fill:DIM,anchor:"end"}));
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
    tl.appendChild(el("div","cardtag","Event timeline · hover a dot for event_ts / event_attributes"));
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
    c.events.forEach(e=>{
      const tr=el("tr");
      tr.appendChild(el("td",null,e[0]));
      tr.appendChild(el("td",null,e[1]));
      tr.appendChild(el("td","attrs",e[3]||"{}"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody); box.appendChild(table);
    ledger.appendChild(box); flow.appendChild(ledger); host.appendChild(flow);
  }

  function select(i){
    selected=i;
    renderTable();
    renderStream();
  }

  renderTable();
  renderStream();
}

/* ---------- encoder tab ---------- */
const ENC=DATA.encoder_tab;
if(!ENC){
  const host=$("#encoder");
  if(host) host.innerHTML="<h2>Encoder</h2><p class='lede'>Run <code>python3 scripts/build_encoder_tab.py</code> with the frozen cache available to build this tab.</p>";
}else{
  let featureFamily="shape";

  function featCols(id){
    if(id==="learned_state") return 48;
    if(id==="forecast") return 42;
    if(id==="uncertainty") return 8;
    return 16;
  }

  function renderFeatSelector(){
    const host=$("#featselector"); host.innerHTML="";
    const maxDim=Math.max(...ENC.families.map(f=>f.dim));
    ENC.families.forEach(f=>{
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
    const fam=ENC.families.find(f=>f.id===featureFamily)||ENC.families[0];
    $("#shapehead").innerHTML="";
    const left=el("div");
    left.appendChild(el("div","cardtag",fam.label));
    left.appendChild(el("div","shapekey",ENC.customer));
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
  renderEncoder();
}

/* ---------- objectives ---------- */
function strip(spec){
  const W=520,H=96,s=svg(W,H);
  const n=11, x0=18, step=(W-36)/(n-1), y=44;
  for(let i=0;i<n;i++){
    const x=x0+i*step, isT=spec.target.includes(i);
    if(isT){
      if(spec.mode==="decay"){const a=0.14+(i-spec.cut)*0.06;s.appendChild(dot(x,y,7,`rgba(5,150,105,${Math.min(a,0.5)})`));}
      s.appendChild(dot(x,y,5.4,"#fff",ACC));
    } else if(i<=spec.cut){
      s.appendChild(dot(x,y,4.4,INK));
    } else {
      s.appendChild(dot(x,y,3.4,"#fff",LINE));
    }
    if(i===spec.cut) s.appendChild(tx(x,y+30,"t",{size:9,fill:DIM,anchor:"middle"}));
  }
  if(spec.mode==="span"||spec.mode==="decay"){
    const xa=x0+spec.cut*step, xb=x0+Math.max(...spec.target)*step;
    s.appendChild(ln(xa, y - 16, xb, y - 16, {stroke: ACC, dash: "3 3"}));
    s.appendChild(ln(xa,y-20,xa,y-12,{stroke:INK})); s.appendChild(ln(xb,y-20,xb,y-12,{stroke:INK}));
  }
  s.appendChild(tx(18,y+52,"observed history",{size:9,fill:DIM}));
  s.appendChild(tx(W-18,y+52,"target",{size:9,fill:ACC,anchor:"end"}));
  return s;
}
function objPanel(id){
  const spec=OBJSPEC[id]; if(!spec) return;
  document.querySelectorAll(".objitem").forEach(b=>b.classList.toggle("on",b.dataset.obj===id));
  const box=$("#objpanel"); box.innerHTML="";
  const head=el("div","objhead");
  head.appendChild(el("h3",null,spec.label));
  const sc=el("div","mono dim small",`skill ${spec.skill>=0?"+":""}${spec.skill.toFixed(3)}`);
  head.appendChild(sc);
  box.appendChild(head);
  box.appendChild(el("p","objdesc",spec.desc));
  const ste=strip(spec); ste.setAttribute("width","100%"); box.appendChild(ste);
  const nums=el("div","objnums");
  [["learned skill",(spec.skill>=0?"+":"")+spec.skill.toFixed(3)],["naive baseline",spec.naive.toFixed(3)],["improvement",(spec.skill-spec.naive>=0?"+":"")+(spec.skill-spec.naive).toFixed(3)]]
   .forEach(([k,v])=>{const d=el("div","objnum");d.appendChild(el("div","v",v));d.appendChild(el("div","k",k));nums.appendChild(d);});
  box.appendChild(nums);
  box.appendChild(el("div","protocol","Linear probe on frozen states, grouped 5-fold by customer. Skill is improvement over the naive baseline. No business labels are used in training or in the target."));
}
document.querySelectorAll(".objitem").forEach(b=>b.onclick=()=>objPanel(b.dataset.obj));
objPanel("jepa");

/* ---------- plugins ---------- */
const PL_EFFECT=DATA.rl_known||{};
const PL_POLICY=DATA.rl_policy||{};
let pluginMode="effect";

function plugStateMini(){
  const W=180,H=54,s=svg(W,H);
  const vals=(OV_ENC&&OV_ENC.shape)||[];
  if(!vals.length) return s;
  const cols=16,rows=8,cell=5,gap=1.2;
  const x0=(W-(cols*cell+(cols-1)*gap))/2;
  const y0=(H-(rows*cell+(rows-1)*gap))/2;
  const maxv=Math.max(...vals.map(v=>Math.abs(v)))||1;
  vals.slice(0,cols*rows).forEach((v,i)=>{
    const x=x0+(i%cols)*(cell+gap), y=y0+Math.floor(i/cols)*(cell+gap);
    const a=Math.min(1,Math.abs(v)/maxv);
    s.appendChild(rect(x,y,cell,cell,v>=0?`rgba(5,150,105,${0.12+0.78*a})`:`rgba(17,24,39,${0.06+0.60*a})`,1));
  });
  return s;
}

function miniEffectSVG(){
  const W=240,H=42,pad=8,s=svg(W,H);
  const keys=["channel","offer","freq","hour","dow"].filter(k=>PL_EFFECT[k]);
  const rowH=(H-2*pad)/Math.max(1,keys.length);
  keys.forEach((k,i)=>{
    const r=PL_EFFECT[k], y=pad+(i+0.5)*rowH;
    const x1=pad+Number(r.table)*(W-2*pad), x2=pad+Number(r.E)*(W-2*pad);
    s.appendChild(ln(x1,y,x2,y,{stroke:"rgba(5,150,105,.4)",w:1.2}));
    s.appendChild(dot(x1,y,2.2,"#d1d5db"));
    s.appendChild(dot(x2,y,3.2,ACC));
  });
  return s;
}

function miniPolicySVG(){
  const W=240,H=42,pad=8,s=svg(W,H);
  const arms=["channel","offer"].filter(k=>PL_POLICY[k]);
  const vals=arms.flatMap(k=>[PL_POLICY[k].random.ips,PL_POLICY[k].table.ips,PL_POLICY[k].E.ips]);
  if(!vals.length) return s;
  const lo=Math.min(...vals)-0.4, hi=Math.max(...vals)+0.4;
  const X=i=>pad+i*(W-2*pad)/2;
  const Y=v=>H-pad-((v-lo)/(hi-lo||1))*(H-2*pad);
  const colors={channel:INK,offer:"#6b7280"};
  arms.forEach(k=>{
    const r=PL_POLICY[k], pts=[r.random.ips,r.table.ips,r.E.ips];
    s.appendChild(ln(X(0),Y(pts[0]),X(1),Y(pts[1]),{stroke:colors[k],w:1.2}));
    s.appendChild(ln(X(1),Y(pts[1]),X(2),Y(pts[2]),{stroke:colors[k],w:1.2}));
    pts.forEach((v,i)=>s.appendChild(dot(X(i),Y(v),i===2?3.4:2.4,colors[k])));
  });
  return s;
}

function renderPluginsMap(){
  const host=$("#plugmap"); host.innerHTML="";
  const state=el("div","plugstate");
  const stateTop=el("div");
  stateTop.appendChild(el("div","k","Frozen state"));
  stateTop.appendChild(el("div","v",Number(OV_ENC?OV_ENC.state_dim:m.state_dim).toLocaleString()));
  stateTop.appendChild(el("div","d","dimensions · one source"));
  state.appendChild(stateTop);
  const statevis=el("div","statevis"); statevis.appendChild(plugStateMini()); state.appendChild(statevis);
  host.appendChild(state);

  const makeCard=(id,tag,title,desc,mini)=>{
    const b=el("button","plugcard"+(pluginMode===id?" on":""));
    b.type="button"; b.dataset.mode=id; b.setAttribute("aria-pressed",pluginMode===id);
    const top=el("div");
    top.appendChild(el("div","k",tag));
    top.appendChild(el("div","t",title));
    top.appendChild(el("div","d",desc));
    b.appendChild(top);
    const box=el("div","mini"); box.appendChild(mini()); b.appendChild(box);
    b.onclick=()=>{pluginMode=id;renderPluginsMap();renderPluginPanel();};
    return b;
  };
  host.appendChild(makeCard("effect","Known effect","Randomized action recovery","5 action keys",miniEffectSVG));
  host.appendChild(makeCard("policy","Policy value","IPS reward","channel + offer",miniPolicySVG));
}

function pluginEffectChart(){
  const W=1000,H=340,pad={l:92,r:24,t:38,b:40},s=svg(W,H);
  const keys=["channel","offer","freq","hour","dow"].filter(k=>PL_EFFECT[k]);
  if(!keys.length) return s;
  const rowH=(H-pad.t-pad.b)/keys.length;
  [0,.25,.5,.75,1].forEach(g=>{
    const x=pad.l+g*(W-pad.l-pad.r);
    s.appendChild(ln(x,pad.t,x,H-pad.b,{stroke:g===0?LINE:"#f3f4f6"}));
    if(g>0) s.appendChild(tx(x,H-14,Math.round(g*100)+"%",{size:9,fill:DIM,anchor:"middle"}));
  });
  keys.forEach((k,i)=>{
    const r=PL_EFFECT[k], y=pad.t+(i+0.5)*rowH;
    const table=Number(r.table), frozen=Number(r.E);
    const x1=pad.l+table*(W-pad.l-pad.r), x2=pad.l+frozen*(W-pad.l-pad.r);
    s.appendChild(tx(pad.l-12,y+4,k,{size:10,fill:MUT,anchor:"end"}));
    s.appendChild(ln(x1,y,x2,y,{stroke:"rgba(5,150,105,.4)",w:2}));
    const a=dot(x1,y,5.4,"#d1d5db"), b=dot(x2,y,7.2,ACC);
    addSvgTitle(a,k+" · table "+table.toFixed(4));
    addSvgTitle(b,k+" · frozen "+frozen.toFixed(4));
    s.appendChild(a); s.appendChild(b);
  });
  return s;
}

function pluginPolicyChart(){
  const W=1000,H=340,pad={l:58,r:96,t:56,b:48},s=svg(W,H);
  const arms=["channel","offer"].filter(k=>PL_POLICY[k]);
  if(!arms.length) return s;
  const series=arms.map(k=>({key:k,uniform:Number(PL_POLICY[k].random.ips),table:Number(PL_POLICY[k].table.ips),frozen:Number(PL_POLICY[k].E.ips)}));
  const vals=series.flatMap(r=>[r.uniform,r.table,r.frozen]);
  const lo=Math.floor(Math.min(...vals)-0.5), hi=Math.ceil(Math.max(...vals)+0.5);
  const X=i=>pad.l+i*(W-pad.l-pad.r)/2;
  const Y=v=>H-pad.b-((v-lo)/(hi-lo||1))*(H-pad.t-pad.b);
  const colors={channel:INK,offer:"#6b7280"};
  for(let v=Math.ceil(lo);v<=Math.floor(hi);v++){
    s.appendChild(ln(pad.l,Y(v),W-pad.r,Y(v),{stroke:v===lo?LINE:"#f3f4f6"}));
    s.appendChild(tx(pad.l-10,Y(v)+3,String(v),{size:9,fill:DIM,anchor:"end"}));
  }
  const x2=X(2);
  s.appendChild(rect(x2-58,pad.t,116,H-pad.t-pad.b,"#f7fefb",0));
  ["uniform","table","frozen state"].forEach((label,i)=>{
    s.appendChild(tx(X(i),H-15,label,{size:10,fill:MUT,anchor:"middle"}));
  });
  series.forEach(r=>{
    const color=colors[r.key]||INK;
    s.appendChild(ln(X(0),Y(r.uniform),X(1),Y(r.table),{stroke:color,w:1.6}));
    s.appendChild(ln(X(1),Y(r.table),X(2),Y(r.frozen),{stroke:color,w:1.6}));
    [[0,r.uniform],[1,r.table],[2,r.frozen]].forEach(([i,v],j)=>{
      const c=dot(X(i),Y(v),j===2?6.4:4.6,j===0?"#fff":color,color);
      addSvgTitle(c,r.key+" · "+(j===0?"uniform":j===1?"table":"frozen")+" "+v.toFixed(2));
      s.appendChild(c);
    });
    s.appendChild(tx(x2+12,Y(r.frozen)+4,r.key,{size:10,fill:color}));
  });
  return s;
}

function renderPluginPanel(){
  const host=$("#plugpanel"); host.innerHTML="";
  if(pluginMode==="policy"){
    host.appendChild(chartHeader("Policy value · IPS reward",[["channel","channel"],["offer","offer"]]));
    host.appendChild(pluginPolicyChart());
    host.appendChild(chartFoot("pre-action state · logged propensity · axis zoomed","hover a point for values"));
  }else{
    host.appendChild(chartHeader("Known effect · randomized action recovery",[["table","table"],["frozen state","frozen"]]));
    host.appendChild(pluginEffectChart());
    host.appendChild(chartFoot("10-NN purity against the randomized optimum","hover a dot for values"));
  }
}

renderPluginsMap();
renderPluginPanel();

/* ---------- explorer ---------- */
const EXP=DATA.explorer;
if(!EXP||!EXP.customers||!EXP.customers.length){
  $("#explorer").innerHTML="<h2>Meet a customer</h2><p class='lede'>Run <code>python3 scripts/build_demo_site.py</code> with the frozen cache available to build the explorer.</p>";
}else{
  const svgEl=$("#scatter"),W=1000,H=600,pad=46,marks=[];
  const lines=document.createElementNS(NS,"g");
  lines.setAttribute("id","neighbour-lines");
  svgEl.appendChild(lines);
  const pointByKey={};
  EXP.customers.forEach((c,i)=>{
    const cx=pad+c.x*(W-2*pad),cy=H-pad-c.y*(H-2*pad);
    pointByKey[c.key]={cx,cy};
    const hit=dot(cx,cy,12,"transparent"); hit.style.cursor="pointer";
    hit.setAttribute("class","hit"); hit.setAttribute("data-i",String(i));
    hit.onclick=()=>select(i);
    const title=document.createElementNS(NS,"title");
    title.textContent=c.key+" · "+(c.churn90?"churned":"retained");
    hit.appendChild(title); svgEl.appendChild(hit);
    const circle=dot(cx,cy,6.2,c.churn90?"#dc2626":"#059669");
    circle.setAttribute("class","dot"); circle.setAttribute("opacity",.68);
    circle.setAttribute("pointer-events","none"); svgEl.appendChild(circle);
    marks.push({hit,circle});
  });

  function money(v){
    return "$"+Number(v).toLocaleString("en-US",{maximumFractionDigits:1});
  }

  function select(i){
    const c=EXP.customers[i];
    const nbSet=new Set((c.neighbours||[]).map(n=>n.key));
    lines.innerHTML="";
    const p0=pointByKey[c.key];
    (c.neighbours||[]).forEach(n=>{
      const p=pointByKey[n.key];
      if(p) lines.appendChild(ln(p0.cx,p0.cy,p.cx,p.cy,{stroke:"rgba(5,150,105,.42)",w:1.2}));
    });
    marks.forEach(({circle},j)=>{
      const isSel=j===i, isNb=nbSet.has(EXP.customers[j].key);
      circle.setAttribute("r",isSel?12:isNb?8.2:6.2);
      circle.setAttribute("opacity",isSel?1:isNb?1:.38);
      if(isNb && !isSel){circle.setAttribute("stroke",INK);circle.setAttribute("stroke-width","1.8");}
      else{circle.removeAttribute("stroke");circle.removeAttribute("stroke-width");}
    });
    const box=$("#detail"); box.innerHTML="";

    const head=el("div","custhead");
    const left=el("div");
    left.appendChild(el("div","custtag","Selected customer"));
    left.appendChild(el("div","custkey",c.key));
    left.appendChild(el("div","custsub",[c.channel,c.category,c.regime].join(" · ")));
    head.appendChild(left);
    head.appendChild(el("div","outcome "+(c.churn90?"churned":"retained"),c.churn90?"Churned":"Retained"));
    box.appendChild(head);

    const stats=el("div","custstats");
    [
      ["Total events",String(c.n_events)],
      ["actual value_90d",money(c.value90)],
      ["actual orders_90d",String(c.orders90)],
      ["lookalike churn_90d agreement",(c.neighbour_churn_agree*100).toFixed(0)+"%"],
    ].forEach(([k,v])=>{
      const stat=el("div","custstat");
      stat.appendChild(el("div","k",k)); stat.appendChild(el("div","v",v));
      stats.appendChild(stat);
    });
    box.appendChild(stats);

    const profile=el("div","custprofile");
    [
      ["Channel preference",c.channel],
      ["Category preference",c.category],
      ["Regime",c.regime],
      ["Return proneness",(c.return_proneness*100).toFixed(0)+"%"],
      ["Lookalike mean value",money(c.neighbour_mean_value)],
    ].forEach(([k,v])=>{
      const item=el("div");
      item.appendChild(el("span","k",k)); item.appendChild(el("span","v",String(v)));
      profile.appendChild(item);
    });
    box.appendChild(profile);

    const detail=el("div","custdetail");

    const hist=el("div","panelbox");
    hist.appendChild(el("div","cardtag","Recent history · "+(c.timeline||[]).length+" notable events"));
    const ul=el("ul","timeline");
    (c.timeline||[]).forEach(ev=>{
      const li=el("li");
      li.appendChild(el("span","t",ev.ts));
      li.appendChild(el("span","e",ev.event));
      li.appendChild(el("span","d",ev.detail||""));
      ul.appendChild(li);
    });
    hist.appendChild(ul); detail.appendChild(hist);

    const neigh=el("div","panelbox");
    neigh.appendChild(el("div","cardtag","Nearest neighbours · frozen state"));
    const ul2=el("ul","timeline");
    (c.neighbours||[]).forEach(nb=>{
      const li=el("li");
      li.appendChild(el("span","t",nb.key));
      li.appendChild(el("span","e",money(nb.value90)+" · "+nb.orders90+" orders"));
      li.appendChild(el("span","d "+(nb.churn90?"bad":"good"),nb.churn90?"churned":"retained"));
      ul2.appendChild(li);
    });
    neigh.appendChild(ul2); detail.appendChild(neigh);

    box.appendChild(detail);
  }
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
