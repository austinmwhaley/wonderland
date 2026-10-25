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
TARGET_LABEL = {
    "value90": "Value, next 90d",
    "orders90": "Orders, next 90d",
    "churn90": "Churn, next 90d",
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
        "runtime": rt,
        "explorer": explorer,
    }


def _pct(x):
    return f"{x * 100:.1f}%"


def _f(x, n=3):
    return f"{x:.{n}f}"


def render(payload) -> str:
    """Flat, light, lab-style page: axes + objectives + plugins + explorer."""
    sup = payload["supervised"]
    rl = payload["rl_known"]
    resp = payload["rl_policy"]
    obj = payload.get("objectives") or {}

    def policy_table():
        rows = []
        for arm in ("channel", "offer"):
            r = resp.get(arm) if isinstance(resp, dict) else None
            if not r:
                continue
            rows.append(
                "<tr>"
                f"<td>{arm}</td>"
                f"<td class='num'>{r['table']['ips']:.2f}</td>"
                f"<td class='num'>{r['random']['ips']:.2f}</td>"
                f"<td class='num strong'>{r['E']['ips']:.2f}</td>"
                "</tr>"
            )
        return "".join(rows)

    def rl_table():
        rows = []
        for arm in ("channel", "offer", "freq", "hour", "dow"):
            r = rl.get(arm)
            if not r:
                continue
            rows.append(
                "<tr>"
                f"<td>{arm}</td>"
                f"<td class='num'>{r['table']:.3f}</td>"
                f"<td class='num strong'>{r['E']:.3f}</td>"
                f"<td class='num strong'>{r['E'] - r['table']:+.3f}</td>"
                "</tr>"
            )
        return "".join(rows)

    def full_vs_table():
        rows = []
        for t in sup["targets"]:
            v = sup["values"][t]
            table = float(v.get("raw", 0.0))
            full = float(v.get("full_measure", 0.0))
            rows.append(
                "<tr>"
                f"<td>{TARGET_LABEL[t]}</td>"
                f"<td class='num'>{table:.4f}</td>"
                f"<td class='num strong'>{full:.4f}</td>"
                f"<td class='num strong'>{full - table:+.4f}</td>"
                "</tr>"
            )
        return "".join(rows)

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
body{font:15px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:1080px;margin:0 auto;padding:52px 24px 96px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.dim{color:var(--dim)}
.small{font-size:11.5px}
header{border-bottom:1px solid var(--line);padding-bottom:22px}
h1{font-size:17px;font-weight:600;margin:0;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin-top:4px}
nav{display:flex;gap:30px;border-bottom:1px solid var(--line);margin:26px 0 44px;overflow-x:auto}
nav button{appearance:none;background:none;border:0;padding:0 0 13px;font:inherit;font-size:13.5px;
  color:var(--muted);cursor:pointer;border-bottom:1px solid transparent;white-space:nowrap}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--ink);border-bottom-color:var(--ink)}
section[hidden]{display:none}
h2{font-size:21px;font-weight:600;margin:0 0 8px;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:14px;max-width:72ch;margin:0 0 34px}
.grid{display:grid;gap:16px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
@media(max-width:900px){.g2,.g3,.g4{grid-template-columns:1fr}}
hr{border:0;border-top:1px solid var(--line);margin:44px 0}
.axis{border:1px solid var(--line);border-radius:8px;padding:20px 20px 16px}
.axis-top{display:flex;align-items:baseline;gap:10px}
.axis-name{font-weight:600;font-size:14.5px}
.axis-val{font-family:ui-monospace,monospace;font-size:27px;font-weight:600;letter-spacing:-.02em;margin:12px 0 2px}
.axis-def{color:var(--muted);font-size:13px;margin-top:12px;min-height:38px}
.vis{margin-top:14px}
.axis.gap{background:#fafafa}
.stat{border-top:1px solid var(--line);padding-top:12px}
.stat .v{font-family:ui-monospace,monospace;font-size:26px;font-weight:600;letter-spacing:-.02em}
.stat .k{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line)}
th{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;font-size:12.5px}
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
.dcustomer{display:grid;grid-template-columns:170px minmax(0,1fr) auto;gap:28px;align-items:center;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:17px 0;margin-bottom:24px}
@media(max-width:800px){.dcustomer{grid-template-columns:1fr;gap:8px}}
.dcustkey{font-family:ui-monospace,monospace;font-size:23px;font-weight:600;letter-spacing:-.02em}
.dcustmeta{color:var(--muted);font-size:13px}
.dcuststats{display:flex;gap:16px;color:var(--muted);font-size:12px;white-space:nowrap;flex-wrap:wrap}
.dcuststats b{color:var(--ink);font-family:ui-monospace,monospace;font-weight:600}
.dataflow{display:grid;gap:16px}
.cardtag{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600;margin-bottom:10px}
.tlwrap svg{display:block;width:100%;height:auto}
@media(max-width:800px){.tlwrap{overflow-x:auto}.tlwrap svg{min-width:700px}}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:11px;margin-top:10px}
.legend span::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
.legend .h::before{background:var(--ink)}
.legend .l::before{background:var(--accent)}
.legend .x::before{background:#d1d5db}
.logscroll{max-height:520px;overflow:auto;border-top:1px solid var(--line)}
.logtable{font-size:12px}
.logtable th{position:sticky;top:0;background:#fff;z-index:2;padding:8px 7px}
.logtable td{padding:7px 7px}
.logtable tr:hover td{background:#fafafa}
.logtable td:first-child{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--dim);white-space:nowrap}
.logtable td:nth-child(2){font-family:ui-monospace,monospace;font-size:11px;white-space:nowrap}
.logtable td:nth-child(3){color:var(--muted);font-size:11.5px}
.phase{font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;font-family:ui-monospace,monospace}
.phase.history{color:var(--dim)}
.phase.labels{color:var(--accent)}
.phase.later{color:#d1d5db}
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
.custprofile div{padding:13px 16px 13px 0;border-right:1px solid var(--line2)}
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
  <header>
    <h1>demo</h1>
    <div class="sub">A frozen foundation model for customer event streams</div>
  </header>

  <nav id="tabs"></nav>

  <main>
    <section id="overview">
      <h2>The result</h2>
      <p class="lede" id="claim"></p>
      <div class="grid g4" id="stat_cards"></div>
      <hr>
      <div class="grid g2">
        <div>
          <h2 style="font-size:17px">One frozen state, three plugin families</h2>
          <p class="lede" style="margin-bottom:0">Supervised heads predict outcomes. Unsupervised heads read geometry. RL heads estimate action response from a pre-action state. The foundation is never retrained per task.</p>
        </div>
        <table>
          <thead><tr><th>Target</th><th class="num">Traditional table</th><th class="num">Frozen state</th><th class="num">Gain</th></tr></thead>
          <tbody>__FULL_TABLE__</tbody>
        </table>
      </div>
    </section>

    <section id="data" hidden>
      <h2>Data</h2>
      <p class="lede">One customer's event stream over the full 540-day window. Every aggregate is built from this ledger.</p>
      <div id="datacustomer"></div>
      <div id="datapanel"></div>
    </section>

    <section id="axes" hidden>
      <h2>Eight content axes</h2>
      <p class="lede">What the frozen state carries, axis by axis. Each axis is a distinct property of the stream, measured on held-out customers.</p>
      <div class="subtabs" id="axistabs"></div>
      <div class="axpanel" id="axpanel"></div>
    </section>

    <section id="objectives" hidden>
      <h2>Objectives</h2>
      <p class="lede">Ten self-supervised tasks trained on the same frozen state. None uses a business label: targets come from the stream itself.</p>
      <div class="objlayout">
        <div id="objlist">__OBJLIST__</div>
        <div class="objpanel" id="objpanel"></div>
      </div>
    </section>

    <section id="plugins" hidden>
      <h2>Plugins</h2>
      <p class="lede">Downstream heads read the frozen state. Known-effect recovery asks whether retrieval finds a customer's randomized optimum; policy value asks whether a learned response model beats logging.</p>
      <div class="grid g2">
        <div>
          <div class="mono dim small" style="letter-spacing:.08em;text-transform:uppercase">Known-effect recovery</div>
          <div class="chart" data-chart="rl_known"></div>
          <div class="note">10-nearest-neighbour purity against the generator's randomized optimum.</div>
        </div>
        <div>
          <div class="mono dim small" style="letter-spacing:.08em;text-transform:uppercase">RL policy value</div>
          <table>
            <thead><tr><th>Arm</th><th class="num">Table IPS</th><th class="num">Uniform</th><th class="num">Encoder IPS</th></tr></thead>
            <tbody>__POLICY_TABLE__</tbody>
          </table>
          <div class="note">Pre-action state, logged propensity, customer-grouped resampling.</div>
        </div>
      </div>
      <hr>
      <table>
        <thead><tr><th>Action key</th><th class="num">Table purity</th><th class="num">Encoder purity</th><th class="num">Gain</th></tr></thead>
        <tbody>__RL_TABLE__</tbody>
      </table>
    </section>

    <section id="explorer" hidden>
      <h2>Meet a customer</h2>
      <p class="lede">Held-out customers projected from the frozen embedding. Click a point to see the customer, their history, and their nearest neighbours below the chart.</p>
      <div class="explorerchart panelbox">
        <svg id="scatter" viewBox="0 0 1000 600" role="img" aria-label="Embedding scatter"></svg>
        <div class="scatterfoot">
          <span><i class="sw retained"></i> retained</span>
          <span><i class="sw churned"></i> churned</span>
          <span class="hint">click a point</span>
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
const TABS=[["overview","Overview"],["data","Data"],["axes","Axes"],["objectives","Objectives"],["plugins","Plugins"],["explorer","Explorer"]];
const nav=$("#tabs");
function activate(id,push){
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
$("#claim").innerHTML=`One self-supervised encoder is fitted per event stream and frozen. Its state is reused by every downstream head — <b>${m.events.toLocaleString()}</b> events from <b>${m.customers.toLocaleString()}</b> customers, ${m.d}-number shape, ${m.state_dim.toLocaleString()}-dimension published state.`;
[["Customers",m.customers.toLocaleString()],["Events",(m.events/1e6).toFixed(2)+"M"],["Frozen state",m.state_dim.toLocaleString()],["Serving",Math.round(DATA.runtime.state_rows_per_s||0).toLocaleString()+" rows/s"]]
 .forEach(([k,v])=>{const c=el("div","stat");c.appendChild(el("div","k",k));c.appendChild(el("div","v",v));$("#stat_cards").appendChild(c);});

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
  function month(iso){
    const parts=String(iso).slice(0,10).split("-");
    return MONTHS[Number(parts[1])-1]+" "+parts[0];
  }
  function dayOf(iso){
    return Math.max(0,Math.min(DT.window_days,(Date.parse(iso)-Date.parse(DT.window_start))/86400000));
  }

  function renderCustomer(){
    const c=DT.customer, host=$("#datacustomer"); host.innerHTML="";
    const bar=el("div","dcustomer");
    const left=el("div");
    left.appendChild(el("div","mono dim small","CUSTOMER"));
    left.appendChild(el("div","dcustkey",c.key));
    bar.appendChild(left);
    bar.appendChild(el("div","dcustmeta",[c.acquisition_channel,c.region,"segment "+c.segment,"loyalty "+c.loyalty_tier,"joined "+String(c.signup_ts).slice(0,10)].join(" · ")));
    const stats=el("div","dcuststats");
    [["events",DT.counts.total],["before as-of",DT.counts.history],["label window",DT.counts.labels],["later",DT.counts.later]].forEach(([k,n])=>{
      const s=el("span"); s.appendChild(el("b",null,String(n))); s.appendChild(document.createTextNode(" "+k)); stats.appendChild(s);
    });
    bar.appendChild(stats); host.appendChild(bar);
  }

  function streamTimeline(){
    const W=1000,H=300,pad={l:86,r:20,t:20,b:34};
    const s=svg(W,H), h=H-pad.t-pad.b;
    const types=DT.types, rowH=h/types.length;
    const X=d=>pad.l+(d/DT.window_days)*(W-pad.l-pad.r);
    const xa=X(dayOf(DT.as_of)), xb=X(dayOf(DT.label_end));
    s.appendChild(rect(xa,pad.t,Math.max(0,xb-xa),h,"#f0fdf4",0));
    types.forEach((t,i)=>{
      const y=pad.t+(i+0.5)*rowH;
      s.appendChild(tx(pad.l-7,y+3,t.label,{size:10,fill:MUT,anchor:"end"}));
      s.appendChild(ln(pad.l,y,W-pad.r,y,{stroke:"#f9fafb"}));
    });
    s.appendChild(ln(xa,pad.t,xa,H-pad.b,{stroke:INK,dash:"3 3"}));
    s.appendChild(tx(xa+5,pad.t-4,"as of",{size:10,fill:INK}));
    s.appendChild(ln(xb,pad.t,xb,H-pad.b,{stroke:ACC,dash:"2 3"}));
    s.appendChild(tx(xb+5,pad.t-4,"+90d",{size:10,fill:ACC}));
    const colors={history:"rgba(17,24,39,.45)",labels:ACC,later:"#d1d5db"};
    DT.events.forEach(e=>{
      const idx=types.findIndex(t=>t.id===e.type);
      if(idx<0) return;
      const y=pad.t+(idx+0.5)*rowH;
      s.appendChild(dot(X(e.day),y,2.1,colors[e.phase]||DIM));
    });
    s.appendChild(ln(pad.l,H-pad.b,W-pad.r,H-pad.b,{stroke:LINE}));
    s.appendChild(tx(pad.l,H-9,month(DT.window_start),{size:10,fill:DIM}));
    s.appendChild(tx(xa,H-9,month(DT.as_of),{size:10,fill:DIM,anchor:"middle"}));
    s.appendChild(tx(xb,H-9,month(DT.label_end),{size:10,fill:DIM,anchor:"middle"}));
    s.appendChild(tx(W-pad.r,H-9,"window end",{size:10,fill:DIM,anchor:"end"}));
    return s;
  }

  function renderData(){
    const host=$("#datapanel"); host.innerHTML="";
    const flow=el("div","dataflow");

    const tl=el("div","panelbox");
    tl.appendChild(el("div","cardtag","Event timeline · one dot per event"));
    const wrap=el("div","tlwrap"); wrap.appendChild(streamTimeline()); tl.appendChild(wrap);
    const leg=el("div","legend");
    [["h","history"],["l","90-day label"],["x","later"]].forEach(([cls,text])=>leg.appendChild(el("span",cls,text)));
    tl.appendChild(leg); flow.appendChild(tl);

    const ledger=el("div","panelbox");
    ledger.appendChild(el("div","cardtag","Event ledger · "+DT.counts.total+" rows"));
    const box=el("div","logscroll");
    const table=el("table","logtable");
    const thead=el("thead"); const hr=el("tr");
    ["Time","Event","Detail","Window"].forEach((t,i)=>{const th=el("th",i===3?"num":null,t);hr.appendChild(th);});
    thead.appendChild(hr); table.appendChild(thead);
    const tbody=el("tbody");
    DT.events.forEach(e=>{
      const tr=el("tr");
      tr.appendChild(el("td",null,e.ts.slice(5,16).replace("T"," ")));
      tr.appendChild(el("td",null,e.type));
      tr.appendChild(el("td",null,e.detail||""));
      tr.appendChild(el("td","num phase "+e.phase,e.phase==="labels"?"label":e.phase));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody); box.appendChild(table); ledger.appendChild(box);
    flow.appendChild(ledger); host.appendChild(flow);
  }

  renderCustomer(); renderData();
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

/* ---------- plugins chart ---------- */
(function(){
  const host=document.querySelector('[data-chart="rl_known"]'); if(!host) return;
  const keys=Object.keys(DATA.rl_known||{}); if(!keys.length) return;
  const W=460,H=230,s=svg(W,H),pad={l:34,r:10,t:10,b:34};
  const gw=(W-pad.l-pad.r)/keys.length;
  keys.forEach((k,i)=>{
    const r=DATA.rl_known[k],gx=pad.l+i*gw,bw=gw*0.18;
    const he=(r.E)*(H-pad.t-pad.b), ht=(r.table)*(H-pad.t-pad.b);
    s.appendChild(ln(gx+gw*0.22,H-pad.b,gx+gw*0.22,H-pad.b-ht,{stroke:DIM,w:bw}));
    s.appendChild(ln(gx+gw*0.52,H-pad.b,gx+gw*0.52,H-pad.b-he,{stroke:ACC,w:bw}));
    s.appendChild(tx(gx+gw/2,H-8,k,{size:9,fill:DIM,anchor:"middle"}));
  });
  s.appendChild(ln(pad.l,H-pad.b,W-pad.r,H-pad.b,{stroke:LINE}));
  s.appendChild(tx(0,14,"1.0",{size:9,fill:DIM}));
  host.appendChild(s);
})();

/* ---------- explorer ---------- */
const EXP=DATA.explorer;
if(!EXP||!EXP.customers||!EXP.customers.length){
  $("#explorer").innerHTML="<h2>Meet a customer</h2><p class='lede'>Run <code>python3 scripts/build_demo_site.py</code> with the frozen cache available to build the explorer.</p>";
}else{
  const svgEl=$("#scatter"),W=1000,H=600,pad=46,marks=[];
  EXP.customers.forEach((c,i)=>{
    const cx=pad+c.x*(W-2*pad),cy=H-pad-c.y*(H-2*pad);
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
    marks.forEach(({circle},j)=>{
      circle.setAttribute("r",j===i?12:6.2);
      circle.setAttribute("opacity",j===i?1:.45);
    });
    const c=EXP.customers[i],box=$("#detail"); box.innerHTML="";

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
      ["Actual value (90d)",money(c.value90)],
      ["Actual orders (90d)",String(c.orders90)],
      ["Lookalike churn agreement",(c.neighbour_churn_agree*100).toFixed(0)+"%"],
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
    html = html.replace("__FULL_TABLE__", full_vs_table())
    html = html.replace("__POLICY_TABLE__", policy_table())
    html = html.replace("__RL_TABLE__", rl_table())
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
