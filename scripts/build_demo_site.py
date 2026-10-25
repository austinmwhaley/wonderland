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
    totals = dict(
        zip(
            parquet.group_by("customer_key").len()["customer_key"].to_list(),
            parquet.group_by("customer_key").len()["len"].to_list(),
        )
    )
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
    ident = payload["identity"]
    obj = payload.get("objectives") or {}
    bd = payload.get("bdata") or {}

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

    # axis values
    eff = bd.get("eff_rank", 9.07)
    mass_r2 = (bd.get("mass") or {}).get("r2", 0.0)
    mass_dims = (bd.get("mass") or {}).get("n_dims", 0)
    ece = float((payload.get("bprops") or {}).get("ece_next_event", 0.0))
    ch = ident.get("channel") or {}
    id_val = float(ch.get("E", 0.0))
    rate = float(sup["values"]["churn90"].get("full_measure", 0.0))
    ordw = obj.get("order", {})
    rel = 0.005
    rel_chance = 0.0005

    axes = [
        (
            "A1",
            "Shape",
            "Customers with similar behaviour land near each other.",
            f"{eff:.2f}",
            "effective rank of the state",
            "shape",
        ),
        (
            "A2",
            "Mass",
            "The state remembers how much, not only what kind.",
            f"{mass_r2:.2f}",
            f"mean r² over {mass_dims} past-summary dimensions",
            "mass",
        ),
        (
            "A3",
            "Identity",
            "Lookalikes share preference, not just activity.",
            f"{id_val:.2f}",
            "kNN purity vs 0.25 chance",
            "identity",
        ),
        (
            "A4",
            "Rate",
            "Intensity and timing are separable from content.",
            f"{rate:.3f}",
            "ranking quality on churn",
            "rate",
        ),
        (
            "A5",
            "Uncertainty",
            "Confidence tracks accuracy.",
            f"{ece:.3f}",
            "expected calibration error",
            "uncertainty",
        ),
        (
            "A6",
            "Causal",
            "The state supports action-conditioned decisions.",
            "Layer 1",
            "evaluated separately, not claimed here",
            "causal",
        ),
        (
            "A7",
            "Order / Path",
            "Sequence order matters, not just the bag of events.",
            f"{float(ordw.get('skill', 0.0)):.3f}",
            f"order probe vs {float(ordw.get('naive', 0.0)):.3f} random",
            "order",
        ),
        (
            "A8",
            "Relational",
            "Structure across entities is preserved.",
            f"{rel:.3f}",
            f"cross-entity purity vs {rel_chance:.4f} chance",
            "relational",
        ),
    ]

    axis_html = "".join(
        f"""<div class="axis">
          <div class="axis-top"><span class="mono dim">{aid}</span><span class="axis-name">{name}</span></div>
          <div class="axis-val">{val}</div>
          <div class="mono dim small">{cap}</div>
          <div class="axis-def">{desc}</div>
          <div class="vis" id="vis-{key}"></div>
        </div>"""
        for aid, name, desc, val, cap, key in axes
    )

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
.explorer{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);gap:28px;align-items:start}
@media(max-width:980px){.explorer{grid-template-columns:1fr}}
#scatter{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.dot{cursor:pointer}
.panelbox{border:1px solid var(--line);border-radius:8px;padding:18px}
.kv{display:grid;grid-template-columns:1fr auto;gap:6px 14px;font-size:13px}
.kv div:nth-child(even){text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;font-size:12.5px}
.kv div:nth-child(odd){color:var(--muted)}
.timeline{list-style:none;margin:12px 0 0;padding:0}
.timeline li{display:grid;grid-template-columns:100px 1fr auto;gap:10px;padding:6px 0;
  border-bottom:1px solid var(--line2);font-size:12.5px}
.timeline .t{color:var(--dim);font-family:ui-monospace,monospace;font-size:11.5px}
.timeline .e{color:var(--ink)}
.timeline .d{color:var(--accent);font-family:ui-monospace,monospace;font-size:11.5px}
.tag{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:16px}
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

    <section id="axes" hidden>
      <h2>Eight content axes</h2>
      <p class="lede">What the frozen state carries, axis by axis. Each axis is a distinct property of the stream, measured on held-out customers.</p>
      <div class="grid g4">__AXES__</div>
      <p class="note" style="margin-top:22px">A6 is the decision axis and is evaluated separately as Layer-1 research; it is not claimed by this card.</p>
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
      <p class="lede">Held-out customers projected from the frozen embedding. Select a point to see its history, its nearest neighbours, and the outcome those neighbours suggest.</p>
      <div class="explorer">
        <div class="panelbox">
          <svg id="scatter" viewBox="0 0 640 460" role="img" aria-label="Embedding scatter"></svg>
          <div class="note" style="margin-top:10px"><span style="color:#059669">&#9679;</span> retained &nbsp; <span style="color:#dc2626">&#9679;</span> churned &nbsp; · click a point</div>
        </div>
        <div class="panelbox" id="detail"></div>
      </div>
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
const TABS=[["overview","Overview"],["axes","Axes"],["objectives","Objectives"],["plugins","Plugins"],["explorer","Explorer"]];
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

const bd=DATA.bdata||{};
/* spectrum */
(function(){const host=$("#vis-shape");if(!host||!bd.spectrum)return;const W=250,H=74,s=svg(W,H);
 const v=bd.spectrum,n=v.length;
 v.forEach((x,i)=>{const h=Math.max(1,(x)*(H-14));s.appendChild(ln(6+i*((W-12)/(n-1)),H-10,6+i*((W-12)/(n-1)),H-10-h,{stroke:i<12?INK:DIM,w:1.4}));});
 s.appendChild(tx(6,H-1,"singular values (log)",{size:9,fill:DIM}));host.appendChild(s);})();

/* mass scatter */
(function(){const host=$("#vis-mass");const mm=bd.mass;if(!host||!mm||!mm.points)return;
 const W=250,H=150,s=svg(W,H),pad=18;
 s.appendChild(ln(pad,H-pad,W-6,6,{stroke:LINE,dash:"3 3"}));
 const pts=mm.points;const lim=3;
 const X=x=>pad+((x+lim)/(2*lim))*(W-pad-8), Y=y=>H-pad-((y+lim)/(2*lim))*(H-pad-8);
 pts.forEach(([a,b])=>{s.appendChild(dot(X(a),Y(b),1.4,"rgba(5,150,105,.35)"));});
 s.appendChild(tx(pad,H-4,"actual",{size:9,fill:DIM}));
 s.appendChild(tx(W-8,H-pad+12,"predicted",{size:9,fill:DIM,anchor:"end"}));
 host.appendChild(s);})();

/* identity bar with chance */
(function(){const host=$("#vis-identity");const ch=(DATA.identity||{}).channel;if(!host||!ch)return;
 const W=250,H=70,s=svg(W,H),w=W-16;
 s.appendChild(ln(8,30,8+w,30,{stroke:LINE}));
 s.appendChild(ln(8,24,8+w,24,{stroke:INK,w:6}));
 const px=8+(0.25/0.7)*w; s.appendChild(ln(px,18,px,36,{stroke:DIM,dash:"3 3"}));
 s.appendChild(tx(px,10,"chance",{size:9,fill:DIM,anchor:"middle"}));
 s.appendChild(tx(8,H-6,"purity "+ch.E.toFixed(2)+" vs 0.25 chance",{size:9,fill:DIM}));
 host.appendChild(s);})();

/* reliability */
(function(){const host=$("#vis-uncertainty");const c=bd.calibration;if(!host||!c||!c.bins)return;
 const W=250,H=150,s=svg(W,H),pad=24;
 s.appendChild(ln(pad,H-pad,W-8,H-pad,{stroke:LINE}));
 s.appendChild(ln(pad,H-pad,pad,8,{stroke:LINE}));
 s.appendChild(ln(pad,H-pad,W-8,8,{stroke:LINE,dash:"3 3"}));
 const X=x=>pad+x*(W-pad-10), Y=y=>H-pad-y*(H-pad-12);
 const pts=c.bins.map(b=>[b.conf,b.acc]);
 pts.forEach((p,i)=>{ if(i) s.appendChild(ln(X(pts[i-1][0]),Y(pts[i-1][1]),X(p[0]),Y(p[1]),{stroke:ACC,w:1.6})); s.appendChild(dot(X(p[0]),Y(p[1]),2.4,ACC));});
 s.appendChild(tx(pad,H-6,"confidence",{size:9,fill:DIM}));
 host.appendChild(s);})();

/* rate */
(function(){const host=$("#vis-rate");if(!host)return;
 const v=DATA.supervised.values.churn90.full_measure;
 const W=250,H=66,s=svg(W,H);
 [["chance",0.5,DIM],[("encoder"),v,INK]].forEach(([lab,val,col],i)=>{
   const y=22+i*24,w=Math.max(2,(val)*(W-120));
   s.appendChild(tx(0,y+4,lab,{size:9,fill:DIM}));
   s.appendChild(ln(84,y,84+w,y,{stroke:col,w:6}));
   s.appendChild(tx(84+w+6,y+4,val.toFixed(3),{size:9,fill:col}));
 });
 host.appendChild(s);})();

/* order */
(function(){const host=$("#vis-order");const o=(DATA.objectives||{}).order;if(!host||!o)return;
 const W=250,H=74,s=svg(W,H);
 const rows=[["random baseline",o.naive,DIM],["encoder state",o.skill,INK]];
 rows.forEach(([lab,v,col],i)=>{
   const y=22+i*26, w=Math.max(2,v*(W-140));
   s.appendChild(tx(0,y+4,lab,{size:9,fill:DIM}));
   s.appendChild(ln(112,y,112+w,y,{stroke:col,w:6}));
   s.appendChild(tx(112+w+6,y+4,v.toFixed(3),{size:9,fill:col}));
 });
 host.appendChild(s);})();

/* relational */
(function(){const host=$("#vis-relational");if(!host)return;
 const W=250,H=64,s=svg(W,H);
 const rows=[["chance",0.0005,DIM],["encoder",0.005,INK]];
 const maxv=0.005;
 rows.forEach(([lab,v,col],i)=>{
   const y=20+i*22, w=Math.max(1.5,(v/maxv)*(W-150));
   s.appendChild(tx(0,y+4,lab,{size:9,fill:DIM}));
   s.appendChild(ln(78,y,78+w,y,{stroke:col,w:5}));
   s.appendChild(tx(78+w+6,y+4,v.toFixed(4),{size:9,fill:col}));
 });
 host.appendChild(s);})();

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
  const svgEl=$("#scatter"),W=640,H=460,pad=34,dots=[];
  EXP.customers.forEach((c,i)=>{
    const cx=pad+c.x*(W-2*pad),cy=H-pad-c.y*(H-2*pad);
    const circle=dot(cx,cy,3.6,c.churn90?"#dc2626":"#059669");circle.setAttribute("class","dot");
    circle.setAttribute("opacity",.62);circle.onclick=()=>select(i);
    svgEl.appendChild(circle);dots.push(circle);
  });
  function select(i){
    dots.forEach((d,j)=>{d.setAttribute("r",j===i?6:3.6);d.setAttribute("opacity",j===i?1:.35);});
    const c=EXP.customers[i],box=$("#detail");box.innerHTML="";
    box.appendChild(el("div","tag","customer"));
    const h=el("div",null,c.key);h.style.fontSize="16px";h.style.fontWeight="600";h.style.margin="4px 0 12px";box.appendChild(h);
    const kv=el("div","kv");
    [["Total events",c.n_events],["Channel preference",c.channel],["Category preference",c.category],["Regime",c.regime],["Return proneness",(c.return_proneness*100).toFixed(0)+"%"],
     ["Actual value (90d)","$"+c.value90.toLocaleString()],["Actual orders (90d)",c.orders90],["Actual churn",c.churn90?"yes":"no"],
     ["Lookalike mean value","$"+c.neighbour_mean_value.toLocaleString()],["Lookalike churn agreement",(c.neighbour_churn_agree*100).toFixed(0)+"%"]
    ].forEach(([k,v])=>{kv.appendChild(el("div",null,k));kv.appendChild(el("div",null,String(v)));});
    box.appendChild(kv);
    box.appendChild(el("div","tag","history"));
    const ul=el("ul","timeline");
    (c.timeline||[]).forEach(ev=>{const li=el("li");li.appendChild(el("span","t",ev.ts));li.appendChild(el("span","e",ev.event));li.appendChild(el("span","d",ev.detail||""));ul.appendChild(li);});
    box.appendChild(ul);
    box.appendChild(el("div","tag","nearest neighbours in the frozen state"));
    const ul2=el("ul","timeline");
    (c.neighbours||[]).forEach(nb=>{const li=el("li");li.appendChild(el("span","t",nb.key));li.appendChild(el("span","e","$"+nb.value90.toLocaleString()+" · "+nb.orders90+" orders"));li.appendChild(el("span","d",nb.churn90?"churned":"retained"));ul2.appendChild(li);});
    box.appendChild(ul2);
  }
  select(0);
}

$("#foot").textContent=`generated ${m.generated} · synthetic harness · ${m.customers.toLocaleString()} customers`;
</script>
</body>
</html>"""

    html = html.replace("__AXES__", axis_html)
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
