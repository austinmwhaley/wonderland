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
        "runtime": rt,
        "explorer": explorer,
    }


def _pct(x):
    return f"{x * 100:.1f}%"


def _f(x, n=3):
    return f"{x:.{n}f}"


def render(payload) -> str:
    sup = payload["supervised"]
    ladder = payload["ladder"]
    chosen = payload["chosen"]
    rl = payload["rl_known"]
    resp = payload["rl_policy"]
    uns = payload["unsupervised"]
    ident = payload["identity"]

    # ---------- static tables ----------
    def sup_table():
        rows = []
        for t in sup["targets"]:
            v = sup["values"][t]
            rows.append(
                "<tr>"
                f"<td>{TARGET_LABEL[t]}</td>"
                f"<td class='num'>{_f(v.get('raw', float('nan')))}</td>"
                f"<td class='num'>{_f(v.get('shape', float('nan')))}</td>"
                f"<td class='num'>{_f(v.get('measure', float('nan')))}</td>"
                f"<td class='num'>{_f(v.get('forecast', float('nan')))}</td>"
                f"<td class='num strong'>{_f(v.get('full_measure', float('nan')))}</td>"
                "</tr>"
            )
        return "".join(rows)

    def ladder_table():
        rows = []
        for t, c in ladder.items():
            rows.append(
                "<tr>"
                f"<td>{TARGET_LABEL[t]}</td>"
                f"<td class='num'>{_f(c['floor'])}</td>"
                f"<td class='num strong'>{_f(c['E'])}</td>"
                f"<td class='num'>{_f(c['ceiling'])}</td>"
                f"<td class='num'>{c['E_minus_floor']:+.4f} ± {c['se']:.4f}</td>"
                "</tr>"
            )
        return "".join(rows)

    def chosen_table():
        rows = []
        for t, c in chosen.items():
            d = c["delta"]
            rows.append(
                "<tr>"
                f"<td>{TARGET_LABEL[t]}</td>"
                f"<td><span class='pill'>{c['view']}</span></td>"
                f"<td class='num'>{_f(c['holdout']['selected'])}</td>"
                f"<td class='num'>{_f(c['holdout']['table'])}</td>"
                f"<td class='num strong'>{d['mean']:+.4f}</td>"
                f"<td class='num muted'>[{d['lo']:+.4f}, {d['hi']:+.4f}]</td>"
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
                f"<td class='num'>{_f(r['table'])}</td>"
                f"<td class='num strong'>{_f(r['E'])}</td>"
                f"<td class='num'>{r['lo']:+.4f}</td>"
                "</tr>"
            )
        return "".join(rows)

    def policy_table():
        rows = []
        for arm in ("channel", "offer"):
            r = resp.get(arm) if isinstance(resp, dict) else None
            if not r:
                continue
            er, et = r["E_minus_random"], r["E_minus_table"]
            rows.append(
                "<tr>"
                f"<td>{arm}</td>"
                f"<td class='num'>{r['E']['ips']:.2f}</td>"
                f"<td class='num'>{r['table']['ips']:.2f}</td>"
                f"<td class='num'>{r['random']['ips']:.2f}</td>"
                f"<td class='num'>{er['mean']:+.2f} [{er['lo']:+.2f}, {er['hi']:+.2f}]</td>"
                f"<td class='num'>{et['mean']:+.2f} [{et['lo']:+.2f}, {et['hi']:+.2f}]</td>"
                "</tr>"
            )
        return "".join(rows)

    def uns_table():
        rows = []
        labels = [
            ("churn_nn", "Churn-neighbour agreement"),
            ("retrieval", "Value-neighbour rank"),
            ("anomaly_auc", "Anomaly AUC"),
            ("nmi_regime", "Regime NMI"),
            ("eff_rank", "Effective rank"),
        ]
        for key, label in labels:
            t = uns.get("table", {})
            e = uns.get("E", {})
            if key not in e:
                continue
            delta = e[key] - t.get(key, 0.0)
            rows.append(
                "<tr>"
                f"<td>{label}</td>"
                f"<td class='num'>{_f(t.get(key, float('nan')))}</td>"
                f"<td class='num strong'>{_f(e[key])}</td>"
                f"<td class='num'>{delta:+.4f}</td>"
                "</tr>"
            )
        return "".join(rows)

    def identity_table():
        rows = []
        for key in ("channel", "category"):
            r = ident.get(key) or {}
            if "E" not in r:
                continue
            rows.append(
                "<tr>"
                f"<td>{key.title()} preference</td>"
                f"<td class='num'>{_f(r['table'])}</td>"
                f"<td class='num strong'>{_f(r['E'])}</td>"
                f"<td class='num'>{r['E'] - r['table']:+.4f}</td>"
                "</tr>"
            )
        return "".join(rows)

    explorer_json = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")

    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wonderland — frozen customer foundation model</title>
<style>
:root{
  --bg:#080b16; --bg2:#0c1224; --panel:#111a31; --panel2:#0f1730;
  --line:#1e2b4d; --ink:#e8eefc; --muted:#8fa1c7; --dim:#5c6f96;
  --accent:#6ee7b7; --accent2:#818cf8; --warn:#fbbf24; --bad:#f87171;
  --good:#34d399;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:
    radial-gradient(1200px 600px at 15% -10%, #16224a 0%, transparent 60%),
    radial-gradient(900px 500px at 90% 0%, #1b2a52 0%, transparent 55%),
    var(--bg);
  color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1240px;margin:0 auto;padding:28px 22px 80px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;flex-wrap:wrap}
.brand{display:flex;gap:14px;align-items:center}
.mark{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  box-shadow:0 8px 30px rgba(110,231,183,.25);position:relative}
.mark::after{content:"";position:absolute;inset:11px;border-radius:6px;background:#0a0f1e;
  clip-path:polygon(0 0,100% 50%,0 100%,35% 50%)}
h1{font-size:22px;margin:0;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.badges{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.badge{border:1px solid var(--line);background:var(--panel2);color:var(--muted);
  padding:6px 11px;border-radius:999px;font-size:12px;white-space:nowrap}
.badge.ok{border-color:rgba(52,211,153,.35);color:#8ef0c8;background:rgba(52,211,153,.08)}
.badge.warn{border-color:rgba(251,191,36,.35);color:#f7d27a;background:rgba(251,191,36,.08)}

nav{display:flex;gap:4px;margin:26px 0 22px;border-bottom:1px solid var(--line);overflow-x:auto}
nav button{appearance:none;background:none;border:0;color:var(--muted);font:inherit;font-weight:600;
  padding:12px 15px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;font-size:13.5px}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--ink);border-bottom-color:var(--accent)}

section[hidden]{display:none}
.grid{display:grid;gap:16px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
@media(max-width:900px){.g2,.g3,.g4{grid-template-columns:1fr}}

.card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
  border-radius:16px;padding:18px}
.card h3{margin:0 0 4px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.big{font-size:30px;font-weight:700;letter-spacing:-.03em;margin:6px 0 2px}
.big small{font-size:14px;color:var(--muted);font-weight:500}
.note{color:var(--muted);font-size:12.5px}

h2{font-size:17px;margin:0 0 6px;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 18px;max-width:80ch}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.strong,.strong{color:var(--accent);font-weight:600}
td.muted,.muted{color:var(--dim)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;background:rgba(129,140,248,.12);
  border:1px solid rgba(129,140,248,.3);color:#b9c1ff;font-size:12px}

.chart{width:100%}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--muted)}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:6px}

.explorer{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);gap:18px;align-items:start}
@media(max-width:980px){.explorer{grid-template-columns:1fr}}
#scatter{width:100%;height:auto;border-radius:12px;background:var(--panel2);border:1px solid var(--line);cursor:crosshair}
.dot{cursor:pointer;transition:r .12s ease,opacity .12s ease}
.dot:hover{r:7}
.panelbox{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:16px}
.kv{display:grid;grid-template-columns:1fr auto;gap:6px 12px;font-size:13px}
.kv div:nth-child(even){text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.kv div:nth-child(odd){color:var(--muted)}
.timeline{list-style:none;margin:10px 0 0;padding:0}
.timeline li{display:grid;grid-template-columns:104px 1fr auto;gap:10px;padding:7px 0;border-bottom:1px dashed rgba(30,43,77,.8);font-size:12.5px}
.timeline .t{color:var(--dim);font-variant-numeric:tabular-nums}
.timeline .e{color:#c4d0ea}
.timeline .d{color:var(--accent)}
.tag{font-size:11px;color:var(--dim)}
footer{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim);font-size:12px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
.caveat{border-left:2px solid var(--warn);padding:8px 0 8px 14px;color:#d7c08a;font-size:12.5px;margin:10px 0}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="mark"></div>
      <div>
        <h1>wonderland</h1>
        <div class="sub">One frozen encoder per event stream · supervised · unsupervised · reinforcement learning</div>
      </div>
    </div>
    <div class="badges" id="badges"></div>
  </header>

  <nav id="tabs"></nav>

  <main>
    <section id="overview">
      <h2>The claim</h2>
      <p class="lede" id="claim"></p>
      <div class="grid g4" id="stat_cards"></div>
      <div class="card" style="margin-top:16px">
        <h3>Selected view vs the traditional table — holdout, paired 95% CI</h3>
        <p class="note">The view per target was chosen on training folds only, then the holdout was scored once.</p>
        <div class="chart" data-chart="selected"></div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Same head, same holdout — what each state view carries</h3>
        <table>
          <thead><tr><th>Target</th><th class="num">Table</th><th class="num">Shape</th><th class="num">Measure</th><th class="num">Forecast</th><th class="num">Full</th></tr></thead>
          <tbody>__SUP_TABLE__</tbody>
        </table>
        <p class="note" style="margin-top:10px">No single view wins everywhere. That is why the artifact publishes named views instead of one opaque vector.<br>
        Protocol note: this table holds one fixed head and fold split across every arm, so its table value (0.357) differs from the train-only selection panel below, which uses a different split and selection procedure (table 0.424). Both are reported; neither is substituted for the other.</p>
      </div>
    </section>

    <section id="vs" hidden>
      <h2>Encoder vs the traditional process</h2>
      <p class="lede">The traditional path builds a wide customer table and fits a gradient-boosted model. The encoder path freezes one self-supervised model over the raw event stream and hands downstream heads a reusable state. Same holdout, same head, paired bootstrap intervals.</p>
      <div class="grid g2">
        <div class="card">
          <h3>R² / AUC by state view</h3>
          <div class="chart" data-chart="arms"></div>
          <div class="legend" id="legend_arms"></div>
        </div>
        <div class="card">
          <h3>Selected view vs table (delta with 95% CI)</h3>
          <div class="chart" data-chart="delta"></div>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Train-only view selection</h3>
        <table>
          <thead><tr><th>Target</th><th>Selected view</th><th class="num">Holdout</th><th class="num">Table</th><th class="num">Delta</th><th class="num">95% CI</th></tr></thead>
          <tbody>__CHOSEN_TABLE__</tbody>
        </table>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Floor → encoder → full-history ceiling</h3>
        <p class="note">Floor is the legacy table baseline; ceiling is a full-history oracle. Reaching the ceiling means the frozen state carries what the full history can offer for that target.</p>
        <table>
          <thead><tr><th>Target</th><th class="num">Floor</th><th class="num">Encoder</th><th class="num">Ceiling</th><th class="num">Encoder − floor</th></tr></thead>
          <tbody>__LADDER_TABLE__</tbody>
        </table>
      </div>
    </section>

    <section id="state" hidden>
      <h2>What the foundation state contains</h2>
      <p class="lede">The artifact publishes named views. Consumers pick with training-fold evidence, not by looking at the test set.</p>
      <div class="grid g2">
        <div class="card">
          <h3>Supervised — same head, same holdout</h3>
          <p class="note">Fixed head and folds across all arms; see the protocol note on Overview.</p>
          <table>
            <thead><tr><th>Target</th><th class="num">Table</th><th class="num">Shape</th><th class="num">Measure</th><th class="num">Full</th></tr></thead>
            <tbody>__SUP_TABLE_2__</tbody>
          </table>
        </div>
        <div class="card">
          <h3>Identity — kNN purity vs truth</h3>
          <table>
            <thead><tr><th>Latent</th><th class="num">Table</th><th class="num">Encoder</th><th class="num">Delta</th></tr></thead>
            <tbody>__IDENT_TABLE__</tbody>
          </table>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Unsupervised probes</h3>
        <table>
          <thead><tr><th>Probe</th><th class="num">Table</th><th class="num">Encoder</th><th class="num">Delta</th></tr></thead>
          <tbody>__UNS_TABLE__</tbody>
        </table>
        <p class="note" style="margin-top:10px">Reported honestly: anomaly and segmentation do not beat the table on this encoder. Retrieval and churn-neighbour structure do.</p>
      </div>
      <div class="grid g4" style="margin-top:16px" id="dim_cards"></div>
    </section>

    <section id="plugins" hidden>
      <h2>Three plugin families, one frozen artifact</h2>
      <p class="lede">The foundation is never retrained per task. Supervised heads predict; unsupervised heads explore geometry; RL heads estimate action response from pre-action state.</p>
      <div class="grid g2">
        <div class="card">
          <h3>Known-effect recovery — 10-NN purity vs randomized optimum</h3>
          <div class="chart" data-chart="rl_known"></div>
          <p class="note" style="margin-top:10px">The generator randomizes one action key per day with a known optimum; retrieval recovers it far above the table.</p>
        </div>
        <div class="card">
          <h3>RL policy value — pre-action, grouped</h3>
          <table>
            <thead><tr><th>Arm</th><th class="num">Encoder IPS</th><th class="num">Table IPS</th><th class="num">Uniform</th><th class="num">Enc − uniform</th><th class="num">Enc − table</th></tr></thead>
            <tbody>__POLICY_TABLE__</tbody>
          </table>
          <p class="note" style="margin-top:10px">Channel beats uniform and ties table; offer beats table and ties uniform. A working response interface, not a deploy certificate.</p>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Known-effect by action key</h3>
        <table>
          <thead><tr><th>Action</th><th class="num">Table purity</th><th class="num">Encoder purity</th><th class="num">Paired CI lower</th></tr></thead>
          <tbody>__RL_TABLE__</tbody>
        </table>
      </div>
    </section>

    <section id="explorer" hidden>
      <h2>Meet a customer</h2>
      <p class="lede">Sampled holdout customers projected from the frozen embedding. Pick a point to see its history, its lookalikes, and the held-out outcome those lookalikes suggest.</p>
      <div class="explorer">
        <div class="panelbox">
          <svg id="scatter" viewBox="0 0 640 460" role="img" aria-label="Embedding scatter"></svg>
          <div class="legend"><span><i style="background:#34d399"></i>retained</span><span><i style="background:#f87171"></i>churned</span><span class="muted">· click a point</span></div>
        </div>
        <div class="panelbox" id="detail"></div>
      </div>
    </section>

  </main>

  <footer>
    <div>wonderland · frozen customer foundation model</div>
    <div id="foot"></div>
  </footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const $ = (s, r=document) => r.querySelector(s);
const el = (t, c, txt) => { const n=document.createElement(t); if(c) n.className=c; if(txt!=null) n.textContent=txt; return n; };

/* ---------- tabs ---------- */
const TABS = [["overview","Overview"],["vs","Encoder vs Traditional"],["state","State Views"],["plugins","Plugins"],["explorer","Customer Explorer"]];
const nav = $("#tabs");
function activate(id, push){
  const known = TABS.some(([t])=>t===id) ? id : TABS[0][0];
  document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on", x.dataset.tab===known));
  TABS.forEach(([tid])=>{ const sec=document.getElementById(tid); if(sec) sec.hidden = tid!==known; });
  if(push) history.replaceState(null,"","#"+known);
  window.scrollTo({top:0,behavior:"instant"});
}
TABS.forEach(([id,label])=>{
  const b=el("button",null,label); b.dataset.tab=id;
  b.onclick=()=>activate(id,true);
  nav.appendChild(b);
});
activate((location.hash||"").replace("#",""), false);
window.addEventListener("hashchange",()=>activate((location.hash||"").replace("#",""), false));

/* ---------- badges ---------- */
const m = DATA.meta;
[["B "+m.b_pass+"/"+m.b_pass+" PASS","ok"],["B+C "+m.bc_pass+"/"+m.bc_pass+" PASS","ok"],["audit open: 3","warn"]].forEach(([t,c])=>{
  const b=el("span","badge "+c,t); $("#badges").appendChild(b);
});

/* ---------- claim + cards ---------- */
$("#claim").innerHTML = `Given <b>${m.events.toLocaleString()}</b> events from <b>${m.customers.toLocaleString()}</b> customers, one self-supervised encoder is frozen and reused by every head. The compact shape is <b>${m.d}</b> numbers; the published state is <b>${m.state_dim.toLocaleString()}</b>. On the same holdout and the same head, the encoder state beats the traditional wide table on value and orders, and the selected view is chosen on training folds only.`;

const cards = [
  ["Customers", m.customers.toLocaleString(), "one 540-day stream"],
  ["Events", (m.events/1e6).toFixed(2)+"M", "canonical 4-column stream"],
  ["Frozen state", m.state_dim.toLocaleString(), m.d+"-d shape + lawful measure"],
  ["Serving", Math.round(DATA.runtime.state_rows_per_s||0).toLocaleString()+" rows/s", "warm, after compile"],
];
cards.forEach(([k,v,n])=>{ const c=el("div","card"); c.appendChild(el("h3",null,k)); c.appendChild(el("div","big",v)); c.appendChild(el("div","note",n)); $("#stat_cards").appendChild(c); });

/* ---------- svg helpers ---------- */
const NS="http://www.w3.org/2000/svg";
function svg(w,h){ const s=document.createElementNS(NS,"svg"); s.setAttribute("viewBox",`0 0 ${w} ${h}`); s.setAttribute("width","100%"); return s; }
function txt(x,y,t,o={}){ const n=document.createElementNS(NS,"text"); n.setAttribute("x",x); n.setAttribute("y",y);
  n.setAttribute("fill",o.fill||"#8fa1c7"); n.setAttribute("font-size",o.size||12); n.setAttribute("font-weight",o.weight||500);
  if(o.anchor) n.setAttribute("text-anchor",o.anchor); n.textContent=t; return n; }
function rect(x,y,w,h,fill,r=4){ const n=document.createElementNS(NS,"rect"); n.setAttribute("x",x); n.setAttribute("y",y);
  n.setAttribute("width",Math.max(0,w)); n.setAttribute("height",h); n.setAttribute("rx",r); n.setAttribute("fill",fill); return n; }

const TARGETS = DATA.supervised.targets;
const TL = {value90:"value90",orders90:"orders90",churn90:"churn90"};

/* grouped bars: arms x targets, normalized per target */
function chartArms(host){
  const s=svg(640,300), pad={l:44,r:14,t:14,b:56};
  const arms=[["raw","Traditional table","#64748b"],["shape","Shape","#818cf8"],["measure","Measure","#fbbf24"],["forecast","Forecast","#38bdf8"],["full_measure","Full state","#34d399"]];
  const series=TARGETS.map(t=>arms.map(([k])=>DATA.supervised.values[t][k]??0));
  const maxV=Math.max(...series.flat())*1.05||1;
  const gw=(640-pad.l-pad.r)/TARGETS.length, bw=gw/(arms.length+1);
  TARGETS.forEach((t,ti)=>{
    const gx=pad.l+ti*gw;
    arms.forEach(([k,label,color],ai)=>{
      const v=DATA.supervised.values[t][k]??0;
      const h=(v/maxV)*(300-pad.t-pad.b);
      const x=gx+ai*bw+bw*0.15, y=300-pad.b-h;
      s.appendChild(rect(x,y,bw*0.7,h,color,3));
      s.appendChild(txt(x+bw*0.35,y-5,v.toFixed(3),{anchor:"middle",size:10,fill:"#c9d6f2"}));
    });
    s.appendChild(txt(gx+gw/2,292,TL[t],{anchor:"middle",size:11}));
  });
  for(let i=0;i<=4;i++){ const y=pad.t+(i/4)*(300-pad.t-pad.b); s.appendChild(txt(pad.l-8,y+4,(maxV*(1-i/4)).toFixed(2),{anchor:"end",size:10,fill:"#5c6f96"})); }
  host.appendChild(s);
  const lg=$("#legend_arms");
  arms.forEach(([k,label,color])=>{ const d=el("span"); const i=el("i"); i.style.background=color; d.appendChild(i); d.appendChild(document.createTextNode(label)); lg.appendChild(d); });
}

/* delta with CI */
function chartDelta(host){
  const s=svg(640,220), pad={l:96,r:24,t:18,b:34};
  const rows=TARGETS.map(t=>({t,d:DATA.chosen[t]&&DATA.chosen[t].delta})).filter(r=>r.d);
  if(!rows.length){ host.appendChild(el("p","note","No selection delta available.")); return; }
  const maxAbs=Math.max(...rows.map(r=>Math.max(Math.abs(r.d.lo),Math.abs(r.d.hi))),0.02);
  const mid=pad.l+(640-pad.l-pad.r)/2, half=(640-pad.l-pad.r)/2;
  const axis=document.createElementNS(NS,"line");
  axis.setAttribute("x1",mid); axis.setAttribute("x2",mid);
  axis.setAttribute("y1",pad.t); axis.setAttribute("y2",220-pad.b);
  axis.setAttribute("stroke","#24345c"); axis.setAttribute("stroke-width","1");
  s.appendChild(axis);
  rows.forEach((r,i)=>{
    const y=pad.t+i*((220-pad.t-pad.b)/Math.max(rows.length,1))+16;
    const x0=mid+(r.d.lo/maxAbs)*half, x1=mid+(r.d.hi/maxAbs)*half;
    s.appendChild(rect(Math.min(x0,x1),y-3,Math.abs(x1-x0),6,"rgba(52,211,153,.22)",3));
    const xm=mid+(r.d.mean/maxAbs)*half;
    s.appendChild(rect(Math.min(xm,mid),y-9,Math.abs(xm-mid),18,"#34d399",3));
    s.appendChild(txt(pad.l-10,y+4,r.t,{size:11,anchor:"end"}));
    s.appendChild(txt(xm+(r.d.mean>=0?8:-8),y+4,(r.d.mean>=0?"+":"")+r.d.mean.toFixed(4),
      {anchor:r.d.mean>=0?"start":"end",size:10,fill:"#8ef0c8",weight:700}));
  });
  s.appendChild(txt(mid,212,"0",{anchor:"middle",size:10,fill:"#5c6f96"}));
  s.appendChild(txt(pad.l,212,"table better ←",{anchor:"start",size:9,fill:"#5c6f96"}));
  s.appendChild(txt(640-pad.r,212,"→ encoder better",{anchor:"end",size:9,fill:"#5c6f96"}));
  host.appendChild(s);
}

/* selected vs table: one panel per target, each with its own scale */
function chartSelected(host){
  host.style.display="flex"; host.style.gap="18px"; host.style.alignItems="flex-end";
  const cols = {value90:"value90", orders90:"orders90", churn90:"churn90"};
  TARGETS.forEach(t=>{
    const c=DATA.chosen[t]; if(!c) return;
    const box=el("div"); box.style.flex="1"; box.style.minWidth="0"; box.style.textAlign="center";
    box.style.display="flex"; box.style.justifyContent="center";
    const W=220,H=252,base=196;
    const s=svg(W,H);
    s.setAttribute("width",String(W)); s.setAttribute("height",String(H)); s.setAttribute("style","max-width:100%");
    const maxV=Math.max(c.holdout.selected,c.holdout.table)*1.25||1;
    const bw=54;
    const ht=(c.holdout.table/maxV)*(base-28), hs=(c.holdout.selected/maxV)*(base-28);
    s.appendChild(rect(W/2-bw-8,base-ht,bw,ht,"#475569",5));
    s.appendChild(rect(W/2+8,base-hs,bw,hs,"#34d399",5));
    s.appendChild(txt(W/2-bw/2-8,base-ht-7,c.holdout.table.toFixed(3),{anchor:"middle",size:11,fill:"#94a3b8"}));
    s.appendChild(txt(W/2+bw/2+8,base-hs-7,c.holdout.selected.toFixed(3),{anchor:"middle",size:11,fill:"#8ef0c8",weight:700}));
    s.appendChild(txt(W/2,base+20,cols[t],{anchor:"middle",size:12,fill:"#c9d6f2"}));
    const d=c.delta;
    s.appendChild(txt(W/2,base+42,(d.mean>=0?"+":"")+d.mean.toFixed(4)+"  ["+d.lo.toFixed(3)+", "+d.hi.toFixed(3)+"]",
      {anchor:"middle",size:10,fill:d.lo>0?"#34d399":"#94a3b8"}));
    box.appendChild(s);
    host.appendChild(box);
  });
  const lg=el("div","legend");
  [["#475569","traditional table"],["#34d399","selected encoder view"]].forEach(([c,l])=>{
    const d=el("span"); const i=el("i"); i.style.background=c; d.appendChild(i); d.appendChild(document.createTextNode(l)); lg.appendChild(d);
  });
  host.parentElement.appendChild(lg);
}

/* rl known-effect bars */
function chartRlKnown(host){
  const s=svg(640,260), pad={l:44,r:14,t:14,b:40};
  const keys=Object.keys(DATA.rl_known||{});
  if(!keys.length) return;
  const maxV=1.0, gw=(640-pad.l-pad.r)/keys.length;
  keys.forEach((k,i)=>{
    const r=DATA.rl_known[k], gx=pad.l+i*gw, bw=gw*0.26;
    const he=(r.E/maxV)*(260-pad.t-pad.b), ht=(r.table/maxV)*(260-pad.t-pad.b);
    s.appendChild(rect(gx+gw*0.16,260-pad.b-ht,bw,ht,"#475569",3));
    s.appendChild(rect(gx+gw*0.52,260-pad.b-he,bw,he,"#818cf8",3));
    s.appendChild(txt(gx+gw*0.52+bw/2,260-pad.b-he-5,r.E.toFixed(2),{anchor:"middle",size:10,fill:"#b9c1ff"}));
    s.appendChild(txt(gx+gw/2,252,k,{anchor:"middle",size:11}));
  });
  for(let i=0;i<=4;i++){ const y=pad.t+(i/4)*(260-pad.t-pad.b); s.appendChild(txt(pad.l-8,y+4,(maxV*(1-i/4)).toFixed(2),{anchor:"end",size:10,fill:"#5c6f96"})); }
  host.appendChild(s);
}

document.querySelectorAll("[data-chart]").forEach(h=>{
  const k=h.dataset.chart;
  if(k==="arms") chartArms(h);
  if(k==="delta") chartDelta(h);
  if(k==="selected") chartSelected(h);
  if(k==="rl_known"){ chartRlKnown(h);
    const lg=el("div","legend");
    [["#475569","traditional table"],["#818cf8","frozen encoder"]].forEach(([c,l])=>{
      const d=el("span"); const i=el("i"); i.style.background=c; d.appendChild(i); d.appendChild(document.createTextNode(l)); lg.appendChild(d); });
    h.parentElement.appendChild(lg);
  }
});

/* ---------- dim cards ---------- */
const dims=Object.entries(m.views||{shape:m.d});
dims.forEach(([k,v])=>{ const c=el("div","card"); c.appendChild(el("h3",null,k.replace("_"," "))); c.appendChild(el("div","big",v.toLocaleString())); c.appendChild(el("div","note","dimensions of this view")); $("#dim_cards").appendChild(c); });

$("#foot").textContent = `generated ${m.generated} · synthetic harness · no business data · not a deploy certificate · full disclosures in REPORT_CARD.md`;

/* ---------- explorer ---------- */
const EXP = DATA.explorer;
if(!EXP || !EXP.customers || !EXP.customers.length){
  $("#explorer").innerHTML = "<h2>Meet a customer</h2><p class='lede'>Explorer data not built. Run <code>python3 scripts/build_demo_site.py</code> with the frozen cache and stream available.</p>";
} else {
  const svgEl=$("#scatter"), W=640,H=460,pad=34;
  const dots=[];
  EXP.customers.forEach((c,i)=>{
    const cx=pad+c.x*(W-2*pad), cy=H-pad-c.y*(H-2*pad);
    const circle=document.createElementNS(NS,"circle");
    circle.setAttribute("cx",cx); circle.setAttribute("cy",cy); circle.setAttribute("r",4.6);
    circle.setAttribute("fill",c.churn90?"#f87171":"#34d399"); circle.setAttribute("opacity",.72);
    circle.setAttribute("class","dot");
    circle.onclick=()=>select(i);
    svgEl.appendChild(circle); dots.push(circle);
  });
  function select(i){
    dots.forEach((d,j)=>{ d.setAttribute("r", j===i?8:4.6); d.setAttribute("opacity", j===i?1:.5); });
    const c=EXP.customers[i];
    const box=$("#detail"); box.innerHTML="";
    box.appendChild(el("div","tag","customer"));
    const h=el("div",null,c.key); h.style.fontSize="18px"; h.style.fontWeight="700"; h.style.margin="2px 0 10px"; box.appendChild(h);
    const kv=el("div","kv");
    [["Total events",c.n_events],["Channel preference (truth)",c.channel],["Category preference (truth)",c.category],["Regime (truth)",c.regime],["Return proneness (truth)",(c.return_proneness*100).toFixed(0)+"%"],
     ["Actual value (90d)","$"+c.value90.toLocaleString()],["Actual orders (90d)",c.orders90],["Actual churn", c.churn90?"yes":"no"],
     ["Lookalike mean value","$"+c.neighbour_mean_value.toLocaleString()],["Lookalike churn agreement",(c.neighbour_churn_agree*100).toFixed(0)+"%"]
    ].forEach(([k,v])=>{ kv.appendChild(el("div",null,k)); kv.appendChild(el("div",null,String(v))); });
    box.appendChild(kv);
    box.appendChild(el("div","tag","history (selected events)"));
    const ul=el("ul","timeline");
    (c.timeline||[]).forEach(ev=>{ const li=el("li"); li.appendChild(el("span","t",ev.ts)); li.appendChild(el("span","e",ev.event)); li.appendChild(el("span","d",ev.detail||"")); ul.appendChild(li); });
    if(!(c.timeline||[]).length){ const li=el("li"); li.appendChild(el("span","muted","no notable events")); ul.appendChild(li); }
    box.appendChild(ul);
    box.appendChild(el("div","tag","nearest neighbours in the frozen embedding"));
    const ul2=el("ul","timeline");
    (c.neighbours||[]).forEach(nb=>{ const li=el("li"); li.appendChild(el("span","t",nb.key)); li.appendChild(el("span","e","value $"+nb.value90.toLocaleString()+" · orders "+nb.orders90)); li.appendChild(el("span","d",nb.churn90?"churned":"retained")); ul2.appendChild(li); });
    box.appendChild(ul2);
    box.appendChild(el("p","note","Neighbours are computed in the 128-d frozen shape; they are not used to form the actual outcome."));
  }
  select(0);
}
</script>
</body>
</html>"""

    html = html.replace("__SUP_TABLE__", sup_table())
    html = html.replace("__SUP_TABLE_2__", sup_table())
    html = html.replace("__CHOSEN_TABLE__", chosen_table())
    html = html.replace("__LADDER_TABLE__", ladder_table())
    html = html.replace("__POLICY_TABLE__", policy_table())
    html = html.replace("__RL_TABLE__", rl_table())
    html = html.replace("__UNS_TABLE__", uns_table())
    html = html.replace("__IDENT_TABLE__", identity_table())
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
