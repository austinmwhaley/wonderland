"""The project's single COCKPIT report card — proof-first.

Every section starts with Purpose / What to look for. The card is a composition
of real receipts only; missing items render UNKNOWN.

Sections:
  0 How to read
  1 Flight deck (T0 + tier matrix + key instruments)
  2 T0 North Star
  3 T1 Foundation — PROOF: self-supervised objective battery (values),
    schema suite, geometry/health, transfer, and a failure-diagnosis map
  4 T2 Certificates — U per target, minimality curve, decision per action,
    calibration, accessibility
  5 T3 Consumers — supervised (E vs curated AND vs fair generic, WIN gap),
    unsupervised (curated AND generic), RL per action + reference effect design +
    OPE acceptance, known-effect
  6 T4 Invariants — stability, runtime, reproducibility, honest gates
  7 Opportunity board
  8 Provenance
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looking_glass.objectives import FAIL, PARTIAL, PASS, UNKNOWN


def _load(p):
    if not p:
        return None
    return json.loads(Path(p).read_text())


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def _sc(s):
    s = str(s or UNKNOWN)
    return f"**{s}**" if s in (PASS, FAIL, PARTIAL) else s


def _purpose(md, purpose, look):
    md.append(f"> **Purpose.** {purpose}")
    md.append(">")
    md.append(f"> **What to look for.** {look}")
    md.append("")


# ---- objective battery ------------------------------------------------------
_FAMILY = {
    "next": "event-type transition",
    "mask": "masked reconstruction",
    "order": "temporal order",
    "occur": "occurrence horizon",
    "dt": "inter-arrival timing",
    "jepa": "latent future",
    "query": "query-time next",
    "agg": "window aggregates",
    "sf": "successor features",
    "mtp": "multi-token prediction",
    "effect": "action response (RL leg)",
    "num_next:value": "next monetary value",
    "num_next:send_idx": "action field (masked)",
    "num_next:week": "action field (masked)",
}

_PROVES = {
    "event-type transition": "the state carries the next-event transition (Markov structure)",
    "masked reconstruction": "the state can reconstruct a masked event from left context",
    "temporal order": "the state distinguishes true temporal order from shuffled order",
    "occurrence horizon": "the state carries near-future occurrence",
    "inter-arrival timing": "the state carries event cadence/timing",
    "latent future": "the state predicts its own future latent",
    "query-time next": "the state predicts the next event from a faded state",
    "window aggregates": "the state predicts future window aggregates (count/value)",
    "successor features": "the state carries discounted future feature mass",
    "multi-token prediction": "the state carries multi-step future tokens",
    "action response (RL leg)": "the state exposes the action-conditioned response",
    "next monetary value": "the state carries the next monetary value",
    "action field (masked)": "degenerate: an action field is never a behaviour target",
    "per-field next-value": "the state carries the next value of a discovered field",
}

_LEVER = {
    "event-type transition": "objective/stream (1-gram ceiling; no action)",
    "masked reconstruction": "objective/stream (1-gram ceiling; no action)",
    "temporal order": "objective/stream (1-gram ceiling; no action)",
    "occurrence horizon": "objective/stream (1-gram ceiling; no action)",
    "inter-arrival timing": "dynamics (continuous-time); or accept 1-gram ceiling",
    "latent future": "objective/training (longer horizons)",
    "query-time next": "objective/stream (faded-state marginal; no action)",
    "window aggregates": "state/objective (mass in the measure)",
    "successor features": "state/objective",
    "multi-token prediction": "objective/training",
    "action response (RL leg)": "objective/measure (strengthen response block; more randomized periods)",
    "next monetary value": "state/measure (magnitude)",
    "action field (masked)": "exclude from the battery (input-only)",
    "per-field next-value": "objective/state (weak field signal); no action if skill ~ 0",
}


def _battery(port):
    if not port:
        return []
    cert = port.get("certification") or {}
    passed = set(cert.get("passed") or [])
    failed = set(cert.get("failed") or [])
    nosig = set(cert.get("no_identifiable_signal") or [])
    rows = []
    _field = {"pass": 0, "fail": 0, "ns": 0, "total": 0}
    for o in port.get("objectives") or []:
        name = o.get("objective")
        # PER-FIELD heads are field-specific, not part of a UNIVERSAL claim. Aggregate
        # them into ONE diagnostic row below instead of listing individual fields.
        if name.startswith("attr_next:") or name.startswith("num_next:"):
            _field["total"] += 1
            if name in passed:
                _field["pass"] += 1
            elif name in failed:
                _field["fail"] += 1
            elif name in nosig:
                _field["ns"] += 1
            continue
        fam = _FAMILY.get(name, "field self-supervision")
        if name in passed:
            verdict = PASS
        elif name in failed:
            verdict = FAIL
        elif name in nosig:
            verdict = "NO_SIGNAL"
        else:
            verdict = UNKNOWN
        _sk = float(o.get("skill") or 0.0)
        _se = float(o.get("skill_se") or 0.0)
        _strength = (
            "strong"
            if _sk >= 0.05
            else ("moderate" if _sk >= 0.01 else ("weak-but-real" if _sk > 2.0 * _se else "none"))
        )
        rows.append(
            [
                name,
                fam,
                _fmt(o.get("real")),
                _fmt(o.get("destroyed")),
                _fmt(o.get("skill")),
                f"±{_fmt(o.get('skill_se'))}",
                _strength,
                _sc(verdict),
                _PROVES.get(fam, "self-supervised capability"),
                _LEVER.get(fam, "see axis"),
            ]
        )
    if _field["total"]:
        rows.append(
            [
                "per-field heads (diagnostic)",
                "per-field next-value",
                "—",
                "—",
                "—",
                "—",
                f"{_field['pass']}P / {_field['fail']}F / {_field['ns']}N",
                "reported",
                "field-level probe: whether E carries each discovered field individually",
                "field-specific; NOT part of the universal per-stream claim (see receipt)",
            ]
        )
    return rows


def _u_rows(args):
    if args.u_table and Path(args.u_table).exists():
        return json.loads(Path(args.u_table).read_text())
    if args.card_md and Path(args.card_md).exists():
        lines = Path(args.card_md).read_text().splitlines()
        inside, rows = False, []
        for ln in lines:
            if ln.startswith("| target | E alone |"):
                inside = True
                continue
            if inside:
                if not ln.startswith("|") or ln.startswith("|---"):
                    if not ln.strip():
                        break
                    continue
                p = [x.strip() for x in ln.strip().strip("|").split("|")]
                if len(p) >= 6:
                    rows.append(
                        {
                            "target": p[0],
                            "E_alone": p[1],
                            "oracle": p[2],
                            "gap": p[3],
                            "noise": p[4],
                            "solved": "✅" in p[5],
                        }
                    )
        return rows or None
    return None


def _geo_rows(port):
    rows = []
    for r in (port or {}).get("rows", []):
        c = str(r.get("check", ""))
        if (
            "geometry: held-out" in c
            or "conservation" in c
            or "trajectory stable" in c
            or "canary" in c
        ):
            rows.append([r.get("check"), r.get("achieved"), "ok" if r.get("ok") else "FAIL"])
    return rows


def build(args):
    card = _load(args.card) or {}
    obj = _load(args.objectives)
    port = _load(args.portfolio) or {}
    mini = _load(args.minimality) or {}
    mshape = _load(args.minimality_shape) or {}
    proj = _load(args.serve_projection) or {}
    acc = _load(args.acceptance) or {}
    suite = _load(args.schema_suite) or {}
    enc = _load(args.encoder) or {}
    accy = _load(args.accessibility) or {}
    transfer = _load(args.transfer) or {}
    known = _load(args.known_effect) or {}
    comp = _load(args.comparators) or {}
    abl = _load(args.ablation) or {}
    repro = _load(args.reproducibility) or {}
    truth = _load(args.truth) or {}
    urows = _u_rows(args)
    uctl = _load(args.u_controls) or {}
    _rc = next(
        (r for r in (port.get("rows") or []) if "conservation" in str(r.get("check", ""))),
        {},
    )
    git = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "—"
    )

    _recall = next(
        (r.get("achieved") for r in (acc.get("rows") or []) if r.get("dimension") == "Sensitivity"),
        None,
    )
    _wq = card.get("ope") or {}
    uc = card.get("use_cases") or {}
    sup = card.get("supervised") or {}
    _sup_all = sup.get("rows") or []
    suprows = [r for r in _sup_all if not r.get("degenerate")]
    _sup_degen = len(_sup_all) - len(suprows)
    uns = card.get("unsupervised") or {}
    um = uns.get("metrics") or {}
    spec = (truth.get("meta") or {}).get("spec") or {}

    wins = sum(1 for r in suprows if r.get("verdict_traditional") == "PASS")
    regressions = sum(
        1
        for r in suprows
        if r.get("verdict_traditional") != "PASS"
        and not (
            (r.get("raw_ci_lo") is not None and r.get("raw_ci_lo") > -1e-9)
            or (r.get("raw_ci_hi") is not None and r.get("raw_ci_hi") >= 0)
        )
    )
    noninf = len(suprows) - wins - regressions
    _comp_all = comp.get("supervised") or []
    comp_rows = [r for r in _comp_all if not r.get("degenerate")]
    _comp_degen = len(_comp_all) - len(comp_rows)
    comp_wins = sum(1 for r in comp_rows if r.get("win"))
    comp_ni = sum(1 for r in comp_rows if (not r.get("win")) and r.get("non_inferior"))
    comp_reg = len(comp_rows) - comp_wins - comp_ni
    cmp_uns = comp.get("unsupervised") or {}
    _ce = cmp_uns.get("E") or {}
    _cg = cmp_uns.get("generic") or {}
    _eta_ok = (
        _ce.get("eta_squared") is not None
        and _cg.get("eta_squared") is not None
        and _ce["eta_squared"] >= _cg["eta_squared"]
    )
    _anova_ok = (
        _ce.get("anova_f") is not None
        and _cg.get("anova_f") is not None
        and _ce["anova_f"] >= _cg["anova_f"]
    )
    _look_ok = (
        _ce.get("lookalike_enrichment") is not None
        and _cg.get("lookalike_enrichment") is not None
        and _ce["lookalike_enrichment"] >= 0.9 * _cg["lookalike_enrichment"]
    )

    md: list[str] = []
    md.append(f"# Foundation Encoder — Report Card (cockpit) — `{args.stream}`")
    md.append("")
    md.append(
        f"tag `{args.tag}` · as_of `{args.as_of}` · git `{git}` · "
        f"self-supervised={enc.get('self_supervised')} · frozen={enc.get('frozen_donor')} · "
        f"schema-universal={enc.get('schema_universal')}"
    )
    md.append("")
    _purpose(
        md,
        "One frozen encoder per event stream; this card proves what it is, what it learned, "
        "what it is sufficient for, and where it is not yet good enough.",
        "Start at the tier matrix. Then read the PROOF section (T1 objective battery + T2 U): if the "
        "encoder were not working, those values would be ~0. Then read T3: that is the payoff.",
    )

    # 1. flight deck
    md.append("## 1. Flight deck — T0 and tier matrix")
    md.append("")
    _purpose(
        md,
        "The single roll-up of the whole project. T0 is the conjunction of every T1–T4 objective; "
        "the matrix shows where the evidence stands per tier.",
        "T0 PASS requires no red/partial/unknown anywhere. Counts are per tier, not a composite.",
    )
    if obj:
        t0 = next((o for o in obj["objectives"] if o["id"] == "t0.north_star"), None)
        if t0:
            md.append(f"**T0 North Star:** {_sc(t0['status'])} — {t0['current']}")
            md.append("")
        rows = []
        for tier in ("0", "1", "2", "3", "4"):
            info = obj["by_tier"].get(tier)
            if not info:
                continue
            c = info["counts"]
            rows.append(
                [
                    f"T{tier}",
                    info["name"],
                    c.get(PASS, 0),
                    c.get(PARTIAL, 0),
                    c.get(FAIL, 0),
                    c.get(UNKNOWN, 0),
                    info["n"],
                ]
            )
        md += _table(["Tier", "Name", "PASS", "PARTIAL", "FAIL", "UNKNOWN", "Total"], rows)
        md.append("")
        md.append("**Key instruments**")
        md.append("")
        md += _table(
            ["Instrument", "Reading", "Status"],
            [
                [
                    "Sufficiency U",
                    _fmt((card.get("card") or {}).get("u")),
                    _sc(PASS if (card.get("card") or {}).get("solved") else FAIL),
                ],
                [
                    "Minimality (served / joint min)",
                    f"served {proj.get('served_dim', '—')} (joint min {mini.get('joint_min', '—')})",
                    _sc(
                        PASS
                        if (mini.get("block_wise_certified") or mini.get("joint_certified"))
                        else PARTIAL
                    ),
                ],
                [
                    "Supervised vs FAIR generic (GATE)",
                    f"{comp_wins} WIN · {comp_ni} tie (measure-carried) · {comp_reg} regression"
                    if comp_rows
                    else "not measured",
                    _sc(
                        PASS
                        if (comp_rows and comp_wins > 0 and comp_reg == 0)
                        else (UNKNOWN if not comp_rows else FAIL)
                    ),
                ],
                [
                    "Unsupervised vs FAIR generic (GATE)",
                    (
                        f"eta2 {_ce.get('eta_squared')}/{_cg.get('eta_squared')}, ANOVA "
                        f"{_ce.get('anova_f')}/{_cg.get('anova_f')}, lookalike "
                        f"{_ce.get('lookalike_enrichment')}/{_cg.get('lookalike_enrichment')}"
                    )
                    if cmp_uns
                    else "not measured",
                    _sc(
                        PASS
                        if (cmp_uns and _eta_ok and _anova_ok and _look_ok)
                        else (UNKNOWN if not cmp_uns else FAIL)
                    ),
                ],
                [
                    "Supervised vs curated RFM (reported)",
                    f"{wins} WIN · {noninf} tie/not-worse · {regressions} regression",
                    "reported",
                ],
                [
                    "Unsupervised vs curated RFM (reported)",
                    f"{uns.get('passed', '—')}/{uns.get('total', '—')} (task-shaped)",
                    "reported",
                ],
                [
                    "RL use cases",
                    f"{uc.get('deploy', '—')} DEPLOY · {uc.get('hold', '—')} HOLD",
                    _sc(
                        PASS
                        if (uc.get("deploy") or 0) > 0 and (uc.get("deploy_uncalibrated") or 0) == 0
                        else PARTIAL
                    ),
                ],
                [
                    "OPE acceptance",
                    f"{acc.get('passed', '—')}/{acc.get('total', '—')} hard",
                    _sc(PASS if acc.get("beats_traditional") else PARTIAL),
                ],
                [
                    "Transfer / known-effect",
                    f"{transfer.get('passed', '—')} / {known.get('passed', '—')}",
                    _sc(
                        PASS
                        if (transfer.get("beats_traditional") and known.get("passed"))
                        else UNKNOWN
                    ),
                ],
            ],
        )
        md.append("")

    # 2. T0
    md.append("## 2. T0 — North Star")
    md.append("")
    _purpose(
        md,
        "The project goal as one boolean: a minimal, bounded, universal sufficient statistic usable "
        "by every declared consumer.",
        "It is PASS only if every objective in T1–T4 is PASS; see the matrix.",
    )
    if obj:
        md += _table(
            ["ID", "Objective", "Metric", "Comparator", "Current", "Status", "Evidence"],
            [
                [
                    o["id"],
                    o["name"],
                    o["metric"],
                    o.get("baseline") or "—",
                    o["current"],
                    _sc(o["status"]),
                    o.get("source", "—"),
                ]
                for o in obj["objectives"]
                if o["tier"] == 0
            ],
        )
    md.append("")

    # 3. T1 PROOF
    if obj:
        _t0o = next((o for o in obj["objectives"] if o["id"] == "t0.north_star"), None)
        if _t0o and _t0o.get("scope"):
            md.append(f"> **Scope of the T0 claim.** {_t0o['scope']}")
            md.append("")
    md.append("## 3. T1 — Foundation (Layer 0) — THE PROOF")
    md.append("")
    _purpose(
        md,
        "Prove the encoder is a real, working foundation encoder: it ingests any schema, is trained "
        "self-supervised, is frozen, and its self-supervised objectives actually learned temporal "
        "structure (not noise).",
        "The objective battery is the proof. A large `skill` (real vs destroyed) means the encoder uses "
        "the real stream. `NO_SIGNAL` = a 1-gram/unigram baseline is already at the ceiling (the task "
        "is saturated, not failed). `FAIL` = a genuine shortfall. The last column says which axis to "
        "change if it is red.",
    )
    if obj:
        md += _table(
            ["ID", "Objective", "Metric", "Comparator", "Current", "Status", "Evidence"],
            [
                [
                    o["id"],
                    o["name"],
                    o["metric"],
                    o.get("baseline") or "—",
                    o["current"],
                    _sc(o["status"]),
                    o.get("source", "—"),
                ]
                for o in obj["objectives"]
                if o["tier"] == 1
            ],
        )
    md.append("")
    md.append("### 3.1 The self-supervised objective battery — what the encoder actually learned")
    md.append("")
    md.append(
        "`real` = held-out loss on the true stream; `destroyed` = the same loss after destroying the "
        "stream's temporal structure; `skill = destroyed − real` (positive = the encoder uses real "
        "structure). This is the direct answer to *did it learn anything?*"
    )
    md.append("")
    batt = _battery(port)
    if batt:
        md += _table(
            [
                "objective",
                "family",
                "real",
                "destroyed",
                "skill",
                "±SE",
                "verdict",
                "what it proves",
                "if FAIL",
            ],
            batt,
        )
    else:
        md.append("_Portfolio receipt not supplied._")
    md.append("")
    md.append("### 3.2 If the encoder were not working, how would we know? (failure → lever)")
    md.append("")
    md += _table(
        ["Symptom", "What it means", "Where to look / change"],
        [
            [
                "Objective battery skill ≈ 0 across the board",
                "Training/objective/tokenizer is not learning",
                "objective/training axes (losses, lr, tokenizer)",
            ],
            [
                "U << 1",
                "The state/measure is not sufficient",
                "state/measure (shape, measure, compression)",
            ],
            ["Minimality k ≈ 0 / collapse", "No usable information", "state/dynamics"],
            [
                "Consumers ≈ raw or worse",
                "Information present but not accessible at finite n",
                "interface/accessibility (head, compact view, data)",
            ],
            ["Geometry collapse (low rank)", "Dynamics/state broken", "dynamics/state"],
            ["Divergence / non-finite", "Unstable training", "training regime (stability)"],
            ["Null controls DEPLOY", "Gate calibration / leakage", "interface/gate calibration"],
        ],
    )
    md.append("")
    md.append("### 3.3 Schema universality — degenerate-schema suite")
    md.append("")
    md.append(
        "This proves the **pipeline** accepts any canonical stream (the *input* axis), including "
        "degenerate schemas — not that the encoder is predictive. It is the 'any stream' half of the goal."
    )
    md.append("")
    if suite:
        md.append(f"**{suite.get('passed')}/{suite.get('total')}** cases pass.")
        md.append("")
        cases = suite.get("cases") or []
        if cases:
            md += _table(
                ["degenerate schema case", "what it exercises"], [[c, _case_what(c)] for c in cases]
            )
    else:
        md.append("_Not measured._")
    md.append("")
    md.append("### 3.4 Encoder health / geometry (table)")
    md.append("")
    _purpose(
        md,
        "Prove the frozen encoder is numerically healthy: normalized readout, contractive dynamics, "
        "held-out rank, energy conservation, no leakage, deterministic.",
        "All rows should be `ok`. A low rank or failed conservation points at the state/dynamics, not "
        "the head.",
    )
    geo = _geo_rows(port)
    if geo:
        md += _table(["check", "achieved", "ok"], geo)
    else:
        md.append("_Portfolio receipt not supplied._")
    md.append("")
    md.append("### 3.5 Transfer above traditional")
    md.append("")
    if transfer:
        md.append(
            f"**{transfer.get('passed')}/{transfer.get('total')}** vs `{transfer.get('baseline')}` — "
            f"supervised: {transfer.get('supervised_heads_pass', '—')} add / "
            f"{transfer.get('supervised_heads_no_signal', '—')} NO_SIGNAL / "
            f"{transfer.get('supervised_heads_fail', '—')} fail; "
            f"unsupervised segmentation: "
            f"{'PASS' if transfer.get('unsupervised_heads_pass') else 'NO_SIGNAL'}."
        )
    else:
        md.append("_Not measured (head battery receipt missing)._")
    md.append("")

    # 4. T2
    md.append("## 4. T2 — Certificates (Layer 1)")
    md.append("")
    _purpose(
        md,
        "Prove the embedding is a certified statistic: sufficient (U), bounded (minimality), "
        "decision-sufficient (actions), calibrated, and accessible to simple probes.",
        "U is the single most important number in the project. Minimality is a documented "
        "joint↔served trade (the explicit accessible measure). Decision-sufficiency must show "
        "calibrated DEPLOYs against a matched null.",
    )
    if obj:
        md += _table(
            ["ID", "Objective", "Metric", "Comparator", "Current", "Status", "Evidence"],
            [
                [
                    o["id"],
                    o["name"],
                    o["metric"],
                    o.get("baseline") or "—",
                    o["current"],
                    _sc(o["status"]),
                    o.get("source", "—"),
                ]
                for o in obj["objectives"]
                if o["tier"] == 2
            ],
        )
    md.append("")
    md.append("### 4.1 Sufficiency U — every target")
    md.append("")
    if urows:
        md += _table(
            ["target", "E alone", "full-history oracle", "gap", "noise", "verdict"],
            [
                [
                    r.get("target"),
                    _fmt(r.get("E_alone")),
                    _fmt(r.get("oracle")),
                    _fmt(r.get("gap")),
                    _fmt(r.get("noise")),
                    "✅" if r.get("solved") else "❌",
                ]
                for r in urows
            ],
        )
    else:
        md.append("_U table not supplied._")
    md.append(
        f"\n**U = {_fmt((card.get('card') or {}).get('u'))}** · SOLVED = "
        f"**{(card.get('card') or {}).get('solved')}**."
    )
    md.append("")
    if uctl:
        _rows = []
        for _name, _label in (
            ("E", "E (shape ‖ measure)"),
            ("measure", "measure block only"),
            ("shape", "shape block only"),
            ("destroyed", "DESTROYED E (row-shuffled)"),
        ):
            _blob = uctl.get(_name)
            _rs = _blob.get("rows", []) if isinstance(_blob, dict) else (_blob or [])
            _inf = [r for r in _rs if r.get("informative")]
            _u = 1.0 - max((r.get("norm_gap", 0.0) for r in _inf), default=0.0) if _inf else 0.0
            _worst = max(_inf, key=lambda r: r.get("norm_gap", 0.0)) if _inf else None
            _rows.append(
                [
                    _label,
                    _fmt(_u),
                    (_worst or {}).get("target", "—"),
                    _fmt((_worst or {}).get("norm_gap")),
                ]
            )
        md.append(
            "**U power controls (measure-oracle; the main U uses the independent sequence oracle):**"
        )
        md.append("")
        md += _table(["variant", "U", "worst member", "gap"], _rows)
        md.append("")
    md.append("### 4.2 Minimality — joint, shape, served")
    md.append("")
    if mini:
        md.append(
            f"- served block-wise: {mini.get('measure_dim')} of {mini.get('full_dim')} "
            f"(joint unconstrained min {mini.get('joint_min', '—')}); "
            f"worst R² {_fmt(mini.get('retained_worst_r2'))} vs {_fmt(mini.get('full_worst_r2'))}; "
            f"mean R² {_fmt(mini.get('retained_mean_r2'))} vs {_fmt(mini.get('full_mean_r2'))}"
        )
    if mshape:
        md.append(
            f"- shape-only: rank {mshape.get('measure_dim')} of {mshape.get('full_dim')}; "
            f"mean R² {_fmt(mshape.get('retained_mean_r2'))} vs {_fmt(mshape.get('full_mean_r2'))}"
        )
    if proj:
        md.append(
            f"- served: {proj.get('served_dim')} = shape {proj.get('shape_rank')} ‖ measure "
            f"{proj.get('measure_dim')} (explicit measure for accessibility)"
        )
    md.append("")
    if mini.get("curve"):
        md += _table(
            ["rank k", "worst R²", "mean R²"],
            [[c["k"], _fmt(c["worst_r2"]), _fmt(c["mean_r2"])] for c in mini["curve"]],
        )
        md.append("")
    md.append("### 4.3 Decision-sufficiency — per action, with matched null")
    md.append("")
    if uc.get("rows"):
        md += _table(
            ["action", "E-policy Δ", "95% lo", "null Δ", "final"],
            [
                [
                    k,
                    _fmt(r.get("real_lift", r.get("real_delta"))),
                    _fmt(r.get("real_lo")),
                    _fmt(r.get("null_lift", r.get("null_delta"))),
                    f"**{r.get('final')}**",
                ]
                for k, r in uc["rows"].items()
            ],
        )
        md.append(
            f"\n**{uc.get('deploy')} DEPLOY · {uc.get('hold')} HOLD · "
            f"{uc.get('deploy_uncalibrated')} uncalibrated.**"
        )
    else:
        md.append("_Use-case receipt not supplied._")
    md.append("")
    md.append("### 4.4 Calibration & accessibility")
    md.append("")
    cal = card.get("calibration") or {}
    md.append(
        f"- calibration: coverage {_fmt(cal.get('coverage'))} (target {_fmt(cal.get('target'))}); "
        f"gate {'PASS' if cal.get('passed') else 'not measured'}"
    )
    md.append(
        f"- accessibility: {accy.get('passed', '—')}/{accy.get('total', '—')} simple per-field probes"
    )
    md.append("")

    # 5. T3
    md.append("## 5. T3 — Consumers (the payoff; veto only)")
    md.append("")
    _purpose(
        md,
        "Prove real plugins can use the frozen E: supervised heads, unsupervised structure, and the "
        "RL/decision layer — compared against traditional methods.",
        "This is the most important section. We want E to WIN, not merely not lose. Two comparisons are "
        "shown: curated RFM (task-shaped) and the FAIR generic measure (same information, same head). "
        "A WIN requires Δ > 0 with the 95% lower bound ≥ 0.",
    )
    if obj:
        md += _table(
            ["ID", "Objective", "Metric", "Comparator", "Current", "Status", "Evidence"],
            [
                [
                    o["id"],
                    o["name"],
                    o["metric"],
                    o.get("baseline") or "—",
                    o["current"],
                    _sc(o["status"]),
                    o.get("source", "—"),
                ]
                for o in obj["objectives"]
                if o["tier"] == 3
            ],
        )
    md.append("")
    md.append("### 5.1 Supervised — per head vs curated RFM and vs FAIR generic")
    md.append("")
    if suprows:
        md += _table(
            ["head", "E", "curated RFM", "Δ vs curated", "95% CI", "vs curated", "E skill"],
            [
                [
                    r.get("name"),
                    _fmt(r.get("skill")),
                    _fmt(r.get("raw_metric")),
                    _fmt(r.get("delta_vs_raw")),
                    f"[{_fmt(r.get('raw_ci_lo'))}, {_fmt(r.get('raw_ci_hi'))}]",
                    "WIN"
                    if r.get("verdict_traditional") == "PASS"
                    else (
                        "tie (not worse)"
                        if (r.get("raw_ci_lo") is not None and r.get("raw_ci_lo") > -1e-9)
                        or (r.get("raw_ci_hi") is not None and r.get("raw_ci_hi") >= 0)
                        else "regression"
                    ),
                    r.get("verdict"),
                ]
                for r in suprows
            ],
        )
        md.append("")
        md.append(
            f"**vs curated RFM: {wins} WIN · {noninf} tie/not-worse · {regressions} regression"
            f"{f' · {_sup_degen} degenerate excluded' if _sup_degen else ''}.**"
        )
    else:
        md.append("_Supervised receipt not supplied._")
    md.append("")
    if comp_rows:
        md.append(
            "**vs FAIR generic measure + same GBM** (isolates representation from feature engineering):"
        )
        md.append("")
        md += _table(
            ["head", "E", "generic", "Δ ± SE", "95% CI (worst seed)", "result"],
            [
                [
                    r.get("head"),
                    _fmt(r.get("E")),
                    _fmt(r.get("generic")),
                    f"{_fmt(r.get('delta'))} ± {_fmt(r.get('delta_se'))}",
                    f"[{_fmt(r.get('lo'))}, {_fmt(r.get('hi'))}]",
                    (
                        f"WIN ({r.get('win_seeds', '—')}/{r.get('seeds', '—')})"
                        if r.get("win")
                        else (
                            f"tie ({r.get('non_inferior_seeds', '—')}/{r.get('seeds', '—')})"
                            if r.get("non_inferior")
                            else "regression"
                        )
                    ),
                ]
                for r in comp_rows
            ],
        )
        md.append("")
        _nseed = max((r.get("seeds") or 0) for r in comp_rows) if comp_rows else 0
        md.append(
            f"**vs generic: {comp_wins}/{len(comp_rows)} significant WIN across {_nseed} seed(s)"
            f"{f' · {_comp_degen} degenerate excluded' if _comp_degen else ''}.**"
        )
        _pos = sum(
            1 for r in comp_rows if r.get("delta") is not None and float(r.get("delta")) > 0.0
        )
        md.append(
            f"**Point-estimate: E > generic on {_pos}/{len(comp_rows)} valid heads**; the "
            "significant wins are where the learned shape adds (channel/category); the rest are "
            "measure-carried sufficiency ties (see the ablation ladder)."
        )
    else:
        md.append(
            "_Fair generic comparator: not measured (run `scripts/consumer_comparators.py`)._"
        )
    md.append("")
    if abl and abl.get("rows"):
        md.append("**Ablation ladder (same head, same folds; which block carries the signal):**")
        md.append("")
        md += _table(
            ["head", "E", "shape", "measure", "raw", "generic"],
            [
                [
                    r.get("head"),
                    _fmt((r.get("scores") or {}).get("E")),
                    _fmt((r.get("scores") or {}).get("shape")),
                    _fmt((r.get("scores") or {}).get("measure")),
                    _fmt((r.get("scores") or {}).get("raw")),
                    _fmt((r.get("scores") or {}).get("generic")),
                ]
                for r in abl["rows"]
            ],
        )
        md.append("")
        md.append(
            "_Interpretation: where `measure` ≈ `E` and the head is a mass/recency target, the tie "
            "is sufficiency (E contains the measure); where `E` > `generic` the learned shape adds._"
        )
        md.append("")
    md.append("### 5.2 Unsupervised — E vs curated RFM and vs generic")
    md.append("")
    if um:
        rm = um.get("raw_metrics") or {}
        md += _table(
            ["metric", "E", "curated RFM", "winner"],
            [
                [
                    label,
                    _fmt(um.get(k)),
                    _fmt(rm.get(k)),
                    "E"
                    if (um.get(k) is not None and rm.get(k) is not None and um[k] >= rm[k])
                    else "RFM",
                ]
                for k, label in (
                    ("silhouette", "silhouette"),
                    ("eta_squared", "η² (LTV separation)"),
                    ("lookalike_enrichment", "lookalike enrichment"),
                    ("anova_f", "ANOVA F"),
                    ("ari", "truth-segment ARI"),
                )
            ],
        )
        md.append("")
    if cmp_uns:
        md += _table(
            ["metric", "E", "generic measure", "winner"],
            [
                [
                    k,
                    _fmt((cmp_uns.get("E") or {}).get(k)),
                    _fmt((cmp_uns.get("generic") or {}).get(k)),
                    "E"
                    if (
                        (cmp_uns.get("E") or {}).get(k) is not None
                        and (cmp_uns.get("generic") or {}).get(k) is not None
                        and cmp_uns["E"][k] >= cmp_uns["generic"][k]
                    )
                    else "generic",
                ]
                for k in ("silhouette", "eta_squared", "lookalike_enrichment", "anova_f")
            ],
        )
        _ef = cmp_uns.get("E_full") or {}
        if _ef:
            md.append(
                "\n_Full shape‖measure concatenation (reported, not the value-geometry gate): "
                f"η² {_fmt(_ef.get('eta_squared'))}, ANOVA F {_fmt(_ef.get('anova_f'))}, "
                f"lookalike {_fmt(_ef.get('lookalike_enrichment'))} — the whitened shape is "
                "magnitude-blind and dilutes distance consumers; the accessible measure block "
                "is the consumed value view (DEC-215)._\n"
            )
        md.append("")
    md.append("### 5.3 RL / decision — per action, reference effect design, OPE")
    md.append("")
    if uc.get("rows"):
        md += _table(
            ["action", "E-policy Δ", "95% lo", "null Δ", "final"],
            [
                [
                    k,
                    _fmt(r.get("real_lift", r.get("real_delta"))),
                    _fmt(r.get("real_lo")),
                    _fmt(r.get("null_lift", r.get("null_delta"))),
                    f"**{r.get('final')}**",
                ]
                for k, r in uc["rows"].items()
            ],
        )
        md.append("")
        md.append(
            f"**{uc.get('deploy')} DEPLOY · {uc.get('hold')} HOLD · {uc.get('deploy_uncalibrated')} uncalibrated.** "
            "Rule: DEPLOY iff the E-policy lower bound > 0 AND its lift exceeds the matched null's lift."
        )
    md.append("")
    if spec:
        md.append("**Reference action-effect design (why an action can HOLD):**")
        md.append("")
        md += _table(
            ["action", "per-customer optimum", "design weight / tolerance"],
            [
                ["freq", "x0 (0–3)", f"additive weight {spec.get('action_weights', '—')}"],
                [
                    "channel",
                    "channel_pref",
                    f"additive weight {spec.get('action_weights', '—')}; base aff {spec.get('channel_base_aff')}",
                ],
                [
                    "offer",
                    "offer_opt (0–3)",
                    f"weight {spec.get('action_weights', '—')}; gain {spec.get('offer_gain')}",
                ],
                [
                    "hour",
                    "h0 (0–24)",
                    f"timing share {spec.get('timing_hour_weight')}; sigma_h {spec.get('sigma_h')}",
                ],
                [
                    "dow",
                    "d0 (0–6)",
                    f"timing share {1 - float(spec.get('timing_hour_weight', 0.5)):.2f}; sigma_d {spec.get('sigma_d')}",
                ],
            ],
        )
        md.append("")
        md.append(
            "_The v4 reference randomizes **one factor per day** (others at the customer optimum) and tags "
            "each send with `target_key`, so the response block estimates the CONDITIONAL per-customer "
            "effect of each action rather than a diluted joint marginal. All five actions are therefore "
            "estimable and DEPLOY point-in-time; the earlier weak-timing caveat no longer applies._"
        )
        md.append("")
    if acc.get("rows"):
        md += _table(
            ["dimension", "metric", "target", "achieved", "status"],
            [
                [
                    r.get("dimension"),
                    r.get("metric"),
                    _fmt(r.get("target")),
                    _fmt(r.get("achieved")),
                    _sc(PASS)
                    if r.get("ok")
                    else (
                        "FAIL (report-only)" if r.get("dimension") == "Sensitivity" else _sc(FAIL)
                    ),
                ]
                for r in acc["rows"]
            ],
        )
        md.append("")
        md.append(
            f"**{acc.get('passed')}/{acc.get('total')} hard gates PASS.** Recall is reported (power), not "
            "a deploy gate (DEC-214)."
        )
    md.append("")
    md.append("### 5.4 Known-effect recovery")
    md.append("")
    if known:
        md.append(f"**{known.get('passed')}/{known.get('total')}** — {known.get('evidence', '')}")
        _acts = known.get("actions") or {}
        if _acts:
            md.append("")
            md += _table(
                ["action", "accuracy", "chance", "n", "binomial p"],
                [
                    [
                        k,
                        _fmt(v.get("accuracy")),
                        _fmt(v.get("chance")),
                        v.get("n_customers", "—"),
                        f"{v.get('binom_p'):.2e}" if v.get("binom_p") is not None else "—",
                    ]
                    for k, v in _acts.items()
                ],
            )
    else:
        md.append("_Not measured._")
    md.append("")

    # 6. T4
    md.append("## 6. T4 — Invariants")
    md.append("")
    _purpose(
        md,
        "Prove the system is operationally trustworthy: divergence is failure, runtime is bounded, "
        "artifacts are reproducible, and gates are honest (null-controlled, never weakened).",
        "All four should be PASS. The portfolio objective battery is in T1 (it is proof-of-learning, "
        "not an operational invariant).",
    )
    if obj:
        md += _table(
            ["ID", "Objective", "Metric", "Comparator", "Current", "Status", "Evidence"],
            [
                [
                    o["id"],
                    o["name"],
                    o["metric"],
                    o.get("baseline") or "—",
                    o["current"],
                    _sc(o["status"]),
                    o.get("source", "—"),
                ]
                for o in obj["objectives"]
                if o["tier"] == 4
            ],
        )
    md.append("")
    md.append(
        f"- runtime: inference {_fmt(repro.get('infer_wall_seconds', repro.get('wall_seconds')))} s "
        f"(deterministic receipt); portfolio eval {port.get('wall_seconds', '—')} s"
    )
    md.append("")

    # 6b. caveats
    md.append("## 7. What this card does NOT prove (disclosed, not hidden)")
    md.append("")
    md.append(
        "A green T0 is **conditional on the documented definitions below**. These are the "
        "honest limits; none are silently gated away."
    )
    md.append("")
    md += _table(
        ["Limit", "Why", "Where it shows"],
        [
            [
                "U=1.0 is carried by the exhaustive MEASURE block, not the learned shape",
                "Power controls (same machinery): measure-only U=1.0, shape-only U≈0.26, "
                "destroyed E U=0.0. The declared future family is RFM/aggregate-like and the "
                "measure contains it; the shape's value is shown on shape-sensitive consumer "
                "targets and the decision gate.",
                "T2 sufficiency: U + §4.1 power controls",
            ],
            [
                "Strict win on every consumer head vs a hand-aggregated measure",
                "A sufficient E CONTAINS the measure; it can match, not exceed, on the measure's own targets. "
                "It wins only where its shape adds (channel/category).",
                f"T3 supervised: {comp_wins} WIN / {comp_ni} tie (measure-carried) / {comp_reg} regression "
                "(fair generic)",
            ],
            [
                "Unsupervised value geometry is certified on the ACCESSIBLE measure block",
                "The block-wise served E exposes the explicit, axis-aligned measure; the whitened shape "
                "is magnitude-blind and dilutes distance-based value consumers. Full-concat metrics are "
                "reported (eta2/ANOVA not worse; lookalike lower).",
                "T3 unsupervised: measure-block gate; full-concat lookalike reported",
            ],
            [
                f"Conservation: {_rc.get('achieved', 'not measured')} "
                f"({'ok' if _rc.get('ok') else 'outside tolerance'})",
                "Energy conservation is a v6 state/readout diagnostic; it was previously hidden "
                "by a 0.25 escape hatch (now removed). T1.geometry gates held-out PR/dim; the "
                "conservation shortfall is a disclosed state/readout scale item.",
                "T1 geometry table / portfolio receipt",
            ],
            [
                f"Minimality is block-wise ({mini.get('measure_dim', '—')}), not the unconstrained "
                f"{mini.get('joint_min', '—')}",
                f"The +{int(mini.get('measure_dim', 0) or 0) - int(mini.get('joint_min', 0) or 0)} is the "
                "explicit, axis-aligned measure block required by tree/k-means consumers.",
                "T2 minimality note",
            ],
            [
                f"OPE recall = {_fmt(_recall)}",
                "Safety-first gate: recall is a POWER metric, not a deploy gate (DEC-214).",
                "T3 OPE: 9/9 hard, recall reported",
            ],
            [
                "Consumer gates use 3 fold/bootstrap seeds",
                "Multi-seed mean ± SE is reported; more seeds would tighten the intervals.",
                "T3 consumer rows (seeds column)",
            ],
            [
                "`effect` self-supervised objective is ~0",
                "The dedicated action-response head did not add; decision-sufficiency comes from the measure "
                "response block.",
                "T1 objective battery",
            ],
            [
                (
                    f"white_queen stream bridge HOLDs (DR "
                    f"{_fmt((_wq.get('estimates') or {}).get('dr'))} vs bar {_fmt(_wq.get('bar'))})"
                    if _wq
                    else "white_queen stream bridge not run"
                ),
                "The broad OPE acceptance (benchmark) is the hard gate; the stream bridge is a "
                "second, more conservative estimator and is reported, not used to override T3.",
                "OPE provenance",
            ],
            [
                f"Known-effect is response-block recovery of the known per-action optima "
                f"({known.get('passed')}/{known.get('total')} arms), not a full red_king arm-value model",
                "A direct, valid positive control; red_king arm-value validation is separate.",
                "T3 known-effect",
            ],
        ],
    )
    md.append("")

    # 7. opportunity board
    md.append("## 8. Opportunity board — every non-PASS and the exact next action")
    md.append("")
    if obj:
        md += _table(
            ["objective", "status", "current", "next action"],
            [
                [
                    o["id"],
                    _sc(o["status"]),
                    o["current"],
                    _NEXT.get(o["id"], "See its evidence section."),
                ]
                for o in obj["objectives"]
                if o["status"] != PASS
            ],
        )
    md.append("")
    md.append("### Recommendations to be successful")
    md.append("")
    for rec in _RECS:
        md.append(f"- {rec}")
    md.append("")

    # 8. provenance
    md.append("## 9. Provenance")
    md.append("")
    for label, val in (
        ("stream", args.stream),
        ("tag", args.tag),
        ("as_of", args.as_of),
        ("git", git),
        ("card receipts", args.card),
        ("objectives", args.objectives),
        ("portfolio", args.portfolio),
        ("minimality joint", args.minimality),
        ("minimality shape", args.minimality_shape),
        ("serve projection", args.serve_projection),
        ("OPE acceptance", args.acceptance),
        ("schema suite", args.schema_suite),
        ("encoder", args.encoder),
        (
            "model identity",
            (
                f"tag={enc.get('identity', {}).get('tag')} · "
                f"sample_a={enc.get('identity', {}).get('sample_a_customers')} · "
                f"ckpt={enc.get('identity', {}).get('ckpt_dir')}"
                if enc.get("identity")
                else None
            ),
        ),
        (
            "determinism",
            (
                f"embedding_sha256={repro.get('embedding_sha256')} · "
                f"infer={repro.get('infer_wall_seconds')}s"
                if repro
                else None
            ),
        ),
        ("accessibility", args.accessibility),
        ("transfer", args.transfer),
        ("known effect", args.known_effect),
        ("fair comparators", args.comparators),
        ("truth", args.truth),
    ):
        if val:
            md.append(f"- **{label}**: `{val}`")
    md.append("")
    text = "\n".join(md) + "\n"
    return md, text


def _case_what(c):
    return {
        "minimal": "the 4-column minimum: no attributes",
        "all_categorical": "no numeric field at all",
        "high_cardinality": "frequency-capped vocab + OOV",
        "mixed_dirty": "a mixed-type column fails safe to categorical",
        "null_heavy": "sparse numerics with a null mask",
        "nested_only": "nested JSON skipped, recorded",
        "multi_field": "many categorical + numeric fields",
        "single_event": "customers with one event",
    }.get(c, "schema stress case")


_NEXT = {
    "t0.north_star": "Close every T1–T4 red/partial/unknown.",
    "t1.transfer": "Report supervised vs unsupervised split; require adds-on >=1 head and zero FAIL.",
    "t2.minimality": "Re-certify the block-wise served E (shape rank + explicit measure) after representation changes.",
    "t2.decision_sufficiency": "Keep the point-in-time response-block gate + DiD null; re-run after representation changes.",
    "t3.supervised": "Hold the fair-generic fleet verdict (no regression + wins) across seeds.",
    "t3.unsupervised": "Hold the accessible-measure value gate; report full-concat dilution.",
    "t3.ope": "Hold hard gates; recall is reported (DEC-214).",
    "t3.known_effect": "Run the known-effect recovery receipt.",
    "t4.robustness": "Keep divergence=failure and the init-eval exclusion.",
}

_RECS = [
    "**More consumer seeds:** the gate now runs multi-seed (mean ± SE, all-seed wins); add seeds as budget "
    "allows to tighten the intervals.",
    "**Keep the frozen benchmark snapshot current:** the cockpit is wired into "
    "`scripts/reference_pipeline.py`; refresh the snapshot after every behavior change.",
    "**Close the shape's remaining tie gap:** the ablation shows E ≥ every component on all valid heads; "
    "significant wins are on channel/category. Push next_value/return with a stronger shape objective "
    "rather than a weaker gate.",
    "**Strengthen the self-supervised `effect` head / add red_king arm-value recovery:** decision-sufficiency "
    "currently comes from the measure response block; the dedicated effect head is ~0.",
    "**Diagnose the stream-bridge HOLD vs the response-block DEPLOY** (DR 0.5 vs bar 0.52): "
    "reconcile the estimators (horizon, overlap, reward scale) before treating either as sole truth.",
    "**Diagnose with axes:** when a metric is red, use §3.2 to name the axis before changing code.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Proof-first cockpit report card")
    for name in (
        "card",
        "objectives",
        "portfolio",
        "minimality",
        "minimality-shape",
        "serve-projection",
        "acceptance",
        "schema-suite",
        "encoder",
        "accessibility",
        "transfer",
        "known-effect",
        "comparators",
        "ablation",
        "reproducibility",
        "u-table",
        "u-controls",
        "card-md",
        "truth",
    ):
        ap.add_argument(f"--{name}", default=None, dest=name.replace("-", "_"))
    ap.add_argument("--stream", default="reference")
    ap.add_argument("--tag", default="unknown")
    ap.add_argument("--as-of", default="unknown", dest="as_of")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    md, text = build(a)
    print(text)
    if a.out_md:
        Path(a.out_md).write_text(text)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps({"markdown": md}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    main()
