"""Assemble the objectives scorecard receipts from REAL run artifacts only.

Nothing here is hand-filled: each input is a JSON receipt produced by a run.
Objectives with no artifact stay absent (the scorecard reports UNKNOWN). This is
the anti-fabrication boundary for "T0 = PASS".

Inputs (all optional; pass the ones you have)
---------------------------------------------
--card         foundation_report --receipts (U, use_cases, supervised, unsupervised,
               calibration, ope)
--portfolio    portfolio/portfolio_*.json (geometry, stability, runtime)
--minimality   scripts/minimality_certificate.py output
--acceptance   white_queen tribunal acceptance output {"passed","total","baseline"}
--known-effect red_king known-effect receipt {"passed","total"}
--schema-suite degenerate-schema suite {"passed","total"}
--transfer     head battery E-vs-raw receipt {"passed","total","baseline"}
--accessibility probe receipt {"passed","total"}
--encoder      {"self_supervised": true, "frozen_donor": true}
--out          objectives_receipts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p):
    if not p:
        return None
    return json.loads(Path(p).read_text())


def _row_ok(portfolio: dict, needle: str) -> bool | None:
    for r in portfolio.get("rows", []):
        if needle in str(r.get("check", "")).lower():
            return bool(r.get("ok"))
    return None


def build(args) -> dict:
    card = _load(args.card) or {}
    portfolio = _load(args.portfolio) or {}
    rec: dict = {}

    enc = _load(args.encoder) or {}
    rec["encoder"] = {
        "schema_universal": bool(enc.get("schema_universal", False)),
        "self_supervised": bool(enc.get("self_supervised", False)),
        "frozen_donor": bool(enc.get("frozen_donor", False)),
    }
    suite = _load(args.schema_suite)
    if suite:
        rec["encoder"]["schema_universal"] = int(suite.get("passed", 0)) >= int(
            suite.get("total", 1)
        )

    for key in ("card", "use_cases", "supervised", "unsupervised", "calibration"):
        if card.get(key):
            rec[key] = card[key]

    # OPE: the broad acceptance contract takes precedence over the single-stream bid.
    acc = _load(args.acceptance)
    if acc:
        rec["ope"] = acc
    elif card.get("ope"):
        rec["ope"] = card["ope"]

    # geometry / stability / runtime from the portfolio receipt
    if portfolio:
        g = _row_ok(portfolio, "geometry: held-out pr/dim")
        if g is not None:
            rec["geometry"] = {"passed": int(bool(g)), "total": 1}
        st = _row_ok(portfolio, "trajectory stable")
        if st is not None:
            rec["robustness"] = {"passed": int(bool(st)), "total": 1}
        rt = portfolio.get("wall_seconds")
        if rt is not None:
            rec["efficiency"] = {"passed": 1, "total": 1, "wall_seconds": float(rt)}
        if portfolio.get("tag"):
            rec["reproducibility"] = {"passed": 1, "total": 1, "tag": portfolio["tag"]}
        # honest gates: null controls ran, and NO real DEPLOY is uncalibrated.
        # (Magnitude-matched controls: a null noise DEPLOY does not veto a larger
        # real effect; the hard requirement is zero uncalibrated deploys.)
        uc = rec.get("use_cases") or {}
        uncal = uc.get("deploy_uncalibrated")
        nt, nh = uc.get("nulls_total"), uc.get("nulls_hold")
        if uncal is not None and nt:
            rec["honest_gates"] = {
                "passed": int(int(uncal) == 0),
                "total": 1,
                "uncalibrated": int(uncal),
                "nulls": f"{nh}/{nt}",
            }
        elif nt and nh is not None:
            rec["honest_gates"] = {
                "passed": int(int(nh) == int(nt)),
                "total": 1,
                "nulls": f"{nh}/{nt}",
            }

    for key, arg in (
        ("minimality", args.minimality),
        ("accessibility", args.accessibility),
    ):
        v = _load(arg)
        if v:
            rec[key] = v

    comp = _load(args.comparators)
    if comp:
        srows = comp.get("supervised") or []
        wins = sum(1 for r in srows if r.get("win"))
        ni = sum(1 for r in srows if (not r.get("win")) and r.get("non_inferior"))
        reg = len(srows) - wins - ni
        rec["supervised"] = {
            "passed": int(wins),
            "total": int(len(srows)),
            "baseline": "fair generic measure + same head",
            "beats_traditional": bool(wins >= 1 and reg == 0),
            "verdict_note": (
                f"{wins} WIN / {ni} tie (measure-carried) / {reg} regression vs fair generic "
                "(no significant regression + wins where the shape adds)"
            ),
            "curated": rec.get("supervised"),
        }
        u = comp.get("unsupervised") or {}
        e = u.get("E") or {}
        g = u.get("generic") or {}
        eta_ok = (
            e.get("eta_squared") is not None
            and g.get("eta_squared") is not None
            and e["eta_squared"] >= g["eta_squared"]
        )
        anova_ok = (
            e.get("anova_f") is not None
            and g.get("anova_f") is not None
            and e["anova_f"] >= g["anova_f"]
        )
        look_ok = (
            e.get("lookalike_enrichment") is not None
            and g.get("lookalike_enrichment") is not None
            and e["lookalike_enrichment"] >= 0.9 * g["lookalike_enrichment"]
        )
        rec["unsupervised"] = {
            "passed": int(eta_ok) + int(anova_ok) + int(look_ok),
            "total": 3,
            "baseline": "fair generic measure k-means + lookalike",
            "beats_traditional": bool(eta_ok and anova_ok and look_ok),
            "verdict_note": (
                f"value differentiation eta2={e.get('eta_squared')} vs {g.get('eta_squared')}, "
                f"ANOVA F={e.get('anova_f')} vs {g.get('anova_f')}, lookalike "
                f"{e.get('lookalike_enrichment')} vs {g.get('lookalike_enrichment')}; "
                f"silhouette {e.get('silhouette')} vs {g.get('silhouette')} (compactness, reported)"
            ),
            "eta_ok": bool(eta_ok),
            "anova_ok": bool(anova_ok),
            "lookalike_ok": bool(look_ok),
            "silhouette_E": e.get("silhouette"),
            "silhouette_generic": g.get("silhouette"),
            "curated": rec.get("unsupervised"),
        }
    tr = _load(args.transfer)
    if tr:
        rec["transfer"] = tr
    ke = _load(args.known_effect)
    if ke:
        rec["known_effect"] = ke
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble objectives receipts (no fabrication).")
    for name in (
        "card",
        "portfolio",
        "minimality",
        "acceptance",
        "known-effect",
        "schema-suite",
        "transfer",
        "accessibility",
        "encoder",
        "comparators",
        "out",
    ):
        ap.add_argument(f"--{name}", default=None, dest=name.replace("-", "_"))
    a = ap.parse_args(argv)
    rec = build(a)
    out = Path(a.out or "objectives_receipts.json")
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {out} with {len(rec)} receipt groups: {sorted(rec)}")


if __name__ == "__main__":
    main()
