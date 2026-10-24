"""Generate docs/REPORT_CARD.md from receipts + the frozen gate spec.

Usage: python3 scripts/report_card.py [--out docs/REPORT_CARD.md] [--receipts docs/report_cards/experiments]

The card is a pure function of (receipts, card_spec). It is never hand-edited.
Every row resolves to a receipt field; a missing source renders NOT MEASURED
(never silently dropped). The verdict is computed from the rows, so the header
cannot contradict the body.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import card_spec as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "report_cards"


def res(doc, path):
    """Resolve a dotted path with list indices into a loaded receipt."""
    cur = doc
    if path in (None, ""):
        return cur
    for part in str(path).split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _load(receipts, name):
    p = receipts / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def evaluate(row, doc):
    """Return (value_str, verdict, note). verdict in PASS/FAIL/NOT MEASURED."""
    if row.kind == "unmeasured" or doc is None:
        return ("—", "NOT MEASURED", "no receipt")
    v = res(doc, row.source[1]) if row.source else None
    k = row.kind
    try:
        if k == "le":
            if v is None:
                return ("—", "NOT MEASURED", "")
            ok = float(v) <= row.thresh
            return (f"{float(v):.3g}", "PASS" if ok else "FAIL", f"gate {row.gate}")
        if k == "ge":
            if v is None:
                return ("—", "NOT MEASURED", "")
            ok = float(v) >= row.thresh
            return (f"{float(v):.3g}", "PASS" if ok else "FAIL", f"gate {row.gate}")
        if k == "gt":
            if v is None:
                return ("—", "NOT MEASURED", "")
            ok = float(v) > row.thresh
            return (f"{float(v):.3g}", "PASS" if ok else "FAIL", f"gate {row.gate}")
        if k == "present":
            ok = bool(v)
            return (str(v) if v else "—", "PASS" if ok else "FAIL", "")
        if k == "bool":
            ok = bool(v)
            return (str(bool(v)), "PASS" if ok else "FAIL", "")
        if k == "identity":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            ec, tc = v.get("E_IB_channel"), v.get("table_channel")
            ea, ta = v.get("E_IB_cat"), v.get("table_cat")
            ok = ec >= tc and ea >= ta
            return (f"ch {ec:.3f}/{tc:.3f} · cat {ea:.3f}/{ta:.3f}", "PASS" if ok else "FAIL", "both >= table")
        if k == "rate":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            e, t = v.get("E_churn_auc"), v.get("table_churn_auc")
            ok = e >= t
            return (f"E {e:.4f} vs {t:.4f}", "PASS" if ok else "FAIL", "E >= table")
        if k == "causal":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            ok = v.get("lo", 0) > 0 and v.get("diff", 0) > 0
            return (f"diff {v.get('diff'):+.1f} [{v.get('lo'):+.1f},{v.get('hi'):+.1f}]",
                    "PASS" if ok else "FAIL", "E recovers effect, CI>0")
        if k == "shuffle":
            if v is None:
                return ("—", "NOT MEASURED", "")
            return (f"{float(v):.4f} (E vs shuffled)", "PASS" if float(v) != 0 else "FAIL", "E is order-sensitive")
        if k == "identity2":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            ec, tc = v.get("identity_channel"), v.get("table_channel")
            ea, ta = v.get("identity_category"), v.get("table_category")
            ok = ec >= tc and ea >= ta
            return (f"ch {ec:.3f}/{tc:.3f} · cat {ea:.3f}/{ta:.3f}", "PASS" if ok else "FAIL", "both >= table")
        if k == "rate2":
            if not isinstance(v, dict) or "churn90" not in v:
                return ("—", "NOT MEASURED", "")
            e, t = v["churn90"].get("E"), v["churn90"].get("table")
            ok = e >= t
            return (f"E {e:.4f} vs {t:.4f}", "PASS" if ok else "FAIL", "E >= table")
        if k == "shuffle2":
            if not isinstance(v, dict) or "shuffle" not in v:
                return ("—", "NOT MEASURED", "")
            drop = v["shuffle"]["value90"] - v["value90"]["E"]
            return (f"drop {drop:+.4f}", "PASS" if drop < 0 else "FAIL", "E order-sensitive")
        def _ctrl(arm):
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            rs = v.get("supervised", {})
            vals = []
            ok = True
            for t in ("value90", "orders90", "churn90"):
                e = rs.get(t, {}).get("E")
                a = rs.get(t, {}).get(arm)
                if e is None or a is None:
                    return ("—", "NOT MEASURED", "missing arm")
                vals.append(e - a)
                if a >= e:
                    ok = False
            return (" / ".join(f"{d:+.3f}" for d in vals), "PASS" if ok else "FAIL", f"E > {arm} all 3")
        if k == "ctrl_random":
            return _ctrl("random")
        if k == "ctrl_destroyed":
            return _ctrl("destroyed")
        if k == "obj":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            if v.get("degenerate"):
                return (f"skill {v.get('skill'):+.3f} vs naive {v.get('naive'):+.3f}",
                        "EXCLUDED", "degenerate target (excluded, reported)")
            ok = bool(v.get("beats_naive_2se"))
            return (f"skill {v.get('skill'):+.3f} vs naive {v.get('naive'):+.3f}",
                    "PASS" if ok else "FAIL", "beats naive by >2SE")
        if k == "relational":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            pu, rd = v.get("order_id_purity"), v.get("order_id_random")
            ok = pu > rd
            return (f"{pu} vs chance {rd}", "PASS" if ok else "FAIL", "> chance")
        def _r2(x):
            return x if isinstance(x, (int, float)) else (x or {}).get("r2")
        if k in ("donor_raw", "donor_fair"):
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            rs = v.get("results", {})
            margins = []
            ok = True
            for t in ("value90", "orders90", "churn90"):
                r = rs.get(t, {})
                e = _r2(r.get("E"))
                tb = _r2(r.get("table"))
                if e is None or tb is None:
                    return ("—", "NOT MEASURED", "missing arm")
                margins.append(e - tb)
                if e < tb:
                    ok = False
            return (" / ".join(f"{m:+.3f}" for m in margins), "PASS" if ok else "FAIL", "E >= table all 3")
        if k == "donor_add":
            if not isinstance(v, dict):
                return ("—", "NOT MEASURED", "")
            rs = v.get("results", {})
            deltas = []
            ok = True
            for t in ("value90", "orders90", "churn90"):
                r = rs.get(t, {})
                a = _r2(r.get("table_seq_E"))
                b = _r2(r.get("table_seq"))
                if a is None or b is None:
                    return ("—", "NOT MEASURED", "missing arm")
                deltas.append(a - b)
                if a - b <= 0:
                    ok = False
            return (" / ".join(f"{d:+.3f}" for d in deltas), "PASS" if ok else "FAIL", "delta > 0 all 3")
    except Exception as e:
        return ("—", "NOT MEASURED", f"eval error {e}")
    return ("—", "NOT MEASURED", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "REPORT_CARD.md"))
    ap.add_argument("--receipts", default=str(DEMO / "experiments"))
    a = ap.parse_args()
    receipts = Path(a.receipts)
    demo = _load(DEMO, "demo_run.json")

    results = []
    for row in S.ROWS:
        doc = demo if (row.source and row.source[0] == "demo_run.json") else _load(receipts, row.source[0]) if row.source else None
        val, verdict, note = evaluate(row, doc)
        results.append((row, val, verdict, note))

    counts = {"PASS": 0, "FAIL": 0, "NOT MEASURED": 0}
    for _, _, v, _ in results:
        counts[v] += 1
    mastered = counts["FAIL"] == 0 and counts["NOT MEASURED"] == 0

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    now = datetime.date.today().isoformat()
    ehash = (demo or {}).get("hash", "n/a")
    runtime = (demo or {}).get("runtime_s", "n/a")
    nseeds = (demo or {}).get("n_seeds", "n/a")

    L = []
    L.append("# REPORT_CARD — Layer-B (encoder) [GENERATED]")
    L.append("")
    L.append("> **Generated** by `python3 scripts/report_card.py` — DO NOT EDIT. "
             "Gates live in `scripts/card_spec.py`; every row resolves to a receipt "
             "(`docs/report_cards/experiments/`). A missing receipt renders NOT MEASURED.")
    L.append("")
    L.append(f"- **Generated:** {now} · **commit:** `{commit}` · **branch:** primary-encoder")
    L.append(f"- **Data:** `relational_v3` synthetic reference (+ schema variant) · harness, no business data.")
    L.append(f"- **Latest demo receipt:** `{ehash}` · seeds `{nseeds}` · runtime `{runtime}s`.")
    L.append(f"- **B mastery:** **{'MASTERED' if mastered else 'NOT YET MASTERED'}** — "
             f"PASS {counts['PASS']} · FAIL {counts['FAIL']} · NOT MEASURED {counts['NOT MEASURED']}.")
    L.append("")

    L.append("## Project goal (AGENTS.md)")
    L.append("")
    L.append("One pretrained, **self-supervised, schema-agnostic encoder per stream** whose FROZEN output is the")
    L.append("**universal donor** every consumer builds on. **Layer 0 (now):** `E`-only matches/beats a fair table and")
    L.append("adds signal beyond `table+seq`; no table/counter/summary inside `E`. **Layer 1 (next):** certification --")
    L.append("sufficiency / minimality / decision-sufficiency via `U = 1 - worst-case oracle gap` vs full history.")
    L.append("**Anti-overfit:** Layer B is self-supervised only; known use cases may VETO, never SHAPE.")
    L.append("")
    L.append("## Principles (charter; conflict order Robust > Universal > Adaptive > Lawful > Efficient > Honest)")
    L.append("")
    L.append("1. **Universal** -- one architecture for any stream; a model is trained per stream, never cross-applied.")
    L.append("2. **Adaptive & Dynamic** -- everything resolves or is learned from the input; no magic literals.")
    L.append("3. **Lawful** -- compositionality, equivariance, and the correct algebra per quantity.")
    L.append("4. **Efficient** -- linear-time, constant-state, minimal capacity; slow or bloated is not done.")
    L.append("5. **Robust** -- fail safe, calibrated uncertainty, deterministic; divergence is a hard failure.")
    L.append("6. **Honest** -- held-out/falsifiable, label-free, never weakened to pass a gate or tuned to the judged metric.")
    L.append("7. **Accessible** -- information recoverable by simple consumers, not merely present.")
    L.append("8. **Diagnosable** -- attribute a shortfall to a specific axis with a matched control.")
    L.append("")
    L.append("## Encoder architecture (Layer B)")
    L.append("")
    L.append("- **Tokenizer** -- auto-discovers fields (categorical / numeric / ISO-time); additive per-field embeddings")
    L.append("  with learned gates; time features = `log1p(gap)` + data-derived Fourier periods (FFT peaks); raw numerics via signed `log1p`.")
    L.append("- **Dynamics** -- continuous-time gated SSM (learned decay, tau initialized from data gap-quantiles);")
    L.append("  affine prefix `_scan` (monoid-exact); width/layers derived from data.")
    L.append("- **State** -- contractive shape state (bounded) + additive mass state (extensive algebra: sums add, no `1-decay`).")
    L.append("- **Readout** -- ensemble (last / mean / max / attention-pooled / per-scale states / mass) with learned positive")
    L.append("  weights -> `E` (d-dim, frozen).")
    L.append("- **Self-supervised objectives (heads)** -- next-event; gap/rate; categorical + numeric reconstruction; identity;")
    L.append("  uncertainty (missingness); past-summary; future-summary (one head per data-derived horizon); relational/context.")
    L.append("- **Training** -- AdamW; trust-region adaptive lr (seed bound; expand/shrink on held-out response); no gradient")
    L.append("  clip; MAD-based early stop; deterministic kernels; learned loss weights.")
    L.append("")
    sections = [(S.CONSTRUCTION, "Construction — invariants by construction"),
                (S.DONOR, "Donor — does not lose / adds / controls"),
                (S.AXES, "Content axes (8)"),
                (S.OBJECTIVES, "Objectives (self-supervised battery)")]
    for sec, title in sections:
        L.append(f"## {title}")
        L.append("")
        L.append("| # | Item | Metric | Comparator | Gate | Value | Verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for row, val, verdict, note in results:
            if row.section != sec:
                continue
            L.append(f"| {row.ident} | {row.name} | {row.metric} | {row.comparator} | {row.gate} | {val} | **{verdict}** |")
        L.append("")

    L.append("## Verdict")
    L.append("")
    if mastered:
        L.append("All construction invariants hold, the donor does not lose and adds over table+seq, all 8 axes pass, "
                 "and every objective beats its naive baseline. **B is mastered — graduate to B+C.**")
    else:
        L.append(f"B is **not yet mastered**: {counts['NOT MEASURED']} row(s) unmeasured, {counts['FAIL']} failing. "
                 "Unmeasured rows need receipts; failing rows need an axis fix (one axis per run).")
    L.append("")
    L.append("## Disclosures")
    L.append("")
    L.append("- Synthetic harness; no business-value or causal claim beyond the randomized holdout.")
    L.append("- Some axis receipts predate DEC-280/281 (future-objective + ladder); refresh required before claiming them.")
    L.append("- Objective battery (`next/dt/jepa/mask/value/entity/order/agg/query/sf`) is defined in "
             "`looking_glass/probes.py` and not yet wired to this encoder -> NOT MEASURED.")
    (Path(a.out)).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("WROTE", a.out)
    print(f"SUMMARY PASS={counts['PASS']} FAIL={counts['FAIL']} NM={counts['NOT MEASURED']} mastered={mastered}")


if __name__ == "__main__":
    main()
