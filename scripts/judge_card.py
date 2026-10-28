#!/usr/bin/env python3
"""Render the JUDGE card (DEC-390) — the project's single steering artifact.

    python3 scripts/judge_card.py [--receipts DIR] [--out JUDGE_CARD.md]

Design rules, all enforced in code rather than in prose:

* **The verdict is computed, never narrated.** This file contains no numbers; every value comes
  from a receipt under the receipts dir.
* **Absence is UNKNOWN, never PASS.** A missing receipt names the exact file that must be
  produced.
* **The freshness law is mechanical** (DEC-390): each source receipt must carry the `behavior_hash`
  of the encoder/serving code that produced it. If that code changed since, the row is UNKNOWN.
* **No PASS without a finite comparator, a paired CI excluding zero, controls below the brain,
  and a recorded N** (certification-integrity rules 3-5).
* A receipt that defines its own criterion is UNKNOWN, not evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_spec as J  # noqa: E402

EXP = ROOT / "docs" / "report_cards" / "experiments"


# --------------------------------------------------------------------------- freshness


def behavior_hash() -> str:
    """sha256 over the behavior-bearing source tree (encoder + data path).

    Derived from the package trees, not a curated file list: a newly added module is covered by
    the freshness law automatically, which is the whole point of the law.
    """
    h = hashlib.sha256()
    files: list[Path] = []
    for tree in J.BEHAVIOR_TREES:
        base = ROOT / tree
        if base.is_dir():
            files.extend(p for p in sorted(base.rglob("*")) if p.suffix in J.BEHAVIOR_SUFFIXES)
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def load(receipts: Path, name: str):
    p = receipts / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {"_load_error": repr(exc)}


# --------------------------------------------------------------------------- evaluation


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _meta_problem(receipt, current_hash: str, bhash_key: str = "behavior_hash") -> str | None:
    """Why this receipt cannot be trusted as fresh/current, or None if it can."""
    if receipt is None:
        return "receipt missing"
    if "_load_error" in receipt:
        return f"receipt unreadable: {receipt['_load_error']}"
    if J.CRITERION_KEYS & set(receipt.keys()):
        return J.UNREGISTERED_NOTE
    meta = receipt.get("_meta") or {}
    if not meta.get("generated_at"):
        return "receipt has no generated_at"
    if not meta.get("git_commit"):
        return "receipt has no git_commit"
    if not meta.get(bhash_key):
        return "receipt has no behavior_hash (freshness unverifiable)"
    if meta[bhash_key] != current_hash:
        return (
            f"stale: encoder behavior changed since receipt ({meta[bhash_key]} != {current_hash})"
        )
    return None


def eval_tasks(tasks, bhash_note: str | None) -> tuple[str, str, list]:
    """Return (verdict, reason, per-task notes) for one leg column's task list."""
    if bhash_note:
        return J.UNKNOWN, bhash_note, []
    if not isinstance(tasks, list) or not tasks:
        return J.UNKNOWN, "no tasks recorded in this column", []

    notes, live = [], []
    for t in tasks:
        name = str(t.get("name", "?"))
        if t.get("degenerate"):
            notes.append(
                f"{name}: EXCLUDED (degenerate — {t.get('degenerate_reason', 'no variance')})"
            )
            continue
        missing = [f for f in J.REQUIRED_TASK_FIELDS if f not in t]
        if missing:
            return J.UNKNOWN, f"{name}: missing {missing}", notes
        if not all(_finite(t[f]) for f in ("brain", "pathA", "ci_lo", "ci_hi")) or not _finite(
            t.get("n")
        ):
            return J.UNKNOWN, f"{name}: non-finite measurement or N", notes
        if t["ci_lo"] > t["ci_hi"]:
            return J.UNKNOWN, f"{name}: ci_lo > ci_hi", notes
        comp = str(t.get("comparator", "PATH A (GBM incumbent)")).strip().lower()
        if comp in J.INVALID_COMPARATORS:
            return J.UNKNOWN, f"{name}: comparator '{comp}' is not a fair comparator", notes
        if t["ci_lo"] == t["ci_hi"]:
            notes.append(f"{name}: zero-width CI (point estimate only)")
            t = dict(t, _zero_width=True)
        for ctl in ("random", "destroyed"):
            if ctl not in t or not _finite(t[ctl]):
                return J.UNKNOWN, f"{name}: control '{ctl}' missing (H4)", notes
            if t[ctl] > t["brain"]:
                return J.FAIL, f"{name}: {ctl} control >= brain ({t[ctl]} vs {t['brain']})", notes
        live.append(t)
        notes.append(
            f"{name}: brain {t['brain']:.4g} vs PATH A {t['pathA']:.4g} "
            f"(delta {t['brain'] - t['pathA']:+.4g}, CI [{t['ci_lo']:+.4g}, {t['ci_hi']:+.4g}], n={t['n']})"
        )

    if not live:
        return J.EXCLUDED, "every task degenerate (reported, not forced)", notes
    failing = [t for t in live if not (t["ci_lo"] > 0 and t["brain"] > t["pathA"])]
    if failing:
        names = ", ".join(t["name"] for t in failing)
        return J.FAIL, f"does not beat PATH A with CI excluding 0: {names}", notes
    return (
        J.PASS,
        f"{len(live)} task(s) beat PATH A; paired CI excludes 0; controls below brain",
        notes,
    )


# --------------------------------------------------------------------------- rendering


def _badge(v: str) -> str:
    return {
        J.PASS: "✅ PASS",
        J.FAIL: "❌ FAIL",
        J.UNKNOWN: "❔ UNKNOWN",
        J.EXCLUDED: "⚪ EXCLUDED",
    }[v]


def render(receipts: Path, current_hash: str) -> tuple[str, dict]:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"pass": 0, "fail": 0, "unknown": 0, "excluded": 0}
    blocked: list[str] = []

    def record(verdict: str, why: str, needs: str | None = None):
        summary[verdict.lower()] += 1
        if verdict in (J.UNKNOWN, J.FAIL) and needs:
            blocked.append(f"{needs} — {why}")
        return verdict, why

    # ---- top line: the judge ----------------------------------------------------
    judge_lines, verdicts = [], {}
    judge_lines += [
        "| Leg | Column | Verdict | Evidence |",
        "|---|---|---|---|",
    ]
    for row in J.LEG_ROWS:
        rc = load(receipts, row.source)
        note = _meta_problem(rc, current_hash)
        tasks = None
        if rc and not note:
            tasks = (rc.get("columns") or {}).get(row.column, {}).get("tasks")
            if tasks is None:
                note = f"receipt {row.source} has no columns.{row.column}.tasks"
        verdict, why, _ = eval_tasks(tasks, note)
        verdict, why = record(verdict, why, None if note is None else f"{row.source}")
        if verdict in (J.FAIL,):
            blocked.append(f"{row.source} :: {row.column} — {why}")
        verdicts[row.ident] = verdict
        judge_lines.append(
            f"| {row.leg} | {J.COLUMN_LABEL[row.column]} | {_badge(verdict)} | {why} |"
        )

    # ---- stage ladder -----------------------------------------------------------
    stage_lines = ["| Stage | Working-state proof | Verdict | Evidence |", "|---|---|---|---|"]
    stages_rc = load(receipts, "judge_stages.json")
    stage_note = _meta_problem(stages_rc, current_hash)
    for s in J.STAGE_ROWS:
        val = (stages_rc or {}).get(s.ident) if not stage_note else None
        if stage_note:
            verdict, why = (
                J.UNKNOWN,
                stage_note if "missing" not in stage_note else "judge_stages.json missing",
            )
        elif val is None:
            verdict, why = J.UNKNOWN, f"no measurement recorded for {s.ident}"
        elif isinstance(val, bool):
            verdict, why = (J.PASS, "reported working") if val else (J.FAIL, "reported not working")
        elif isinstance(val, dict) and "working" in val:
            verdict = J.PASS if val["working"] else J.FAIL
            why = str(val.get("evidence", ""))[:160]
        else:
            verdict, why = J.UNKNOWN, f"malformed entry for {s.ident}"
        verdict, why = record(verdict, why, "judge_stages.json")
        verdicts[s.ident] = verdict
        stage_lines.append(f"| **{s.ident}** {s.name} | {s.gate} | {_badge(verdict)} | {why} |")

    # ---- hygiene ----------------------------------------------------------------
    hyg_lines = ["| # | Check | Verdict | Evidence |", "|---|---|---|---|"]
    env_rc = load(receipts, "env_receipt.json")
    env_ok = bool(env_rc and (env_rc.get("torch") or {}).get("matmul_smoke", {}).get("ok"))
    checks = {
        "H1": (
            env_ok,
            "env_receipt.json device matmul ok" if env_ok else "run scripts/env_receipt.py",
        ),
        "H2": (bool(current_hash), f"behavior_hash {current_hash}"),
        "H3": (True, "enforced per row (no PASS without N)"),
        "H4": (True, "enforced per row (no PASS without controls)"),
        "H5": (True, "enforced per row (comparator must be finite and fair)"),
    }
    for h in J.HYGIENE_ROWS:
        ok, why = checks[h.ident]
        verdict, why = record(J.PASS if ok else J.UNKNOWN, why, h.ident)
        verdicts[h.ident] = verdict
        hyg_lines.append(f"| {h.ident} | {h.name} | {_badge(verdict)} | {why} |")

    # ---- verdict rollup ---------------------------------------------------------
    legs_green = all(verdicts[r.ident] == J.PASS for r in J.LEG_ROWS)
    stages_green = all(verdicts[s.ident] == J.PASS for s in J.STAGE_ROWS)
    if legs_green and stages_green:
        assessment = (
            "ENCODER DONE (v1): all four legs PASS in both columns and every stage is working "
            "by receipt. Capacity certification (k_min) is reported separately below."
        )
    else:
        assessment = (
            f"NOT DONE. Legs green: {sum(1 for r in J.LEG_ROWS if verdicts[r.ident] == J.PASS)}"
            f"/{len(J.LEG_ROWS)} · stages green: "
            f"{sum(1 for s in J.STAGE_ROWS if verdicts[s.ident] == J.PASS)}/{len(J.STAGE_ROWS)}. "
            "This is a statement about receipts, not about effort; nothing here is a claim of "
            "universality (DEC-410: that word appears only when the coverage rows are green)."
        )

    md = "\n".join(
        [
            "# JUDGE CARD — PATH B > PATH A",
            "",
            f"*Generated `{generated}` · commit `{git('rev-parse', '--short', 'HEAD')}` "
            f"(`{git('rev-parse', '--abbrev-ref', 'HEAD')}`{' · dirty' if git('status', '--porcelain') else ''}) "
            f"· encoder behavior `{current_hash}`*",
            "",
            f"**Assessment:** {assessment}",
            "",
            "The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.",
            "Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.",
            "",
            "## The judge — four legs × two numbers",
            "",
            *judge_lines,
            "",
            "## The stage ladder — what is broken when a leg is red",
            "",
            *stage_lines,
            "",
            "## Certification hygiene",
            "",
            *hyg_lines,
            "",
            "## What is blocking (exact missing receipts)",
            "",
            *([f"- `{b}`" for b in blocked] or ["- nothing — all rows evaluated"]),
            "",
            "---",
            "",
            "*The inherited status-quo card (`REPORT_CARD.md`, generated by `scripts/report_card.py`) "
            "measures the pipeline as it was. This card measures the rebuild; the two are not "
            "interchangeable and neither overrides the other (DEC-415).*",
            "",
        ]
    )
    return md, {"summary": summary, "assessment": assessment, "verdicts": verdicts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipts", default=str(EXP))
    ap.add_argument("--out", default=str(ROOT / "JUDGE_CARD.md"))
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args()

    current = behavior_hash()
    md, info = render(Path(a.receipts), current)
    Path(a.out).write_text(md)
    print(f"judge card -> {a.out}")
    print(f"behavior hash: {current}")
    print("summary:", json.dumps(info["summary"], sort_keys=True))
    print(info["assessment"])
    if a.do_print:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
