#!/usr/bin/env python3
"""Render the REPORT_CARD (DEC-390) — the project's single steering artifact.

    python3 scripts/report_card.py [--receipts DIR] [--out REPORT_CARD.md]

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
BACKWARD_REQUIREMENTS = ROOT / "docs" / "backward_requirements.json"
# the card is a committed GENERATED artifact: its own bytes are the only thing a render is
# allowed to leave uncommitted, so they must not decide the `dirty` flag (see _worktree_dirty)
CARD_NAME = "REPORT_CARD.md"


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


def _worktree_dirty(status_porcelain: str) -> bool:
    """Working-tree state, EXCLUDING the card's own output.

    The card is a committed generated artifact, so its own bytes are the only thing a render is
    allowed to dirty. Counting them meant every render printed `dirty` forever and certification
    rule 6 ("regenerated at a clean, recorded commit") was unreachable — a flag that can never
    say the truth is worse than no flag, because readers learn to ignore it.
    """
    # `git()` returns "?" when a command produced NO output, and an empty `status --porcelain`
    # IS the clean tree. Taking the sentinel at face value made every truly clean tree render
    # `dirty` — the same "a flag that always says one thing is not a flag" defect DEC-430 removed
    # for the card's own bytes, and it was worse this way round: the honest state was unreachable
    # and a reader would learn to ignore the only signal rule 6 has.
    if status_porcelain.strip() in ("", "?"):
        return False
    return any(line.strip() and CARD_NAME not in line for line in status_porcelain.splitlines())


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


def _block_problem(block, current_hash: str) -> str | None:
    """Per-PRODUCER freshness (certification rules 6 and 7).

    `judge_stages.json` is a MERGE: five stage blocks written by four different scripts at four
    different times. The file's own `_meta` therefore belongs to whoever wrote it LAST, which
    let one re-run of `measure_views_law` make every other block look current — including blocks
    whose inputs (`/tmp/opencode/...`) had been deleted. A block must show it was measured
    against the encoder behavior now being judged, or it is not evidence about this encoder.
    """
    if not isinstance(block, dict):
        return None  # scalars carry no stamp of their own; the file-level check covers them
    stamp = block.get("_stamp")
    if not isinstance(stamp, dict):
        return "block has no producer stamp (legacy receipt) — re-run its producer"
    bhash = str(stamp.get("behavior_hash") or "")
    if not bhash:
        return "block stamp records no behavior_hash — re-run its producer"
    if bhash != current_hash:
        return (
            f"stale: measured by {stamp.get('producer', '?')} against behavior {bhash[:12]} "
            f"(current {current_hash[:12]}) — re-run that producer"
        )
    # A block that contradicts itself is not evidence either (rule 7: self-consistency). The
    # frozen criterion is still the block's `working` flag — this only refuses a block whose own
    # recorded checks deny it, rather than letting `working: true` outrank its receipts.
    checks = block.get("checks")
    if isinstance(checks, dict) and checks and block.get("working"):
        denied = [k for k, v in checks.items() if not (isinstance(v, dict) and v.get("working"))]
        if denied:
            return (
                f"self-contradictory: reports working=True but "
                f"{len(denied)} check(s) did not ({', '.join(sorted(denied)[:3])}) — re-run its producer"
            )
    return None


def _stage_why(val: dict) -> str:
    """Evidence for a stage row — names WHAT FAILED rather than whatever the summary starts with.

    A red stage is a diagnosis to act on (principle 8), so the row has to name the failing check
    and its measured reason. The block-level `evidence` string is a concatenation of every check
    in insertion order, so on S4/S5 it began with an OK row and a red stage read like a green one.
    """
    checks = val.get("checks") or {}
    # fail safe: a check that cannot say it worked did not work (reject, do not assume)
    failing = {k: v for k, v in checks.items() if not (isinstance(v, dict) and v.get("working"))}
    if failing:
        name, check = next(iter(failing.items()))
        rest = f" (+{len(failing) - 1} more)" if len(failing) > 1 else ""
        return f"FAILED {name}: {str(check.get('evidence', ''))[:110]}{rest}"
    return str(val.get("evidence", ""))[:140]


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
        # H4 is only satisfied if a control COULD have won. A control at R2 << 0 is not beaten, it
        # is absurd: beating it proves the head is not insane, not that the representation carries
        # signal. Such a control is reported as vacuous, and a column may not PASS on vacuity.
        readable = [
            ctl
            for ctl in ("random", "destroyed")
            if _finite(t[ctl]) and t[ctl] >= J.VACUOUS_CONTROL_R2
        ]
        if not readable:
            t = dict(t, _no_readable_control=True)
            notes.append(
                f"{name}: H4 VACUOUS — both controls below R2={J.VACUOUS_CONTROL_R2} "
                f"(random {t['random']:.4g}, destroyed {t['destroyed']:.4g}); beating them is "
                f"not evidence of signal"
            )
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
    # A green row must rest on a control that could have caught it (DEC-433). This can only turn a
    # would-be PASS into UNKNOWN — it never turns a red row green, so it tightens H4 rather than
    # moving a verdict in our favor.
    vacuous = [t["name"] for t in live if t.get("_no_readable_control")]
    if vacuous:
        return (
            J.UNKNOWN,
            f"beats PATH A but H4 is vacuous on {len(vacuous)}/{len(live)} trait(s) "
            f"({', '.join(vacuous[:3])}): no control readable enough to have failed",
            notes,
        )
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


def _products_id(block) -> tuple | None:
    p = (block or {}).get("products") or {}
    if not p.get("registry_sha256") and not p.get("tag"):
        return None
    return (p.get("tag"), p.get("registry_sha256"))


def _artifact_mismatch(receipts: Path, stages_rc: dict | None) -> dict:
    """Rows measured against a DIFFERENT products artifact than the rest of the card are
    refused (UNKNOWN), not quietly rendered side by side (DEC-448, closing DEC-445).

    Two encoder protocols once shipped under one tag, so a card assembled from legs of both
    would have rendered as self-consistent. Identity is the registry (tag + content hash);
    receipts without an id (older artifacts) are exempt — this refuses a demonstrated MIX,
    it is not a new capability gate. Deterministic: the majority artifact wins; ties break
    lexicographically.
    """
    ids: dict[str, tuple] = {}
    for row in J.LEG_ROWS:
        i = _products_id(load(receipts, row.source))
        if i:
            ids[row.source] = i
    for s in J.STAGE_ROWS:
        if s.ident in ("S4", "S5"):
            i = _products_id((stages_rc or {}).get(s.ident))
            if i:
                ids[f"stage::{s.ident}"] = i
    distinct = set(ids.values())
    if len(distinct) <= 1:
        return {}
    counts: dict[tuple, int] = {}
    for v in ids.values():
        counts[v] = counts.get(v, 0) + 1
    top = max(sorted(counts), key=lambda k: counts[k])
    return {
        k: (
            f"card mixes artifacts: this row measured tag={i[0]} sha={i[1]}, "
            f"but the card's artifact is tag={top[0]} sha={top[1]}"
        )
        for k, i in ids.items()
        if i != top
    }


def _requirements() -> list[dict]:
    """Backward findings become card requirements (DEC-497).

    The renderer never invents a requirement: it displays the structured file BACKWARD writes.
    An unreadable file is a BREAKING requirement, not an empty list — absence of evidence is
    never a green.
    """
    if not BACKWARD_REQUIREMENTS.exists():
        return []
    try:
        data = json.loads(BACKWARD_REQUIREMENTS.read_text())
    except Exception as exc:
        return [
            {
                "id": "B-???",
                "severity": "BREAKING",
                "status": "open",
                "finding": f"backward requirements file is unreadable: {exc!r}",
                "forward_subgoal": "repair docs/backward_requirements.json",
            }
        ]
    reqs = data.get("requirements", [])
    return [r for r in reqs if isinstance(r, dict)]


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
    # Loaded before the legs so the single-artifact rule (DEC-448) can see every
    # products-consuming receipt on the card, stage blocks included.
    stages_rc = load(receipts, "judge_stages.json")
    mix = _artifact_mismatch(receipts, stages_rc)
    judge_lines, verdicts = [], {}
    judge_lines += [
        "| Leg | Column | Verdict | Evidence |",
        "|---|---|---|---|",
    ]
    for row in J.LEG_ROWS:
        rc = load(receipts, row.source)
        note = _meta_problem(rc, current_hash)
        if rc and not note:
            note = mix.get(row.source)
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
            block_note = _block_problem(val, current_hash)
            if not block_note:
                block_note = mix.get(f"stage::{s.ident}")
            if block_note:
                verdict, why = J.UNKNOWN, block_note
            else:
                verdict = J.PASS if val["working"] else J.FAIL
                # provenance travels with the number: a reader must be able to see WHICH artifact
                # this block describes without opening the JSON and guessing from the merge order
                st = val["_stamp"]
                why = (
                    _stage_why(val)
                    + f"  ·{st.get('producer', '?')} on {str(st.get('git_commit', ''))[:7]}"
                )
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

    # ---- backward requirements (DEC-497) ---------------------------------------
    reqs = _requirements()
    open_blocking = [
        r
        for r in reqs
        if str(r.get("status", "open")) == "open"
        and str(r.get("severity", "")).upper() in ("BREAKING", "MAJOR")
    ]
    req_lines = [
        "| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |",
        "|---|---|---|---|---|---|---|",
    ]
    if not reqs:
        req_lines.append(
            "| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |"
        )
    else:
        for r in reqs:
            req_lines.append(
                f"| {r.get('id', '?')} | {r.get('severity', '?')} | {r.get('principle', '—')} | "
                f"{r.get('status', '?')} | {str(r.get('finding', ''))[:140]} | "
                f"{str(r.get('forward_subgoal', ''))[:140]} | {r.get('audit_ref', '—')} |"
            )

    # ---- verdict rollup ---------------------------------------------------------
    legs_green = all(verdicts[r.ident] == J.PASS for r in J.LEG_ROWS)
    stages_green = all(verdicts[s.ident] == J.PASS for s in J.STAGE_ROWS)
    if legs_green and stages_green and not open_blocking:
        assessment = (
            "ENCODER DONE (v1): all four legs PASS in both columns, every stage is working by "
            "receipt, and no BREAKING/MAJOR backward or principle requirement is open. Capacity certification "
            "(k_min) is reported separately below."
        )
    else:
        assessment = (
            f"NOT DONE. Legs green: {sum(1 for r in J.LEG_ROWS if verdicts[r.ident] == J.PASS)}"
            f"/{len(J.LEG_ROWS)} · stages green: "
            f"{sum(1 for s in J.STAGE_ROWS if verdicts[s.ident] == J.PASS)}/{len(J.STAGE_ROWS)}"
            f" · backward open: {len(open_blocking)}. "
            "This is a statement about receipts, not about effort; nothing here is a claim of "
            "universality (DEC-410: that word appears only when the coverage rows are green)."
        )

    for r in open_blocking:
        blocked.append(f"backward {r.get('id', '?')} — {r.get('finding', '')}")
    summary["backward_open"] = len(open_blocking)

    md = "\n".join(
        [
            "# REPORT_CARD — PATH B > PATH A",
            "",
            f"*Generated `{generated}` · commit `{git('rev-parse', '--short', 'HEAD')}` "
            f"(`{git('rev-parse', '--abbrev-ref', 'HEAD')}`"
            f"{' · dirty' if _worktree_dirty(git('status', '--porcelain')) else ' · clean'}) "
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
            "## Backward requirements (from PROMPT_BACKWARD)",
            "",
            "Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.",
            "",
            *req_lines,
            "",
            "---",
            "",
            "*The inherited status-quo card is archived at "
            "`docs/archive/REPORT_CARD_INHERITED.md`; it measures the pipeline as it was and is "
            "disclosure only (DEC-497).*",
            "",
        ]
    )
    return md, {"summary": summary, "assessment": assessment, "verdicts": verdicts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipts", default=str(EXP))
    ap.add_argument("--out", default=str(ROOT / "REPORT_CARD.md"))
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args()

    current = behavior_hash()
    md, info = render(Path(a.receipts), current)
    Path(a.out).write_text(md)
    print(f"report card -> {a.out}")
    print(f"behavior hash: {current}")
    print("summary:", json.dumps(info["summary"], sort_keys=True))
    print(info["assessment"])
    if a.do_print:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
