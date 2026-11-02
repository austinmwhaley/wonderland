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

import b_spec as B  # noqa: E402
import judge_spec as J  # noqa: E402

EXP = ROOT / "docs" / "report_cards" / "experiments"
AUDIT_MD = ROOT / "AUDIT.md"
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


def _num(x) -> str:
    """Compact number for the card's detail table; a missing/non-finite value is an em dash."""
    return f"{float(x):.4g}" if _finite(x) else "—"


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


def _get_path(obj, path: str):
    """Resolve a dotted receipt path; list indices are integers. Missing -> None."""
    if not path:
        return obj
    for part in path.split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def _eval_b_row(receipts: Path, row: B.BRow, current_hash: str) -> tuple[str, str]:
    """One B-only row: PASS only on a CURRENT receipt that meets its gate.

    Missing and legacy rows are the honest answer, not a failure to be narrated away. A legacy
    number can be useful context, but it was measured against an encoder that no longer exists,
    so it is UNKNOWN and names the producer that must be re-run.
    """
    if row.kind == "missing" or row.source is None:
        return J.UNKNOWN, f"missing producer: {row.producer}"
    name, path = row.source
    rc = load(receipts, name)
    if rc is None:
        return J.UNKNOWN, f"missing receipt {name}; produce {row.producer}"
    if row.kind == "stage":
        if not isinstance(rc, dict):
            return J.UNKNOWN, f"{name} is malformed (not an object)"
        block = rc.get(path)
        if not isinstance(block, dict):
            return J.UNKNOWN, f"{name}: no {path} block; run {row.producer}"
        note = _block_problem(block, current_hash)
        if note:
            return J.UNKNOWN, note
        if block.get("working"):
            return J.PASS, f"{name}:{path} working=True"
        return J.FAIL, _stage_why(block) or f"{name}:{path} working=False"
    if row.kind == "stage_decomposition":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, note
        if not rc.get("complete"):
            return J.FAIL, f"{name}: stage decomposition incomplete"
        st = rc.get("stages") or {}
        val = rc.get("validation") or {}
        att = rc.get("attribution") or {}

        def sm(stage: str, metric: str) -> str:
            return _num((st.get(stage) or {}).get(metric))

        return (
            J.PASS,
            "complete · persona M0 "
            f"{sm('M0_exact_counter', 'persona_geometry')} → M1 "
            f"{sm('M1_fold_only', 'persona_geometry')} → served "
            f"{sm('M4_served', 'persona_geometry')} · future M0 "
            f"{sm('M0_exact_counter', 'future_geometry')} → M1 "
            f"{sm('M1_fold_only', 'future_geometry')} → served "
            f"{sm('M4_served', 'future_geometry')} · largest drop "
            f"{att.get('persona_geometry', '?')} · M3/M4 "
            f"{_num(val.get('m3_m4_distance_alignment'))}",
        )
    if row.kind == "block_serving":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, note
        if not rc.get("complete"):
            return J.FAIL, f"{name}: block-serving probe incomplete"
        arms = rc.get("arms") or {}
        sens = rc.get("sensitivity_counter_weight") or {}

        def am(arm: str, metric: str) -> str:
            return _num((arms.get(arm) or {}).get(metric))

        vals = [v for v in sens.values() if _finite(v)]
        flat = len(set(vals)) <= 1
        return (
            J.PASS,
            "complete · persona served "
            f"{am('served', 'persona_geometry')} / counter "
            f"{am('exact_counter', 'persona_geometry')} / block "
            f"{am('block_equal_variance', 'persona_geometry')} / PATH A "
            f"{am('path_a', 'persona_geometry')} · future served "
            f"{am('served', 'future_geometry')} / counter "
            f"{am('exact_counter', 'future_geometry')} / block "
            f"{am('block_equal_variance', 'future_geometry')} / PATH A "
            f"{am('path_a', 'future_geometry')} · counter-weight sensitivity "
            + (
                "flat (block scale erased by per-dim z-score; dimension count is the lever)"
                if flat
                else str(sens)
            ),
        )
    if row.kind == "linear_access":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, note
        la = rc.get("linear_access") or {}
        if not la.get("complete"):
            return J.FAIL, f"{name}: linear-access probe incomplete"
        arms = la.get("arms") or {}
        served = (arms.get("served") or {}).get("mean_r2")
        path = (arms.get("path_a") or {}).get("mean_r2")
        if not _finite(served) or not _finite(path):
            return J.UNKNOWN, f"{name}: linear-access values missing"
        delta = float(served) - float(path)
        verdict = J.PASS if delta >= -0.01 else J.FAIL
        return (
            verdict,
            f"served {_num(served)} vs PATH A {_num(path)} "
            f"(delta {delta:+.4f}, margin -0.01; "
            f"n_fields {(arms.get('served') or {}).get('n_fields')})",
        )
    if row.kind == "equal_dimension":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, note
        ed = rc.get("equal_dimension") or {}
        if not ed.get("complete"):
            return J.FAIL, f"{name}: equal-dimension probe incomplete"
        arms = ed.get("arms") or {}

        def em(arm: str, metric: str) -> str:
            return _num((arms.get(arm) or {}).get(metric))

        return (
            J.PASS,
            "complete · persona equal_39 "
            f"{em('equal_39', 'persona_geometry')} / equal_128 "
            f"{em('equal_128', 'persona_geometry')} / counter "
            f"{em('counter_tile_39', 'persona_geometry')} / shape_pca_39 "
            f"{em('shape_pca_39', 'persona_geometry')} / PATH A "
            f"{em('path_a', 'persona_geometry')} · future equal_39 "
            f"{em('equal_39', 'future_geometry')} / equal_128 "
            f"{em('equal_128', 'future_geometry')} / counter "
            f"{em('counter_tile_39', 'future_geometry')} / PATH A "
            f"{em('path_a', 'future_geometry')}",
        )
    if row.kind == "served_object":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, note
        if not rc.get("complete"):
            return J.FAIL, f"{name}: served-object X-ray incomplete"
        arms = rc.get("arms") or {}
        ledger_names = rc.get("ledger_arms") or []
        scored = {
            n: (arms.get(n) or {}).get("persona_geometry")
            for n in ledger_names
            if _finite((arms.get(n) or {}).get("persona_geometry"))
        }
        if not scored:
            return J.UNKNOWN, f"{name}: no ledger arm scored a finite persona geometry"
        best = max(sorted(scored), key=lambda n: scored[n])
        branch = B.o0_branch(scored[best])
        la = (rc.get("linear_access") or {}).get("arms") or {}

        def am(arm: str, metric: str) -> str:
            return _num((arms.get(arm) or {}).get(metric))

        def ra(arm: str) -> str:
            return _num((la.get(arm) or {}).get("mean_r2"))

        return (
            J.PASS,
            "complete · branch "
            f"{branch} (DEC-569: best ledger arm persona "
            f"{am(best, 'persona_geometry')} from {best}; "
            f"full {am('ledger_full', 'persona_geometry')} / "
            f"pca_39 {am('ledger_pca_39', 'persona_geometry')} / "
            f"pca_128 {am('ledger_pca_128', 'persona_geometry')}) · "
            f"references persona counter {am('exact_counter', 'persona_geometry')} / "
            f"shape {am('shape_served', 'persona_geometry')} / PATH A "
            f"{am('path_a', 'persona_geometry')} · "
            f"linear access ledger_full {ra('ledger_full')} / counter {ra('exact_counter')} / "
            f"shape {ra('shape_served')} / PATH A {ra('path_a')}",
        )
    if row.kind == "legacy":
        return J.UNKNOWN, f"legacy receipt {name} (no behavior stamp); re-run {row.producer}"
    note = _meta_problem(rc, current_hash)
    if note:
        return J.UNKNOWN, note
    val = _get_path(rc, path)
    if val is None:
        return J.UNKNOWN, f"{name}:{path} missing"
    if row.kind == "bool":
        return (J.PASS if bool(val) else J.FAIL), f"{name}:{path}={bool(val)}"
    if row.kind == "present":
        return J.PASS, f"{name}:{path} present"
    if not _finite(val):
        return J.UNKNOWN, f"{name}:{path} is not a finite number"
    ok = {
        "le": float(val) <= row.thresh,
        "lt": float(val) < row.thresh,
        "ge": float(val) >= row.thresh,
        "gt": float(val) > row.thresh,
        "eq": float(val) == row.thresh,
    }.get(row.kind)
    if ok is None:
        return J.UNKNOWN, f"unknown B gate kind {row.kind!r}"
    return (J.PASS if ok else J.FAIL), f"{name}:{path}={val:g} (gate {row.kind} {row.thresh:g})"


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


def _task_passes(t: dict, delivery: bool) -> bool:
    """The per-task gate. One definition, shared by the leg summary and the detail table."""
    if not delivery:
        return bool(t["ci_lo"] > 0 and t["brain"] > t["pathA"])
    sst = t.get("sst")
    margin = (
        J.DELIVERY_MARGIN_R2 * float(sst)
        if isinstance(sst, (int, float)) and float(sst) > 0
        else J.DELIVERY_MARGIN_RAW
    )
    point_margin = (
        J.DELIVERY_MARGIN_R2
        if isinstance(sst, (int, float)) and float(sst) > 0
        else J.DELIVERY_MARGIN_RAW
    )
    return bool(t["brain"] >= t["pathA"] - point_margin and t["ci_lo"] >= -margin)


def _task_verdict(t: dict, delivery: bool) -> str:
    """Per-task verdict for the detail table; UNKNOWN for a malformed row, never PASS."""
    if t.get("degenerate"):
        return J.EXCLUDED
    if not all(_finite(t.get(f)) for f in ("brain", "pathA", "ci_lo", "ci_hi", "n")):
        return J.UNKNOWN
    for ctl in ("random", "destroyed"):
        if ctl not in t or not _finite(t[ctl]):
            return J.UNKNOWN
        if t[ctl] > t["brain"]:
            return J.FAIL
    return J.PASS if _task_passes(t, delivery) else J.FAIL


def eval_tasks(tasks, bhash_note: str | None, delivery: bool = False) -> tuple[str, str, list]:
    """Return (verdict, reason, per-task notes) for one leg column's task list.

    Failure attribution is by TASK ORDER: a control failure on a later task may not hide an
    earlier task's CI failure (the price-control bug that masked x0/return in DEC-510).
    """
    if bhash_note:
        return J.UNKNOWN, bhash_note, []
    if not isinstance(tasks, list) or not tasks:
        return J.UNKNOWN, "no tasks recorded in this column", []

    notes, live, failures = [], [], []
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

        control_fail = None
        for ctl in ("random", "destroyed"):
            if ctl not in t or not _finite(t[ctl]):
                return J.UNKNOWN, f"{name}: control '{ctl}' missing (H4)", notes
            if t[ctl] > t["brain"]:
                control_fail = f"{ctl} control >= brain ({t[ctl]} vs {t['brain']})"
                break
        if control_fail:
            failures.append((name, control_fail))
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
        if not control_fail and not _task_passes(t, delivery):
            gate = (
                f"outside the delivery non-inferiority margin ({J.DELIVERY_MARGIN_R2:g} R2)"
                if delivery
                else "does not beat PATH A with CI excluding 0"
            )
            failures.append((name, gate))
        live.append(t)
        notes.append(
            f"{name}: brain {t['brain']:.4g} vs PATH A {t['pathA']:.4g} "
            f"(delta {t['brain'] - t['pathA']:+.4g}, CI [{t['ci_lo']:+.4g}, {t['ci_hi']:+.4g}], n={t['n']})"
        )

    if not live:
        return J.EXCLUDED, "every task degenerate (reported, not forced)", notes
    if failures:
        first_name, first_reason = failures[0]
        names = ", ".join(dict.fromkeys(n for n, _ in failures))
        return J.FAIL, f"{first_name}: {first_reason}; failing: {names}", notes
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
    """Backward findings become card requirements (DEC-505).

    BACKWARD writes ONE file, `AUDIT.md`; its `BACKWARD_REQUIREMENTS_JSON` block is the
    machine-readable gate. The renderer never invents a requirement. A file that exists but
    cannot be parsed is a BREAKING requirement, not an empty list — absence of evidence is
    never a green. A missing file means no review has run yet.
    """
    if not AUDIT_MD.exists():
        return []
    text = AUDIT_MD.read_text(errors="replace")
    start = text.find("<!-- BACKWARD_REQUIREMENTS_JSON")
    if start < 0:
        return [
            {
                "id": "B-???",
                "severity": "BREAKING",
                "status": "open",
                "finding": "AUDIT.md has no BACKWARD_REQUIREMENTS_JSON block",
                "forward_subgoal": "repair AUDIT.md (one .md, one JSON block)",
            }
        ]
    end = text.find("-->", start)
    brace = text.find("{", start)
    if end < 0 or brace < 0 or brace > end:
        return [
            {
                "id": "B-???",
                "severity": "BREAKING",
                "status": "open",
                "finding": "AUDIT.md has a malformed BACKWARD_REQUIREMENTS_JSON block",
                "forward_subgoal": "repair AUDIT.md (one .md, one JSON block)",
            }
        ]
    try:
        data = json.loads(text[brace:end].strip())
    except Exception as exc:
        return [
            {
                "id": "B-???",
                "severity": "BREAKING",
                "status": "open",
                "finding": f"AUDIT.md requirements block is unreadable: {exc!r}",
                "forward_subgoal": "repair AUDIT.md (one .md, one JSON block)",
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
    detail_lines: list[str] = []
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
        verdict, why, _ = eval_tasks(tasks, note, delivery=(row.column == "brain_ledger"))
        verdict, why = record(verdict, why, None if note is None else f"{row.source}")
        if verdict in (J.FAIL,):
            blocked.append(f"{row.source} :: {row.column} — {why}")
        verdicts[row.ident] = verdict
        judge_lines.append(
            f"| {row.leg} | {J.COLUMN_LABEL[row.column]} | {_badge(verdict)} | {why} |"
        )
        if isinstance(tasks, list):
            delivery = row.column == "brain_ledger"
            for t in tasks:
                name = str(t.get("name", "?"))
                brain, path_a = t.get("brain"), t.get("pathA")
                delta = (
                    f"{float(brain) - float(path_a):+.4g}"
                    if _finite(brain) and _finite(path_a)
                    else "—"
                )
                detail_lines.append(
                    f"| {row.leg} | {J.COLUMN_LABEL[row.column]} | {name} | "
                    f"{_num(brain)} | {_num(path_a)} | {delta} | "
                    f"[{_num(t.get('ci_lo'))}, {_num(t.get('ci_hi'))}] | "
                    f"{_num(t.get('random'))} | {_num(t.get('destroyed'))} | "
                    f"{_num(t.get('n'))} | {_badge(_task_verdict(t, delivery))} |"
                )
    if not detail_lines:
        detail_lines.append(
            "| — | — | no task rows (receipts missing or stale) | — | — | — | — | — | — | — | ❔ UNKNOWN |"
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
    # B is evaluated below; the three scope lines are built there so the top-line cannot say
    # "done" while B is uncertified (DEC-563).
    for r in open_blocking:
        blocked.append(f"backward {r.get('id', '?')} — {r.get('finding', '')}")
    summary["backward_open"] = len(open_blocking)

    # ---- B: the foundational scorecard (DEC-562) --------------------------------
    # The judge below is B+C: it can only tell us whether a consumer benefits. This section is
    # B itself. A row PASSes only on a receipt carrying the CURRENT behavior hash and meeting
    # its gate; a legacy or missing receipt is UNKNOWN and names the producer that must run.
    b_lines: list[str] = []
    b_verdicts: dict[str, str] = {}
    b_evals: dict[str, tuple[str, str]] = {}
    b_summary = {"pass": 0, "fail": 0, "unknown": 0, "missing": 0}
    b_sections: dict[str, str] = {}
    for sec in B.SECTIONS:
        rows = [r for r in B.ROWS if r.section == sec.ident]
        evals = {r.ident: _eval_b_row(receipts, r, current_hash) for r in rows}
        b_evals.update(evals)
        vs = [v for v, _ in evals.values()]
        if any(v == J.FAIL for v in vs):
            sec_v = J.FAIL
        elif all(v == J.PASS for v in vs):
            sec_v = J.PASS
        else:
            sec_v = J.UNKNOWN
        b_sections[sec.ident] = sec_v
        b_lines += [
            f"### {sec.ident} — {sec.name} · {_badge(sec_v)}",
            "",
            f"*{sec.question}*",
            "",
            "| # | Property | Metric | Verdict | Evidence / missing producer |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            verdict, why = evals[row.ident]
            b_verdicts[row.ident] = verdict
            if verdict == J.PASS:
                b_summary["pass"] += 1
            elif verdict == J.FAIL:
                b_summary["fail"] += 1
            else:
                b_summary["unknown"] += 1
                if row.kind in ("missing", "legacy"):
                    b_summary["missing"] += 1
            if verdict in (J.UNKNOWN, J.FAIL):
                blocked.append(f"B {row.ident} {row.name} — {why}")
            b_lines.append(
                f"| {row.ident} | {row.name} | {row.metric} | {_badge(verdict)} | {why} |"
            )
        b_lines.append("")
    # ---- B capability top line (O8 sprint; DEC-568/D2) -------------------------
    # The sprint's unit of progress is one capability delta on these axes. Each axis is the
    # WORST verdict of the B rows that own it, so the top line is a pure function of the rows
    # below it: it can never say "done" while a detail row is red, and it cannot be narrated
    # green. B+C stays a consequence section — reported, never gated here.
    cap_lines = [
        "| Capability | B rows | Verdict | Headline |",
        "|---|---|---|---|",
    ]
    cap_verdicts: dict[str, str] = {}
    for axis, idents in B.CAPABILITY_AXES:
        vs = [b_evals.get(i, (J.UNKNOWN, ""))[0] for i in idents]
        if J.FAIL in vs:
            axis_v = J.FAIL
        elif all(v == J.PASS for v in vs):
            axis_v = J.PASS
        else:
            axis_v = J.UNKNOWN
        cap_verdicts[axis] = axis_v
        head = str(b_evals.get(idents[0], (J.UNKNOWN, "no B row evaluated"))[1])
        if len(head) > 120:  # the headline is a pointer, not a quote: cut at a word boundary
            cut = head[:120]
            head = cut[: cut.rfind(" ")].rstrip(",;:") + " …" if cut.rfind(" ") > 0 else cut
        cap_lines.append(f"| {axis} | {', '.join(idents)} | {_badge(axis_v)} | {head} |")
    cap_verdicts_all = all(v == J.PASS for v in cap_verdicts.values())

    b_green_n = sum(1 for v in b_sections.values() if v == J.PASS)
    b_all_green = b_green_n == len(B.SECTIONS)
    legs_green_n = sum(1 for r in J.LEG_ROWS if verdicts[r.ident] == J.PASS)
    stages_green_n = sum(1 for s in J.STAGE_ROWS if verdicts[s.ident] == J.PASS)
    legs_all = legs_green_n == len(J.LEG_ROWS)
    stages_all = stages_green_n == len(J.STAGE_ROWS)

    if b_all_green:
        b_scope = (
            f"✅ **B foundational:** CERTIFIED — {b_green_n}/{len(B.SECTIONS)} sections green."
        )
    else:
        b_scope = (
            f"❌ **B foundational:** NOT CERTIFIED — the current gate. "
            f"{b_green_n}/{len(B.SECTIONS)} sections green · {b_summary['pass']} rows PASS · "
            f"{b_summary['fail']} FAIL · {b_summary['unknown']} UNKNOWN/MISSING."
        )
    if legs_all and stages_all and not open_blocking:
        c_scope = (
            f"✅ **B+C consequence:** DONE (ENCODER DONE v1) — {legs_green_n}/{len(J.LEG_ROWS)} "
            f"legs green · stages {stages_green_n}/{len(J.STAGE_ROWS)} · no BREAKING/MAJOR "
            "backward requirement."
        )
    else:
        c_scope = (
            f"❌ **B+C consequence:** NOT DONE — {legs_green_n}/{len(J.LEG_ROWS)} legs green · "
            f"stages {stages_green_n}/{len(J.STAGE_ROWS)} · backward open: {len(open_blocking)}. "
            "Reported for diagnosis; downstream of B."
        )
    p_scope = "⚪ **Project v1:** not assessed on this card (full stack is a separate scope)."
    assessment_lines = [b_scope, c_scope, p_scope]
    assessment = " | ".join(
        line.replace("**", "").replace("✅ ", "").replace("❌ ", "").replace("⚪ ", "")
        for line in assessment_lines
    )
    summary["b"] = dict(
        b_summary,
        sections_green=b_green_n,
        sections_total=len(B.SECTIONS),
        all_green=b_all_green,
    )

    md = "\n".join(
        [
            "# REPORT_CARD — B foundational, then PATH B > PATH A",
            "",
            f"*Generated `{generated}` · commit `{git('rev-parse', '--short', 'HEAD')}` "
            f"(`{git('rev-parse', '--abbrev-ref', 'HEAD')}`"
            f"{' · dirty' if _worktree_dirty(git('status', '--porcelain')) else ' · clean'}) "
            f"· encoder behavior `{current_hash}`*",
            "",
            "**Assessment:**",
            "",
            *[f"- {line}" for line in assessment_lines],
            "",
            "The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.",
            "Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.",
            "",
            "## B capability — top line (O8 sprint; DEC-568/D2)",
            "",
            "The sprint's unit of progress is one capability delta on these axes. Each axis is the "
            "worst verdict of the B rows that own it, computed from the receipts below — B+C "
            "remains the reported consequence section, never a top-line axis.",
            "",
            *cap_lines,
            "",
            "## B — foundational scorecard (B-only, DEC-562)",
            "",
            "The judge below is the **consequence** (B+C). This section is **B itself**: a row is "
            "PASS only on a current receipt that meets its gate; a missing or legacy receipt is "
            "UNKNOWN and names the producer that must be run. Legacy numbers are not evidence.",
            "",
            *b_lines,
            "## The judge — four legs × two numbers",
            "",
            "Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a "
            f"**non-inferiority** claim vs PATH A (margin {J.DELIVERY_MARGIN_R2:g} R2); a delivery "
            "tie is not a brain win.",
            "",
            *judge_lines,
            "",
            "## Measured values — every task on the card",
            "",
            "The leg verdict above is computed from these rows; a task can PASS while its leg fails "
            "on another task. Every number below is read from the receipts, never narrated.",
            "",
            "| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *detail_lines,
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
            "*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from "
            "`scripts/judge_spec.py`; both read the receipts under `docs/report_cards/experiments/`. "
            "There is no separate audit card, and no row may be parked elsewhere (DEC-512).*",
            "",
        ]
    )
    return md, {
        "summary": summary,
        "assessment": assessment,
        "verdicts": verdicts,
        "b_verdicts": b_verdicts,
        "b_sections": b_sections,
        "cap_verdicts": cap_verdicts,
        "cap_all_green": cap_verdicts_all,
    }


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
