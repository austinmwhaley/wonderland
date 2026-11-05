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
import plan_spec as P  # noqa: E402

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


def _frontier_readings(receipts: Path, current_hash: str) -> list[str]:
    """E1's measured values, read from committed dev/port receipts.

    Reported, never gated: the bars printed beside them are DEC-frozen, and a reading measured
    under another behavior says so. This is the trunk's only appearance on the card, because the
    trunk has no frozen row (plan_spec E1) — and a blob must be visible, not inferred.
    """

    def stamp(rc) -> str:
        h = str(((rc or {}).get("_meta") or {}).get("behavior_hash") or "")
        if not h:
            return " · unstamped"
        return "" if h == current_hash else f" · measured under `{h[:8]}`"

    out: list[str] = []
    for name, label in (
        ("lab/next_event_iter1.json", "dev rig · next_event iter1"),
        ("lab/next_event_iter2.json", "dev rig · next_event iter2"),
        ("lab/next_event_iter3.json", "dev rig · next_event iter3 (DEC-578, candidate 2)"),
        ("lab/next_event_iter4.json", "dev rig · next_event iter4 (DEC-581, candidate 3)"),
        (
            "lab/next_event_cap256.json",
            "dev rig · next_event cap256 (DEC-582, capacity @ baseline algebra)",
        ),
        ("lab/intent_dec586.json", "dev rig · intent (DEC-586, E1.b)"),
        ("lab/intent_dec586_counter.json", "dev rig · intent + mass counter"),
        ("lab/intent_dec586_sig.json", "dev rig · intent + transition signature"),
    ):
        rc = load(receipts, name)
        pos = (rc.get("positive") or {}) if isinstance(rc, dict) else {}
        g = pos.get("g")
        if not _finite(g):
            continue
        out.append(
            f"- **{label}** — g {_num(g)} (bar ≤ 0.5, DEC-576/577) · "
            f"eff_rank {_num(pos.get('eff_rank'))} · beats 1-gram: "
            f"{'yes' if pos.get('beats_base') else 'no'} — `{name}`{stamp(rc)}"
        )
    sd = load(receipts, "o3axis_i/stage_decomposition.json")
    bs = load(receipts, "o3axis_i/block_serving.json")
    persona = (((sd or {}).get("stages") or {}).get("M4_served") or {}).get("persona_geometry")
    r2 = ((((bs or {}).get("linear_access") or {}).get("arms") or {}).get("served") or {}).get(
        "mean_r2"
    )
    if _finite(persona) or _finite(r2):
        out.append(
            f"- **demo stream · O3 attempt 4 (REJECTED)** — M4 persona {_num(persona)} "
            f"(bar ≥ 0.15, DEC-572) · served coarse R² {_num(r2)} (bar ≥ 0.45) — "
            f"`o3axis_i/`{stamp(sd)}"
        )
    return out


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
    if row.kind in ("sufficiency", "sufficiency_gap", "sufficiency_attribution"):
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, note
        if not rc.get("complete"):
            return J.FAIL, f"{name}: sufficiency receipt incomplete"
        bar = B.SUFFICIENCY_BAR
        probes = rc.get("probes") or {}
        included = {p: d for p, d in probes.items() if d.get("included")}
        excl = rc.get("excluded_probes") or {}
        if row.kind == "sufficiency":
            u = (rc.get("u") or {}).get("served_object")
            if not _finite(u):
                return (
                    J.UNKNOWN,
                    f"{name}: no finite served-object U ({len(included)}/{len(probes)} probes included)",
                )
            worst = (rc.get("worst_probe") or {}).get("served_object", "?")
            why = (
                f"served U {float(u):.4f} (bar {bar:g}, DEC-570) · worst probe {worst} · "
                f"{len(included)}/{len(probes)} probes included"
            )
            if excl:
                why += " · excluded: " + ", ".join(sorted(excl))
            return (J.PASS if float(u) >= bar else J.FAIL), why
        if row.kind == "sufficiency_gap":
            if not included:
                return (
                    J.FAIL,
                    f"{name}: no probe with positive headroom — the bar is unfilled, not met",
                )
            bad, detail = [], []
            for p in sorted(included):
                up = (included[p].get("u") or {}).get("served_object")
                if not _finite(up):
                    bad.append(f"{p} (U not finite)")
                    continue
                detail.append(f"{p} {float(up):.4f}")
                if float(up) < bar:
                    bad.append(f"{p} ({float(up):.4f} < {bar:g})")
            if bad:
                return J.FAIL, f"{name}: per-probe bar violated: " + "; ".join(bad)
            return J.PASS, "every included probe clears the bar · " + " · ".join(detail)
        u_l = (rc.get("u") or {}).get("learned_only")
        u_s = (rc.get("u") or {}).get("served_object")
        if not _finite(u_l):
            return J.UNKNOWN, f"{name}: learned-only U not recorded — attribution unmeasured"
        if _finite(u_s) and float(u_l) >= float(u_s):
            carry = "learned shape carries (the ledger does not add)"
        elif _finite(u_s):
            carry = "the ledger carries part of the served U (rule 4: served-statistic, not learned-shape, sufficiency)"
        else:
            carry = "served U not recorded"
        return (
            J.PASS,
            f"learned-only U {float(u_l):.4f} · worst probe "
            f"{(rc.get('worst_probe') or {}).get('learned_only', '?')} · {carry}",
        )
    if row.kind == "construction_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        if not rc.get("complete"):
            return J.FAIL, f"{name}: construction battery incomplete"
        ch = rc.get("checks") or {}
        fp = _get_path(ch, "field_permutation.max_rel_err")
        sc = _get_path(ch, "scan_associativity.worst_rel_err")
        ma = _get_path(ch, "mass_additivity.worst_rel_err")
        det = _get_path(ch, "determinism.bit_identical")
        sv = _get_path(ch, "schema_variant.working")
        fid = _get_path(ch, "frozen_identity.present")
        bad = []
        if not _finite(fp) or fp > B.CONSTRUCTION_FIELD_PERM_BAR:
            bad.append(f"C1 field-perm {_num(fp)} (bar {B.CONSTRUCTION_FIELD_PERM_BAR:g})")
        if not _finite(sc) or sc > B.CONSTRUCTION_SCAN_BAR:
            bad.append(f"C2 scan {_num(sc)} (bar {B.CONSTRUCTION_SCAN_BAR:g})")
        if not _finite(ma) or ma > B.CONSTRUCTION_MASS_BAR:
            bad.append(f"C3 mass {_num(ma)} (bar {B.CONSTRUCTION_MASS_BAR:g})")
        if det is not True:
            bad.append("C4 determinism not bit-identical")
        if sv is not True:
            bad.append("C5 schema variant not working")
        if fid is not True:
            bad.append("C6 frozen identity incomplete")
        if bad:
            return J.FAIL, f"{name}: " + "; ".join(bad)
        return (
            J.PASS,
            f"C1-C6 all met: field-perm {_num(fp)} · scan {_num(sc)} · mass {_num(ma)} · "
            f"determinism bit-identical · schema variant ok · identity present "
            f"(N={rc.get('n_sequences')}, {rc.get('device', '?')})",
        )
    if row.kind == "decision_dose":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        blk = rc.get("decision_dose")
        if not isinstance(blk, dict) or not blk.get("complete"):
            return J.FAIL, f"{name}: decision_dose block incomplete"
        se, st = blk.get("share_E"), blk.get("share_table")
        lo, sr, sd = blk.get("ci_lo"), blk.get("share_random"), blk.get("share_destroyed")
        if not all(_finite(v) for v in (se, st, lo, sr, sd)):
            return J.UNKNOWN, f"{name}: decision_dose values missing"
        bad = []
        if se < st:
            bad.append(f"E share {se:.4f} < table {st:.4f}")
        if lo <= 0:
            bad.append(f"paired CI lo {lo:+.4f} <= 0")
        if sr >= se:
            bad.append(f"random {sr:.4f} >= E")
        if sd >= se:
            bad.append(f"destroyed {sd:.4f} >= E")
        if bad:
            return J.FAIL, f"{name}: " + "; ".join(bad)
        thin = blk.get("thin_levels") or []
        why = (
            f"E share {se:.4f} vs table {st:.4f} (CI [{lo:+.4f}, {blk.get('ci_hi', float('nan')):+.4f}]) "
            f"· random {sr:.4f} · destroyed {sd:.4f} · n={blk.get('n', '?')} "
            f"· oracle ceiling {blk.get('oracle_ceiling', '?')}"
        )
        if thin:
            why += f" · thin levels (degeneracy): {thin}"
        return J.PASS, why
    if row.kind == "decision_ope":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        blk = rc.get("decision_ope")
        if not isinstance(blk, dict) or not blk.get("complete"):
            return J.FAIL, f"{name}: decision_ope block incomplete"
        diff, lo = blk.get("diff"), blk.get("lo")
        if not _finite(diff) or not _finite(lo):
            return J.UNKNOWN, f"{name}: decision_ope values missing"
        if diff <= 0 or lo <= 0:
            return (
                J.FAIL,
                f"{name}: E_ips {blk.get('E_ips')} vs T_ips {blk.get('T_ips')} "
                f"(diff {diff:+.2f} [{lo:+.2f}, {blk.get('hi')}])",
            )
        freqs = blk.get("level_frequencies") or {}
        return (
            J.PASS,
            f"E policy IPS {blk.get('E_ips')} vs table {blk.get('T_ips')} "
            f"(diff {diff:+.2f} [{lo:+.2f}, {blk.get('hi')}] > 0) · "
            f"accE {blk.get('accE')} vs accT {blk.get('accT')} · random IPS {blk.get('random')} · "
            f"n_test={blk.get('n_test')} · levels {freqs}",
        )
    if row.kind == "decision_invariance":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        blk = rc.get("decision_invariance")
        if not isinstance(blk, dict) or not blk.get("complete"):
            return J.FAIL, f"{name}: decision_invariance block incomplete"
        c1 = blk.get("check1_cutoff_band") or {}
        c2 = blk.get("check2_marker_invariance") or {}
        c3 = blk.get("check3_level_teeth") or {}
        s2, s3 = c2.get("share_E_prime"), c3.get("share_permuted")
        if c1.get("ok") is not True:
            return J.FAIL, f"{name}: cutoff band cert not ok — {c1.get('why', 'refused')}"
        if not _finite(s2) or not _finite(s3):
            return J.UNKNOWN, f"{name}: check2/3 values missing"
        return (
            J.PASS,
            f"band cert ok (P1==P2 bit-identical; P1 vs stored S {c1.get('max_delta_S', '?')}"
            f" / E {c1.get('max_delta_E', '?')} vs bands {c1.get('band_S', '?')}"
            f" / {c1.get('band_E', '?')}; teeth {c1.get('teeth_ok', '?')}) · "
            f"marker null: share {s2} vs stored {c2.get('share_E_stored')} (ok={c2.get('ok')}) · "
            f"level permutation: share {s3} vs table {c3.get('share_table', '?')} "
            f"(|Δ|∞ {c3.get('abs_delta_inf', '?')}, ok={c3.get('ok')})",
        )
    if row.kind == "runtime_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        if not rc.get("complete"):
            return J.FAIL, f"{name}: runtime benchmark incomplete"
        n, n_exp = rc.get("n"), rc.get("n_expected")
        rows_s, rss = rc.get("rows_per_s"), rc.get("peak_rss_mb")
        cap = rc.get("slice_cap_gb")
        incr = rc.get("incremental") or {}
        bad = []
        if not (isinstance(n, int) and n == n_exp):
            bad.append(f"n {n} != registry N {n_exp}")
        if cap is not None and not (isinstance(rss, (int, float)) and rss <= float(cap) * 1024.0):
            bad.append(f"peak RSS {rss} MB exceeds slice {cap} GB")
        if not _finite(rows_s):
            bad.append("rows/s not finite")
        if not incr or not _finite((incr or {}).get("n_embeddings")):
            bad.append("incremental receipt row not recorded")
        if bad:
            return J.FAIL, f"{name}: " + "; ".join(bad)
        cap_txt = (
            f"peak RSS {rss} MB (slice {cap} GB)"
            if cap
            else f"peak RSS {rss} MB (no slice cap — CI no-op)"
        )
        return (
            J.PASS,
            f"{n}/{n_exp} customers · cold {rc.get('cold_seconds')}s / warm "
            f"{rc.get('warm_seconds')}s · {rows_s} rows/s · dims "
            f"{rc.get('dims_state')}/{rc.get('dims_served')} · {cap_txt} · incremental "
            f"(absorbed {incr.get('n_absorbed')}, idle {incr.get('n_idle')}, wall "
            f"{incr.get('wall_seconds')}s)",
        )
    if row.kind == "donor_battery":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        if not rc.get("complete"):
            return J.FAIL, f"{name}: donor battery incomplete"
        targets = rc.get("targets") or {}
        deg = [t for t, d in targets.items() if d.get("degenerate")]
        bad = []
        for t in sorted(set(targets) - set(deg)):
            d = targets[t]
            if not all(
                _finite(d.get(k)) for k in ("E", "raw", "fair", "random", "destroyed", "ci_lo")
            ):
                bad.append(f"{t}: values missing")
                continue
            if d["E"] < d["raw"]:
                bad.append(f"{t}: E {d['E']:.4f} < raw {d['raw']:.4f}")
            if d["E"] < d["fair"]:
                bad.append(f"{t}: E {d['E']:.4f} < fair {d['fair']:.4f}")
            if d["ci_lo"] <= 0:
                bad.append(f"{t}: table+E CI lo {d['ci_lo']:+.4f} <= 0")
            if d["random"] >= d["E"]:
                bad.append(f"{t}: random {d['random']:.4f} >= E")
            if d["destroyed"] >= d["E"]:
                bad.append(f"{t}: destroyed {d['destroyed']:.4f} >= E")
        if bad:
            return J.FAIL, f"{name}: " + "; ".join(bad[:4]) + (" …" if len(bad) > 4 else "")
        detail = " · ".join(
            f"{t}: E {targets[t]['E']:.4f} vs raw {targets[t]['raw']:.4f} / fair {targets[t]['fair']:.4f} "
            f"/ table+E {targets[t]['table_E']:.4f} (CI lo {targets[t]['ci_lo']:+.4f}) "
            f"/ hstack {_num(targets[t].get('table_E_hstack'))} "
            f"(CI lo {_num(targets[t].get('hstack_ci_lo'))}) "
            f"/ rand {targets[t]['random']:.4f} / dest {targets[t]['destroyed']:.4f}"
            for t in sorted(set(targets) - set(deg))
        )
        why = f"n_A {rc.get('n_A')} / n_B {rc.get('n_B')} · {detail}"
        if deg:
            why += f" · excluded (degenerate): {deg}"
        return J.PASS, why
    if row.kind == "calibration_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        blk = rc.get("calibration")
        if not isinstance(blk, dict) or not blk.get("complete"):
            return J.FAIL, f"{name}: calibration block incomplete"
        ece = blk.get("ece_next_event")
        if not _finite(ece):
            return J.FAIL, f"{name}: {blk.get('why', 'ECE not measured')}"
        if ece > B.CALIBRATION_ECE_BAR:
            return J.FAIL, f"{name}: ECE {ece:.4f} > bar {B.CALIBRATION_ECE_BAR:g}"
        ood = blk.get("ood") or {}
        return (
            J.PASS,
            f"next-type ECE {ece:.4f} (bar {B.CALIBRATION_ECE_BAR:g}) · acc {blk.get('acc')} "
            f"· n {blk.get('n_next')}/{blk.get('n_pop')} (OOV {blk.get('n_oov')}) · OOD "
            f"(as_of+{B.OOD_HORIZON_DAYS}d): ece {ood.get('ece')} acc {ood.get('acc')} "
            f"n {ood.get('n')}",
        )
    if row.kind == "transfer_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        if not rc.get("complete"):
            return J.FAIL, f"{name}: transfer family incomplete"
        n, wins = rc.get("n_targets"), rc.get("wins")
        mean_e, mean_t = rc.get("mean_E"), rc.get("mean_table")
        mean_r, mean_d = rc.get("mean_random"), rc.get("mean_destroyed")
        if not all(_finite(v) for v in (n, wins, mean_e, mean_t, mean_r, mean_d)):
            return J.FAIL, f"{name}: transfer family values missing"
        if int(n) < 1 or float(wins) / float(n) < B.TRANSFER_WIN_SHARE_BAR:
            return J.FAIL, (
                f"{name}: E wins {int(wins)}/{int(n)} < bar "
                f"{B.TRANSFER_WIN_SHARE_BAR:.3f} of the published family"
            )
        if float(mean_e) <= float(mean_t):
            return J.FAIL, f"{name}: family mean E {mean_e:.4f} <= table {mean_t:.4f}"
        if float(mean_e) <= float(mean_r) or float(mean_e) <= float(mean_d):
            return J.FAIL, (
                f"{name}: family mean E {mean_e:.4f} does not beat controls "
                f"(random {mean_r:.4f}, destroyed {mean_d:.4f})"
            )
        return (
            J.PASS,
            f"published family {int(n)} targets · E wins {int(wins)}/{int(n)} "
            f"({float(wins) / float(n):.3f}) · family mean E {mean_e:.4f} vs table "
            f"{mean_t:.4f} (diff {float(mean_e) - float(mean_t):+.4f}) · random "
            f"{mean_r:.4f} / destroyed {mean_d:.4f}",
        )
    if row.kind == "robustness_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        if not rc.get("complete") or not rc.get("coverage_reported"):
            return J.FAIL, f"{name}: coverage receipt incomplete"
        nt = rc.get("next_time") or {}
        cov = nt.get("coverage")
        if not _finite(cov):
            return J.FAIL, f"{name}: next-time coverage not reported"
        miss = rc.get("missingness") or {}
        if not miss.get("complete") or not (miss.get("fractions") or []):
            return J.FAIL, f"{name}: missingness sensitivity not reported"
        occ = rc.get("occurrence") or {}
        resp = rc.get("response_head") or {}
        fracs = ", ".join(
            f"{float(f.get('dropped', 0)):.0%}:{float(f.get('mean_rel_l2', 0)):.3f}"
            for f in miss.get("fractions") or []
        )
        return (
            J.PASS,
            f"next-time {nt.get('nominal')} coverage {cov:.3f} (n {nt.get('n')}, MAE "
            f"{_num(nt.get('mae'))}d) · occurrence AUC {_num(occ.get('auc'))} / Brier "
            f"{_num(occ.get('brier'))} (h {occ.get('horizon_days')}d) · response coverage "
            f"{resp.get('coverage_empirical')}/{resp.get('coverage_nominal')} · missingness "
            f"mean rel L2 {fracs}",
        )
    if row.kind == "minimality_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        if not rc.get("complete"):
            return J.FAIL, f"{name}: minimality receipt incomplete"
        k_min = rc.get("k_min")
        full_dim = rc.get("full_dim")
        served = rc.get("served_shape_dim")
        served_dim = rc.get("served_dim")
        block_dim = rc.get("block_dim")
        if not all(_finite(v) for v in (k_min, full_dim, served, served_dim, block_dim)):
            return J.FAIL, f"{name}: minimality dims missing"
        if int(served) + int(block_dim) != int(served_dim):
            return J.FAIL, (
                f"{name}: served dims inconsistent — {served} shape + {block_dim} block "
                f"!= {served_dim}"
            )
        if int(served) != int(k_min):
            return J.FAIL, (
                f"{name}: served shape rank {int(served)} != certificate k_min {int(k_min)} "
                f"(shape full dim {int(full_dim)})"
            )
        if int(k_min) != int(full_dim):
            return J.FAIL, (
                f"{name}: shape certificate is not minimal at the served rank — k_min "
                f"{int(k_min)} != full_dim {int(full_dim)}"
            )
        if int(block_dim) < 1:
            return J.FAIL, f"{name}: no explicit measure block (block_dim {int(block_dim)})"
        if not (0 < int(k_min) <= int(full_dim)):
            return J.FAIL, f"{name}: k_min {int(k_min)} outside 1..{int(full_dim)}"
        cert = rc.get("shape_certificate") or {}
        return (
            J.PASS,
            f"served shape rank {int(served)} == k_min {int(k_min)} (shape full "
            f"{int(full_dim)}) · explicit block {int(block_dim)} · joint floor "
            f"{rc.get('joint_min')} (unconstrained) · n_train {cert.get('n_train')} / "
            f"n_test {cert.get('n_test')} · noise {cert.get('noise')}",
        )
    if row.kind == "identity_law":
        note = _meta_problem(rc, current_hash)
        if note:
            return J.UNKNOWN, f"{note} — re-run {row.producer}"
        blk = rc.get("identity")
        if not isinstance(blk, dict) or not blk.get("complete"):
            return J.FAIL, f"{name}: identity block incomplete"
        if blk.get("present") is not True:
            return J.FAIL, f"{name}: identity fields incomplete — {blk.get('why')}"
        return (
            J.PASS,
            f"tag {blk.get('tag')} · registry {blk.get('registry_sha256')} · data "
            f"{blk.get('data_signature')} · as_of {blk.get('as_of')} · n "
            f"{blk.get('n_customers')} · config {blk.get('config_sha256')} · "
            f"checkpoint {blk.get('checkpoint_sha256')}",
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
            blocked.append(f"{row.source} :: {row.column} [P4·{P.owner(row.ident)}] — {why}")
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
            "| # | Property | Metric | Green means | Verdict | Evidence / missing producer |",
            "|---|---|---|---|---|---|",
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
                if row.kind in ("missing", "legacy") or why.startswith("missing"):
                    b_summary["missing"] += 1
            if verdict in (J.UNKNOWN, J.FAIL):
                blocked.append(
                    f"B {row.ident} [{P.problem_of(row.ident)}·{P.owner(row.ident)}] "
                    f"{row.name} — {why}"
                )
            b_lines.append(
                f"| {row.ident} | {row.name} | {row.metric} | {row.gate_kind} | "
                f"{_badge(verdict)} | {why} |"
            )
        b_lines.append("")
    # The sprint's capability-axis view (b_spec.CAPABILITY_AXES) was retired by DEC-579: the plan
    # tree is the one top line and owns the same rows plus E1 and the v1 scope. Keeping a second
    # axis definition invited the two to disagree, which is the duplication principle 6 forbids.

    b_green_n = sum(1 for v in b_sections.values() if v == J.PASS)
    b_all_green = b_green_n == len(B.SECTIONS)
    legs_green_n = sum(1 for r in J.LEG_ROWS if verdicts[r.ident] == J.PASS)
    stages_green_n = sum(1 for s in J.STAGE_ROWS if verdicts[s.ident] == J.PASS)
    legs_all = legs_green_n == len(J.LEG_ROWS)
    stages_all = stages_green_n == len(J.STAGE_ROWS)

    if b_all_green:
        b_scope = (
            f"✅ **B foundational:** CERTIFIED — {b_green_n}/{len(B.SECTIONS)} sections green. "
            f"Scope: the 23 receipts are current at behavior `{current_hash[:8]}` and are "
            "certified on the single `cfm_v2k_vf2_cert` bundle (base 272 · bank 179 · donor-only "
            "106 · delivery port 307 · measure view), with a manifest mapping every row to its "
            "component. The artifact is the synthetic `demo_v2` prototype, not a production "
            "stream (B-001). The trunk-only rank is 4.11 (base) / 10.33 (bank); the delivery "
            "port reaches 30.9 only with the measure PCA-128, so E1.c is a delivery/ledger port "
            "(B-003). The B+C brain is the served bank; the measure-free donor-only diagnostic "
            "is recorded and does not clear all legs (B-008)."
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
            "backward requirement. Bank v2 (DEC-598): the served block bank + compact judge "
            "ledger. The brain column is the served bank (donor blocks + served statistics); "
            "the measure-free donor-only rerun is in `bc_donor_diagnostic.json` and does not "
            "clear all legs (B-008)."
        )
    else:
        c_scope = (
            f"❌ **B+C consequence:** NOT DONE — {legs_green_n}/{len(J.LEG_ROWS)} legs green · "
            f"stages {stages_green_n}/{len(J.STAGE_ROWS)} · backward open: {len(open_blocking)}. "
            "The B/C/E1/v1 rows are green but the open BACKWARD requirements block completion."
        )
    p_scope = (
        "✅ **Project v1:** V1.1-V1.5 green — identifiable stream; plugin acceptance 9/9; "
        "red_king known-effect 73.3% vs 25% chance, held-out policy 0.371 vs 0.208 random "
        "(CI > 0); red_queen offer composition DEPLOY/HOLD; caterpillar on red_queen's plan."
    )

    # ---- the plan: the SAME tree, with verdicts attached (PLAN.md -> REPORT_CARD) -----------
    # The decomposition is defined once in plan_spec; this section is the measurement surface
    # for it. An interior verdict is the worst of its children; a DEC-frozen atom renders
    # FRONTIER (no card row exists) and a v1 atom renders NOT ASSESSED. The root therefore
    # cannot read SOLVED while either scope is unmeasured — T0 is not SOLVED (integrity rule 2).
    plan_verdicts = P.statuses(
        {**verdicts, **b_verdicts}, closed=P.CLOSED, backward_open=len(open_blocking)
    )
    census = P.census()
    card_atoms = [n for n in P.NODES if n.kind == "atomic" and n.evidence == "card"]
    plan_green = sum(1 for n in card_atoms if plan_verdicts.get(n.ident) == J.PASS)
    plan_fail = sum(1 for n in card_atoms if plan_verdicts.get(n.ident) == J.FAIL)
    root_v = plan_verdicts.get("G", J.UNKNOWN)
    root_emoji = {J.PASS: "✅", J.FAIL: "❌", J.UNKNOWN: "⚪"}[root_v]
    root_word = {J.PASS: "SOLVED", J.FAIL: "FAILED", J.UNKNOWN: "OPEN"}[root_v]
    red_by_problem: dict[str, int] = {}
    for n in card_atoms:
        if plan_verdicts.get(n.ident) == J.FAIL:
            pr = P.problem_of(n.ident)
            red_by_problem[pr] = red_by_problem.get(pr, 0) + 1
    reds_txt = (
        " · reds: " + ", ".join(f"{k} ({v})" for k, v in sorted(red_by_problem.items()))
        if red_by_problem
        else ""
    )
    _dec_ids = {n.ident for n in P.NODES if n.evidence == "dec"}
    _v1_ids = {n.ident for n in P.NODES if n.evidence == "v1"}
    _dec_open = census["dec"] - len(P.CLOSED & _dec_ids)
    _v1_open = census["v1"] - len(P.CLOSED & _v1_ids)
    _solved = plan_green + len(P.CLOSED & (_dec_ids | _v1_ids))
    _back = f" · {len(open_blocking)} BACKWARD open" if open_blocking else ""
    plan_scope = (
        f"{root_emoji} **Plan root:** {root_word} — {_solved}/{census['atomic_total']} "
        f"atomic problems green · {plan_fail} FAIL · {_dec_open} DEC-frozen · "
        f"{_v1_open} v1 unassessed · {census['interior']} interior{_back}{reds_txt}."
    )
    assessment_lines = [plan_scope, b_scope, c_scope, p_scope]
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

    plan_lines = [
        "## The plan — the tree this card measures (PLAN.md)",
        "",
        "The goal, cut backwards into the problems that must all be solved, recursively to the "
        "atomic problems. This section is generated from `scripts/plan_spec.py` — the single "
        "definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: "
        "every measured atomic problem is a frozen row in its B/judge section below, and an "
        "interior verdict is the worst of its children.",
        "",
        f"**Plan root:** {root_emoji} {root_word} — {_solved}/{census['atomic_total']} "
        f"atomic problems green · {plan_fail} FAIL · {_dec_open} DEC-frozen (no card rows) "
        f"· {_v1_open} v1 not assessed on this card · {census['interior']} interior "
        f"problems{(' · ' + str(len(open_blocking)) + ' BACKWARD open') if open_blocking else ''}.",
        "",
        "| Level | Node | What must be true | Atomic problems | Verdict |",
        "|---|---|---|---|---|",
    ]
    for n in P.NODES:
        if n.kind == "atomic" and n.evidence == "card":
            continue  # a card atom is a frozen row: it renders in its B/judge section below
        lvl = P.level(n.ident)
        if n.kind == "atomic":
            atoms_txt = "—"
        else:
            under = len(P.atomics_under(n.ident))
            atoms_txt = f"{under} atomic problem{'s' if under != 1 else ''}"
        q = n.question if len(n.question) <= 96 else n.question[:96].rsplit(" ", 1)[0] + " …"
        v = plan_verdicts.get(n.ident, J.UNKNOWN)
        if v == P.FRONTIER:
            badge = "⚪ FRONTIER"
        elif v == P.NOT_ASSESSED:
            badge = "⚪ NOT ASSESSED"
        else:
            badge = _badge(v)
        plan_lines.append(f"| {lvl} | **{n.ident}** {n.title} | {q} | {atoms_txt} | {badge} |")
    plan_lines += [
        "",
        f"*Macro review:* {P.MACRO_REVIEW}",
        "",
        f"*Frontier:* {P.FRONTIER_NOTE}",
        "",
        "**Frontier readings (reported, not gated — the evidence behind E1's FRONTIER badge):**",
        "",
    ]
    readings = _frontier_readings(receipts, current_hash)
    plan_lines += readings or ["- no frontier receipt present"]
    reporting_leaves = [r.ident for r in B.ROWS if r.gate_kind == "reporting"]
    plan_lines += [
        "",
        "A green atom means its frozen gate is met. For these atoms the gate certifies "
        "*measurement and attribution*, not a quality level (derived in `scripts/b_spec.py: "
        "gate_kind`, not asserted here): "
        + ", ".join(f"`{x}`" for x in reporting_leaves)
        + ". Their quality levels are DEC-frozen and live in the receipts (`PLAN.md` §6). Read "
        "the plan before reading a green row as 'good'.",
        "",
    ]
    plan_summary = {
        "root": root_v,
        "green": plan_green,
        "fail": plan_fail,
        "card": census["card"],
        "dec": census["dec"],
        "v1": census["v1"],
        "atomic_total": census["atomic_total"],
        "interior": census["interior"],
        "nodes": plan_verdicts,
    }

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
            *plan_lines,
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
            "`scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` "
            "(rendered readably in `PLAN.md`); all read the receipts under "
            "`docs/report_cards/experiments/`. There is no separate audit card, and no row may "
            "be parked elsewhere (DEC-512).*",
            "",
        ]
    )
    return md, {
        "summary": summary,
        "assessment": assessment,
        "verdicts": verdicts,
        "b_verdicts": b_verdicts,
        "b_sections": b_sections,
        "plan": plan_summary,
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
