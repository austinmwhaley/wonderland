#!/usr/bin/env python3
"""S6: the card must be rebuildable by ONE repo command, at a recorded commit, with hashes (rule 6).

    python3 scripts/measure_certification_law.py [--products DIR] [--demo DIR]

Why this file exists. `report_card.py` renders the card from whatever receipts happen to be on
disk, so "I regenerated the card" was never evidence about the card: its five stage blocks were
last written by four different scripts, at four different times, two of them against `/tmp` inputs
that no longer exist (DEC-430 found exactly this). A card nobody can rebuild from a commit is not
a certificate — it is a transcript. This is the single command the S6 gate demands, and it
measures whether that command works, in the order the dependencies require:

    S1 S2  measure_input_law        universal input, vocabulary
    S3     measure_dynamics_law     the engine's laws
    S4     measure_views_law        the outlets
    S5     measure_battery_law      the objective battery
    legs   measure_bc_bank          the four-leg top line, both columns (bank v2, DEC-598;
                                    rl/simulator delegate to the judge_law LEGs)
    prov   measure_path_a_provenance  the comparator's per-column provenance (DEC-446)
    B      the thirteen B-row producers  the receipts the B scorecard reads (CHAIN_B)
    S6     this file                the regeneration itself
    card   report_card              rendered last, from the receipts above

Stage B (CHAIN_B) exists because the B scorecard reads thirteen receipts from producers this
chain never ran: every behavior change left 15-20 B rows UNKNOWN until an operator re-ran them by
hand (three times). It is orchestration only — every producer keeps its own gates, bars, defaults
and semantics. Each entry declares the artifact its frozen protocol names: the object under
certification (the passed `--products`), or a sibling representation (the 272-dim served base, the
59-dim donor slice, the 48-dim compact served object, the 307-dim delivery port) derived from the
products tree's parent. A derived artifact that is absent SKIPS its producer with a printed note;
it never fails the chain and it is never substituted by the bank — measuring the wrong object is
the defect class this stage exists to remove. `--skip-slow-b` skips the two slowest producers
(runtime ~7.5 min, sufficiency ~1 min); the default is to RUN them, because one command
regenerating the card is the point.

What "regenerated" has to mean, as checks rather than prose: the whole chain ran in ONE invocation
and every producer exited 0; the card renders from those receipts deterministically (twice,
byte-identical apart from its own timestamp line); the commit is recorded and the tree is clean
apart from the files this command is authorized to rewrite; the inputs the receipts describe are
hashed; and every row the card publishes reports its N.

Like every other producer here, this file records measurements. It contains no verdict string, and
`report_card.py` — not this file — decides what S6 means (certification rule 3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import judge_spec as JS  # noqa: E402  (the one definition of the certified legs)
import report_card as RC  # noqa: E402  (one definition of the card, and of git state)
import safe_run as SR  # noqa: E402  (one definition of the memory cap)
from measure_input_law import _rollup, stamp_merge_write  # noqa: E402  (one definition)

EXP = ROOT / "docs/report_cards/experiments"
CARD = ROOT / "REPORT_CARD.md"
ALL_LEGS = "supervised,unsupervised,rl,simulator"

#: the chain, in dependency order. Each entry is (label, script, argv template, the stage blocks
#: that producer is required to write). Placeholders are filled from the run's paths, so the chain
#: is one definition rather than five copies of a shell history.
CHAIN = (
    ("S1+S2", "measure_input_law.py", ("--data", "{demo}"), ("S1", "S2")),
    (
        "S3",
        "measure_dynamics_law.py",
        ("--ckpt-dir", "{products}", "--db", "{stream}", "--train-log", "{train_log}"),
        ("S3",),
    ),
    ("S4", "measure_views_law.py", ("--products", "{products}", "--demo", "{demo}"), ("S4",)),
    ("S5", "measure_battery_law.py", ("--products", "{products_db}", "--demo", "{demo}"), ("S5",)),
    (
        "legs",
        # bank v2 (DEC-598): bc_bank owns the supervised/unsupervised legs; it delegates the
        # rl/simulator legs to the judge_law LEGs, so one producer stamps all four files.
        "measure_bc_bank.py",
        ("--products", "{products}", "--demo", "{demo}", "--legs", "{legs}"),
        (),
    ),
    (
        "prov",
        "measure_path_a_provenance.py",
        ("--demo", "{demo}", "--products-dir", "{products}", "--out", "{prov_out}"),
        (),
    ),
)


class BProducer(NamedTuple):
    """One B-scorecard producer and the artifact its frozen protocol names.

    `needs` is "products" (the object this invocation certifies) or a sibling representation:
    "base" (272-dim served base = trunk 256 + level 16), "c32rvd" (59-dim served donor slice),
    "compact" (48-dim compact served object = 32 shape + 16 level), "port" (the 307-dim delivery
    port = bank + measure PCA-128). A missing sibling SKIPS the producer with a printed note; it
    never fails the chain, and it never falls back to a different representation (the row then
    reads UNKNOWN, visibly — measuring the wrong object is worse than measuring nothing).
    `receipt` is the filename the card reads, so `--out` is explicit rather than left to a default.
    """

    label: str
    script: str
    needs: str
    receipt: str
    argv: tuple[str, ...]


#: Stage B, in dependency order. These thirteen receipts are what the B scorecard reads; the
#: arguments are the producers' own frozen protocols (DEC-585/595/655/656), not preferences:
#:   decision/sufficiency/served-object  the 272-dim served base
#:   donor/transfer                      the 59-dim served donor slice
#:   minimality                          the 48-dim compact served object the certificate certifies
#:   stage/block_serving                 the 307-dim delivery port (bank + measure PCA-128), so the
#:                                       B3 blocking rows are measured on the object they certify
#:   encoder_win                         the served bank, with the ADOPTED logistic gate head
#: Everything else is the object under certification (normally the served bank). A missing
#: representation SKIPS its producer; it is never substituted by another (that defect class —
#: measuring the wrong artifact — is exactly what makes a re-stamp destroy the card).
CHAIN_B = (
    BProducer(
        "b_props",
        "measure_b_props.py",
        "products",
        "encoder_b_props.json",
        ("--products", "{products}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "construction",
        "measure_construction_law.py",
        "products",
        "encoder_lawfulness.json",
        ("--products", "{products}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "decision",
        "measure_decision_law.py",
        "base",
        "encoder_decision_law.json",
        ("--products", "{base}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "donor",
        "measure_donor.py",
        "c32rvd",
        "encoder_donor.json",
        ("--products", "{c32rvd}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "minimality",
        "measure_minimality_law.py",
        "compact",
        "encoder_minimality.json",
        ("--products", "{compact}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "robustness",
        "measure_robustness_law.py",
        "base",
        "encoder_robustness_law.json",
        ("--products", "{base}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "runtime",
        "measure_runtime.py",
        "products",
        "runtime.json",
        ("--products", "{products}", "--out", "{b_out}"),
    ),
    BProducer(
        "transfer",
        "measure_transfer_law.py",
        "c32rvd",
        "encoder_transfer_law.json",
        ("--products", "{c32rvd}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "sufficiency",
        "measure_sufficiency_law.py",
        "base",
        "encoder_sufficiency_law.json",
        ("--products", "{base}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "served_object",
        "measure_served_object.py",
        "base",
        "encoder_served_object.json",
        ("--products", "{base}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "stage",
        "measure_stage_decomposition.py",
        "port",
        "encoder_stage_decomposition.json",
        ("--products", "{port}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "block_serving",
        "measure_block_serving.py",
        "port",
        "encoder_block_serving.json",
        ("--products", "{port}", "--demo", "{demo}", "--out", "{b_out}"),
    ),
    BProducer(
        "encoder_win",
        "measure_encoder_win.py",
        "products",
        "encoder_win.json",
        # DEC-649: the adopted gate head is the canonical multinomial logistic; the receipt's
        # protocol must match the gate it certifies.
        ("--products", "{products}", "--demo", "{demo}", "--out", "{b_out}", "--head", "logistic"),
    ),
)

#: The two producers whose runtime dominates the stage (~7.5 min / ~1 min); skipped only on ask.
SLOW_B = frozenset({"runtime", "sufficiency"})


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _artifact(path: Path) -> dict:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _has_products(path: Path) -> bool:
    """A products DIRECTORY, by the artifact it cannot run without (not just `is_dir`)."""
    return (path / "cfm_products.duckdb").is_file()


def _generated_at(path: Path) -> str:
    """A receipt's own freshness stamp; '' when absent, unreadable, or unstamped."""
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    return str((body.get("_meta") or {}).get("generated_at", "")) if isinstance(body, dict) else ""


def _run_producer(script: str, argv: list[str], cap_gb: float) -> dict:
    """One producer, as a real child process, inside its own hard memory cap.

    The chain has to survive being invoked the way an operator invokes it, from a clean shell, not
    only as an import inside this interpreter — a chain that only works in-process is a chain that
    will silently stop working the first time a producer gains a `main()` side effect.

    It also has to be uncrashable. Each child is launched through `safe_run`'s cgroup cap, so a
    producer that runs away is killed by the kernel at the cap and the machine keeps running; the
    chain then reports that producer as failed instead of half-writing a receipt.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / script), *argv]
    run_cmd, mechanism = SR.capped_command(cmd, cap_gb)
    t0 = time.perf_counter()
    proc = subprocess.run(run_cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    return {
        "command": " ".join([script, *argv]),
        "exit": proc.returncode,
        "seconds": round(time.perf_counter() - t0, 1),
        "last_line": tail[-1][:160] if tail else "",
        "cap_gb": round(cap_gb, 1),
        "cap_mechanism": mechanism,
    }


def changed_paths(porcelain: str) -> list[str]:
    """Paths from `git status --porcelain -z` output, exactly.

    `-z` terminates each entry with NUL and never quotes, so no status/path boundary is guessed.
    The previous version sliced `line[3:]` on the line-oriented form, which was wrong for the
    FIRST line of the output: `report_card.git()` strips the whole stdout, eating the leading space
    of " M path" and shifting every later slice by one — measured, it reported a modified file as
    `ECISIONS.md`. A parser that mangles the path of the only file it names is not a check.
    """
    out = []
    for entry in porcelain.split("\0"):
        if len(entry) < 4:
            continue
        path = entry[3:]
        if " -> " in path:  # rename: the destination is the file that exists now
            path = path.split(" -> ")[-1]
        out.append(path)
    return out


def _tree_dirty(exempt: set[str]) -> tuple[bool, list]:
    """Working-tree state, excluding the artifacts a regeneration is allowed to rewrite.

    These are committed generated files: a command whose whole job is to rewrite them can never
    report a clean tree *after* it ran, and a rule that can never be satisfied is the rule DEC-430
    removed for the card's own bytes. Source edits still count, which is the whole point.
    """
    raw = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    changed = [p for p in changed_paths(raw) if Path(p).name not in exempt]
    return bool(changed), changed


def _rows_report_n(receipts: Path) -> tuple[int, list]:
    """Count the live leg rows the card will publish, and name any that carry no N.

    H3 enforces N at render time; measuring it here makes S6 a statement about the regenerated
    receipts rather than about the renderer's good intentions.

    Only the four certified gate-of-record files are read, by name from `judge_spec.LEGS`. A glob
    would also pick up `judge_leg_v1flat_<leg>.json` — `measure_judge_law`'s v1-flat DIAGNOSTIC —
    and count diagnostic rows as live leg rows (the receipt-ownership change that moved it off the
    gate filenames).
    """
    rows, missing = 0, []
    for leg in JS.LEGS:
        f = receipts / f"judge_leg_{leg}.json"
        if not f.exists():
            continue
        body = json.loads(f.read_text())
        for col, block in (body.get("columns") or {}).items():
            for t in block.get("tasks", []):
                if t.get("degenerate"):
                    continue
                rows += 1
                if not t.get("n"):
                    missing.append(f"{f.stem}::{col}::{t.get('name', '?')}")
    return rows, missing


STAGES = ("S1", "S2", "S3", "S4", "S5")


def chain_verdict(
    exits: dict[str, int],
    fresh_stages: set,
    want_stages: set,
    fresh_legs: set,
    want_legs: set,
    n_producers: int,
) -> tuple[bool, dict]:
    """Did the chain RUN, as distinct from did it measure everything green? (DEC-441)

    Three outcomes are routinely confused, and only one of them is a broken chain:
    * exit 0 — ran, every check passed;
    * exit 1 — ran, at least one check FAILED. That is a measurement, and the card is supposed to
      carry failing rows: `measure_views_law` documents exactly this contract.
    * exit >= 2, or no fresh artifact — the producer did not run. THIS is the chain failing.
    A gate that demanded exit 0 would push producers to report green in order to look healthy,
    which is the incentive this whole card exists to remove.
    """
    fatal = {k: v for k, v in exits.items() if v >= 2}
    missing = sorted(want_stages - fresh_stages) + sorted(want_legs - fresh_legs)
    ok = not fatal and not missing and fresh_legs == want_legs and len(exits) == n_producers
    return ok, {
        "fatal": fatal,
        "missing": missing,
        "reported_failing": sorted(k for k, v in exits.items() if v == 1),
    }


def comparator_provenance_fact(prov_body: dict | None, fresh: bool) -> dict:
    """The S6 check over the comparator's provenance receipt (DEC-446), as a pure predicate.

    A table computed with the future included would make every "does not beat PATH A" an
    artifact of the comparison, not a finding (DEC-437). The audit producer rebuilds the table
    from the lab's own inputs and verifies the builder's per-column stamp against it; this says
    what that receipt means: it must EXIST, be FRESH, and have every sub-check working.
    """
    if prov_body is None:
        return {
            "working": False,
            "evidence": "no receipt — the provenance producer did not write it",
        }
    sub = prov_body.get("checks", {})
    failed = sorted(k for k, c in sub.items() if not c.get("working"))
    return {
        "working": bool(prov_body.get("working")) and fresh and bool(sub),
        "evidence": (
            f"{len(prov_body.get('comparator_columns', []))} comparator column(s), each stamped "
            f"pre-`as_of` by the builder and confirmed against a rebuild from the lab's own "
            f"relational inputs (as_of {prov_body.get('as_of')}); sub-checks {sorted(sub)}"
            + (f"; FAILED: {failed}" if failed else "")
            + ("" if fresh else "; STALE (written before this run)")
        ),
    }


def _diff_line(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a.splitlines(), b.splitlines())):
        if x != y:
            return i
    return -1


def main() -> int:
    SR.ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=str(ROOT / "looking_glass/artifacts/cfm"))
    ap.add_argument("--demo", default=str(ROOT / "rabbit_hole/data"))
    ap.add_argument(
        "--train-log", default="", help="defaults to <products parent>/cfm_train_baseline.log"
    )
    ap.add_argument("--legs", default=ALL_LEGS)
    ap.add_argument(
        "--floor-gb",
        type=float,
        default=SR.DEFAULT_FLOOR_GB,
        help="system memory to leave free; each producer is capped at what remains",
    )
    ap.add_argument("--receipts", default=str(EXP))
    ap.add_argument("--out", default=str(EXP / "judge_stages.json"))
    ap.add_argument("--card", default=str(CARD))
    ap.add_argument(
        "--skip-slow-b",
        action="store_true",
        help="skip the two slow stage-B producers (runtime ~7.5 min, sufficiency ~1 min); "
        "the default is to run them so one command regenerates the whole card",
    )
    a = ap.parse_args()

    products = Path(a.products)
    products_db = products / "cfm_products.duckdb"
    if products.is_file():
        products_db, products = products, products.parent
    demo = Path(a.demo)
    receipts, card, stages_out = Path(a.receipts), Path(a.card), Path(a.out)
    stream = demo / "customer_event_stream.parquet"
    truth = demo / "reference_truth.json"
    table = demo / "path_a" / "path_a_wide_table.parquet"
    train_log = Path(a.train_log) if a.train_log else products.parent / "cfm_train_baseline.log"

    # Stage B's sibling artifacts. The producers' frozen protocols name these representations, so
    # they are derived from the products tree — never from a stale default or the CWD. When the
    # passed --products IS one of them (e.g. the base), the sibling resolves to the same directory.
    # `port` is the repo-resident 307-dim delivery port (bank + measure PCA-128) that the B3
    # blocking rows certify; it is built by scripts/build_port_artifact.py, and NEVER substituted
    # by the bank (a 179-dim E measures a different object and can flip those rows).
    sources = {
        "products": products,
        "base": products.parent / "cfm_v2k_vf2",
        "c32rvd": products.parent / "cfm_v2k_vf2_c32rvd",
        "compact": products.parent / "cfm_v2k_vf2_c32",
        "port": products.parent / "cfm_v2k_vf2_port",
    }

    for label, path in (
        ("products DB", products_db),
        ("demo stream", stream),
        ("reference truth", truth),
        ("PATH A table", table),
    ):
        if not Path(path).exists():
            print(
                f"certification: no {label} at {path} — the chain cannot be regenerated without it",
                file=sys.stderr,
            )
            return 2

    fmt = {
        "products": str(products),
        "products_db": str(products_db),
        "demo": str(demo),
        "stream": str(stream),
        "train_log": str(train_log),
        "legs": a.legs,
        "stages_out": str(stages_out),
        "prov_out": str(receipts / "path_a_provenance.json"),
    }

    # ---- 0. the memory budget, derived from the machine before anything is launched -----------
    cap_gb, cap_ok, cap_why = SR.resolve_cap(SR.DEFAULT_SLICE, a.floor_gb)
    if not cap_ok:
        print(f"certification: REFUSING to run the chain — {cap_why}", file=sys.stderr)
        return 2
    print(f"  [cap   ] {cap_gb:.1f} GB per producer, via {SR.capped_command(['x'], cap_gb)[1]}")

    # ---- 1. the chain, run for real ---------------------------------------------------------
    t_start = time.time()
    t_start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    runs: dict[str, dict] = {}
    for label, script, template, blocks in CHAIN:
        argv = [arg.format(**fmt) for arg in template]
        if blocks:  # stage producers merge into one file; the legs write a directory
            argv += ["--out", str(stages_out)]
        elif "--out" not in template:  # single-receipt producers name their own file via {prov_out}
            argv += ["--out-dir", str(receipts)]
        runs[label] = _run_producer(script, argv, cap_gb)
        r = runs[label]
        print(f"  [{label:5s}] {script} exit={r['exit']} {r['seconds']}s — {r['last_line'][:88]}")

    # ---- 1b. stage B: the thirteen receipts the B scorecard reads ------------------------------
    # Before this stage existed, every behavior change left 15-20 B rows UNKNOWN until an operator
    # re-ran these producers by hand (three times). Orchestration only: each producer keeps its
    # gates, bars and semantics; the chain supplies the artifact its frozen protocol names. A
    # missing derived artifact SKIPS its producer with a note — the row then reads UNKNOWN on the
    # card, which is the truthful state, instead of the chain dying and regenerating nothing.
    b_fmt = {**fmt, **{k: str(v) for k, v in sources.items()}}
    runs_b: dict[str, dict] = {}
    skipped_b: dict[str, str] = {}
    for p in CHAIN_B:
        if a.skip_slow_b and p.label in SLOW_B:
            skipped_b[p.label] = "--skip-slow-b"
            print(f"[B/{p.label}] SKIPPED — --skip-slow-b")
            continue
        src = sources[p.needs]
        if not _has_products(src):
            if p.needs == "port":
                reason = (
                    "no repo-resident 307-dim port artifact (run scripts/build_port_artifact.py)"
                )
            elif p.needs == "compact":
                reason = f"no 48-dim compact served object at {src}"
            else:
                reason = f"no products at {src}"
            skipped_b[p.label] = reason
            print(f"[B/{p.label}] SKIPPED — {reason}")
            continue
        receipt = receipts / p.receipt
        argv = [arg.format(**b_fmt, b_out=str(receipt)) for arg in p.argv]
        r = _run_producer(p.script, argv, cap_gb)
        r["receipt"], r["fresh"] = p.receipt, _generated_at(receipt) >= t_start_iso
        runs_b[p.label] = r
        print(f"[B/{p.label}] exit={r['exit']} {r['seconds']}s")
        if r["exit"] != 0:  # the one line stays the required one; the tail answers "why"
            print(f"      ↳ {r['last_line'][:118]}")

    # A producer exits 1 when a CHECK failed (measure_views_law: `0 if all working else 1`) and 2
    # when it could not run at all. Those are different things, and conflating them would report a
    # legitimate failing measurement as a broken chain — or, worse, would tempt someone to make a
    # producer exit 0 to get a green S6. The gate here is "the card was REGENERATED", so the
    # evidence is fresh artifacts, not happy exit codes.
    want_legs = {s.strip() for s in a.legs.split(",") if s.strip()}
    # By name from judge_spec.LEGS, never a glob: `measure_judge_law` writes its v1-flat
    # DIAGNOSTIC as judge_leg_v1flat_<leg>.json, which a glob would count as a certified leg.
    fresh_legs = set()
    for leg in JS.LEGS:
        f = receipts / f"judge_leg_{leg}.json"
        if f.exists() and f.stat().st_mtime >= t_start - 1:
            fresh_legs.add(leg)
    stages_now = json.loads(stages_out.read_text()) if stages_out.exists() else {}
    fresh_stages = {
        name
        for name in STAGES
        if str(stages_now.get(name, {}).get("_stamp", {}).get("generated_at", "")) >= t_start_iso
    }
    ok_chain, detail = chain_verdict(
        {k: v["exit"] for k, v in runs.items()},
        fresh_stages,
        set(STAGES),
        fresh_legs,
        want_legs,
        len(CHAIN),
    )
    timing = ", ".join(f"{k} {v['seconds']}s" for k, v in runs.items())

    checks: dict[str, dict] = {
        "full_chain_in_one_invocation": {
            "working": ok_chain,
            "evidence": (
                f"{len(runs)} producer(s) run by this one command ({timing}); "
                f"fresh receipts: stages {sorted(fresh_stages)}, legs {sorted(fresh_legs)}"
                + (
                    f"; {len(detail['reported_failing'])} reported a FAILING CHECK (a "
                    f"measurement, not a chain failure): {detail['reported_failing']}"
                    if detail["reported_failing"]
                    else ""
                )
                + (f"; COULD NOT RUN (exit>=2): {detail['fatal']}" if detail["fatal"] else "")
                + (f"; MISSING FRESH EVIDENCE: {detail['missing']}" if detail["missing"] else "")
            ),
        }
    }

    # ---- 2. the reader is a pure function of those receipts ---------------------------------
    # Measured with every producer's receipt present and S6's own block ABSENT: a check may not be
    # taken over bytes that include its own result (DEC-430's rule, applied to the render).
    strip = lambda md: "\n".join(  # noqa: E731 - the timestamp line is the only thing allowed to move
        ln for ln in md.splitlines() if not ln.startswith("*Generated `")
    )

    renders = [strip(RC.render(receipts, RC.behavior_hash())[0]) for _ in range(2)]
    same = renders[0] == renders[1] and bool(renders[0])
    checks["render_is_deterministic"] = {
        "working": same,
        "evidence": (
            "rendered the card twice from the same receipts: "
            + (
                f"identical apart from the timestamp line ({len(renders[0])} chars, "
                f"{renders[0].count(chr(10)) + 1} lines)"
                if same
                else f"DIFFERS, first at line {_diff_line(renders[0], renders[1])}"
            )
        ),
    }

    # ---- 3. commit, tree, and the hashes of what the receipts describe ----------------------
    commit, branch = RC.git("rev-parse", "HEAD"), RC.git("rev-parse", "--abbrev-ref", "HEAD")
    dirty, changed = _tree_dirty(_exempt(receipts, card))
    checks["commit_recorded_clean"] = {
        "working": bool(commit) and commit != "?" and not dirty,
        "evidence": (
            f"commit {commit[:12]} on {branch}, behavior {RC.behavior_hash()[:16]}; "
            + (
                "tree clean apart from the generated artifacts this command rewrites"
                if not dirty
                else f"tree carries {len(changed)} non-generated change(s): {', '.join(changed[:4])}"
            )
        ),
    }

    artifacts: dict[str, dict] = {}
    missing: list[str] = []
    for path in [
        products_db,
        stream,
        truth,
        table,
        *sorted(products.glob("cfm_*.pt")),
        *sorted(products.glob("registry_*.json")),
        *([train_log] if train_log.exists() else []),
    ]:
        if not Path(path).exists():
            missing.append(Path(path).name)
            continue
        artifacts[Path(path).name] = _artifact(Path(path))
    checks["artifact_hashes_recorded"] = {
        "working": bool(artifacts) and not missing,
        "evidence": (
            f"{len(artifacts)} input artifact(s) hashed "
            f"({sum(v['bytes'] for v in artifacts.values()) / 1e6:.1f} MB) — the bytes the receipts "
            f"were measured against" + (f"; MISSING: {', '.join(missing)}" if missing else "")
        ),
    }

    # ---- 4. every published row carries its N ----------------------------------------------
    n_rows, no_n = _rows_report_n(receipts)
    checks["every_row_reports_n"] = {
        "working": n_rows > 0 and not no_n,
        "evidence": (
            f"{n_rows} live leg-row(s) in the regenerated receipts, "
            + ("all report N" if not no_n else f"{len(no_n)} without N: {', '.join(no_n[:4])}")
        ),
    }

    # ---- 5. the comparator's provenance: every column the FAILs compare against is stamped ----
    prov_receipt = receipts / "path_a_provenance.json"
    prov_body = json.loads(prov_receipt.read_text()) if prov_receipt.exists() else None
    checks["comparator_provenance_certified"] = comparator_provenance_fact(
        prov_body, str((prov_body or {}).get("generated_at", "")) >= t_start_iso
    )

    # ---- write S6, then render the card it describes ---------------------------------------
    body = _rollup(checks)
    body["chain"] = runs
    # Stage B is recorded, not gated: the B producers carry measurements (exit 1 = a failing row,
    # which the card is supposed to publish), and S6's meaning is the regeneration itself. What a
    # reader must be able to reconstruct is exactly which B receipts this invocation rewrote and
    # which it skipped, and why.
    body["chain_b"] = {
        "ran": runs_b,
        "skipped": skipped_b,
        "skip_slow_b": a.skip_slow_b,
        "sources": {k: str(v) for k, v in sources.items()},
    }
    body["memory"] = {
        "cap_gb": round(cap_gb, 1),
        "derivation": cap_why,
        "mechanism": runs["legs"]["cap_mechanism"],
    }
    body["artifacts"] = artifacts
    body["inputs"] = {"products": str(products), "demo": str(demo), "legs": a.legs}
    body["n_rows"] = n_rows
    stamp_merge_write(stages_out, {"S6": body}, " ".join([Path(__file__).name] + sys.argv[1:]))
    md, info = RC.render(receipts, RC.behavior_hash())
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(md)

    print(f"\nS6 working={body['working']} · card -> {card}")
    print(f"card summary: {json.dumps(info['summary'], sort_keys=True)}")
    print(info["assessment"])
    for name, c in checks.items():
        print(f"  {'OK  ' if c['working'] else 'FAIL'} {name}: {c['evidence'][:118]}")
    b_fatal = {k: v["exit"] for k, v in runs_b.items() if v["exit"] >= 2}
    print(
        f"B stage: {len(runs_b)}/{len(CHAIN_B)} ran, {len(skipped_b)} skipped"
        + (f" ({', '.join(sorted(skipped_b))})" if skipped_b else "")
        + (f"; COULD NOT RUN (exit>=2): {b_fatal}" if b_fatal else "")
    )
    if skipped_b:
        print(
            f"WARNING: stage B is PARTIAL — {len(skipped_b)}/{len(CHAIN_B)} producer(s) skipped "
            f"({', '.join(sorted(skipped_b))}); those B receipts were NOT re-stamped"
        )
    return 0 if body["working"] else 1


def _exempt(receipts: Path, card: Path) -> set[str]:
    """Names a full regeneration is allowed to leave modified: the receipts and the card."""
    allowed = {card.name}
    if receipts.is_dir():
        allowed |= {p.name for p in receipts.iterdir() if p.is_file()}
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
