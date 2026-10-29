#!/usr/bin/env python3
"""S6: the card must be rebuildable by ONE repo command, at a recorded commit, with hashes (rule 6).

    python3 scripts/measure_certification_law.py [--products DIR] [--demo DIR]

Why this file exists. `judge_card.py` renders the card from whatever receipts happen to be on
disk, so "I regenerated the card" was never evidence about the card: its five stage blocks were
last written by four different scripts, at four different times, two of them against `/tmp` inputs
that no longer exist (DEC-430 found exactly this). A card nobody can rebuild from a commit is not
a certificate — it is a transcript. This is the single command the S6 gate demands, and it
measures whether that command works, in the order the dependencies require:

    S1 S2  measure_input_law        universal input, vocabulary
    S3     measure_dynamics_law     the engine's laws
    S4     measure_views_law        the outlets
    S5     measure_battery_law      the objective battery
    legs   measure_judge_law        the four-leg top line, both columns
    S6     this file                the regeneration itself
    card   judge_card               rendered last, from the receipts above

What "regenerated" has to mean, as checks rather than prose: the whole chain ran in ONE invocation
and every producer exited 0; the card renders from those receipts deterministically (twice,
byte-identical apart from its own timestamp line); the commit is recorded and the tree is clean
apart from the files this command is authorized to rewrite; the inputs the receipts describe are
hashed; and every row the card publishes reports its N.

Like every other producer here, this file records measurements. It contains no verdict string, and
`judge_card.py` — not this file — decides what S6 means (certification rule 3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import judge_card as JC  # noqa: E402  (one definition of the card, and of git state)
import safe_run as SR  # noqa: E402  (one definition of the memory cap)
from measure_input_law import _rollup, stamp_merge_write  # noqa: E402  (one definition)

EXP = ROOT / "docs/report_cards/experiments"
CARD = ROOT / "docs/JUDGE_CARD.md"
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
        "measure_judge_law.py",
        ("--products", "{products}", "--demo", "{demo}", "--legs", "{legs}"),
        (),
    ),
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _artifact(path: Path) -> dict:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


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


def _tree_dirty(exempt: set[str]) -> tuple[bool, list]:
    """Working-tree state, excluding the artifacts a regeneration is allowed to rewrite.

    These are committed generated files: a command whose whole job is to rewrite them can never
    report a clean tree *after* it ran, and a rule that can never be satisfied is the rule DEC-430
    removed for the card's own bytes — a flag that always says the same thing is not a flag.
    Source edits still count, which is the entire point of the check.
    """
    changed = []
    for line in JC.git("status", "--porcelain").splitlines():
        # porcelain is "XY PATH", or "XY OLD -> NEW" for a rename: the New side is the file that
        # exists now, and the two status columns must be dropped or "? scripts/x.py" is "changed"
        path = line[3:].split(" -> ")[-1].strip() if len(line) > 3 else ""
        if path and Path(path).name not in exempt:
            changed.append(path)
    return bool(changed), changed


def _rows_report_n(receipts: Path) -> tuple[int, list]:
    """Count the live leg rows the card will publish, and name any that carry no N.

    H3 enforces N at render time; measuring it here makes S6 a statement about the regenerated
    receipts rather than about the renderer's good intentions.
    """
    rows, missing = 0, []
    for f in sorted(receipts.glob("judge_leg_*.json")):
        body = json.loads(f.read_text())
        for col, block in (body.get("columns") or {}).items():
            for t in block.get("tasks", []):
                if t.get("degenerate"):
                    continue
                rows += 1
                if not t.get("n"):
                    missing.append(f"{f.stem}::{col}::{t.get('name', '?')}")
    return rows, missing


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
    }

    # ---- 0. the memory budget, derived from the machine before anything is launched -----------
    cap_gb, cap_ok, cap_why = SR.plan(SR.mem_available_gb(), a.floor_gb)
    if not cap_ok:
        print(f"certification: REFUSING to run the chain — {cap_why}", file=sys.stderr)
        return 2
    print(f"  [cap   ] {cap_gb:.1f} GB per producer, via {SR.capped_command(['x'], cap_gb)[1]}")

    # ---- 1. the chain, run for real ---------------------------------------------------------
    t_start = time.time()
    runs: dict[str, dict] = {}
    for label, script, template, blocks in CHAIN:
        argv = [arg.format(**fmt) for arg in template]
        if blocks:  # stage producers merge into one file; the legs write a directory
            argv += ["--out", str(stages_out)]
        else:
            argv += ["--out-dir", str(receipts)]
        runs[label] = _run_producer(script, argv, cap_gb)
        r = runs[label]
        print(f"  [{label:5s}] {script} exit={r['exit']} {r['seconds']}s — {r['last_line'][:88]}")
    failed = {k: v["exit"] for k, v in runs.items() if v["exit"] != 0}
    timing = ", ".join(f"{k} {v['seconds']}s" for k, v in runs.items())
    # a "full" regeneration that quietly re-measured one leg would satisfy the count and lie about
    # the substance, so the top line's completeness is measured off the receipts' mtimes
    want_legs = {s.strip() for s in a.legs.split(",") if s.strip()}
    fresh_legs = {
        f.stem.removeprefix("judge_leg_")
        for f in receipts.glob("judge_leg_*.json")
        if f.stat().st_mtime >= t_start - 1
    }
    legs_complete = fresh_legs == want_legs

    checks: dict[str, dict] = {
        "full_chain_in_one_invocation": {
            "working": not failed and len(runs) == len(CHAIN) and legs_complete,
            "evidence": (
                f"{len(runs)} producer(s) run by this one command ({timing}); "
                f"legs regenerated: {sorted(fresh_legs) or 'none'} of {sorted(want_legs)}; "
                + ("all exited 0" if not failed else f"non-zero exit: {failed}")
            ),
        }
    }

    # ---- 2. the reader is a pure function of those receipts ---------------------------------
    # Measured with every producer's receipt present and S6's own block ABSENT: a check may not be
    # taken over bytes that include its own result (DEC-430's rule, applied to the render).
    strip = lambda md: "\n".join(  # noqa: E731 - the timestamp line is the only thing allowed to move
        ln for ln in md.splitlines() if not ln.startswith("*Generated `")
    )

    renders = [strip(JC.render(receipts, JC.behavior_hash())[0]) for _ in range(2)]
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
    commit, branch = JC.git("rev-parse", "HEAD"), JC.git("rev-parse", "--abbrev-ref", "HEAD")
    dirty, changed = _tree_dirty(_exempt(receipts, card))
    checks["commit_recorded_clean"] = {
        "working": bool(commit) and commit != "?" and not dirty,
        "evidence": (
            f"commit {commit[:12]} on {branch}, behavior {JC.behavior_hash()[:16]}; "
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

    # ---- write S6, then render the card it describes ---------------------------------------
    body = _rollup(checks)
    body["chain"] = runs
    body["memory"] = {
        "cap_gb": round(cap_gb, 1),
        "derivation": cap_why,
        "mechanism": runs["legs"]["cap_mechanism"],
    }
    body["artifacts"] = artifacts
    body["inputs"] = {"products": str(products), "demo": str(demo), "legs": a.legs}
    body["n_rows"] = n_rows
    stamp_merge_write(stages_out, {"S6": body}, " ".join([Path(__file__).name] + sys.argv[1:]))
    md, info = JC.render(receipts, JC.behavior_hash())
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(md)

    print(f"\nS6 working={body['working']} · card -> {card}")
    print(f"card summary: {json.dumps(info['summary'], sort_keys=True)}")
    print(info["assessment"])
    for name, c in checks.items():
        print(f"  {'OK  ' if c['working'] else 'FAIL'} {name}: {c['evidence'][:118]}")
    return 0 if body["working"] else 1


def _exempt(receipts: Path, card: Path) -> set[str]:
    """Names a full regeneration is allowed to leave modified: the receipts and the card."""
    allowed = {card.name}
    if receipts.is_dir():
        allowed |= {p.name for p in receipts.iterdir() if p.is_file()}
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
