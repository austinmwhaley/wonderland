"""Flag principle violations in code that is RUN, not just code that changed.

Usage:
  python3 scripts/principle_audit.py [files...] [--strict]     # default: changed vs HEAD
  python3 scripts/principle_audit.py --all [--strict]           # the WHOLE behavior surface
  python3 scripts/principle_audit.py --all --summary            # counts by file and reason
  python3 scripts/principle_audit.py --all --write-baseline      # accept today's findings

Doctrine (AGENTS.md "Principle audit"): a finding is BLOCKING unless it is fixed or carries an
inline `# principle-ok: <reason>`. `--strict` exits non-zero on any BLOCKING finding that is not
in `scripts/principle_audit_baseline.json`. The baseline is a RATCHET: it may only shrink.

Severity matters. A physical unit is not a cadence assumption:
  * `x / 86400.0`  — seconds -> days, a unit conversion. NOTE, never blocking.
  * `30 * 86400.0` — "assume a 30-day window". BLOCKING: a cadence decision in code.
  * `604800`       — "assume a week". BLOCKING.
An audit that flags the unit 150 times is an audit people learn to ignore, which is worse than
no audit at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "principle_audit_baseline.json"

#: production hot paths — a finding here is BLOCKING
BLOCK_ROOTS = (
    "looking_glass/",
    "plugins/",
    "white_queen/",
    "red_king/",
    "red_queen/",
    "caterpillar/",
    "rabbit_hole/",
    "scripts/",
)
#: not hot paths (tests, the separately-owned RL library, offline data generation)
NOTE_ROOTS = ("tests/", "eighth_square/", "rabbit_hole/generators/")

#: (regex, why) — ordered: the first match on a line wins, so the specific cadence-window
#: patterns must precede the generic unit pattern.
BLOCK = [
    (re.compile(r"for\s+\w+\s+in\s+range\(\s*T\s*\)"), "Python loop over T (use _scan/vectorize)"),
    (re.compile(r"for\s+\w+\s+in\s+range\(\s*len\("), "Python loop over rows/time"),
    (re.compile(r"\.iterrows\(\)|\.itertuples\(\)"), "row-wise pandas loop"),
    # cadence WINDOWS: an integer (>=2) times a day/week, or the week constant itself
    (
        re.compile(r"\b(?:[2-9]|\d{2,})\s*\*\s*86400(?:\.0)?\b"),
        "cadence window literal (derive the window from data)",
    ),
    (
        re.compile(r"\b(?:[2-9]|\d{2,})\s*\*\s*24\s*\*\s*3600(?:\.0)?\b"),
        "cadence window literal (derive the window from data)",
    ),
    (re.compile(r"\b604800(?:\.0)?\b"), "week-cadence literal (derive the period from data)"),
    (re.compile(r"\bseq_len\s*=\s*\d+"), "fixed seq_len literal"),
    (re.compile(r"\bdim\s*=\s*(?:32|64|128|256|512|1024)\b"), "fixed width literal"),
    (re.compile(r"\bn_?layers?\s*=\s*\d+|\blayers\s*=\s*[2-9]\b"), "fixed depth literal"),
    (re.compile(r"criterion\s*=\s*[\"']"), "new criterion string (must be frozen/pre-registered)"),
]

#: allowed with a note; the single-constant cleanup is tracked, not a blocker
NOTE = [
    (
        re.compile(r"\b86400(?:\.0)?\b"),
        "unit literal (seconds per day) — prefer one named constant",
    ),
]


def _scope(path: str) -> str:
    """ "block" | "note" | "skip"."""
    p = path.replace("\\", "/")
    if any(p.startswith(r) for r in NOTE_ROOTS):
        return "note"
    if any(p.startswith(r) for r in BLOCK_ROOTS):
        return "block"
    return "skip"


def _key(path: str, why: str, code: str) -> str:
    return hashlib.sha1(f"{path}|{why}|{code}".encode()).hexdigest()[:16]


def _code_lines(text: str) -> set[int] | None:
    """Line numbers that carry executable tokens (NAME/OP/NUMBER), not comments/docstrings.

    Without this the audit flags prose: a comment `# seq_len=128` and a `--set dim=128` help
    string both read as fixed-capacity literals. Precision is what makes an audit usable.
    """
    import io
    import tokenize

    lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
                lines.add(tok.start[0])
    except (tokenize.TokenError, IndentationError):
        return None
    return lines


def collect(files: list[str]) -> list[dict]:
    hits: list[dict] = []
    for f in files:
        p = Path(f)
        if not p.exists() or p.suffix != ".py" or p.name == "principle_audit.py":
            continue
        scope = _scope(str(f))
        if scope == "skip":
            continue
        text = p.read_text(errors="ignore")
        code_only = _code_lines(text)
        for i, line in enumerate(text.splitlines(), 1):
            if code_only is not None and i not in code_only:
                continue
            if "# principle-ok:" in line:
                continue
            code = line.strip()[:120]
            for rx, why in BLOCK:
                if rx.search(line):
                    sev = "block" if scope == "block" else "note"
                    hits.append(
                        {
                            "file": str(f),
                            "line": i,
                            "why": why,
                            "code": code,
                            "severity": sev,
                            "key": _key(str(f), why, code),
                        }
                    )
                    break
            else:
                for rx, why in NOTE:
                    if rx.search(line):
                        hits.append(
                            {
                                "file": str(f),
                                "line": i,
                                "why": why,
                                "code": code,
                                "severity": "note",
                                "key": _key(str(f), why, code),
                            }
                        )
                        break
    return hits


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return set(json.loads(BASELINE.read_text()).get("keys", []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--all", action="store_true", help="audit the whole tracked surface")
    ap.add_argument("--summary", action="store_true", help="counts by file and reason, no lines")
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args(argv)

    if a.all:
        files = subprocess.run(
            ["git", "ls-files", "*.py"], capture_output=True, text=True, cwd=ROOT
        ).stdout.split()
    else:
        files = (
            a.files
            or subprocess.run(
                ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, cwd=ROOT
            ).stdout.split()
        )

    hits = collect(files)
    blocking = [h for h in hits if h["severity"] == "block"]
    notes = [h for h in hits if h["severity"] == "note"]

    if a.write_baseline:
        keys = sorted({h["key"] for h in blocking})
        BASELINE.write_text(
            json.dumps(
                {
                    "_note": (
                        "Accepted BLOCKING findings (ratchet — may only shrink). "
                        "Generated on a whole-surface audit; regenerate only when a finding is "
                        "genuinely resolved or newly accepted by the operator."
                    ),
                    "count": len(keys),
                    "keys": keys,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline written: {len(keys)} blocking finding(s) -> {BASELINE}")

    baseline = _load_baseline()
    new = [h for h in blocking if h["key"] not in baseline]

    if a.summary:
        by_file: dict[str, int] = {}
        by_why: dict[str, int] = {}
        for h in hits:
            tag = f"{h['file']}"
            if h["severity"] == "block":
                by_file[tag] = by_file.get(tag, 0) + 1
            by_why[h["why"]] = by_why.get(h["why"], 0) + 1
        for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
            print(f"{n:4d}  {f}")
        print()
        for w, n in sorted(by_why.items(), key=lambda kv: -kv[1]):
            print(f"{n:4d}  {w}")
    else:
        for h in hits:
            mark = "" if h["severity"] == "note" else ("NEW " if h["key"] not in baseline else "")
            print(f"{mark}{h['file']}:{h['line']}: [{h['severity']}] {h['why']}\n    {h['code']}")

    print(
        f"[principle-audit] {len(blocking)} blocking / {len(notes)} note / "
        f"{len(hits)} total; {len(new)} new blocking beyond the {len(baseline)}-entry baseline",
        file=sys.stderr,
    )
    if a.strict and new:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
