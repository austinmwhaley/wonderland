"""Flag principle violations in changed files: hot-path loops, cadence/capacity literals, gates.

Usage:
  python3 scripts/principle_audit.py [files...] [--strict]
Default: changed files vs HEAD (`git diff --name-only`).
A line can be exempted with `# principle-ok: <reason>`.
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = [
    (re.compile(r"for\s+\w+\s+in\s+range\(\s*T\s*\)"), "Python loop over T (use _scan/vectorize)"),
    (re.compile(r"for\s+\w+\s+in\s+range\(\s*len\("), "Python loop over rows/time"),
    (re.compile(r"\.iterrows\(\)|\.itertuples\(\)"), "row-wise pandas loop"),
    (re.compile(r"\b604800\b|\b86400\b"), "cadence literal (derive from data)"),
    (re.compile(r"\bseq_len\s*=\s*\d+"), "fixed seq_len literal"),
    (re.compile(r"\bdim\s*=\s*(?:32|64|128|256|512|1024)\b"), "fixed width literal"),
    (re.compile(r"\bn_?layers?\s*=\s*\d+|\blayers\s*=\s*[2-9]\b"), "fixed depth literal"),
    (re.compile(r"criterion\s*=\s*[\"']"), "new criterion string (must be frozen/pre-registered)"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    files = (
        a.files
        or subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True
        ).stdout.split()
    )
    hits = []
    for f in files:
        p = Path(f)
        if not p.exists() or p.suffix not in (".py",) or p.name == "principle_audit.py":
            continue
        if "rabbit_hole/generators/" in str(p):
            continue  # offline data generation is not a hot path
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if "# principle-ok:" in line:
                continue
            for rx, why in PATTERNS:
                if rx.search(line):
                    hits.append((str(p), i, why, line.strip()[:100]))
    for f, i, why, line in hits:
        print(f"{f}:{i}: {why}\n    {line}")
    print(f"[principle-audit] {len(hits)} finding(s)", file=sys.stderr)
    if a.strict and hits:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
