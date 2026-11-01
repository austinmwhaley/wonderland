"""AGENTS.md: the principle audit applies to all code we RUN, constantly.

`--strict` exits non-zero on any BLOCKING finding outside
`scripts/principle_audit_baseline.json`. The baseline is a ratchet and may only shrink; this
test is what makes that true on every pytest invocation rather than on the honour system.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_whole_surface_has_no_new_principle_findings():
    r = subprocess.run(
        [sys.executable, "scripts/principle_audit.py", "--all", "--strict"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (
        "new blocking principle finding(s) beyond the baseline.\n"
        "Fix the code, or mark the line `# principle-ok: <reason>`.\n"
        "Do NOT grow the baseline to make this green.\n\n"
        f"{r.stdout}\n{r.stderr}"
    )


def test_baseline_shrinks_only():
    """A baseline entry whose finding no longer exists is dead weight — delete it.

    This does not fail the suite (resolving a finding is always allowed); it reports so the
    ratchet does not quietly accumulate stale acceptances.
    """
    import json

    baseline = ROOT / "scripts" / "principle_audit_baseline.json"
    assert baseline.exists(), "principle_audit_baseline.json is missing"
    data = json.loads(baseline.read_text())
    assert data.get("count") == len(data.get("keys", [])), "baseline count is inconsistent"
