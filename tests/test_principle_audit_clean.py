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


#: Pre-existing stale baseline entries (findings that no longer exist). The ratchet may only
#: shrink, so this set is a ceiling: paying down one of these passes without an edit, while a NEW
#: stale entry — a resolved finding whose acceptance was left behind — fails the suite. Compute
#: the current debt with `set(baseline_keys) - {h["key"] for h in collect(...) if blocking}`.
_KNOWN_STALE: frozenset[str] = frozenset()  # the baseline is a hard ratchet: zero stale


def _current_blocking_keys() -> set[str]:
    """Re-run the same whole-surface audit the `--all` gate runs and return its finding keys.

    Paths are repo-relative (the audit's scope matching and key hashing are path-based), so the
    caller runs with `cwd=ROOT`. The tree walk covers the audit's own BLOCK/NOTE roots: there is
    no tracked-file list to drift from the files actually on disk.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import principle_audit as PA

    files = [
        str(p.relative_to(ROOT))
        for tree in PA.BLOCK_ROOTS + PA.NOTE_ROOTS
        for p in (ROOT / tree).rglob("*.py")
    ]
    return {h["key"] for h in PA.collect(files) if h["severity"] == "block"}


def test_baseline_shrinks_only(monkeypatch):
    """A baseline entry whose finding no longer exists is dead weight — delete it.

    The baseline is a ratchet: it may only shrink. This re-runs the audit and compares its
    BLOCKING findings with the accepted set, so an acceptance cannot outlive the finding it
    accepted. The pre-existing stale entries are recorded above; any NEW stale entry fails.
    """
    import json

    baseline = ROOT / "scripts" / "principle_audit_baseline.json"
    assert baseline.exists(), "principle_audit_baseline.json is missing"
    data = json.loads(baseline.read_text())
    keys = data.get("keys", [])
    assert data.get("count") == len(keys), "baseline count is inconsistent"
    assert len(keys) == len(set(keys)), "baseline contains duplicate keys"

    monkeypatch.chdir(ROOT)
    current = _current_blocking_keys()
    accepted = set(keys)
    # a finding with no acceptance is already a strict failure; assert it here too so this
    # ratchet check cannot pass on a surface whose gate is red
    assert current <= accepted, (
        f"blocking finding(s) not accepted by the baseline: {sorted(current - accepted)}"
    )
    stale = accepted - current
    assert stale <= _KNOWN_STALE, (
        f"new stale baseline entry/entries {sorted(stale - _KNOWN_STALE)}: the finding no "
        f"longer exists, so delete its key from {baseline.name} (the ratchet may only shrink)"
    )
