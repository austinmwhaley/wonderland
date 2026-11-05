"""Decision-log citation integrity (B-012 record repair; DEC-607).

The steering documents cite DEC ids as load-bearing protocol references. Every such
citation in the log's range must resolve to an entry in DECISIONS.md — or to a
documented reserved/sealed line — so that a record hole fails a test instead of
relying on a reader's memory.

Scope rules (frozen with the B-012 repair, DEC-607):
* the log starts at its oldest entry (DEC-275); citations below that id are
  pre-log legacy artifacts and out of scope (the log cannot backfill history it
  never had);
* elided citations ("DEC-586/600/601") expand to all members;
* a "reserved" line in DECISIONS.md documents a sealed/unallocated id (DEC-574;
  the B-012 seals for DEC-597/DEC-602);
* entry ids from DEC-590 on (the log's disciplined region, which contains the
  B-012 repair region) must stay non-decreasing in file order; the legacy
  log's ordering below that is out of scope, like the pre-275 citations.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOG = ROOT / "DECISIONS.md"

STEERING = (
    ROOT / "PLAN.md",
    ROOT / "REPORT_CARD.md",
    ROOT / "STATUS.md",
    ROOT / "AUDIT.md",
    LOG,
    ROOT / "scripts" / "plan_spec.py",
    ROOT / "scripts" / "report_card.py",
)

# A real entry line: a heading, or a bullet whose first word is the DEC id
# (bold or not). Wrapped prose that merely begins with "DEC-NNN …" is NOT an entry.
ENTRY = re.compile(r"^\s*(?:#{1,4}\s+|[-*]\s+\*{0,2})DEC-(\d+)\b")
DEC_CITE = re.compile(r"DEC-(\d{2,4})(?:\s*/\s*(\d{2,4}))*")
RESERVED = re.compile(r"reserved", re.I)

#: The log's ordering-discipline region (DEC-607): the repair region and everything
#: after it. Legacy ordering below this is an artifact, like pre-275 citations.
DISCIPLINED_FROM = 590


def _entries() -> list[tuple[int, int]]:
    """(dec id, line number) of every entry line, in file order."""
    out: list[tuple[int, int]] = []
    for i, line in enumerate(LOG.read_text().splitlines(), 1):
        m = ENTRY.match(line)
        if m:
            out.append((int(m.group(1)), i))
    return out


def _resolved() -> set[int]:
    """Dec ids that resolve: entry lines + documented reserved/sealed lines."""
    ids = {i for i, _ in _entries()}
    for line in LOG.read_text().splitlines():
        if RESERVED.search(line):
            ids |= {int(m.group(1)) for m in DEC_CITE.finditer(line)}
    return ids


def test_citations_resolve_to_entries_or_seals() -> None:
    entries = _entries()
    assert entries, "DECISIONS.md has no entry lines — the detector is broken"
    floor = min(i for i, _ in entries)  # the log's oldest entry (DEC-275)
    resolved = _resolved()
    bad: dict[int, list[str]] = {}
    for path in STEERING:
        assert path.exists(), f"steering document missing: {path}"
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for m in DEC_CITE.finditer(line):
                cands = [int(m.group(1))] + [int(g) for g in m.groups()[1:] if g]
                for c in cands:
                    # in-log-range citation; must resolve (elided form expands)
                    if c >= floor and c not in resolved:
                        bad.setdefault(c, []).append(f"{path.name}:{i}")
    assert not bad, (
        "DEC citations with no entry (or documented seal) in DECISIONS.md: "
        f"{bad} — log the decision (or seal the id if unallocated) before citing it"
    )


def test_entry_ids_are_monotone_in_the_disciplined_region() -> None:
    tail = [(dec, ln) for dec, ln in _entries() if dec >= DISCIPLINED_FROM]
    assert len(tail) >= 3, "disciplined region lost its entries — detector broken"
    for (a, la), (b, lb) in zip(tail, tail[1:]):
        assert a <= b, (
            f"DECISIONS.md entry order regressed: DEC-{b} (line {lb}) follows "
            f"DEC-{a} (line {la}); the log grows by appending in id order from "
            f"DEC-{DISCIPLINED_FROM} on (DEC-607)"
        )
