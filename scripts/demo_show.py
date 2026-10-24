"""Present the wonderland demo from the generated report card.

    python3 scripts/demo_show.py

The demo IS the project running on the synthetic dataset.  `REPORT_CARD.md` (repo
root) is the artifact; this script narrates it and prints it.  No number is invented
here: everything is read from the card.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "REPORT_CARD.md"
AUDIT = ROOT / "docs/AUDIT_CARD.md"


def _bullet(card: str, prefix: str) -> str:
    for line in card.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _section(card: str, heading: str) -> list[str]:
    lines = card.splitlines()
    start = next((i for i, x in enumerate(lines) if x.strip() == f"## {heading}"), None)
    if start is None:
        return []
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    return lines[start + 1 : end]


def _table_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if line.startswith("|") and not line.startswith("|---") and "| # |" not in line:
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def main():
    card = CARD.read_text()
    W = 78
    print("=" * W)
    print("  WONDERLAND — demo")
    print("=" * W)
    print(_bullet(card, "- **Data:**"))
    print(_bullet(card, "- **Scale:**"))
    print(_bullet(card, "- **Encoder config:**"))
    print()
    print(f"B  (encoder alone)  : {_bullet(card, '- **B (encoder alone):**')}")
    print(f"B+C (plugins)       : {_bullet(card, '- **B+C (plugins):**')}")
    print()
    print("Plugins (2 per group):")
    for r in _table_rows(_section(card, "B+C — encoder + plugins")):
        if r and r[0][:3] in ("SUP", "UNS", "RL"):
            print(f"  {r[0]:<6} {r[1]:<34} {r[5]:<58} {r[6]}")
    print()
    verdict = " ".join(_section(card, "Verdict"))
    print("Verdict:", verdict[:300])
    print()
    print("Not claimed (see the audit):")
    for line in _section(card, "Limitations / Layer 1 (not claimed)"):
        if line.startswith("- **"):
            print("  " + line[2:120])
    print()
    print(f"Full card : {CARD}")
    print(f"Audit     : {AUDIT}")
    print("=" * W)


if __name__ == "__main__":
    main()
