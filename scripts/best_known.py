#!/usr/bin/env python3
"""The best-known record — the monotone frontier, generated from receipts.

`PROMPT_FORWARD.md` is a monotone loop: the unit of progress is beating the best-known
value on a published axis, not "the card is green" and not "we beat the 1-gram" (that
crossing is milestone M1 — a floor). This script is the record. It reads the CURRENT
receipts, compares each published metric to the stored record, and:

    python3 scripts/best_known.py            # report (read-only)
    python3 scripts/best_known.py --update   # promote any improvement into the record

Rules enforced here, not narrated:
  * a value may enter the record ONLY from a receipt whose `_meta.behavior_hash` matches the
    current tree (a stale receipt cannot set a record);
  * a record is only ever IMPROVED (strictly better, in the metric's own direction);
  * M1 stays in the record forever; new milestones are appended, never substituted;
  * the file records the receipt, the value, and the date that set each record.

The gate definitions stay where they are (`b_spec.py`); this file is a frontier ledger,
never a gate, and nothing in the card's verdicts reads it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "report_cards" / "experiments"
RECORD = EXPERIMENTS / "best_known.json"

sys.path.insert(0, str(ROOT / "scripts"))

#: principle-ok: below this many scored rows a per-class estimate is noise, not a record.
MIN_CLASS_ROWS = 8


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _current_hash() -> str | None:
    try:
        import report_card as RC

        return RC.behavior_hash()
    except Exception:  # noqa: BLE001 - the record must not crash the loop
        return None


def _fresh(rc: dict | None, current: str | None) -> bool:
    if not isinstance(rc, dict) or current is None:
        return False
    return (rc.get("_meta") or {}).get("behavior_hash") == current


def _candidates(current: str | None) -> list[dict]:
    """(metric key, value, direction, receipt, detail) from the current receipts.

    direction = "lower" means a smaller value is better; "higher" the reverse.
    Only metrics with a protocol that is stable across runs are listed — a metric whose
    definition/N moves between runs is not a record, it is a different measurement.
    """
    out: list[dict] = []

    # ---- the demo_v2 outcome receipt (B9.1's producer) --------------------------------
    ew = _load(EXPERIMENTS / "encoder_win.json")
    if _fresh(ew, current):
        b = ew.get("encoder_win") or {}
        nll_enc, nll_base = b.get("nll_enc"), b.get("nll_base")
        if isinstance(nll_enc, (int, float)) and isinstance(nll_base, (int, float)):
            out.append(
                {
                    "metric": "demo_v2.next_event.nll_enc",
                    "value": float(nll_enc),
                    "direction": "lower",
                    "receipt": "encoder_win.json",
                    "detail": f"nll_base {nll_base:.6f} · n_B {b.get('n_B_rows')}",
                }
            )
            out.append(
                {
                    "metric": "demo_v2.next_event.margin_vs_1gram",
                    "value": float(nll_enc) - float(nll_base),
                    "direction": "lower",
                    "receipt": "encoder_win.json",
                    "detail": f"paired CI [{b.get('ci_lo')}, {b.get('ci_hi')}]",
                }
            )
        per_class = b.get("per_class_nll_enc") or {}
        per_n = b.get("per_class_n_B") or {}
        for cls, val in sorted(per_class.items()):
            n_b = per_n.get(cls)
            # A class with almost no rows in the scored split is not a record: a 3-row
            # estimate is noise (DEC-649). Same floor the producer uses to allow a fit.
            if n_b is not None and int(n_b) < MIN_CLASS_ROWS:
                continue
            if isinstance(val, (int, float)):
                out.append(
                    {
                        "metric": f"demo_v2.next_event.nll_enc[{cls}]",
                        "value": float(val),
                        "direction": "lower",
                        "receipt": "encoder_win.json",
                        "detail": "per-class nll_enc on split B",
                    }
                )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="promote improvements into the record")
    a = ap.parse_args(argv)

    current = _current_hash()
    record = _load(RECORD) or {"kind": "best_known", "records": {}, "history": []}
    records: dict = record.setdefault("records", {})
    history: list = record.setdefault("history", [])

    cands = _candidates(current)
    if not cands:
        print(f"best_known: no fresh receipts (current behavior {current}); record untouched")
        return 0

    improved, held = [], []
    for c in cands:
        key, val, direction = c["metric"], c["value"], c["direction"]
        prev = records.get(key)
        better = prev is None or (
            val < float(prev["value"]) if direction == "lower" else val > float(prev["value"])
        )
        status = (
            "NEW"
            if prev is None
            else (
                "BEAT"
                if better
                else ("tie" if abs(val - float(prev["value"])) < 1e-12 else "worse")
            )
        )
        line = (
            f"  {key:46s} {val:+.6f}  "
            + (f"(record {float(prev['value']):+.6f}) " if prev else "")
            + status
        )
        if better:
            improved.append((c, status))
            print(line)
        else:
            held.append(key)
            print(line)

    if a.update and improved:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for c, status in improved:
            records[c["metric"]] = {
                "value": c["value"],
                "direction": c["direction"],
                "receipt": c["receipt"],
                "detail": c["detail"],
                "behavior_hash": current,
                "set_at": now,
                "milestone": status,
            }
            history.append(
                {
                    "metric": c["metric"],
                    "value": c["value"],
                    "receipt": c["receipt"],
                    "behavior_hash": current,
                    "set_at": now,
                }
            )
        RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"\nbest_known: {len(improved)} record(s) improved -> {RECORD.relative_to(ROOT)}")
    elif improved:
        print(f"\nbest_known: {len(improved)} improvement(s) available — run --update to record")
    else:
        print(
            f"\nbest_known: no improvement on {len(held)} metric(s) — this run reproduced the record"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
