"""Layer-A laboratory gate (plan step 5): assert the defensibility criteria.

For every use case it checks the fixture ships all six pieces and that the three
controls behave:
  * positive  — the oracle says the target is CARRIED (incremental > 0),
  * negative  — the snr=0 null is NOT carried (a correct gate must read NO_SIGNAL),
  * adversarial — a leakage trap: the target is NOT carried, yet a never-masked
    attribute ECHOES it, so a leaky gate would false-PASS.

Run: python3 -m looking_glass.lab_gate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from looking_glass.capability_card import transition_carried
from rabbit_hole.lab import coverage_report

LAB = Path("rabbit_hole/data/lab")
_TYPES = ("view", "cart", "purchase")


def _rows(parquet: Path):
    df = pl.read_parquet(parquet).sort(["customer_key", "event_ts"])
    rows = []
    for sub in df.partition_by("customer_key", maintain_order=True):
        et = [_TYPES.index(x) for x in sub["event_type"].to_list() if x in _TYPES]
        if len(et) >= 4:
            rows.append({"et": np.asarray(et), "co": np.zeros(len(et))})
    return rows


def _leak_score(parquet: Path) -> float:
    """How well the echoed attribute predicts the next event (a leaky gate fails this)."""
    df = pl.read_parquet(parquet).sort(["customer_key", "event_ts"])
    df = df.with_columns(
        pl.col("event_attributes").str.json_path_match("$.echo_next").alias("echo")
    )
    df = df.with_columns(pl.col("event_type").shift(-1).over("customer_key").alias("nxt"))
    d = df.drop_nulls(["echo", "nxt"])
    d = d.filter(pl.col("echo") != "")
    if d.height == 0:
        return 0.0
    return float((d["echo"] == d["nxt"]).mean())


_TRANSITION = {
    "next_event",
    "mask",
    "order",
    "next",
    "next_entity",
    "anomaly",
    "boundary",
    "uncertainty",
}


def evaluate(lab: Path = LAB) -> dict:
    cov = coverage_report(lab)
    rows = {}
    for uid, crow in cov["rows"].items():
        uc_dir = Path(lab) / uid
        checks = {"complete": crow["complete"]}
        orc = uc_dir / "oracle.json"
        checks["oracle_carried"] = bool(
            orc.exists() and json.loads(orc.read_text()).get("incremental", 0) > 0
        )
        if uid in _TRANSITION:
            neg = uc_dir / "negative" / "customer_event_stream.parquet"
            adv = uc_dir / "adversarial" / "customer_event_stream.parquet"
            if neg.exists():
                carried, _ = transition_carried(_rows(neg), len(_TYPES))
                checks["negative_not_carried"] = not carried
            if adv.exists():
                carried_a, _ = transition_carried(_rows(adv), len(_TYPES))
                checks["adversarial_not_carried"] = not carried_a
                checks["adversarial_leak_has_teeth"] = _leak_score(adv) > 0.9
        rows[uid] = checks
    n_ok = sum(1 for c in rows.values() if all(c.values()))
    return {"n_use_cases": len(rows), "n_pass": n_ok, "rows": rows}


def main():  # pragma: no cover - CLI
    rep = evaluate()
    print(f"lab gate: {rep['n_pass']}/{rep['n_use_cases']} use cases fully defensible")
    for uid, c in rep["rows"].items():
        bad = [k for k, v in c.items() if not v]
        if bad:
            print(f"  {uid:28s} FAIL {bad}")


if __name__ == "__main__":
    main()
