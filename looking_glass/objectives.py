"""Machine-readable objectives scorecard: tiers that roll up to the project goal.

Each objective is ONE metric, ONE ground-truth gate, a control, the axis it
diagnoses, and the artifact that proves it. The scorecard is assembled from
receipts (never hand-maintained) and reports per-capability status; there is no
single composite that hides a failure (root doctrine, principle 9).

Tiers
-----
T0  North Star  — the project goal (derived from T1/T2 certificates)
T1  Foundation  — is this a foundation encoder at all? (all must hold)
T2  Certificates— Layer-1 quality guarantees
T3  Consumers   — downstream gates; these may VETO, never SHAPE the encoder
T4  Invariants  — operational guarantees every run must keep

Artifact contract (``artifacts.json``)::

    {
      "encoder":          {"schema_universal": true, "self_supervised": true,
                           "frozen_donor": true},
      "card":             {"solved": true, "u": 0.88},
      "minimality":       {"measure_dim": 128, "joint_certified": false},
      "use_cases":        {"deploy": 3, "hold": 2, "nulls_hold": 5, "nulls_total": 5},
      "calibration":      {"coverage": 0.90, "target": 0.90},
      "accessibility":    {"passed": 1, "total": 1},
      "supervised":       {"passed": 2, "total": 4},
      "unsupervised":     {"passed": 1, "total": 1},
      "ope":              {"passed": 8, "total": 10},
      "known_effect":     {"passed": 1, "total": 1},
      "robustness":       {"passed": 1, "total": 1},
      "efficiency":       {"passed": 1, "total": 1},
      "reproducibility":  {"passed": 1, "total": 1},
      "honest_gates":     {"passed": 1, "total": 1}
    }

Every reader accepts ``{"status": "PASS|PARTIAL|FAIL"}`` as an override, so a
producer can emit a verdict directly instead of the raw metric.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

PASS = "PASS"
PARTIAL = "PARTIAL"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
STATUSES = (PASS, PARTIAL, FAIL, UNKNOWN)
TIER_NAMES = {
    0: "North Star",
    1: "Foundation",
    2: "Certificates",
    3: "Consumers (veto only)",
    4: "Invariants",
}


@dataclass(frozen=True)
class Objective:
    id: str
    tier: int
    name: str
    metric: str
    gate: str
    control: str
    axis: str
    source: str


def record(status: str, current: str = "", evidence: str = "") -> dict:
    """A normalized objective result."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    return {"status": status, "current": str(current), "evidence": str(evidence)}


# ---------------------------------------------------------------------------
# readers: artifacts -> normalized record. Missing artifact => UNKNOWN (honest).
# ---------------------------------------------------------------------------
def _override(art) -> dict | None:
    if isinstance(art, str) and art in STATUSES:
        return record(art)
    if isinstance(art, dict) and art.get("status") in STATUSES:
        return record(art["status"], art.get("current", ""), art.get("evidence", ""))
    return None


def _bool_reader(key: str) -> Callable[[dict], dict]:
    def read(artifacts: dict) -> dict:
        enc = artifacts.get("encoder") or {}
        if not isinstance(enc, dict):
            return record(UNKNOWN, evidence="encoder receipt absent")
        ov = _override(enc)
        if ov is not None:
            return ov
        val = enc.get(key)
        if val is None:
            return record(UNKNOWN, evidence=f"encoder.{key} absent")
        return record(PASS if val else FAIL, current=str(bool(val)), evidence="encoder receipt")

    return read


def _count_reader(key: str, source: str) -> Callable[[dict], dict]:
    def read(artifacts: dict) -> dict:
        art = artifacts.get(key)
        if not isinstance(art, dict):
            return record(UNKNOWN, evidence=f"{source} absent")
        ov = _override(art)
        if ov is not None:
            return ov
        passed = int(art.get("passed", 0) or 0)
        total = int(art.get("total", 0) or 0)
        if total <= 0:
            return record(UNKNOWN, current=f"{passed}/{total}", evidence=source)
        status = PASS if passed >= total else (PARTIAL if passed > 0 else FAIL)
        return record(status, current=f"{passed}/{total}", evidence=source)

    return read


def _read_sufficiency(artifacts: dict) -> dict:
    art = artifacts.get("card")
    if not isinstance(art, dict):
        return record(UNKNOWN, evidence="foundation card absent")
    ov = _override(art)
    if ov is not None:
        return ov
    u = art.get("u")
    solved = art.get("solved")
    cur = f"U={u}" if u is not None else ""
    if solved is not None:
        return record(PASS if solved else FAIL, current=cur, evidence="foundation card")
    if u is None:
        return record(UNKNOWN, evidence="card has neither solved nor u")
    # no re-tuned threshold: U>0 with no gate is evidence of signal, not certification
    return record(PARTIAL if float(u) > 0 else FAIL, current=cur, evidence="foundation card")


def _read_minimality(artifacts: dict) -> dict:
    art = artifacts.get("minimality")
    if not isinstance(art, dict):
        return record(UNKNOWN, evidence="minimality receipt absent")
    ov = _override(art)
    if ov is not None:
        return ov
    dim = art.get("measure_dim")
    joint = art.get("joint_certified")
    cur = f"measure_dim={dim}, joint={bool(joint)}"
    if joint:
        return record(PASS, current=cur, evidence="joint shape+measure certificate")
    if dim is not None:
        return record(PARTIAL, current=cur, evidence="measure-only certificate")
    return record(FAIL, current=cur, evidence="no minimality evidence")


def _read_use_cases(artifacts: dict) -> dict:
    art = artifacts.get("use_cases")
    if not isinstance(art, dict):
        return record(UNKNOWN, evidence="use-case receipt absent")
    ov = _override(art)
    if ov is not None:
        return ov
    deploy = int(art.get("deploy", 0) or 0)
    hold = int(art.get("hold", 0) or 0)
    nh = art.get("nulls_hold")
    nt = art.get("nulls_total")
    cur = f"DEPLOY {deploy}, HOLD {hold}, nulls {nh}/{nt}"
    if deploy <= 0:
        return record(FAIL, current=cur, evidence="no use case deployed")
    if nh is not None and nt is not None and int(nh) == int(nt):
        return record(PASS, current=cur, evidence="all matching nulls HOLD")
    return record(PARTIAL, current=cur, evidence="deploy present; null control incomplete")


def _read_calibration(artifacts: dict) -> dict:
    art = artifacts.get("calibration")
    if not isinstance(art, dict):
        return record(UNKNOWN, evidence="calibration receipt absent")
    ov = _override(art)
    if ov is not None:
        return ov
    cov = art.get("coverage")
    tgt = art.get("target")
    if cov is None or tgt is None:
        return record(UNKNOWN, evidence="coverage/target absent")
    cur = f"coverage={cov} target={tgt}"
    return record(PASS if float(cov) >= float(tgt) else FAIL, current=cur, evidence="conformal")


READERS: dict[str, Callable[[dict], dict]] = {
    "t1.schema_universal": _bool_reader("schema_universal"),
    "t1.self_supervised": _bool_reader("self_supervised"),
    "t1.frozen_donor": _bool_reader("frozen_donor"),
    "t1.sufficiency": _read_sufficiency,
    "t1.geometry": _count_reader("geometry", "portfolio geometry gate"),
    "t1.transfer": _count_reader("transfer", "battery transfer"),
    "t2.sufficiency": _read_sufficiency,
    "t2.minimality": _read_minimality,
    "t2.decision_sufficiency": _read_use_cases,
    "t2.calibration": _read_calibration,
    "t2.accessibility": _count_reader("accessibility", "per-field simple probes"),
    "t3.supervised": _count_reader("supervised", "supervised gate"),
    "t3.unsupervised": _count_reader("unsupervised", "unsupervised gate"),
    "t3.ope": _count_reader("ope", "white_queen acceptance"),
    "t3.known_effect": _count_reader("known_effect", "red_king known-effect recovery"),
    "t4.robustness": _count_reader("robustness", "stability/divergence gate"),
    "t4.efficiency": _count_reader("efficiency", "runtime receipts"),
    "t4.reproducibility": _count_reader("reproducibility", "seeded/versioned artifacts"),
    "t4.honest_gates": _count_reader("honest_gates", "null controls / no gate weakening"),
}


def _t0(records: dict[str, dict]) -> dict:
    parts = ("t1.sufficiency", "t2.minimality", "t2.decision_sufficiency")
    st = [records[p]["status"] for p in parts]
    cur = " / ".join(f"{p.split('.')[1]}={records[p]['status']}" for p in parts)
    if all(s == PASS for s in st):
        return record(PASS, current=cur, evidence="all three certificates PASS")
    if all(s == UNKNOWN for s in st):
        return record(UNKNOWN, current=cur, evidence="no certificate receipts")
    if any(s == PASS for s in st):
        return record(PARTIAL, current=cur, evidence="some certificates PASS")
    return record(FAIL, current=cur, evidence="no certificate PASS")


OBJECTIVES: tuple[Objective, ...] = (
    Objective(
        "t0.north_star",
        0,
        "Minimal, bounded, universal sufficient statistic",
        "U->1 AND minimality AND decision-sufficiency",
        "all three certificates PASS",
        "full-history oracle + null actions",
        "state/objective",
        "derived from T1/T2",
    ),
    # T1 — definition of a foundation encoder
    Objective(
        "t1.schema_universal",
        1,
        "Schema-universal ingestion",
        "any canonical 4-column event stream",
        "no named JSON keys; whole event_attributes consumed",
        "heterogeneous-field fixture",
        "input/tokenizer",
        "encoder receipt",
    ),
    Objective(
        "t1.self_supervised",
        1,
        "Self-supervised only",
        "no downstream labels in training",
        "label-free objectives only",
        "code/architecture",
        "objective",
        "encoder receipt",
    ),
    Objective(
        "t1.frozen_donor",
        1,
        "Frozen / reusable donor",
        "downstream reads E; encoder never mutated",
        "frozen at serve",
        "state-store reload",
        "interface",
        "encoder receipt",
    ),
    Objective(
        "t1.sufficiency",
        1,
        "Sufficiency U",
        "excess loss vs full-history oracle",
        "B gate SOLVED",
        "full-history oracle",
        "state",
        "foundation card",
    ),
    Objective(
        "t1.geometry",
        1,
        "Shared geometry",
        "isotropy / effective rank of consumed E",
        "portfolio geometry gate",
        "white-noise null",
        "state",
        "portfolio receipt",
    ),
    Objective(
        "t1.transfer",
        1,
        "Transfer above raw",
        "frozen E adds signal over raw features",
        "battery beats raw",
        "raw model-free baseline",
        "interface",
        "battery receipt",
    ),
    # T2 — certification layer
    Objective(
        "t2.sufficiency",
        2,
        "Sufficiency certificate",
        "U worst-case over the task family",
        "SOLVED",
        "full-history oracle",
        "state",
        "foundation card",
    ),
    Objective(
        "t2.minimality",
        2,
        "Minimality certificate",
        "joint shape+measure dimension",
        "certified minimal",
        "reconstruction of full targets",
        "state",
        "minimality receipt",
    ),
    Objective(
        "t2.decision_sufficiency",
        2,
        "Decision-sufficiency",
        "action response exposed; causal value",
        "DEPLOY with matching null HOLD",
        "null action (snr=0)",
        "objective",
        "use-case receipt",
    ),
    Objective(
        "t2.calibration",
        2,
        "Calibration",
        "empirical coverage of intervals",
        "coverage >= target",
        "held-out conformal",
        "interface",
        "calibration receipt",
    ),
    Objective(
        "t2.accessibility",
        2,
        "Accessibility",
        "simple probes recover information",
        "per-field probe PASS",
        "chance / raw probe",
        "interface",
        "probe receipt",
    ),
    # T3 — consumers (veto only)
    Objective(
        "t3.supervised",
        3,
        "Supervised gate",
        "LTV / return / churn / next-purchase",
        "gated heads PASS with null controls",
        "shuffled-label null",
        "interface",
        "supervised gate",
    ),
    Objective(
        "t3.unsupervised",
        3,
        "Unsupervised gate",
        "silhouette / lookalike / ANOVA",
        "structure gate PASS",
        "random partition null",
        "interface",
        "unsupervised gate",
    ),
    Objective(
        "t3.ope",
        3,
        "Decision/OPE gate",
        "coverage / false deploys / precision / recall / rank",
        "acceptance PASS",
        "truth-in-CI + safe-worse policy",
        "interface",
        "white_queen acceptance",
    ),
    Objective(
        "t3.known_effect",
        3,
        "Known-effect recovery",
        "ordering of randomized arm effects",
        "recovers known effect",
        "known truth",
        "objective",
        "red_king receipt",
    ),
    # T4 — invariants
    Objective(
        "t4.robustness",
        4,
        "Robustness",
        "divergence = hard failure; determinism",
        "no divergence, deterministic forward",
        "repeat-run",
        "training regime",
        "stability receipt",
    ),
    Objective(
        "t4.efficiency",
        4,
        "Efficiency",
        "linear-time, constant-state, runtime receipt",
        "budget met",
        "measured runtime",
        "training regime",
        "runtime receipt",
    ),
    Objective(
        "t4.reproducibility",
        4,
        "Reproducibility",
        "seeded, versioned artifacts",
        "same config -> same result",
        "re-run",
        "interface",
        "registry receipt",
    ),
    Objective(
        "t4.honest_gates",
        4,
        "Honest gates",
        "null controls; never weaken a gate",
        "nulls HOLD",
        "null control",
        "objective",
        "gate receipts",
    ),
)


def scorecard(artifacts: dict) -> dict:
    """Assemble the objective records and per-tier rollups from receipts."""
    records: dict[str, dict] = {}
    for obj in OBJECTIVES:
        if obj.id == "t0.north_star":
            continue
        records[obj.id] = READERS[obj.id](artifacts)
    records["t0.north_star"] = _t0(records)

    by_tier: dict[str, dict] = {}
    for tier in sorted(TIER_NAMES):
        ids = [o.id for o in OBJECTIVES if o.tier == tier]
        counts = {s: 0 for s in STATUSES}
        for oid in ids:
            counts[records[oid]["status"]] += 1
        by_tier[str(tier)] = {"name": TIER_NAMES[tier], "counts": counts, "n": len(ids)}

    order = {o.id: i for i, o in enumerate(OBJECTIVES)}
    return {
        "version": "1",
        "objectives": [{**asdict(o), **records[o.id]} for o in OBJECTIVES],
        "order": order,
        "by_tier": by_tier,
    }


def to_markdown(card: dict) -> str:
    lines = [
        "## Objectives scorecard",
        "",
        "| Tier | Objective | Metric | Gate | Current | Status |",
        "|---|---|---|---|---|---|",
    ]
    for o in card["objectives"]:
        lines.append(
            f"| T{o['tier']} | {o['name']} | {o['metric']} | {o['gate']} | "
            f"{o['current']} | **{o['status']}** |"
        )
    lines += ["", "### Roll-up (per tier; no composite)", ""]
    for tier, t in card["by_tier"].items():
        c = t["counts"]
        lines.append(
            f"- **T{tier} {t['name']}**: "
            f"{c[PASS]} PASS / {c[PARTIAL]} PARTIAL / {c[FAIL]} FAIL / {c[UNKNOWN]} UNKNOWN "
            f"(of {t['n']})"
        )
    return "\n".join(lines) + "\n"


def load(path: str | Path) -> dict:
    """Load an artifacts JSON object."""
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Assemble the objectives scorecard from receipts.")
    ap.add_argument(
        "--artifacts", required=True, help="JSON dict of receipts (see module docstring)"
    )
    ap.add_argument("--json", default=None, help="write the scorecard JSON here")
    ap.add_argument("--markdown", default=None, help="write the markdown table here")
    args = ap.parse_args(argv)

    card = scorecard(load(args.artifacts))
    md = to_markdown(card)
    print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(card, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(md)
    return card


if __name__ == "__main__":
    main()
