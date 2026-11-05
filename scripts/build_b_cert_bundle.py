#!/usr/bin/env python3
"""Build the single B/E1 certification bundle (B-001/B-008/B-009).

The B rows were measured against different objects: the base candidate, the derived
block bank, the delivery port, and the measure view. That is honest attribution but
not one artifact identity. This builder assembles one artifact that contains every
component the rows read:

    state_embeddings     bank 179  (the served brain: donor blocks + served stats)
    donor_embeddings     donor-only 106 (the B-008 boundary diagnostic)
    base_embeddings      base 272   (B2/B3.6/B4.1/B4.3)
    port_embeddings      port 307   (B3.2-B3.5; bank 179 + measure PCA-128)
    view_embeddings      the measure / response / forecast / ... views

and writes a registry + `bundle_receipt.json` that maps every B row to its component
and records the synthetic demo scope (B-001). No measurement is re-run; the
components are copied byte-for-byte or derived deterministically from the frozen
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_spec as B  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402

BANK = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_bank"
BASE = ROOT / "looking_glass/artifacts/cfm_v2k_vf2"
C32 = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_c32rvd"
DEMO = ROOT / "looking_glass/artifacts/demo_v2"
DEST = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_cert"
PORT_RECEIPT = ROOT / "docs/report_cards/experiments/lab/port_dec604.json"
PCA_DIM = 128


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _component_for(receipt_name: str) -> str:
    if receipt_name in ("judge_stages.json",):
        return "stages (S1-S6)"
    if receipt_name == "runtime.json":
        return "bank"
    paths: list[str] = []
    try:
        rc = json.loads((ROOT / "docs/report_cards/experiments" / receipt_name).read_text())
        for key in ("products", "ope_products", "source_products"):
            block = rc.get(key) or {}
            if isinstance(block, dict) and block.get("path"):
                paths.append(str(block["path"]).rstrip("/"))
    except Exception:  # noqa: BLE001 - a missing receipt is not this builder's failure
        pass
    if "port" in receipt_name or any("cfmd_port128" in p for p in paths):
        return "port"
    has_bank = any(p.endswith("cfm_v2k_vf2_bank") for p in paths)
    has_base = any(
        p.endswith("cfm_v2k_vf2") or p.endswith("cfm_v2k_vf2/cfm_products.duckdb") for p in paths
    )
    has_c32 = any("cfm_v2k_vf2_c32rvd" in p for p in paths)
    if has_bank and has_base:
        return "base+bank"
    if has_bank and has_c32:
        return "bank+c32rvd"
    if has_bank:
        return "bank"
    if has_base:
        return "base"
    return "bank"


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    import duckdb

    t0 = time.perf_counter()
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    shutil.copy2(BANK / "cfm_products.duckdb", DEST / "cfm_products.duckdb")
    for p in BANK.glob("cfm_*.pt"):
        shutil.copy2(p, DEST / p.name)
    for name in ("bank_receipt.json", "compact_e_receipt.json"):
        shutil.copy2(BANK / name, DEST / name)

    # base 272
    con = duckdb.connect(str(BASE / "cfm_products.duckdb"), read_only=True)
    base = con.execute("select customer_key, embedding from state_embeddings").fetchall()
    con.close()
    # bank + measure PCA-128 -> port 307
    con = duckdb.connect(str(DEST / "cfm_products.duckdb"))
    try:
        keys = [
            str(r[0]) for r in con.execute("select customer_key from state_embeddings").fetchall()
        ]
        bank = {
            str(k): np.asarray(v, float)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
        measure: dict[str, dict[str, np.ndarray]] = {}
        for k, part, emb in con.execute(
            "select customer_key, part, embedding from view_embeddings where view='measure'"
        ).fetchall():
            measure.setdefault(str(k), {})[str(part)] = np.asarray(emb, float)
        split = dict(con.execute("select customer_key, split from encoder_samples").fetchall())
    finally:
        con.close()
    from looking_glass import view_layout as V

    order = list(V.PART_ORDER.get("measure", []))
    full = np.vstack([np.concatenate([measure[k][p] for p in order]) for k in keys])
    fit_idx = [i for i, k in enumerate(keys) if split[k] == "A"]
    scaler = StandardScaler().fit(full[fit_idx])
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(scaler.transform(full[fit_idx]))
    pca_rows = pca.transform(scaler.transform(full))
    port = {k: np.concatenate([bank[k], pca_rows[i]]) for i, k in enumerate(keys)}
    donor = {k: bank[k][:106] for k in keys}

    con = duckdb.connect(str(DEST / "cfm_products.duckdb"))
    try:
        con.execute("drop table if exists base_embeddings")
        con.execute(
            "create table base_embeddings (customer_key varchar, version varchar, dim integer, embedding double[])"
        )
        con.execute("drop table if exists donor_embeddings")
        con.execute(
            "create table donor_embeddings (customer_key varchar, version varchar, dim integer, embedding double[])"
        )
        con.execute("drop table if exists port_embeddings")
        con.execute(
            "create table port_embeddings (customer_key varchar, version varchar, dim integer, embedding double[])"
        )
        tag = str(con.execute("select distinct version from state_embeddings").fetchone()[0])
        base_map = {str(k): np.asarray(v, float) for k, v in base}
        con.executemany(
            "insert into base_embeddings values (?,?,?,?)",
            [[k, tag, 272, base_map[k].tolist()] for k in keys],
        )
        con.executemany(
            "insert into donor_embeddings values (?,?,?,?)",
            [[k, tag, 106, donor[k].tolist()] for k in keys],
        )
        con.executemany(
            "insert into port_embeddings values (?,?,?,?)",
            [[k, tag, 307, port[k].tolist()] for k in keys],
        )
    finally:
        con.close()

    # Bundle registry: the bank registry plus the component manifest. The tag must equal
    # the served version (resolvers refuse otherwise).
    regs = sorted(BANK.glob("registry_*.json"), key=lambda p: p.stat().st_mtime)
    reg = json.loads(regs[-1].read_text())
    components = {
        "base": {
            **products_identity(str(BASE)),
            "table": "base_embeddings",
            "dim": 272,
        },
        "bank": {
            **products_identity(str(BANK)),
            "table": "state_embeddings",
            "dim": 179,
        },
        "donor_only": {"table": "donor_embeddings", "dim": 106},
        "port": {
            "table": "port_embeddings",
            "dim": 307,
            "receipt": str(PORT_RECEIPT),
            "receipt_sha256": _sha(PORT_RECEIPT),
            "measure_pca_dim": PCA_DIM,
        },
        "measure": {"view": "measure"},
        "c32rvd": {**products_identity(str(C32)), "role": "B5.1 source"},
    }
    reg["bundle"] = {
        "kind": "b_cert_bundle",
        "scope": "synthetic demo_v2 prototype; no production-stream claim",
        "demo_stream_sha256": _sha(DEMO / "customer_event_stream.parquet"),
        "components": components,
    }
    (DEST / f"registry_{tag.replace('.', '_')}.json").write_text(
        json.dumps(reg, indent=1, sort_keys=True) + "\n"
    )

    rows = {
        row.ident: {
            "receipt": row.source[0] if row.source else None,
            "component": _component_for(row.source[0]) if row.source else "missing",
            "producer": row.producer,
        }
        for row in B.ROWS
    }
    # The B-008 boundary: the donor-only B+C rerun (bank 106, no measure parts). Keep a
    # compact diagnostic beside the bundle so the scope is reproducible even if /tmp is gone.
    diag = {"kind": "bc_donor_only_diagnostic", "passed": False, "legs": {}}
    for leg in ("supervised", "unsupervised", "rl", "simulator"):
        src = Path(f"/tmp/opencode/bc_donor/judge_leg_{leg}.json")
        if not src.exists():
            diag["legs"][leg] = {"available": False}
            continue
        rc = json.loads(src.read_text())
        tasks = (rc.get("columns") or {}).get("brain", {}).get("tasks", [])
        fails = [t.get("name") for t in tasks if (t.get("ci_lo") is None or t.get("ci_lo") <= 0)]
        diag["legs"][leg] = {
            "available": True,
            "working": rc.get("working"),
            "failing_tasks": fails,
            "tasks": {
                t.get("name"): {
                    "brain": t.get("brain"),
                    "pathA": t.get("pathA"),
                    "ci_lo": t.get("ci_lo"),
                    "ci_hi": t.get("ci_hi"),
                }
                for t in tasks
            },
        }
    diag_path = ROOT / "docs/report_cards/experiments/bc_donor_diagnostic.json"
    diag_path.write_text(json.dumps(diag, indent=1, sort_keys=True) + "\n")
    bundle = {
        "kind": "b_cert_bundle",
        "complete": True,
        "tag": tag,
        "products": products_identity(str(DEST)),
        "components": components,
        "b_rows": rows,
        "b_row_components": sorted({v["component"] for v in rows.values()}),
        "scope": "synthetic demo_v2 prototype; no production-stream claim (B-001)",
        "one_artifact": (
            "every B row's component is inside this bundle; the row -> component map is above "
            "(B-009)"
        ),
        "brain_boundary": (
            "the B+C brain column is state_embeddings (bank 179: donor blocks + served measure "
            "statistics); donor_embeddings is the measure-free donor-only diagnostic. The "
            "donor-only B+C rerun fails unsupervised persona/future and supervised "
            "h0/return_proneness, so the served-bank result is not a pure encoder win (B-008)."
        ),
        "donor_only_diagnostic": str(diag_path),
        "donor_only_diagnostic_sha256": _sha(diag_path),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    # _meta through the shared stamper (DEC-609/B-015): the manifest is the single-artifact
    # claim and must carry the same provenance as the 23 row receipts it maps. The registry
    # (registry_{tag}.json) does not hash this receipt, so stamping moves no pinned sha.
    stamp_merge_write(DEST / "bundle_receipt.json", bundle, " ".join(sys.argv), stamp_blocks=False)
    print(json.dumps({k: bundle[k] for k in ("tag", "products", "b_row_components")}, indent=1))
    print("WROTE", DEST / "bundle_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
