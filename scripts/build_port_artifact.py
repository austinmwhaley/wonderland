#!/usr/bin/env python3
"""Build the DEC-604 delivery-port products (bank 179 + measure PCA-128 = 307).

B3.2 (`measure_stage_decomposition`) and B3.4 (`measure_block_serving`) were measured against
`/tmp/opencode/cfmd_port128`, a 307-dim products directory that existed only in /tmp and had no
repo builder. Clearing /tmp therefore made both receipts unreproducible (certification rule 6).

This builder materialises that object from repo data, deterministically:

    source 1  cfm_v2k_vf2_cert/port_embeddings (307, when the bundle carries it)
    source 2  bank 179 + measure view PCA-128, composed EXACTLY as
              `build_b_cert_bundle.py` does — StandardScaler + PCA(128, random_state=0),
              fit on split A only

It copies the served bank products (tables, checkpoint, registry) and replaces
`state_embeddings` with the port (adding `port_embeddings` as the explicit table), so the two
producers read the same object they certify: `customer_state` (M0-M3 stages),
`encoder_samples` (split), `view_embeddings` (the measure ledger) and the `w_counter`/`proj`
checkpoint. A missing source raises; the builder never fabricates a port from a wrong-width
served object.

    python3 scripts/build_port_artifact.py
    python3 scripts/build_port_artifact.py --dest looking_glass/artifacts/cfm_v2k_vf2_port
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import products_identity, stamp_merge_write  # noqa: E402

BANK = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_bank"
CERT = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_cert"
DEST = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_port"
# DEC-604 delivery-port contract (frozen): bank 179 + measure PCA-128 = 307.
PORT_DIM = 307
PCA_DIM = 128
RANDOM_STATE = 0  # principle-ok: frozen PCA contract (matches build_b_cert_bundle.py)


# --------------------------------------------------------------------------- pure logic


def _require_products(path: Path, what: str) -> Path:
    """The products DB this builder cannot run without; refuse instead of substituting."""
    db = Path(path) / "cfm_products.duckdb"
    if not db.is_file():
        raise FileNotFoundError(
            f"port builder: no {what} products at {path} ({db.name} missing) — refusing to "
            "fabricate a port; restore the artifact or rebuild the bank first"
        )
    return db


def _source_products(bank_dir: Path, cert_dir: Path) -> Path:
    """The products directory to copy: the served bank, else the cert bundle that carries it."""
    for path in (Path(bank_dir), Path(cert_dir)):
        if (path / "cfm_products.duckdb").is_file():
            return path
    raise FileNotFoundError(
        f"port builder: no source products — neither {bank_dir} nor {cert_dir} has "
        "cfm_products.duckdb; the DEC-604 port cannot be materialised from repo data"
    )


def _uniform_width(rows: dict[str, np.ndarray], what: str) -> int:
    """The one width every row shares; a ragged/2-D source is a refusal, not a reshape."""
    widths = {np.asarray(v).shape for v in rows.values()}
    if len(widths) != 1:
        raise ValueError(f"{what}: ragged row shapes {sorted(str(w) for w in widths)[:3]}")
    shape = next(iter(widths))
    if len(shape) != 1:
        raise ValueError(f"{what}: rows are not 1-D (shape {shape})")
    return int(shape[0])


def measure_part_order() -> list[str]:
    """The published measure-view part order (one definition: `view_layout`)."""
    from looking_glass import view_layout as V

    return list(V.PART_ORDER["measure"])


def compose_port(
    bank: dict[str, np.ndarray],
    measure: dict[str, dict[str, np.ndarray]],
    split: dict[str, str],
    *,
    pca_dim: int = PCA_DIM,
    part_order: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """bank 179 + measure PCA-128 — `build_b_cert_bundle.py`'s DEC-604 composition.

    Rows are emitted in `bank` order and aligned by `customer_key`, never by dict insertion
    order in `measure`: a customer's PCA row is the transform of THAT customer's measure parts.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    keys = list(bank)
    if not keys:
        raise ValueError("port composition: the bank has no rows")
    order = list(part_order) if part_order is not None else measure_part_order()
    if not order:
        raise ValueError("port composition: the measure view has no parts")
    missing_keys = [k for k in keys if k not in measure]
    if missing_keys:
        raise ValueError(
            f"port composition: {len(missing_keys)} bank row(s) have no measure view row"
        )
    missing_parts = sorted({p for k in keys for p in order if p not in measure[k]})
    if missing_parts:
        raise ValueError(f"port composition: measure view has no parts {missing_parts}")
    full = np.vstack([np.concatenate([measure[k][p] for p in order]) for k in keys]).astype(
        np.float64
    )
    if pca_dim > min(full.shape):
        raise ValueError(
            f"port composition: PCA({pca_dim}) cannot fit a {full.shape[0]}x{full.shape[1]} "
            "measure matrix"
        )
    fit_idx = [i for i, k in enumerate(keys) if split.get(k) == "A"]
    if len(fit_idx) < pca_dim:
        raise ValueError(
            f"port composition: {len(fit_idx)} split-A row(s) cannot fit PCA({pca_dim})"
        )
    scaler = StandardScaler().fit(full[fit_idx])
    pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit(scaler.transform(full[fit_idx]))
    rows = pca.transform(scaler.transform(full))
    return {
        k: np.concatenate([np.asarray(bank[k], dtype=np.float64), rows[i]])
        for i, k in enumerate(keys)
    }


def validate_port(
    port: dict[str, np.ndarray],
    *,
    bank_keys: object | None = None,
    dim: int = PORT_DIM,
) -> int:
    """Refuse a port that is not `dim` wide, bank-key aligned and finite. Returns the row count."""
    if not port:
        raise ValueError("port validation: the port has no rows")
    if bank_keys is not None:
        want, got = {str(k) for k in bank_keys}, set(port)
        missing, extra = want - got, got - want
        if missing or extra:
            raise ValueError(
                "port validation: rows do not align with the bank by customer_key "
                f"(missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]})"
            )
    widths = set()
    bad: list[str] = []
    for k, v in port.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"port validation: row {k!r} is not 1-D (shape {arr.shape})")
        widths.add(int(arr.shape[0]))
        if not np.all(np.isfinite(arr)):
            bad.append(k)
    if widths != {dim}:
        raise ValueError(f"port validation: port width is {sorted(widths)}, not {dim}")
    if bad:
        raise ValueError(f"port validation: non-finite rows {sorted(bad)[:3]}")
    return len(port)


# --------------------------------------------------------------------------- repo sources


def _read_bank(
    db: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, str],
    dict[str, tuple[float, str]],
]:
    """(bank rows, measure parts, split, (as_of_epoch, version)) from a products DB."""
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            "select customer_key, embedding, as_of_epoch, version from state_embeddings"
        ).fetchall()
        measure_rows = con.execute(
            "select customer_key, part, embedding from view_embeddings where view='measure'"
        ).fetchall()
        split = {
            str(k): str(s)
            for k, s in con.execute("select customer_key, split from encoder_samples").fetchall()
        }
    finally:
        con.close()
    bank: dict[str, np.ndarray] = {}
    meta: dict[str, tuple[float, str]] = {}
    for key, emb, as_of, version in rows:
        k = str(key)
        bank[k] = np.asarray(emb, dtype=np.float64)
        meta[k] = (float(as_of), str(version))
    measure: dict[str, dict[str, np.ndarray]] = {}
    for key, part, emb in measure_rows:
        measure.setdefault(str(key), {})[str(part)] = np.asarray(emb, dtype=np.float64)
    if not bank:
        raise ValueError(f"port builder: state_embeddings at {db} is empty")
    return bank, measure, split, meta


def read_cert_port(cert_dir: Path) -> tuple[dict[str, np.ndarray], set[str]] | None:
    """The cert bundle's `port_embeddings` (307) plus its versions; None when the table is absent."""
    db = Path(cert_dir) / "cfm_products.duckdb"
    if not db.is_file():
        return None
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        tables = {
            str(r[0])
            for r in con.execute("select table_name from information_schema.tables").fetchall()
        }
        if "port_embeddings" not in tables:
            return None
        rows = con.execute("select customer_key, embedding from port_embeddings").fetchall()
        versions = {
            str(r[0])
            for r in con.execute("select distinct version from port_embeddings").fetchall()
        }
    finally:
        con.close()
    if not rows:
        raise ValueError(f"port builder: {db} has an empty port_embeddings table")
    port = {str(k): np.asarray(v, dtype=np.float64) for k, v in rows}
    if len(port) != len(rows):
        raise ValueError(f"port builder: {db} has duplicate customer_key rows in port_embeddings")
    return port, versions


def resolve_port(bank_dir: Path, cert_dir: Path) -> tuple[dict[str, np.ndarray], str, dict]:
    """(port, source label, inputs) from repo artifacts only; a missing source raises."""
    source = _source_products(bank_dir, cert_dir)
    bank, measure, split, meta = _read_bank(source / "cfm_products.duckdb")
    bank_dim = _uniform_width(bank, "port builder: bank state_embeddings")
    expected = bank_dim + PCA_DIM
    if expected != PORT_DIM:
        raise ValueError(
            f"port builder: bank {bank_dim} + measure PCA-{PCA_DIM} = {expected}, but the "
            f"DEC-604 port is {PORT_DIM}; the source bank has drifted"
        )
    cert_port = read_cert_port(cert_dir)
    if cert_port is not None:
        port, versions = cert_port
        want = {version for _as_of, version in meta.values()}
        if versions != want:
            raise ValueError(
                "port builder: the cert port_embeddings versions "
                f"{sorted(versions)} do not match the bank versions {sorted(want)}"
            )
        label = f"cert port_embeddings ({Path(cert_dir) / 'cfm_products.duckdb'})"
        source_kind = "cert_port_embeddings"
    else:
        port = compose_port(bank, measure, split)
        label = (
            f"composed bank {bank_dim} + measure PCA-{PCA_DIM} ({source / 'cfm_products.duckdb'})"
        )
        source_kind = "composed_bank_measure_pca"
    n = validate_port(port, bank_keys=bank.keys(), dim=PORT_DIM)
    info = {
        "bank": bank,
        "measure": measure,
        "split": split,
        "meta": meta,
        "source": source,
        "source_kind": source_kind,
        "bank_dim": bank_dim,
        "n_fit_A": sum(1 for k in bank if split.get(k) == "A"),
        "rows": n,
    }
    return port, label, info


# --------------------------------------------------------------------------- build


def build(
    bank_dir: Path,
    cert_dir: Path,
    dest: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Materialise the 307-dim port products at `dest`; refuse rather than guess."""
    t0 = time.perf_counter()
    bank_dir, cert_dir, dest = Path(bank_dir), Path(cert_dir), Path(dest)
    source = _source_products(bank_dir, cert_dir)
    if dest.resolve() in {
        Path(bank_dir).resolve(),
        Path(cert_dir).resolve(),
        source.resolve(),
    }:
        raise SystemExit("port builder: --dest must differ from the source artifacts")
    # Resolve/validate the source BEFORE touching an existing dest: a corrupt source must not
    # delete the previous good artifact.
    port, label, info = resolve_port(bank_dir, cert_dir)
    if dest.exists():
        if dest.is_file():
            if not overwrite:
                raise SystemExit(f"port builder: {dest} is a file; pass --overwrite to replace it")
            dest.unlink()
        elif any(dest.iterdir()):
            if not overwrite:
                raise SystemExit(f"port builder: {dest} exists; pass --overwrite to rebuild it")
            shutil.rmtree(dest)
        else:
            dest.rmdir()

    source_id = products_identity(str(source))
    cert_used = info["source_kind"] == "cert_port_embeddings"

    shutil.copytree(source, dest)

    import duckdb

    db = dest / "cfm_products.duckdb"
    versions = {version for _as_of, version in info["meta"].values()}
    if len(versions) != 1:
        raise ValueError(f"port builder: state_embeddings carries multiple versions {versions}")
    tag = next(iter(versions))
    keys = list(port)
    con = duckdb.connect(str(db))
    try:
        import pyarrow as pa

        port_rows = pa.table(
            {
                "customer_key": keys,
                "dim": pa.array([PORT_DIM] * len(keys), type=pa.int32()),
                "embedding": pa.array(
                    [port[k].astype(np.float32).tolist() for k in keys],
                    type=pa.list_(pa.float32()),
                ),
            }
        )
        con.register("port_rows", port_rows)
        try:
            con.execute("drop table if exists port_embeddings")
            con.execute(
                "create table port_embeddings as select customer_key, "
                "cast(? as varchar) as version, dim, embedding from port_rows",
                [tag],
            )
            # One set-based update, not one statement per customer: the per-row path made a
            # deterministic rebuild take minutes on the 5000x307 rows.
            con.execute(
                "update state_embeddings as s set embedding = p.embedding, dim = p.dim "
                "from port_rows as p where s.customer_key = p.customer_key"
            )
        finally:
            con.unregister("port_rows")
        dims = {
            int(r[0]) for r in con.execute("select distinct dim from state_embeddings").fetchall()
        }
        if dims != {PORT_DIM}:
            raise SystemExit(f"port builder: state_embeddings dims did not update cleanly: {dims}")
        state = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
        explicit = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in con.execute(
                "select customer_key, embedding from port_embeddings"
            ).fetchall()
        }
        if set(state) != set(keys) or set(explicit) != set(keys):
            raise SystemExit("port builder: state_embeddings/port_embeddings keys disagree")
        for k in keys:
            if not np.array_equal(state[k], explicit[k]):
                raise SystemExit(f"port builder: stored port row for {k!r} differs between tables")
    finally:
        con.close()
    n = len(port)

    regs = sorted(source.glob("registry_*.json"), key=lambda p: p.stat().st_mtime)
    if not regs:
        raise SystemExit(f"port builder: no registry_*.json under {source}")
    transform = {
        "method": (
            "state_embeddings = DEC-604 delivery port = bank (179) + measure view PCA-128 "
            "(StandardScaler + PCA(random_state=0), fit on split A only), concatenated by "
            "customer_key; the same composition build_b_cert_bundle.py writes to "
            "port_embeddings"
        ),
        "source_kind": info["source_kind"],
        "source": label,
        "source_products": source_id,
        "cert": products_identity(str(cert_dir)) if cert_used else None,
        "source_tag": tag,
        "bank_dim": int(info["bank_dim"]),
        "measure_pca_dim": PCA_DIM,
        "served_dim": PORT_DIM,
        "random_state": RANDOM_STATE,
        "n_fit_A": int(info["n_fit_A"]),
        "n_rows": n,
        "table": "port_embeddings",
        "state_table": "state_embeddings",
    }
    derived = json.loads(regs[-1].read_text())
    derived["served_transform"] = transform
    (dest / f"registry_{tag.replace('.', '_')}.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True, default=str) + "\n"
    )

    body = {
        "kind": "delivery_port_artifact",
        "complete": True,
        "tag": tag,
        "products": products_identity(str(dest)),
        "source": {"kind": info["source_kind"], "label": label, "products": source_id},
        "port": {
            "table": "port_embeddings",
            "state_table": "state_embeddings",
            "dim": PORT_DIM,
            "rows": n,
            "bank_dim": int(info["bank_dim"]),
            "measure_pca_dim": PCA_DIM,
            "fit": "split A only",
            "random_state": RANDOM_STATE,
            "n_fit_A": int(info["n_fit_A"]),
        },
        "why": (
            "B3.2/B3.4 certify the 307-dim DEC-604 delivery port; this artifact makes that "
            "object repo-resident so both receipts are reproducible without /tmp"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(
        dest / "port_receipt.json",
        body,
        " ".join([Path(__file__).name] + sys.argv[1:]),
        stamp_blocks=False,
    )
    body["source_label"] = label
    return body


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default=str(BANK), help="served bank products (179-dim source)")
    ap.add_argument("--cert", default=str(CERT), help="cert bundle carrying port_embeddings")
    ap.add_argument("--dest", default=str(DEST), help="port products directory to write")
    ap.add_argument("--overwrite", action="store_true", help="rebuild an existing --dest")
    a = ap.parse_args()
    try:
        body = build(Path(a.bank), Path(a.cert), Path(a.dest), overwrite=a.overwrite)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"port builder: {exc}") from exc
    print(f"port artifact -> {body['products']['path']} ({body['seconds']}s)")
    print(
        f"  state_embeddings: {body['port']['dim']}-d x {body['port']['rows']} rows "
        f"({body['port']['table']} mirrors it)"
    )
    print(f"  source: {body['source_label']}")
    print(f"  WROTE {Path(a.dest) / 'port_receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
