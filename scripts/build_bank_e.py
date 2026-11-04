"""Build the served BLOCK BANK for the B+C judge (DEC-598).

The B+C gap is geometry: the flat served E loses the static segment/attribute mass and the
judge's delivery arm appends the full 595-dim measure view, which dilutes it. This builder
materialises one served object with the donor blocks a simple consumer can select:

    c32rvd blocks (32 shape + 16 level + 4 response + 3 RFM + 4 action)
  + exact counter (39, `customer_state.embedding`)
  + schema-derived profile (low-cardinality signup attributes, one-hot)
  + numeric_stats (60), event_mass (7), return_ratio (6) from the measure view
  = 179-dim block bank

The full measure view stays untouched (the B3.1/B3.6 ledger rows still read it). A compact
`judge_ledger` view (PCA-39 of the measure, fit on split A) is added for the delivery column.

The bank receipt records every block's slice and the category levels; the B5.1 producer reads
the paired `compact_e_receipt.json` (rank 32, block_dim 147, served_dim 179).

    python scripts/build_bank_e.py --source .../cfm_v2k_vf2_c32rvd --dest .../cfm_v2k_vf2_bank

CPU, deterministic; the source artifact is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cfm_products import _db_path, resolve_served_tag  # noqa: E402
from measure_input_law import products_identity  # noqa: E402

DEFAULT_SOURCE = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_c32rvd"
DEFAULT_DEST = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_bank"
DEFAULT_DEMO = ROOT / "looking_glass/artifacts/demo_v2"


def _profile_block(demo: Path, keys: list[str]) -> tuple[np.ndarray, dict]:
    """One-hot profile of low-cardinality signup attributes, levels fit on split A."""
    import polars as pl

    raw = pl.read_parquet(demo / "customer_event_stream.parquet")
    sig = raw.filter(pl.col("event_type") == "customer_signup")
    if sig.height == 0:
        return np.zeros((len(keys), 0)), {"fields": [], "levels": {}}
    attrs = []
    for a in sig["event_attributes"].to_list():
        try:
            attrs.append(json.loads(a))
        except Exception:  # noqa: BLE001 - malformed attribute: no profile entry
            attrs.append({})
    fields = sorted(
        {
            k
            for obj in attrs
            for k, v in obj.items()
            if k not in ("signup_ts", "target_key") and isinstance(v, (str, int, float, bool))
        }
    )
    by_key = {str(k): obj for k, obj in zip(sig["customer_key"], attrs, strict=True)}
    levels: dict[str, list] = {}
    for f in fields:
        vals = sorted({str(by_key[k].get(f)) for k in keys if f in by_key.get(k, {})})
        if 1 < len(vals) <= 32:
            levels[f] = vals
    cols = []
    for f, vals in levels.items():
        for v in vals:
            cols.append(
                np.asarray(
                    [1.0 if str(by_key.get(k, {}).get(f)) == v else 0.0 for k in keys],
                    dtype=np.float64,
                )
            )
    mat = np.column_stack(cols) if cols else np.zeros((len(keys), 0))
    return mat, {"fields": list(levels), "levels": {f: list(v) for f, v in levels.items()}}


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--demo", default=str(DEFAULT_DEMO))
    ap.add_argument("--ledger-dim", type=int, default=39)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    import duckdb
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    t0 = time.perf_counter()
    source, dest, demo = Path(a.source), Path(a.dest), Path(a.demo)
    if source.resolve() == dest.resolve():
        raise SystemExit("--source and --dest must differ")
    if dest.exists() and any(dest.iterdir()) and not a.overwrite:
        raise SystemExit(f"{dest} exists; pass --overwrite to rebuild it")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("cfm_products.duckdb", "bank_receipt.json", "compact_e_receipt.json"):
        (dest / name).unlink(missing_ok=True)
    for pat in ("cfm_*.pt", "registry_*.json", "train.log"):
        for p in dest.glob(pat):
            p.unlink()

    tag, source_reg = resolve_served_tag(str(source))
    source_meta = json.loads(source_reg.read_text())
    src_id = products_identity(str(source))
    shutil.copy2(_db_path(str(source)), dest / "cfm_products.duckdb")
    for p in source.glob("cfm_*.pt"):
        shutil.copy2(p, dest / p.name)
    if (source / "train.log").exists():
        shutil.copy2(source / "train.log", dest / "train.log")

    compact = json.loads((source / "compact_e_receipt.json").read_text())
    tr = compact["transform"]
    base_dim = int(tr["served_dim"])
    shape_dim = int(tr["rank"])
    if int(tr["shape_dim"]) != 256:
        raise SystemExit(f"source shape_dim {tr['shape_dim']} is not the 256 trunk shape")
    if base_dim - shape_dim != int(tr["block_dim"]):
        raise SystemExit("source compact receipt is inconsistent")

    con = duckdb.connect(str(dest / "cfm_products.duckdb"))
    try:
        rows = con.execute("select customer_key, embedding from state_embeddings").fetchall()
        keys = [str(r[0]) for r in rows]
        base = {str(r[0]): np.asarray(r[1], dtype=np.float64) for r in rows}
        split = dict(con.execute("select customer_key, split from encoder_samples").fetchall())
        counter = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in con.execute("select customer_key, embedding from customer_state").fetchall()
        }
        measure: dict[str, dict[str, np.ndarray]] = {}
        for k, part, emb in con.execute(
            "select customer_key, part, embedding from view_embeddings where view='measure'"
        ).fetchall():
            measure.setdefault(str(k), {})[str(part)] = np.asarray(emb, dtype=np.float64)
    finally:
        con.close()

    missing = [k for k in keys if k not in counter or k not in measure]
    if missing:
        raise SystemExit(f"{len(missing)} customers lack a counter/measure row; refusing")
    part_order = ["numeric_stats", "event_mass", "return_ratio"]
    if any(p not in measure[keys[0]] for p in part_order):
        raise SystemExit(f"source measure view lacks {part_order}")

    profile, profile_meta = _profile_block(demo, keys)
    blocks = [
        {"name": "shape", "start": 0, "dim": shape_dim},
        {"name": "level", "start": shape_dim, "dim": 16},
        {"name": "response", "start": shape_dim + 16, "dim": 4},
        {"name": "rfm", "start": shape_dim + 20, "dim": 3},
        {"name": "action", "start": shape_dim + 23, "dim": 4},
        {"name": "counter", "start": base_dim, "dim": int(counter[keys[0]].size)},
        {
            "name": "profile",
            "start": base_dim + int(counter[keys[0]].size),
            "dim": profile.shape[1],
        },
    ]
    pos = blocks[-1]["start"] + blocks[-1]["dim"]
    for p in part_order:
        d = int(measure[keys[0]][p].size)
        blocks.append({"name": p, "start": pos, "dim": d})
        pos += d
    bank_dim = pos

    bank = np.zeros((len(keys), bank_dim), dtype=np.float64)
    for i, k in enumerate(keys):
        bank[i] = np.concatenate(
            [
                base[k],
                counter[k],
                profile[i],
                *(measure[k][p] for p in part_order),
            ]
        )
    if bank.shape != (len(keys), bank_dim):
        raise SystemExit(f"bank shape {bank.shape} inconsistent")

    # Compact judge ledger: PCA of the full measure view, fit on split A only.
    full_measure = np.vstack(
        [np.concatenate([measure[k][p] for p in sorted(measure[k])]) for k in keys]
    )
    fit = np.asarray([split[k] == "A" for k in keys], dtype=bool)
    scaler = StandardScaler().fit(full_measure[fit])
    pca = PCA(n_components=int(a.ledger_dim), random_state=0).fit(
        scaler.transform(full_measure[fit])
    )
    ledger = pca.transform(scaler.transform(full_measure))

    con = duckdb.connect(str(dest / "cfm_products.duckdb"))
    try:
        for k, v in zip(keys, bank):
            con.execute(
                "update state_embeddings set embedding=?, dim=? where customer_key=?",
                [v.tolist(), bank_dim, k],
            )
        con.execute("delete from view_embeddings where view='judge_ledger'")
        for k, v in zip(keys, ledger):
            con.execute(
                "insert into view_embeddings values (?,?,?,?,?,?,?)",
                [k, 0.0, tag, "judge_ledger", "pca", int(ledger.shape[1]), v.tolist()],
            )
        dims = [r[0] for r in con.execute("select distinct dim from state_embeddings").fetchall()]
        if dims != [bank_dim]:
            raise SystemExit(f"state_embeddings dims did not update cleanly: {dims}")
    finally:
        con.close()

    block_meta = {
        "method": (
            "c32rvd blocks + exact counter + schema-derived low-cardinality signup profile "
            "+ measure parts numeric_stats/event_mass/return_ratio; the full measure view is "
            "kept and a compact judge_ledger (PCA of the measure, fit on split A) is added"
        ),
        "source": src_id,
        "source_tag": tag,
        "base_dim": base_dim,
        "shape_dim": shape_dim,
        "bank_dim": bank_dim,
        "block_dim": bank_dim - shape_dim,
        "blocks": blocks,
        "profile": profile_meta,
        "measure_parts": part_order,
        "judge_ledger": {
            "dim": int(ledger.shape[1]),
            "fit": "split A only",
            "explained_variance_ratio_sum": round(float(pca.explained_variance_ratio_.sum()), 8),
            "components_sha256": hashlib.sha256(pca.components_.tobytes()).hexdigest()[:16],
        },
        "n_fit_A": int(fit.sum()),
        "n_customers": len(keys),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (dest / "bank_receipt.json").write_text(
        json.dumps(block_meta, indent=2, sort_keys=True, default=str) + "\n"
    )
    compact_body = {
        "complete": True,
        "tag": tag,
        "source": src_id,
        "dest": None,
        "transform": {
            "method": block_meta["method"],
            "source_registry_sha256": src_id.get("registry_sha256"),
            "source_tag": tag,
            "source_dim": base_dim,
            "shape_dim": shape_dim,
            "rank": shape_dim,
            "level_block_dim": 16,
            "append_blocks": blocks[1:],
            "append_dim": bank_dim - shape_dim - 16,
            "block_dim": bank_dim - shape_dim,
            "served_dim": bank_dim,
            "n_fit_A": int(fit.sum()),
        },
        "why": (
            "one served block bank for the B+C judge; the shape stays the certified 32 PCs and "
            "every extra block is explicit and schema-derived; the compact judge ledger is a "
            "view, not a replacement for the full measure ledger"
        ),
    }
    derived = dict(source_meta)
    derived["served_transform"] = compact_body["transform"]
    dest_reg = dest / f"registry_{tag.replace('.', '_')}.json"
    dest_reg.write_text(json.dumps(derived, indent=2, sort_keys=True, default=str) + "\n")
    compact_body["dest"] = products_identity(str(dest))
    (dest / "compact_e_receipt.json").write_text(
        json.dumps(compact_body, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({k: v for k, v in block_meta.items() if k != "blocks"}, indent=2, default=str))
    print("WROTE", dest / "bank_receipt.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
