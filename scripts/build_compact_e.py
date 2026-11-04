"""Build the compact SERVED embedding from a trained candidate (B5.1/B6.2).

The full served E of the 0.11.1 candidate is 272 dims = 256 shape dims + the 16-dim level
block. The minimality certificate (DEC-588) says the shape block is compressible to its
first 32 PCs without losing the self-supervised future family beyond the held-out noise
floor. This producer materialises exactly that object:

    shape block (first `dim - level_block_dim` columns) -> standardize (fit on split A)
                                                        -> PCA(rank) (fit on split A)
                                                        -> rank PCs
    level block   (trailing level_block_dim columns)    -> unchanged
    appended view (optional, e.g. the 4-dim response)   -> unchanged
    served E      = [rank shape PCs | level block | appended view]

DEC-592: every served block is in the one composition, not only in a diagnostic view.
The response view is the statistical bank's response block; appending it makes the
compact object the same E every consumer reads.

Split A is the fit side; split B is never touched. The builder writes a NEW products
directory (checkpoint + products DB + a DERIVED registry that records the transform and
the source registry sha), so the compact object has its own identity instead of silently
reusing the full one.

    python scripts/build_compact_e.py --source .../cfm_v2k_vf2 --dest .../cfm_v2k_vf2_c32

CPU, deterministic (PCA random_state=0); the DB rows are rewritten in place on a fresh
copy of the source products database.
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

DEFAULT_SOURCE = ROOT / "looking_glass/artifacts/cfm_v2k_vf2"
DEFAULT_DEST = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_c32"


def _load_rows(db: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute("select customer_key, embedding from state_embeddings").fetchall()
        split = dict(con.execute("select customer_key, split from encoder_samples").fetchall())
    finally:
        con.close()
    keys = [str(r[0]) for r in rows]
    E = np.vstack([np.asarray(r[1], dtype=np.float64) for r in rows])
    return keys, E, split


def _load_view(db: Path, view: str) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Read one published view's per-customer parts from `view_embeddings`."""
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            "select customer_key, part, embedding from view_embeddings where view = ?", [view]
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(f"view {view!r} has no rows in view_embeddings")
    by_key: dict[str, dict[str, np.ndarray]] = {}
    parts: set[str] = set()
    for k, part, emb in rows:
        by_key.setdefault(str(k), {})[str(part)] = np.asarray(emb, dtype=np.float64)
        parts.add(str(part))
    return by_key, sorted(parts)


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument(
        "--level-block-dim",
        type=int,
        default=None,
        help="trailing level-block columns kept unchanged (default: len(LEVEL_COLS))",
    )
    ap.add_argument(
        "--append-block",
        action="append",
        default=[],
        dest="append_blocks",
        help=(
            "published view block to append, as view[:part[,part...]] (e.g. response, "
            "learned_state:rfm_recon); repeatable, order preserved"
        ),
    )
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from looking_glass.stat_bank import LEVEL_COLS

    t0 = time.perf_counter()
    source, dest = Path(a.source), Path(a.dest)
    if source.resolve() == dest.resolve():
        raise SystemExit("--source and --dest must differ")
    if dest.exists() and any(dest.iterdir()) and not a.overwrite:
        raise SystemExit(f"{dest} exists; pass --overwrite to rebuild it")
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "cfm_products.duckdb",
        "compact_e_receipt.json",
    ):
        (dest / name).unlink(missing_ok=True)
    for name in ("cfm_*.pt", "registry_*.json", "train.log"):
        for p in dest.glob(name):
            p.unlink()

    tag, source_reg = resolve_served_tag(str(source))
    source_meta = json.loads(source_reg.read_text())
    src_id = products_identity(str(source))
    src_db = _db_path(str(source))
    dst_db = dest / "cfm_products.duckdb"
    shutil.copy2(src_db, dst_db)
    for p in source.glob("cfm_*.pt"):
        shutil.copy2(p, dest / p.name)
    train_log = source / "train.log"
    if train_log.exists():
        shutil.copy2(train_log, dest / train_log.name)

    keys, E, split = _load_rows(dst_db)
    if E.ndim != 2 or E.shape[0] != len(keys):
        raise SystemExit(f"malformed state_embeddings: {E.shape} for {len(keys)} keys")
    dim = int(E.shape[1])
    level_dim = int(a.level_block_dim if a.level_block_dim is not None else len(LEVEL_COLS))
    shape_dim = dim - level_dim
    if shape_dim <= 0 or a.rank <= 0 or a.rank > shape_dim:
        raise SystemExit(
            f"bad dims: served {dim}, level block {level_dim}, shape {shape_dim}, rank {a.rank}"
        )
    fit = np.array([split.get(k) == "A" for k in keys])
    if int(fit.sum()) < max(10, a.rank):
        raise SystemExit(f"too few split-A customers to fit the transform: {int(fit.sum())}")

    extra_mats: list[np.ndarray] = []
    append_blocks: list[dict] = []
    for spec in a.append_blocks:
        view_name, _, parts_s = str(spec).partition(":")
        view, all_parts = _load_view(dst_db, view_name)
        want = (
            all_parts
            if not parts_s.strip()
            else [p.strip() for p in parts_s.split(",") if p.strip()]
        )
        unknown = [p for p in want if p not in all_parts]
        if unknown:
            raise SystemExit(f"view {view_name!r} has no parts {unknown}; available {all_parts}")
        missing = [k for k in keys if k not in view]
        if missing:
            raise SystemExit(f"{len(missing)} customers have no {view_name!r} view row; refusing")
        mat = np.vstack([np.concatenate([view[k][p] for p in want]) for k in keys]).astype(
            np.float64
        )
        if mat.shape[1] == 0:
            raise SystemExit(f"view {view_name!r} has no parts")
        extra_mats.append(mat)
        append_blocks.append({"view": view_name, "parts": want, "dim": int(mat.shape[1])})
    extra = np.hstack(extra_mats) if extra_mats else np.zeros((len(keys), 0), dtype=np.float64)

    shape, block = E[:, :shape_dim], E[:, shape_dim:]
    scaler = StandardScaler().fit(shape[fit])
    pca = PCA(n_components=int(a.rank), random_state=0).fit(scaler.transform(shape[fit]))
    compact = np.hstack([pca.transform(scaler.transform(shape)), block, extra]).astype(np.float64)
    block_dim = int(block.shape[1] + extra.shape[1])
    if compact.shape != (len(keys), a.rank + block_dim):
        raise SystemExit(f"compact shape {compact.shape} inconsistent")

    import duckdb

    con = duckdb.connect(str(dst_db))
    try:
        for k, v in zip(keys, compact):
            con.execute(
                "update state_embeddings set embedding=?, dim=? where customer_key=?",
                [v.tolist(), int(compact.shape[1]), k],
            )
        dims = [r[0] for r in con.execute("select distinct dim from state_embeddings").fetchall()]
        if dims != [int(compact.shape[1])]:
            raise SystemExit(f"state_embeddings dims did not update cleanly: {dims}")
    finally:
        con.close()

    transform = {
        "method": (
            "shape block (leading dim-level_block_dim columns) standardized and PCA-projected "
            "to rank k on split A; level block (trailing level_block_dim columns) unchanged; "
            "published view appended unchanged"
        ),
        "source_registry_sha256": src_id.get("registry_sha256"),
        "source_tag": tag,
        "source_dim": dim,
        "shape_dim": int(shape_dim),
        "rank": int(a.rank),
        "level_block_dim": int(level_dim),
        "append_blocks": append_blocks,
        "append_dim": int(extra.shape[1]),
        "block_dim": int(block_dim),
        "served_dim": int(compact.shape[1]),
        "n_fit_A": int(fit.sum()),
        "explained_variance_ratio_sum": round(float(pca.explained_variance_ratio_.sum()), 8),
        "scaler_mean_sha256": hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:16],
        "pca_components_sha256": hashlib.sha256(pca.components_.tobytes()).hexdigest()[:16],
    }
    derived = dict(source_meta)
    derived["served_transform"] = transform
    dest_reg = dest / f"registry_{tag.replace('.', '_')}.json"
    dest_reg.write_text(json.dumps(derived, indent=2, sort_keys=True, default=str) + "\n")

    body = {
        "complete": True,
        "tag": tag,
        "source": src_id,
        "dest": products_identity(str(dest)),
        "transform": transform,
        "why": (
            "the compact served object is a deterministic rank-k reconstruction of the shape "
            "block (fit on split A only), with the level block and any appended published "
            "view carried unchanged; the derived registry records the source registry sha and "
            "the transform so the compact artifact has its own identity"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (dest / "compact_e_receipt.json").write_text(
        json.dumps(body, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({k: v for k, v in body.items() if k != "why"}, indent=2, default=str))
    print("WROTE", dest / "compact_e_receipt.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
