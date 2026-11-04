#!/usr/bin/env python3
"""PATH A provenance audit: the comparator behind every top-line FAIL, certified at the column level.

    .venv/bin/python scripts/measure_path_a_provenance.py \
        --products-dir /home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_b20 \
        --demo /home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo

Why (DEC-437/DEC-446). The card's top line is `brain > PATH A`, and PATH A is `table_features`
over the lab's wide table. If the table had been computed with the future included, every
"does not beat PATH A" FAIL would be an artifact of the comparison rather than a finding about
the encoder — and nothing on the card would say so. DEC-437 audited this by candidate search and
was withdrawn when version 2 OOM-killed the host; the close it named is here: the BUILDER stamps
which cut of the stream each column came from (`path_a_provenance`), and this audit verifies the
published table against that stamp — rebuild from the demo's own `relational/` inputs with the
current builder, require the columns and values to match the published artifact, then require
every column the comparator selects to be stamped `hist|customers` and every `fut` column to be
dropped by the comparator's own rule (`table_feature_columns`, imported from where it is defined).

Cost discipline (the reason DEC-437 died): NO candidate search, NO DuckDB planner surface, NO
hand-typed recipe map. One bounded polars aggregation (the builder itself, ~0.7 s on this demo)
plus a column-wise compare of a 6000-row table. The receipt embeds the provenance map itself, so
the published artifact carries the per-column facts, not a summary of them.

Exit codes follow the chain contract (DEC-441): 0 all checks pass, 1 a check FAILED (a
measurement), 2 the audit could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import _rollup  # noqa: E402  (one definition of a check block)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _value_diffs(published: pl.DataFrame, rebuilt: pl.DataFrame, cols: list[str]) -> list[dict]:
    """Column-wise compare of two frames already aligned and sorted on `customer_key`."""
    bad = []
    for c in cols:
        a, b = published[c], rebuilt[c]
        if a.dtype in (pl.Float32, pl.Float64) or b.dtype in (pl.Float32, pl.Float64):
            av = a.cast(pl.Float64).fill_null(np.nan).to_numpy()
            bv = b.cast(pl.Float64).fill_null(np.nan).to_numpy()
            if not np.array_equal(av, bv, equal_nan=True):
                nd = int(np.count_nonzero(av != bv))
                md = float(np.nanmax(np.abs(av - bv))) if nd else 0.0
                bad.append({"column": c, "n_different": nd, "max_abs_diff": md})
        else:
            if not a.equals(b):
                nd = int((a != b).fill_null(False).sum())
                bad.append({"column": c, "n_different": nd, "max_abs_diff": None})
    return bad


def audit(demo: Path, products_dir: Path | None) -> dict:
    """The four facts. Each returns (working, evidence); the receipt embeds all of them."""
    manifest = json.loads((demo / "manifest.json").read_text())
    as_of = datetime.fromisoformat(manifest["as_of"])
    table_path = demo / "path_a" / "path_a_wide_table.parquet"
    published = pl.read_parquet(table_path)

    from rabbit_hole.generators.path_a_build import build_path_a

    from demo.auto_encoder import table_feature_columns  # the comparator rule, from its definition

    base = pl.read_parquet(demo / "relational" / "events.parquet")
    customers = pl.read_parquet(demo / "relational" / "customers.parquet")
    features, labels, wide, state = build_path_a(base, customers, as_of, engine=None)
    prov = state["provenance"]

    checks: dict[str, dict] = {}

    # 1. columns: the stamp describes THIS table, not a hypothetical one.
    t0 = time.perf_counter()
    pub_cols, rebuilt_cols = set(published.columns), set(wide.columns)
    cols_ok = pub_cols == rebuilt_cols
    checks["columns_match"] = {
        "working": cols_ok,
        "evidence": (
            f"published {len(pub_cols)} cols vs rebuilt {len(rebuilt_cols)} cols "
            f"(missing: {sorted(pub_cols - rebuilt_cols)[:6]}, extra: {sorted(rebuilt_cols - pub_cols)[:6]})"
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }

    # 2. values: same inputs + current builder == the bytes every verdict measured. A drift here
    #    means the artifact predates this builder, so the stamp describes different bytes than the
    #    ones on disk — the audit must then say so, not grade the stamp in isolation.
    t0 = time.perf_counter()
    pub = published.sort("customer_key")
    reb = wide.sort("customer_key")
    keys_ok = pub["customer_key"].equals(reb["customer_key"])
    bad = (
        _value_diffs(pub, reb, sorted(pub_cols & rebuilt_cols - {"customer_key"}))
        if keys_ok
        else None
    )
    vals_ok = bool(keys_ok) and bad == []
    checks["values_match"] = {
        "working": vals_ok,
        "evidence": (
            (
                f"{pub.height} rows joined on customer_key, {len(sorted(pub_cols & rebuilt_cols)) - 1} "
                f"columns compared exactly (CPU-built artifact, CPU rebuild): identical"
                if bad == []
                else f"{len(bad)} column(s) differ from a rebuild of the demo's own relational inputs: "
                + "; ".join(
                    f"{d['column']} n={d['n_different']} max|Δ|={d['max_abs_diff']}"
                    for d in bad[:5]
                )
            )
            if keys_ok
            else "customer_key sets differ between published and rebuilt table"
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }

    # 3. completeness: every published column is attributable to a cut (or is the key).
    t0 = time.perf_counter()
    unexplained = sorted(pub_cols - {"customer_key"} - set(prov))
    checks["provenance_complete"] = {
        "working": not unexplained,
        "evidence": (
            f"{len(prov)} of {len(pub_cols) - 1} published columns stamped "
            f"(hist={sum(1 for v in prov.values() if v['source'] == 'hist')}, "
            f"fut={sum(1 for v in prov.values() if v['source'] == 'fut')}, "
            f"customers={sum(1 for v in prov.values() if v['source'] == 'customers')})"
            + (f"; UNEXPLAINED: {unexplained[:8]}" if unexplained else "")
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }

    # 4. the actual claim: what the comparator reads is pre-as_of, and what is post-as_of is
    #    exactly what the comparator drops. The drop rule is imported from its definition.
    t0 = time.perf_counter()
    comparator = table_feature_columns(published)
    violators = sorted(
        c for c in comparator if prov.get(c, {}).get("source") not in ("hist", "customers")
    )
    fut_cols = {c for c, v in prov.items() if v["source"] == "fut"}
    leaked_fut = sorted(c for c in fut_cols if c in set(comparator))
    dropped_fut = sorted(c for c in published.columns if c in fut_cols and c not in set(comparator))
    comp_ok = (
        bool(comparator) and not violators and not leaked_fut and len(dropped_fut) == len(fut_cols)
    )
    checks["comparator_clean"] = {
        "working": comp_ok,
        "evidence": (
            f"table_features selects {len(comparator)} columns, all stamped hist|customers; "
            f"all {len(fut_cols)} fut columns are dropped by the comparator's own rule "
            f"({len(dropped_fut)} dropped)"
            + (f"; IN COMPARATOR BUT NOT PRE-as_of: {violators[:6]}" if violators else "")
            + (f"; FUT COLUMN IN COMPARATOR: {leaked_fut}" if leaked_fut else "")
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }

    artifacts = {
        "path_a_wide_table.parquet": {
            "sha256": _sha256(table_path),
            "bytes": table_path.stat().st_size,
        },
        "relational/events.parquet": {
            "sha256": _sha256(demo / "relational" / "events.parquet"),
            "bytes": (demo / "relational" / "events.parquet").stat().st_size,
        },
        "relational/customers.parquet": {
            "sha256": _sha256(demo / "relational" / "customers.parquet"),
            "bytes": (demo / "relational" / "customers.parquet").stat().st_size,
        },
    }
    body = _rollup(checks)
    body["as_of"] = manifest["as_of"]
    body["provenance"] = prov
    body["comparator_columns"] = comparator
    body["artifacts"] = artifacts
    body["inputs"] = {
        "demo": str(demo),
        "products_dir": str(products_dir) if products_dir else None,
    }
    body["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return body


def main() -> int:
    # the same ceiling as every other producer (DEC-440): never run unbounded on a contested box
    from safe_run import ensure_boundary

    ensure_boundary()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--products-dir", default=None, help="recorded in the receipt for provenance")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    try:
        body = audit(Path(a.demo), Path(a.products_dir) if a.products_dir else None)
    except FileNotFoundError as exc:
        print(f"path_a_provenance: cannot run — {exc}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2))
    print(f"path_a provenance -> {out}")
    for name, c in body["checks"].items():
        print(f"  PATH-A :: {name}: {'PASS' if c['working'] else 'FAIL'}  ({c['seconds']}s)")
    print(f"path_a_provenance working={body['working']}")
    return 0 if body["working"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
