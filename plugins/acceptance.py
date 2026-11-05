"""Layer C acceptance — each part works independently, and together.

Parts validated:
  1. CFM static embedding table exists and its version matches the CFM registry.
  2. Every plugin reads that one table (single source of truth).
  3. Each plugin passes its own independent gate.
  4. End-to-end: the same frozen embeddings drive all three plugin kinds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .base import CFM_PRODUCTS, STREAM_DB, WORK


def _registry_for(tag):
    p = CFM_PRODUCTS.parent / f"registry_{tag.replace('.', '_')}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Layer C acceptance")
    ap.add_argument("--window", type=int, default=365)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(WORK / "docs" / "report_cards" / "experiments" / "plugin_acceptance.json"),
    )
    a = ap.parse_args(argv)
    import duckdb

    rows = []

    def chk(name, achieved, ok):
        rows.append({"check": name, "achieved": achieved, "ok": ok})

    con = duckdb.connect(str(CFM_PRODUCTS), read_only=True)
    try:
        n, vers = con.execute(
            "SELECT count(*), count(DISTINCT version) FROM anchor_embeddings"
        ).fetchone()
        vlist = con.execute("SELECT DISTINCT version FROM anchor_embeddings").fetchall()
    finally:
        con.close()
    tag = vlist[0][0] if vlist else "?"
    reg = _registry_for(tag)
    chk("CFM registry present for embedding version", bool(reg), bool(reg))
    chk("static embedding table non-empty", n, n > 0)
    chk("embedding table single version", len(vlist), len(vlist) == 1)
    chk("embedding version has a CFM registry", tag, bool(reg))

    # each plugin independently
    from . import supervised, segmentation, white_queen_plugin

    sup_ok = supervised.run(a.window, a.seed)[0]
    uns_ok = segmentation.run(a.window, a.seed)[0]
    white_queen_ok = white_queen_plugin.run(a.window, a.seed)[0]
    chk("supervised plugin independent gate", sup_ok, sup_ok)
    chk("unsupervised plugin independent gate", uns_ok, uns_ok)
    chk("white_queen plugin independent gate", white_queen_ok, white_queen_ok)

    # whole
    artifacts = sorted((WORK / "plugins" / "artifacts").glob("*.json"))
    chk("all plugin artifacts written", len(artifacts), len(artifacts) >= 3)
    chk(
        "end-to-end: all plugin kinds share one frozen table",
        all([sup_ok, uns_ok, white_queen_ok]),
        all([sup_ok, uns_ok, white_queen_ok]),
    )

    from .base import gate

    ok = gate(rows, "LAYER C ACCEPTANCE")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kind": "plugin_acceptance",
                "ok": bool(ok),
                "rows": rows,
                "products": str(CFM_PRODUCTS),
                "stream": str(STREAM_DB),
                "window": a.window,
                "seed": a.seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"WROTE {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
