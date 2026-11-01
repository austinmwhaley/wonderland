#!/usr/bin/env python3
"""Rank attainability — is the geometry target supported by the stream? (read-only diagnostic)

DEC-514. The card asks the encoder for `rank_target` (~30 of 256 directions) while the frozen
state measures ~6. Before spending more compute chasing 0.117, ask whether the stream's own
exhaustive full-history measure spans 30 directions at all.

This script MEASURES and never adopts anything. It writes a receipt and prints. It does not
touch a gate, a threshold, or the encoder.

    python3 scripts/measure_rank_attainability.py \
        --products /home/austin_whaley/wonderland_work/cfm_m8_counter \
        --db /home/austin_whaley/wonderland_work/demo/customer_event_stream.parquet

What it reports, per representation (learned shape / exhaustive measure / served E):

* `eff_rank` and its column-permuted null (`looking_glass.portfolio._geometry`, the SAME
  definition the portfolio gate grades, so there is one definition of rank in the repo);
* `signal_dims`: correlation eigenvalues above the Marchenko-Pastur edge
  `lambda+ = (1 + sqrt(D/N))^2` — how many directions carry more than noise;
* `attainable_target`: `signal_dims / dim`, the target the derivation should have produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from looking_glass.cfm_config import CFMConfig  # noqa: E402
from looking_glass.layer_b import generic_measure  # noqa: E402
from looking_glass.portfolio import _geometry  # noqa: E402


def _signal_dims(x: np.ndarray, seed: int = 0) -> dict:
    """How many directions carry more than sampling noise — two scale-free criteria.

    1. `mp_signal_dims`: eigenvalues above the Marchenko-Pastur edge
       `lambda+ = (1 + sqrt(D/N))^2` (a fixed threshold; over-counts when D is large).
    2. `parallel_dims`: order-matched **parallel analysis** — compare each eigenvalue of the
       observed correlation matrix against the SAME-order eigenvalue of the column-permuted
       null (marginals kept, joint structure destroyed). This is the standard criterion and
       is not biased by D. It is the number to quote.

    Both are computed on the CORRELATION matrix, so raw column scale (counts vs sums vs rates)
    cannot masquerade as rank. Vectorized: two eigvalsh over D x D, no per-row loop.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    z = x - x.mean(0, keepdims=True)
    sd = z.std(0, ddof=1).clip(min=1e-12)
    zs = z / sd
    corr = zs.T @ zs / max(n - 1, 1)
    ev = np.linalg.eigvalsh(corr).clip(min=0.0)

    null = np.column_stack([x[:, c][rng.permutation(n)] for c in range(d)])
    zn = null - null.mean(0, keepdims=True)
    sdn = zn.std(0, ddof=1).clip(min=1e-12)
    corr_n = (zn / sdn).T @ (zn / sdn) / max(n - 1, 1)
    ev_n = np.linalg.eigvalsh(corr_n).clip(min=0.0)

    edge = float((1.0 + np.sqrt(d / max(n, 1))) ** 2)
    return {
        "n": int(n),
        "dim": int(d),
        "mp_edge": round(edge, 4),
        "mp_signal_dims": int((ev > edge).sum()),
        "parallel_dims": int((ev > ev_n).sum()),
        "parallel_frac": round(float((ev > ev_n).sum()) / max(d, 1), 4),
        "top_eig": round(float(ev[-1]), 3),
        "top5_eig": [round(float(v), 3) for v in ev[-5:][::-1]],
        "eig_above_edge_sum_frac": round(float(ev[ev > edge].sum() / ev.sum()), 4),
    }


def _block(x: np.ndarray, seed: int) -> dict:
    """The portfolio gate's geometry definition + the scale-free signal count, in one block."""
    out = _geometry(np.asarray(x, dtype=np.float64), seed)
    out.update(_signal_dims(x, seed))
    return out


def _load_cfg(reg: dict, db: Path) -> CFMConfig:
    """Rebuild the training config from the registry so the measure matches the run."""
    raw = dict(reg.get("config") or {})
    fields = {k for k, f in CFMConfig.__dataclass_fields__.items() if f.init}
    cfg = CFMConfig(**{k: v for k, v in raw.items() if k in fields})
    # the registry records the container path; point at the local stream
    cfg.db = str(db)
    for k in ("action_fields", "effect_actions", "numeric_fields", "attr_fields", "action_keys"):
        if k in raw and raw[k] is not None:
            setattr(cfg, k, tuple(raw[k]))
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", required=True, help="dir holding cfm_products.duckdb + registry")
    ap.add_argument("--db", required=True, help="the event stream the encoder trained on")
    ap.add_argument("--as-of", default=None, help="override the registry as_of")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=str(ROOT / "docs/report_cards/experiments/rank_attainability.json")
    )
    a = ap.parse_args(argv)

    import duckdb

    prod = Path(a.products)
    db = prod if prod.is_file() else prod / "cfm_products.duckdb"
    regs = sorted(prod.glob("registry_*.json"))
    if not regs:
        print(f"no registry_*.json under {prod}", file=sys.stderr)
        return 2
    reg_path = regs[-1]
    reg = json.loads(reg_path.read_text())
    cfg = _load_cfg(reg, Path(a.db))
    if a.as_of:
        cfg.as_of = a.as_of

    con = duckdb.connect(str(db), read_only=True)
    try:
        bkeys = [
            r[0]
            for r in con.execute(
                "select customer_key from encoder_samples where split='B'"
            ).fetchall()
        ]
        emb = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
        vers = [
            r[0] for r in con.execute("select distinct version from state_embeddings").fetchall()
        ]
    finally:
        con.close()

    keys = [k for k in bkeys if k in emb]
    if len(keys) < 32:
        print(f"only {len(keys)} held-out B states — too few to measure", file=sys.stderr)
        return 2
    shape = np.vstack([emb[k] for k in keys])

    measure, names = generic_measure(cfg, keys)
    measure = np.asarray(measure, dtype=np.float64)
    if measure.shape[0] != len(keys):
        print(
            f"measure rows {measure.shape[0]} != keys {len(keys)} — refusing to misalign",
            file=sys.stderr,
        )
        return 2

    blocks = {
        "learned_shape": _block(shape, a.seed),
        "full_history_measure": _block(measure, a.seed),
        "served_E": _block(np.hstack([shape, measure]), a.seed),
    }
    target = float(getattr(cfg, "rank_target", 0.0) or 0.0)
    attainable = blocks["full_history_measure"]["parallel_frac"]
    body = {
        "_meta": {
            "producer": Path(__file__).name,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join([Path(__file__).name] + sys.argv[1:]),
            "read_only": True,
            "products": str(prod),
            "registry": reg_path.name,
            "registry_sha256": hashlib.sha256(reg_path.read_bytes()).hexdigest()[:16],
            "encoder_tag": reg.get("tag"),
            "encoder_versions_served": vers,
            "db": str(a.db),
            "as_of": getattr(cfg, "as_of", None),
            "n": int(len(keys)),
            "n_measure_columns": int(measure.shape[1]),
            "measure_fields": len(names),
        },
        "geometry": blocks,
        "comparison": {
            "card_rank_target": round(target, 4),
            "card_rank_target_dims": round(target * shape.shape[1], 2),
            "learned_eff_rank": blocks["learned_shape"]["eff_rank"],
            "learned_null_eff_rank": blocks["learned_shape"]["eff_rank_null"],
            "measure_parallel_dims": blocks["full_history_measure"]["parallel_dims"],
            "measure_parallel_frac": attainable,
            "attainable_target_frac": attainable,
            "learned_over_attainable": round(
                blocks["learned_shape"]["eff_rank"]
                / max(blocks["full_history_measure"]["parallel_dims"], 1e-9),
                4,
            ),
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n")

    c = body["comparison"]
    print(f"\nrank attainability -> {out}")
    print(
        f"  n={body['_meta']['n']}  shape_dim={shape.shape[1]}  "
        f"measure_dim={measure.shape[1]}  tag={body['_meta']['encoder_tag']}"
    )
    for name, g in blocks.items():
        print(
            f"  {name:22s} eff_rank {g['eff_rank']:8.3f} / null {g['eff_rank_null']:8.3f}"
            f"  (ratio {g['eff_rank_ratio']:.3f})  parallel_dims {g['parallel_dims']:3d}"
            f"  ({g['parallel_frac']:.4f})  mp_dims {g['mp_signal_dims']:3d}  pr_score {g['pr_score']:+.3f}"
        )
    print(
        f"  target {c['card_rank_target']:.4f} ({c['card_rank_target_dims']:.1f} dims)"
        f"  vs measure parallel_frac {c['measure_parallel_frac']:.4f}"
        f"  ({c['measure_parallel_dims']} dims above its own null)"
    )
    print(
        f"  learned_eff_rank / measure_parallel_dims = {c['learned_over_attainable']:.2f}"
        "  (>1: the shape already spans more directions than the measure's signal count)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
