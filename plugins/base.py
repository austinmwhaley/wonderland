"""Layer C plugins — independent heads over the frozen CFM embedding table.

Every plugin reads the SAME static table `anchor_embeddings` (produced by the
looking_glass encoder for sample-B customers at random uniform anchor days) and
declares its own target window. Anchors lacking enough forward data for a
plugin's window are dropped by that plugin only — the embedding layer is
agnostic.

Plugins are fully independent: separate artifacts, separate validation.
Kinds: `supervised`, `unsupervised`, `white_queen`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM_DB = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
OUT = WORK / "plugins" / "artifacts"


@dataclass
class PluginSpec:
    name: str
    kind: str  # supervised | unsupervised | white_queen
    target: str  # e.g. "gross_margin"
    window_days: int  # label horizon this plugin owns (1, 365, ...)
    version: str = "v1.0.0"
    revision: int = 1

    @property
    def tag(self) -> str:
        return f"{self.name}_{self.version}r{self.revision}"


@dataclass
class Dataset:
    spec: PluginSpec
    keys: np.ndarray  # customer keys per anchor row
    anchor_epoch: np.ndarray
    X: np.ndarray  # frozen embeddings
    y: np.ndarray  # realized target over the window
    x_base: np.ndarray  # trailing-window target (baseline feature)
    price_mat: np.ndarray | None = None  # additional covariates (e.g. sends)
    data_end: float = 0.0
    meta: dict = field(default_factory=dict)


def _anchor_table(con, table="anchor_embeddings"):
    return con.execute(f"SELECT customer_key, anchor_epoch, embedding FROM {table}").pl()


def _iso_epoch(s: str) -> float:
    from datetime import datetime, timezone

    d = datetime.fromisoformat(str(s))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def load_dataset(
    window_days: int,
    *,
    cfm_products=CFM_PRODUCTS,
    stream_db=STREAM_DB,
    target_col: str = "gross_margin",
    target=None,
    as_of: str | None = None,
    feature_table: str | None = None,
) -> Dataset:
    """Layer C owns its TARGET: read Layer B's state table, then compute the
    target against rabbit_hole at train time (no labels baked into Layer B).

    Additive extensions (legacy behavior unchanged when both are default):
      * ``target`` (plugins.targets.Target): kind drives the label contract —
        binary -> "any purchase in (anchor, anchor+window]"; continuous ->
        the summed ``target_col`` (the original CLV contract).
      * ``as_of`` (ISO date): point-in-time labels — orders are cut to <= as_of
        and only anchors whose FORWARD window is fully closed by as_of are
        eligible (``anchor + window <= as_of``). This is what makes monthly
        training leak-free: on the 1st you may only learn from closed windows.
    """
    import duckdb
    import polars as pl

    table = feature_table or getattr(target, "feature_table", "anchor_embeddings")
    pc = duckdb.connect(str(cfm_products), read_only=True)
    try:
        anchors = _anchor_table(pc, table)
        vers = [r[0] for r in pc.execute(f"SELECT DISTINCT version FROM {table}").fetchall()]
    finally:
        pc.close()
    encoder_version = vers[0] if len(vers) == 1 else None

    cutoff = _iso_epoch(as_of) if as_of is not None else None
    kind = getattr(target, "kind", "continuous")
    con = duckdb.connect(str(stream_db), read_only=True)
    try:
        data_end = float(
            con.execute(
                "SELECT max(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM customer_events"
            ).fetchone()[0]
        )
        orders_end = float(
            con.execute("SELECT max(epoch(CAST(order_ts AS TIMESTAMPTZ))) FROM orders").fetchone()[
                0
            ]
        )
        # labels can only be observed up to the LAST ORDER (not the last event):
        # capping at events_end would silently score truncated windows as zeros.
        eff_end = cutoff if cutoff is not None else min(data_end, orders_end)
        con.register("rh_anchors", anchors)
        W = int(window_days) * 86400
        time_cut = f"AND t <= {cutoff}" if cutoff is not None else ""
        if kind == "binary":
            # label = any order inside the forward window; baseline = same for
            # the trailing window (a trivial "did they buy recently" predictor)
            y_expr = (
                f"COALESCE(MAX(CASE WHEN t > a.anchor_epoch "
                f"AND t <= a.anchor_epoch + {W} THEN 1 ELSE 0 END), 0)"
            )
            xb_expr = (
                f"COALESCE(MAX(CASE WHEN t > a.anchor_epoch - {W} "
                f"AND t <= a.anchor_epoch THEN 1 ELSE 0 END), 0)"
            )
        else:
            y_expr = (
                f"COALESCE(SUM(CASE WHEN t > a.anchor_epoch "
                f"AND t <= a.anchor_epoch + {W} THEN o.{target_col} END), 0.0)"
            )
            xb_expr = (
                f"COALESCE(SUM(CASE WHEN t > a.anchor_epoch - {W} "
                f"AND t <= a.anchor_epoch THEN o.{target_col} END), 0.0)"
            )
        labels = con.execute(f"""
			SELECT a.customer_key, a.anchor_epoch,
				{y_expr} AS y,
				{xb_expr} AS x_base
			FROM rh_anchors a
			LEFT JOIN LATERAL (
				SELECT epoch(CAST(order_ts AS TIMESTAMPTZ)) AS t, gross_margin
				FROM orders WHERE customer_id = a.customer_key {time_cut}
			) o ON TRUE
			GROUP BY 1, 2
		""").pl()
    finally:
        con.close()
    flt = anchors.join(labels, on=["customer_key", "anchor_epoch"], how="inner")
    flt = flt.filter(pl.col("anchor_epoch") + W <= eff_end)
    if flt.height == 0:
        raise ValueError(f"no anchors with a full {window_days}d forward window")
    keys = flt["customer_key"].to_numpy()
    anchor_epoch = flt["anchor_epoch"].to_numpy().astype(np.float64)
    X = np.stack(flt["embedding"].to_list()).astype(np.float32)
    y = flt["y"].to_numpy()
    y = y.astype(np.float64) if kind == "continuous" else y.astype(np.int64)
    x_base = flt["x_base"].to_numpy()
    x_base = x_base.astype(np.float64) if kind == "continuous" else x_base.astype(np.int64)
    spec = (
        target.spec()
        if target is not None
        else PluginSpec(name="", kind="", target=target_col, window_days=window_days)
    )
    return Dataset(
        spec=spec,
        keys=keys,
        anchor_epoch=anchor_epoch,
        X=X,
        y=y,
        x_base=x_base,
        data_end=eff_end,
        meta={
            "n": int(flt.height),
            "n_customers": int(len(set(keys.tolist()))),
            "y_mean": float(y.mean()),
            "y_pos_rate": float((y > 0).mean()),
            "kind": kind,
            "target": getattr(target, "name", target_col),
            "as_of": as_of,
            "cutoff_epoch": cutoff,
            "orders_end": orders_end,
            "events_end": data_end,
            "feature_table": table,
            "encoder_version": encoder_version,
            "encoder_versions": vers,
        },
    )


def save_artifact(spec: PluginSpec, payload: dict, out: Path | None = None) -> Path:
    out = Path(out) if out is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{spec.tag}.json"
    p.write_text(
        json.dumps(
            {"spec": asdict(spec), "tag": spec.tag, "payload": payload}, indent=1, default=float
        )
    )
    return p


def gate(rows, name):
    ok = all(r["ok"] for r in rows)
    print(f"\n== {name} ==")
    print(f"{'check':46s} {'achieved':>16s}  status")
    for r in rows:
        print(f"{r['check']:46s} {str(r['achieved']):>16s}  {'PASS' if r['ok'] else 'FAIL'}")
    n = sum(1 for r in rows if r["ok"])
    print(f"completion: {n}/{len(rows)} ({100 * n / len(rows):.0f}%)")
    return ok
