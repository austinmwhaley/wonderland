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
	kind: str                 # supervised | unsupervised | white_queen
	target: str               # e.g. "gross_margin"
	window_days: int          # label horizon this plugin owns (1, 365, ...)
	version: str = "v1.0.0"
	revision: int = 1

	@property
	def tag(self) -> str:
		return f"{self.name}_{self.version}r{self.revision}"


@dataclass
class Dataset:
	spec: PluginSpec
	keys: np.ndarray          # customer keys per anchor row
	anchor_epoch: np.ndarray
	X: np.ndarray             # frozen embeddings
	y: np.ndarray             # realized target over the window
	x_base: np.ndarray        # trailing-window target (baseline feature)
	price_mat: np.ndarray | None = None   # additional covariates (e.g. sends)
	data_end: float = 0.0
	meta: dict = field(default_factory=dict)


def _anchor_table(con, table="anchor_embeddings"):
	return con.execute(
		f"SELECT customer_key, anchor_epoch, embedding FROM {table}").pl()


def load_dataset(window_days: int, *, cfm_products=CFM_PRODUCTS,
				 stream_db=STREAM_DB, target_col: str = "gross_margin") -> Dataset:
	"""Layer C owns its TARGET: read Layer B's state table, then compute the
	target against rabbit_hole at train time (no labels baked into Layer B)."""
	import duckdb
	import polars as pl
	pc = duckdb.connect(str(cfm_products), read_only=True)
	try:
		anchors = _anchor_table(pc)
	finally:
		pc.close()
	con = duckdb.connect(str(stream_db), read_only=True)
	try:
		data_end = float(con.execute(
			"SELECT max(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM customer_events"
		).fetchone()[0])
		con.register("rh_anchors", anchors)
		W = int(window_days) * 86400
		labels = con.execute(f"""
			SELECT a.customer_key, a.anchor_epoch,
				COALESCE(SUM(CASE WHEN t > a.anchor_epoch
					AND t <= a.anchor_epoch + {W} THEN o.{target_col} END), 0.0) AS y,
				COALESCE(SUM(CASE WHEN t > a.anchor_epoch - {W}
					AND t <= a.anchor_epoch THEN o.{target_col} END), 0.0) AS x_base
			FROM rh_anchors a
			LEFT JOIN LATERAL (
				SELECT epoch(CAST(order_ts AS TIMESTAMPTZ)) AS t, gross_margin
				FROM orders WHERE customer_id = a.customer_key
			) o ON TRUE
			GROUP BY 1, 2
		""").pl()
	finally:
		con.close()
	flt = anchors.join(labels, on=["customer_key", "anchor_epoch"], how="inner")
	flt = flt.filter(pl.col("anchor_epoch") + W <= data_end)
	if flt.height == 0:
		raise ValueError(f"no anchors with a full {window_days}d forward window")
	keys = flt["customer_key"].to_numpy()
	anchor_epoch = flt["anchor_epoch"].to_numpy().astype(np.float64)
	X = np.stack(flt["embedding"].to_list()).astype(np.float32)
	y = flt["y"].to_numpy().astype(np.float64)
	x_base = flt["x_base"].to_numpy().astype(np.float64)
	return Dataset(spec=PluginSpec(name="", kind="", target=target_col,
								   window_days=window_days),
				   keys=keys, anchor_epoch=anchor_epoch, X=X, y=y,
				   x_base=x_base, data_end=data_end,
				   meta={"n": int(flt.height),
						 "n_customers": int(len(set(keys.tolist()))),
						 "y_mean": float(y.mean()),
						 "y_pos_rate": float((y > 0).mean())})


def save_artifact(spec: PluginSpec, payload: dict) -> Path:
	OUT.mkdir(parents=True, exist_ok=True)
	p = OUT / f"{spec.tag}.json"
	p.write_text(json.dumps({
		"spec": asdict(spec), "tag": spec.tag,
		"payload": payload}, indent=1, default=float))
	return p


def gate(rows, name):
	ok = all(r["ok"] for r in rows)
	print(f"\n== {name} ==")
	print(f"{'check':46s} {'achieved':>16s}  status")
	for r in rows:
		print(f"{r['check']:46s} {str(r['achieved']):>16s}  "
			  f"{'PASS' if r['ok'] else 'FAIL'}")
	n = sum(1 for r in rows if r["ok"])
	print(f"completion: {n}/{len(rows)} ({100*n/len(rows):.0f}%)")
	return ok
