"""Layer B solo success test — does the frozen CFM embedding work on its own?

One number, one verdict. After the CFM is trained, this answers:
    "Is the representation good enough (by itself) to feed downstream plugins?"

Method (independent of training): a FROZEN-embedding linear probe on sample-B
anchors, compared against (a) raw causal features and (b) a scrambled embedding.
Targets are self-supervised and horizon-free and, crucially, are CUSTOMER
actions only — company actions are exogenous and never predicted (they are
excluded, matching the encoder's contract).

Score = mean, over probes, of how much of the raw>scrambled gap the CFM closes
(clipped to [0,1]). SUCCESS iff CFM beats scrambled on every probe AND is not
materially below raw on any probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

CFM_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "cfm"


def _company_actions(tag):
	for p in CFM_DIR.glob(f"registry_{tag.replace('.', '_')}.json"):
		cfg = json.loads(p.read_text()).get("config", {})
		return tuple(cfg.get("company_actions", ("email_send",)))
	return ("email_send",)


def _read_stream(db):
	if str(db).endswith((".arrow", ".feather", ".ipc")):
		return pl.read_ipc(db, memory_map=True)
	import duckdb
	con = duckdb.connect(db, read_only=True)
	try:
		return con.execute("SELECT customer_key, event_ts, event_type, value "
						   "FROM customer_events ORDER BY customer_key, event_ts").pl()
	finally:
		con.close()


def _read_anchors(products):
	import duckdb
	con = duckdb.connect(products, read_only=True)
	try:
		return con.execute(
			"SELECT customer_key, anchor_epoch, version, embedding FROM anchor_embeddings").pl()
	finally:
		con.close()


def _epoch(s):
	from datetime import datetime
	try:
		return datetime.fromisoformat(str(s)).timestamp()
	except Exception:
		return np.nan


def _v(x):
	try:
		return float(x)
	except (TypeError, ValueError):
		return 0.0


def build_rows(df, anchors, company):
	cols = {c: df[c].to_list() for c in df.columns}
	n = df.height
	by, i = {}, 0
	while i < n:
		k = cols["customer_key"][i]; j = i
		while j < n and cols["customer_key"][j] == k:
			j += 1
		ts = np.array([_epoch(x) for x in cols["event_ts"][i:j]], dtype=np.float64)
		by[k] = (ts, cols["event_type"][i:j], np.array([_v(x) for x in cols["value"][i:j]]))
		i = j
	etypes = sorted(set(map(str, cols["event_type"])) - set(company))
	Xc, Xr, y_type, y_val, y_dt, groups = [], [], [], [], [], []
	for row in anchors.iter_rows(named=True):
		k = row["customer_key"]; a = float(row["anchor_epoch"])
		if k not in by:
			continue
		ts, et, val = by[k]
		past = ts <= a
		fut = np.flatnonzero(ts > a)
		if past.sum() < 3 or len(fut) == 0:
			continue
		# next CUSTOMER action (skip exogenous company actions)
		nn = next((j for j in fut if str(et[j]) not in company), None)
		if nn is None:
			continue
		pt, pv = ts[past], val[past]
		pe = np.array(et)[past]
		span = max(pt[-1] - pt[0], 1.0)
		feats = [a - pt[-1], a - pt[0], float(past.sum()), float(pv.sum()), span,
				 float(np.mean(np.diff(pt))) if len(pt) > 1 else 0.0, float(np.mean(pv))]
		feats += [float(np.sum(pe == t)) for t in etypes]
		Xr.append(feats); Xc.append(list(row["embedding"]))
		y_type.append(etypes.index(str(et[nn])) if str(et[nn]) in etypes else 0)
		op = next((j for j in fut if str(et[j]) == "order_placed"), None)
		y_val.append(_v(val[op]) if op is not None else np.nan)
		y_dt.append(float(np.log1p(max(ts[nn] - a, 0.0))))
		groups.append(k)
	return (np.array(Xc, dtype=np.float64), np.array(Xr, dtype=np.float64),
			np.array(y_type), np.array(y_val, dtype=np.float64),
			np.array(y_dt, dtype=np.float64), np.array(groups))


def _clf(X, y, groups):
	if len(np.unique(y)) < 2:
		return float("nan")
	tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, groups))
	if len(np.unique(y[tr])) < 2:
		return float("nan")
	sc = StandardScaler().fit(X[tr])
	m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
	return float(f1_score(y[te], m.predict(sc.transform(X[te])), average="macro"))


def _reg(X, y, groups, seed=0):
	tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
	sc = StandardScaler().fit(X[tr])
	m = Ridge(alpha=1e-3 * float(np.mean(np.diag(X[tr].T @ X[tr]))) + 1e-9).fit(sc.transform(X[tr]), y[tr])
	return float(spearmanr(m.predict(sc.transform(X[te])), y[te]).statistic)


def evaluate(db, products, customers=500):
	anchors = _read_anchors(products)
	tag = anchors["version"][0]
	company = _company_actions(tag)
	df = _read_stream(db)
	keys = df["customer_key"].unique(maintain_order=True).to_list()[:customers]
	df = df.filter(pl.col("customer_key").is_in(keys))
	anchors = anchors.filter(pl.col("customer_key").is_in(keys))
	Xc, Xr, y_type, y_val, y_dt, groups = build_rows(df, anchors, company)
	if len(groups) < 20:
		return {"error": "too few anchors", "n": int(len(groups))}
	Xs = Xc[np.random.default_rng(0).permutation(len(Xc))]
	tasks = [
		("next customer-action type (macro-F1)", _clf, y_type, True),
		("next purchase value (spearman)", _reg, y_val, False),
		("next action delay (spearman)", _reg, y_dt, True),
	]
	rows, skills, informative = [], [], 0
	for name, fn, y, dense in tasks:
		m = np.isfinite(y) if not dense else np.ones(len(y), dtype=bool)
		cfm = fn(Xc[m], y[m], groups[m]); raw = fn(Xr[m], y[m], groups[m])
		scr = fn(Xs[m], y[m], groups[m])
		# A probe only counts if the target is learnable at all (raw beats chance).
		has_signal = (raw - scr) > 0.02
		if not has_signal:
			rows.append({"task": name + " [no signal]", "CFM": cfm, "raw": raw,
						 "scrambled": scr, "skill": 0.0, "ok": True})
			continue
		informative += 1
		gap = (raw - scr)
		skill = (cfm - scr) / gap
		skills.append(min(max(skill, 0.0), 1.0))
		rows.append({"task": name, "CFM": cfm, "raw": raw, "scrambled": scr,
					 "skill": skill,
					 "ok": bool(cfm > scr + 1e-6 and cfm >= raw - 0.05)})
	score = float(np.mean(skills)) if skills else 0.0
	success = informative >= 2 and all(r["ok"] for r in rows) and score >= 0.5
	return {"n": int(len(groups)), "version": str(tag), "company_actions": list(company),
			"rows": rows, "score": score, "informative": informative, "success": success}


def _show(res):
	if "error" in res:
		print("CFM solo test:", res); return False
	print(f"== CFM SOLO SUCCESS TEST ==  version={res['version']}  n={res['n']} "
		  f"(exogenous={','.join(res['company_actions'])})")
	print(f"{'probe':36s} {'CFM':>7s} {'raw':>7s} {'scram':>7s} {'skill':>6s}  ok")
	for r in res["rows"]:
		print(f"{r['task']:36s} {r['CFM']:7.3f} {r['raw']:7.3f} {r['scrambled']:7.3f} "
			  f"{r['skill']:6.2f}  {'Y' if r['ok'] else 'N'}")
	print(f"\nCFM SOLO SCORE: {res['score']:.2f}   (0..1; ≥0.5 and CFM>scrambled on all)")
	print("VERDICT:", "SUCCESS — safe for downstream plugins" if res["success"]
		  else "NOT YET — not independently validated for downstream use")
	return res["success"]


def main(argv=None):
	ap = argparse.ArgumentParser(description="Layer B solo success test")
	ap.add_argument("--db", default="data/arrow/customer_event_stream.feather")
	ap.add_argument("--products", default="artifacts/cfm/cfm_products.duckdb")
	ap.add_argument("--customers", type=int, default=500)
	a = ap.parse_args(argv)
	return 0 if _show(evaluate(a.db, a.products, a.customers)) else 1


if __name__ == "__main__":
	raise SystemExit(main())
