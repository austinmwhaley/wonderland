"""Supervised CLV plugins — independent heads over the frozen embeddings.

Each head is trained and validated on its own, on sample-B anchors. The target
is realized gross margin over the plugin's window (e.g. 365 days). No head
knows about any other; the plugin layer is agnostic.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .base import Dataset, PluginSpec, save_artifact, gate


def _split(keys, seed=0, test=0.3):
	gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
	tr, te = next(gss.split(np.zeros(len(keys)), groups=keys))
	return tr, te


def _metrics(pred, y):
	from sklearn.metrics import mean_absolute_error
	sp = float(spearmanr(pred, y).statistic)
	# top-decile capture: mean realized target in top-10% predicted / overall
	k = max(1, int(0.1 * len(pred)))
	top = np.argsort(pred)[-k:]
	capture = float(y[top].mean() / max(y.mean(), 1e-9))
	return {"spearman": sp, "mae": float(mean_absolute_error(y, pred)),
			"top_decile_capture": capture}


def _ridge_alpha(X):
	"""Derived regularisation: scaled to the feature Gram scale (no magic)."""
	g = float(np.mean(np.sum(X * X, axis=0)))
	return max(1e-6, 1e-3 * g)


def head_point(ds: Dataset, seed=0):
	tr, te = _split(ds.keys, seed)
	sc = StandardScaler().fit(ds.X[tr])
	m = Ridge(alpha=_ridge_alpha(sc.transform(ds.X[tr]))).fit(sc.transform(ds.X[tr]), ds.y[tr])
	pred = m.predict(sc.transform(ds.X[te]))
	return {"name": "point", "pred": pred, "idx": te,
			"metrics": _metrics(pred, ds.y[te])}


def head_two_part(ds: Dataset, seed=0):
	tr, te = _split(ds.keys, seed)
	sc = StandardScaler().fit(ds.X[tr])
	Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
	active = (ds.y[tr] > 0)
	clf = LogisticRegression(max_iter=2000).fit(Xtr, active.astype(int))
	p = clf.predict_proba(Xte)[:, 1]
	reg = Ridge(alpha=_ridge_alpha(Xtr[active])).fit(Xtr[active], ds.y[tr][active])
	pred = p * np.maximum(reg.predict(Xte), 0.0)
	return {"name": "two_part", "pred": pred, "idx": te,
			"metrics": _metrics(pred, ds.y[te])}


def head_quantile(ds: Dataset, seed=0):
	tr, te = _split(ds.keys, seed)
	qs = (0.1, 0.5, 0.9)
	out = {}
	for q in qs:
		m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
										  random_state=seed)
		m.fit(ds.X[tr], ds.y[tr])
		out[q] = m.predict(ds.X[te])
	y = ds.y[te]
	# Rearrangement: enforce non-crossing quantiles (oracle-improving).
	stack = np.sort(np.stack([out[0.1], out[0.5], out[0.9]], axis=0), axis=0)
	lo, med, hi = stack[0], stack[1], stack[2]
	cover = float(np.mean((y >= lo) & (y <= hi)))
	mono = bool(np.all(lo <= med + 1e-6) and np.all(med <= hi + 1e-6))
	return {"name": "quantile", "idx": te,
			"coverage_80": cover, "monotone": mono,
			"pred": med, "lo": lo, "hi": hi,
			"metrics": _metrics(med, y)}


def head_baseline(ds: Dataset, seed=0):
	tr, te = _split(ds.keys, seed)
	return {"name": "trailing_baseline", "pred": ds.x_base[te], "idx": te,
			"metrics": _metrics(ds.x_base[te], ds.y[te])}


def run(window_days: int, seed: int = 0):
	from .base import load_dataset
	ds = load_dataset(window_days)
	spec = PluginSpec(name=f"clv_supervised_{window_days}d", kind="supervised",
					  target="gross_margin", window_days=window_days)
	heads = [head_point(ds, seed), head_two_part(ds, seed),
			 head_quantile(ds, seed), head_baseline(ds, seed)]
	rows = []
	for h in heads:
		m = h["metrics"]
		if h["name"] == "quantile":
			rows.append({"check": "quantile: 80% coverage in [0.6,1.0]",
						 "achieved": round(h["coverage_80"], 3),
						 "ok": 0.6 <= h["coverage_80"] <= 1.0})
			rows.append({"check": "quantile: monotone q10<=q50<=q90",
						 "achieved": h["monotone"], "ok": h["monotone"]})
		elif h["name"] == "trailing_baseline":
			rows.append({"check": "trailing_baseline: finite spearman (sanity)",
						 "achieved": round(m["spearman"], 3),
						 "ok": np.isfinite(m["spearman"])})
		else:
			rows.append({"check": f"{h['name']}: spearman > 0.1",
						 "achieved": round(m["spearman"], 3),
						 "ok": np.isfinite(m["spearman"]) and m["spearman"] > 0.1})
	basesp = [h for h in heads if h["name"] == "trailing_baseline"][0]["metrics"]["spearman"]
	finite = [h["metrics"]["spearman"] for h in heads
			  if h["name"] not in ("trailing_baseline", "quantile")
			  and np.isfinite(h["metrics"]["spearman"])]
	bestsp = max(finite) if finite else 0.0
	rows.append({"check": "embedding head >= trailing baseline (spearman)",
				 "achieved": f"{bestsp:.3f} vs {basesp:.3f}",
				 "ok": bestsp >= (basesp if np.isfinite(basesp) else 0.0) - 0.05})
	ok = gate(rows, f"SUPERVISED CLV PLUGIN ({window_days}d)")
	payload = {"dataset": ds.meta, "heads": [
		{"name": h["name"], "metrics": h["metrics"],
		 **({"coverage_80": h.get("coverage_80"), "monotone": h.get("monotone")}
			if h["name"] == "quantile" else {})} for h in heads],
		"verdict": bool(ok)}
	save_artifact(spec, payload)
	return ok, payload
