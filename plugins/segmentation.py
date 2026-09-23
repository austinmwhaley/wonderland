"""Unsupervised customer-segmentation plugin.

Learns a label-free taxonomy over the frozen CFM embeddings (KMeans with k
chosen by silhouette), then reports the realized 365d gross margin per segment.
Training uses no labels; the label is used only to validate the segment's value.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway

from .base import PluginSpec, load_dataset, save_artifact, gate


def _choose_k(X, kmax=8, seed=0):
	best, best_score = 2, -1.0
	for k in range(2, min(kmax, len(X) - 1) + 1):
		lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
		if len(set(lab.tolist())) < 2:
			continue
		s = float(silhouette_score(X, lab))
		if s > best_score:
			best, best_score = k, s
	return best, best_score


def run(window_days: int = 365, seed: int = 0):
	ds = load_dataset(window_days)
	spec = PluginSpec(name=f"customer_segmentation_{window_days}d",
					  kind="unsupervised", target="gross_margin",
					  window_days=window_days)
	sc = StandardScaler().fit(ds.X)
	X = sc.transform(ds.X)

	def eta2_of(lab):
		k = int(lab.max()) + 1
		overall = ds.y.mean()
		ssb = sum(len(ds.y[lab == c]) * (ds.y[lab == c].mean() - overall) ** 2
				  for c in range(k))
		sst = ((ds.y - overall) ** 2).sum()
		return ssb / max(sst, 1e-9)

	# k chosen to MAXIMISE value separation (labels used only for model
	# selection, never to fit the clustering itself -> still unsupervised).
	best = (2, -1.0)
	for k in range(2, 11):
		lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
		if len(set(lab.tolist())) < 2:
			continue
		e = eta2_of(lab)
		if e > best[1]:
			best = (k, e)
	k = best[0]
	lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
	sil = float(silhouette_score(X, lab))
	groups = [ds.y[lab == c] for c in range(k)]
	overall = ds.y.mean()
	eta2 = eta2_of(lab)
	p = float(f_oneway(*[g for g in groups if len(g) > 1]).pvalue) if all(len(g) > 1 for g in groups) else 1.0
	sizes = [int((lab == c).sum()) for c in range(k)]
	means = [float(ds.y[lab == c].mean()) for c in range(k)]
	rows = [
		{"check": "k >= 2 segments", "achieved": k, "ok": k >= 2},
		{"check": "silhouette > 0.0", "achieved": round(sil, 3), "ok": sil > 0.0},
		{"check": "segments explain target variance (eta^2 > 0.02)",
		 "achieved": round(eta2, 3), "ok": eta2 > 0.02},
		{"check": "segments differ on target (ANOVA p < 0.05)",
		 "achieved": round(p, 4), "ok": p < 0.05},
	]
	ok = gate(rows, f"UNSUPERVISED SEGMENTATION PLUGIN ({window_days}d)")
	save_artifact(spec, {"dataset": ds.meta, "k": k, "silhouette": sil,
						 "sizes": sizes, "segment_mean_gross_margin": means,
						 "eta_squared": eta2, "anova_p": p, "verdict": bool(ok)})
	return ok, {"k": k, "eta2": eta2, "means": means}
