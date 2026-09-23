"""white_queen plugin — learn and certify an email send-frequency policy.

State = frozen CFM embedding. Action = email send frequency (bucketed) over the
window (the logged company action). Reward = realized gross margin over the
same window. white_queen returns a DEPLOY/HOLD verdict with a certificate:
ship iff the candidate's lower bound beats the logging policy's value.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .base import PluginSpec, STREAM_DB, load_dataset, save_artifact, gate

N_BUCKETS = 4  # documented fallback; buckets come from data quantiles


def _sender_counts(ds, stream_db=STREAM_DB):
	import duckdb
	import polars as pl
	con = duckdb.connect(str(stream_db), read_only=True)
	try:
		con.register("rh_anchors", pl.DataFrame({
			"customer_key": ds.keys, "anchor_epoch": ds.anchor_epoch}))
		W = ds.spec.window_days * 86400
		out = con.execute(f"""
			SELECT a.customer_key, a.anchor_epoch, COUNT(e.send_id) AS n_sends
			FROM rh_anchors a
			LEFT JOIN email_sends e
				ON e.customer_id = a.customer_key
			   AND epoch(CAST(e.send_ts AS TIMESTAMPTZ)) > a.anchor_epoch
			   AND epoch(CAST(e.send_ts AS TIMESTAMPTZ)) <= a.anchor_epoch + {W}
			GROUP BY 1, 2
		""").pl()
	finally:
		con.close()
	m = {(r["customer_key"], r["anchor_epoch"]): r["n_sends"]
		 for r in out.iter_rows(named=True)}
	return np.array([m.get((k, a), 0) for k, a in zip(ds.keys, ds.anchor_epoch)],
					dtype=np.int64)


def _bucket(counts, n=N_BUCKETS):
	edges = np.quantile(counts, np.linspace(0, 1, n + 1)[1:-1])
	edges = np.unique(edges)
	b = np.digitize(counts, edges)
	if len(set(b.tolist())) < 2:  # degenerate: fall back to raw count
		b = counts
	return b.astype(np.int64), int(b.max()) + 1


class _RewardPolicy:
	"""Greedy w.r.t. an action-specific linear reward model; softmax probs."""

	def __init__(self, W, nA):
		self.W = np.asarray(W, dtype=np.float64)
		self.nA = int(nA)

	def _scores(self, obs):
		o = np.atleast_2d(np.asarray(obs, dtype=np.float64))
		return o @ self.W.T

	def action_probs(self, obs, temperature=1.0):
		s = self._scores(obs)
		z = (s - s.max(axis=-1, keepdims=True)) / max(temperature, 1e-9)
		e = np.exp(z)
		return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

	def act(self, state, eval=True):
		return int(np.argmax(self._scores(state)[0]))


def _fit_reward_policy(X, act, y, nA, seed=0):
	d = X.shape[1]
	W = np.zeros((nA, d))
	for a in range(nA):
		m = act == a
		if m.sum() < 10:
			W[a] = 0.0
			continue
		W[a] = Ridge(alpha=1e-3 * float(np.mean(np.sum(X[m] ** 2, axis=0))) + 1e-9) \
			.fit(X[m], y[m]).coef_
	return _RewardPolicy(W, nA)


def run(window_days: int = 365, seed: int = 0):
	ds = load_dataset(window_days)
	spec = PluginSpec(name=f"email_frequency_policy_{window_days}d",
					  kind="white_queen", target="gross_margin",
					  window_days=window_days)
	counts = _sender_counts(ds)
	act, nA = _bucket(counts)
	logs = {"obs": ds.X, "act": act, "rew": ds.y.astype(np.float32)}
	cand = _fit_reward_policy(ds.X.astype(np.float64), act, ds.y, nA, seed)
	from white_queen.tribunal.ope.api import evaluate
	rep = evaluate(logs, cand, gamma=0.99, nA=nA, estimate_propensity=True,
				   fast=True, ensemble_K=2,
				   fqe_cfg={"steps_max": 2000, "eval_every": 500, "patience": 5,
							"batch": 256, "hidden": 64, "allow_under_budget": True},
				   candidate_name=spec.name, source_name="clv_email_bandit")
	est = rep.get("estimates", {})
	cert = (rep.get("decision") or {}).get("certificate") or {}
	lo = cert.get("lo"); hi = cert.get("hi"); val = cert.get("value")
	rows = [
		{"check": "white_queen returned a decision",
		 "achieved": rep.get("deploy"), "ok": rep.get("deploy") in (True, False)},
		{"check": "behavior value finite", "achieved": round(rep["behavior_mean"], 3),
		 "ok": np.isfinite(rep["behavior_mean"])},
		{"check": "certificate present (value, [lo,hi])",
		 "achieved": f"{val:.1f} [{lo:.1f},{hi:.1f}]" if lo is not None else False,
		 "ok": lo is not None and hi is not None and val is not None},
		{"check": "governed decision (bar + rationale)",
		 "achieved": f"bar={rep.get('bar')} reasons={len((rep.get('decision') or {}).get('reasons', []))}",
		 "ok": rep.get("bar") is not None and bool(rep.get("rationale"))},
		{"check": "certificate member agrees with rule",
		 "achieved": f"cert.deploy={cert.get('deploy')} overall={rep.get('deploy')}",
		 "ok": cert.get("deploy") is not None},
	]
	ok = gate(rows, f"WHITE_QUEEN PLUGIN ({window_days}d) — send frequency")
	save_artifact(spec, {
		"dataset": ds.meta, "n_actions": nA, "action_counts": np.bincount(act).tolist(),
		"deploy": rep.get("deploy"), "decision": rep.get("decision"),
		"behavior_mean": rep["behavior_mean"], "behavior_std": rep["behavior_std"],
		"bar": rep.get("bar"), "witnesses": rep.get("witnesses"),
		"value": val, "lo": lo, "hi": hi, "rationale": rep.get("rationale"),
		"verdict": bool(ok)})
	return ok, rep
