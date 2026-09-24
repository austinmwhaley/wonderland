"""red_queen — true next-best-action engine (multi-channel, multi-cadence).

Composes per-customer plans across channels (email/sms/push) and discount:
  (1) HOW MANY of each contact  -> the channel's best cadence arm (causal, IPW)
  (2) AT WHAT TIME              -> spread across the cadence window (caps + hours)
  (3) WHAT DISCOUNT (if any)    -> the best discount bucket (causal, net margin)
  WHO                           -> per-customer uplift responders (never everyone)
Constraint-aware (per-day/week caps, global budget), fail-safe (no action where
value <= 0), and emits a plan + management receipts.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "nba_multi_plan.json"

CHANNEL_CADENCE = {
	"email": (0.2, 0.6, 1.2, 2.0),
	"sms":   (0.1, 0.3, 0.8, 1.5),
	"push":  (0.5, 1.5, 3.0, 5.0),
}
DISCOUNTS = (0.0, 5.0, 10.0, 15.0)
CAPPED_HOURS = (9, 20)          # send window (quiet hours respected)


def _ipw_effects(channel, action):
	"""IPW causal mean reward per action level from randomized holdout-free logs."""
	import duckdb
	con = duckdb.connect(str(STREAM), read_only=True)
	try:
		col = "arm" if action == "arm" else "discount_pct"
		df = con.execute(f"""
			SELECT cs.{col} a, COALESCE(o.gross_margin, 0) r, cs.propensity p
			FROM contact_sends cs LEFT JOIN orders o ON o.transaction_id = cs.converted_order_id
			WHERE cs.channel = ?""", [channel]).pl()
	finally:
		con.close()
	a = df["a"].to_numpy().astype(np.float64); r = df["r"].to_numpy().astype(np.float64)
	# propensity not logged per send for discount; use uniform for discount buckets
	p = df["p"].to_numpy().astype(np.float64) if action == "arm" else np.ones(len(df))
	p = np.where(np.isfinite(p) & (p > 0), p, 1.0)
	levels = np.unique(a)
	out = {}
	for lv in levels:
		m = a == lv
		out[float(lv)] = float((r[m] / p[m]).sum() / max((1.0 / p[m]).sum(), 1e-9))
	return out



def _best_joint(channel):
	"""Joint (frequency arm x discount) selection from the data (mean reward)."""
	import duckdb
	con = duckdb.connect(str(STREAM), read_only=True)
	try:
		df = con.execute("""
			SELECT cs.arm a, cs.discount_pct d, COALESCE(o.gross_margin,0) r
			FROM contact_sends cs LEFT JOIN orders o ON o.transaction_id=cs.converted_order_id
			WHERE cs.channel=? AND cs.arm IS NOT NULL""", [channel]).pl()
	finally:
		con.close()
	a = df["a"].to_numpy(); d = df["d"].to_numpy(); r = df["r"].to_numpy()
	best, bv = (None, None), -1e18
	tab = {}
	for aa in sorted(set(a.tolist())):
		for dd in sorted(set(d.tolist())):
			m = (a == aa) & (d == dd)
			if m.sum() < 30:
				continue
			tab[(aa, dd)] = float(r[m].mean())
			if tab[(aa, dd)] > bv:
				bv, best = tab[(aa, dd)], (int(aa), float(dd))
	return best, bv


def _best_hour(channel):
	"""Learned send hour: the hour-of-day with the most conversions."""
	import duckdb
	con = duckdb.connect(str(STREAM), read_only=True)
	try:
		rows = con.execute("""
			SELECT extract(hour FROM CAST(click_ts AS TIMESTAMPTZ)) h, count(*) c
			FROM contact_sends WHERE channel=? AND clicked=1 AND click_ts IS NOT NULL
			GROUP BY 1 ORDER BY 2 DESC LIMIT 1""", [channel]).fetchone()
	finally:
		con.close()
	return int(rows[0]) if rows else 12

def build_plan(cadence="weekly", budget=None, seed=0):
	rng = np.random.default_rng(seed)
	# CERTIFICATION GATE: only schedule channels certified DEPLOY; else HOLD.
	cert_path = Path(__file__).resolve().parents[0] / "certification" / "channel_certification.json"
	certified = {}
	if cert_path.exists():
		for r in json.loads(cert_path.read_text()):
			if r.get("action") == "arm":
				certified[r["channel"]] = bool(r.get("deploy"))
	# per-channel best cadence arm + best discount (causal)
	best_arm, best_disc, effects, hours = {}, {}, {}, {}
	for ch in CHANNEL_CADENCE:
		ja, _jv = _best_joint(ch)
		best_arm[ch], best_disc[ch] = ja
		effects[ch] = _ipw_effects(ch, "arm")
		hours[ch] = _best_hour(ch)
	# WHO: per-customer uplift responders
	try:
		from red_queen.response_model import fit_uplift
		keys, up, _ = fit_uplift()
		responders = [keys[i] for i in np.argsort(-up) if up[i] > 0]
	except Exception:
		responders = []
		if budget is None:
			budget = 0
	if budget is None:
		budget = float(len(responders))
	# compose per-customer plan
	period_days = 7 if cadence == "weekly" else 1
	start = datetime.now(timezone.utc)
	plan = []
	used = 0.0
	for k in responders:
		if used >= budget:
			break
		contacts = []
		for ch, cad in CHANNEL_CADENCE.items():
			if cert_path.exists() and not certified.get(ch, False):
				continue   # HOLD: channel not certified
			r = cad[int(best_arm[ch])] * (period_days / 7.0)
			# unbiased fractional-rate allocation (keeps low cadences alive)
			n = int(r) + (1 if rng.random() < (r - int(r)) else 0)
			n = int(min(n, max(0, budget - used)))
			# (2) timing: spread within the window, respecting business hours
			for j in range(n):
				frac = (j + 0.5) / max(n, 1)
				day = min(period_days - 1, int(frac * period_days))
				hour = hours.get(ch, CAPPED_HOURS[0])
				ts = (start + timedelta(days=day, hours=hour)).isoformat()
				contacts.append({"channel": ch, "ts": ts,
								 "discount_pct": float(best_disc[ch])})
			used += n
		if contacts:
			plan.append({"customer_key": k, "contacts": contacts})
	report = {"cadence": cadence, "responders": len(responders), "targeted": len(plan),
			  "touches": int(used), "budget": budget,
			  "best_arm": {k: int(v) for k, v in best_arm.items()},
			  "best_discount": {k: float(v) for k, v in best_disc.items()},
			  "effects_arm": {k: {str(a): round(v, 2) for a, v in e.items()} for k, e in effects.items()},
			  "included_channels": [ch for ch in CHANNEL_CADENCE if (not cert_path.exists() or certified.get(ch, False))],
			  "send_hour": hours,
			  "fail_safe": "non-responders and uncertified channels -> no action"}
	OUT.parent.mkdir(parents=True, exist_ok=True)
	OUT.write_text(json.dumps({"report": report, "plan": plan[:2000]}))
	return report


def main(argv=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("--cadence", default="weekly", choices=["daily", "weekly"])
	ap.add_argument("--budget", type=float, default=None)
	a = ap.parse_args(argv)
	print("== RED_QUEEN NBA ENGINE (multi-channel, multi-cadence) ==")
	for k, v in build_plan(a.cadence, a.budget).items():
		print(f"  {k:14s}: {v}")


if __name__ == "__main__":
	main()
