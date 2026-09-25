"""Marketing INCREMENTALITY from persistent holdouts.

The persistent holdout is a RANDOMIZED control (5% of customer-periods get NO
marketing), so the difference in outcomes between treated and held-out periods is
CAUSAL. This gives:
  * POPULATION incrementality (E[margin | treated] - E[margin | holdout]);
  * PER-CUSTOMER incrementality for customers observed in both states
    (who responds to marketing) -> the per-customer value we want;
  * whether that per-customer response is PREDICTABLE from the frozen state
    (i.e., can we personalize whom to target).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
N_PERIODS = 12


def load():
    import duckdb
    import polars as pl

    con = duckdb.connect(str(STREAM), read_only=True)
    try:
        min_t, max_t = con.execute(
            "SELECT min(epoch(CAST(event_ts AS TIMESTAMPTZ))), "
            "max(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM customer_events"
        ).fetchone()
        pdays = (max_t - min_t) / N_PERIODS
        orders = con.execute(f"""
			SELECT customer_id k,
			       least({N_PERIODS - 1}, greatest(0,
			         floor((epoch(CAST(order_ts AS TIMESTAMPTZ)) - {min_t}) / {pdays})::int)) p,
			       SUM(gross_margin) gm
			FROM orders GROUP BY 1, 2""").pl()
        hold = con.execute("SELECT customer_id k, period p, holdout FROM email_holdout").pl()
    finally:
        con.close()
    grid = hold.join(orders, on=["k", "p"], how="left").with_columns(pl.col("gm").fill_null(0.0))
    return grid


def run(seed=0):

    gr = load()
    y = gr["gm"].to_numpy()
    h = gr["holdout"].to_numpy().astype(bool)
    k = gr["k"].to_numpy()
    treated, held = ~h, h
    pop = float(y[treated].mean() - y[held].mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(400):
        j = rng.integers(0, len(y), len(y))
        yy = y[j]
        hh = h[j]
        boot.append(float(yy[~hh].mean() - yy[hh].mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # per-customer incrementality (both states observed)
    incr = {}
    for t in np.unique(k):
        m = k == t
        tt = m & treated
        hh = m & held
        if tt.any() and hh.any():
            incr[t] = float(y[tt].mean() - y[hh].mean())
    # predictability from the frozen state
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    import duckdb

    con = duckdb.connect(str(CFM), read_only=True)
    emb = con.execute("""
		SELECT customer_key, embedding FROM (
			SELECT customer_key, anchor_epoch, embedding,
			       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
			FROM anchor_embeddings) WHERE rn=1""").pl()
    con.close()
    emap = {r["customer_key"]: r["embedding"] for r in emb.iter_rows(named=True)}
    ks = [t for t in incr if t in emap]
    X = np.stack([emap[t] for t in ks])
    yv = np.array([incr[t] for t in ks])
    sc = StandardScaler().fit(X)
    pred = Ridge(alpha=1.0).fit(sc.transform(X), yv).predict(sc.transform(X))
    rho = float(spearmanr(pred, yv).statistic) if len(ks) > 5 else float("nan")
    return {
        "customer_periods": len(y),
        "treated": int(treated.sum()),
        "heldout": int(held.sum()),
        "population_incrementality": round(pop, 2),
        "pop_ci95": [round(lo, 2), round(hi, 2)],
        "customers_with_both": len(incr),
        "per_customer_incr_mean": round(float(np.mean(list(incr.values()))), 2) if incr else None,
        "per_customer_incr_std": round(float(np.std(list(incr.values()))), 2) if incr else None,
        "predictable_from_state_rho": round(rho, 3) if np.isfinite(rho) else None,
    }


if __name__ == "__main__":
    print("== MARKETING INCREMENTALITY (persistent holdouts) ==")
    for kk, v in run().items():
        print(f"  {kk:30s}: {v}")
