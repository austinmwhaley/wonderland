"""Sequential email dataset for offline RL.

Turns the anchor stream into per-customer TRAJECTORIES of decision steps:
    step = (state s, action a, reward r, next-state s', done, dt_days)
  * state   = frozen donor embedding at anchor t
  * action  = email send-frequency bucket over (t, t']
  * reward  = INCREMENTAL gross margin (email-caused orders) over (t, t']
  * next    = donor embedding at the next anchor
  * dt_days = elapsed time (for time-discounting)

Reward standard (AGENTS.md): long-term INCREMENTAL gross margin, discounted.
Saves trajectories + discounted returns to red_king/data/seq_email.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parents[0] / "data" / "seq_email.npz"
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM_DB = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
GAMMA_DAY = 0.999  # per-day discount (long-term)


def _load():
    import duckdb

    pc = duckdb.connect(str(CFM_PRODUCTS), read_only=True)
    try:
        anch = pc.execute(
            "SELECT customer_key, anchor_epoch, embedding "
            "FROM anchor_embeddings ORDER BY customer_key, anchor_epoch"
        ).pl()
    finally:
        pc.close()
    sc = duckdb.connect(str(STREAM_DB), read_only=True)
    try:
        sends = sc.execute(
            "SELECT customer_key, epoch(CAST(event_ts AS TIMESTAMPTZ)) t "
            "FROM customer_events WHERE event_type='email_send'"
        ).pl()
        inc = sc.execute("""
			SELECT s.customer_id AS customer_key,
			       epoch(CAST(o.order_ts AS TIMESTAMPTZ)) t, o.gross_margin gm
			FROM email_sends s JOIN orders o
			  ON o.customer_id = s.customer_id AND o.session_id = s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked = 1
		""").pl()
    finally:
        sc.close()
    return anch, sends, inc


def build(nA=4):
    anch, sends, inc = _load()
    sby, iby = {}, {}
    for r in sends.iter_rows(named=True):
        sby.setdefault(r["customer_key"], []).append(float(r["t"]))
    for r in inc.iter_rows(named=True):
        iby.setdefault(r["customer_key"], []).append((float(r["t"]), float(r["gm"])))
    rows = list(anch.iter_rows(named=True))
    steps = []  # flattened steps
    i, n, tid = 0, len(rows), 0
    while i < n:
        k = rows[i]["customer_key"]
        j = i
        while j < n and rows[j]["customer_key"] == k:
            j += 1
        grp = rows[i:j]
        st = np.array(sby.get(k, []))
        io = iby.get(k, [])
        it = np.array([x[0] for x in io])
        ig = np.array([x[1] for x in io])
        if len(grp) >= 2:
            for m in range(len(grp) - 1):
                t0 = float(grp[m]["anchor_epoch"])
                t1 = float(grp[m + 1]["anchor_epoch"])
                if t1 <= t0:
                    continue
                a = float(np.sum((st > t0) & (st <= t1)))
                r = float(ig[(it > t0) & (it <= t1)].sum())
                done = 1.0 if m == len(grp) - 2 else 0.0
                steps.append(
                    (
                        grp[m]["embedding"],
                        a,
                        r,
                        grp[m + 1]["embedding"],
                        done,
                        (t1 - t0) / 86400.0,
                        tid,
                    )
                )
            tid += 1
        i = j
    S = np.array([s[0] for s in steps], np.float32)
    Araw = np.array([s[1] for s in steps], np.float32)
    R = np.array([s[2] for s in steps], np.float32)
    S2 = np.array([s[3] for s in steps], np.float32)
    D = np.array([s[4] for s in steps], np.float32)
    DT = np.array([s[5] for s in steps], np.float32)
    TID = np.array([s[6] for s in steps], np.int64)
    edges = np.unique(np.quantile(Araw, np.linspace(0, 1, nA + 1)[1:-1]))
    A = np.digitize(Araw, edges).astype(np.int64)
    # time-discounted return per step (backward within each trajectory)
    G = np.zeros(len(R), np.float32)
    for t in np.unique(TID):
        idx = np.where(TID == t)[0]
        acc = 0.0
        for m in range(len(idx) - 1, -1, -1):
            k = idx[m]
            acc = R[k] + (GAMMA_DAY ** DT[k]) * acc
            G[k] = acc
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, S=S, A=A, R=R, S2=S2, D=D, DT=DT, TID=TID, G=G)
    return {
        "n_steps": int(len(S)),
        "n_traj": int(tid),
        "dim": int(S.shape[1]),
        "steps_per_traj": round(len(S) / max(tid, 1), 2),
        "reward_mean": float(R.mean()),
        "reward_nonzero": float((R > 0).mean()),
        "return_mean": float(G.mean()),
        "out": str(OUT),
    }


def main():
    r = build()
    print("== SEQUENTIAL EMAIL DATASET (discounted incremental margin) ==")
    for k, v in r.items():
        print(f"  {k:16s}: {v}")
    # quick state-signal check: does the donor predict the discounted return?
    import numpy as np

    z = np.load(OUT)
    S, G = z["S"], z["G"]
    tr = np.arange(len(S)) % 3 != 0
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(S[tr])
    m = Ridge(alpha=1.0).fit(sc.transform(S[tr]), G[tr])
    te = ~tr
    from scipy.stats import spearmanr

    print(
        "  donor->discounted-return spearman:",
        round(float(spearmanr(m.predict(sc.transform(S[te])), G[te]).statistic), 3),
    )


if __name__ == "__main__":
    main()
