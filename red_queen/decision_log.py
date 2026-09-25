"""Decision log — the sequential, MULTI-CADENCE, MULTI-ACTION RL dataset.

One row per (customer, epoch). Epochs are consecutive anchor windows; each carries
the state (donor embedding), an ACTION SET (a frequency vector — how many touches
of each kind happened, i.e. MULTIPLE actions per cadence), the incremental-GP
reward, elapsed time, a cadence label, and done.

  state       : donor embedding at anchor t
  action      : [n_sends, n_arm0, n_arm1, n_arm2, n_arm3]   (frequency vector)
  reward      : incremental gross margin over (t, t']
  dt_days     : elapsed time -> cadence label (daily/weekly/monthly)
  next_state, done

Consumers: red_king (world model over action sets), white_queen (offline RL +
certification), red_queen (sequential controller).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "decision_log.npz"
ACTION_DIM = 5  # [n_sends, n_arm0, n_arm1, n_arm2, n_arm3]


def _cadence(dt_days: float) -> int:
    return 0 if dt_days <= 1.0 else (1 if dt_days <= 7.0 else 2)  # daily/weekly/monthly


def build():
    import duckdb

    pc = duckdb.connect(str(CFM), read_only=True)
    try:
        anch = pc.execute(
            "SELECT customer_key, anchor_epoch, embedding FROM anchor_embeddings "
            "ORDER BY customer_key, anchor_epoch"
        ).pl()
        splits = pc.execute("SELECT customer_key, split FROM encoder_samples").pl()
    finally:
        pc.close()
    B = set(splits.filter(splits["split"] == "B")["customer_key"].to_list())
    anch = anch.filter(anch["customer_key"].is_in(list(B)))
    sc = duckdb.connect(str(STREAM), read_only=True)
    try:
        snd = sc.execute(
            "SELECT customer_id k, epoch(CAST(send_ts AS TIMESTAMPTZ)) t, arm "
            "FROM email_sends WHERE arm IS NOT NULL"
        ).pl()
        inc = sc.execute("""
			SELECT s.customer_id k, epoch(CAST(o.order_ts AS TIMESTAMPTZ)) t, o.gross_margin gm
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1""").pl()
    finally:
        sc.close()
    sby = {}
    for r in snd.iter_rows(named=True):
        sby.setdefault(r["k"], []).append((float(r["t"]), int(r["arm"])))
    iby = {}
    for r in inc.iter_rows(named=True):
        iby.setdefault(r["k"], []).append((float(r["t"]), float(r["gm"])))
    rows = list(anch.iter_rows(named=True))
    i, n = 0, len(rows)
    S, S2, A, R, DT, D, CAD, TID = [], [], [], [], [], [], [], []
    tid = 0
    while i < n:
        k = rows[i]["customer_key"]
        j = i
        while j < n and rows[j]["customer_key"] == k:
            j += 1
        grp = rows[i:j]
        io = iby.get(k, [])
        it = np.array([x[0] for x in io])
        ig = np.array([x[1] for x in io])
        sm = sby.get(k, [])
        if len(grp) >= 2:
            for m in range(len(grp) - 1):
                t0 = float(grp[m]["anchor_epoch"])
                t1 = float(grp[m + 1]["anchor_epoch"])
                if t1 <= t0:
                    continue
                win = [b for (a, b) in sm if t0 < a <= t1]
                act = np.zeros(ACTION_DIM, np.float32)
                act[0] = len(win)
                for b in win:
                    if 0 <= b < 4:
                        act[1 + b] += 1
                S.append(grp[m]["embedding"])
                S2.append(grp[m + 1]["embedding"])
                A.append(act)
                R.append(float(ig[(it > t0) & (it <= t1)].sum()) if len(io) else 0.0)
                dt = (t1 - t0) / 86400.0
                DT.append(dt)
                CAD.append(_cadence(dt))
                D.append(1.0 if m == len(grp) - 2 else 0.0)
                TID.append(tid)
            tid += 1
        i = j
    out = {
        "state": np.array(S, np.float32),
        "next_state": np.array(S2, np.float32),
        "action": np.array(A, np.float32),
        "reward": np.array(R, np.float32),
        "dt_days": np.array(DT, np.float32),
        "cadence": np.array(CAD, np.int64),
        "done": np.array(D, np.float32),
        "traj": np.array(TID, np.int64),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    return {
        "steps": len(R),
        "trajectories": tid,
        "state_dim": len(S[0]),
        "action_dim": ACTION_DIM,
        "cadence_mix": {
            c: int((out["cadence"] == i).sum())
            for i, c in enumerate(("daily", "weekly", "monthly"))
        },
        "mean_actions_per_epoch": round(float(out["action"][:, 0].mean()), 2),
        "max_actions_per_epoch": int(out["action"][:, 0].max()),
        "reward_mean": round(float(out["reward"].mean()), 2),
        "out": str(OUT),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="multi-cadence multi-action decision log")
    ap.parse_args(argv)
    print("== DECISION LOG (multi-cadence, multi-action) ==")
    for k, v in build().items():
        print(f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()
