"""Per-channel frequency optimization + discount optimization (white_queen).

For each marketing channel (email/sms/push) and for discount, build logs from
rabbit_hole's unified `contact_sends`, then run white_queen's offline-RL pipeline
to LEARN and CERTIFY a policy with a DEPLOY/HOLD verdict.

  state  = frozen donor embedding (customer)
  action = cadence arm (frequency) OR discount bucket
  reward = incremental gross margin (order linked to the contact, else 0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"


def _state():
    import duckdb

    con = duckdb.connect(str(CFM), read_only=True)
    try:
        df = con.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn=1""").pl()
    finally:
        con.close()
    return df


def _logs(channel, action):
    import duckdb

    con = duckdb.connect(str(STREAM), read_only=True)
    try:
        if action == "arm":
            df = con.execute(
                """
				SELECT cs.customer_id k, cs.arm a, COALESCE(o.gross_margin,0) r
				FROM contact_sends cs LEFT JOIN orders o ON o.transaction_id = cs.converted_order_id
				WHERE cs.channel = ? AND cs.arm IS NOT NULL LIMIT 300000""",
                [channel],
            ).pl()
        else:  # discount
            df = con.execute(
                """
				SELECT cs.customer_id k, cs.discount_pct a, COALESCE(o.gross_margin,0) r
				FROM contact_sends cs LEFT JOIN orders o ON o.transaction_id = cs.converted_order_id
				WHERE cs.channel = ? AND cs.discount_pct IS NOT NULL LIMIT 300000""",
                [channel],
            ).pl()
    finally:
        con.close()
    emb = _state()
    return df.join(emb, left_on="k", right_on="customer_key", how="inner")


class _Greedy:
    """Reward-greedy candidate policy (per-arm linear reward model)."""

    def __init__(self, W, nA):
        self.W = W
        self.nA = nA

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        s = o @ self.W.T
        z = (s - s.max(-1, keepdims=True)) / max(temperature, 1e-9)
        e = np.exp(z)
        return (e / e.sum(-1, keepdims=True)).astype(np.float32)

    def act(self, state, eval=True):
        o = np.atleast_2d(np.asarray(state, dtype=np.float64))
        return int(np.argmax(o @ self.W.T))


def optimize(channel, action="arm", max_rows=800, seed=0):
    from sklearn.linear_model import Ridge
    from white_queen.tribunal.ope.api import evaluate

    df = _logs(channel, action)
    a = df["a"].to_numpy().astype(np.float64)
    if action == "discount":
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 5)[1:-1]))
        act = np.digitize(a, edges).astype(np.int64)
    else:
        act = a.astype(np.int64)
    nA = int(act.max()) + 1
    obs = np.stack(df["embedding"].to_list()).astype(np.float32)
    rew = df["r"].to_numpy().astype(np.float32)
    n = len(obs)
    idx = np.arange(n)
    if n > max_rows:
        idx = np.random.default_rng(seed).choice(n, max_rows, replace=False)
    # speed: PCA the 512-d donor state to 64-d on the SAME (sampled) rows, fit the
    # candidate on that reduced space so it matches what the OPE panel queries.
    from sklearn.decomposition import PCA

    k = min(64, obs.shape[1], len(idx) - 1)
    Z = (
        PCA(n_components=k, random_state=seed)
        .fit_transform(obs[idx].astype(np.float64))
        .astype(np.float32)
    )
    A, R = act[idx], rew[idx].astype(np.float64)
    W = np.zeros((nA, k))
    for arm in range(nA):
        m = A == arm
        W[arm] = Ridge(alpha=1.0).fit(Z[m], R[m]).coef_ if m.sum() > 30 else np.zeros(k)
    _FAST = {
        "steps_max": 400,
        "eval_every": 100,
        "patience": 3,
        "batch": 128,
        "hidden": 32,
        "allow_under_budget": True,
    }
    rep = evaluate(
        {"obs": Z, "act": A, "rew": rew[idx]},
        _Greedy(W, nA),
        nA=nA,
        fast=True,
        ensemble_K=1,
        fqe_cfg=_FAST,
        candidate_name=f"{channel}_{action}",
    )
    return {
        "channel": channel,
        "action": action,
        "rows": int(len(idx)),
        "nA": nA,
        "deploy": bool(rep.get("deploy")),
        "behavior": round(rep.get("behavior_mean", 0), 2),
        "bar": round(rep.get("bar", 0), 2),
        "provenance": rep.get("provenance", {}).get("propensity"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_rows", type=int, default=15000)
    a = ap.parse_args(argv)
    print("== CHANNEL / DISCOUNT OPTIMIZATION (white_queen) ==")
    res = [optimize(chan, "arm", a.max_rows) for chan in ("email", "sms", "push")]
    res.append(optimize("email", "discount", a.max_rows))
    import json

    out = WORK / "red_queen" / "certification" / "channel_certification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for r in res:
        print("  ", r)
    print("  wrote", out)


if __name__ == "__main__":
    main()
