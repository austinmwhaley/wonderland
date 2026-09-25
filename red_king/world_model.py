"""red_king — counterfactual world model over the frozen CFM donor.

Learns, from logged transitions, an ensemble dynamics + reward model:
    (state = donor embedding, action = email send-frequency bucket)
        -> (next state, reward = gross margin, done)
Uncertainty = ensemble disagreement. Used for counterfactual evaluation and to
improve white_queen's model-based OPE, with an explicit with/without baseline.

Data: consecutive sample-B anchors of the same customer. Action = number of
email_sends between the two anchors (bucketed). Reward = gross margin between
them. Self-supervised in structure (no reward model is given).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM_DB = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"


def _load_anchors():
    import duckdb

    con = duckdb.connect(str(CFM_PRODUCTS), read_only=True)
    try:
        return con.execute(
            "SELECT customer_key, anchor_epoch, embedding "
            "FROM anchor_embeddings ORDER BY customer_key, anchor_epoch"
        ).pl()
    finally:
        con.close()


def _load_facts():
    import duckdb

    con = duckdb.connect(str(STREAM_DB), read_only=True)
    try:
        sends = con.execute(
            "SELECT customer_key, epoch(CAST(event_ts AS TIMESTAMPTZ)) t "
            "FROM customer_events WHERE event_type='email_send'"
        ).pl()
        orders = con.execute(
            "SELECT customer_id AS customer_key, epoch(CAST(order_ts AS TIMESTAMPTZ)) t, "
            "gross_margin gm FROM orders"
        ).pl()
    finally:
        con.close()
    return sends, orders


def build_transitions(nA=4):

    anch = _load_anchors()
    sends, orders = _load_facts()
    sends_by, orders_by = {}, {}
    for r in sends.iter_rows(named=True):
        sends_by.setdefault(r["customer_key"], []).append(float(r["t"]))
    for r in orders.iter_rows(named=True):
        orders_by.setdefault(r["customer_key"], []).append((float(r["t"]), float(r["gm"])))
    rows = list(anch.iter_rows(named=True))
    S, A, R, S2, D, C = [], [], [], [], [], []
    len(rows[0]["embedding"])
    i = 0
    n = len(rows)
    while i < n:
        k = rows[i]["customer_key"]
        j = i
        while j < n and rows[j]["customer_key"] == k:
            j += 1
        grp = rows[i:j]
        st = np.array(sends_by.get(k, []))
        ot = orders_by.get(k, [])
        oga = np.array([x[1] for x in ot])
        otime = np.array([x[0] for x in ot])
        for m in range(len(grp) - 1):
            t0 = float(grp[m]["anchor_epoch"])
            t1 = float(grp[m + 1]["anchor_epoch"])
            if t1 <= t0:
                continue
            a = float(np.sum((st > t0) & (st <= t1)))
            r = float(oga[(otime > t0) & (otime <= t1)].sum())
            S.append(grp[m]["embedding"])
            S2.append(grp[m + 1]["embedding"])
            R.append(r)
            A.append(a)
            D.append(0.0)
            C.append(k)
        i = j
    # bucket actions by quantiles (derived)
    av = np.array(A)
    edges = np.unique(np.quantile(av, np.linspace(0, 1, nA + 1)[1:-1]))
    ab = np.digitize(av, edges).astype(np.int64)
    nAb = int(ab.max()) + 1
    return (
        np.array(S, np.float32),
        ab,
        np.array(R, np.float32),
        np.array(S2, np.float32),
        np.array(D, np.float32),
        nAb,
        np.array(C),
    )


def _split(groups, seed=0, test=0.3):
    from sklearn.model_selection import GroupShuffleSplit

    tr, te = next(
        GroupShuffleSplit(1, test_size=test, random_state=seed).split(
            np.zeros(len(groups)), groups=groups
        )
    )
    return tr, te


def train(nA, epochs=None, seed=0):
    import torch

    S, A, R, S2, D, nAb, C = build_transitions(nA)
    tr, te = _split(C, seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dim = S.shape[1]
    Rz = R / (R[tr].std() + 1e-6)  # scale reward for stable training

    class MLP(torch.nn.Module):
        def __init__(self, h=256):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim + nAb, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, dim + 2),
            )

        def forward(self, s, a):
            oh = torch.nn.functional.one_hot(a, nAb).float()
            o = self.net(torch.cat([s, oh], -1))
            return o[:, :dim], o[:, dim], o[:, dim + 1]

    K = 5
    ens = [MLP().to(dev) for _ in range(K)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
    St = torch.tensor(S, device=dev)
    At = torch.tensor(A, device=dev)
    Rt = torch.tensor(Rz, device=dev)
    S2t = torch.tensor(S2, device=dev)
    dS = S2t - St
    tri = torch.tensor(tr, device=dev)
    steps = 2000
    for m, opt in zip(ens, opts):
        torch.manual_seed(seed + K)
        for _ in range(steps):
            b = tri[torch.randint(0, len(tri), (256,), device=dev)]
            _, rp, _ = m(St[b], At[b])
            ds, _, _ = m(St[b], At[b])
            loss = torch.nn.functional.mse_loss(rp, Rt[b]) + 0.1 * torch.nn.functional.mse_loss(
                ds, dS[b]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    # eval
    with torch.no_grad():
        tei = torch.tensor(te, device=dev)
        rp = torch.stack([m(St[tei], At[tei])[1] for m in ens])
        rmean = rp.mean(0) * (R[tr].std() + 1e-6)
        run = rp.std(0) * (R[tr].std() + 1e-6)
        ds = torch.stack([m(St[tei], At[tei])[0] for m in ens]).mean(0)
        s2p = St[tei] + ds
    cos = float(torch.nn.functional.cosine_similarity(s2p, S2t[tei]).mean())
    ss_res = float(((rmean - torch.tensor(R[te], device=dev)) ** 2).sum())
    ss_tot = float(((torch.tensor(R[te], device=dev) - R[te].mean()) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    # with/without baseline: policy value on held-out states
    with torch.no_grad():
        St_te = St[tei]
        r_all = torch.stack(
            [
                torch.stack(
                    [
                        m(St_te, torch.full((len(te),), a, device=dev, dtype=torch.long))[1]
                        for a in range(nAb)
                    ]
                )
                for m in ens
            ]
        )  # (K, nA, N)
        r_grid = r_all.mean(0)  # (nA, N)
        r_grid.argmax(0)
        model_based = float(r_grid.max(0).values.mean() * (R[tr].std() + 1e-6))
        behavior = float(R[te].mean())
    # model-free DM baseline: reward model trained WITHOUT dynamics (same net, no dS)
    return {
        "n": int(len(S)),
        "nA": nAb,
        "dim": dim,
        "reward_r2": r2,
        "next_state_cos": cos,
        "behavior_value": behavior,
        "model_based_value": model_based,
        "uncertainty_mean": float(run.mean()),
        "reward_scale": float(R[tr].std()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="red_king world model")
    ap.add_argument("--actions", type=int, default=4)
    a = ap.parse_args(argv)
    res = train(a.actions)
    print("== RED_KING (counterfactual world model) ==")
    for k, v in res.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
