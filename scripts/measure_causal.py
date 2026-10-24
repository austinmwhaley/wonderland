"""A6 known-effect recovery: reproduce the frozen IPS protocol on current data.

Protocol v1 (what the card currently uses, made reproducible):

  state      : frozen encoder state at as-of (E readout + identity/mass blocks)
               or the wide table
  action     : the channel / offer in the customer's last logged `email_send`
  reward     : y_margin_90
  split      : 70/30 by customer, seed 0
  learner    : XGBoost (200 trees, depth 5, lr 0.06), [state, one-hot(action)]
  policy     : argmax predicted reward over the observed arm levels
  value      : IPS with uniform propensity 1/K, paired bootstrap 400, seed 1
  gate       : first arm (channel) diff > 0 and CI lower bound > 0

This is a *reproduction*, not a gate change.  It exists so the card can resolve
A6 from a repo command instead of a `/tmp` script.  The last-send action and the
as-of state overlap in time; that limitation is reported, not hidden.

Usage:
  FROZEN=/tmp/opencode/frozen_6k python3 scripts/measure_causal.py \
      --data /tmp/opencode/t0_loop/ref_v3b_6k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

import xgboost as xgb  # noqa: E402

OUT = ROOT / "docs/report_cards/experiments/encoder_causal_ci.json"
ARMS = ("channel", "offer")


def _load_frozen(path, data):
    """Load the frozen cache; fall back to training if FROZEN is unset."""
    if path:
        import frozen

        return frozen.load(path)
    import auto_encoder as AE

    return AE.train_encoder(data, seed=0, epochs=12, d=128, layers=2)  # principle-ok: frozen-config fallback


def _last_send(data: Path) -> dict:
    stream = pl.read_parquet(data / "customer_event_stream.parquet")
    last = (
        stream.filter(pl.col("event_type") == "email_send")
        .sort("event_ts")
        .group_by("customer_key", maintain_order=True)
        .agg(pl.col("event_attributes").last().alias("attrs"))
    )
    return {r["customer_key"]: json.loads(r["attrs"]) for r in last.iter_rows(named=True)}


def _onehot(vals, levels):
    lev = {v: i for i, v in enumerate(levels)}
    M = np.zeros((len(vals), len(levels)))
    for i, v in enumerate(vals):
        M[i, lev.get(v, 0)] = 1.0
    return M


def _z_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _arm(name, state_E, state_T, actions, truth, reward, seed=0):
    levels = sorted(set(truth) | set(actions))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(reward))
    ntr = int(0.7 * len(reward))
    tr, te = idx[:ntr], idx[ntr:]
    p0 = 1.0 / len(levels)

    def fit_predict(X, label):
        mu, sd = _z_fit(X[tr])
        Z = (X - mu) / sd
        Za = np.hstack([Z[tr], _onehot([actions[i] for i in tr], levels)])
        m = xgb.XGBRegressor(
            device="cpu", tree_method="hist", n_estimators=200, max_depth=5, learning_rate=0.06
        )
        m.fit(Za, reward[tr])
        # every test row x every action, one vectorized predict
        base = np.repeat(Z[te], len(levels), axis=0)
        acts = np.tile(levels, len(te))
        pred = m.predict(np.hstack([base, _onehot(acts, levels)])).reshape(len(te), len(levels))
        return np.array(levels)[pred.argmax(1)], pred, label

    pol_E, _, _ = fit_predict(state_E, "E")
    pol_T, _, _ = fit_predict(state_T, "table")

    def ips(pol):
        return float(np.mean((pol == actions[te]) * reward[te] / p0))

    rng_b = np.random.default_rng(seed + 1)
    d = []
    for _ in range(400):
        bi = rng_b.integers(0, len(te), len(te))
        d.append(
            float(
                np.mean((pol_E[bi] == actions[te][bi]) * reward[te][bi] / p0)
                - np.mean((pol_T[bi] == actions[te][bi]) * reward[te][bi] / p0)
            )
        )
    d = np.array(d)
    return {
        "arm": name,
        "levels": [str(x) for x in levels],
        "E_ips": round(ips(pol_E), 2),
        "T_ips": round(ips(pol_T), 2),
        "random": round(float(np.mean(reward[te])), 2),
        "diff": round(float(d.mean()), 2),
        "lo": round(float(np.quantile(d, 0.025)), 2),
        "hi": round(float(np.quantile(d, 0.975)), 2),
        "accE": round(float(np.mean(pol_E == truth[te])), 3),
        "accT": round(float(np.mean(pol_T == truth[te])), 3),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "protocol": "v1 (last-send; as-of state; uniform IPS)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = Path(args.data)

    E, Xtab, labels, hold, seg, model, vs, _B, best, BLK = _load_frozen(
        args.frozen or __import__("os").environ.get("FROZEN"), str(data)
    )
    E = np.nan_to_num(E)
    Xtab = np.nan_to_num(Xtab)
    BLK = np.nan_to_num(BLK)
    Xe = np.hstack([E, BLK])
    truth = {c["customer"]: c for c in json.load(open(data / "reference_truth.json"))["customers"]}
    acts = _last_send(data)
    reward = np.asarray(labels["y_margin_90"], float)

    from rabbit_hole.generators.relational_reference import CHANNELS

    out = []
    for arm in ARMS:
        if arm == "channel":
            raw = [str(acts.get(k, {}).get("channel", "")) for k in hold]
            actions = np.array(
                [CHANNELS[int(x)] if x.isdigit() and int(x) < len(CHANNELS) else x for x in raw],
                dtype=object,
            )
            wanted = np.array([str(truth[k]["channel_pref"]) for k in hold], dtype=object)
        else:
            actions = np.array([int(acts.get(k, {}).get("offer", -1)) for k in hold])
            wanted = np.array([int(truth[k]["offer_opt"]) for k in hold])
        out.append(_arm(arm, Xe, Xtab, actions, wanted, reward, seed=0))
        print(out[-1], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
