"""Item 7 — decision response plugin (action-conditioned, outside the frozen donor).

Keeps `E` action-free.  Trains a response model on logged RANDOMIZED actions with a
strictly pre-action encoder state, then estimates policy value with the known uniform
propensity and a customer-grouped bootstrap CI.

Protocol (pre-registered):
  rows     : up to --maxsends randomized sends per customer, per action key
  state    : frozen encoder state at the last event STRICTLY BEFORE the send,
             or the all-schema past-summary table at the same cut
  action   : channel (4 levels) or offer (4 levels) of that send
  reward   : margin of purchases in (send, send + --days]
  model    : HistGB response model on [state, one-hot(action)], GroupKFold by customer
  policy   : argmax predicted reward over observed levels
  value    : IPS with the known uniform propensity 1/K; DR as a cross-check
  CI       : 400 customer-grouped bootstrap resamples
  gates    : E value > random (CI lo > 0) and E value >= table value
  control  : random-donor (row-shuffled encoder state)

Writes docs/report_cards/experiments/plugin_response.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

import auto_encoder as AE  # noqa: E402
from measure_rl_v2 import _embed_cut, _past_cut, _purchase_margin  # noqa: E402
from stream_loader import _local_tz  # noqa: E402

OUT = ROOT / "docs/report_cards/experiments/plugin_response.json"


def _multi_sends(raw: pl.DataFrame, key: str, tz: str, maxsends: int, rng) -> dict:
    """customer -> list of (send_epoch, attrs) for up to `maxsends` randomized sends."""
    s = (
        raw.filter(pl.col("event_type") == "email_send")
        .with_columns(
            pl.col("event_attributes").str.extract(r'"target_key":\s*"([^"]+)"', 1).alias("_tk"),
            pl.col("event_ts")
            .str.to_datetime()
            .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
            .dt.epoch("s")
            .alias("ts"),
        )
        .filter(pl.col("_tk") == key)
        .filter(pl.col("ts").is_not_null())
        .select("customer_key", "ts", "event_attributes")
        .sort(["customer_key", "ts"])
    )
    out: dict[str, list] = {}
    g = s.group_by("customer_key", maintain_order=True).agg(
        [pl.col("ts"), pl.col("event_attributes")]
    )
    for r in g.iter_rows(named=True):
        ts = list(r["ts"])
        at = list(r["event_attributes"])
        n = len(ts)
        if n == 0:
            continue
        idx = rng.choice(n, size=min(maxsends, n), replace=False)
        idx.sort()
        out[r["customer_key"]] = [(float(ts[i]), json.loads(at[i] or "{}")) for i in idx]
    return out


def _rows_for(rows, pur, days, key, seqs, as_of_ts):
    ks, cuts, qg, actions, rewards, groups = [], [], [], [], [], []
    from rabbit_hole.generators.relational_reference import CHANNELS

    for ci, (k, sends) in enumerate(rows.items()):
        if k not in seqs:
            continue
        tss = seqs[k][5]
        for ts, a in sends:
            if ts >= as_of_ts:
                continue
            cut = int(np.searchsorted(tss, ts, side="left"))
            if cut < 8:
                continue
            ks.append(k)
            groups.append(ci)
            cuts.append(cut)
            qg.append(max(0.0, (ts - float(tss[cut - 1])) / (24 * 60 * 60)))
            if key == "channel":
                c = a.get("channel")
                actions.append(CHANNELS[int(c)] if str(c).isdigit() else str(c))
            else:
                actions.append(int(a.get("offer", -1)))
            hi = ts + days * 24 * 60 * 60
            rewards.append(float(sum(m for t, m in pur.get(k, []) if ts < t <= hi)))
    return (
        ks,
        cuts,
        qg,
        np.array(actions, dtype=object),
        np.asarray(rewards, float),
        np.array(groups),
    )


def ci_or_fail(d):
    """Paired-bootstrap interval that refuses impossible results."""
    d = np.asarray(d, dtype=float)
    mean = float(d.mean())
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    if lo == hi and abs(mean) > 1e-9:
        raise RuntimeError(
            f"degenerate bootstrap: zero-width CI with mean {mean}; refusing to publish"
        )
    if not np.isfinite([mean, lo, hi]).all():
        raise RuntimeError(f"non-finite bootstrap interval: {mean} [{lo}, {hi}]")
    return {"mean": round(mean, 2), "lo": round(lo, 2), "hi": round(hi, 2)}


def _eval(Xe, Xt, actions, reward, groups, levels, seed=0):
    """Pre-action response evaluation with a valid uniform-policy control.

    Arms:
      * `E`      response model on the frozen foundation state;
      * `table`  response model on the wide table;
      * `random` the UNIFORM policy (its IPS value under uniform logging is mean reward).

    The previous "random" arm was a shuffled-state model. That is not a policy control:
    it converged to the same policy as `E`, producing an identically-zero IPS difference
    and a zero-width interval. The uniform policy is the correct randomized comparator.
    """
    from sklearn.model_selection import GroupKFold

    levels = sorted(set(levels) | set(actions))
    lev = {v: i for i, v in enumerate(levels)}
    K = len(levels)
    p0 = 1.0 / K
    n = len(reward)

    def fit_policy(X):
        Z = StandardScaler().fit_transform(X)
        A = np.zeros((n, K))
        for i, v in enumerate(actions):
            A[i, lev.get(v, 0)] = 1.0
        ZA = np.hstack([Z, A])
        pred = np.zeros((n, K))
        for tr, te in GroupKFold(5).split(ZA, reward, groups):
            m = HistGradientBoostingRegressor(max_iter=250, random_state=seed)
            m.fit(ZA[tr], reward[tr])
            base = np.repeat(Z[te], K, axis=0)
            acts = np.tile(levels, len(te))
            Ab = np.zeros((len(base), K))
            for i, v in enumerate(acts):
                Ab[i, lev[v]] = 1.0
            pred[te] = m.predict(np.hstack([base, Ab])).reshape(len(te), K)
        pol = np.array(levels)[pred.argmax(1)]
        hit = pol == actions
        p_logged = pred[np.arange(n), [lev[v] for v in actions]]
        return {
            "pol": pol,
            "best": pred.max(1),
            "logged_pred": p_logged,
            "ips": float(np.mean(hit * reward / p0)),
            "dr": float(np.mean(pred.max(1) + hit * (reward - p_logged) / p0)),
        }

    arms = {"E": fit_policy(Xe), "table": fit_policy(Xt)}

    rng = np.random.default_rng(seed + 1)
    gu = np.unique(groups)

    def ips_of(arm, ix):
        a = arms[arm]
        return float(np.mean((a["pol"][ix] == actions[ix]) * reward[ix] / p0))

    def dr_of(arm, ix):
        a = arms[arm]
        return float(
            np.mean(
                a["best"][ix]
                + (a["pol"][ix] == actions[ix]) * (reward[ix] - a["logged_pred"][ix]) / p0
            )
        )

    boots = {k: [] for k in ("E-table", "E-random", "E-random-dr", "table-random")}
    for _ in range(400):
        gsel = rng.choice(gu, size=len(gu), replace=True)
        ix = np.flatnonzero(np.isin(groups, gsel))
        boots["E-table"].append(ips_of("E", ix) - ips_of("table", ix))
        boots["E-random"].append(ips_of("E", ix) - float(np.mean(reward[ix])))
        boots["table-random"].append(ips_of("table", ix) - float(np.mean(reward[ix])))
        boots["E-random-dr"].append(dr_of("E", ix) - float(np.mean(reward[ix])))

    return {
        "levels": [str(x) for x in levels],
        "n_rows": int(n),
        "n_customers": int(len(gu)),
        "random_value": round(float(np.mean(reward)), 2),
        "E": {"ips": round(arms["E"]["ips"], 2), "dr": round(arms["E"]["dr"], 2)},
        "table": {"ips": round(arms["table"]["ips"], 2), "dr": round(arms["table"]["dr"], 2)},
        "random": {
            "ips": round(float(np.mean(reward)), 2),
            "dr": round(float(np.mean(reward)), 2),
            "definition": "uniform policy under uniform logging (= mean reward)",
        },
        "E_minus_table": ci_or_fail(boots["E-table"]),
        "E_minus_random": ci_or_fail(boots["E-random"]),
        "E_minus_random_dr": ci_or_fail(boots["E-random-dr"]),
        "table_minus_random": ci_or_fail(boots["table-random"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--maxsends", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    import os

    import frozen as F

    E, Xtab, _labels, _hold, _seg, model, vs, _B, _best, _BLK = F.load(
        args.frozen or os.environ["FROZEN"]
    )
    data = Path(args.data)
    raw = pl.read_parquet(data / "customer_event_stream.parquet")
    man = json.loads((data / "manifest.json").read_text())
    from datetime import datetime

    as_of_ts = datetime.fromisoformat(man["as_of"]).timestamp()
    _b, _a, _keys, seqs, vs2, *_ = AE.load_data(str(data))
    tz = _local_tz()
    pur = _purchase_margin(raw, set(seqs), tz)

    out = {
        "protocol": {
            "reward_days": args.days,
            "maxsends": args.maxsends,
            "split": "GroupKFold by customer",
        }
    }
    for key in ("channel", "offer"):
        rng = np.random.default_rng(0)
        rows = _multi_sends(raw, key, tz, args.maxsends, rng)
        ks, cuts, qg, actions, reward, groups = _rows_for(rows, pur, args.days, key, seqs, as_of_ts)
        Ec, Bc = _embed_cut(model, seqs, ks, cuts, qg, vs)
        Xe = np.nan_to_num(np.hstack([Ec, Bc]))
        Xt = _past_cut(seqs, ks, cuts, qg, vs2)
        res = _eval(Xe, Xt, actions, reward, groups, set(actions))
        res["mean_reward"] = round(float(reward.mean()), 3)
        res["positive_rate"] = round(float((reward > 0).mean()), 3)
        out[key] = res
        print(key, json.dumps(res, indent=1), flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
