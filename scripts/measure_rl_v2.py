"""RL plugin v2: a clean contextual-bandit test on the randomized action days.

The reference generator randomizes exactly one action key per day; the other four
keys sit at that customer's optimum.  Selecting the last `target_key == "channel"`
(or `"offer"`) send therefore gives a uniformly randomized action with a known
propensity, and we can cut the encoder state strictly BEFORE the send.

Protocol (pre-registered, written before the run):
  data     : synthetic relational reference; all customers with a matching send
  action   : channel (4 levels) or offer (4 levels) of that send
  state    : frozen encoder state at the last event strictly before the send,
             or the all-schema past-summary table at the same cut
  reward   : margin of purchases in (send, send + 3d]  (primary; 1d sensitivity)
  split    : 70/30 by customer, seed 0
  learner  : HistGradientBoosting response model on [state, one-hot(action)]
  policy   : argmax predicted reward over the observed levels
  value    : IPS with the known uniform propensity 1/K; paired bootstrap 95% CI
  gates    : E value > random (CI lower > 0) and E value >= table value
  truth    : generator's known per-customer optimum, reported as accuracy

Writes docs/report_cards/experiments/plugin_rl.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
import torch  # noqa: E402

OUT = ROOT / "docs/report_cards/experiments/plugin_rl.json"


def _embed_cut(model, seqs, ks, cuts, qgaps, vs, batch=32):
    """Frozen encoder state at an arbitrary prefix cut (causal: only events < the send)."""
    out_e, out_b = [], []
    with torch.no_grad():
        for i in range(0, len(ks), batch):
            bk = ks[i : i + batch]
            bc = cuts[i : i + batch]
            bq = qgaps[i : i + batch]
            t, _, qg, lens, _ = AE.collate(seqs, bk, bc, bq, vs, False, None, None)
            Hs, E, _, *blocks = model(
                t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
            )
            out_e.append(E.cpu().numpy())
            out_b.append(
                np.hstack([b.cpu().numpy() for b in blocks] + [model.past(E).cpu().numpy()])
            )
    return np.concatenate(out_e, 0), np.concatenate(out_b, 0)


def _past_cut(seqs, ks, cuts, qgaps, vs):
    t, P, _, _, _ = AE.collate(seqs, ks, cuts, qgaps, vs, True, None, None)
    return np.nan_to_num(P.cpu().numpy())


def _purchase_margin(raw: pl.DataFrame, keys: set[str], tz: str) -> dict:
    """customer -> sorted (epoch_seconds, margin) purchase events."""
    pur = (
        raw.filter(pl.col("event_type") == "purchase")
        .with_columns(
            pl.col("event_ts")
            .str.to_datetime()
            .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
            .dt.epoch("s")
            .alias("ts"),
            pl.col("event_attributes")
            .str.extract(r'"margin":\s*([-0-9.eE+]+)', 1)
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("margin"),
        )
        .filter(pl.col("ts").is_not_null())
        .select("customer_key", "ts", "margin")
        .sort(["customer_key", "ts"])
    )
    out: dict[str, list[tuple[float, float]]] = {}
    g = pur.group_by("customer_key", maintain_order=True).agg([pl.col("ts"), pl.col("margin")])
    for r in g.iter_rows(named=True):
        if r["customer_key"] in keys:
            out[r["customer_key"]] = list(zip(r["ts"], r["margin"]))
    return out


def _send_rows(raw: pl.DataFrame, key: str, tz: str) -> dict:
    """customer -> (send_epoch, attrs) for the last randomized send of `key`."""
    s = (
        raw.filter(pl.col("event_type") == "email_send")
        .with_columns(
            pl.col("event_attributes")
            .str.extract(r'"target_key":\s*"([^"]+)"', 1)
            .alias("_target_key")
        )
        .filter(pl.col("_target_key") == key)
        .with_columns(
            pl.col("event_ts")
            .str.to_datetime()
            .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
            .dt.epoch("s")
            .alias("ts")
        )
        .filter(pl.col("ts").is_not_null())
        .sort(["customer_key", "ts"])
        .group_by("customer_key", maintain_order=True)
        .agg(pl.col("ts").last(), pl.col("event_attributes").last().alias("attrs"))
    )
    return {
        r["customer_key"]: (float(r["ts"]), json.loads(r["attrs"] or "{}"))
        for r in s.iter_rows(named=True)
    }


def _reward(pur: dict, k: str, ts: float, days: float) -> float:
    hi = ts + days * 24 * 60 * 60  # principle-ok: unit conversion, not a cadence
    return float(sum(m for t, m in pur.get(k, []) if ts < t <= hi))


def _fit_eval(Xe, Xt, actions, truth, reward, seed=0):
    from scipy.stats import binomtest

    levels = sorted(set(actions) | set(truth))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(reward))
    ntr = int(0.7 * len(reward))
    tr, te = idx[:ntr], idx[ntr:]
    p0 = 1.0 / len(levels)

    def oh(vals):
        m = {v: i for i, v in enumerate(levels)}
        M = np.zeros((len(vals), len(levels)))
        for i, v in enumerate(vals):
            M[i, m.get(v, 0)] = 1.0
        return M

    def run(X, name):
        sc = StandardScaler().fit(X[tr])
        Z = sc.transform(X)
        # vectorized over test rows x levels
        base = np.repeat(Z[te], len(levels), axis=0)
        acts = np.tile(levels, len(te))
        m = HistGradientBoostingRegressor(max_iter=250, random_state=0)
        m.fit(np.hstack([Z[tr], oh([actions[i] for i in tr])]), reward[tr])
        pred = m.predict(np.hstack([base, oh(acts)])).reshape(len(te), len(levels))
        pol = np.array(levels)[pred.argmax(1)]
        logged = actions[te]
        p_logged = pred[np.arange(len(te)), [levels.index(v) for v in logged]]
        ips = float(np.mean((pol == logged) * reward[te] / p0))
        dr = float(np.mean(pred.max(1) + (pol == logged) * (reward[te] - p_logged) / p0))
        hits = int(np.sum(pol == truth[te]))
        return {
            "ips": round(ips, 2),
            "dr": round(dr, 2),
            "acc_truth": round(hits / max(len(te), 1), 3),
            "acc_p": float(binomtest(hits, len(te), p0, alternative="greater").pvalue),
            "pol": pol,
            "best": pred.max(1),
            "logged_pred": p_logged,
            name: pred,
        }

    e = run(Xe, "E")
    t = run(Xt, "table")
    logged = actions[te]
    r = reward[te]
    rng_b = np.random.default_rng(seed + 1)
    bi = rng_b.integers(0, len(te), (400, len(te)))

    def ip(pol, ix):
        return np.mean((pol[ix] == logged[ix]) * r[ix] / p0)

    def drv(o, ix):
        return np.mean(
            o["best"][ix] + (o["pol"][ix] == logged[ix]) * (r[ix] - o["logged_pred"][ix]) / p0
        )

    e_ips = np.array([ip(e["pol"], ix) for ix in bi])
    t_ips = np.array([ip(t["pol"], ix) for ix in bi])
    e_dr = np.array([drv(e, ix) for ix in bi])
    t_dr = np.array([drv(t, ix) for ix in bi])
    rand = np.array([float(np.mean(r[ix])) for ix in bi])

    def ci(d):
        return {
            "mean": round(float(np.mean(d)), 2),
            "lo": round(float(np.quantile(d, 0.025)), 2),
            "hi": round(float(np.quantile(d, 0.975)), 2),
        }

    return {
        "levels": [str(x) for x in levels],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "random_value": round(float(np.mean(r)), 2),
        "E": {k: e[k] for k in ("ips", "dr", "acc_truth", "acc_p")},
        "table": {k: t[k] for k in ("ips", "dr", "acc_truth", "acc_p")},
        "E_minus_table": {"ips": ci(e_ips - t_ips), "dr": ci(e_dr - t_dr)},
        "E_minus_random": {"ips": ci(e_ips - rand), "dr": ci(e_dr - rand)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--days", type=float, default=3.0)
    args = ap.parse_args()
    data = Path(args.data)
    import os

    if args.frozen or os.environ.get("FROZEN"):
        import frozen

        _E, _X, _lab, _hold, _seg, model, vs, _B, _best, _BLK = frozen.load(
            args.frozen or os.environ["FROZEN"]
        )
    else:
        raise SystemExit("Set FROZEN to the frozen cache (or pass --frozen).")

    raw = pl.read_parquet(data / "customer_event_stream.parquet")
    man = json.loads((data / "manifest.json").read_text())
    as_of = datetime.fromisoformat(man["as_of"])
    as_of_ts = as_of.timestamp()
    from stream_loader import _local_tz

    tz = _local_tz()
    _B, _A, _keys, seqs, vs2, *_ = AE.load_data(str(data))
    truth = {c["customer"]: c for c in json.load(open(data / "reference_truth.json"))["customers"]}
    pur = _purchase_margin(raw, set(seqs), tz)

    out = {"protocol": {"reward_days": args.days, "split": "70/30 by customer", "seed": 0}}
    for key in ("channel", "offer"):
        sends = _send_rows(raw, key, tz)
        ks, cuts, qg, actions, rewards = [], [], [], [], []
        for k, (ts, a) in sends.items():
            if k not in seqs or ts >= as_of_ts:
                continue
            tss = seqs[k][5]
            cut = int(np.searchsorted(tss, ts, side="left"))
            if cut < 8:
                continue
            ks.append(k)
            cuts.append(cut)
            qg.append(
                max(0.0, (ts - float(tss[cut - 1])) / (24 * 60 * 60))
            )  # principle-ok: seconds per day
            if key == "channel":
                from rabbit_hole.generators.relational_reference import CHANNELS

                c = a.get("channel")
                actions.append(CHANNELS[int(c)] if str(c).isdigit() else str(c))
            else:
                actions.append(int(a.get("offer", -1)))
            rewards.append(_reward(pur, k, ts, args.days))
        actions = np.array(actions, dtype=object)
        truth_v = np.array(
            [
                str(truth[k]["channel_pref"]) if key == "channel" else int(truth[k]["offer_opt"])
                for k in ks
            ],
            dtype=object,
        )
        Ecut, Bcut = _embed_cut(model, seqs, ks, cuts, qg, vs)
        Xe = np.nan_to_num(np.hstack([Ecut, Bcut]))
        Xt = _past_cut(seqs, ks, cuts, qg, vs2)
        res = _fit_eval(Xe, Xt, actions, truth_v, np.asarray(rewards, float), seed=0)
        res["n_customers"] = len(ks)
        res["mean_reward"] = round(float(np.mean(rewards)), 3)
        res["positive_rate"] = round(float(np.mean(np.asarray(rewards) > 0)), 3)
        out[key] = res
        print(
            key,
            json.dumps({k: v for k, v in res.items() if k not in ("E", "table")}, indent=1),
            flush=True,
        )
        print("  E", res["E"], "table", res["table"], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
