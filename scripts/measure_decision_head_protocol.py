#!/usr/bin/env python3
"""DEC-589 decision/value head protocol v2 — DIAGNOSTIC, not a gate.

Same cfm_v2k E, same splits, same targets, same frozen bars; only the HEAD changes:

  * OPE (B4.2): the frozen head is a ridge on [E, onehot(action)] — additive, so
    `V(E,a) = w_E·E + w_a` has the same argmax for every customer (the measured constant
    policy collapse). The T-learner fits a separate ridge per action level
    (`V(E,a) = E·w_a`), so the action effect can vary by customer.
  * Dose (B4.1): the gate's per-level ridge heads stay; a multinomial logistic on the
    optimal offer level is the diagnostic for how much E carries directly.
  * Donor (B6.2): ridge vs Tweedie/log-link and log1p+ridge on y_value_90 / y_orders_90.

The frozen receipts and gates are untouched; this receipt is diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import (  # noqa: E402
    MIN_ARM_DAYS,
    _load_all,
    _ridge_head,
    arm_cells,
    identified_dose_response,
)
from measure_decision_law import _onehot  # noqa: E402
from measure_judge_law import _table_matrix  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "decision_head_protocol.json"
DEMO = Path("/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2")


def _round(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _pop_arrays(emb, bkeys, truth, stream, demo: Path):
    """The OPE population: last logged offer, y_margin_90, table, truth."""
    con = duckdb.connect()
    con.register("s", stream)
    try:
        last = con.execute(
            """
            select customer_key,
                   arg_max(try_cast(json_extract_string(event_attributes,'$.offer') as integer),
                           event_ts) as off
            from s where event_type = 'email_send'
            group by 1
            """
        ).pl()
    finally:
        con.close()
    acts = {
        r["customer_key"]: int(r["off"]) for r in last.iter_rows(named=True) if r["off"] is not None
    }
    labels = pl.read_parquet(demo / "path_a" / "customer_labels.parquet")
    lab = {r["customer_key"]: float(r["y_margin_90"]) for r in labels.iter_rows(named=True)}
    table, _ = _table_matrix(demo, bkeys)
    pop = [k for k in bkeys if k in lab and k in acts and k in emb and k in table and k in truth]
    actions = np.asarray([acts[k] for k in pop], dtype=int)
    reward = np.asarray([lab[k] for k in pop], dtype=float)
    want = np.asarray([int(truth[k]["offer_opt"]) for k in pop], dtype=int)
    X_E = np.vstack([emb[k] for k in pop])
    X_T = np.vstack([table[k] for k in pop])
    return pop, actions, reward, want, X_E, X_T


def _policy_additive(X, actions, reward, levels, tr, te):
    """The frozen head: ridge on [E, onehot(action)] — additive, constant argmax."""
    m = _ridge_head()
    m.fit(np.hstack([X[tr], _onehot(actions[tr], levels)]), reward[tr])
    base = np.repeat(X[te], len(levels), axis=0)
    pred = m.predict(
        np.hstack([base, _onehot(np.tile(np.asarray(levels), len(te)), levels)])
    ).reshape(len(te), len(levels))
    return pred.argmax(1)


def _policy_tlearner(X, actions, reward, levels, tr, te):
    """Per-action ridge: V(E,a) = E·w_a — the action effect may vary by customer."""
    preds = np.zeros((len(te), len(levels)))
    for j, lvl in enumerate(levels):
        m = actions[tr] == lvl
        if int(m.sum()) >= 20:
            h = _ridge_head()
            h.fit(X[tr][m], reward[tr][m])
            preds[:, j] = h.predict(X[te])
        else:
            preds[:, j] = float(reward[tr][m].mean()) if bool(m.any()) else float(reward[tr].mean())
    return preds.argmax(1)


def _ips(pol, actions, reward, te, p0):
    return float(np.mean((pol == actions[te]) * reward[te] / p0))


def _boot_diff(pol_E, pol_T, actions, reward, te, p0, n=400, seed=1):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        bi = rng.integers(0, len(te), len(te))
        d.append(
            float(
                np.mean((pol_E[bi] == actions[te][bi]) * reward[te][bi] / p0)
                - np.mean((pol_T[bi] == actions[te][bi]) * reward[te][bi] / p0)
            )
        )
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _response_block(stream, keys, levels):
    """Per-customer response curve: mean next-purchase outcome per offer level in the
    randomized history, shrunk to the population mean. A served statistic (no table)."""
    from looking_glass.stream_adapter import to_ts_expr

    df = stream.with_columns(to_ts_expr(pl.col("event_ts")).alias("ts"))
    send = (
        df.filter(pl.col("event_type") == "email_send")
        .with_columns(
            pl.col("event_attributes").str.json_path_match("$.offer").cast(pl.Int64).alias("offer")
        )
        .select(["customer_key", "ts", "offer"])
        .sort(["customer_key", "ts"])
    )
    send = send.with_columns(pl.col("ts").shift(-1).over("customer_key").alias("nsend"))
    purch = (
        df.filter(pl.col("event_type") == "purchase")
        .select(["customer_key", "ts", "value"])
        .sort(["customer_key", "ts"])
    )
    j = send.join_asof(
        purch.rename({"ts": "pts", "value": "pvalue"}),
        left_on="ts",
        right_on="pts",
        by="customer_key",
        strategy="forward",
    )
    j = j.with_columns(
        pl.when(
            pl.col("pts").is_not_null()
            & pl.col("nsend").is_not_null()
            & (pl.col("pts") <= pl.col("nsend"))
        )
        .then(pl.col("pvalue"))
        .otherwise(0.0)
        .alias("outcome")
    )
    per = j.group_by(["customer_key", "offer"]).agg(
        pl.col("outcome").mean().alias("resp"), pl.len().alias("n")
    )
    pop_mean = (
        per.group_by("offer")
        .agg((pl.col("resp") * pl.col("n")).sum() / pl.col("n").sum())
        .rename({"resp": "pop"})
    )
    k = float(per["n"].median() or 1.0)
    per = per.join(pop_mean, on="offer").with_columns(
        ((pl.col("resp") * pl.col("n") + k * pl.col("pop")) / (pl.col("n") + k)).alias("shrunk")
    )
    piv = per.pivot(index="customer_key", columns="offer", values="shrunk")
    pm = dict(zip(pop_mean["offer"].to_list(), pop_mean["pop"].to_list()))
    key_df = pl.DataFrame({"customer_key": keys})
    m = key_df.join(piv, on="customer_key", how="left")
    for lvl in levels:
        col = str(lvl)
        if col not in m.columns:
            m = m.with_columns(pl.lit(pm.get(lvl, 0.0)).alias(col))
        else:
            m = m.with_columns(pl.col(col).fill_null(pm.get(lvl, 0.0)))
    return m.select([str(lvl) for lvl in levels]).to_numpy().astype(float)


def main(argv=None) -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k")
    ap.add_argument("--demo", default=str(DEMO))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    demo = Path(a.demo)
    products = Path(a.products)
    _reg, truth, emb, _raw, stream, bkeys, akeys, _as_of = _load_all(
        str(products / "cfm_products.duckdb"), demo
    )
    A = set(akeys)
    B = set(bkeys)

    # ---- OPE: additive (frozen) vs T-learner ---------------------------------
    pop, actions, reward, want, X_E, X_T = _pop_arrays(emb, bkeys, truth, stream, demo)
    levels = sorted(set(int(w) for w in want) | set(int(x) for x in actions))
    K = len(levels)
    p0 = 1.0 / K
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(reward))
    ntr = int(0.7 * len(reward))
    tr, te = idx[:ntr], idx[ntr:]
    res = {}
    for arm, X in (("E", X_E), ("table", X_T)):
        for head, fn in (("additive", _policy_additive), ("t_learner", _policy_tlearner)):
            pol = fn(X, actions, reward, levels, tr, te)
            res[f"{arm}_{head}"] = {
                "ips": round(_ips(pol, actions, reward, te, p0), 2),
                "n_unique_actions": int(len(np.unique(pol))),
                "action_share": {str(lvl): round(float(np.mean(pol == lvl)), 4) for lvl in levels},
            }
    resp = _response_block(stream, pop, levels)
    E_resp = np.hstack([X_E, resp])
    for head, fn in (("t_learner", _policy_tlearner),):
        pol = fn(E_resp, actions, reward, levels, tr, te)
        res[f"E_resp_{head}"] = {
            "ips": round(_ips(pol, actions, reward, te, p0), 2),
            "n_unique_actions": int(len(np.unique(pol))),
            "action_share": {str(lvl): round(float(np.mean(pol == lvl)), 4) for lvl in levels},
        }
    res["response_block_argmax_share"] = round(float(np.mean(resp.argmax(1) == want)), 4)
    lo_e, hi_e = _boot_diff(
        _policy_tlearner(X_E, actions, reward, levels, tr, te),
        _policy_tlearner(X_T, actions, reward, levels, tr, te),
        actions,
        reward,
        te,
        p0,
    )
    res["t_learner_diff_E_minus_table"] = {
        "point": round(res["E_t_learner"]["ips"] - res["table_t_learner"]["ips"], 2),
        "ci95": [round(lo_e, 2), round(hi_e, 2)],
    }
    lo_r, hi_r = _boot_diff(
        _policy_tlearner(E_resp, actions, reward, levels, tr, te),
        _policy_tlearner(X_T, actions, reward, levels, tr, te),
        actions,
        reward,
        te,
        p0,
    )
    res["resp_diff_E_minus_table"] = {
        "point": round(res["E_resp_t_learner"]["ips"] - res["table_t_learner"]["ips"], 2),
        "ci95": [round(lo_r, 2), round(hi_r, 2)],
    }

    # ---- Dose: per-level ridge (gate) vs multinomial logistic (diagnostic) ----
    dr = identified_dose_response(arm_cells(stream, truth), MIN_ARM_DAYS)
    dlevels = sorted({lvl for k in dr for lvl in dr[k]})
    ak = [k for k in A if k in dr and k in emb]
    bk = [k for k in B if k in dr and k in emb]
    if ak and bk:
        Ea = np.vstack([emb[k] for k in ak])
        Eb = np.vstack([emb[k] for k in bk])
        ya = np.asarray([int(truth[k]["offer_opt"]) for k in ak])
        yb = np.asarray([int(truth[k]["offer_opt"]) for k in bk])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(Ea, ya)
        got = m.predict(Eb)
        res["dose_logistic"] = {
            "levels": dlevels,
            "n_A": len(ak),
            "n_B": len(bk),
            "share_E_logistic": round(float(accuracy_score(yb, got)), 6),
            "chance": round(1.0 / max(len(dlevels), 1), 4),
        }
    else:
        res["dose_logistic"] = {"complete": False, "n_A": len(ak), "n_B": len(bk)}

    # ---- Donor: ridge vs Tweedie/log1p on the 90-day targets ----------------
    labels = pl.read_parquet(demo / "path_a" / "customer_labels.parquet")
    lab = {r["customer_key"]: r for r in labels.iter_rows(named=True)}
    keys = [k for k in emb if k in lab]
    X = np.vstack([emb[k] for k in keys])
    a_mask = np.asarray([k in A for k in keys])
    b_mask = ~a_mask
    donor = {}
    for tgt in ("y_value_90", "y_orders_90"):
        y = np.asarray([float(lab[k][tgt]) for k in keys], dtype=float)
        out = {}
        m = _ridge_head()
        m.fit(X[a_mask], y[a_mask])
        out["ridge_r2"] = _round(r2_score(y[b_mask], m.predict(X[b_mask])))
        for name, head, transform, inv in (
            ("tweedie", TweedieRegressor(power=1.5, link="log", max_iter=3000), None, None),
            ("log1p_ridge", _ridge_head(), np.log1p, np.expm1),
        ):
            yy = transform(y) if transform is not None else y
            h = head
            h.fit(X[a_mask], yy[a_mask])
            p = h.predict(X[b_mask])
            if inv is not None:
                p = inv(p)
            out[f"{name}_r2"] = _round(r2_score(y[b_mask], p))
        out["n_B"] = int(b_mask.sum())
        donor[tgt] = out
    res["donor_heads"] = donor

    body = {
        "kind": "decision_head_protocol_diagnostic",
        "protocol": (
            "DEC-589 v2: same cfm_v2k E, splits, targets and frozen bars; OPE additive vs "
            "per-action T-learner; dose multinomial logistic diagnostic; donor ridge vs "
            "Tweedie/log1p. DIAGNOSTIC — the frozen receipts/gates are untouched."
        ),
        "products_identity": products_identity(products),
        "results": res,
        "complete": bool(res),
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out), body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
    )
    print(f"decision head protocol -> {a.out} ({body['seconds']}s)")
    print(json.dumps(res, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
