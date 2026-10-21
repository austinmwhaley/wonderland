"""Certify Layer A (PATH A + PATH B) as ready for encoder development.

This is stricter than the PATH A gate. It checks:
  1. PATH A gate passed (`path_a_validation.json`).
  2. PATH B is the canonical 4-column stream, sorted, complete, non-degenerate.
  3. Action design is usable: all 5 keys, >=2 levels each, >=50 sends/level, and a
     measurable conversion gradient by declared level.
  4. The pipeline ingests PATH B: `_read_stream`, `generic_measure` (no target_key),
     response block present, all action keys discovered.
  5. Sequence HEADROOM: sequence features from <= as_of (last/previous event types,
     last category/touch, gaps, 7/30d counts) must add over PATH A + GBM. If they
     don't, the table is not demonstrably lossy and the encoder has no target.
  6. Truth receipt has the known-effect fields and latent diagnostics.

Writes <dir>/layer_a_certificate.json. Exit 1 if not certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looking_glass.cfm_config import CFMConfig  # noqa: E402
from looking_glass.cfm_data import _customer_keys, _read_stream  # noqa: E402
from looking_glass.layer_b import generic_measure  # noqa: E402
from looking_glass.use_case_gate import discover_action_keys  # noqa: E402


def _attrs(s):
    return json.loads(s)


def _cv(X, y, kind, seed=0):
    y = np.asarray(y, float)
    oof = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        if kind == "clf":
            m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
    return oof


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repro-dir", default=None, dest="repro_dir")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text())
    pa = json.loads((d / "path_a_validation.json").read_text())
    A = pl.read_parquet(d / "path_a/path_a_wide_table.parquet")
    B = pl.read_parquet(d / "path_b/path_b_customer_event_stream.parquet")
    truth = json.loads((d / "reference_truth.json").read_text())
    as_of = datetime.fromisoformat(man["as_of"])
    failures: list[str] = []
    cert: dict = {"dir": str(d), "as_of": man["as_of"], "join_key": "customer_key"}

    # 1. PATH A gate
    cert["path_a"] = {"ok": pa["ok"], "failures": pa["failures"], "baseline": pa["baseline"]}
    if not pa["ok"]:
        failures.append("PATH A gate failed")

    # 2. PATH B integrity
    pb = {}
    pb["columns"] = B.columns
    pb["rows"] = B.height
    pb["customers"] = B["customer_key"].n_unique()
    if B.columns != ["customer_key", "event_ts", "event_type", "event_attributes"]:
        failures.append(f"PATH B columns are not canonical: {B.columns}")
    if set(B["customer_key"]) != set(A["customer_key"]):
        failures.append("PATH B customer_key set differs from PATH A")
    Bs = B.with_columns(pl.col("event_ts").str.to_datetime().alias("ts_dt"))
    pb["sorted"] = bool(Bs.equals(Bs.sort(["customer_key", "ts_dt"])))
    if not pb["sorted"]:
        failures.append("PATH B is not sorted by customer_key, event_ts")
    lens = B.group_by("customer_key").len()["len"]
    pb["seq_len"] = {
        "min": int(lens.min()),
        "median": float(lens.median()),
        "max": int(lens.max()),
        "mean": round(float(lens.mean()), 1),
    }
    if int(lens.min()) < 20:
        failures.append(f"a customer has <20 events: min={int(lens.min())}")
    types = B.group_by("event_type").len().sort("len", descending=True)
    pb["event_types"] = {r["event_type"]: int(r["len"]) for r in types.to_dicts()}
    for et in ("email_send", "purchase", "customer_signup"):
        if not pb["event_types"].get(et):
            failures.append(f"missing event type {et}")
    dup = Bs.group_by(["customer_key", "ts_dt"]).len().filter(pl.col("len") > 1).height
    pb["duplicate_timestamps"] = int(dup)
    if dup / B.height > 0.01:
        failures.append(f"too many duplicate timestamps: {dup}")
    pre = Bs.filter(pl.col("ts_dt") <= as_of)
    fut = Bs.filter(pl.col("ts_dt") > as_of)
    pb["prefix_rows"] = pre.height
    pb["future_rows"] = fut.height
    if pre.height == 0 or fut.height == 0:
        failures.append("as_of split leaves no prefix or no future events")
    cert["path_b"] = pb

    # 3. action design: coverage + conversion gradient
    sends = B.filter(pl.col("event_type") == "email_send")
    srows = sends.sample(n=min(40000, sends.height), seed=a.seed).to_dicts()
    purch = B.filter(pl.col("event_type") == "purchase").with_columns(
        pl.col("event_ts").str.to_datetime().dt.date().alias("day")
    )
    pset = {(r["customer_key"], r["day"]) for r in purch.select(["customer_key", "day"]).to_dicts()}
    conv: dict[str, dict[str, list]] = {}
    for r in srows:
        at = _attrs(r["event_attributes"])
        key = at.get("target_key")
        if not key or key not in ("freq", "hour", "dow", "channel", "offer"):
            continue
        lvl = str(at.get(key))
        day = datetime.fromisoformat(r["event_ts"]).date()
        conv.setdefault(key, {}).setdefault(lvl, [0, 0])
        conv[key][lvl][0] += 1
        conv[key][lvl][1] += int((r["customer_key"], day) in pset)
    action = {}
    for key, levels in conv.items():
        rates = {lvl: round(v[1] / v[0], 4) for lvl, v in levels.items() if v[0] >= 50}
        n = {lvl: v[0] for lvl, v in levels.items()}
        spread = (max(rates.values()) - min(rates.values())) if len(rates) >= 2 else 0.0
        action[key] = {"n": n, "conversion": rates, "spread": round(spread, 4)}
        if len(rates) < 2:
            failures.append(f"action {key}: <2 levels with >=50 sends")
        elif spread < 0.005:
            failures.append(f"action {key}: conversion spread {spread:.4f} < 0.005")
    cert["actions"] = action

    # 4. pipeline ingestion
    cfg = CFMConfig()
    cfg.db = str(d / "path_b/path_b_customer_event_stream.parquet")
    cfg.as_of = man["as_of"]
    cfg.sample_customers = 6000
    cfg.effect_actions = ("freq", "hour", "dow", "channel", "offer")
    cfg.action_fields = ("freq", "hour", "dow", "channel", "offer", "discount_pct")
    cfg.action_period_seconds = 86400.0
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    m, names = generic_measure(cfg, keys[:200])
    ingest = {
        "read_rows": df.height,
        "customers": len(keys),
        "measure_cols": int(m.shape[1]),
        "response_cols": len([n for n in names if n.startswith("resp_")]),
        "target_key_in_measure": any("target_key" in n for n in names),
        "actions": discover_action_keys(cfg),
    }
    cert["ingestion"] = ingest
    if ingest["target_key_in_measure"]:
        failures.append("target_key leaked into the measure")
    if ingest["response_cols"] == 0:
        failures.append("no response block in the measure")
    if set(ingest["actions"]) != {"freq", "hour", "dow", "channel", "offer"}:
        failures.append(f"action discovery mismatch: {ingest['actions']}")

    # 5. sequence headroom over PATH A
    p = pre.sort(["customer_key", "ts_dt"]).with_columns(
        pl.col("event_type").shift(1).over("customer_key").alias("prev_et"),
        pl.col("event_type").shift(2).over("customer_key").alias("prev2_et"),
        (pl.col("ts_dt") - pl.col("ts_dt").shift(1).over("customer_key"))
        .dt.total_days()
        .alias("gap_days"),
    )
    last = p.group_by("customer_key", maintain_order=True).agg(
        pl.col("event_type").last().alias("last_et"),
        pl.col("prev_et").last().alias("prev_et"),
        pl.col("prev2_et").last().alias("prev2_et"),
        pl.col("event_attributes").last().alias("last_attrs"),
        pl.col("ts_dt").last().alias("last_ts"),
        pl.col("gap_days").last().alias("last_gap_days"),
        pl.col("gap_days").mean().alias("mean_gap_days"),
        pl.len().alias("seq_len"),
    )
    cnt7 = (
        pre.filter(pl.col("ts_dt") > as_of - __import__("datetime").timedelta(days=7))
        .group_by("customer_key")
        .len()
        .rename({"len": "n_events_7d"})
    )
    cnt30 = (
        pre.filter(pl.col("ts_dt") > as_of - __import__("datetime").timedelta(days=30))
        .group_by("customer_key")
        .len()
        .rename({"len": "n_events_30d"})
    )
    last = last.join(cnt7, on="customer_key", how="left").join(cnt30, on="customer_key", how="left")
    last = last.with_columns(
        (pl.lit(as_of) - pl.col("last_ts")).dt.total_days().alias("days_since_last"),
        pl.col("n_events_7d").fill_null(0),
        pl.col("n_events_30d").fill_null(0),
    )
    cats = []
    for r in last.to_dicts():
        at = _attrs(r["last_attrs"])
        cats.append((at.get("category_code"), at.get("touch") or at.get("channel")))
    last = last.with_columns(
        pl.Series("last_category", [c[0] for c in cats]),
        pl.Series("last_touch", [c[1] for c in cats]),
    ).drop("last_attrs", "last_ts")
    seq_cols = ["last_et", "prev_et", "prev2_et", "last_category", "last_touch"]
    seq_num = [
        "last_gap_days",
        "mean_gap_days",
        "seq_len",
        "n_events_7d",
        "n_events_30d",
        "days_since_last",
    ]
    S = A.select(
        ["customer_key"]
        + [
            c
            for c in A.columns
            if not c.startswith("y_") and c not in ("customer_key", "signup_ts")
        ]
    )
    S = S.join(last, on="customer_key", how="left")
    base_cats = [
        c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in S.columns
    ]
    Xb = (
        S.select([c for c in S.columns if c not in seq_cols + seq_num + ["customer_key"]])
        .to_dummies(columns=base_cats, drop_first=True)
        .to_numpy()
        .astype(float)
    )
    Xs = (
        S.select([c for c in S.columns if c != "customer_key"])
        .to_dummies(columns=base_cats + seq_cols, drop_first=True)
        .to_numpy()
        .astype(float)
    )
    yc = np.asarray(A["y_churn_90"], float)
    yv = np.asarray(A["y_value_90"], float)
    pb_c = _cv(Xb, yc, "clf", a.seed)
    ps_c = _cv(Xs, yc, "clf", a.seed)
    pb_r = _cv(Xb, yv, "reg", a.seed)
    ps_r = _cv(Xs, yv, "reg", a.seed)
    head = {
        "churn90_base_auc": round(roc_auc_score(yc, pb_c), 4),
        "churn90_seq_auc": round(roc_auc_score(yc, ps_c), 4),
        "churn90_delta_auc": round(roc_auc_score(yc, ps_c) - roc_auc_score(yc, pb_c), 4),
        "value90_base_r2": round(r2_score(yv, pb_r), 4),
        "value90_seq_r2": round(r2_score(yv, ps_r), 4),
        "value90_delta_r2": round(r2_score(yv, ps_r) - r2_score(yv, pb_r), 4),
    }
    cert["sequence_headroom"] = head
    if head["churn90_delta_auc"] < 0.01 and head["value90_delta_r2"] < 0.03:
        failures.append(
            "no sequence headroom: PATH A already captures the order information "
            f"(dAUC={head['churn90_delta_auc']}, dR2={head['value90_delta_r2']})"
        )

    # 6. truth / known-effect fields
    tr = truth.get("customers") or []
    need = ("x0", "h0", "d0", "offer_opt")
    missing = [k for c in tr for k in need if k not in c]
    cert["truth"] = {
        "customers": len(tr),
        "known_effect_fields_ok": not missing,
        "regime_diagnostics": all("regime_runs" in c for c in tr),
    }
    if len(tr) != A.height or missing:
        failures.append(f"truth receipt invalid: n={len(tr)} missing={missing[:5]}")

    # hashes
    cert["hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            d / "path_a/path_a_wide_table.parquet",
            d / "path_b/path_b_customer_event_stream.parquet",
            d / "reference_truth.json",
            d / "manifest.json",
        )
    }
    if a.repro_dir:
        rd = Path(a.repro_dir)
        rels = [
            "path_a/path_a_wide_table.parquet",
            "path_b/path_b_customer_event_stream.parquet",
            "reference_truth.json",
            "manifest.json",
        ]
        same = {
            rel: hashlib.sha256((d / rel).read_bytes()).hexdigest()
            == hashlib.sha256((rd / rel).read_bytes()).hexdigest()
            for rel in rels
        }
        cert["reproducible"] = all(same.values())
        cert["repro_files"] = same
        if not all(same.values()):
            failures.append(f"reproducibility mismatch: {same}")

    cert["failures"] = failures
    cert["certified"] = not failures
    (d / "layer_a_certificate.json").write_text(json.dumps(cert, indent=2) + "\n")
    print(
        json.dumps(
            {
                "certified": cert["certified"],
                "failures": failures,
                "headroom": head,
                "actions": action,
            },
            indent=1,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
