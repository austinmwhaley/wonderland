#!/usr/bin/env python3
"""DEV-RIG speedometer + controls (DEC-576) — a development instrument, NOT card evidence.

Frozen protocol (DEC-576, 2026-11-03; the operator's pivot: make ANY foundation model on a
known-truth lab stream before THE one on the real stream). For one lab use case this script
measures, on the SAME held-out B-future rows:

  * the encoder's next-type NLL (one-vs-rest shipping `_ridge_head` per class, fit on the A
    customers only, softmax with a 1e-15 p-floor),
  * the 1-gram baseline NLL (Laplace +1 MLE over the full history, model-free),
  * the closed-form oracle ceiling (oracle.json, nats),
  * the speedometer g = (nll_enc - nll_oracle) / (nll_base - nll_oracle)  (g=0 ceiling, g=1
    baseline; frozen fallback: if nll_base <= nll_oracle on the finite sample, the
    closed-form baseline_ce replaces the denominator and both facts are recorded),
  * the paired bootstrap CI of (nll_enc - nll_base) over B rows (SEED/N_BOOT from the battery),
  * the run's own eff_rank (portfolio receipt referenced by the products' train.log),
  * the NEGATIVE control (snr=0 fixture through the same pipeline: must read NO_SIGNAL = does
    not beat the 1-gram) and the ADVERSARIAL control (echo-next leak: the head WILL beat the
    1-gram, but the structure claim reads NO_SIGNAL because the types carry no transition —
    both numbers are recorded; a naive "beats baseline" verdict on that fixture is the false
    PASS the control exists to catch).

The A-side fixture defensibility line comes from `looking_glass.lab_gate.evaluate()`.

Discipline (binding): receipts land under docs/report_cards/experiments/lab/ — no B-row source
in b_spec.py/judge_spec.py names them, and a toy-green can only SUGGEST the next real-data run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import looking_glass.lab_gate as LG  # noqa: E402
from looking_glass.stream_adapter import to_ts_expr  # noqa: E402
from measure_battery_law import N_BOOT, SEED, _ridge_head  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from safe_run import ensure_boundary  # noqa: E402

LAB_RECEIPTS = ROOT / "docs" / "report_cards" / "experiments" / "lab"
A_FRAC = 0.7  # frozen head-fit fraction (mirrors the train_encoder internal split)
P_FLOOR = 1e-15  # frozen softmax p-floor (same as the U next-type probe)
BAR_G = 0.5  # frozen "mastered" speedometer bar (DEC-576)


def _round(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _load_served(products: Path) -> dict:
    import duckdb

    db = products / "cfm_products.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    emb = {
        k: np.asarray(v, float)
        for k, v in con.execute("select customer_key, embedding from state_embeddings").fetchall()
    }
    con.close()
    return emb


def _eff_rank(products: Path) -> tuple[float | None, str | None]:
    """The run's own portfolio receipt (single source of the geometry bar).

    Resolution: (1) the canonical layout's `<products>/train.log` `receipt ->` line; (2) the
    portfolio receipt named by the products' own tag — the trainer writes it under
    looking_glass/artifacts/cfm/portfolio/ and the filename carries the tag, so no launcher
    log is needed (the lab and O3 launches tee their stdout elsewhere)."""
    log = products / "train.log"
    if log.exists():
        path = None
        for line in log.read_text().splitlines():
            if "receipt ->" in line:
                path = line.split("receipt ->")[-1].strip()
        if path:
            p = Path(path)
            if p.exists():
                geo = json.loads(p.read_text()).get("geometry") or {}
                return _round(geo.get("eff_rank")), str(p)
    regs = sorted(products.glob("registry_*.json"), key=lambda q: q.stat().st_mtime)
    if not regs:
        return None, "no registry_*.json in the products dir — tag unresolvable"
    tag = regs[-1].name.removeprefix("registry_").removesuffix(".json")
    cands = sorted(
        (ROOT / "looking_glass" / "artifacts" / "cfm" / "portfolio").glob(
            f"portfolio_{tag}_*.json"
        ),
        key=lambda q: q.stat().st_mtime,
    )
    if not cands:
        return None, f"no portfolio receipt for tag {tag}"
    geo = json.loads(cands[-1].read_text()).get("geometry") or {}
    return _round(geo.get("eff_rank")), str(cands[-1])


def _split_customers(keys: list[str]) -> tuple[list[str], list[str]]:
    """Frozen A/B customer partition: one seeded permutation, 70/30. Same seed + same keys
    (all fixtures of a use case share them) -> the identical partition on every fixture."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(keys))
    cut = int(A_FRAC * len(keys))
    return [keys[i] for i in sorted(order[:cut])], [keys[i] for i in sorted(order[cut:])]


def _bigram_counts(hist: pl.DataFrame, classes: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Laplace +1 transition counts over the history (all customers, vectorized) and the
    last-history-type per customer. `hist` is pre-filtered to class types by the caller."""
    idx = {c: i for i, c in enumerate(classes)}
    T = np.ones((len(classes), len(classes)))
    if hist.height == 0:
        return T, {}
    keys_h = hist["customer_key"].to_numpy()
    types_h = np.array([idx[t] for t in hist["event_type"].to_numpy()])
    boundary = np.ones(len(types_h), bool)
    boundary[1:] = keys_h[1:] != keys_h[:-1]
    # same-customer transitions: position i pairs with i-1 when i is not a boundary
    pos = np.nonzero(~boundary & (np.arange(len(types_h)) > 0))[0]
    np.add.at(T, (types_h[pos - 1], types_h[pos]), 1.0)
    # last event of each customer: i is last when i+1 starts a new customer (or i is final)
    is_last = np.zeros(len(types_h), bool)
    is_last[:-1] = boundary[1:]
    is_last[-1] = True
    last_hist = {str(k): int(t) for k, t in zip(keys_h[is_last], types_h[is_last], strict=False)}
    return T, last_hist


def _measure_pair(fixture_dir: Path, products: Path, as_of) -> dict:
    stream = pl.read_parquet(fixture_dir / "customer_event_stream.parquet")
    keys = sorted(stream["customer_key"].unique().to_list())
    akeys, bkeys = _split_customers(keys)

    fut = stream.filter(to_ts_expr(pl.col("event_ts")) > as_of).sort(
        ["customer_key", "event_ts", "event_type"]
    )
    nxt = fut.unique("customer_key", keep="first")
    next_of = dict(zip(nxt["customer_key"].to_list(), nxt["event_type"].to_list(), strict=False))
    classes = sorted(set(next_of.values()))

    emb = _load_served(products)
    hist = (
        stream.filter(to_ts_expr(pl.col("event_ts")) <= as_of)
        .filter(pl.col("event_type").is_in(classes))
        .sort(["customer_key", "event_ts", "event_type"])
    )
    T, last_hist = _bigram_counts(hist, classes)
    P = T / T.sum(1, keepdims=True)
    b_rows = [k for k in bkeys if k in next_of and k in emb]

    nll_base = np.array(
        [
            -math.log(P[last_hist[k], classes.index(next_of[k])])
            if k in last_hist and next_of[k] in classes
            else np.nan
            for k in b_rows
        ]
    )

    a_rows = [k for k in akeys if k in next_of and k in emb]
    n_A = {c: int(sum(1 for k in a_rows if next_of[k] == c)) for c in classes}
    if not a_rows or not b_rows or not any(n >= 8 for n in n_A.values()):
        return {"fatal": "no A/B rows with a next type", "n_B_rows": 0}
    ya = np.array([next_of[k] for k in a_rows])
    yb = np.array([next_of[k] for k in b_rows])
    Ea = np.vstack([emb[k] for k in a_rows])
    Eb = np.vstack([emb[k] for k in b_rows])
    excluded = [c for c in classes if n_A[c] < 8]
    kept = [c for c in classes if c not in excluded]
    scores = []
    for c in kept:
        p, _ = _ridge_oos(Ea, (ya == c).astype(float), Eb, (yb == c).astype(float))
        if p is None:
            excluded.append(c)
            kept = [k2 for k2 in kept if k2 != c]
            continue
        scores.append(p)
    if not kept:
        return {"fatal": "no class head fit", "n_B_rows": 0, "classes_excluded": excluded}
    S = np.column_stack(scores)
    S = S - S.max(1, keepdims=True)
    Pm = np.exp(S)
    Pm = np.clip(Pm, P_FLOOR, None)
    Pm /= Pm.sum(1, keepdims=True)
    ki = np.array([kept.index(next_of[k]) for k in b_rows])
    nll_enc = -np.log(Pm[np.arange(len(b_rows)), ki])

    orc_path = fixture_dir / "oracle.json"
    orc = json.loads(orc_path.read_text()) if orc_path.exists() else {}
    # transition_oracle writes NATS (np.log); the receipt and every NLL here are nats.
    # The old x ln2 read was a unit error (DEC-583/586): it deflated the oracle by 30%.
    nll_oracle = orc.get("oracle_ce")
    baseline_ce = orc.get("baseline_ce")

    m = ~np.isnan(nll_base) & ~np.isnan(nll_enc)
    nll_base, nll_enc = nll_base[m], nll_enc[m]
    delta = nll_enc - nll_base  # negative = the encoder's NLL is lower (better)
    ci_lo, ci_hi = _paired_ci(delta) if len(delta) else (None, None)
    # beats_base: the WHOLE (enc - base) CI must sit below zero — the encoder is
    # significantly lower-NLL than the 1-gram. (ci_lo > 0 was the sign-inverted test:
    # it read "significantly worse" as a win.)
    beats_base = bool(ci_hi is not None and ci_hi < 0)

    g = None
    g_fallback = False
    if nll_oracle is not None and len(delta):
        denom = float(nll_base.mean()) - nll_oracle
        if denom <= 0:
            denom = (baseline_ce - nll_oracle) if baseline_ce is not None else None
            g_fallback = True
        if denom:
            g = (float(nll_enc.mean()) - nll_oracle) / denom

    eff_rank, eff_rank_src = _eff_rank(products)
    return {
        "n_customers": len(keys),
        "n_A": len(a_rows),
        "n_B_rows": int(len(m)),
        "classes": classes,
        "classes_excluded": excluded,
        "per_class_n_A": n_A,
        "nll_enc": _round(nll_enc.mean()) if len(m) else None,
        "nll_base": _round(nll_base.mean()) if len(m) else None,
        "nll_oracle_nats": _round(nll_oracle),
        "nll_baseline_ce_nats": _round(baseline_ce),
        "g": _round(g),
        "g_fallback_to_closed_form_denominator": g_fallback,
        "paired_ci_enc_minus_base": [ci_lo, ci_hi],
        "beats_base": beats_base,
        "eff_rank": eff_rank,
        "eff_rank_source": eff_rank_src,
    }


def _measure_pair_hier(fixture_dir: Path, products: Path, as_of) -> dict:
    """E1 head-form v2: per-customer transition counts shrunk to the pooled 1-gram.

    The served signature block is the exact per-customer pair counter. The head predicts the
    next type from the last pre-as_of type with `p = (N_c[last] + alpha * P0[last]) /
    (sum N_c[last] + alpha)`; alpha is chosen on split A only. This is the order statistic
    the mean state cannot carry, read by a simple consumer.
    """
    stream = pl.read_parquet(fixture_dir / "customer_event_stream.parquet")
    keys = sorted(stream["customer_key"].unique().to_list())
    akeys, bkeys = _split_customers(keys)
    fut = stream.filter(to_ts_expr(pl.col("event_ts")) > as_of).sort(
        ["customer_key", "event_ts", "event_type"]
    )
    nxt = fut.unique("customer_key", keep="first")
    next_of = dict(zip(nxt["customer_key"].to_list(), nxt["event_type"].to_list(), strict=False))
    classes = sorted(set(next_of.values()))
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    hist = (
        stream.filter(to_ts_expr(pl.col("event_ts")) <= as_of)
        .filter(pl.col("event_type").is_in(classes))
        .sort(["customer_key", "event_ts", "event_type"])
    )
    T, last_hist = _bigram_counts(hist, classes)
    P0 = T / T.sum(1, keepdims=True)
    per: dict[str, tuple[np.ndarray, int]] = {}
    for g in hist.partition_by("customer_key", maintain_order=True):
        k = str(g["customer_key"][0])
        types = np.array([idx[t] for t in g["event_type"]])
        C = np.zeros((K, K), dtype=np.float64)
        for a, b in zip(types[:-1], types[1:]):
            C[a, b] += 1.0
        per[k] = (C, int(types[-1]))
    if not per:
        return {"fatal": "no history for the signature head", "n_B_rows": 0}

    def nll(ks, alpha):
        vals = []
        for k in ks:
            if k not in next_of or k not in per:
                continue
            C, last = per[k]
            p = (C[last] + alpha * P0[last]) / (C[last].sum() + alpha)
            vals.append(-math.log(max(float(p[idx[next_of[k]]]), P_FLOOR)))
        return np.asarray(vals)

    grid = np.logspace(-2, 4, 25)
    best = None
    for al in grid:
        v = nll(akeys, al)
        if len(v):
            score = float(v.mean())
            if best is None or score < best[0]:
                best = (score, float(al))
    if best is None:
        return {"fatal": "no A rows for the signature head", "n_B_rows": 0}
    alpha = best[1]
    nll_enc = nll(bkeys, alpha)
    nll_base = np.asarray(
        [-math.log(P0[per[k][1], idx[next_of[k]]]) for k in bkeys if k in next_of and k in per]
    )
    m = np.isfinite(nll_enc) & np.isfinite(nll_base)
    nll_enc, nll_base = nll_enc[m], nll_base[m]
    delta = nll_enc - nll_base
    ci_lo, ci_hi = _paired_ci(delta) if len(delta) else (None, None)
    beats_base = bool(ci_hi is not None and ci_hi < 0)
    orc = (
        json.loads((fixture_dir / "oracle.json").read_text())
        if (fixture_dir / "oracle.json").exists()
        else {}
    )
    nll_oracle = orc.get("oracle_ce")
    baseline_ce = orc.get("baseline_ce")
    g = None
    if nll_oracle is not None and len(delta):
        denom = float(nll_base.mean()) - nll_oracle
        if denom <= 0:
            denom = (baseline_ce - nll_oracle) if baseline_ce is not None else None
        if denom:
            g = (float(nll_enc.mean()) - nll_oracle) / denom
    eff_rank, eff_rank_src = _eff_rank(products)
    a_rows = [k for k in akeys if k in next_of and k in per]
    n_A = {c: int(sum(1 for k in a_rows if next_of[k] == c)) for c in classes}
    return {
        "head_form": "hierarchical_dirichlet",
        "alpha": round(alpha, 6),
        "n_customers": len(keys),
        "n_A": len(a_rows),
        "n_B_rows": int(len(m)),
        "classes": classes,
        "classes_excluded": [],
        "per_class_n_A": n_A,
        "nll_enc": _round(nll_enc.mean()) if len(m) else None,
        "nll_base": _round(nll_base.mean()) if len(m) else None,
        "nll_oracle_nats": _round(nll_oracle),
        "nll_baseline_ce_nats": _round(baseline_ce),
        "g": _round(g),
        "paired_ci_enc_minus_base": [ci_lo, ci_hi],
        "beats_base": beats_base,
        "eff_rank": eff_rank,
        "eff_rank_source": eff_rank_src,
    }


def _ridge_oos(Ea, ya, Eb, yb):
    """(predictions, R2) with the battery's honest-OOS guards (fixed capacity, A-fit/B-score)."""
    if len(Ea) < 8 or float(np.std(ya)) < 1e-12 or float(np.std(yb)) < 1e-12:
        return None, None
    m = _ridge_head()
    m.fit(Ea, ya)
    p = m.predict(Eb)
    return p, None


def _paired_ci(deltas: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        means[i] = deltas[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixture", required=True, help="use-case dir, e.g. rabbit_hole/data/lab/next_event"
    )
    ap.add_argument(
        "--products", required=True, help="CFM products dir trained on the positive fixture"
    )
    ap.add_argument("--as-of", required=True, help="training boundary (YYYY-MM-DD)")
    ap.add_argument("--neg-products", default=None, help="products trained on the negative fixture")
    ap.add_argument(
        "--adv-products", default=None, help="products trained on the adversarial fixture"
    )
    ap.add_argument("--iter", default="iter1", help="iteration label recorded in the receipt")
    ap.add_argument(
        "--head",
        choices=("ridge", "hier"),
        default="ridge",
        help="consumer head: the shipping ridge, or the hierarchical Dirichlet signature head",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.perf_counter()
    fixture = Path(a.fixture)
    use_case = fixture.name
    as_of = datetime.strptime(a.as_of, "%Y-%m-%d")
    out_path = Path(a.out) if a.out else LAB_RECEIPTS / f"{use_case}_{a.iter}.json"

    lab_gate = LG.evaluate()["rows"].get(use_case, {})
    measure = _measure_pair_hier if a.head == "hier" else _measure_pair
    pos = measure(fixture, Path(a.products), as_of)
    pos["mastered"] = bool(
        "g" in pos
        and pos["g"] is not None
        and pos["g"] <= BAR_G
        and pos["beats_base"]
        and pos["eff_rank"] is not None
    )

    body = {
        "kind": "lab_dev_instrument",
        "use_case": use_case,
        "iteration": a.iter,
        "protocol": "DEC-576 dev-rig speedometer + controls (development instrument; never B-row evidence)",
        "head_form": a.head,
        "as_of": a.as_of,
        "fixture_spec": json.loads((fixture / f"{use_case}_truth.json").read_text())["spec"],
        "lab_gate_a_side": lab_gate,
        "positive": pos,
        "bar_g": BAR_G,
        "a_frac": A_FRAC,
        "p_floor": P_FLOOR,
        "seed": SEED,
        "n_boot": N_BOOT,
        "discipline": (
            "toy-green can only SUGGEST the next real-data run (DEC-576); certification of B "
            "stays on the real stream (cfm_v2k / O3 / DEC-575); no B-row source names this receipt"
        ),
        "products_identity": products_identity(Path(a.products)),
        "complete": False,
        "seconds": 0.0,
    }

    if a.neg_products:
        neg = measure(fixture / "negative", Path(a.neg_products), as_of)
        neg["expected"] = "NO_SIGNAL"
        ci = neg.get("paired_ci_enc_minus_base")
        # NO_SIGNAL = the snr=0 null is NOT significantly beaten by the head (the CI of
        # enc-base does not sit entirely below 0). A measurable CI is required to say so.
        neg["reads_no_signal"] = bool(ci and ci[1] is not None and not neg["beats_base"])
        body["negative_control"] = neg
    if a.adv_products:
        adv = measure(fixture / "adversarial", Path(a.adv_products), as_of)
        adv["expected"] = "structure claim NO_SIGNAL despite beats_base via the echo"
        # A side: adversarial_not_carried=True means the TYPES carry no structure (snr=0).
        # So structure_carried is the NEGATION — this is the claim the B head must NOT make.
        adv["structure_carried"] = not bool(lab_gate.get("adversarial_not_carried", False))
        adv["leak_has_teeth_a_side"] = bool(lab_gate.get("adversarial_leak_has_teeth", False))
        adv["reads_no_signal_structure"] = bool(not adv["structure_carried"])
        # The control reads as expected only when BOTH: the structure claim is NO_SIGNAL
        # (types carry nothing) AND the head still beats the 1-gram via the echoed attribute
        # (leak has teeth — a naive "beats baseline" gate would false-PASS here).
        if a.head == "hier":
            # A type-only head cannot see the echoed attribute. The control is: the structure
            # claim stays NO_SIGNAL, the echo still has teeth A-side, and the head does NOT beat
            # the 1-gram on the null types (a non-leaky read).
            adv["reads_as_expected"] = bool(
                adv["reads_no_signal_structure"]
                and not adv.get("beats_base")
                and adv["leak_has_teeth_a_side"]
            )
            adv["head_ignores_echo"] = True
        else:
            adv["reads_as_expected"] = bool(
                adv["reads_no_signal_structure"]
                and adv.get("beats_base")
                and adv["leak_has_teeth_a_side"]
            )
        body["adversarial_control"] = adv

    body["complete"] = bool(
        pos.get("n_B_rows", 0) > 0
        and pos.get("nll_enc") is not None
        and pos.get("nll_base") is not None
        and all(lab_gate.get(k) for k in ("complete", "oracle_carried", "negative_not_carried"))
        and ("adversarial_not_carried" not in lab_gate or lab_gate["adversarial_not_carried"])
    )
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        out_path, body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
    )
    print(f"lab {use_case} {a.iter} -> {out_path} ({body['seconds']}s)")
    print(
        f"  g={pos.get('g')} (bar {BAR_G})  nll enc {pos.get('nll_enc')} / base {pos.get('nll_base')} / "
        f"oracle {pos.get('nll_oracle_nats')}  CI(enc-base) {pos.get('paired_ci_enc_minus_base')}  "
        f"beats_base={pos.get('beats_base')}  eff_rank={pos.get('eff_rank')}  mastered={pos.get('mastered')}"
    )
    for ctl in ("negative_control", "adversarial_control"):
        if ctl in body:
            c = body[ctl]
            ok = (
                c.get("reads_no_signal")
                if ctl == "negative_control"
                else c.get("reads_as_expected")
            )
            print(
                f"  {ctl}: nll enc {c.get('nll_enc')} / base {c.get('nll_base')}  "
                f"beats_base={c.get('beats_base')}  reads as expected={ok}"
            )
    print(f"  complete={body['complete']}")
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
