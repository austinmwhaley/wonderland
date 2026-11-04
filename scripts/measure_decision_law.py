"""B4 decisive laws, stamped on the cfm_v2k products (DEC-585).

decision_dose        B-only, judge-simulator-v2 geometry: one ridge head per identified
                     offer level, fit on split-A reps, queried on split-B reps; argmax vs
                     the lab's declared optimum (ties -> lowest level, the judge's hits()
                     semantics); E arm = stored 256-d state_embeddings (no re-encode),
                     table arm = PATH A's wide table under the SAME head; random/destroyed
                     controls broken on BOTH fit and score sides; paired CI on
                     per-customer (hit_E - hit_table).
decision_ope         legacy protocol v1 port, B-only: last-logged email_send offer level
                     over the FULL stream, reward y_margin_90, 70/30 split seed 0, ridge
                     head on [reps; onehot(action)] (the pre-registered ridge replaces the
                     legacy XGB — B-only simple-head doctrine, same head both arms),
                     policy argmax over K levels, IPS uniform p0 = 1/K, paired bootstrap
                     400 seed 1. demo_v2 randomizes ONLY the offer arm (one_factor_weekly),
                     so the legacy channel arm is RETIRED (pre-registered protocol delta).
decision_invariance  check 1: as-of-cutoff band cert — two independent fresh loads (P1/P2)
                     must be bit-identical and sit within CUTOFF_ULP_COVER ULP of the
                     stored state/E; an LG_SCAN=eager witness (P3) within 4x band;
                     per-customer teeth for gapped customers. check 2: target_key -> None
                     re-encode (fresh counter) refits the dose surface — recorded, not
                     gated. check 3: a seeded level-axis permutation of the E-arm surface
                     (teeth) — recorded, not gated.

GPU: every forward must run on the build device cuda:0 (CPU vs GPU forward maxdiff
2.03e-06 exceeds every bar); the producer refuses and records if unavailable.
Writes docs/report_cards/experiments/encoder_decision_law.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cfm_products import (  # noqa: E402
    _db_path,
    close_at,
    load_products,
    resolve_served_tag,
    stored_tables,
    stream_sequences,
)
from measure_battery_law import (  # noqa: E402
    MIN_ARM_DAYS,
    SEED,
    _load_all,
    _paired_ci,
    _ridge_head,
    arm_cells,
    controls_matched,
    identified_dose_response,
)
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import _table_matrix  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_decision_law.json"
WORK = Path("/tmp/opencode/decision_law")

# the judge's simulator-v2 oracle-argmax ceiling (DEC-559): recorded as CONTEXT for the
# B-only dose share, never a B bar.
ORACLE_CEILING = 0.6404


def _onehot(vals, levels):
    lev = {v: i for i, v in enumerate(levels)}
    M = np.zeros((len(vals), len(levels)))
    for i, v in enumerate(vals):
        M[i, lev.get(v, 0)] = 1.0
    return M


# --------------------------------------------------------------------------- check 1


def _spawn_replica(kind: str, products: str, out_npz: Path, env_extra: dict | None = None) -> Path:
    """One independent fresh load -> forward all customers -> E (build op order, stored
    counters) -> exact fade replica, saved to npz. A SUBPROCESS on purpose: LG_SCAN is read
    at cfm_model import time, and P1/P2 must be independent of this process (and of each
    other) — the reproducibility bar is the point of the cert."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--replica",
        kind,
        "--products",
        products,
        "--replica-out",
        str(out_npz),
    ]
    env = dict(os.environ, **(env_extra or {}))
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    return out_npz


def _replica_worker(kind: str, products: str, out_npz: Path) -> None:
    import torch

    from looking_glass.cfm_config import LN2, TIME_UNIT_SECONDS
    from looking_glass.cfm_state import donor_states
    from safe_run import ensure_boundary

    ensure_boundary()
    tag, _reg = resolve_served_tag(products)
    model, cfg, _meta = load_products(tag, products)  # build device (cuda)
    dev = next(model.parameters()).device
    if str(dev) != "cuda:0":
        raise SystemExit(f"replica refuses: model on {dev}, expected cuda:0 (the build device)")
    stored = stored_tables(products)
    _df, _keys, _split, seqs = stream_sequences(cfg, min_events=1)
    keys = [s["customer"] for s in seqs]
    last_epochs = []
    H = []
    with torch.no_grad():
        for seq in seqs:
            if getattr(model, "row_readout_on", False):
                from looking_glass.cfm_training import _collate

                _t = _collate([seq], model.vocab, model._dev())
                _yb, hb = model.encode(_t)
                _y, h = _yb[0], hb[0]
            else:
                _y, h = model(seq)
            H.append(h.cpu().numpy().astype(np.float32))
            last_epochs.append(float(seq["ts"][-1]))
    H = np.vstack(H)
    # E with the STORED counters, in the build op order (materialize FIRST, then fade)
    ctr = np.vstack([np.asarray(stored["customer_state"][k][1], dtype=np.float32) for k in keys])
    E = donor_states(model, H, ctr)
    from looking_glass.stat_bank import level_block

    E = np.hstack([E, level_block(_df, keys, close_at(products))])
    # the exact day-1 close fade, mirrored op-for-op (CPU float32, as fade_idle computes it)
    ca = close_at(products)
    hl = float(cfg.state_half_life_days)
    dt = torch.tensor([float(ca) - le for le in last_epochs], dtype=torch.float32).unsqueeze(1)
    decay = torch.exp(-(LN2 * dt) / max(hl * TIME_UNIT_SECONDS, 1.0)).clamp(max=1.0)
    S = (torch.as_tensor(H, dtype=torch.float32) * decay).numpy()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        H=H,
        S=S,
        E=E,
        keys=np.array(keys, dtype=object),
        last_epochs=np.asarray(last_epochs),
        close_at=np.float64(ca),
    )
    print(
        f"[decision_law replica:{kind}] wrote {out_npz} (n={len(keys)}, device={dev})", flush=True
    )


def _index(rep, prefix):
    keys = rep["keys"]
    mat = rep[prefix]
    return {k: mat[i] for i, k in enumerate(keys)}


def check1_cutoff_band(products: str) -> dict:
    """The as-of-cutoff band cert (DEC-585): P1/P2 bit-identical; P1 within
    CUTOFF_ULP_COVER ULP of the stored state/E; P3 (eager scan) within 4x band; the
    cutoff-sensitive teeth for gapped customers."""
    import b_spec as B

    stored = stored_tables(products)
    S_st = {k: np.asarray(v[0], dtype=np.float32) for k, v in stored["customer_state"].items()}
    E_st = {k: np.asarray(v, dtype=np.float32) for k, v in stored["embeddings"].items()}
    ca = close_at(products)

    WORK.mkdir(parents=True, exist_ok=True)
    p1 = _spawn_replica("canonical", products, WORK / "replica_p1.npz")
    p2 = _spawn_replica("canonical", products, WORK / "replica_p2.npz")
    p3 = _spawn_replica("eager", products, WORK / "replica_p3.npz", env_extra={"LG_SCAN": "eager"})
    R1, R2, R3 = (np.load(p, allow_pickle=True) for p in (p1, p2, p3))

    bit_p1_p2 = bool(
        (R1["H"] == R2["H"]).all() and (R1["S"] == R2["S"]).all() and (R1["E"] == R2["E"]).all()
    )
    S_st_all = np.vstack(list(S_st.values()))
    E_st_all = np.vstack(list(E_st.values()))
    ulp = float(np.float32(2.0) ** -23)
    band_S = B.CUTOFF_ULP_COVER * ulp * float(np.abs(S_st_all).max())
    band_E = B.CUTOFF_ULP_COVER * ulp * float(np.abs(E_st_all).max())

    def max_delta(rep_indexed, st):
        md = 0.0
        for k, v in rep_indexed.items():
            if k in st:
                md = max(md, float(np.abs(v - st[k]).max()))
        return md

    S1, H1, E1 = _index(R1, "S"), _index(R1, "H"), _index(R1, "E")
    S3, E3 = _index(R3, "S"), _index(R3, "E")
    d_S = max_delta(S1, S_st)
    d_E = max_delta(E1, E_st)
    d_S3 = max_delta(S3, S_st)
    d_E3 = max_delta(E3, E_st)

    # teeth: for customers with an idle gap, the FADED replica must sit closer to the
    # stored state than the RAW (pre-fade) forward does — the cert is cutoff-sensitive.
    gaps = {k: float(ca) - float(R1["last_epochs"][i]) for i, k in enumerate(R1["keys"])}
    gapped = [k for k in S1 if gaps.get(k, 0.0) > 0.0]
    teeth_faded = max(float(np.abs(S1[k] - S_st[k]).max()) for k in gapped) if gapped else 0.0
    teeth_raw = (
        max(float(np.abs(H1[k] - S_st[k]).max()) for k in gapped) if gapped else float("inf")
    )
    teeth_ok = bool(len(gapped) > 0 and teeth_faded < teeth_raw)

    ok = bool(
        bit_p1_p2
        and d_S <= band_S
        and d_E <= band_E
        and d_S3 <= 4.0 * band_S
        and d_E3 <= 4.0 * band_E
        and teeth_ok
    )
    return {
        "ok": ok,
        "why": (
            f"P1/P2 bit-identical={bit_p1_p2}; P1 vs stored S {d_S:.3e} (band {band_S:.3e}) / "
            f"E {d_E:.3e} (band {band_E:.3e}); P3 (eager) S {d_S3:.3e} / E {d_E3:.3e} "
            f"(<= 4x band); teeth faded {teeth_faded:.3e} < raw {teeth_raw:.3e} over "
            f"{len(gapped)} gapped customers; close_at {ca}"
        ),
        "device": "cuda:0",
        "n": int(len(R1["keys"])),
        "close_at": ca,
        "p1_p2_bit_identical": bit_p1_p2,
        "max_delta_S": d_S,
        "band_S": band_S,
        "max_delta_E": d_E,
        "band_E": band_E,
        "p3_max_delta_S": d_S3,
        "p3_max_delta_E": d_E3,
        "p3_allow_S": 4.0 * band_S,
        "p3_allow_E": 4.0 * band_E,
        "teeth_ok": teeth_ok,
        "teeth_faded": teeth_faded,
        "teeth_raw": teeth_raw,
        "n_customers_gapped": len(gapped),
        "ulp": ulp,
        "ulp_cover": B.CUTOFF_ULP_COVER,
    }


# --------------------------------------------------------------------------- B4.1


def decision_dose(stream, truth, emb, akeys, bkeys, demo: Path) -> tuple[dict, dict]:
    """The B-only dose response. Returns (body, refs); refs carry the fitted heads and
    arrays the invariance checks refit/reuse — popped before the body is JSON."""
    t0 = time.perf_counter()
    cells = arm_cells(stream, truth)
    dr = identified_dose_response(cells, MIN_ARM_DAYS)
    levels = sorted({lvl for k in dr for lvl in dr[k]})
    ak = [k for k in akeys if k in dr and k in emb]
    bk = [k for k in bkeys if k in dr and k in emb]
    OPT = {k: int(truth[k]["offer_opt"]) for k in bk}
    if not ak or not bk or not levels:
        return (
            {
                "complete": False,
                "error": "empty identified population (need >=1 A and >=1 B customer, >=1 level)",
                "n": 0,
                "levels": [],
                "share_E": None,
                "share_table": None,
                "ci_lo": None,
                "share_random": None,
                "share_destroyed": None,
                "seconds": round(time.perf_counter() - t0, 1),
            },
            {},
        )

    Ea = np.vstack([emb[k] for k in ak])
    Eb = np.vstack([emb[k] for k in bk])

    table, tinfo = _table_matrix(demo, ak + bk)
    if table is None:
        return (
            {
                "complete": False,
                "error": f"table arm unavailable: {tinfo}",
                "n": 0,
                "levels": levels,
                "share_E": None,
                "share_table": None,
                "ci_lo": None,
                "share_random": None,
                "share_destroyed": None,
                "seconds": round(time.perf_counter() - t0, 1),
            },
            {},
        )
    Ta = np.vstack([table[k] for k in ak])
    Tb = np.vstack([table[k] for k in bk])

    def level_heads(reps, keys):
        heads = {}
        for lvl in levels:
            idx = [i for i, k in enumerate(keys) if lvl in dr.get(k, {})]
            if not idx:
                continue
            heads[lvl] = _ridge_head().fit(
                reps[idx], np.asarray([dr[keys[i]][lvl][0] for i in idx])
            )
        return heads

    def surface(heads, reps, n_b):
        P = np.empty((n_b, len(levels)))
        for j, lvl in enumerate(levels):
            P[:, j] = heads[lvl].predict(reps) if lvl in heads else 0.0
        return P

    heads_E = level_heads(Ea, ak)
    heads_T = level_heads(Ta, ak)
    P_E = surface(heads_E, Eb, len(bk))
    P_T = surface(heads_T, Tb, len(bk))
    got_E = np.array([levels[i] for i in P_E.argmax(1)], dtype=int)
    got_T = np.array([levels[i] for i in P_T.argmax(1)], dtype=int)
    want = np.asarray([OPT[k] for k in bk])
    hit_E = (got_E == want).astype(float)
    hit_T = (got_T == want).astype(float)
    share_E = float(hit_E.mean())
    share_table = float(hit_T.mean())
    _mean, lo, hi = _paired_ci(hit_E - hit_T)

    # negative controls, broken on BOTH fit and score sides (the judge's own usage):
    rng = np.random.default_rng(SEED)
    rand_b, dest_b = controls_matched(rng, Ea, Eb)
    rand_a, dest_a = controls_matched(rng, Ea, Ea)

    def control_share(fit_reps, score_reps):
        heads = level_heads(fit_reps, ak)
        P = surface(heads, score_reps, len(bk))
        got = np.array([levels[i] for i in P.argmax(1)], dtype=int)
        return float(np.mean(got == want))

    share_random = control_share(rand_a, rand_b)
    share_destroyed = control_share(dest_a, dest_b)
    thin = [lvl for lvl in levels if sum(1 for k in ak if lvl in dr.get(k, {})) < 30]

    body = {
        "complete": True,
        "n": len(bk),
        "n_levels": len(levels),
        "levels": [int(x) for x in levels],
        "share_E": share_E,
        "share_table": share_table,
        "ci_lo": lo,
        "ci_hi": hi,
        "share_random": share_random,
        "share_destroyed": share_destroyed,
        "thin_levels": [int(x) for x in thin],
        "oracle_ceiling": ORACLE_CEILING,
        "table_columns": int(tinfo.get("columns", 0)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    refs = {
        "heads_E": heads_E,
        "reps_B": Eb,
        "P_E": P_E,
        "levels": levels,
        "dr": dr,
        "ak": ak,
        "bk": bk,
        "OPT": OPT,
        "share_E": share_E,
        "share_table": share_table,
    }
    return body, refs


# --------------------------------------------------------------------------- B4.3 c2/c3


def check2_marker_invariance(model, seqs, refs) -> dict:
    """Re-encode ALL pre-as_of sequences with attrs[target_key] -> None (the 'none'
    bucket), a FRESH counter via the build loop, E' = donor_states(pre-fade h', ctr');
    refit the dose surface on E'. Recorded, not gated (pre-registered FAIL risk:
    target_key is a lab marker the trunk may have learned)."""
    import torch

    from looking_glass.cfm_model import signed_log1p
    from looking_glass.cfm_state import donor_states
    from looking_glass.cfm_training import _collate

    t0 = time.perf_counter()
    if "target_key" not in model.vocab.attrs:
        return {
            "complete": True,
            "share_E_prime": None,
            "share_E_stored": refs.get("share_E"),
            "ok": None,
            "none_bucket_idx": None,
            "n": 0,
            "note": "target_key not in the served vocab — marker refit not measurable",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    none_idx = int(model.vocab.attrs["target_key"].get("none", 0))
    levels, dr, ak, bk = refs["levels"], refs["dr"], refs["ak"], refs["bk"]
    H2, ctr2 = [], []
    with torch.no_grad():
        for seq in seqs:
            n = len(seq["event_type"])
            seq2 = {**seq, "attrs": {**seq["attrs"], "target_key": [None] * n}}
            _y, h = model(seq2)
            H2.append(h.cpu().numpy().astype(np.float32))
            t = _collate([seq2], model.vocab, model._dev())
            tok = model.tokens_batch(t)
            _cre, M = model.counter(tok, model.raw_fields(t), mask=t.get("mask"))
            ctr2.append(np.asarray(signed_log1p(M[0, -1]).cpu().numpy(), dtype=np.float32))
    keys = [s["customer"] for s in seqs]
    E2 = donor_states(model, np.vstack(H2), np.vstack(ctr2))
    by_key = {k: E2[i] for i, k in enumerate(keys)}
    Ea2 = np.vstack([by_key[k] for k in ak])
    Eb2 = np.vstack([by_key[k] for k in bk])
    heads2 = {}
    for lvl in levels:
        idx = [i for i, k in enumerate(ak) if lvl in dr.get(k, {})]
        if not idx:
            continue
        heads2[lvl] = _ridge_head().fit(Ea2[idx], np.asarray([dr[ak[i]][lvl][0] for i in idx]))
    P2 = np.empty((len(bk), len(levels)))
    for j, lvl in enumerate(levels):
        P2[:, j] = heads2[lvl].predict(Eb2) if lvl in heads2 else 0.0
    got = np.array([levels[i] for i in P2.argmax(1)], dtype=int)
    want = np.asarray([refs["OPT"][k] for k in bk])
    share_E_prime = float(np.mean(got == want))
    return {
        "complete": True,
        "share_E_prime": share_E_prime,
        "share_E_stored": refs.get("share_E"),
        "ok": bool(share_E_prime >= refs["share_E"]) if refs.get("share_E") is not None else None,
        "none_bucket_idx": none_idx,
        "n": len(bk),
        "note": (
            "target_key is a lab marker (which arm randomized this event); the pre-registered "
            "risk is that the trunk learned it — recorded, not gated"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def check3_level_teeth(refs) -> dict:
    """A seeded permutation of the level axis applied to the E-arm recovered surface:
    the surface must respond to the axis (teeth). Recorded, not gated."""
    levels, heads_E, reps_B, P_E = refs["levels"], refs["heads_E"], refs["reps_B"], refs["P_E"]
    perm = np.random.default_rng(SEED).permutation(len(levels))

    # one-shot surface: stack the per-level ridge coefficients and do a single
    # (n, K) matmul — algebraically the per-level .predict() (Ridge: X @ coef_ +
    # intercept_) with zero rows for un-fitted levels, no level-axis loop
    def _coef(h):
        return h.coef_ if hasattr(h, "coef_") else h[-1].coef_

    def _icpt(h):
        return h.intercept_ if hasattr(h, "intercept_") else h[-1].intercept_

    coefs = np.array(
        [_coef(heads_E[lvl]) if lvl in heads_E else np.zeros(reps_B.shape[1]) for lvl in levels]
    )
    consts = np.array([float(_icpt(heads_E[lvl])) if lvl in heads_E else 0.0 for lvl in levels])
    P_perm = reps_B @ coefs[perm].T + consts[perm]
    got = np.asarray(levels, dtype=int)[P_perm.argmax(1)]
    want = np.asarray([refs["OPT"][k] for k in refs["bk"]])
    d_inf = float(np.abs(P_perm - P_E).max())
    return {
        "complete": True,
        "share_permuted": float(np.mean(got == want)),
        "share_table": refs.get("share_table"),
        "abs_delta_inf": d_inf,
        "ok": bool(d_inf > 0.0),
        "permutation": [int(x) for x in perm],
    }


# --------------------------------------------------------------------------- B4.2


def decision_ope(bkeys, emb, truth, demo: Path) -> dict:
    """Protocol v1, offer arm only (DEC-585): last-logged email_send offer over the FULL
    stream, reward y_margin_90, 70/30 split seed 0, ridge head on [reps; onehot(action)],
    policy argmax over K, IPS uniform p0 = 1/K, paired bootstrap 400 seed 1."""
    import duckdb

    t0 = time.perf_counter()
    full = pl.read_parquet(demo / "customer_event_stream.parquet")
    con = duckdb.connect()
    con.register("s", full)
    try:
        last = con.execute(
            """
            select customer_key,
                   arg_max(try_cast(json_extract_string(event_attributes,'$.offer') as integer),
                           event_ts) as off,
                   count(*) as n_sends
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

    table, tinfo = _table_matrix(demo, bkeys)
    if table is None:
        return {
            "complete": False,
            "error": f"table arm unavailable: {tinfo}",
            "n_pop": 0,
            "seconds": round(time.perf_counter() - t0, 1),
        }
    pop = [k for k in bkeys if k in lab and k in acts and k in emb and k in table and k in truth]
    n_excluded = len(bkeys) - len(pop)
    if len(pop) < 30:
        return {
            "complete": False,
            "error": f"population too small ({len(pop)} < 30) — excluded {n_excluded}",
            "n_pop": len(pop),
            "n_excluded": n_excluded,
            "seconds": round(time.perf_counter() - t0, 1),
        }
    actions = np.asarray([acts[k] for k in pop], dtype=int)
    reward = np.asarray([lab[k] for k in pop], dtype=float)
    want = np.asarray([int(truth[k]["offer_opt"]) for k in pop], dtype=int)
    X_E = np.vstack([emb[k] for k in pop])
    X_T = np.vstack([table[k] for k in pop])
    levels = sorted(set(int(w) for w in want) | set(int(a) for a in actions))
    K = len(levels)
    p0 = 1.0 / K
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(reward))
    ntr = int(0.7 * len(reward))
    tr, te = idx[:ntr], idx[ntr:]

    def arm_policy(X):
        m = _ridge_head()
        m.fit(np.hstack([X[tr], _onehot(actions[tr], levels)]), reward[tr])
        base = np.repeat(X[te], K, axis=0)
        pred = m.predict(
            np.hstack([base, _onehot(np.tile(np.asarray(levels), len(te)), levels)])
        ).reshape(len(te), K)
        return pred.argmax(1), pred

    pol_E, _pred_E = arm_policy(X_E)
    pol_T, _pred_T = arm_policy(X_T)

    def ips(pol):
        return float(np.mean((pol == actions[te]) * reward[te] / p0))

    e_ips, t_ips = ips(pol_E), ips(pol_T)
    rng_b = np.random.default_rng(1)
    diffs = []
    for _ in range(400):
        bi = rng_b.integers(0, len(te), len(te))
        diffs.append(
            float(
                np.mean((pol_E[bi] == actions[te][bi]) * reward[te][bi] / p0)
                - np.mean((pol_T[bi] == actions[te][bi]) * reward[te][bi] / p0)
            )
        )
    diffs = np.asarray(diffs)
    freq = {str(lvl): round(float(np.mean(actions[te] == lvl)), 4) for lvl in levels}
    return {
        "complete": True,
        "arm": "offer",
        "protocol": (
            "v1 port (DEC-585): last-logged email_send offer; as-of state; uniform IPS "
            "p0=1/K; standardized-ridge head (replaces legacy XGB — B-only simple-head "
            "doctrine); legacy channel arm RETIRED (demo_v2 randomizes offer only)"
        ),
        "levels": [int(x) for x in levels],
        "E_ips": round(e_ips, 2),
        "T_ips": round(t_ips, 2),
        "random": round(float(np.mean(reward[te])), 2),
        "diff": round(float(diffs.mean()), 2),
        "lo": round(float(np.quantile(diffs, 0.025)), 2),
        "hi": round(float(np.quantile(diffs, 0.975)), 2),
        "accE": round(float(np.mean(pol_E == want[te])), 3),
        "accT": round(float(np.mean(pol_T == want[te])), 3),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_pop": int(len(pop)),
        "n_excluded": int(n_excluded),
        "level_frequencies": freq,
        "ips_audit": (
            "uniform p0=1/K pre-registered; observed test-population level frequencies "
            "recorded above; last-logged send and as-of state overlap in time (reported, "
            "not hidden)"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/wonderland_work/cfm_v2k")
    ap.add_argument("--demo", default="/home/austin_whaley/wonderland_work/demo_v2")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--replica", choices=["canonical", "eager"], default=None)
    ap.add_argument("--replica-out", default=None)
    a = ap.parse_args()

    if a.replica:  # the hidden subprocess entry (check-1 replicas)
        _replica_worker(
            a.replica, a.products, Path(a.replica_out or str(WORK / f"replica_{a.replica}.npz"))
        )
        return 0

    from safe_run import ensure_boundary

    ensure_boundary()
    t0 = time.perf_counter()
    demo = Path(a.demo)

    # GPU or refuse + record: every forward in this producer needs the build device.
    try:
        tag, _reg = resolve_served_tag(a.products)
        model, cfg, _meta = load_products(tag, a.products, require_gpu=True)
    except (RuntimeError, SystemExit) as exc:
        body = {
            "complete": False,
            "refused": str(exc),
            "products": products_identity(a.products),
            "decision_dose": {"complete": False, "error": str(exc)},
            "decision_ope": {"complete": False, "error": str(exc)},
            "decision_invariance": {
                "complete": False,
                "check1_cutoff_band": {"ok": False, "why": f"refused: {exc}"},
                "check2_marker_invariance": {"complete": True, "share_E_prime": None},
                "check3_level_teeth": {"complete": True, "share_permuted": None},
            },
            "seconds": round(time.perf_counter() - t0, 1),
        }
        stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
        print(f"[decision_law] REFUSED: {exc}", flush=True)
        return 1

    dev = str(next(model.parameters()).device)
    _reg, truth, emb, _raw_state, stream, bkeys, akeys, _as_of = _load_all(
        str(_db_path(a.products)), demo
    )
    print(
        f"[decision_law] {tag} on {dev}: A={len(akeys)} B={len(bkeys)} — dose, ope, "
        "then invariance (3 replica subprocesses + marker re-encode)",
        flush=True,
    )

    dose, refs = decision_dose(stream, truth, emb, akeys, bkeys, demo)
    print(
        json.dumps(
            {
                k: dose.get(k)
                for k in ("share_E", "share_table", "ci_lo", "share_random", "share_destroyed")
            },
            indent=1,
        ),
        flush=True,
    )

    ope = decision_ope(bkeys, emb, truth, demo)
    print(
        f"[decision_law] ope: complete={ope.get('complete')} "
        f"E_ips={ope.get('E_ips')} T_ips={ope.get('T_ips')} diff={ope.get('diff')} "
        f"lo={ope.get('lo')}",
        flush=True,
    )

    c1 = check1_cutoff_band(a.products)
    print(f"[decision_law] check1: ok={c1.get('ok')} {c1.get('why')}", flush=True)

    seqs_all = list(stream_sequences(cfg, min_events=1)[3])
    c2 = (
        check2_marker_invariance(model, seqs_all, refs)
        if refs
        else {
            "complete": True,
            "share_E_prime": None,
            "share_E_stored": None,
            "ok": None,
            "n": 0,
            "note": "dose surface unavailable — marker refit not measurable",
        }
    )
    print(f"[decision_law] check2: share_E_prime={c2.get('share_E_prime')}", flush=True)

    c3 = (
        check3_level_teeth(refs)
        if refs
        else {
            "complete": True,
            "share_permuted": None,
            "share_table": dose.get("share_table"),
            "abs_delta_inf": None,
            "ok": None,
        }
    )
    print(f"[decision_law] check3: share_permuted={c3.get('share_permuted')}", flush=True)

    body = {
        "complete": bool(
            dose.get("complete")
            and ope.get("complete")
            and c2.get("complete")
            and c3.get("complete")
        ),
        "device": dev,
        "tag": tag,
        "n_customers": int(len(emb)),
        "n_A": int(len(akeys)),
        "n_B": int(len(bkeys)),
        "products": products_identity(a.products),
        "decision_dose": dose,
        "decision_ope": ope,
        "decision_invariance": {
            "complete": bool(
                c1.get("ok") is not None and c2.get("complete") and c3.get("complete")
            ),
            "check1_cutoff_band": c1,
            "check2_marker_invariance": c2,
            "check3_level_teeth": c3,
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(
        json.dumps(
            {
                "dose": {
                    k: dose.get(k)
                    for k in ("share_E", "share_table", "ci_lo", "share_random", "share_destroyed")
                },
                "ope": {k: ope.get(k) for k in ("E_ips", "T_ips", "diff", "lo")},
                "check1": c1.get("why"),
                "check2": c2.get("share_E_prime"),
                "check3": c3.get("share_permuted"),
            },
            indent=1,
        ),
        flush=True,
    )
    print("WROTE", a.out)
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
