"""Standardized downstream probe harness for the FROZEN foundation encoder
(DEC-052).

The encoder is a finished *donor*: these objectives are strictly DOWNSTREAM
probes. Encoder weights are frozen (``requires_grad=False``); only the probe head
trains. Two head families (Linear, shallow MLP) separate linearly-separable from
nonlinearly-accessible information.

Objectives (10): next, dt, jepa, mask, value, entity, order, agg, query, sf.

Protocol
--------
* One standardized fold split (grouped by held-out customer) and one target mask
  (drop company-action and end-of-sequence-pad target positions), shared by all
  probes.
* Plain unweighted CE / MSE — no focal loss, class weights, or label smoothing.
* Expected Calibration Error (ECE) for categorical probes.
* Model-free baselines on the exact same target set:
    - next/mask/entity: unigram (marginal) and 1-gram (previous event id);
    - dt/value/query/agg/sf/jepa: historical mean and previous-value (naive).
* Feature = the frozen per-position readout ``z_t`` (final state ``h`` for the
  sequence-level `order` probe). Forward is verified grad-free (fail-fast).

Verdict (DEC-046, shared with the portfolio): PASS / FAIL /
NO_IDENTIFIABLE_SIGNAL. The trivial predictor (unigram / historical mean) is the
null; a stronger model-free baseline that beats the null by > 2xSE proves signal
exists; a probe PASSes only if it beats that stronger baseline by > 2xSE.

Run:  python3 -m looking_glass.probes --db <stream.duckdb> --as-of <date> \
          --ckpt-dir <dir> --tag <tag> [--out report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from looking_glass.cfm_data import (
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
    draw_sample,
)
from looking_glass.cfm_training import _collate

OBJECTIVES = ("next", "dt", "jepa", "mask", "value", "entity", "order", "agg", "query", "sf")
CATEGORICAL = {"next", "mask", "entity", "order"}
N_FOLDS = 5
PAD = -1  # pad marker inside this module


def freeze_encoder(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bad = [n for n, p in model.named_parameters() if p.requires_grad]
    if bad:
        raise RuntimeError(f"probe harness: encoder not frozen: {bad[:5]}")


# ---------------------------------------------------------------------------
# one frozen encoding pass (+ a masked pass for the `mask` probe)
# ---------------------------------------------------------------------------
def _encode(model, vocab, seqs, batch=128, mask_frac=0.15, seed=0):
    dev = model._dev()
    NE = vocab.n_et
    rows = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i : i + batch]
        t = _collate(chunk, vocab, dev)
        with torch.no_grad():
            y, h = model.encode(t)
            rand = (torch.rand(t["et"].shape, device=dev) < mask_frac) & (t["mask"] > 0)
            t2 = dict(t)
            t2["et"] = torch.where(rand, torch.full_like(t["et"], NE), t["et"])
            # Redact ALL content channels at the masked position (not just the
            # event type) — otherwise brand/entity/value leak the masked event
            # through the readout (measured: mask probe CE 0.47 vs unigram 1.10 on
            # the L0 negative control). Matches training `_masked_forward`.
            t2["br"] = torch.where(rand, torch.full_like(t["br"], vocab.n_brand), t["br"])
            t2["en"] = torch.where(rand, torch.full_like(t["en"], vocab.n_ent), t["en"])
            t2["val"] = torch.where(rand.unsqueeze(-1), torch.zeros_like(t["val"]), t["val"])
            if "eid" in t:
                t2["eid"] = torch.where(rand, torch.full_like(t["eid"], vocab.n_eid), t["eid"])
            y_m, _ = model.encode(t2)
        y = y.float().cpu().numpy()
        h = h.float().cpu().numpy()
        y_m = y_m.float().cpu().numpy()
        et = t["et"].cpu().numpy()
        en = t["en"].cpu().numpy()
        val = t["val"].squeeze(-1).cpu().numpy()
        dt = t["dt"].squeeze(-1).cpu().numpy()
        m = t["mask"].cpu().numpy()
        co = t["co"][:, :, 0].cpu().numpy()
        rmask = rand.cpu().numpy()
        _pv = None
        if getattr(model, "prod_vec_table", None) is not None and model.prod_vec_table.shape[1] > 0:
            _pv = model.prod_vec_table[t["eid"]].float().cpu().numpy()
        for b in range(et.shape[0]):
            L = int(m[b].sum())
            if L < 3:
                continue
            rows.append(
                {
                    "cust": chunk[b]["customer"],
                    "Z": y[b, :L],
                    "H": h[b],
                    "Zm": y_m[b, :L],
                    "rmask": rmask[b, :L],
                    "et": et[b, :L],
                    "en": en[b, :L],
                    "val": val[b, :L],
                    "dt": dt[b, :L],
                    "co": co[b, :L],
                    **({"pvec": _pv[b, :L]} if _pv is not None else {}),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# build (X, y, kind, prev_id, prev_val, seq) per objective — same mask everywhere
# ---------------------------------------------------------------------------
def _build(name, rows, n_ent, n_ev, horizon):
    X, Y, PREVID, PREVVAL, S = [], [], [], [], []
    for r in rows:
        L = len(r["et"])
        if name == "order":
            X.append(r["H"])
            Y.append(1.0 if (r["et"] == 1).any() else 0.0)
            PREVID.append(-1)
            PREVVAL.append(0.0)
            S.append(r["cust"])
            continue
        if name == "sf":
            # Successor features MATCHING the training objective (cfm_training
            # `sf`): the time-discounted future phi = [log1p|value|, event one-hot],
            # scale-free. (The old raw future-COUNT target was a different,
            # magnitude-heavy quantity the model was never trained on.)
            etr = np.asarray(r["et"], int)
            valr = np.log1p(np.abs(np.asarray(r["val"], float)))
            gap = np.expm1(np.asarray(r["dt"], float))
            tau = np.cumsum(gap)
            w = np.zeros((L, n_ev))
            w[:, 0] = valr
            _pvec = r.get("pvec")
            ok = (etr >= 0) & (etr < n_ev - 1)
            w[np.arange(L)[ok], 1 + etr[ok]] = 1.0
            gam, unit = 0.5, 86400.0
            gpow = gam ** (tau / unit)
            wg = w * gpow[:, None]
            rc = np.cumsum(wg[::-1], axis=0)[::-1]
            ssum = np.zeros_like(w)
            ssum[:-1] = rc[1:]
            Rrow = ssum * (gam ** (-tau / unit))[:, None]
            if _pvec is not None:
                Rrow = np.concatenate([Rrow, _pvec], axis=1)
        for t in range(L):
            if r["co"][t] > 0:
                continue
            z = r["Z"][t]
            if name in ("next", "dt", "value", "entity", "agg", "query", "jepa", "sf"):
                if t + 1 >= L or r["co"][t + 1] > 0 or r["et"][t + 1] == n_ev - 1:
                    continue
            elif name == "mask":
                if not r["rmask"][t]:
                    continue
            if name == "next":
                X.append(z)
                Y.append(r["et"][t + 1])
                PREVID.append(r["et"][t])
                PREVVAL.append(0.0)
            elif name == "dt":
                X.append(z)
                Y.append(r["dt"][t + 1])
                PREVID.append(-1)
                PREVVAL.append(r["dt"][t])
            elif name == "value":
                X.append(z)
                Y.append(np.log1p(abs(r["val"][t + 1])))
                PREVID.append(-1)
                PREVVAL.append(np.log1p(abs(r["val"][t])))
            elif name == "entity":
                X.append(z)
                Y.append(r["en"][t + 1])
                PREVID.append(r["et"][t])
                PREVVAL.append(0.0)
            elif name == "mask":
                X.append(r["Zm"][t])
                Y.append(r["et"][t])
                # 1-gram predictor for a masked event = the PREVIOUS observed
                # event (NOT the target itself — that leaked the answer and gave
                # a bogus ~0 baseline).
                PREVID.append(r["et"][t - 1] if t >= 1 else -1)
                PREVVAL.append(0.0)
            elif name == "agg":
                hi = min(t + 1 + horizon, L)
                cnt = hi - (t + 1)
                vs = float(np.abs(r["val"][t + 1 : hi]).sum()) if hi > t + 1 else 0.0
                X.append(z)
                Y.append([np.log1p(cnt), np.log1p(vs)])
                PREVID.append(-1)
                PREVVAL.append(0.0)
            elif name == "query":
                last = next((u for u in range(t, -1, -1) if r["et"][u] == 1), -1)
                X.append(z)
                Y.append(np.log1p(0.0 if last < 0 else float(t - last)))
                PREVID.append(-1)
                PREVVAL.append(0.0)
            elif name == "jepa":
                hi = min(t + 1 + horizon, L)
                fut = r["Z"][t + 1 : hi]
                X.append(z)
                Y.append((fut.mean(0) if len(fut) else z) @ _jepa_proj(z.shape[0]))
                PREVID.append(-1)
                PREVVAL.append(0.0)
            elif name == "sf":
                X.append(z)
                Y.append(Rrow[t])
                PREVID.append(-1)
                PREVVAL.append(0.0)
            S.append(r["cust"])
    if not X:
        return None
    Y = np.asarray(Y)
    if name == "sf":
        Y = (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-6)  # scale-free (DEC-018)
    return (
        np.asarray(X, float),
        Y,
        np.asarray(PREVID),
        np.asarray(PREVVAL),
        np.asarray(S),
    )


_JP = {}


def _jepa_proj(dim):
    """Fixed random projection of the future-window embedding to 16 dims (bounds
    the multi-output probe cost; the projection is fixed, not learned)."""
    if dim not in _JP:
        g = np.random.default_rng(7)
        p = g.normal(size=(dim, 16)).astype(np.float64)
        _JP[dim] = p
    return _JP[dim]


def _kind(name):
    if name in CATEGORICAL:
        return "cat"
    if name in ("agg", "sf", "jepa"):
        return "multi"
    return "reg"


def _head(kind, family):
    # Standardize features inside the head (fit on train only): raw frozen states
    # are O(10-100) scale, which makes LBFGS crawl and MLPs mistune. The scaler is
    # part of the fitted pipeline => no leakage; it is a conditioning fix, not a
    # change to the graded quantity.
    if kind == "cat":
        return (
            make_pipeline(StandardScaler(with_std=False), LogisticRegression(max_iter=100))
            if family == "lin"
            else make_pipeline(
                StandardScaler(with_std=False),
                MLPClassifier(hidden_layer_sizes=(128,), max_iter=25, early_stopping=True),
            )
        )
    if kind == "multi":
        return (
            make_pipeline(StandardScaler(with_std=False), LinearRegression())
            if family == "lin"
            else make_pipeline(
                StandardScaler(with_std=False),
                MLPRegressor(hidden_layer_sizes=(128,), max_iter=25, early_stopping=True),
            )
        )
    return (
        make_pipeline(StandardScaler(with_std=False), LinearRegression())
        if family == "lin"
        else make_pipeline(
            StandardScaler(with_std=False),
            MLPRegressor(hidden_layer_sizes=(128,), max_iter=25, early_stopping=True),
        )
    )


def _loss(kind, head, X, y):
    if kind == "cat":
        P = head.predict_proba(X)
        cls = list(head.classes_)
        idx = {c: i for i, c in enumerate(cls)}
        return np.array([-np.log(max(P[i, idx.get(int(t), 0)], 1e-12)) for i, t in enumerate(y)])
    if kind == "multi":
        return np.mean((head.predict(X) - y) ** 2, axis=1)
    return (head.predict(X) - y) ** 2


def _ece(kind, head, X, y, n_bins=10):
    if kind != "cat":
        return None
    P = head.predict_proba(X)
    conf = P.max(1)
    pred = P.argmax(1)
    acc = pred == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        msk = (conf > edges[i]) & (conf <= edges[i + 1])
        if msk.sum() > 0:
            e += msk.mean() * abs(acc[msk].mean() - conf[msk].mean())
    return float(e)


def _unigram_loss(ytr, yva):
    k = int(max(ytr.max(), yva.max())) + 1
    c = np.bincount(ytr.astype(int), minlength=k).astype(float) + 0.5
    p = c / c.sum()
    return np.array([-np.log(max(p[int(t)], 1e-12)) for t in yva])


def _gram_loss(ytr, yva, prevtr, prevva):
    kc = int(max(ytr.max(), yva.max())) + 1
    kp = int(max(prevtr.max(), prevva.max(), 0)) + 1
    marg = np.bincount(ytr.astype(int), minlength=kc).astype(float) + 0.5
    tab = [None] * kp
    for pid, t in zip(prevtr.tolist(), ytr.tolist()):
        if pid < 0:
            continue
        if tab[pid] is None:
            tab[pid] = np.zeros(kc) + 0.5
        tab[pid][t] += 1
    out = []
    for pid, t in zip(prevva.tolist(), yva.tolist()):
        row = tab[pid] if 0 <= pid < kp and tab[pid] is not None else marg
        out.append(-np.log(row[int(t)] / row.sum()))
    return np.array(out)


def _mean_loss(ytr, yva):
    if ytr.ndim == 1:
        return (yva - ytr.mean()) ** 2
    mu = ytr.mean(0)
    return np.mean((yva - mu) ** 2, axis=1)


def _prevval_loss(yva, prev):
    if yva.ndim == 1:
        return (yva - prev) ** 2
    return np.mean((yva - prev) ** 2, axis=1)


def _verdict(model_m, stronger_base, null_base, se):
    floor = 2.0 * se
    if stronger_base < null_base - floor:
        # a model-free baseline beats the trivial predictor => signal exists;
        # the probe must beat that stronger baseline.
        return "PASS" if model_m < stronger_base - floor else "FAIL"
    # no model-free signal; PASS only if the probe beats the trivial predictor.
    return "PASS" if model_m < null_base - floor else "NO_IDENTIFIABLE_SIGNAL"


def _bootstrap_ci(delta, S, n_boot=1000, seed=0):
    """Customer-level bootstrap 95% CI of the mean per-sample delta
    (delta = strong_baseline_loss - model_loss, >0 = model beats the baseline).
    Resamples HELD-OUT CUSTOMERS (the §9.6 unit), so within-customer correlation
    does not inflate the interval. Returns (mean, lo, hi)."""
    d = np.asarray(delta, float)
    S = np.asarray(S)
    ok = np.isfinite(d)
    d, S = d[ok], S[ok]
    if len(d) < 8:
        return float(np.mean(d)) if len(d) else 0.0, 0.0, 0.0
    cust = np.asarray(sorted(set(S.tolist())), dtype=object)
    idx = {c: i for i, c in enumerate(cust)}
    sums = np.zeros(len(cust))
    cnts = np.zeros(len(cust))
    np.add.at(sums, [idx[c] for c in S], d)
    np.add.at(cnts, [idx[c] for c in S], 1.0)
    cmean = sums / np.maximum(cnts, 1.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(cmean, size=(n_boot, len(cmean)), replace=True).mean(axis=1)
    return float(cmean.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _verdict_ci(mean, lo, hi, stronger_base, null_base, floor):
    """§9.6 / §11.3 defensible verdict: PASS iff the bootstrap 95% CI of the
    delta vs the STRONGER model-free baseline excludes 0 and mean>0; NO_SIGNAL iff
    that baseline is itself at the trivial-predictor floor; else FAIL/UNCERTAIN."""
    signal = stronger_base < null_base - floor
    if lo > 0:
        return "PASS"
    if not signal:
        return "NO_SIGNAL"
    if hi < 0:
        return "FAIL"
    return "UNCERTAIN"


def _stronger_per_sample(bnull, bstrong, Bnull, Bstrong):
    """The stronger model-free baseline (chosen by MEAN loss), applied per sample
    and used for the CI. Deliberately NOT `np.fmin(Bnull, Bstrong)`: a per-sample
    best-of oracle is unbeatable by construction and FALSE-FAILs every model."""
    return Bstrong if bstrong <= bnull else Bnull


def _identity_decodability(rows, fof):
    """Permanent intrinsic diagnostic: can the frozen readout z_t linearly expose
    the CURRENT event type, versus the raw event id e_t? Reports acc / macro-F1 /
    CE out-of-fold. If z_t cannot but e_t can, the readout drops immediate event
    identity."""
    Z, E, Y, S = [], [], [], []
    for r in rows:
        for t in range(len(r["et"])):
            if r["co"][t] > 0:
                continue
            Z.append(r["Z"][t])
            E.append(r["et"][t])
            Y.append(r["et"][t])
            S.append(r["cust"])
    Z = np.asarray(Z)
    E = np.asarray(E)
    Y = np.asarray(Y)
    fold = np.array([fof[c] for c in S])
    k = int(Y.max()) + 1
    onehot = np.zeros((len(E), k))
    onehot[np.arange(len(E)), E] = 1.0
    feats = {"z": Z, "e_t": onehot, "[z,e_t]": np.hstack([Z, onehot])}
    out = {}
    for nm, F in feats.items():
        accs, f1s, ces = [], [], []
        for fi in range(N_FOLDS):
            tr, va = fold != fi, fold == fi
            if not va.any() or not tr.any():
                continue
            clf = LogisticRegression(max_iter=500).fit(F[tr], Y[tr])
            pred = clf.predict(F[va])
            accs.append((pred == Y[va]).mean())
            # macro-F1
            f1 = []
            for c in range(k):
                tp = ((pred == c) & (Y[va] == c)).sum()
                fp = ((pred == c) & (Y[va] != c)).sum()
                fn = ((pred != c) & (Y[va] == c)).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1.append(2 * prec * rec / max(prec + rec, 1e-9))
            f1s.append(np.mean(f1))
            P = clf.predict_proba(F[va])
            cls = list(clf.classes_)
            idx = {c: i for i, c in enumerate(cls)}
            ces.append(
                np.mean(
                    [-np.log(max(P[i, idx.get(int(t), 0)], 1e-12)) for i, t in enumerate(Y[va])]
                )
            )
        out[nm] = {
            "acc": round(float(np.mean(accs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "ce": round(float(np.mean(ces)), 4),
        }
    return out


def _next_slices(X, Y, PREVID, fof, S, vocab):
    """Ecommerce next-event slices: CE by next/prev event type, rare slice, and
    empirical P(purchase | prev=cart) (vs the probe's mean estimate)."""
    fold = np.array([fof[c] for c in S])
    ce = np.full(len(Y), np.nan)
    for fi in range(N_FOLDS):
        tr, va = fold != fi, fold == fi
        if not va.any() or not tr.any():
            continue
        clf = LogisticRegression(max_iter=500).fit(X[tr], Y[tr])
        P = clf.predict_proba(X[va])
        cls = list(clf.classes_)
        idx = {c: i for i, c in enumerate(cls)}
        ce[va] = [-np.log(max(P[i, idx.get(int(t), 0)], 1e-12)) for i, t in enumerate(Y[va])]
    name = {i: v for v, i in vocab.et.items()}

    def nm(i):
        return name.get(int(i), f"id{i}")

    slices = {}
    for i in sorted(set(int(v) for v in Y.tolist())):
        m = Y == i
        slices[f"next={nm(i)}"] = round(float(np.nanmean(ce[m])), 4)
    for i in sorted(set(int(v) for v in PREVID.tolist()) - {-1}):
        m = PREVID == i
        if m.any():
            slices[f"prev={nm(i)}"] = round(float(np.nanmean(ce[m])), 4)
    rare = np.array([nm(y) != "view" for y in Y])
    if rare.any():
        slices["rare(next!=view)"] = round(float(np.nanmean(ce[rare])), 4)
    if "cart" in vocab.et and "purchase" in vocab.et:
        cart, buy = vocab.et["cart"], vocab.et["purchase"]
        m = PREVID == cart
        if m.any():
            slices["empirical_P(purchase|prev=cart)"] = round(float((Y[m] == buy).mean()), 4)
            slices["base_P(purchase)"] = round(float((Y == buy).mean()), 4)
    return slices


def _capability_probes(model, vocab, rows, seqs, fof):
    """Phase-5 capability probes (DEC-056): pacing, boundary, uncertainty.
    Frozen encoder; plain MSE / AUROC; uncertainty is a DERIVED deterministic
    channel (entropy of the frozen next-head), never a trained head."""
    import torch
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score

    dev = model._dev()
    out = {}

    # ---- pacing: session velocity = (mean, std) of log1p(dt), from the mean state
    Xp, Yp, Sp = [], [], []
    for r in rows:
        d = r["dt"][1:]
        if len(d) < 3:
            continue
        Xp.append(r["Z"].mean(0))
        Yp.append([float(d.mean()), float(d.std())])
        Sp.append(r["cust"])
    if Xp:
        Xp = np.asarray(Xp)
        Yp = np.asarray(Yp)
        fp = np.array([fof[c] for c in Sp])
        loss = np.full(len(Yp), np.nan)
        for fi in range(N_FOLDS):
            tr, va = fp != fi, fp == fi
            if va.any() and tr.any():
                h = Ridge().fit(Xp[tr], Yp[tr])
                loss[va] = ((h.predict(Xp[va]) - Yp[va]) ** 2).mean(1)
        base = ((Yp - Yp.mean(0)) ** 2).mean(1)
        m, b = float(np.nanmean(loss)), float(base.mean())
        fm = [np.nanmean(loss[fp == fi]) for fi in range(N_FOLDS) if (fp == fi).any()]
        se = float(np.std(fm) / np.sqrt(max(len(fm), 1)))
        out["pacing"] = {
            "model": m,
            "baseline": b,
            "ceiling": b,
            "se": se,
            "delta": b - m,
            "verdict": "PASS" if m < b - 2 * se else "FAIL",
        }

    # ---- boundary: next event is a session reset (long gap) -> AUROC
    alldt = np.concatenate([r["dt"][1:] for r in rows if len(r["dt"]) > 1])
    thr = float(np.quantile(alldt, 0.95)) if alldt.size else 0.0
    Xb, Yb, Sb = [], [], []
    for r in rows:
        for t in range(len(r["et"]) - 1):
            if r["co"][t] > 0 or r["co"][t + 1] > 0:
                continue
            Xb.append(r["Z"][t])
            Yb.append(1.0 if r["dt"][t + 1] > thr else 0.0)
            Sb.append(r["cust"])
    if Xb and 0 < np.mean(Yb) < 1:
        Xb = np.asarray(Xb)
        Yb = np.asarray(Yb)
        fb = np.array([fof[c] for c in Sb])
        auc = []
        for fi in range(N_FOLDS):
            tr, va = fb != fi, fb == fi
            if va.any() and tr.any() and len(np.unique(Yb[tr])) > 1 and len(np.unique(Yb[va])) > 1:
                clf = LogisticRegression(max_iter=300).fit(Xb[tr], Yb[tr])
                auc.append(roc_auc_score(Yb[va], clf.predict_proba(Xb[va])[:, 1]))
        a = float(np.mean(auc)) if auc else 0.5
        se = float(np.std(auc) / np.sqrt(len(auc))) if auc else 0.0
        out["boundary"] = {
            "model": a,
            "baseline": 0.5,
            "ceiling": 1.0,
            "se": se,
            "delta": a - 0.5,
            "verdict": "PASS" if a > 0.5 + 2 * se else "FAIL",
        }

    # ---- uncertainty: DERIVED U_t = H(softmax(head_next(z_t))). OOD = synthetic
    # anomalous events (novel event id injected); AUROC(U, in-dist vs OOD).
    rng = np.random.default_rng(0)
    Un, Uo = [], []
    for i in range(0, len(seqs), 128):
        chunk = seqs[i : i + 128]
        t = _collate(chunk, vocab, dev)
        with torch.no_grad():
            y, _ = model.encode(t)
            lg = model.head_next(y[:, :-1])
            p = torch.softmax(lg, -1)
            U = (-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).cpu().numpy()
            et2 = t["et"].clone()
            pos = rng.integers(1, max(t["et"].shape[1], 2), size=t["et"].shape[0])
            for b in range(et2.shape[0]):
                et2[b, pos[b]] = vocab.n_et  # novel event
            y2, _ = model.encode({**t, "et": et2})
            lg2 = model.head_next(y2[:, :-1])
            p2 = torch.softmax(lg2, -1)
            U2 = (-(p2 * torch.log(p2.clamp_min(1e-12))).sum(-1)).cpu().numpy()
        m = t["mask"].cpu().numpy()
        for b in range(t["et"].shape[0]):
            L = int(m[b].sum())
            j = int(pos[b]) - 1
            if 0 <= j < L - 1:
                Un.append(float(U[b, j]))
                Uo.append(float(U2[b, j]))
    if Un and Uo and len(set(Un + Uo)) > 1:
        auc = roc_auc_score([0] * len(Un) + [1] * len(Uo), Un + Uo)
        out["uncertainty"] = {
            "model": float(auc),
            "baseline": 0.5,
            "ceiling": 1.0,
            "se": 0.0,
            "delta": float(auc) - 0.5,
            "verdict": "PASS" if auc > 0.5 else "FAIL",
        }
    return out


def run_probe_report(model, vocab, cfg, seqs, tag, light=False):
    freeze_encoder(model)
    dev = model._dev()
    with torch.no_grad():
        t = _collate(seqs[:4], vocab, dev)
        y, _ = model.encode(t)
    if y.requires_grad:
        raise RuntimeError("probe harness: encoder forward returned a grad tensor")

    rows = _encode(model, vocab, seqs)
    n_ent = vocab.n_ent + 1
    n_ev = vocab.n_et + 1
    horizon = max(1, int(np.median([len(r["et"]) for r in rows]) * 0.1))
    custs = sorted({r["cust"] for r in rows})
    rng = np.random.default_rng(1)
    rng.shuffle(custs)
    folds = np.array_split(np.asarray(custs, dtype=object), N_FOLDS)
    fof = {c: fi for fi, f in enumerate(folds) for c in f}

    report = {}
    for name in OBJECTIVES:
        built = _build(name, rows, n_ent, n_ev, horizon)
        if built is None:
            continue
        X, Y, PREVID, PREVVAL, S = built
        cap = 8000  # bound probe cost (speed principle)
        if len(X) > cap:
            sub = rng.choice(len(X), cap, replace=False)
            X, Y, PREVID, PREVVAL, S = [a[sub] for a in built]
        kind = _kind(name)
        # GRADE THE DONOR WAY (SPEC §9.6 req.2 / §9): the model probe reads
        # [E, raw] and the baseline reads raw — the PASS bar is that E ADDS
        # incremental signal over the strong model-free baseline. raw = the
        # previous-event one-hot (cat) / previous value (reg); the multi-output
        # tasks (agg/jepa/sf) have no per-sample raw feature, so they stay E-only
        # vs the bag-of-events mean.
        if kind == "cat":
            raw = np.zeros((len(PREVID), n_ev))
            mm = PREVID >= 0
            raw[np.arange(len(PREVID))[mm], PREVID[mm]] = 1.0
            Xm = np.hstack([X, raw])
        elif kind == "reg":
            Xm = np.hstack([X, np.asarray(PREVVAL, float).reshape(-1, 1)])
        else:
            Xm = X
        fold = np.array([fof[c] for c in S])
        Llin = np.full(len(Y), np.nan)
        Lmlp = np.full(len(Y), np.nan)
        Bnull = np.full(len(Y), np.nan)
        Bstrong = np.full(len(Y), np.nan)
        ece_acc = []
        for fi in range(N_FOLDS):
            tr, va = fold != fi, fold == fi
            if not va.any() or not tr.any():
                continue
            if kind == "cat" and len(np.unique(Y[tr])) < 2:
                # degenerate fold (single class): no probe can be fit; fall back
                # to the trivial predictor for both model and baseline.
                Bnull[va] = _unigram_loss(Y[tr], Y[va])
                Bstrong[va] = Bnull[va]
                Llin[va] = Bnull[va]
                Lmlp[va] = Bnull[va]
                continue
            hl = _head(kind, "lin")
            hl.fit(Xm[tr], Y[tr])
            Llin[va] = _loss(kind, hl, Xm[va], Y[va])
            if not light:
                hm = _head(kind, "mlp")
                hm.fit(Xm[tr], Y[tr])
                Lmlp[va] = _loss(kind, hm, Xm[va], Y[va])
            if kind == "cat":
                Bnull[va] = _unigram_loss(Y[tr], Y[va])
                Bstrong[va] = _gram_loss(Y[tr], Y[va], PREVID[tr], PREVID[va])
                e = _ece(kind, hl, Xm[va], Y[va])
                if e is not None:
                    ece_acc.append(e)
            elif kind == "reg":
                Bnull[va] = _mean_loss(Y[tr], Y[va])
                Bstrong[va] = _prevval_loss(Y[va], PREVVAL[va])
            else:  # multi-output (agg / jepa / sf)
                # The trivial predictor for a multi-output target IS the mean, so
                # the null loss equals the strong-baseline loss (no stronger
                # model-free predictor exists). Previously the null was set to
                # `2*var`, which is always LARGER than the mean loss -> "signal
                # present" -> these objectives could never be NO_SIGNAL (they
                # FALSE-FAILed on the L0 negative control). Now: PASS iff the
                # model beats the mean, else NO_SIGNAL.
                Bstrong[va] = _mean_loss(Y[tr], Y[va])
                Bnull[va] = Bstrong[va]
        lin_m = float(np.nanmean(Llin))
        mlp_m = float(np.nanmean(Lmlp)) if not light else float("nan")
        bnull, bstrong = float(np.nanmean(Bnull)), float(np.nanmean(Bstrong))
        fold_means = [np.nanmean(Llin[fold == fi]) for fi in range(N_FOLDS) if np.any(fold == fi)]
        se = float(np.std(fold_means) / np.sqrt(max(len(fold_means), 1)))
        stronger = min(bnull, bstrong)
        verdict = _verdict(lin_m, stronger, bnull, se)
        # §9.6-defensible verdict: bootstrap 95% CI of the delta vs the stronger
        # model-free baseline (chosen by MEAN loss — the SAME baseline the verdict
        # uses; never a per-sample fmin oracle, which no model could beat),
        # resampled over held-out customers.
        stronger_ps = _stronger_per_sample(bnull, bstrong, Bnull, Bstrong)
        dmean, lo, hi = _bootstrap_ci(stronger_ps - Llin, S, seed=(abs(hash(name)) % (2**31)))
        floor = 2.0 * se
        report[name] = {
            "model_lin": lin_m,
            "model_mlp": mlp_m,
            "se": se,
            "baseline": bstrong,
            "null": bnull,
            "ceiling": stronger,
            "delta": bstrong - lin_m,
            "delta_mean": dmean,
            "ci_lo": lo,
            "ci_hi": hi,
            "verdict": verdict,
            "verdict_ci": _verdict_ci(dmean, lo, hi, stronger, bnull, floor),
            "ece": round(float(np.mean(ece_acc)), 4) if ece_acc else None,
        }
    # §20.4 capability card: if a k-event context does NOT predict the next event
    # better than the 1-gram, the 1-gram is the ceiling and the transition
    # capability is NO_SIGNAL (¬CARRIED beyond the baseline) — a model must not be
    # required to beat an optimal baseline. Keep PASS if the model actually beat it.
    card = {}
    try:
        from looking_glass.capability_card import transition_carried

        carried, card = transition_carried(rows, n_ev)
        if not carried:
            for o in ("next", "mask"):
                if o in report and report[o]["verdict_ci"] in ("FAIL", "UNCERTAIN"):
                    report[o]["verdict_ci"] = "NO_SIGNAL"
                    report[o]["capability_card"] = "1-gram is the ceiling"
    except Exception as exc:  # noqa: BLE001
        card = {"error": str(exc)}
    # intrinsic diagnostics (spec §6/§7) — skipped in `light` mode (the card only
    # needs the per-objective verdicts; these are for the standalone report).
    ident, slices, caps = {}, {}, {}
    if not light:
        try:
            ident = _identity_decodability(rows, fof)
        except Exception as exc:  # noqa: BLE001
            ident = {"error": str(exc)}
        bn = _build("next", rows, n_ent, n_ev, horizon)
        if bn is not None:
            try:
                slices = _next_slices(bn[0], bn[1], bn[2], fof, bn[4], vocab)
                if "next" in report and "rare(next!=view)" in slices:
                    report["next"]["rare"] = slices["rare(next!=view)"]
            except Exception as exc:  # noqa: BLE001
                slices = {"error": str(exc)}
        # Phase-5 capability probes (pacing / boundary / uncertainty)
        try:
            caps = _capability_probes(model, vocab, rows, seqs, fof)
        except Exception as exc:  # noqa: BLE001
            caps = {"error": str(exc)}
    md = _to_markdown(report, tag, ident, slices, caps)
    receipt = {
        "tag": tag,
        "frozen": True,
        "n_sequences": len(rows),
        "objectives": report,
        "capability_card": card,
        "identity_decodability": ident,
        "next_slices": slices,
        "capabilities": caps,
    }
    return md, receipt


def _to_markdown(report, tag, ident=None, slices=None, caps=None):
    lines = [
        f"### Frozen-encoder probe report — {tag}",
        "",
        "| Objective | Probe Type | Model Score (± SE) | Baseline Score | Ceiling Score "
        "| Δ vs Baseline | Rare slice | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in OBJECTIVES:
        if name not in report:
            continue
        r = report[name]
        lines.append(
            f"| {name} | linear (mlp {r['model_mlp']:.4f}) | {r['model_lin']:.4f} ± {r['se']:.4f} "
            f"| {r['baseline']:.4f} | {r['ceiling']:.4f} | {r['delta']:+.4f} | "
            f"{r.get('rare', '')} | {r['verdict']} |"
        )
    if ident:
        lines += [
            "",
            "#### Event-identity decodability (z_t -> current event type)",
            "",
            "| features | acc | macro-F1 | CE |",
            "|---|---|---|---|",
        ]
        for nm, v in ident.items():
            if isinstance(v, dict) and "acc" in v:
                lines.append(f"| {nm} | {v['acc']} | {v['macro_f1']} | {v['ce']} |")
    if slices:
        lines += [
            "",
            "#### ecommerce_2019 next-event slices (CE)",
            "",
            "| slice | value |",
            "|---|---|",
        ]
        for k, v in slices.items():
            lines.append(f"| {k} | {v} |")
    if caps:
        lines += [
            "",
            "#### Phase-5 capabilities (pacing / boundary / uncertainty)",
            "",
            "| capability | Model | Baseline | Δ | Verdict |",
            "|---|---|---|---|---|",
        ]
        for k, v in caps.items():
            if isinstance(v, dict) and "model" in v:
                lines.append(
                    f"| {k} | {v['model']:.4f} | {v['baseline']:.4f} | "
                    f"{v['delta']:+.4f} | {v['verdict']} |"
                )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--sample-a", type=int, default=20000, dest="sample_a")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_state import load_frozen_encoder

    model, _ = load_frozen_encoder(args.tag, args.ckpt_dir)
    dcfg = CFMConfig()
    dcfg.db, dcfg.as_of = args.db, args.as_of
    dcfg.sample_customers, dcfg.sample_a_customers = args.customers, args.sample_a
    df = _read_stream(dcfg)
    keys = _customer_keys(df, dcfg)
    split = assign_split(keys, dcfg)
    ak = draw_sample(keys, split, "A", args.sample_a, dcfg.split_seed)
    seqs = build_sequences(df, ak, dcfg, split, with_anchors=False)
    md, receipt = run_probe_report(model, model.vocab, dcfg, seqs, args.tag)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        out.with_suffix(".json").write_text(json.dumps(receipt, indent=1, default=float))
    return receipt


if __name__ == "__main__":
    main()
