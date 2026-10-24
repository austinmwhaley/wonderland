"""Iteration-3 measurement: the self-supervised objective battery.

Adapter of looking_glass/probes.py OBJECTIVES (next, dt, jepa, mask, value,
entity, order, agg, query, sf) to the demo encoder's per-step states Hs.

Protocol (mirrors probes.py): grouped 5-fold by customer; Linear head; per-target
targets; model-free baselines (unigram / historical mean / previous value);
PASS only if the probe beats the stronger baseline by > 2*SE.

Writes docs/report_cards/experiments/encoder_objectives.json
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/austin-whaley/Documents/wonderland")
sys.path.insert(0, "/home/austin-whaley/Documents/wonderland/scripts/demo")
import torch  # noqa: E402
import auto_encoder as AE  # noqa: E402
from sklearn.linear_model import Ridge, LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import r2_score, accuracy_score  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/t0_loop/relational_v3"
OUT = Path("/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_objectives.json")
H = 4  # horizon (steps) for agg/jepa/sf

E, _, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
torch.cuda.empty_cache()
_, A, keys, seqs, vs2, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = AE.load_data(DATA)
AE.AS_OF_TS = as_of_ts
n_ev = vs2["et"]
_cand = [j for j in range(len(catspec)) if j not in vs2["id_idx"]]
_uc = {_j: len(set(np.concatenate([seqs[k][1][:, _j] for k in hold[:300]]).tolist())) for _j in _cand} if _cand else {}
_ok = [j for j in _cand if 2 <= _uc[j] <= 50]
_ent_j = max(_ok, key=lambda j: _uc[j]) if _ok else (min(_uc, key=lambda j: _uc[j]) if _uc else 0)
id_idx = _ent_j  # varying, low-cardinality categorical for the entity probe

# per-step states for holdout (chunked)
Z, Ls, cust = [], [], []
with torch.no_grad():
    for i in range(0, len(hold), 48):
        bk = hold[i : i + 48]
        fc = [len(seqs[k][5]) for k in bk]
        fq = [(as_of_ts - seqs[k][5][-1]) / 86400.0 for k in bk]
        t, _p, qg, lens, _f = AE.collate(seqs, bk, fc, fq, vs2, False, None, None)
        Hs, Eh, w, *_r = model(t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"])
        Hs = Hs.cpu().numpy()
        L = lens.cpu().numpy().astype(int)
        for b, k in enumerate(bk):
            Z.append(Hs[b])
            Ls.append(L[b])
            cust.append(k)

Xall, Y, PREVID, PREVVAL, S = {n: [] for n in ("next", "dt", "jepa", "mask", "value", "entity", "order", "agg", "query", "sf")}, {}, {}, {}, {}
for name in Xall:
    Y[name], PREVID[name], PREVVAL[name], S[name] = [], [], [], []

_jp = {}
def jproj(dim):
    if dim not in _jp:
        _jp[dim] = np.random.default_rng(7).normal(size=(dim, 16))
    return _jp[dim]

for z, L, c in zip(Z, Ls, cust):
    etk, gck, gnk, tmk, gpk, tt, grk = seqs[c]
    val = np.log1p(np.abs(grk).sum(1))
    for t in range(0, L - 1):
        zt = z[t]
        # next / order share the next-event target; order uses the final state
        Xall["next"].append(zt); Y["next"].append(etk[t + 1]); PREVID["next"].append(etk[t]); PREVVAL["next"].append(0.0); S["next"].append(c)
        Xall["dt"].append(zt); Y["dt"].append(gpk[t + 1]); PREVID["dt"].append(-1); PREVVAL["dt"].append(gpk[t]); S["dt"].append(c)
        Xall["value"].append(zt); Y["value"].append(val[t + 1]); PREVID["value"].append(-1); PREVVAL["value"].append(val[t]); S["value"].append(c)
        Xall["entity"].append(zt); Y["entity"].append(gck[t + 1, id_idx]); PREVID["entity"].append(etk[t]); PREVVAL["entity"].append(0.0); S["entity"].append(c)
        Xall["mask"].append(zt); Y["mask"].append(etk[t]); PREVID["mask"].append(etk[t - 1] if t >= 1 else -1); PREVVAL["mask"].append(0.0); S["mask"].append(c)
        Xall["query"].append(zt); last = np.where(etk[: t + 1] == 1)[0]; Y["query"].append(np.log1p(t - last[-1]) if len(last) else np.log1p(0.0)); PREVID["query"].append(-1); PREVVAL["query"].append(0.0); S["query"].append(c)
        hi = min(t + 1 + H, L); cnt = hi - (t + 1)
        vs_ = float(np.abs(grk[t + 1 : hi]).sum()) if hi > t + 1 else 0.0
        Xall["agg"].append(zt); Y["agg"].append([np.log1p(cnt), np.log1p(vs_)]); PREVID["agg"].append(-1); PREVVAL["agg"].append(0.0); S["agg"].append(c)
        fut = z[t + 1 : hi]
        Xall["jepa"].append(zt); Y["jepa"].append(((fut.mean(0) if len(fut) else zt) @ jproj(z.shape[1])).tolist()); PREVID["jepa"].append(-1); PREVVAL["jepa"].append(0.0); S["jepa"].append(c)
        # sf: decayed future successor features [value, event one-hot]
        gam = 0.5
        R = np.zeros(n_ev)
        for u in range(t + 1, hi):
            R[etk[u]] += gam ** (u - t)
            R[0] += gam ** (u - t) * val[u]
        Xall["sf"].append(zt); Y["sf"].append(R.tolist()); PREVID["sf"].append(-1); PREVVAL["sf"].append(0.0); S["sf"].append(c)
    # order: sequence-level (final state) -> terminal event
    Xall["order"].append(z[L - 1]); Y["order"].append(etk[L - 1]); PREVID["order"].append(etk[0]); PREVVAL["order"].append(0.0); S["order"].append(c)


def fit_eval(name, X, y, prev_id, prev_val, folds):
    cat = name in ("next", "mask", "entity", "order")
    multi = name in ("agg", "jepa", "sf")
    X = np.asarray(X, float)
    y = np.asarray(y)
    vals, naive = [], []
    from collections import Counter, defaultdict
    for tr, te in folds:
        if cat:
            mc = Counter(y[tr]).most_common(1)[0][0]
            uni = accuracy_score(y[te], np.full(len(te), mc))
            # 1-gram predictor: most frequent target conditioned on previous id
            g = defaultdict(Counter)
            for yy, pp in zip(y[tr], prev_id[tr]):
                g[pp][yy] += 1
            pred1 = np.array([(g[pp].most_common(1)[0][0] if pp in g else mc) for pp in prev_id[te]])
            one = accuracy_score(y[te], pred1)
            naive.append(max(uni, one))
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
            try:
                m.fit(X[tr], y[tr]); vals.append(accuracy_score(y[te], m.predict(X[te])))
            except Exception:
                vals.append(0.0)
        else:
            yy = y if multi else y.reshape(-1, 1)
            base = yy[tr].mean(0)
            naive.append(r2_score(yy[te], np.tile(base, (len(te), 1)), multioutput="uniform_average"))
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(X[tr], yy[tr]); vals.append(r2_score(yy[te], m.predict(X[te]), multioutput="uniform_average"))
    return float(np.mean(vals)), float(np.std(vals) / max(np.sqrt(len(vals)), 1)), float(np.mean(naive))


out = {}
CAP = 40000
for name in Xall:
    X = np.asarray(Xall[name], float)
    y = np.asarray(Y[name])
    Sarr = np.asarray(S[name])
    if len(X) > CAP:
        _ix = np.random.default_rng(0).choice(len(X), CAP, replace=False)
        X, y, Sarr = X[_ix], y[_ix], Sarr[_ix]
    folds = list(GroupKFold(5).split(X, y, Sarr))
    skill, se, naive = fit_eval(name, X, y, np.asarray(PREVID[name]), np.asarray(PREVVAL[name]), folds)
    deg = bool(name in ("next", "mask", "entity", "order") and naive >= 0.999)
    out[name] = {"skill": skill, "se": se, "naive": naive,
                 "beats_naive_2se": bool((skill - naive) > 2 * se), "degenerate": bool(deg)}
    print(f"[obj] {name:7s} skill {skill:+.4f} ± {se:.4f} vs naive {naive:+.4f} -> {out[name]['beats_naive_2se']}", flush=True)

OUT.write_text(json.dumps({"config": {"data": DATA, "seed": 0, "epochs": 12, "horizon": H}, "objectives": out}, indent=1))
print("WROTE", OUT)
