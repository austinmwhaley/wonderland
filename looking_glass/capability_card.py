"""Model-free capability card (SPEC §20.4): decide PASS-able vs NO_SIGNAL BEFORE
grading the model.

For a categorical next-event target, the honest question is not "can the state
predict the next event?" (a 1-gram usually can) but "is there structure BEYOND the
1-gram to recover?". This computes, from the raw stream only, the incremental
predictability of the next event given a longer context (last k events) over and
above the previous event (1-gram). If the longer context adds nothing, the 1-gram
IS the ceiling and the capability is NO_SIGNAL — no model can be required to beat it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_ALPHA = 1.0  # add-alpha smoothing


def _pairs(rows, k):
    ctx1, ctxk, nxt = [], [], []
    for r in rows:
        et = np.asarray(r["et"], int)
        co = np.asarray(r["co"])
        for t in range(k, len(et)):
            if co[t] > 0:
                continue
            ctx1.append(int(et[t - 1]))
            ctxk.append(tuple(int(x) for x in et[t - k : t]))
            nxt.append(int(et[t]))
    return ctx1, ctxk, np.asarray(nxt, int)


def _ce_fold(ctx_tr, y_tr, ctx_va, y_va, n_classes):
    tab: dict = defaultdict(lambda: np.full(n_classes, _ALPHA))
    for c, y in zip(ctx_tr, y_tr, strict=True):
        tab[c][y] += 1.0
    marg = np.full(n_classes, _ALPHA)
    for y in y_tr:
        marg[y] += 1.0
    marg /= marg.sum()
    tot = 0.0
    for c, y in zip(ctx_va, y_va, strict=True):
        row = tab.get(c)
        p = row[y] / row.sum() if row is not None else marg[y]
        tot += -np.log(max(p, 1e-12))
    return tot / max(len(y_va), 1)


def transition_carried(rows, n_classes, k=3, folds=5, seed=0, z=2.0):
    """(carried, receipt). carried True iff a k-event context predicts the next
    event materially better than the 1-gram (target CARRIED beyond the 1-gram);
    False => the 1-gram is the ceiling => transition capability is NO_SIGNAL."""
    ctx1, ctxk, nxt = _pairs(rows, k)
    if len(nxt) < 200:
        return False, {"reason": "too few pairs"}
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(nxt))
    diffs = []
    for f in range(folds):
        tr, va = fold != f, fold == f
        ce1 = _ce_fold(
            [ctx1[i] for i in np.flatnonzero(tr)],
            nxt[tr],
            [ctx1[i] for i in np.flatnonzero(va)],
            nxt[va],
            n_classes,
        )
        cek = _ce_fold(
            [ctxk[i] for i in np.flatnonzero(tr)],
            nxt[tr],
            [ctxk[i] for i in np.flatnonzero(va)],
            nxt[va],
            n_classes,
        )
        diffs.append(ce1 - cek)
    diffs = np.asarray(diffs)
    mu = float(diffs.mean())
    se = float(diffs.std() / np.sqrt(len(diffs)))
    return bool(mu > z * se), {"ce_1gram_minus_ctx": round(mu, 4), "se": round(se, 4), "k": k}
