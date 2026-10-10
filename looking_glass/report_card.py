"""Layer-B report card (SPEC 0.9.4 §1.2 / §9).

One readable "report card" for a given event stream, covering ALL Layer-B
(looking_glass) checks: capability skills, manifold health, the intrinsic
"layer before", and RL-readiness. The card is GREEN only when every capability
the stream carries PASSES (or is certified NO_SIGNAL) and every manifold /
intrinsic / RL check passes — the SPEC no-FAIL verdict law.

Run:
  python3 -m looking_glass.report_card --db <stream> --as-of <date> \
      --ckpt-dir <dir> --tag <tag> --stream <name> [--sample-a N] [--out card.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from looking_glass.cfm_data import (
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
    draw_sample,
)
from looking_glass.cfm_training import _collate
from looking_glass.probes import freeze_encoder

SEND = {"email_send", "sms_send", "push_send"}
N_FOLDS = 5


def _folds(customers, n=N_FOLDS, seed=1):
    cs = sorted(set(np.asarray(customers).tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cs)
    return np.array_split(np.asarray(cs, dtype=object), n)


def _rl_checks(model, vocab, seqs, horizon=10, seed=0):
    """RL-relevant capability checks ON THE FROZEN EMBEDDING (folded into section
    A): (1) action representation, (2) outcome sufficiency, (3) action x state.
    Returns rows with a SPEC verdict (PASS/FAIL/UNCERTAIN)."""
    Z, A, OUT, S = [], [], [], []
    for i in range(0, len(seqs), 128):
        chunk = seqs[i : i + 128]
        t = _collate(chunk, vocab, model._dev())
        with torch.no_grad():
            y, _ = model.encode(t)
        y = y.cpu().numpy()
        m = t["mask"].cpu().numpy()
        for b in range(len(chunk)):
            L = int(m[b].sum())
            et = [str(x) for x in chunk[b]["event_type"]]
            for j in range(L - 1):
                Z.append(y[b, j])
                A.append(et[j + 1] if et[j + 1] in SEND else "none")
                OUT.append(
                    1.0 if any(e == "order_placed" for e in et[j + 1 : j + 1 + horizon]) else 0.0
                )
                S.append(chunk[b]["customer"])
    Z = np.asarray(Z)
    A = np.asarray(A)
    OUT = np.asarray(OUT)
    S = np.asarray(S)
    if len(Z) > 2500:
        sub = np.random.default_rng(0).choice(len(Z), 2500, replace=False)
        Z, A, OUT, S = Z[sub], A[sub], OUT[sub], S[sub]
    fof = {c: fi for fi, f in enumerate(_folds(S, seed=seed)) for c in f}
    fold = np.array([fof[c] for c in S])
    rows = []

    def _st(score, base, se, better_low=True):
        d = (base - score) if better_low else (score - base)  # >0 = better than baseline
        if d > 2 * se:
            return "PASS"
        if d < -2 * se:
            return "FAIL"
        return "UNCERTAIN"

    # (1) action representation: z_t -> next action channel (CE vs unigram)
    ce, marg = [], []
    for fi in range(N_FOLDS):
        tr, va = fold != fi, fold == fi
        if not tr.any() or not va.any():
            continue
        clf = LogisticRegression(max_iter=80, solver="lbfgs").fit(Z[tr], A[tr])
        P = clf.predict_proba(Z[va])
        cls = list(clf.classes_)
        idx = {c: i for i, c in enumerate(cls)}
        ce.append(
            np.mean([-np.log(max(P[i, idx.get(t_, 0)], 1e-12)) for i, t_ in enumerate(A[va])])
        )
        cnt = np.bincount([list(cls).index(x) for x in A[tr]], minlength=len(cls)) + 0.5
        p = cnt / cnt.sum()
        marg.append(np.mean([-np.log(p[list(cls).index(t_)]) for t_ in A[va]]))
    ce_m, marg_m = float(np.mean(ce)), float(np.mean(marg))
    se = float(np.std(ce) / np.sqrt(max(len(ce), 1)))
    rows.append(
        (
            "rl: action representation (z_t->action)",
            f"CE {ce_m:.4f} vs unigram {marg_m:.4f}",
            _st(ce_m, marg_m, se),
        )
    )

    # (2) outcome sufficiency: z_t -> order within horizon (AUROC vs 0.5)
    auc = []
    for fi in range(N_FOLDS):
        tr, va = fold != fi, fold == fi
        if not tr.any() or not va.any() or 0 in (OUT[va].mean(), 1 - OUT[va].mean()):
            continue
        clf = LogisticRegression(max_iter=80, solver="lbfgs").fit(Z[tr], OUT[tr])
        auc.append(roc_auc_score(OUT[va], clf.predict_proba(Z[va])[:, 1]))
    a = float(np.mean(auc))
    se = float(np.std(auc) / np.sqrt(max(len(auc), 1)))
    rows.append(
        (
            f"rl: outcome sufficiency (z_t->order in {horizon})",
            f"AUROC {a:.4f}",
            _st(a, 0.5, se, better_low=False),
        )
    )

    # (3) action x state: within-action AUROC (who responds). Average per fold
    # (over action contexts) so the SE reflects fold noise, not the 4-action spread.
    per_fold = {fi: [] for fi in range(N_FOLDS)}
    for act in ("email_send", "push_send", "sms_send", "none"):
        m_ = A == act
        if m_.sum() < 200 or not 0 < OUT[m_].mean() < 1:
            continue
        for fi in range(N_FOLDS):
            tr, va = (fold != fi) & m_, (fold == fi) & m_
            if not tr.any() or not va.any() or 0 in (OUT[va].mean(), 1 - OUT[va].mean()):
                continue
            clf = LogisticRegression(max_iter=80, solver="lbfgs").fit(Z[tr], OUT[tr])
            per_fold[fi].append(roc_auc_score(OUT[va], clf.predict_proba(Z[va])[:, 1]))
    fmeans = [float(np.mean(v)) for v in per_fold.values() if v]
    wa = float(np.mean(fmeans)) if fmeans else 0.5
    se = float(np.std(fmeans) / np.sqrt(len(fmeans))) if fmeans else 0.0
    rows.append(
        (
            "rl: action x state (who responds)",
            f"mean within-action AUROC {wa:.4f}",
            _st(wa, 0.5, se, better_low=False),
        )
    )
    return rows


def _row(check, achieved, ok):
    return {"check": check, "achieved": achieved, "ok": bool(ok)}


def _latest_receipt(out_dir, kind, tag):
    import glob
    import json as _json

    if not out_dir:
        return None
    fs = sorted(glob.glob(str(Path(out_dir) / f"{kind}_{tag.replace('.', '_')}_*.json")))
    return _json.loads(Path(fs[-1]).read_text()) if fs else None


def build_report_card(model, vocab, cfg, seqs, stream, tag, out_dir=None, seed=0, refresh=False):
    freeze_encoder(model)
    from looking_glass.intrinsic import evaluate as intrinsic_eval
    from looking_glass.portfolio import evaluate as portfolio_eval

    # reuse receipts the training run already wrote (a re-card is then ~seconds)
    port = None if refresh else _latest_receipt(out_dir, "portfolio", tag)
    if port is None:
        port = portfolio_eval(model, vocab, cfg, seqs, seed=seed, out_dir=out_dir, tag=tag)
    intrin = None if refresh else _latest_receipt(out_dir, "intrinsic", tag)
    if intrin is None:
        intrin = intrinsic_eval(model, vocab, cfg, seqs, seed=seed, out_dir=out_dir, tag=tag)
    rl = _rl_checks(model, vocab, seqs, seed=seed)

    # capability verdicts (SPEC §1.2): PASS / NO_SIGNAL / FAIL / UNCERTAIN
    ident = port.get("identifiability", {})
    caps, fails, nosig, unc = [], [], [], []
    for o in port.get("objectives", []):
        skill, se = o.get("skill", 0.0), o.get("skill_se", 0.0)
        ic = ident.get(o["objective"], {})
        ce, cse = ic.get("ceiling"), ic.get("se")
        # NO_SIGNAL only when a ceiling was MEASURED and can't beat the null.
        # No ceiling for an objective (e.g. agg/sf) => judge it on the skill.
        no_signal = ce is not None and ce <= 2.0 * float(cse or 0.0)
        if no_signal:
            st = "NO_SIGNAL"
        elif skill > 2.0 * se:
            st = "PASS"
        elif skill < -2.0 * se:
            st = "FAIL"
        else:
            st = "UNCERTAIN"  # within noise of the threshold; needs more data
        caps.append((o["objective"], f"skill {skill:+.4f} ± {se:.4f}", st))
        if st == "FAIL":
            fails.append(o["objective"])
        elif st == "NO_SIGNAL":
            nosig.append(o["objective"])
        elif st == "UNCERTAIN":
            unc.append(o["objective"])
    # RL-relevant capability checks (frozen embedding) — folded into section A
    for name, meas, st in rl:
        caps.append((name, meas, st))
        if st == "FAIL":
            fails.append(name)
        elif st == "UNCERTAIN":
            unc.append(name)
    # manifold rows (geometry / stability / conservation / canary) from portfolio
    manifold = [r for r in port["rows"] if not r["check"].startswith("portfolio:")]
    for r in manifold:
        if not r["ok"]:
            fails.append(r["check"])
    intrinsic_rows = intrin["rows"]
    for r in intrinsic_rows:
        if not r["ok"]:
            fails.append(r["check"])
    green = len(fails) == 0 and len(unc) == 0
    lines = [
        f"# Layer-B Report Card — {stream}",
        f"encoder `{tag}` · {len(seqs)} sequences · B-only (encoder frozen)",
        "",
        f"## VERDICT: {'GREEN ✅' if green else 'NOT GREEN ❌'}",
    ]
    if fails:
        lines.append(f"BLOCKERS (FAIL): {fails}")
    if nosig:
        lines.append(f"NO_SIGNAL (allowed): {nosig}")
    if unc:
        lines.append(f"UNCERTAIN (needs more data): {unc}")
    lines += [
        "",
        "## A. Capability checks (frozen z_t vs model-free baselines)",
        "",
        "| capability | measured | verdict |",
        "|---|---|---|",
    ]
    for name, meas, st in caps:
        lines.append(f"| {name} | {meas} | {st} |")
    lines += [
        "",
        "## B. Manifold health (geometry / stability / conservation / canary)",
        "",
        "| check | achieved | verdict |",
        "|---|---|---|",
    ]
    for r in manifold:
        lines.append(f"| {r['check']} | {r['achieved']} | {'PASS' if r['ok'] else 'FAIL'} |")
    lines += [
        "",
        "## C. Intrinsic properties (the layer before)",
        "",
        "| check | achieved | verdict |",
        "|---|---|---|",
    ]
    for r in intrinsic_rows:
        lines.append(f"| {r['check']} | {r['achieved']} | {'PASS' if r['ok'] else 'FAIL'} |")
    card = "\n".join(lines) + "\n"
    receipt = {
        "stream": stream,
        "tag": tag,
        "green": green,
        "fails": fails,
        "no_signal": nosig,
        "capabilities": caps,
        "manifold": manifold,
        "intrinsic": intrinsic_rows,
        "rl": rl,
    }
    return card, receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--sample-a", type=int, default=20000, dest="sample_a")
    ap.add_argument("--max-seqs", type=int, default=None, dest="max_seqs")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_state import load_frozen_encoder

    model, _ = load_frozen_encoder(args.tag, args.ckpt_dir)
    dcfg = CFMConfig()
    dcfg.db, dcfg.as_of = args.db, args.as_of
    dcfg.sample_customers, dcfg.sample_a_customers = args.customers, args.sample_a
    # load the checkpoint's own resolved config/receipt so per-stream gates use the
    # right capacity (whitening pr_after) and stability flag.
    import json as _json

    reg = Path(args.ckpt_dir) / f"registry_{args.tag.replace('.', '_')}.json"
    if reg.exists():
        meta = _json.loads(reg.read_text())
        dcfg._whiten_pr_after = (meta.get("whiten") or {}).get("pr_after")
        dcfg._stable = not bool((meta.get("governor") or {}).get("diverged", False))
        for k in (
            "n_experts",
            "dim",
            "seq_len",
            "batch",
            "state_half_life_days",
            "agg_horizons_days",
        ):
            v = (meta.get("config") or {}).get(k)
            if v is not None:
                setattr(dcfg, k, v)
    df = _read_stream(dcfg)
    keys = _customer_keys(df, dcfg)
    split = assign_split(keys, dcfg)
    ak = draw_sample(keys, split, "A", args.sample_a, dcfg.split_seed)
    seqs = build_sequences(df, ak, dcfg, split, with_anchors=False)
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    stream = args.stream or Path(args.db).parent.name
    card, receipt = build_report_card(
        model, model.vocab, dcfg, seqs, stream, args.tag, out_dir=args.ckpt_dir
    )
    print(card)
    if args.out:
        Path(args.out).write_text(card)
    return receipt


if __name__ == "__main__":
    main()
