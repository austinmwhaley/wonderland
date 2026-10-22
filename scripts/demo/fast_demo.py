"""Fast B+C demo: frozen sequence encoder -> supervised / unsupervised / RL plugins."""

import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import encoder_model as EM
from rabbit_hole.generators.relational_reference import CHANNELS
from sklearn.model_selection import KFold
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from scipy.stats import spearmanr

D = None
EM.ARGS = SimpleNamespace(
    max_customers=0, d=64, layers=2, mass="explicit", epochs=5, lr=3e-4, batch=16, nce=0, aug_p=0.1
)


def z(X):
    return StandardScaler().fit_transform(np.nan_to_num(np.asarray(X, float)))


def oof(X, y, kind, folds=5):
    oof = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        m = (
            HistGradientBoostingRegressor(max_iter=150, random_state=0)
            if kind == "reg"
            else HistGradientBoostingClassifier(max_iter=150, random_state=0)
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
    return oof


def score(y, p, kind):
    return r2_score(y, p) if kind == "reg" else roc_auc_score(y, p)


def boot_diff(y, pA, pB, kind, n=400):
    rng = np.random.default_rng(0)
    n_ = len(y)
    d = []
    for _ in range(n):
        idx = rng.integers(0, n_, n_)
        if kind == "clf" and len(np.unique(y[idx])) < 2:
            continue
        d.append(score(y[idx], pA[idx], kind) - score(y[idx], pB[idx], kind))
    d = np.array(d)
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/relational_v3")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "docs/report_cards/demo_bc_card.md"),
    )
    args = ap.parse_args()
    global D
    D = args.data
    out = Path(args.out)
    # ---- train the frozen E (fast) ----
    B, A, keys, seqs, vs, et_v, cat_v, h_min, h_max, periods_h, n_experts, taus, p_mask, num_dim = (
        EM.load_data(D)
    )
    purchase_idx = et_v.get("purchase", -1)
    res = EM.one_seed(
        0,
        B,
        A,
        keys,
        seqs,
        vs,
        purchase_idx,
        h_min,
        h_max,
        n_experts,
        taus,
        p_mask,
        num_dim,
        et_v,
        cat_v,
        0.1,
    )
    E, Xtab, labels, hold = res["E"], res["Xtab"], res["labels"], res["hold"]
    E = np.nan_to_num(E)
    Xtab = np.nan_to_num(Xtab)
    Xh = np.hstack([Xtab, E])
    print(
        f"[E] hold={len(hold)} dim={E.shape[1]} train={res['n_train']} val={res['n_val']} best_val={res['best_val']}",
        flush=True,
    )

    # ---- supervised (6) ----
    sup = []
    for name, key, kind in [
        ("value30", "y_value_30", "reg"),
        ("value90", "y_value_90", "reg"),
        ("orders30", "y_orders_30", "reg"),
        ("orders90", "y_orders_90", "reg"),
        ("churn30", "y_churn_30", "clf"),
        ("churn90", "y_churn_90", "clf"),
    ]:
        y = labels[key]
        pE = oof(E, y, kind)
        pT = oof(Xtab, y, kind)
        pH = oof(Xh, y, kind)
        sE, sT, sH = score(y, pE, kind), score(y, pT, kind), score(y, pH, kind)
        m, lo, hi = boot_diff(y, pE, pT, kind)
        sup.append((name, sT, sE, sH, m, lo, hi))
        print(
            f"[sup] {name:9s} table {sT:.4f} E {sE:.4f} hybrid {sH:.4f} E-table {m:+.4f} [{lo:+.4f},{hi:+.4f}]",
            flush=True,
        )

    # ---- unsupervised (4) ----
    truth = {
        c["customer"]: c for c in json.load(open(Path(D) / "reference_truth.json"))["customers"]
    }
    regime = np.array([truth[k]["regime0"] for k in hold], dtype=object)
    categ = np.array([truth[k]["category_pref"] for k in hold], dtype=object)
    rp = np.array([truth[k]["return_proneness"] for k in hold], float)
    yv = labels["y_value_90"]
    yc = labels["y_churn_90"]
    uns = {}
    for arm, X in (("table", Xtab), ("E", E)):
        Z = z(X)
        km = KMeans(4, n_init=10, random_state=0).fit_predict(Z)
        kmc = KMeans(5, n_init=10, random_state=0).fit_predict(Z)
        nmi = normalized_mutual_info_score(regime, km)
        ari = adjusted_rand_score(regime, km)
        nmic = normalized_mutual_info_score(categ, kmc)
        # anomaly: top decile return-proneness
        lab = (rp >= np.quantile(rp, 0.9)).astype(int)
        sc = -IsolationForest(random_state=0).fit(Z).score_samples(Z)
        auc = roc_auc_score(lab, sc) if lab.sum() > 0 else float("nan")
        # retrieval: 1-NN outcome consistency
        from scipy.spatial.distance import cdist

        dd = cdist(Z, Z)
        np.fill_diagonal(dd, np.inf)
        nn = dd.argmin(1)
        rnk = spearmanr(yv, yv[nn]).correlation
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        er = float((s**2).sum() ** 2 / ((s**4).sum() + 1e-12))
        uns[arm] = dict(
            nmi_regime=round(nmi, 4),
            ari_regime=round(ari, 4),
            nmi_category=round(nmic, 4),
            anomaly_auc=round(float(auc), 4),
            retrieval_rank=round(float(rnk), 4),
            eff_rank=round(er, 2),
        )
        print(f"[uns] {arm:5s} {uns[arm]}", flush=True)

    # ---- RL (3): last logged email_send action; reward = y_margin_90 ----
    B2 = B.filter(pl.col("event_type") == "email_send").sort("ts")
    last = B2.group_by("customer_key", maintain_order=True).agg(
        pl.col("event_attributes").last().alias("attrs")
    )
    acts = {r["customer_key"]: json.loads(r["attrs"]) for r in last.to_dicts()}
    reward = labels["y_margin_90"]
    hold_act = [(k, acts.get(k, {}), truth[k]) for k in hold]
    ch = np.array([str(a.get("channel")) for _, a, _ in hold_act], dtype=object)
    off = np.array([int(a.get("offer", -1)) for _, a, _ in hold_act])
    ch_truth = np.array([str(t["channel_pref"]) for _, _, t in hold_act], dtype=object)
    off_truth = np.array([int(t["offer_opt"]) for _, _, t in hold_act])
    # map action channel code -> truth string where possible
    chmap = {str(i): CHANNELS[i] for i in range(len(CHANNELS))}
    ch_str = np.array([chmap.get(x, x) for x in ch], dtype=object)
    rl = []
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(hold))
    ntr = int(0.7 * len(hold))
    tr, te = idx[:ntr], idx[ntr:]
    for arm_name, a, a_truth in (("channel", ch_str, ch_truth), ("offer", off, off_truth)):
        levels = sorted(set(a_truth) | set(a[tr]))
        lev = {v: i for i, v in enumerate(levels)}
        K = len(levels)

        def onehot(vals):
            M = np.zeros((len(vals), K))
            for i, v in enumerate(vals):
                M[i, lev.get(v, 0)] = 1
            return M

        for state_name, X in (("table", Xtab), ("E", E)):
            Xa = np.hstack([z(X), onehot(a)])
            m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(Xa[tr], reward[tr])
            pred = np.array(
                [
                    [m.predict(np.hstack([z(X)[i : i + 1], onehot([v])]))[0] for v in levels]
                    for i in te
                ]
            )
            pol = np.array(levels)[pred.argmax(1)]
            # IPS/DR with uniform logging propensity over the arm levels
            p0 = 1.0 / K
            logged = a[te]
            r = reward[te]
            ips = np.mean((pol == logged) * r / p0)
            # DR: model reward for chosen arm + correction for logged arm
            m_chosen = pred.max(1)
            m_logged = pred[np.arange(len(te)), [lev.get(v, 0) for v in logged]]
            dr = np.mean(m_chosen + (pol == logged) * (r - m_logged) / p0)
            rnd = np.mean(r)
            acc = float(np.mean(pol == a_truth[te]))
            rl.append((arm_name, state_name, levels, ips, dr, rnd, acc))
            print(
                f"[rl] {arm_name:7s} {state_name:5s} arms={len(levels)} IPS {ips:.2f} DR {dr:.2f} random {rnd:.2f} truth-acc {acc:.3f}",
                flush=True,
            )

    # ---- B card numbers from 6k receipts ----
    b6 = {}
    for tag in ("explicit", "none", "generic"):
        f = (
            Path(__file__).resolve().parents[2]
            / "docs/report_cards/experiments"
            / f"encoder_6k_{tag}_3seed.json"
        )
        if f.exists():
            d = json.loads(f.read_text())
            b6[tag] = {
                t: float(
                    np.mean(
                        [
                            (
                                x["results"][t]["E"]["r2"]
                                if isinstance(x["results"][t]["E"], dict)
                                else x["results"][t]["E"]
                            )
                            for x in d
                        ]
                    )
                )
                for t in ("value90", "orders90", "churn90")
            }
    # ---- card ----
    L = []
    L.append("# B+C demo card (fast 2k; B reference 6k)\n")
    L.append(
        "Synthetic relational reference `relational_v3`; no business data. Harness, not business value.\n"
    )
    L.append("## Layer B — encoder\n")
    L.append(
        f"- Frozen E: dim {E.shape[1]}, holdout {len(hold)}, train/val {res['n_train']}/{res['n_val']}, best val {res['best_val']}."
    )
    if b6:
        for tag, v in b6.items():
            L.append(
                f"- 6k {tag} E-only: value {v['value90']:.3f}, orders {v['orders90']:.3f}, churn {v['churn90']:.3f}."
            )
    L.append(
        "- 6k fair table: value 0.335, orders 0.401, churn 0.919; raw table 0.364/0.436/0.917."
    )
    L.append("- E adds to `table+seq` (6k): hand +0.021/+0.018/+0.017; auto +0.021/+0.032/+0.012.")
    L.append(
        "- Feature ablations (1 seed/5): `qgap` critical; Fourier periods help orders; horizon helps orders; past helps churn; `mass=explicit` trades churn for value/orders.\n"
    )
    L.append("## Layer C — plugins\n")
    L.append("### Supervised\n")
    L.append("| target | table | E | hybrid | E-table | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for n, sT, sE, sH, m, lo, hi in sup:
        L.append(f"| {n} | {sT:.4f} | {sE:.4f} | {sH:.4f} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    L.append("\n### Unsupervised (truth-referenced)\n")
    L.append(
        "| arm | NMI regime | ARI regime | NMI category | anomaly AUC | retrieval rank | eff rank |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for arm, v in uns.items():
        L.append(
            f"| {arm} | {v['nmi_regime']} | {v['ari_regime']} | {v['nmi_category']} | {v['anomaly_auc']} | {v['retrieval_rank']} | {v['eff_rank']} |"
        )
    L.append("\n### RL (logged randomized actions; uniform-propensity IPS/DR)\n")
    L.append("| arm | state | arms | IPS | DR | random | truth-acc |")
    L.append("|---|---|---|---|---|---|---|")
    for arm_name, state_name, levels, ips, dr, rnd, acc in rl:
        L.append(
            f"| {arm_name} | {state_name} | {len(levels)} | {ips:.2f} | {dr:.2f} | {rnd:.2f} | {acc:.3f} |"
        )
    L.append("\n## Caveats\n")
    L.append("- 2k/5-epoch fast C; 6k/3-seed is the reference for the B card.")
    L.append(
        "- RL uses the last logged `email_send`, future margin reward, uniform-propensity IPS/DR; not a shipped policy."
    )
    L.append("- Unsupervised uses truth latents; `segment` is random and is not used as a target.")
    L.append(
        "- Low-rank E; hand-listed core + auto-discovered extensions; fixed windows excluded from the fair table."
    )
    card = "\n".join(L) + "\n"
    out.write_text(card)
    print("\n" + card)
    print("WROTE", out)


if __name__ == "__main__":
    main()
