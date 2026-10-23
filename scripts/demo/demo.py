"""One-command demo: data -> frozen auto E -> B axis probes -> B+C vs XGBoost -> cards."""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_encoder as AE
import fast_demo as FD
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb

def xgb_oof(X, y, kind):
    o = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = (xgb.XGBRegressor(device='cuda', tree_method='hist', n_estimators=300, max_depth=6, learning_rate=0.06)
             if kind == 'reg' else xgb.XGBClassifier(device='cuda', tree_method='hist', n_estimators=300, max_depth=6, learning_rate=0.06, eval_metric='logloss'))
        m.fit(X[tr], y[tr]); o[te] = m.predict(X[te]) if kind == 'reg' else m.predict_proba(X[te])[:, 1]
    return o

def sc(X, y, kind):
    p = xgb_oof(X, y, kind); return float(r2_score(y, p) if kind == 'reg' else roc_auc_score(y, p))

def knn_purity(Z, lab, k=10):
    D = cdist(Z, Z); np.fill_diagonal(D, np.inf); idx = np.argpartition(D, k, axis=1)[:, :k]
    return float(np.mean(lab[idx] == lab[:, None]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', default='docs/report_cards')
    ap.add_argument('--epochs', type=int, default=5); ap.add_argument('--seeds', default='0'); ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=2)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    seeds = [int(x) for x in a.seeds.split(',')]; runs = []
    for seed in seeds:
        E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(a.data, seed=seed, epochs=a.epochs, d=a.d, layers=a.layers, lr=1e-3)
        E = np.nan_to_num(E); Xtab = np.nan_to_num(Xtab); BLK = np.nan_to_num(BLK); Xms = np.hstack([E, BLK])
        _, A, keys, seqs, vs2, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = AE.load_data(a.data)
        truth = {c['customer']: c for c in json.load(open(Path(a.data) / 'reference_truth.json'))['customers']}
        d = {'seed': seed, 'best': float(best)}
        # order/path shuffle (needs the model)
        if seed == seeds[0]:
            shuf = {}
            for k in keys:
                n = len(seqs[k][5]); perm = np.random.default_rng(abs(hash(k)) % 2**32).permutation(n)
                shuf[k] = tuple(np.asarray(x)[perm] for x in seqs[k])
            E2, B2 = AE.embed(model, shuf, hold, vs); Xms2 = np.hstack([np.nan_to_num(E2), np.nan_to_num(B2)])
        # identity
        nn = cdist(FD.z(E), FD.z(E)); np.fill_diagonal(nn, np.inf); nnidx = nn.argmin(1)
        d['retrieval'] = float(spearmanr(labels['y_value_90'], labels['y_value_90'][nnidx]).correlation)
        d['churn_nn'] = float(np.mean(labels['y_churn_90'] == labels['y_churn_90'][nnidx]))
        d['eff_rank'] = float(np.linalg.svd(FD.z(E), compute_uv=False).var())
        # identity block slices (MB=3e*d, IB=nc*d)
        e = model.ssm[0].e; dd = model.d; nc = len(vs['gcat']); ib = BLK[:, 3*e*dd:3*e*dd+nc*dd]
        cat = np.array([truth[k]['category_pref'] for k in hold], dtype=object); ch = np.array([truth[k]['channel_pref'] for k in hold], dtype=object)
        d['identity_channel'] = knn_purity(FD.z(ib), ch); d['identity_category'] = knn_purity(FD.z(ib), cat)
        d['table_channel'] = knn_purity(FD.z(Xtab), ch); d['table_category'] = knn_purity(FD.z(Xtab), cat)
        del model; gc.collect(); torch.cuda.empty_cache()
        # supervised + shuffle scores on GPU XGBoost after freeing the encoder
        for t, kind in (('value90', 'reg'), ('orders90', 'reg'), ('churn90', 'clf')):
            y = labels['y_' + t.replace('90', '_90')]
            d[t] = {'E': sc(Xms, y, kind), 'table': sc(Xtab, y, kind)}
            if seed == seeds[0]: d.setdefault('shuffle', {})[t] = sc(Xms2, y, kind)
        runs.append(d); print(json.dumps(d), flush=True)
    # aggregate + artifact
    def mean(t, arm): return float(np.mean([r[t][arm] for r in runs]))
    res = {'n_seeds': len(seeds), 'runtime_s': round(time.time() - t0, 1), 'runs': runs,
           'agg': {t: {'E': mean(t, 'E'), 'table': mean(t, 'table')} for t in ('value90', 'orders90', 'churn90')}}
    res['hash'] = hashlib.sha256(json.dumps(res['agg'], sort_keys=True).encode()).hexdigest()[:12]
    (out / 'demo_run.json').write_text(json.dumps(res, indent=1))
    L = ['# B/C demo command output', '', f'seeds={seeds} epochs={a.epochs} runtime={res["runtime_s"]}s hash={res["hash"]}', '']
    L.append('| target | table (XGBoost GPU) | frozen E | margin |')
    L.append('|---|---|---|---|')
    for t in ('value90', 'orders90', 'churn90'):
        L.append(f'| {t} | {res["agg"][t]["table"]:.4f} | {res["agg"][t]["E"]:.4f} | {res["agg"][t]["E"]-res["agg"][t]["table"]:+.4f} |')
    L.append(''); L.append('| probe | E | table |'); L.append('|---|---|---|')
    L.append(f'| retrieval | {runs[0]["retrieval"]:.4f} | — |'); L.append(f'| churn-nn | {runs[0]["churn_nn"]:.4f} | — |')
    L.append(f'| identity channel | {runs[0]["identity_channel"]:.4f} | {runs[0]["table_channel"]:.4f} |')
    L.append(f'| identity category | {runs[0]["identity_category"]:.4f} | {runs[0]["table_category"]:.4f} |')
    if 'shuffle' in runs[0]:
        L.append(f'| shuffle value drop | {runs[0]["shuffle"]["value90"]-runs[0]["value90"]["E"]:+.4f} | 0 |')
    (out / 'demo_command_card.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L)); print('WROTE', out / 'demo_run.json')

if __name__ == '__main__':
    main()
