import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from white_queen import db
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate
from algorithms.deep.networks import QNetwork

cfg = dict(PRESETS['quick'])
env = make_env('cartpole', seed=999)
data = db.load_diet('/home/austin-whaley/wq/white_queen/data/white_queen_quick.db',
                    'novice_only')
iql = load_candidate('iql', env, data, cfg,
                     '/home/austin-whaley/wq/white_queen/verdicts/v15/iql_novice_only.pt')
P = iql.action_probs(data['obs'], temperature=1.0)
P2 = iql.action_probs(data['obs2'], temperature=1.0)
# live truth of THIS judged proxy (temperature=1.0), for context
print('context: candidate live argmax truth ~81; this soft proxy live ~52', flush=True)
N = len(data['obs'])
O = torch.as_tensor(data['obs'])
A = torch.as_tensor(data['act'], dtype=torch.long)
R = torch.as_tensor(data['rew']).unsqueeze(1)
O2 = torch.as_tensor(data['obs2'])
D = torch.as_tensor(data['done']).unsqueeze(1)
P2t = torch.as_tensor(P2)


def run(steps, mode, cadence=0, tau=0.0, lr=1e-3, batch=256, hidden=128):
    torch.manual_seed(0)
    q = QNetwork(4, hidden, 2)
    qt = QNetwork(4, hidden, 2)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for s in range(steps):
        i = torch.as_tensor(np.random.default_rng(s).integers(0, N, batch))
        with torch.no_grad():
            v2 = (P2t[i] * qt(O2[i])).sum(1, keepdim=True)
            tgt = R[i] + 0.99 * (1 - D[i]) * v2
        loss = F.mse_loss(q(O[i]).gather(1, A[i].unsqueeze(1)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == 'polyak':
            with torch.no_grad():
                for p, pt in zip(q.parameters(), qt.parameters()):
                    pt.mul_(1 - tau).add_(p, alpha=tau)
        elif mode == 'hard' and cadence and s % cadence == 0:
            qt.load_state_dict(q.state_dict())
    with torch.no_grad():
        v = (torch.as_tensor(P) * q(O)).sum(1)
        # mean V over episode starts
        ep = data['episode']
        starts = np.unique(ep, return_index=True)[1]
        return float(v[starts].mean())


for label, kw in [
    ('hard every 2242 (current)', dict(steps=40000, mode='hard', cadence=2242)),
    ('hard every 1000', dict(steps=40000, mode='hard', cadence=1000)),
    ('hard every 250', dict(steps=40000, mode='hard', cadence=250)),
    ('hard every 100', dict(steps=40000, mode='hard', cadence=100)),
    ('polyak 0.02', dict(steps=40000, mode='polyak', tau=0.02)),
    ('polyak 0.01', dict(steps=40000, mode='polyak', tau=0.01)),
    ('polyak 0.005', dict(steps=40000, mode='polyak', tau=0.005)),
]:
    try:
        print('  %-28s DM=%7.1f' % (label, run(**kw)), flush=True)
    except Exception as e:
        print('  %-28s ERROR %r' % (label, e), flush=True)
print('SWEEP DONE', flush=True)
