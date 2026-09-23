import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np
import torch
torch.set_num_threads(1)
print('cuda', torch.cuda.is_available(), flush=True)
from algorithms.deep.networks import QNetwork
import torch.nn.functional as F

# Clean toy: reward=1 every step, fixed horizon H, no termination, uniform
# policy. EXACT answer = sum_{t=0}^{H-1} gamma^t. No coverage issue at all.
g, H = 0.99, 100
truth = float(sum(g ** t for t in range(H)))
print('exact DM =', round(truth, 2), flush=True)
rng = np.random.default_rng(0)
N = 2000
obs = rng.normal(size=(N, 4)).astype(np.float32)
O = torch.as_tensor(obs)
R = torch.ones(N, 1)
O2 = torch.as_tensor(obs)
D = torch.zeros(N, 1)


def run(steps, lr=1e-3, hidden=128, target=True, sync_every=1000, batch=256):
    torch.manual_seed(0)
    q = QNetwork(4, hidden, 2)
    qt = QNetwork(4, hidden, 2)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for s in range(steps):
        i = torch.as_tensor(rng.integers(0, N, batch))
        with torch.no_grad():
            v2 = (0.5 * qt(O2[i])).sum(1, keepdim=True)
            tgt = R[i] + g * (1 - D[i]) * v2
        pred = q(O[i]).mean(1, keepdim=True)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if target and s % sync_every == 0:
            qt.load_state_dict(q.state_dict())
    with torch.no_grad():
        dm = float(q(O[:H]).mean(1).mean())
    return dm


for label, kw in [('baseline 5k', dict(steps=5000)),
                  ('20k', dict(steps=20000)),
                  ('50k', dict(steps=50000)),
                  ('50k lr3e-3', dict(steps=50000, lr=3e-3)),
                  ('50k no-target', dict(steps=50000, target=False)),
                  ('50k wide256', dict(steps=50000, hidden=256))]:
    print('%16s DM=%7.2f (exact %.1f)' % (label, run(**kw), truth), flush=True)
print('TOY DONE', flush=True)
