import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from algorithms.deep.networks import QNetwork

g = 0.99
TRUTH = 100.0  # infinite-horizon uniform policy, reward 1: 1/(1-0.99)
print('true fixed point =', TRUTH, flush=True)
rng = np.random.default_rng(0)
N = 2000
obs = rng.normal(size=(N, 4)).astype(np.float32)
O, R, O2, D = (torch.as_tensor(obs), torch.ones(N, 1),
               torch.as_tensor(obs), torch.zeros(N, 1))


def run_polyak(steps, tau, lr=1e-3, batch=256):
    torch.manual_seed(0)
    q = QNetwork(4, 128, 2)
    qt = QNetwork(4, 128, 2)
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
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)
    with torch.no_grad():
        return float(q(O[:100]).mean(1).mean())


print('steps=50000, Polyak soft target (tau: smaller = faster tracking):', flush=True)
for tau in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
    dm = run_polyak(50000, tau)
    print('  tau=%.3f DM=%7.2f  (%.0f%% of exact)' % (tau, dm, 100 * dm / TRUTH), flush=True)
print('TOY3 DONE', flush=True)
