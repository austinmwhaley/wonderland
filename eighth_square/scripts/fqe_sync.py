import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from algorithms.deep.networks import QNetwork

g, H = 0.99, 100
truth = float(sum(g ** t for t in range(H)))
print('exact DM =', round(truth, 2), flush=True)
rng = np.random.default_rng(0)
N = 2000
obs = rng.normal(size=(N, 4)).astype(np.float32)
O, R, O2, D = (torch.as_tensor(obs), torch.ones(N, 1),
               torch.as_tensor(obs), torch.zeros(N, 1))


def run(steps, sync_every, lr=1e-3, batch=256):
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
        if s % sync_every == 0:
            qt.load_state_dict(q.state_dict())
    with torch.no_grad():
        return float(q(O[:H]).mean(1).mean())


print('steps=50000, vary target sync cadence:', flush=True)
for se in (50, 100, 200, 500, 1000):
    dm = run(50000, se)
    print('  sync_every=%5d DM=%7.2f  (%.0f%% of exact)' %
          (se, dm, 100 * dm / truth), flush=True)
print('TOY2 DONE', flush=True)
