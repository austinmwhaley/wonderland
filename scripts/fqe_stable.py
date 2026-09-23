import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np, torch
import torch.nn.functional as F
torch.set_num_threads(1)
from white_queen import db
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate
from algorithms.deep.networks import QNetwork
cfg=dict(PRESETS['quick']); env=make_env('cartpole',seed=999)
data=db.load_diet('/home/austin-whaley/wq/white_queen/data/white_queen_quick.db','novice_only')
iql=load_candidate('iql',env,data,cfg,'/home/austin-whaley/wq/white_queen/verdicts/v15/iql_novice_only.pt')
N=len(data['obs']); O=torch.as_tensor(data['obs']); A=torch.as_tensor(data['act'],dtype=torch.long)
R=torch.as_tensor(data['rew']).unsqueeze(1); O2=torch.as_tensor(data['obs2']); D=torch.as_tensor(data['done']).unsqueeze(1)
P=iql.action_probs(data['obs'],temperature=1.0); P2=torch.as_tensor(iql.action_probs(data['obs2'],temperature=1.0))
ep=data['episode']; starts=np.unique(ep,return_index=True)[1]
def run(steps=80000,lr=1e-3,clip=0.0,huber=False,tau=0.02,batch=512,hidden=128,seed=0):
    torch.manual_seed(seed); q=QNetwork(4,hidden,2); qt=QNetwork(4,hidden,2); qt.load_state_dict(q.state_dict())
    opt=torch.optim.Adam(q.parameters(),lr=lr); rng=np.random.default_rng(seed)
    for s in range(steps):
        i=torch.as_tensor(rng.integers(0,N,batch))
        with torch.no_grad():
            v2=(P2[i]*qt(O2[i])).sum(1,keepdim=True); tgt=R[i]+0.99*(1-D[i])*v2
        pred=q(O[i]).gather(1,A[i].unsqueeze(1))
        loss=(F.smooth_l1_loss(pred,tgt) if huber else F.mse_loss(pred,tgt))
        opt.zero_grad(); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(q.parameters(),clip)
        opt.step()
        with torch.no_grad():
            for p,pt in zip(q.parameters(),qt.parameters()): pt.mul_(1-tau).add_(p,alpha=tau)
    with torch.no_grad():
        v=(torch.as_tensor(P)*q(O)).sum(1); return float(v[starts].mean())
print('proxy live ~52; bar 30.0', flush=True)
for label,kw in [('baseline 80k',dict()),
                 ('clip=1.0',dict(clip=1.0)),
                 ('lr=5e-4',dict(lr=5e-4)),
                 ('clip+lr5e-4',dict(clip=1.0,lr=5e-4)),
                 ('huber',dict(huber=True)),
                 ('huber+clip+lr5e-4',dict(huber=True,clip=1.0,lr=5e-4))]:
    print('  %-20s DM=%7.1f' % (label, run(**kw)), flush=True)
print('STABLE DONE', flush=True)
