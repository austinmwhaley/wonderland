import sys; sys.path.insert(0,'/home/austin-whaley/wq')
import numpy as np, torch
print('cuda', torch.cuda.is_available(), flush=True)
from white_queen import db
from white_queen.tribunal.ope import estimators as E
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate
cfg=dict(PRESETS['quick']); env=make_env('cartpole',seed=999)
class U:
    def act(self,s,eval=True): return 0
    def action_probs(self,o,temperature=1.0):
        o=np.asarray(o); return np.full((len(o),2),0.5,dtype=np.float32)
class ArgmaxWrap:
    """Evaluate the SHIPPED argmax policy: action_probs = one-hot at argmax."""
    def __init__(self, inner): self.inner=inner
    def act(self,s,eval=True): return self.inner.act(s,eval=eval)
    def action_probs(self,o,temperature=1.0):
        p=np.asarray(self.inner.action_probs(o,temperature=1.0),dtype=np.float64)
        out=np.zeros_like(p); out[np.arange(len(p)),p.argmax(1)]=1.0
        return out
for diet in ['mixed','novice_only','expert_only']:
    data=db.load_diet('/home/austin-whaley/wq/white_queen/data/white_queen_quick.db',diet)
    for name in ['uniform','iql','bc','cql']:
        if name=='uniform': cand=U()
        else: cand=load_candidate(name,env,data,cfg,f'/home/austin-whaley/wq/white_queen/verdicts/v15/{name}_{diet}.pt')
        # sharp FQE: train on the ARGMAX policy probs, medium budget
        torch.manual_seed(0)
        _,dm,info=E.fit_fqe(data, ArgmaxWrap(cand), 0.99,
            {'steps_max':120000,'eval_every':5000,'patience':10,'batch':512,'hidden':128}, temperature=1.0)
        print(f'{diet:12s} {name:8s} FQE-argmax DM={dm:6.1f} stopped={info["stopped"]}', flush=True)
print('FQE SHARP DONE', flush=True)
