import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np, torch
torch.set_num_threads(1)
from white_queen import db
from white_queen.tribunal.ope import estimators as E
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate
cfg=dict(PRESETS['quick']); env=make_env('cartpole',seed=999)
for diet in ['novice_only','mixed','expert_only']:
    data=db.load_diet('/home/austin-whaley/wq/white_queen/data/white_queen_quick.db',diet)
    for name in ['uniform','iql']:
        if name=='uniform':
            class U:
                def act(self,s,eval=True): return 0
                def action_probs(self,o,temperature=1.0):
                    o=np.asarray(o); return np.full((len(o),2),0.5,dtype=np.float32)
            cand=U()
        else:
            cand=load_candidate(name,env,data,cfg,f'/home/austin-whaley/wq/white_queen/verdicts/v15/{name}_{diet}.pt')
        torch.manual_seed(0)
        _,dm,info=E.fit_fqe(data,cand,0.99,{'device':'cpu','steps_max':80000,'eval_every':4000,'patience':20},temperature=1.0)
        print('%-12s %-8s governed FQE DM=%7.1f stopped=%s tau=%.3f' % (diet,name,dm,info['stopped'],info.get('target_tau',-1)), flush=True)
print('VALIDATE DONE', flush=True)
