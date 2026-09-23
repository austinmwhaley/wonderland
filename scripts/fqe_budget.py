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
data=db.load_diet('/home/austin-whaley/wq/white_queen/data/white_queen_quick.db','novice_only')
iql=load_candidate('iql',env,data,cfg,'/home/austin-whaley/wq/white_queen/verdicts/v15/iql_novice_only.pt')
print('proxy live ~52; bar 30.0', flush=True)
for sm in (11428, 20000, 40000, 80000):
    torch.manual_seed(0)
    _,dm,info=E.fit_fqe(data,iql,0.99,{'device':'cpu','steps_max':sm,'eval_every':max(500,sm//20),'patience':20,'target_tau':0.02},temperature=1.0)
    print('  steps_max=%6d DM=%7.1f stopped=%s steps=%d' % (sm,dm,info['stopped'],info['steps']), flush=True)
print('BUDGET DONE', flush=True)
