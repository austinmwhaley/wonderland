import sys
sys.path.insert(0, '/home/austin-whaley/wq')
import numpy as np
import torch
torch.set_num_threads(1)
from white_queen import db
from white_queen.tribunal.ope import estimators as E
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate

cfg = dict(PRESETS['quick'])
env = make_env('cartpole', seed=999)
data = db.load_diet('/home/austin-whaley/wq/white_queen/data/white_queen_quick.db',
                    'novice_only')
iql = load_candidate('iql', env, data, cfg,
                     '/home/austin-whaley/wq/white_queen/verdicts/v15/iql_novice_only.pt')
print('proxy live ~52; bar 30.0', flush=True)
# governed, full autotune, Polyak fix
for label, base in [
    ('governed default', {'device': 'cpu'}),
    ('patience=999 (no early stop)', {'device': 'cpu', 'patience': 999}),
    ('patience=999 tau=0.02', {'device': 'cpu', 'patience': 999, 'target_tau': 0.02}),
    ('patience=999 tau=0.05', {'device': 'cpu', 'patience': 999, 'target_tau': 0.05}),
]:
    torch.manual_seed(0)
    _, dm, info = E.fit_fqe(data, iql, 0.99, base, temperature=1.0)
    print('  %-30s DM=%7.1f tau=%.4f steps=%s stopped=%s' %
          (label, dm, info.get('target_tau', -1), info['steps'], info['stopped']),
          flush=True)
print('GOVERN DONE', flush=True)
