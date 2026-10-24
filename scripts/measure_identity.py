"""Iteration-6 measurement: identity axis probed from the frozen donor E (not the id-head block).

Writes docs/report_cards/experiments/encoder_identity.json
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/austin-whaley/Documents/wonderland")
sys.path.insert(0, "/home/austin-whaley/Documents/wonderland/scripts/demo")
import torch  # noqa: E402
import auto_encoder as AE  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/t0_loop/relational_6k_v3"
OUT = Path("/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_identity.json")

E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
torch.cuda.empty_cache()


def z(X):
    X = np.nan_to_num(np.asarray(X, float))
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def purity(Z, lab, k=10):
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return float(np.mean(lab[idx] == lab[:, None]))


truth = {c["customer"]: c for c in json.load(open(Path(DATA) / "reference_truth.json"))["customers"]}
ch = np.array([truth[k]["channel_pref"] for k in hold], dtype=object)
cat = np.array([truth[k]["category_pref"] for k in hold], dtype=object)
# id-head block (what the old probe used)
e_ = model.ssm[0].e
dd = model.d
ib = np.nan_to_num(BLK)[:, 3 * e_ * dd : 3 * e_ * dd + sum(vs["id_dims"])]

out = {
    "E_channel": purity(z(E), ch),
    "table_channel": purity(z(Xtab), ch),
    "E_category": purity(z(E), cat),
    "table_category": purity(z(Xtab), cat),
    "IB_channel": purity(z(ib), ch),
    "IB_category": purity(z(ib), cat),
}
print(json.dumps(out, indent=1), flush=True)
OUT.write_text(json.dumps(out, indent=1))
print("WROTE", OUT)
