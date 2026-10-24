"""Identity axis probed from the frozen donor E, with a paired bootstrap CI.

Writes docs/report_cards/experiments/encoder_identity.json with E/table purity and
the paired-bootstrap SE of (E - table) per identity latent.
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

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/t0_loop/v7b"
OUT = Path("/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_identity.json")

E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
torch.cuda.empty_cache()


def z(X):
    X = np.nan_to_num(np.asarray(X, float))
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def nn_ok(Z, lab, k=10):
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return (lab[idx] == lab[:, None]).mean(1)  # per-customer kNN-same-label fraction


def paired(Eok, Tok, n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = len(Eok)
    d = np.array([Eok[rng.integers(0, m, m)].mean() - Tok[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(Eok.mean()), float(Tok.mean()), float(d.std()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


truth = {c["customer"]: c for c in json.load(open(Path(DATA) / "reference_truth.json"))["customers"]}
ch = np.array([truth[k]["channel_pref"] for k in hold], dtype=object)
cat = np.array([truth[k]["category_pref"] for k in hold], dtype=object)
zE, zT = z(E), z(Xtab)

out = {}
for name, lab in (("channel", ch), ("category", cat)):
    eo, to = nn_ok(zE, lab), nn_ok(zT, lab)
    em, tm, se, lo, hi = paired(eo, to)
    out[name] = {"E": round(em, 4), "table": round(tm, 4), "se": round(se, 4),
                 "lo": round(lo, 4), "hi": round(hi, 4), "not_worse_2se": bool((em - tm) >= -2 * se)}
    print(f"[id] {name}: E {em:.3f} table {tm:.3f} diff {em-tm:+.4f} ±{se:.4f} not_worse_2se={out[name]['not_worse_2se']}", flush=True)

OUT.write_text(json.dumps(out, indent=1))
print("WROTE", OUT)
