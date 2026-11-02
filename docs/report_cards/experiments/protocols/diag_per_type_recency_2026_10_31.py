"""CPU ceiling check: per-event-type recency added to the ledger (no GPU, read-only).

The ledger has global silence/tenure/span/count. PATH A's table carries per-event-type
recency (days since last purchase, etc.). This measures whether a name-blind per-type
recency block (time since last occurrence of each event type, log1p days) closes the
churn/return gaps, before any protocol is pre-registered.
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

repo = Path("/home/austin_whaley/Documents/Repos/wonderland")
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "scripts"))
from measure_judge_law import _ridge_oos  # noqa: E402

from looking_glass import ledger_state, view_layout as V  # noqa: E402
from looking_glass.cfm_data import _customer_keys, _read_stream, assign_split  # noqa: E402
from looking_glass.cfm_state import build_sequences, load_frozen_encoder  # noqa: E402
from looking_glass.customer_foundation_model import _replay_run_cfg, _resolve_registry  # noqa: E402

T0 = time.perf_counter()
prod = Path("/home/austin_whaley/wonderland_work/cfm_m2_clean")
demo = Path("/home/austin_whaley/wonderland_work/demo")
manifest = json.loads((demo / "manifest.json").read_text())
as_of = float(np.datetime64(manifest["as_of"]).astype("datetime64[s]").astype(np.int64))

con = duckdb.connect(str(prod / "cfm_products.duckdb"), read_only=True)
A = [
    r[0] for r in con.execute("select customer_key from encoder_samples where split='A'").fetchall()
]
B = [
    r[0] for r in con.execute("select customer_key from encoder_samples where split='B'").fetchall()
]
led = {}
for k, part, vec in con.execute(
    "select customer_key, part, embedding from view_embeddings where view='measure'"
).fetchall():
    led.setdefault(k, {})[part] = np.asarray(vec, dtype=np.float64)
con.close()
present = {q for r in led.values() for q in r}
parts = [p for p in V.PART_ORDER.get("measure", []) if p in present]
parts += sorted(present - set(parts))
LA = {k: np.concatenate([led[k][p] for p in parts]) for k in led}
print(f"[t] ledger read {time.perf_counter() - T0:.1f}s  dim={len(next(iter(LA.values())))}")

tag, meta = _resolve_registry(prod, None)
model, cfg = load_frozen_encoder(tag, prod)
_replay_run_cfg(meta, cfg, prod, cfg)
cfg.db = str(demo / "customer_event_stream.parquet")
cfg.out_dir = str(prod)
df = _read_stream(cfg)
keys = _customer_keys(df, cfg)
split = assign_split(keys, cfg)
seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
vocab = model.vocab
tup = ledger_state.seqs_to_tuples(seqs, vocab, as_of=as_of)
print(f"[t] sequences {time.perf_counter() - T0:.1f}s")

n_et = int(vocab.n_et)
rec = {}
first = {}
for k, t in tup.items():
    et, _, _, ts, _, _, _ = t
    n = len(et)
    if n == 0:
        continue
    d_last = np.full(n_et, np.nan)
    d_first = np.full(n_et, np.nan)
    # last occurrence per type: reverse-unique
    order = np.argsort(ts, kind="stable")
    et_s, ts_s = et[order], ts[order]
    uniq, idx_last = np.unique(et_s[::-1], return_index=True)
    d_last[uniq] = (
        as_of - ts_s[::-1][idx_last]
    ) / 86400.0  # principle-ok: unit scale (DEC-473 precedent)
    uniq0, idx_first = np.unique(et_s, return_index=True)
    d_first[uniq0] = (
        as_of - ts_s[idx_first]
    ) / 86400.0  # principle-ok: unit scale (DEC-473 precedent)
    # observed types: log1p(days since last); unseen types: sentinel = log1p(tenure) (never happened)
    tenure_days = (
        max(as_of - float(ts.min()), 0.0) / 86400.0
    )  # principle-ok: unit scale (DEC-473 precedent)
    seen = ~np.isnan(d_last)
    feat = np.where(seen, np.log1p(np.nan_to_num(d_last)), np.log1p(tenure_days))
    rec[k] = feat
    first[k] = np.log1p(np.nan_to_num(d_first, nan=tenure_days))
print(f"[t] per-type recency {time.perf_counter() - T0:.1f}s  width={n_et}")

truth = {
    c["customer"]: c for c in json.loads((demo / "reference_truth.json").read_text())["customers"]
}
pathA = {"x0": 0.9056, "h0": -0.1213, "return_proneness": 0.6734, "churn_day": 0.7209}
print(
    f"{'trait':18s} {'ledger':>8s} {'+rec':>8s} {'+rec+first':>10s} {'rec-only':>9s} {'PATH A':>8s}"
)
for t in ("x0", "h0", "return_proneness", "churn_day"):
    ya = np.array([float(truth[k][t]) for k in A])
    yb = np.array([float(truth[k][t]) for k in B])
    VA = np.vstack([LA[k] for k in A])
    VB = np.vstack([LA[k] for k in B])
    RA = np.vstack([rec[k] for k in A])
    RB = np.vstack([rec[k] for k in B])
    FA = np.vstack([first[k] for k in A])
    FB = np.vstack([first[k] for k in B])
    _, r0 = _ridge_oos(VA, ya, VB, yb)
    _, r1 = _ridge_oos(np.hstack([VA, RA]), ya, np.hstack([VB, RB]), yb)
    _, r2 = _ridge_oos(np.hstack([VA, RA, FA]), ya, np.hstack([VB, RB, FB]), yb)
    _, r3 = _ridge_oos(np.hstack([RA, FA]), ya, np.hstack([RB, FB]), yb)
    print(f"{t:18s} {r0:8.4f} {r1:8.4f} {r2:10.4f} {r3:9.4f} {pathA[t]:8.4f}")
print(f"[t] total {time.perf_counter() - T0:.1f}s")
