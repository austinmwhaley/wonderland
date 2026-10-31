"""DEC-462/464 pre-verdict diagnostic for protocol N (ledger profile block).

Premise to test BEFORE pre-registering: the ledger's shares+total cannot express raw
counts/modes (x0's measured sufficient statistic, DEC-464 R2 0.909), and name-blind
log-ratios + hazard interactions carry return/churn signal the table reads. Measures
the ceiling on the judge population with judge-identical probes. No GPU, no writes.
"""
import json
import sys
import time
from pathlib import Path

_T0 = time.perf_counter()


def _tick(label):
    print(f"[t] {label}: {time.perf_counter() - _T0:.1f}s", flush=True)

import duckdb
import numpy as np
import polars as pl

repo = Path("/home/austin_whaley/Documents/Repos/wonderland")
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "scripts"))
from measure_judge_law import _ridge_oos  # noqa: E402

from looking_glass import ledger_state  # noqa: E402
from looking_glass.cfm_data import _customer_keys, _read_stream, assign_split  # noqa: E402
from looking_glass.cfm_state import build_sequences, load_frozen_encoder  # noqa: E402
from looking_glass.customer_foundation_model import _resolve_registry  # noqa: E402

prod = Path("/home/austin_whaley/wonderland_work/cfm_m2_clean")
demo = Path("/home/austin_whaley/wonderland_work/demo")
manifest = json.loads((demo / "manifest.json").read_text())
as_of = float(np.datetime64(manifest["as_of"]).astype("datetime64[s]").astype(np.int64))

con = duckdb.connect(str(prod / "cfm_products.duckdb"), read_only=True)
A = [r[0] for r in con.execute("select customer_key from encoder_samples where split='A'").fetchall()]
B = [r[0] for r in con.execute("select customer_key from encoder_samples where split='B'").fetchall()]
led = {}
for k, part, vec in con.execute(
    "select customer_key, part, embedding from view_embeddings where view='measure'"
).fetchall():
    led.setdefault(k, {})[part] = np.asarray(vec, dtype=np.float64)
con.close()
from looking_glass import view_layout as V  # noqa: E402  (the judge's own part order)

_present = {q for r in led.values() for q in r}
_parts = [p for p in V.PART_ORDER.get("measure", []) if p in _present]
_parts += sorted(_present - set(_parts))
LA = {k: np.concatenate([led[k][p] for p in _parts]) for k in led}
_tick("duckdb ledger read")

tag, meta = _resolve_registry(prod, None)
model, cfg = load_frozen_encoder(tag, prod)
from looking_glass.customer_foundation_model import _replay_run_cfg  # noqa: E402

_replay_run_cfg(meta, cfg, prod, cfg)  # exact products-path config replay (2500 customers, split seed)
cfg.db = str(demo / "customer_event_stream.parquet")
cfg.out_dir = str(prod)
df = _read_stream(cfg)
_tick("stream read (parquet)")
keys = _customer_keys(df, cfg)
split = assign_split(keys, cfg)
seqs = build_sequences(df, keys, cfg, split, with_anchors=False, min_events=1)
_tick("build_sequences (polars)")
vocab = model.vocab
train_keys = [k for k in keys if split.get(k, "A") == "A"]
builder, _ = ledger_state.fit_vocabulary(seqs, vocab, train_keys, as_of=as_of, recency=True)
tup = ledger_state.seqs_to_tuples(seqs, vocab, as_of=as_of)
_tick("fit_vocabulary + seqs_to_tuples")


def profile(t):
    et, gcat, _, ts, _, _, _ = t
    n = len(et)
    if n == 0:
        return None
    feats = []
    for j, keep in enumerate(builder.keep):
        col = gcat[:, j]
        cnts = np.bincount(col, minlength=int(builder.vs["gcat"][j])).astype(np.float64)
        feats.append(np.log1p(cnts[keep]))
        if len(keep):
            vi = np.asarray(keep, dtype=np.float64)
            w = cnts[keep]
            feats.append(
                np.array([float((w * vi).sum() / max(w.sum(), 1.0)), float(w.max() / max(w.sum(), 1.0))])
            )
        p = np.bincount(col).astype(np.float64)
        p = p[p > 0] / n
        feats.append(np.array([-(p * np.log(p)).sum() / np.log(max(len(p), 2))]))
    ec = np.bincount(et, minlength=int(builder.vs["et"])).astype(np.float64)
    lc = np.log1p(ec)
    ix = np.triu_indices(len(lc), 1)
    feats.append(lc[ix[0]] - lc[ix[1]])  # all pairwise name-blind log-ratios
    # principle-ok: seconds->days unit scale for probe conditioning (dimensionless),
    # mirrors the authorized recency block exactly (measure_state.py:193-195, DEC-473).
    DAY = 86400.0  # principle-ok: unit conversion, not a cadence decision (DEC-473 precedent)
    silence = max(as_of - float(ts.max()), 0.0) / DAY
    tenure = max(as_of - float(ts.min()), 0.0) / DAY
    span = max(float(ts.max()) - float(ts.min()), 0.0) / DAY
    feats.append(
        np.array(
            [
                np.log1p(silence) * np.log1p(n),
                silence / max(n / max(tenure, 1e-9), 1e-9),
                silence - tenure / max(n, 1.0),
                np.log1p(tenure) / max(np.log1p(silence), 1e-9),
                np.log1p(span),
            ]
        )
    )
    return np.concatenate(feats)


prof = {k: profile(t) for k, t in tup.items()}
prof = {k: v for k, v in prof.items() if v is not None}


def profile_raw(t):
    """DEC-464's exact geometry: RAW per-value counts (no log1p), plus per-field
    concentration/entropy — the 0.909 construction, name-blind over kept vocab."""
    et, gcat, _, ts, _, _, _ = t
    n = len(et)
    if n == 0:
        return None
    feats = []
    for j, keep in enumerate(builder.keep):
        col = gcat[:, j]
        cnts = np.bincount(col, minlength=int(builder.vs["gcat"][j])).astype(np.float64)
        feats.append(cnts[keep])
        if len(keep):
            w = cnts[keep]
            feats.append(np.array([float(w.max() / max(w.sum(), 1.0))]))
    return np.concatenate(feats)


praw = {k: profile_raw(t) for k, t in tup.items()}
praw = {k: v for k, v in praw.items() if v is not None}
_tick("feature build (numpy, per-customer)")
truth = {c["customer"]: c for c in json.loads((demo / "reference_truth.json").read_text())["customers"]}
pathA = {"x0": 0.9056, "h0": -0.1213, "return_proneness": 0.6734, "churn_day": 0.7209}
W = len(next(iter(prof.values())))
Wr = len(next(iter(praw.values())))
print(f"profile width={W} raw width={Wr}  fit={len(A)} score={len(B)}")
print(f"{'trait':18s} {'ledger':>8s} {'+prof':>8s} {'prof':>8s} {'+raw':>8s} {'raw':>8s} {'PATH A':>8s}")
for t in ("x0", "h0", "return_proneness", "churn_day"):
    ya = np.array([float(truth[k][t]) for k in A])
    yb = np.array([float(truth[k][t]) for k in B])
    VA = np.vstack([LA[k] for k in A])
    VB = np.vstack([LA[k] for k in B])
    FA = np.vstack([prof[k] for k in A])
    FB = np.vstack([prof[k] for k in B])
    RA = np.vstack([praw[k] for k in A])
    RB = np.vstack([praw[k] for k in B])
    _, r0 = _ridge_oos(VA, ya, VB, yb)
    _, r1 = _ridge_oos(np.hstack([VA, FA]), ya, np.hstack([VB, FB]), yb)
    _, r2 = _ridge_oos(FA, ya, FB, yb)
    _, r3 = _ridge_oos(np.hstack([VA, RA]), ya, np.hstack([VB, RB]), yb)
    _, r4 = _ridge_oos(RA, ya, RB, yb)
    print(f"{t:18s} {r0:8.4f} {r1:8.4f} {r2:8.4f} {r3:8.4f} {r4:8.4f} {pathA[t]:8.4f}")

# --- M3 premise check: freq-histogram readability on M2's undimmed served brain ---
# K/L receipt (faded serving): E->hist per-bin R2 = 0.643/0.194/0.059/0.533 (DEC-469/471).
con = duckdb.connect(str(prod / "cfm_products.duckdb"), read_only=True)
emb = {
    k: np.asarray(v, dtype=np.float64)
    for k, v in con.execute("select customer_key, embedding from state_embeddings").fetchall()
}
con.close()
fq = df.with_columns(pl.col("freq").cast(pl.Float64, strict=False).round(0).alias("f")).filter(
    pl.col("f").is_not_null()
)
mx = int(fq["f"].max())
# one vectorized bincount over group*n+value (the MeasureBuilder idiom) — no per-event loop
uk = {k: i for i, k in enumerate(sorted(fq["customer_key"].unique().to_list()))}
idx = np.fromiter((uk[k] for k in fq["customer_key"].to_list()), np.int64)
fb = fq["f"].to_numpy().astype(np.int64)
flat = np.bincount(idx * (mx + 1) + fb, minlength=len(uk) * (mx + 1)).reshape(len(uk), mx + 1)
hist = {k: flat[i] for i, k in enumerate(uk)}
if not hist:
    print("no numeric freq column in the stream")
else:
    Wf = len(next(iter(hist.values())))
    EA = np.vstack([emb[k] for k in A]); EB = np.vstack([emb[k] for k in B])
    HA = np.vstack([hist[k] for k in A]); HB = np.vstack([hist[k] for k in B])
    rs = []
    for b in range(Wf):
        _, r = _ridge_oos(EA, HA[:, b], EB, HB[:, b])
        rs.append(r if r is not None else float("nan"))
    print(f"freq bins={Wf} E->hist per-bin R2: " + "/".join(f"{v:.3f}" for v in rs))
    print("K/L faded receipt for comparison:   0.643/0.194/0.059/0.533")
