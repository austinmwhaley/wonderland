"""Principle-aligned auto encoder: all discovered fields, generic objectives, frozen E."""

import collections
import datetime
import json
import math
import sys
import hashlib
from pathlib import Path

import os as _os
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import polars as pl
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from looking_glass.cfm_model import _scan  # noqa: E402
from looking_glass.autotune import derive_batch, derive_dim, derive_lr  # noqa: E402
from looking_glass.cfm_training import embed_batch_size  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_STATS = {"steps": 0, "binds": 0}
AS_OF_TS = 0.0
BATCH = 16
LABELS = (
    "y_value_30",
    "y_orders_30",
    "y_churn_30",
    "y_value_90",
    "y_orders_90",
    "y_churn_90",
    "y_margin_90",
)


from stream_loader import load_data as _stream_load, _prep_rev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

_VENV_CANDIDATES = (
    _os.environ.get("WONDERLAND_CUDF_PY", ""),
    str(Path(__file__).resolve().parents[1] / ".venv-cudf" / "bin" / "python"),
    "/tmp/opencode/cudf_venv_uv/bin/python",
)


def _venv_python():
    for c in _VENV_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def _artifact(data):
    return Path(data) / f"prep_{_prep_rev(data)}.npz"


def _load_artifact(art, data):
    z = np.load(art, allow_pickle=False)
    spec = json.loads(bytes(z["spec"]).decode())
    keys = spec["keys"]
    lens = z["lens"]
    ends = np.cumsum(lens)
    starts = ends - lens
    big = {k: z[k] for k in ("et", "gcat", "gnum", "time", "gaps", "tss", "graw")}
    seqs = {
        k: (big["et"][s:e], big["gcat"][s:e], big["gnum"][s:e], big["time"][s:e], big["gaps"][s:e], big["tss"][s:e], big["graw"][s:e])
        for k, s, e in zip(keys, starts, ends)
    }
    A = pl.read_parquet(Path(data) / "path_a/path_a_wide_table.parquet")
    z.close()
    return (
        None,
        A,
        keys,
        seqs,
        spec["vs"],
        spec["et_v"],
        spec["catspec"],
        [tuple(x) for x in spec["num"]],
        np.array(spec["taus"]),
        spec["p_mask"],
        spec["as_of_ts"],
        spec["periods"],
        np.array(spec["horizons"]),
    )


def load_data(data):
    eng = _os.environ.get("WONDERLAND_LOAD_ENGINE", "auto")
    art = _artifact(data)
    if eng != "cpu" and art.exists():
        z = np.load(art, allow_pickle=False)
        ok = json.loads(bytes(z["spec"]).decode())["rev"] == _prep_rev(data)
        z.close()
        if ok:
            return _load_artifact(art, data)
    vp = None if eng == "cpu" else _venv_python()
    if vp is not None:
        import subprocess

        r = subprocess.run(
            [vp, str(Path(__file__).resolve().parent / "gpu_prep.py"), data, str(art)],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and art.exists():
            print((r.stdout.strip().splitlines() or [""])[-1], flush=True)
            return _load_artifact(art, data)
        print(f"gpu prep failed: {r.stderr.strip()[-300:]}; CPU in-process fallback", flush=True)
    return _stream_load(data, engine=None)


class GatedSSM(nn.Module):
    def __init__(self, d, e, taus):
        super().__init__()
        self.d = d
        self.e = e
        self.Wx = nn.Linear(d, e * d)
        self.out = nn.Linear(d, d)
        self.log_tau = nn.Parameter(torch.log(torch.as_tensor(taus, dtype=torch.float32)))

    def forward(self, x, mask, dt):
        B, T, _ = x.shape
        tau = torch.exp(self.log_tau)
        decay = torch.exp(-dt.unsqueeze(-1) / tau)
        decay = torch.where(mask.unsqueeze(-1) > 0, decay, torch.ones_like(decay))
        inp = self.Wx(x).view(B, T, self.e, self.d)
        b = (1 - decay).unsqueeze(-1) * inp
        b = torch.where(mask.unsqueeze(-1).unsqueeze(-1) > 0, b, torch.zeros_like(b))
        _, H = _scan(decay.unsqueeze(-1), b)
        y = H.sum(2)
        # additive mass state: m_t = decay * m_{t-1} + inp_t (no 1-decay normalization)
        _, M = _scan(decay.unsqueeze(-1), inp)
        return F.gelu(self.out(y)) + y, H, M


def _ident_targets(seqs, keys, vs):
    """Per-customer field histograms (all categorical fields), projected to <=32 dims each."""
    dims = vs.get("ident_dims", [])
    if not dims:
        return {}, 0
    g = np.random.default_rng(1234)
    projs = [g.normal(size=(s, dd)).astype(np.float32) / np.sqrt(max(s, 1)) for s, dd in zip(vs["gcat"], dims)]
    out = {}
    for k in keys:
        gck = seqs[k][1]
        row = np.zeros(vs["ident_dim"], np.float32)
        off = 0
        for j, (sz, dd) in enumerate(zip(vs["gcat"], dims)):
            h = np.bincount(gck[:, j], minlength=sz).astype(np.float32)
            if h.shape[0] > sz:
                h = h[:sz]
            h[0] = 0.0
            tot = h.sum()
            if tot > 0:
                h = h / tot
            row[off : off + dd] = h @ projs[j]
            off += dd
        out[k] = row
    return out, vs["ident_dim"]


class Enc(nn.Module):
    def __init__(self, vs, d, e, layers, taus):
        super().__init__()
        self.d = d
        self.et = nn.Embedding(vs["et"], d)
        self.gcat = nn.ModuleList([nn.Embedding(s, d) for s in vs["gcat"]])
        self.gnum = nn.ModuleList([nn.Linear(1, d) for _ in range(vs["gnum"])])
        self.time = nn.Linear(vs["time_dim"], d)
        self.gw_cat = nn.Parameter(torch.full((len(vs["gcat"]),), 0.5413))
        self.gw_num = nn.Parameter(torch.full((vs["gnum"],), 0.5413))
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ssm = nn.ModuleList([GatedSSM(d, e, taus) for _ in range(layers)])
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.n_read = 4 + 3 * e
        self.rproj = nn.ModuleList([nn.Linear(d, d) for _ in range(self.n_read)])
        self.rw = nn.Parameter(torch.zeros(self.n_read))
        self.raw = nn.Linear(vs["raw"], d)
        self.mass_proj = nn.Linear(vs["raw"] + vs["et"], e * d)
        self.rmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.qgap = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.head_et = nn.Linear(d, vs["et"])
        self.head_gap = nn.Linear(d, 1)
        self.heads_cat = nn.ModuleList([nn.Linear(d, s) for s in vs["gcat"]])
        self.heads_num = nn.ModuleList([nn.Linear(d, 1) for _ in range(vs["gnum"])])
        self.past = nn.Linear(d, vs["past_dim"])
        self.head_fut = nn.ModuleList([nn.Linear(d, vs["fut_dim"]) for _ in vs["horizons"]])
        self.head_id = nn.ModuleList([nn.Linear(d, s2) for s2 in vs["id_dims"]])
        self.head_unc = nn.Linear(d, vs["unc_dim"]); self.rel_idx = list(vs["rel_idx"])
        self.head_ident = nn.Linear(d, vs["ident_dim"]) if vs.get("ident_dim", 0) > 0 else None

    def forward(self, et, gcat, gnum, time, gaps, mask, lengths, qgap, graw):
        x = self.et(et) + self.time(time)
        for i, emb in enumerate(self.gcat):
            x = x + F.softplus(self.gw_cat[i]) * emb(gcat[:, :, i])
        for i, lin in enumerate(self.gnum):
            x = x + F.softplus(self.gw_num[i]) * lin(gnum[:, :, i : i + 1])
        x = self.mlp(x) + self.raw(torch.sign(graw) * torch.log1p(graw.abs()))
        dt = gaps.clamp(min=1e-3)
        H = None; M = None
        for ssm in self.ssm:
            x, H, M = ssm(x, mask, dt)
        Hs = x
        ar = torch.arange(Hs.shape[0], device=Hs.device)
        last = Hs[ar, lengths - 1]
        m = mask.unsqueeze(-1)
        mean = (Hs * m).sum(1) / lengths.unsqueeze(-1)
        mx = Hs.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values
        score = (Hs @ self.q) / math.sqrt(self.d)
        score = score.masked_fill(mask == 0, -1e9)
        att = torch.softmax(score, 1)
        pooled = (Hs * att.unsqueeze(-1)).sum(1)
        Hlast = H.reshape(H.shape[0], H.shape[1], H.shape[2], H.shape[3])[ar, lengths - 1]  # (B,e,d)
        Hmean = (H * mask.unsqueeze(-1).unsqueeze(-1)).sum(1) / lengths.view(-1, 1, 1)
        mass_in = torch.cat([torch.sign(graw) * torch.log1p(graw.abs()),
                             F.one_hot(et, self.et.num_embeddings).float()], -1)
        Mp = self.mass_proj(mass_in)
        # recompute the additive mass state (same decay) for the readout
        # M from the last SSM is already the additive mass; use its last row
        Mlast = M[ar, lengths - 1]  # (B,e,d), recency-weighted raw mass
        rs = [last, mean, mx, pooled]
        for j in range(Hlast.shape[1]):
            rs.append(Hlast[:, j]); rs.append(Hmean[:, j])
        for j in range(Mlast.shape[1]):
            rs.append(Mlast[:, j])
        pr = [self.rproj[i](rs[i]) for i in range(self.n_read)]
        w = F.softplus(self.rw)
        E = sum(w[i] * pr[i] for i in range(self.n_read))
        E = self.rmlp(E)
        E = E + self.qgap(torch.log1p(qgap).unsqueeze(-1))
        MB = torch.cat([Mlast.reshape(Mlast.shape[0], -1), Hlast.reshape(Hlast.shape[0], -1), Hmean.reshape(Hmean.shape[0], -1)], 1)
        # learned blocks (trained heads, not static summaries)
        IB = torch.cat([h(E) for h in self.head_id], 1) if len(self.head_id) else E.new_zeros((E.shape[0], 0))
        UB = self.head_unc(E)
        RB = self.head_gap(E)
        OB = self.head_et(E)
        RelB = torch.cat([self.heads_cat[j](E) for j in self.rel_idx], 1) if self.rel_idx else E.new_zeros((E.shape[0], 0))
        CB = torch.cat([hf(E) for hf in self.head_fut], 1)
        return Hs, E, w / w.sum(), MB, IB, RB, UB, CB, OB, RelB


def collate(
    seqs,
    ks,
    cuts,
    qgaps,
    vs,
    want_past,
    past_mu,
    past_sd,
    want_future=False,
    h=0.0,
    fut_mu=None,
    fut_sd=None,
):
    T = max(cuts)
    Bn = len(ks)
    nc = len(vs["gcat"])
    nnf = vs["gnum"]
    td = vs["time_dim"]
    et = np.zeros((Bn, T), np.int64)
    gcat = np.zeros((Bn, T, nc), np.int64)
    gnum = np.zeros((Bn, T, nnf), np.float32)
    graw = np.zeros((Bn, T, vs["raw"]), np.float32)
    time = np.zeros((Bn, T, td), np.float32)
    gaps = np.zeros((Bn, T), np.float32)
    mask = np.zeros((Bn, T), np.float32)
    tsarr = np.zeros((Bn, T), np.float64)
    past = np.zeros((Bn, vs["past_dim"]), np.float32)
    fut = np.zeros((Bn, vs["fut_dim"]), np.float32)
    id_tgt = np.zeros((Bn, len(vs["id_idx"])), np.int64)
    unc_tgt = np.zeros((Bn, vs["unc_dim"]), np.float32)
    qg = np.zeros(Bn, np.float32)
    for b, (k, cut) in enumerate(zip(ks, cuts)):
        etk, gck, gnk, tmk, gpk, tt, grk = seqs[k]
        L = min(cut, len(tt))
        q = float(qgaps[b])
        qg[b] = q
        et[b, :L] = etk[:L]
        gcat[b, :L] = gck[:L]
        gnum[b, :L] = gnk[:L]
        graw[b, :L] = grk[:L]
        time[b, :L] = tmk[:L]
        gaps[b, :L] = gpk[:L]
        tsarr[b, :L] = tt[:L]
        mask[b, :L] = 1
        for jj, ii in enumerate(vs["id_idx"]):
            id_tgt[b, jj] = gck[0, ii]
        unc_tgt[b, :len(vs["gcat"])] = (gck[:L] == 0).mean(0)
        unc_tgt[b, len(vs["gcat"]):] = (gnk[:L] == 0).mean(0)
        if want_past and L > 0:
            ev = etk[:L]
            etc = [float((ev == i).sum()) for i in range(vs["et"])]
            catd = [float(len(np.unique(gck[:L, j][gck[:L, j] > 0]))) for j in range(nc)]
            stat = []
            for j in range(vs["raw"]):
                x = grk[:L, j]
                stat += [
                    float((x != 0).sum()),
                    float(x.sum()),
                    float(x.mean()),
                    float(x.std()),
                    float(x.min()),
                    float(x.max()),
                ]
            tenure = (tt[L - 1] - tt[0]) / 86400.0
            past[b] = (
                etc
                + catd
                + stat
                + [
                    tenure,
                    float(np.std(tt[:L]) / 86400.0),
                    q,
                    q,
                    tenure,
                    float(L),
                    float(len(np.unique(ev))),
                ]
            )
        if want_future and L > 0:
            ct = tt[L - 1]
            sel = (tt > ct) & (tt <= ct + h * 86400.0)
            etcf = [float((etk[sel] == i).sum()) for i in range(vs["et"])]
            catf = [float(len(np.unique(gck[sel, j][gck[sel, j] > 0]))) for j in range(nc)]
            rawf = []
            for j in range(vs["raw"]):
                x = grk[sel, j]
                rawf += [float(x.sum()), float(x.mean()) if len(x) else 0.0]
            fut[b] = etcf + catf + rawf
    if want_past and past_mu is not None:
        past = (past - past_mu) / past_sd
    if want_future and fut_mu is not None:
        fi = int(np.argmin(np.abs(np.asarray(vs["horizons"]) - h)))
        fut = (fut - fut_mu[fi]) / fut_sd[fi]
    t = dict(
        et=torch.tensor(et, device=DEV),
        gcat=torch.tensor(gcat, device=DEV),
        gnum=torch.tensor(gnum, device=DEV),
        graw=torch.tensor(graw, device=DEV),
        id_tgt=torch.tensor(id_tgt, device=DEV),
        unc_tgt=torch.tensor(unc_tgt, device=DEV),
        time=torch.tensor(time, device=DEV),
        gaps=torch.tensor(gaps, device=DEV),
        mask=torch.tensor(mask, device=DEV),
        tss=torch.tensor(tsarr, device=DEV),
    )
    return (
        t,
        torch.tensor(past, device=DEV),
        torch.tensor(qg, device=DEV),
        torch.tensor(cuts, device=DEV),
        torch.tensor(fut, device=DEV),
    )


def _next_losses(model, t, m, p_mask, rng):
    def ce(head, tgt):
        logit = head(model_hs[:, :-1]).reshape(-1, head.out_features)
        y = tgt[:, 1:].reshape(-1)
        mm = m.reshape(-1) > 0
        return F.cross_entropy(logit[mm], y[mm]) if mm.sum() > 0 else torch.tensor(0.0, device=DEV)

    model_hs = model._hs

    def mse(head, tgt):
        pred = head(model_hs[:, :-1]).squeeze(-1)
        y = tgt[:, 1:].float()
        return (
            (((pred - y) ** 2) * m).sum() / m.sum()
            if m.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )

    def keep():
        return rng.random() < (1.0 - p_mask)

    Ls = [ce(model.head_et, t["et"])]
    for j, head in enumerate(model.heads_cat):
        Ls.append(ce(head, t["gcat"][:, :, j]) if keep() else torch.tensor(0.0, device=DEV))
    for j, head in enumerate(model.heads_num):
        Ls.append(mse(head, t["gnum"][:, :, j]) if keep() else torch.tensor(0.0, device=DEV))
    Ls.append(mse(model.head_gap, t["gaps"]))
    return Ls


def run_epoch(
    model,
    seqs,
    ks,
    opt,
    sched,
    log_s,
    rng,
    vs,
    past_mu,
    past_sd,
    p_mask,
    taus,
    fut_mu,
    fut_sd,
    horizons,
    train=True,
    ident_map=None,
):
    model.train(train)
    order = np.array(ks)[rng.permutation(len(ks))]
    tot = 0
    nb = 0
    for i in range(0, len(order), BATCH):
        bk = list(order[i : i + BATCH])
        cuts = []
        for k in bk:
            lo = seqs[k][5][0]
            hi = seqs[k][5][-1]
            ct = float(rng.uniform(lo, hi))
            cut = int(np.searchsorted(seqs[k][5], ct, side="right"))
            cuts.append(max(8, min(cut, len(seqs[k][5]) - 2)))
        hor = vs["horizons"]
        _hi = int(rng.integers(len(hor)))
        h = float(hor[_hi])
        t, past, qg, lens, fut = collate(
            seqs, bk, cuts, [0.0] * len(bk), vs, False, None, None, True, h, fut_mu, fut_sd
        )
        Hs, E, w, *_blocks = model(t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"])
        model._hs = Hs
        m = t["mask"][:, 1:]
        Ls = _next_losses(model, t, m, p_mask, rng)
        Ls.append(((model.head_fut[_hi](E) - fut) ** 2).mean())
        for jj, h in enumerate(model.head_id):
            Ls.append(F.cross_entropy(h(E), t["id_tgt"][:, jj]))
        Ls.append(((model.head_unc(E) - t["unc_tgt"]) ** 2).mean())
        if model.head_ident is not None and ident_map is not None:
            idt = torch.tensor(np.stack([ident_map[k] for k in bk]), device=DEV)
            Ls.append(((model.head_ident(E) - idt) ** 2).mean())
        loss = sum(0.5 * torch.exp(-log_s[j]) * Ls[j] + 0.5 * log_s[j] for j in range(len(Ls)))
        fcuts = [len(seqs[k][5]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in bk]
        t2, past2, qg2, lens2, _f2 = collate(seqs, bk, fcuts, fq, vs, True, past_mu, past_sd)
        _, E2, _, *_b2 = model(
            t2["et"], t2["gcat"], t2["gnum"], t2["time"], t2["gaps"], t2["mask"], lens2, qg2, t2["graw"]
        )
        lp = ((model.past(E2) - past2) ** 2).mean()
        j = len(Ls)
        loss = loss + 0.5 * torch.exp(-log_s[j]) * lp + 0.5 * log_s[j]
        if train:
            opt.zero_grad()
            loss.backward()
            CLIP_STATS["steps"] += 1
            opt.step()
        tot += float(loss.detach())
        nb += 1
    if train and sched:
        sched.step()
    return tot / max(nb, 1)


@torch.no_grad()
def val_loss(model, seqs, ks, vs, past_mu, past_sd, taus, fut_mu, fut_sd, horizons):
    model.eval()
    tot = 0
    nb = 0
    eb = max(32, BATCH)
    for i in range(0, len(ks), eb):
        bk = ks[i : i + eb]
        fcuts = [len(seqs[k][5]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in bk]
        h = float(vs["horizons"][-1])
        t, past, qg, lens, fut = collate(
            seqs, bk, fcuts, fq, vs, True, past_mu, past_sd, True, h, fut_mu, fut_sd
        )
        Hs, E, w, *_blocks = model(t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"])
        model._hs = Hs
        m = t["mask"][:, 1:]
        Ls = _next_losses(model, t, m, 0.0, np.random.default_rng(0))
        lp = ((model.past(E) - past) ** 2).mean()
        lf = ((model.head_fut[-1](E) - fut) ** 2).mean()
        lid = sum(float(F.cross_entropy(h(E), t["id_tgt"][:, jj])) for jj, h in enumerate(model.head_id))
        luc = float(((model.head_unc(E) - t["unc_tgt"]) ** 2).mean())
        tot += float(sum(Ls) + lp + lf + lid + luc)
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def embed(model, seqs, ks, vs):
    model.eval()
    out = []
    mout = []
    eb = max(32, BATCH)
    for i in range(0, len(ks), eb):
        bk = ks[i : i + eb]
        fcuts = [len(seqs[k][5]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in bk]
        t, _, qg, lens, _f = collate(seqs, bk, fcuts, fq, vs, False, None, None)
        Hs, E, _, MB, IB, RB, UB, CB, OB, RelB = model(t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"])
        out.append(E.cpu().numpy()); mout.append(np.hstack([MB.cpu().numpy(), IB.cpu().numpy(), RB.cpu().numpy(), UB.cpu().numpy(), CB.cpu().numpy(), OB.cpu().numpy(), RelB.cpu().numpy(), model.past(E).cpu().numpy()]))
    return np.concatenate(out, 0), np.concatenate(mout, 0)


def table_features(A, hold):
    amap = {r["customer_key"]: r for r in A.to_dicts()}
    rows = [amap[k] for k in hold]
    feat = [
        c for c in A.columns if not c.startswith("y_") and c not in ("customer_key", "signup_ts")
    ]
    X = pl.DataFrame([{c: r[c] for c in feat} for r in rows]).to_dummies(
        columns=[
            c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in feat
        ],
        drop_first=True,
    )
    labels = {c: np.array([r[c] for r in rows], float) for c in LABELS}
    seg = np.array([str(r["segment"]) for r in rows], dtype=object)
    return X.to_numpy().astype(float), labels, seg


def train_encoder(data, seed=0, d=None, layers=2, epochs=5, lr=None, batch=0):
    global AS_OF_TS, BATCH
    B, A, keys, seqs, vs, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = load_data(data)
    AS_OF_TS = as_of_ts
    if d is None:
        from collections import Counter as _C
        n_seqs = len(keys)
        total = sum(len(seqs[k][5]) for k in keys)
        _cnt = _C()
        for k in keys:
            _cnt.update(seqs[k][0].tolist())
        _tot = sum(_cnt.values())
        _H = -sum((c / _tot) * math.log2(c / _tot) for c in _cnt.values())
        d, _dn = derive_dim(n_seqs, total, [vs["et"]] + list(vs["gcat"]), _H)
        print(f"derived dim {d} ({_dn['dim']})", flush=True)
    if not batch:
        seq_len = int(np.median([len(seqs[k][5]) for k in keys]))
        batch = derive_batch(len(keys), seq_len)[0]
        batch = max(8, int(batch * (128 / d) ** 2))  # capacity-squared token bound (disclosed)
    BATCH = batch
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    nt = int(0.7 * len(keys))
    hold = [keys[i] for i in perm[nt:]]
    tr = [keys[i] for i in perm[:nt]]
    r2 = np.random.default_rng(seed + 100)
    vp = r2.permutation(len(tr))
    vn = max(20, int(0.1 * len(tr)))
    val = [tr[i] for i in vp[:vn]]
    tr = [tr[i] for i in vp[vn:]]
    pkeys = tr[: min(len(tr), 400)]
    fcuts = [len(seqs[k][5]) for k in pkeys]
    fq = [(AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in pkeys]
    if d is None:
        from collections import Counter as _C
        n_seqs = len(keys)
        total = sum(len(seqs[k][5]) for k in keys)
        _cnt = _C()
        for k in keys:
            _cnt.update(seqs[k][0].tolist())
        _tot = sum(_cnt.values())
        _H = -sum((c / _tot) * math.log2(c / _tot) for c in _cnt.values())
        d, _dn = derive_dim(n_seqs, total, [vs["et"]] + list(vs["gcat"]), _H)
        print(f"derived dim {d} ({_dn['dim']})", flush=True)
    _, P, _, _, _ = collate(seqs, pkeys, fcuts, fq, vs, True, None, None)
    P = P.cpu().numpy()
    past_mu, past_sd = P.mean(0), P.std(0)
    past_sd[past_sd < 1e-6] = 1.0
    Fs = []
    for hv in vs["horizons"]:
        rows = []
        for fr in (0.3, 0.5, 0.7, 0.9):
            mc = [max(8, int(fr * len(seqs[k][5]))) for k in pkeys]
            _, _, _, _, Fk = collate(seqs, pkeys, mc, [0.0] * len(pkeys), vs, False, None, None, True, float(hv), None, None)
            rows.append(Fk.cpu().numpy())
        Fs.append(np.concatenate(rows))
    Fcat = np.stack(Fs)
    fut_mu, fut_sd = Fcat.mean(1), Fcat.std(1)
    fut_sd[fut_sd < 1e-6] = 1.0
    ident_map, _ = _ident_targets(seqs, keys, vs)
    torch.manual_seed(seed)
    model = Enc(vs, d, len(taus), layers, taus).to(DEV)
    lr = lr if lr is not None else 1e-3  # documented seed bound; trust-region adapts below
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    log_s = nn.Parameter(torch.zeros(1 + len(vs["gcat"]) + vs["gnum"] + 1 + 1 + 1 + len(vs["id_dims"]) + 1 + 1, device=DEV))
    opt.add_param_group({"params": [log_s]})
    tr_log = []  # trust-region lr trajectory (receipt)
    best = 1e9
    best_state = None
    pat = 0
    hist = []
    for ep in range(epochs):
        eff = lr * (0.5 if ep == 0 else 1.0)  # one warm epoch at half step
        for g in opt.param_groups:
            g["lr"] = eff
        trl = run_epoch(
            model,
            seqs,
            tr,
            opt,
            None,
            log_s,
            rng,
            vs,
            past_mu,
            past_sd,
            p_mask,
            taus,
            fut_mu,
            fut_sd,
            None,
            True,
            ident_map=ident_map,
        )
        vl = val_loss(model, seqs, val, vs, past_mu, past_sd, taus, fut_mu, fut_sd, horizons)
        print(f"seed {seed} epoch {ep + 1} train {trl:.4f} val {vl:.4f}", flush=True)
        if not math.isfinite(trl) or not math.isfinite(vl):
            raise RuntimeError(f"divergence (hard failure, never masked): train {trl} val {vl}")
        hist.append(vl)
        _h = np.array(hist[-5:])
        _mad = float(np.median(np.abs(_h - np.median(_h)))) if len(_h) >= 3 else 0.0
        tol = max(1.4826 * _mad / np.sqrt(len(_h)), 1e-3)
        improved = vl < best - tol
        if improved:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        lr = min(lr * 1.5, 1e-2) if improved else max(lr * 0.5, 1e-5)
        tr_log.append((ep + 1, round(eff, 6), round(vl, 4)))
        if pat >= 3:
            break
    print(f"trust-region lr {tr_log}", flush=True)
    _names = ["et"] + [f"cat{j}" for j in range(len(vs["gcat"]))] + [f"num{j}" for j in range(vs["gnum"])] + ["gap", "fut"] + [f"id{j}" for j in range(len(vs["id_dims"]))] + ["unc", "ident"]
    _ls = log_s.detach().cpu().tolist()
    print("loss weights " + ", ".join(f"{n}={math.exp(-x):.3f}" for n, x in zip(_names, _ls)), flush=True)
    print(f"clip stats: {CLIP_STATS}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    torch.cuda.empty_cache()
    E, BLK = embed(model, seqs, hold, vs)
    Xtab, labels, seg = table_features(A, hold)
    return E, Xtab, labels, hold, seg, model, vs, B, best, BLK
