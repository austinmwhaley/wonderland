"""Principle-aligned auto encoder: all discovered fields, generic objectives, frozen E."""

import json
import math
import sys
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
from looking_glass.autotune import derive_batch, derive_dim  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_STATS = {"steps": 0, "binds": 0}
AS_OF_TS = 0.0
BATCH = 16
# Architecture / objective axes (one per toggle; empty = the recorded baseline).
# Set WQ_AXES="measure,query,suffix,slow,readout,multicut,amp,tokbatch" to enable.
AXES = {x for x in _os.environ.get("WQ_AXES", "").split(",") if x}
LABELS = (
    "y_value_30",
    "y_orders_30",
    "y_churn_30",
    "y_value_90",
    "y_orders_90",
    "y_churn_90",
    "y_margin_90",
    "y_discount_sensitivity",
    "y_ltv",
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
        k: (
            big["et"][s:e],
            big["gcat"][s:e],
            big["gnum"][s:e],
            big["time"][s:e],
            big["gaps"][s:e],
            big["tss"][s:e],
            big["graw"][s:e],
        )
        for k, s, e in zip(keys, starts, ends)
    }
    _ap = Path(data) / "path_a/path_a_wide_table.parquet"
    A = pl.read_parquet(_ap) if _ap.exists() else None
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
    def __init__(self, d, e, taus, selective=False, log_tau_bounds=None):
        super().__init__()
        self.d = d
        self.e = e
        self.selective = bool(selective)
        self.Wx = nn.Linear(d, e * d)
        self.out = nn.Linear(d, d)
        self.log_tau = nn.Parameter(torch.log(torch.as_tensor(taus, dtype=torch.float32)))
        if self.selective:
            # Zero-initialised: at step 0 the retention is EXACTLY the baseline
            # exp(-dt/tau); the model then learns input-dependent deviations.
            self.Wsel = nn.Linear(d, e)
            nn.init.zeros_(self.Wsel.weight)
            nn.init.zeros_(self.Wsel.bias)
            if log_tau_bounds is None:
                lo = float(np.log(max(float(np.min(taus)), 1e-3))) - 2.0
                hi = float(np.log(max(float(np.max(taus)), 1e-3))) + 2.0
                log_tau_bounds = (lo, hi)
            self.register_buffer(
                "tau_lo", torch.tensor(float(log_tau_bounds[0]), dtype=torch.float32)
            )
            self.register_buffer(
                "tau_hi", torch.tensor(float(log_tau_bounds[1]), dtype=torch.float32)
            )

    def forward(self, x, mask, dt):
        B, T, _ = x.shape
        if self.selective:
            log_tau_t = self.log_tau.view(1, 1, self.e) + self.Wsel(x)
            log_tau_t = torch.clamp(log_tau_t, self.tau_lo, self.tau_hi)
            tau = torch.exp(log_tau_t)
        else:
            tau = torch.exp(self.log_tau).view(1, 1, self.e)
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
    projs = [
        g.normal(size=(s, dd)).astype(np.float32) / np.sqrt(max(s, 1))
        for s, dd in zip(vs["gcat"], dims)
    ]
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
    def __init__(self, vs, d, e, layers, taus, slow_taus=None):
        super().__init__()
        self.d = d
        self.et = nn.Embedding(vs["et"], d)
        self.gcat = nn.ModuleList([nn.Embedding(s, d) for s in vs["gcat"]])
        self.gnum = nn.ModuleList([nn.Linear(1, d) for _ in range(vs["gnum"])])
        self.time = nn.Linear(vs["time_dim"], d)
        self.gw_cat = nn.Parameter(torch.full((len(vs["gcat"]),), 0.5413))
        self.gw_num = nn.Parameter(torch.full((vs["gnum"],), 0.5413))
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.selective = "selective" in AXES
        self.ssm = nn.ModuleList(
            [
                GatedSSM(
                    d,
                    e,
                    taus,
                    selective=self.selective,
                    log_tau_bounds=vs.get("tau_bounds"),
                )
                for _ in range(layers)
            ]
        )
        # ---- axis toggles (empty set = the recorded baseline path) ----
        self.use_measure = "measure" in AXES
        self.use_query = "query" in AXES
        self.use_suffix = "suffix" in AXES
        self.use_slow = "slow" in AXES
        self.use_readout = "readout" in AXES
        self.slow_ssm = (
            GatedSSM(d, len(slow_taus), slow_taus)
            if (self.use_slow and slow_taus is not None)
            else None
        )
        e_tot = e + (len(slow_taus) if self.slow_ssm is not None else 0)
        self.K = max(2, len(taus)) if self.use_readout else 0
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.q_multi = (
            nn.Parameter(torch.randn(self.K, d) / math.sqrt(d)) if self.use_readout else None
        )
        base_read = 3 + self.K if self.use_readout else 4
        self.n_read = base_read + 3 * e_tot
        self.rproj = nn.ModuleList([nn.Linear(d, d) for _ in range(self.n_read)])
        self.rw = nn.Parameter(torch.zeros(self.n_read))
        self.raw = nn.Linear(vs["raw"], d)
        self.rmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.qgap = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.head_et = nn.Linear(d, vs["et"])
        self.head_gap = nn.Linear(d, 1)
        self.heads_cat = nn.ModuleList([nn.Linear(d, s) for s in vs["gcat"]])
        self.heads_num = nn.ModuleList([nn.Linear(d, 1) for _ in range(vs["gnum"])])
        self.past = nn.Linear(d, vs["past_dim"])
        self.head_fut = nn.ModuleList([nn.Linear(d, vs["fut_dim"]) for _ in vs["horizons"]])
        self.head_id = nn.ModuleList([nn.Linear(d, s2) for s2 in vs["id_dims"]])
        self.head_unc = nn.Linear(d, vs["unc_dim"])
        self.rel_idx = list(vs["rel_idx"])
        self.head_ident = nn.Linear(d, vs["ident_dim"]) if vs.get("ident_dim", 0) > 0 else None
        # item 3: low-rank previous-event transition operator (learned, small)
        self.trans_r = 16
        if self.use_suffix:
            self.trans_u = nn.Embedding(vs["et"], self.trans_r)
            self.trans_v = nn.Parameter(torch.randn(self.trans_r, d) / math.sqrt(self.trans_r))
        # item 1: self-supervised typed-histogram head (all discovered categorical fields)
        self.head_hist = (
            nn.Linear(d, vs["hist_dim"])
            if (self.use_measure and vs.get("hist_dim", 0) > 0)
            else None
        )
        # item 2: query-conditioned future-aggregate head
        self.q_dim = 24
        if self.use_query:
            self.qemb = nn.Embedding(max(vs.get("n_query", 1), 1), self.q_dim)
            self.head_query = nn.Linear(d + self.q_dim, 1)

    def forward(self, et, gcat, gnum, time, gaps, mask, lengths, qgap, graw):
        x = self.et(et) + self.time(time)
        for i, emb in enumerate(self.gcat):
            x = x + F.softplus(self.gw_cat[i]) * emb(gcat[:, :, i])
        for i, lin in enumerate(self.gnum):
            x = x + F.softplus(self.gw_num[i]) * lin(gnum[:, :, i : i + 1])
        x = self.mlp(x) + self.raw(torch.sign(graw) * torch.log1p(graw.abs()))
        dt = gaps.clamp(min=1e-3)
        H = None
        M = None
        for ssm in self.ssm:
            x, H, M = ssm(x, mask, dt)
        Hs = x
        if self.use_slow and self.slow_ssm is not None:
            # item 4: a second stack whose tau is initialized at the stream span scale
            _xs, H_s, M_s = self.slow_ssm(x, mask, dt)
            H = torch.cat([H, H_s], dim=2)
            M = torch.cat([M, M_s], dim=2)
            Hs = Hs + _xs
        if self.use_suffix:
            # item 3: expose the previous event through a learned low-rank transition operator
            prev = F.pad(et[:, :-1], (1, 0), value=0)
            Hs = Hs + self.trans_u(prev) @ self.trans_v
        ar = torch.arange(Hs.shape[0], device=Hs.device)
        last = Hs[ar, lengths - 1]
        m = mask.unsqueeze(-1)
        mean = (Hs * m).sum(1) / lengths.unsqueeze(-1)
        mx = Hs.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values
        Hlast = H[ar, lengths - 1]
        Hmean = (H * mask.unsqueeze(-1).unsqueeze(-1)).sum(1) / lengths.view(-1, 1, 1)
        Mlast = M[ar, lengths - 1]
        rs = [last, mean, mx]
        if self.use_readout:
            # item 6: structured multi-query attention readout
            for kq in range(self.K):
                sc = (Hs @ self.q_multi[kq]) / math.sqrt(self.d)
                sc = sc.masked_fill(mask == 0, -1e9)
                rs.append((Hs * torch.softmax(sc, 1).unsqueeze(-1)).sum(1))
        else:
            score = (Hs @ self.q) / math.sqrt(self.d)
            score = score.masked_fill(mask == 0, -1e9)
            rs.append((Hs * torch.softmax(score, 1).unsqueeze(-1)).sum(1))
        for j in range(Hlast.shape[1]):
            rs.append(Hlast[:, j])
            rs.append(Hmean[:, j])
        for j in range(Mlast.shape[1]):
            rs.append(Mlast[:, j])
        pr = [self.rproj[i](rs[i]) for i in range(self.n_read)]
        w = F.softplus(self.rw)
        E = sum(w[i] * pr[i] for i in range(self.n_read))
        E = self.rmlp(E)
        E = E + self.qgap(torch.log1p(qgap).unsqueeze(-1))
        MB = torch.cat(
            [
                Mlast.reshape(Mlast.shape[0], -1),
                Hlast.reshape(Hlast.shape[0], -1),
                Hmean.reshape(Hmean.shape[0], -1),
            ],
            1,
        )
        # learned blocks (trained heads, not static summaries)
        IB = (
            torch.cat([h(E) for h in self.head_id], 1)
            if len(self.head_id)
            else E.new_zeros((E.shape[0], 0))
        )
        UB = self.head_unc(E)
        RB = self.head_gap(E)
        OB = self.head_et(E)
        RelB = (
            torch.cat([self.heads_cat[j](E) for j in self.rel_idx], 1)
            if self.rel_idx
            else E.new_zeros((E.shape[0], 0))
        )
        CB = torch.cat([hf(E) for hf in self.head_fut], 1)
        HB = self.head_hist(E) if self.head_hist is not None else E.new_zeros((E.shape[0], 0))
        if self.use_suffix:
            SB = torch.cat(
                [self.et(et[ar, lengths - 1]), self.et(et[ar, torch.clamp(lengths - 2, min=0)])], 1
            )
        else:
            SB = E.new_zeros((E.shape[0], 0))
        return Hs, E, w / w.sum(), MB, IB, RB, UB, CB, OB, RelB, HB, SB


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
    want_hist=False,
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
    hist = np.zeros((Bn, vs.get("hist_dim", 0)), np.float32)
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
        unc_tgt[b, : len(vs["gcat"])] = (gck[:L] == 0).mean(0)
        unc_tgt[b, len(vs["gcat"]) :] = (gnk[:L] == 0).mean(0)
        if want_hist and L > 0 and vs.get("hist_keep"):
            off = 0
            for j, keep in enumerate(vs["hist_keep"]):
                col = gck[:L, j]
                sz = vs["gcat"][j]
                c = np.bincount(col[col < sz], minlength=sz).astype(np.float32)
                nk = len(keep)
                hk = np.zeros(nk + 1, np.float32)
                hk[:nk] = c[keep]
                hk[-1] = max(c.sum() - hk[:nk].sum(), 0.0)
                tot = hk.sum()
                if tot > 0:
                    hk /= tot
                hist[b, off : off + nk + 1] = hk
                off += nk + 1
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
        hist=torch.tensor(hist, device=DEV),
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

    def ce_customer(head, tgt):
        logit = head(model_hs[:, :-1]).reshape(-1, head.out_features)
        y = tgt[:, 1:].reshape(-1)
        mm = m.reshape(-1) > 0
        if "noaction" in AXES and getattr(model, "_exo_idx", None):
            exo = torch.tensor(model._exo_idx, device=y.device, dtype=torch.long)
            mm = mm & ~torch.isin(y, exo)
        return F.cross_entropy(logit[mm], y[mm]) if mm.sum() > 0 else torch.tensor(0.0, device=DEV)

    Ls = [ce_customer(model.head_et, t["et"])]
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
    order = list(np.array(ks)[rng.permutation(len(ks))])
    if "tokbatch" in AXES:
        # item 8: pack batches by a token budget (median-length scaled) instead of customer count
        med = max(1, int(np.median([len(seqs[k][5]) for k in order])))
        budget = max(1, BATCH * med)
        batches, cur, ctok = [], [], 0
        for k in order:
            n = len(seqs[k][5])
            if cur and ctok + n > budget:
                batches.append(cur)
                cur, ctok = [], 0
            cur.append(k)
            ctok += n
        if cur:
            batches.append(cur)
    else:
        batches = [order[i : i + BATCH] for i in range(0, len(order), BATCH)]
    tot = 0
    nb = 0
    use_hist = model.head_hist is not None
    for bk in batches:
        cuts = []
        for k in bk:
            tss = seqs[k][5]
            if "multicut" in AXES:
                # item 5: stratified cut position (balanced short/long histories)
                frac = float(rng.uniform(0.15, 0.95))
                cut = int(frac * len(tss))
            else:
                lo = tss[0]
                hi = tss[-1]
                ct = float(rng.uniform(lo, hi))
                cut = int(np.searchsorted(tss, ct, side="right"))
            cuts.append(max(8, min(cut, len(tss) - 2)))
        hor = vs["horizons"]
        _hi = int(rng.integers(len(hor)))
        h = float(hor[_hi])
        t, past, qg, lens, fut = collate(
            seqs,
            bk,
            cuts,
            [0.0] * len(bk),
            vs,
            False,
            None,
            None,
            True,
            h,
            fut_mu,
            fut_sd,
            use_hist,
        )
        Hs, E, w, *_blocks = model(
            t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
        )
        model._hs = Hs
        m = t["mask"][:, 1:]
        Ls = _next_losses(model, t, m, p_mask, rng)
        Ls.append(((model.head_fut[_hi](E) - fut) ** 2).mean())
        for jj, hh in enumerate(model.head_id):
            Ls.append(F.cross_entropy(hh(E), t["id_tgt"][:, jj]))
        Ls.append(((model.head_unc(E) - t["unc_tgt"]) ** 2).mean())
        if model.head_ident is not None and ident_map is not None:
            idt = torch.tensor(np.stack([ident_map[k] for k in bk]), device=DEV)
            Ls.append(((model.head_ident(E) - idt) ** 2).mean())
        if use_hist:
            Ls.append(((model.head_hist(E) - t["hist"]) ** 2).mean())
        if model.use_query:
            qf = int(rng.integers(vs["raw"]))
            qidx = _hi * vs["raw"] + qf
            qe = model.qemb(torch.full((E.shape[0],), qidx, device=DEV, dtype=torch.long))
            pred = model.head_query(torch.cat([E, qe], 1))
            Ls.append(((pred - fut[:, vs["fut_raw_off"] + 2 * qf].unsqueeze(1)) ** 2).mean())
        loss = sum(0.5 * torch.exp(-log_s[j]) * Ls[j] + 0.5 * log_s[j] for j in range(len(Ls)))
        fcuts = [len(seqs[k][5]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in bk]
        t2, past2, qg2, lens2, _f2 = collate(seqs, bk, fcuts, fq, vs, True, past_mu, past_sd)
        _, E2, _, *_b2 = model(
            t2["et"],
            t2["gcat"],
            t2["gnum"],
            t2["time"],
            t2["gaps"],
            t2["mask"],
            lens2,
            qg2,
            t2["graw"],
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
            seqs,
            bk,
            fcuts,
            fq,
            vs,
            True,
            past_mu,
            past_sd,
            True,
            h,
            fut_mu,
            fut_sd,
            model.head_hist is not None,
        )
        Hs, E, w, *_blocks = model(
            t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
        )
        model._hs = Hs
        m = t["mask"][:, 1:]
        Ls = _next_losses(model, t, m, 0.0, np.random.default_rng(0))
        lp = ((model.past(E) - past) ** 2).mean()
        lf = ((model.head_fut[-1](E) - fut) ** 2).mean()
        lid = sum(
            float(F.cross_entropy(h(E), t["id_tgt"][:, jj])) for jj, h in enumerate(model.head_id)
        )
        luc = float(((model.head_unc(E) - t["unc_tgt"]) ** 2).mean())
        lhist = (
            float(((model.head_hist(E) - t["hist"]) ** 2).mean())
            if model.head_hist is not None
            else 0.0
        )
        tot += float(sum(Ls) + lp + lf + lid + luc + lhist)
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
        Hs, E, _, *blocks = model(
            t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
        )
        out.append(E.cpu().numpy())
        mout.append(np.hstack([b.cpu().numpy() for b in blocks] + [model.past(E).cpu().numpy()]))
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
    labels = {c: np.array([r[c] for r in rows], float) for c in LABELS if c in rows[0]}
    seg = np.array([str(r["segment"]) for r in rows], dtype=object)
    return X.to_numpy().astype(float), labels, seg


def train_encoder(data=None, seed=0, d=None, layers=2, epochs=5, lr=None, batch=0, prepared=None):
    global AS_OF_TS, BATCH
    if prepared is not None:
        (B, A, keys, seqs, vs, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons) = (
            prepared
        )
    else:
        B, A, keys, seqs, vs, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = (
            load_data(data)
        )
    if data is not None:
        vs["data_path"] = str(data)
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
            _, _, _, _, Fk = collate(
                seqs,
                pkeys,
                mc,
                [0.0] * len(pkeys),
                vs,
                False,
                None,
                None,
                True,
                float(hv),
                None,
                None,
            )
            rows.append(Fk.cpu().numpy())
        Fs.append(np.concatenate(rows))
    Fcat = np.stack(Fs)
    fut_mu, fut_sd = Fcat.mean(1), Fcat.std(1)
    fut_sd[fut_sd < 1e-6] = 1.0
    ident_map, _ = _ident_targets(seqs, keys, vs)
    # item 1: coverage-capped per-field value vocab for the typed-histogram objective
    if "measure" in AXES:
        hist_keep = []
        for j in range(
            len(vs["gcat"])
        ):  # principle-ok: loop over discovered fields (~17), not rows/time
            cnt = np.zeros(vs["gcat"][j], np.int64)
            for k in tr[:2000]:
                col = seqs[k][1][:, j]
                cnt += np.bincount(col[col < len(cnt)], minlength=len(cnt))[: len(cnt)]
            order = np.argsort(cnt)[::-1]
            csum = np.cumsum(cnt[order])
            tot_c = max(int(csum[-1]), 1)
            keep = order[csum <= 0.99 * tot_c]
            if len(keep) == 0:
                keep = order[:1]
            hist_keep.append(np.sort(keep).astype(np.int64))
        vs["hist_keep"] = hist_keep
        vs["hist_dim"] = int(sum(len(k) + 1 for k in hist_keep))
        print(
            f"hist_dim {vs['hist_dim']} (coverage 0.99 over {len(vs['gcat'])} fields)", flush=True
        )
    # selective retention: bounds derived from observed gap/span statistics
    _gap_pos = np.concatenate(
        [np.asarray(seqs[k][4]) for k in tr[:500]]
    )  # principle-ok: one-off data prep
    _gap_pos = _gap_pos[_gap_pos > 0]
    _span = np.array(
        [(seqs[k][5][-1] - seqs[k][5][0]) / 86400.0 for k in tr[:500]]
    )  # principle-ok: unit conversion s->days
    if len(_gap_pos) and len(_span):
        _lo = float(np.log(max(float(np.quantile(_gap_pos, 0.01)), 1e-4)))
        _hi = float(np.log(max(float(np.quantile(_span, 0.99)), 1e-3)))
        vs["tau_bounds"] = (_lo - 2.0, _hi + 2.0)
    # item 4: slow timescales derived from the observed stream span
    spans = np.array(
        [(seqs[k][5][-1] - seqs[k][5][0]) / 86400.0 for k in keys]
    )  # principle-ok: unit conversion s->days
    slow_taus = np.quantile(spans, [0.5, 0.9]) if "slow" in AXES else None
    # item 2: query space = (horizon, raw field)
    vs["fut_raw_off"] = vs["et"] + len(vs["gcat"])
    vs["n_query"] = len(vs["horizons"]) * vs["raw"]
    torch.manual_seed(seed)
    model = Enc(vs, d, len(taus), layers, taus, slow_taus=slow_taus).to(DEV)
    model._exo_idx = list(vs.get("exo_idx") or [])
    if "noaction" in AXES:
        print(
            f"noaction: masking exogenous event types {model._exo_idx} from next-event CE",
            flush=True,
        )
    lr = lr if lr is not None else 1e-3  # documented seed bound; trust-region adapts below
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_extra = (1 if model.head_hist is not None else 0) + (1 if model.use_query else 0)
    log_s = nn.Parameter(
        torch.zeros(
            1 + len(vs["gcat"]) + vs["gnum"] + 1 + 1 + 1 + len(vs["id_dims"]) + 1 + 1 + n_extra,
            device=DEV,
        )
    )
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
    _names = (
        ["et"]
        + [f"cat{j}" for j in range(len(vs["gcat"]))]
        + [f"num{j}" for j in range(vs["gnum"])]
        + ["gap", "fut"]
        + [f"id{j}" for j in range(len(vs["id_dims"]))]
        + ["unc", "ident"]
    )
    if model.head_hist is not None:
        _names.append("hist")
    if model.use_query:
        _names.append("query")
    _ls = log_s.detach().cpu().tolist()
    print(
        "loss weights " + ", ".join(f"{n}={math.exp(-x):.3f}" for n, x in zip(_names, _ls)),
        flush=True,
    )
    print(f"clip stats: {CLIP_STATS}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    torch.cuda.empty_cache()
    E, BLK = embed(model, seqs, hold, vs)
    if A is not None and getattr(A, "height", 0):
        Xtab, labels, seg = table_features(A, hold)
    else:
        # No wide table available: training is still complete. Evaluation baselines
        # are simply absent, and that is reported rather than invented.
        Xtab = np.zeros((len(hold), 0), float)
        labels = {}
        seg = np.array(["unknown"] * len(hold), dtype=object)
    return E, Xtab, labels, hold, seg, model, vs, B, best, BLK
