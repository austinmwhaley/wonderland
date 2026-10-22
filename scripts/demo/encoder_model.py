"""Sequence-only experiment v2: stable training, 3 seeds, diagnostics.

No measure/counter/summary in E. Compare E vs PATH A on value/orders/churn.
"""

import argparse, collections, datetime, json, math, sys, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, polars as pl, torch, torch.nn as nn, torch.nn.functional as F
from looking_glass.cfm_model import _scan
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mass", default="explicit", choices=["explicit", "none", "state", "readout"])
    ap.add_argument("--max-customers", type=int, default=0, dest="max_customers")
    ap.add_argument("--nce", type=int, default=1)
    ap.add_argument("--aug-p", type=float, default=-1.0, dest="aug_p")
    return ap.parse_args()


AS_OF = "2025-01-01"
AS_OF_TS = __import__("datetime").datetime.fromisoformat(AS_OF).timestamp()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _vocab(vals):
    u = sorted({str(v) for v in vals if v is not None and str(v) != ""})
    m = {v: i + 1 for i, v in enumerate(u)}
    m[""] = 0
    return m


def load_data(data):
    B = pl.read_parquet(Path(data) / "path_b/path_b_customer_event_stream.parquet")
    A = pl.read_parquet(Path(data) / "path_a/path_a_wide_table.parquet")
    as_of = pl.lit(AS_OF).str.to_datetime()
    B = (
        B.with_columns(pl.col("event_ts").str.to_datetime().alias("ts"))
        .filter(pl.col("ts") <= as_of)
        .sort(["customer_key", "ts"])
    )
    keys = sorted(B["customer_key"].unique().to_list())
    if ARGS.max_customers:
        keys = keys[: ARGS.max_customers]
    B = B.filter(pl.col("customer_key").is_in(keys))
    rows = B.select(["customer_key", "ts", "event_type", "event_attributes"]).to_dicts()
    et_v = _vocab({r["event_type"] for r in rows})
    attrs = [json.loads(r["event_attributes"]) for r in rows]
    prod_v = _vocab({a.get("product_id") for a in attrs})
    cat_v = _vocab({a.get("category_code") for a in attrs})
    brand_v = _vocab({a.get("brand") for a in attrs})
    camp_v = _vocab({a.get("campaign_id") for a in attrs})
    touch_v = _vocab({a.get("touch") for a in attrs})
    acq_v = _vocab({a.get("acquisition_channel") for a in attrs})
    region_v = _vocab({a.get("region") for a in attrs})
    hand_cat = {
        "product_id",
        "category_code",
        "brand",
        "campaign_id",
        "touch",
        "acquisition_channel",
        "region",
    }
    hand_num = {"value", "margin", "price", "qty", "discount"}
    allk = set().union(*[set(a) for a in attrs]) - {"target_key"}
    extra = sorted(allk - hand_cat - hand_num - {"order_id"})
    GCAT = []
    GNUM = []
    for f in extra:
        nn = [a.get(f) for a in attrs if a.get(f) is not None and str(a.get(f)) != ""]
        is_ts = bool(nn) and all(isinstance(v, str) for v in nn[:200])
        if is_ts:
            for v in nn[:200]:
                try:
                    datetime.datetime.fromisoformat(str(v))
                except Exception:
                    is_ts = False
                    break
        if is_ts:
            GNUM.append({"field": f, "kind": "ts"})
            continue
        if nn and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nn):
            u = len(set(map(str, nn)))
            if isinstance(nn[0], int) and u <= 2 * int(math.sqrt(len(rows))):
                GCAT.append({"field": f})
                continue
            GNUM.append({"field": f, "kind": "num"})
            continue
        GCAT.append({"field": f})
    GCAT_SPEC = []
    for g in GCAT:
        f = g["field"]
        cnt = collections.Counter(
            str(a.get(f)) for a in attrs if a.get(f) is not None and str(a.get(f)) != ""
        )
        if len(cnt) > 2 * int(math.sqrt(len(rows))):
            buckets = 1 << max(4, int(math.ceil(math.log2(2 * math.sqrt(len(rows))))))
            GCAT_SPEC.append({"field": f, "mode": "hash", "buckets": buckets, "size": buckets + 1})
        else:
            m = {}
            total = sum(cnt.values())
            acc = 0
            i = 1
            for v, c in cnt.most_common():
                if total and acc / total >= 0.99:
                    break
                m[v] = i
                i += 1
                acc += c
            GCAT_SPEC.append({"field": f, "mode": "vocab", "map": m, "size": i})
    by = {}
    for r, a in zip(rows, attrs):
        by.setdefault(r["customer_key"], []).append((r["ts"], r["event_type"], a))
    seqs = {}
    for k, evs in by.items():
        evs.sort(key=lambda x: x[0])
        n = len(evs)
        arr = {
            c: np.zeros(n, np.int64)
            for c in ("et", "prod", "cat", "brand", "camp", "touch", "acq", "region")
        }
        num = np.zeros((n, 6), np.float32)
        raw = np.zeros((n, 4), np.float32)
        mask = np.ones(n, np.float32)
        tss = np.zeros(n, np.float64)
        prev = None
        gcat = np.zeros((n, len(GCAT_SPEC)), np.int64)
        gnum = np.zeros((n, len(GNUM)), np.float32)
        for i, (ts, et, at) in enumerate(evs):
            arr["et"][i] = et_v.get(et, 0)
            arr["prod"][i] = prod_v.get(str(at.get("product_id") or ""), 0)
            arr["cat"][i] = cat_v.get(str(at.get("category_code") or ""), 0)
            arr["brand"][i] = brand_v.get(str(at.get("brand") or ""), 0)
            arr["camp"][i] = camp_v.get(str(at.get("campaign_id") or ""), 0)
            arr["touch"][i] = touch_v.get(str(at.get("touch") or ""), 0)
            arr["acq"][i] = acq_v.get(str(at.get("acquisition_channel") or ""), 0)
            arr["region"][i] = region_v.get(str(at.get("region") or ""), 0)
            v = float(at.get("value") or 0)
            m = float(at.get("margin") or 0)
            p = float(at.get("price") or 0)
            q = float(at.get("qty") or 0)
            dsc = float(at.get("discount") or 0)
            tss[i] = ts.timestamp()
            gap = 0.0 if prev is None else max(0.0, (ts - prev).total_seconds() / 86400.0)
            prev = ts
            sl = lambda x: math.copysign(math.log1p(abs(x)), x)
            num[i] = [sl(v), sl(m), sl(p), q / 3.0, dsc, math.log1p(gap)]
            raw[i] = [v, m, p * q, 1.0]
            for j, g in enumerate(GCAT_SPEC):
                s = "" if at.get(g["field"]) is None else str(at.get(g["field"]))
                gcat[i, j] = (
                    0
                    if s == ""
                    else (
                        1 + zlib.crc32((g["field"] + "=" + s).encode()) % g["buckets"]
                        if g["mode"] == "hash"
                        else g["map"].get(s, 0)
                    )
                )
            for j, g in enumerate(GNUM):
                val = at.get(g["field"])
                x = 0.0
                if val is not None and str(val) != "":
                    if g["kind"] == "ts":
                        try:
                            x = (
                                ts - datetime.datetime.fromisoformat(str(val))
                            ).total_seconds() / 86400.0
                        except Exception:
                            x = 0.0
                    else:
                        try:
                            x = float(val)
                        except Exception:
                            x = 0.0
                gnum[i, j] = sl(x) if x else 0.0
        seqs[k] = (arr, num, raw, mask, tss, gcat, gnum)
    vs = {
        "et": len(et_v),
        "prod": len(prod_v),
        "cat": len(cat_v),
        "brand": len(brand_v),
        "camp": len(camp_v),
        "touch": len(touch_v),
        "acq": len(acq_v),
        "region": len(region_v),
        "gcat": [g["size"] for g in GCAT_SPEC],
        "gnum": len(GNUM),
    }
    print(
        "[extra fields]",
        [(g["field"], g["mode"], g["size"]) for g in GCAT_SPEC],
        [(g["field"], g["kind"]) for g in GNUM],
        flush=True,
    )
    purchase_idx = et_v.get("purchase", -1)
    gaps = []
    for k in keys:
        tt = seqs[k][4]
        ets = seqs[k][0]["et"]
        pt = tt[ets == purchase_idx]
        if len(pt) > 1:
            gaps.extend((np.diff(pt) / 86400.0).tolist())
    gaps = np.array(gaps)
    h_min = max(1.0, float(np.percentile(gaps, 50)) if len(gaps) else 1.0)
    span = max((seqs[k][4][-1] - seqs[k][4][0]) / 86400.0 for k in keys)
    h_max = max(h_min, float(span))
    # --- data-derived periodicity (hourly event counts -> spectrum peaks) ---
    allt = np.concatenate([seqs[k][4] for k in keys])
    h0 = allt.min()
    hn = int(np.ceil((allt.max() - h0) / 3600.0)) + 1
    counts, _ = np.histogram(allt, bins=hn, range=(h0, h0 + hn * 3600.0))
    spec = np.abs(np.fft.rfft(counts - counts.mean())) ** 2
    freqs = np.fft.rfftfreq(hn, d=1.0)  # cycles/hour
    spec[0] = 0
    peaks = []
    for i in range(1, len(spec) - 1):
        if (
            spec[i] > spec[i - 1]
            and spec[i] >= spec[i + 1]
            and spec[i] > np.percentile(spec[1:], 90)
        ):
            peaks.append((spec[i], freqs[i]))
    peaks = sorted(peaks, reverse=True)[:2]
    periods_h = [1.0 / f for _, f in peaks if f > 0]
    n_periods = len(periods_h)
    n_experts = int(np.clip(max(2, len(peaks)), 2, 6))
    taus = np.quantile(gaps if len(gaps) else np.array([h_min]), np.linspace(0.25, 0.95, n_experts))
    # --- observed input missingness -> target mask rate ---
    miss = []
    for k in keys:
        arr = seqs[k][0]
        for c in ("prod", "cat", "brand", "camp", "touch"):
            miss.append(float((arr[c] == 0).mean()))
    p_mask = float(np.clip(np.mean(miss), 0.05, 0.40))
    # --- append Fourier features from the derived periods ---
    num_dim = 6 + 2 * n_periods
    for k in keys:
        arr, num, raw, mask, tss, gcat, gnum = seqs[k]
        n = len(tss)
        num2 = np.zeros((n, num_dim), np.float32)
        num2[:, :6] = num
        for j, P in enumerate(periods_h):
            ph = 2 * np.pi * (tss / 3600.0) / P
            num2[:, 6 + 2 * j] = np.sin(ph)
            num2[:, 7 + 2 * j] = np.cos(ph)
        seqs[k] = (arr, num2, raw, mask, tss, gcat, gnum)
    print(
        "[derived] periods_h",
        [round(x, 2) for x in periods_h],
        "n_experts",
        n_experts,
        "taus",
        [round(float(x), 2) for x in taus],
        "p_mask",
        round(p_mask, 3),
        flush=True,
    )
    return (
        B,
        A,
        keys,
        seqs,
        vs,
        et_v,
        cat_v,
        h_min,
        h_max,
        periods_h,
        n_experts,
        taus,
        p_mask,
        num_dim,
    )


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
        return F.gelu(self.out(y)) + y


class Enc(nn.Module):
    def __init__(self, vs, d, e, layers, taus, num_dim, mass_mode="explicit", past_dim=20):
        super().__init__()
        self.d = d
        self.et = nn.Embedding(vs["et"], d)
        self.prod = nn.Embedding(vs["prod"], d)
        self.cat = nn.Embedding(vs["cat"], d)
        self.brand = nn.Embedding(vs["brand"], d)
        self.camp = nn.Embedding(vs["camp"], d)
        self.touch = nn.Embedding(vs["touch"], d)
        self.acq = nn.Embedding(vs["acq"], d)
        self.region = nn.Embedding(vs["region"], d)
        self.num = nn.Linear(num_dim, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.gcat = nn.ModuleList([nn.Embedding(s, d) for s in vs["gcat"]])
        self.gnum = nn.Linear(vs["gnum"], d) if vs["gnum"] > 0 else None
        self.gw_cat = nn.Parameter(torch.full((len(vs["gcat"]),), 0.5413))
        self.gw_num = nn.Parameter(torch.tensor(0.5413)) if vs["gnum"] > 0 else None
        self.ssm = nn.ModuleList([GatedSSM(d, e, taus) for _ in range(layers)])
        self.mass_mode = mass_mode
        self.n_read = 4 if mass_mode == "none" else 5
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.rproj = nn.ModuleList([nn.Linear(d, d) for _ in range(self.n_read)])
        self.rw = nn.Parameter(torch.zeros(self.n_read))
        if mass_mode == "explicit":
            self.mag = nn.Linear(4, d)
        elif mass_mode == "state":
            self.mass_w = nn.Linear(d, d)
            self.mag = nn.Linear(d, d)
        elif mass_mode == "readout":
            self.mass_q = nn.Linear(d, 1)
            self.mass_v = nn.Linear(d, d)
            self.mag = nn.Linear(d, d)
        self.rmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.head_et = nn.Linear(d, vs["et"])
        self.head_cat = nn.Linear(d, vs["cat"])
        self.head_brand = nn.Linear(d, vs["brand"])
        self.head_touch = nn.Linear(d, vs["touch"])
        self.head_prod = nn.Linear(d, vs["prod"])
        self.head_val = nn.Linear(d, 1)
        self.head_gap = nn.Linear(d, 1)
        self.g_et = nn.Linear(d, vs["et"])
        self.g_val = nn.Linear(d, 1)
        self.g_gap = nn.Linear(d, 1)
        self.h_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.g_v_e = nn.Linear(d, 1)
        self.g_o_e = nn.Linear(d, 1)
        self.g_v_h = nn.Linear(d, 1)
        self.g_o_h = nn.Linear(d, 1)
        self.past = nn.Linear(d, past_dim)
        self.qgap = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.tensor(2.3))

    def forward(
        self,
        et,
        prod,
        cat,
        brand,
        camp,
        touch,
        acq,
        region,
        num,
        raw,
        mask,
        lengths,
        qgap,
        gcat,
        gnum,
        use_mag=True,
    ):
        x = (
            self.et(et)
            + self.prod(prod)
            + self.cat(cat)
            + self.brand(brand)
            + self.camp(camp)
            + self.touch(touch)
            + self.acq(acq)
            + self.region(region)
            + self.num(num)
        )
        for i, e in enumerate(self.gcat):
            x = x + F.softplus(self.gw_cat[i]) * e(gcat[:, :, i])
        if self.gnum is not None:
            x = x + F.softplus(self.gw_num) * self.gnum(gnum)
        x = self.mlp(x)
        dt = torch.expm1(num[:, :, 5]).clamp(min=1e-3)
        for s in self.ssm:
            x = s(x, mask, dt)
        Hs = x
        ar = torch.arange(Hs.shape[0], device=Hs.device)
        last = Hs[ar, lengths - 1]
        m = mask.unsqueeze(-1)
        mean = (Hs * m).sum(1) / lengths.unsqueeze(-1)
        mx = (Hs.masked_fill(mask.unsqueeze(-1) == 0, -1e9)).max(1).values
        score = (Hs @ self.q) / math.sqrt(self.d)
        score = score.masked_fill(mask == 0, -1e9)
        att = torch.softmax(score, 1)
        pooled = (Hs * att.unsqueeze(-1)).sum(1)
        rs = [last, mean, mx, pooled]
        if self.mass_mode == "explicit":
            mag = torch.cumsum(raw, 1)[ar, lengths - 1]
            mag = torch.sign(mag) * torch.log1p(mag.abs())
            mag = self.mag(mag)
            rs.append(mag if use_mag else torch.zeros_like(mag))
        elif self.mass_mode == "state":
            mass = (self.mass_w(x) * mask.unsqueeze(-1)).sum(1)
            rs.append(self.mag(mass))
        elif self.mass_mode == "readout":
            sc = self.mass_q(x).squeeze(-1).masked_fill(mask == 0, -1e9)
            ww = torch.softmax(sc, 1)
            mass = (ww.unsqueeze(-1) * self.mass_v(x)).sum(1)
            rs.append(self.mag(mass))
        pr = [self.rproj[i](rs[i]) for i in range(self.n_read)]
        w = F.softplus(self.rw)
        E = sum(w[i] * pr[i] for i in range(self.n_read))
        E = self.rmlp(E)
        E = E + self.qgap(torch.log1p(qgap).unsqueeze(-1))
        g = (w / w.sum()).unsqueeze(0).expand(E.shape[0], self.n_read)
        return Hs, E, g


def augment(seq, p, rng):
    arr, num, raw, _mask, tss, gcat, gnum = seq
    n = len(tss)
    if p <= 0 or n < 4:
        return seq
    keep = rng.random(n) > p
    keep[0] = True
    keep[-1] = True
    if keep.sum() < 4:
        return seq
    tss2 = tss[keep]
    arr2 = {c: arr[c][keep] for c in arr}
    num2 = num[keep].copy()
    raw2 = raw[keep].copy()
    gcat2 = gcat[keep]
    gnum2 = gnum[keep]
    gap = np.zeros(len(tss2), np.float64)
    gap[1:] = np.maximum(0.0, np.diff(tss2) / 86400.0)
    num2[:, 5] = np.log1p(gap)
    return (arr2, num2, raw2, np.ones(len(tss2), np.float32), tss2, gcat2, gnum2)


def collate(seqs, ks, cuts, hs, qgaps, purchase_idx, n_et, past_mu, past_sd, want_past):
    T = max(cuts)
    out = {
        c: np.zeros((len(ks), T), np.int64)
        for c in ("et", "prod", "cat", "brand", "camp", "touch", "acq", "region")
    }
    Dnum = seqs[ks[0]][1].shape[1]
    num = np.zeros((len(ks), T, Dnum), np.float32)
    raw = np.zeros((len(ks), T, 4), np.float32)
    mask = np.zeros((len(ks), T), np.float32)
    glob = np.zeros((len(ks), 3), np.float32)
    hor = np.zeros((len(ks), 2), np.float32)
    past = np.zeros((len(ks), n_et + 50), np.float32)
    qg = np.zeros(len(ks), np.float32)
    gcat = np.zeros((len(ks), T, seqs[ks[0]][5].shape[1]), np.int64)
    gnum = np.zeros((len(ks), T, seqs[ks[0]][6].shape[1]), np.float32)
    for b, (k, cut) in enumerate(zip(ks, cuts)):
        arr, nn_, rr, mm, tt, gc, gn = seqs[k]
        L = min(cut, len(mm))
        q = float(qgaps[b])
        qg[b] = q
        for c in out:
            out[c][b, :L] = arr[c][:L]
        num[b, :L] = nn_[:L]
        raw[b, :L] = rr[:L]
        mask[b, :L] = 1
        gcat[b, :L] = gc[:L]
        gnum[b, :L] = gn[:L]
        glob[b] = [arr["et"][L], nn_[L, 0], nn_[L, 5]] if L < len(mm) else [-1, 0, 0]
        ct = tt[L - 1] + q * 86400.0 if L > 0 else 0.0
        if L > 0 and hs[b] > 0:
            h = float(hs[b])
            fut = (tt > ct) & (tt <= ct + h * 86400.0) & (arr["et"] == purchase_idx)
            hor[b, 0] = math.log1p(max(float(rr[fut, 0].sum()), 0.0))
            hor[b, 1] = math.log1p(float(fut.sum()))
        if want_past and L > 0:
            e = arr["et"][:L]
            etc = [float((e == i).sum()) for i in range(n_et)]
            catd = [
                float(len(np.unique(arr[f][:L][arr[f][:L] > 0])))
                for f in ("prod", "cat", "brand", "camp", "touch", "acq", "region")
            ]
            qty = nn_[:L, 3] * 3.0
            price = rr[:L, 2] / np.maximum(qty, 1.0)
            gap = np.expm1(nn_[:L, 5])
            stat = []
            for x in (rr[:L, 0], rr[:L, 1], price, qty, nn_[:L, 4], gap):
                stat += [
                    float((x != 0).sum()),
                    float(x.sum()),
                    float(x.mean()),
                    float(x.std()),
                    float(x.min()),
                    float(x.max()),
                ]
            past[b] = (
                etc
                + catd
                + stat
                + [
                    float((tt[L - 1] - tt[0]) / 86400.0),
                    float(np.std(tt[:L]) / 86400.0),
                    q,
                    q,
                    float((ct - tt[0]) / 86400.0),
                    float(L),
                    float(len(np.unique(e))),
                ]
            )
    if want_past and past_mu is not None:
        past = (past - past_mu) / past_sd
    t = {c: torch.tensor(v, device=DEV) for c, v in out.items()}
    t["gcat"] = torch.tensor(gcat, device=DEV)
    t["gnum"] = torch.tensor(gnum, device=DEV)
    return (
        t,
        torch.tensor(num, device=DEV),
        torch.tensor(raw, device=DEV),
        torch.tensor(mask, device=DEV),
        torch.tensor(glob, device=DEV),
        torch.tensor(hor, device=DEV),
        torch.tensor(past, device=DEV),
        torch.tensor(qg, device=DEV),
        torch.tensor(cuts, device=DEV),
    )


def run_epoch(
    model,
    seqs,
    ks,
    opt,
    sched,
    cw,
    log_s,
    rng,
    purchase_idx,
    h_min,
    h_max,
    p_mask,
    n_et,
    past_mu,
    past_sd,
    train=True,
    aug_p=0.0,
):
    model.train(train)
    order = np.array(ks)[rng.permutation(len(ks))]
    tot = 0
    nb = 0
    for i in range(0, len(order), ARGS.batch):
        bk = list(order[i : i + ARGS.batch])
        cuts = []
        hs = []
        for k in bk:
            h = float(np.exp(rng.uniform(np.log(h_min), np.log(h_max))))
            lo = seqs[k][4][0]
            hi = AS_OF_TS - h * 86400.0
            if hi <= lo:
                h = h_min
                hi = AS_OF_TS - h * 86400.0
            ct = float(rng.uniform(lo, max(lo, hi)))
            cut = int(np.searchsorted(seqs[k][4], ct, side="right"))
            n = len(seqs[k][3])
            cuts.append(max(8, min(cut, n - 2)))
            hs.append(h)
        t, num, raw, mask, glob, hor, past, qg, lens = collate(
            seqs, bk, cuts, hs, [0.0] * len(bk), purchase_idx, n_et, past_mu, past_sd, False
        )
        Hs, E, g = model(
            t["et"],
            t["prod"],
            t["cat"],
            t["brand"],
            t["camp"],
            t["touch"],
            t["acq"],
            t["region"],
            num,
            raw,
            mask,
            lens,
            qg,
            t["gcat"],
            t["gnum"],
        )
        lnce = torch.tensor(0.0, device=DEV)
        if train and ARGS.nce and len(bk) > 1:
            aug = {k: augment(seqs[k], aug_p, rng) for k in bk}
            cuts_a = []
            for k, cut in zip(bk, cuts):
                tcut = seqs[k][4][min(cut, len(seqs[k][4])) - 1]
                ca = int(np.searchsorted(aug[k][4], tcut, side="right"))
                cuts_a.append(max(1, min(ca, len(aug[k][3]))))
            t_a, num_a, raw_a, mask_a, glob_a, hor_a, past_a, qg_a, lens_a = collate(
                aug, bk, cuts_a, hs, [0.0] * len(bk), purchase_idx, n_et, None, None, False
            )
            _, E_a, _ = model(
                t_a["et"],
                t_a["prod"],
                t_a["cat"],
                t_a["brand"],
                t_a["camp"],
                t_a["touch"],
                t_a["acq"],
                t_a["region"],
                num_a,
                raw_a,
                mask_a,
                lens_a,
                qg_a,
                t_a["gcat"],
                t_a["gnum"],
            )
            z1 = F.normalize(E, dim=-1)
            z2 = F.normalize(E_a, dim=-1)
            logits = (z1 @ z2.T) * model.logit_scale.exp()
            lab = torch.arange(len(bk), device=DEV)
            lnce = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
        nxt = {c: t[c][:, 1:] for c in t}
        m = mask[:, 1:]
        num_next = num[:, 1:]

        def ce(head, key, w=None):
            logit = head(Hs[:, :-1]).reshape(-1, head.out_features)
            tgt = nxt[key].reshape(-1)
            mm = m.reshape(-1) > 0
            if mm.sum() == 0:
                return torch.tensor(0.0, device=DEV)
            return F.cross_entropy(logit[mm], tgt[mm], weight=w)

        def mse(head, key):
            pred = head(Hs[:, :-1]).squeeze(-1)
            tgt = (num_next[:, :, 0] if key == "val" else num_next[:, :, 5]).float()
            mm = (m * (tgt != 0)) if key == "val" else m
            if mm.sum() == 0:
                return torch.tensor(0.0, device=DEV)
            return (((pred - tgt) ** 2) * mm).sum() / mm.sum()

        keep = lambda p=None: rng.random() < (1.0 - p_mask if p is None else p)
        Ls = [
            ce(model.head_et, "et", cw),
            ce(model.head_cat, "cat") if keep() else torch.tensor(0.0, device=DEV),
            ce(model.head_brand, "brand") if keep() else torch.tensor(0.0, device=DEV),
            ce(model.head_touch, "touch") if keep() else torch.tensor(0.0, device=DEV),
            ce(model.head_prod, "prod") if keep(0.9) else torch.tensor(0.0, device=DEV),
            mse(model.head_val, "val") if keep() else torch.tensor(0.0, device=DEV),
            mse(model.head_gap, "gap") if keep() else torch.tensor(0.0, device=DEV),
        ]
        gm = glob[:, 0] >= 0
        let = (
            F.cross_entropy(model.g_et(E)[gm], glob[gm, 0].long(), weight=cw)
            if gm.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )
        gv = model.g_val(E).squeeze(-1)
        gvm = gm & (glob[:, 1] != 0)
        lv = (
            (((gv - glob[:, 1]) ** 2) * gvm).sum() / gvm.sum()
            if gvm.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )
        gg = model.g_gap(E).squeeze(-1)
        lgap = (
            (((gg - glob[:, 2]) ** 2) * gm).sum() / gm.sum()
            if gm.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )
        Ls += [let, lv, lgap]
        ht = torch.as_tensor(hs, device=DEV)
        hemb = model.h_mlp(torch.log1p(ht).unsqueeze(-1))
        last = Hs[torch.arange(Hs.shape[0], device=DEV), lens - 1]
        for vec, hd in ((E, model.g_v_e), (last, model.g_v_h)):
            Ls.append((((hd(vec + hemb).squeeze(-1)) - hor[:, 0]) ** 2).mean())
        for vec, hd in ((E, model.g_o_e), (last, model.g_o_h)):
            Ls.append((((hd(vec + hemb).squeeze(-1)) - hor[:, 1]) ** 2).mean())
        Ls.append(lnce)
        loss = sum(0.5 * torch.exp(-log_s[j]) * Ls[j] + 0.5 * log_s[j] for j in range(len(Ls)))
        fcuts = [len(seqs[k][3]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][4][-1]) / 86400.0 for k in bk]
        t2, num2, raw2, mask2, glob2, hor2, past2, qg2, lens2 = collate(
            seqs, bk, fcuts, [0.0] * len(bk), fq, purchase_idx, n_et, past_mu, past_sd, True
        )
        _, E2, _ = model(
            t2["et"],
            t2["prod"],
            t2["cat"],
            t2["brand"],
            t2["camp"],
            t2["touch"],
            t2["acq"],
            t2["region"],
            num2,
            raw2,
            mask2,
            lens2,
            qg2,
            t2["gcat"],
            t2["gnum"],
        )
        lp = ((model.past(E2) - past2) ** 2).mean()
        loss = loss + 0.5 * torch.exp(-log_s[15]) * lp + 0.5 * log_s[15]
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss.detach())
        nb += 1
    if train and sched:
        sched.step()
    return tot / max(nb, 1)


@torch.no_grad()
def val_loss(model, seqs, ks, purchase_idx, h_min, h_max, n_et, past_mu, past_sd):
    model.eval()
    tot = 0
    nb = 0
    hv = float(math.sqrt(h_min * h_max))
    for i in range(0, len(ks), 32):
        bk = ks[i : i + 32]
        cuts = [
            max(
                8,
                min(int(np.searchsorted(seqs[k][4], AS_OF_TS - hv * 86400.0)), len(seqs[k][3]) - 2),
            )
            for k in bk
        ]
        t, num, raw, mask, glob, hor, past, qg, lens = collate(
            seqs,
            bk,
            cuts,
            [hv] * len(bk),
            [0.0] * len(bk),
            purchase_idx,
            n_et,
            past_mu,
            past_sd,
            False,
        )
        Hs, E, g = model(
            t["et"],
            t["prod"],
            t["cat"],
            t["brand"],
            t["camp"],
            t["touch"],
            t["acq"],
            t["region"],
            num,
            raw,
            mask,
            lens,
            qg,
            t["gcat"],
            t["gnum"],
        )
        nxt = {c: t[c][:, 1:] for c in t}
        m = mask[:, 1:]
        num_next = num[:, 1:]
        l_et = (
            F.cross_entropy(
                model.head_et(Hs[:, :-1]).reshape(-1, model.head_et.out_features)[
                    m.reshape(-1) > 0
                ],
                nxt["et"].reshape(-1)[m.reshape(-1) > 0],
            )
            if m.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )
        tgt = num_next[:, :, 0]
        mm = m * (tgt != 0)
        lv = (
            (((model.head_val(Hs[:, :-1]).squeeze(-1) - tgt) ** 2) * mm).sum() / mm.sum()
            if mm.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )
        gm = glob[:, 0] >= 0
        lg = (
            F.cross_entropy(model.g_et(E)[gm], glob[gm, 0].long())
            if gm.sum() > 0
            else torch.tensor(0.0, device=DEV)
        )
        hemb = model.h_mlp(torch.log1p(torch.as_tensor(hv, device=DEV)).unsqueeze(-1).unsqueeze(0))
        last = Hs[torch.arange(Hs.shape[0], device=DEV), lens - 1]
        lh = 0.0
        for vec, hd in ((E, model.g_v_e), (last, model.g_v_h)):
            lh += float((((hd(vec + hemb).squeeze(-1)) - hor[:, 0]) ** 2).mean())
        for vec, hd in ((E, model.g_o_e), (last, model.g_o_h)):
            lh += float((((hd(vec + hemb).squeeze(-1)) - hor[:, 1]) ** 2).mean())
        fcuts = [len(seqs[k][3]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][4][-1]) / 86400.0 for k in bk]
        t2, num2, raw2, mask2, glob2, hor2, past2, qg2, lens2 = collate(
            seqs, bk, fcuts, [0.0] * len(bk), fq, purchase_idx, n_et, past_mu, past_sd, True
        )
        _, E2, _ = model(
            t2["et"],
            t2["prod"],
            t2["cat"],
            t2["brand"],
            t2["camp"],
            t2["touch"],
            t2["acq"],
            t2["region"],
            num2,
            raw2,
            mask2,
            lens2,
            qg2,
            t2["gcat"],
            t2["gnum"],
        )
        lp = float(((model.past(E2) - past2) ** 2).mean())
        tot += float(l_et + 0.5 * lv + lg + 0.5 * lh + 0.5 * lp)
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def embed(model, seqs, ks, n_et, past_mu, past_sd, use_mag=True):
    model.eval()
    out = []
    gates = []
    for i in range(0, len(ks), 32):
        bk = ks[i : i + 32]
        cuts = [len(seqs[k][3]) for k in bk]
        fq = [(AS_OF_TS - seqs[k][4][-1]) / 86400.0 for k in bk]
        t, num, raw, mask, glob, hor, past, qg, lens = collate(
            seqs, bk, cuts, [0.0] * len(bk), fq, 0, n_et, past_mu, past_sd, False
        )
        _, E, g = model(
            t["et"],
            t["prod"],
            t["cat"],
            t["brand"],
            t["camp"],
            t["touch"],
            t["acq"],
            t["region"],
            num,
            raw,
            mask,
            lens,
            qg,
            t["gcat"],
            t["gnum"],
            use_mag=use_mag,
        )
        out.append(E.cpu().numpy())
        gates.append(g.cpu().numpy())
    return np.concatenate(out, 0), np.concatenate(gates, 0)


def table_features(A, keys):
    amap = {r["customer_key"]: r for r in A.to_dicts()}
    rows = [amap[k] for k in keys]
    feat = [
        c for c in A.columns if not c.startswith("y_") and c not in ("customer_key", "signup_ts")
    ]
    X = pl.DataFrame([{c: r[c] for c in feat} for r in rows]).to_dummies(
        columns=[
            c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in feat
        ],
        drop_first=True,
    )
    labels = {
        c: np.array([r[c] for r in rows], float)
        for c in (
            "y_value_30",
            "y_orders_30",
            "y_churn_30",
            "y_value_90",
            "y_orders_90",
            "y_churn_90",
            "y_margin_90",
        )
    }
    seg = np.array([str(r["segment"]) for r in rows], dtype=object)
    return X.to_numpy().astype(float), labels, seg


def seq_features(B, hold):
    p = B.sort(["customer_key", "ts"]).with_columns(
        pl.col("event_type").shift(1).over("customer_key").alias("prev_et"),
        pl.col("event_type").shift(2).over("customer_key").alias("prev2_et"),
        (pl.col("ts") - pl.col("ts").shift(1).over("customer_key")).dt.total_days().alias("gap"),
    )
    last = p.group_by("customer_key", maintain_order=True).agg(
        pl.col("event_type").last().alias("last_et"),
        pl.col("prev_et").last(),
        pl.col("prev2_et").last(),
        pl.col("event_attributes").last().alias("attrs"),
        pl.col("ts").last().alias("last_ts"),
        pl.col("gap").last().alias("last_gap"),
        pl.col("gap").mean().alias("mean_gap"),
        pl.len().alias("seq_len"),
    )
    cnt7 = (
        p.filter(
            pl.col("ts")
            > pl.lit(AS_OF).str.to_datetime() - __import__("datetime").timedelta(days=7)
        )
        .group_by("customer_key")
        .len()
        .rename({"len": "n7"})
    )
    cnt30 = (
        p.filter(
            pl.col("ts")
            > pl.lit(AS_OF).str.to_datetime() - __import__("datetime").timedelta(days=30)
        )
        .group_by("customer_key")
        .len()
        .rename({"len": "n30"})
    )
    last = (
        last.join(cnt7, on="customer_key", how="left")
        .join(cnt30, on="customer_key", how="left")
        .with_columns(
            pl.col("n7").fill_null(0),
            pl.col("n30").fill_null(0),
            (pl.lit(AS_OF).str.to_datetime() - pl.col("last_ts")).dt.total_days().alias("since"),
        )
    )
    cats = [json.loads(a) for a in last["attrs"].to_list()]
    last = last.with_columns(
        pl.Series("last_cat", [c.get("category_code") for c in cats]),
        pl.Series("last_touch", [c.get("touch") for c in cats]),
    ).drop("attrs", "last_ts")
    last = last.filter(pl.col("customer_key").is_in(hold))
    cols = ["last_et", "prev_et", "prev2_et", "last_cat", "last_touch"]
    return last.to_dummies(columns=cols, drop_first=True)


def cv(X, y, kind):
    oof = np.zeros_like(y, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        if kind == "clf":
            m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
    return oof


def geometry_metrics(Z, y_value, y_churn, seg, rng, k=10, npairs=4000):
    Z = np.nan_to_num(np.asarray(Z, float))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    n = len(Z)
    D = squareform(pdist(Z))
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    p = float(np.mean(y_churn))
    agree = float(np.mean(y_churn[nn] == y_churn))
    base = p * p + (1 - p) ** 2
    vr = spearmanr(y_value, y_value[nn]).correlation
    kk = min(k, n - 1)
    idx = np.argpartition(D, kk, axis=1)[:, :kk]
    pur = float(np.mean(seg[idx] == seg[:, None]))
    _u, cnt = np.unique(seg, return_counts=True)
    seg_base = float(np.sum((cnt / cnt.sum()) ** 2))
    i = rng.integers(0, n, npairs)
    j = rng.integers(0, n, npairs)
    m = i != j
    i, j = i[m], j[m]
    dv = spearmanr(D[i, j], np.abs(y_value[i] - y_value[j])).correlation
    dc = spearmanr(D[i, j], (y_churn[i] != y_churn[j]).astype(float)).correlation
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    er = float((s**2).sum() ** 2 / ((s**4).sum() + 1e-12))
    return {
        "churn_nn_agree": round(agree, 4),
        "churn_random": round(base, 4),
        "value_nn_rank": round(float(vr), 4),
        "seg_knn_purity": round(pur, 4),
        "seg_random": round(seg_base, 4),
        "dist_value_spear": round(float(dv), 4),
        "dist_churn_spear": round(float(dc), 4),
        "eff_rank": round(er, 2),
    }


def one_seed(
    seed,
    B,
    A,
    keys,
    seqs,
    vs,
    purchase_idx,
    h_min,
    h_max,
    n_experts,
    taus,
    p_mask,
    num_dim,
    et_v,
    cat_v,
    aug_p,
):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    nt = int(0.7 * len(keys))
    tr = [keys[i] for i in perm[:nt]]
    hold = [keys[i] for i in perm[nt:]]
    r2 = np.random.default_rng(seed + 100)
    pv = np.random.default_rng(seed + 200)
    vperm = r2.permutation(len(tr))
    vn = max(20, int(0.1 * len(tr)))
    val = [tr[i] for i in vperm[:vn]]
    tr = [tr[i] for i in vperm[vn:]]
    n_et = int(vs["et"])
    pkeys = tr[: min(len(tr), 400)]
    pcuts = [len(seqs[k][3]) for k in pkeys]
    pq = [(AS_OF_TS - seqs[k][4][-1]) / 86400.0 for k in pkeys]
    _t, _n, _r, _m, _g, _h, _p, _q, _l = collate(
        seqs, pkeys, pcuts, [0.0] * len(pkeys), pq, 0, n_et, None, None, True
    )
    P = _p.cpu().numpy()
    past_mu = P.mean(0)
    past_sd = P.std(0)
    past_sd[past_sd < 1e-6] = 1.0
    past_dim = P.shape[1]
    torch.manual_seed(seed)
    model = Enc(vs, ARGS.d, n_experts, ARGS.layers, taus, num_dim, ARGS.mass, past_dim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=ARGS.lr, weight_decay=1e-4)
    warm = max(1, int(0.1 * ARGS.epochs))
    cnt = np.zeros(vs["et"])
    for k in tr:
        for e in seqs[k][0]["et"]:
            cnt[e] += 1
    w = 1 / np.maximum(cnt, 1)
    w = w / w.mean()
    cw = torch.tensor(w, dtype=torch.float32, device=DEV)
    log_s = nn.Parameter(torch.zeros(16, device=DEV))
    opt.add_param_group({"params": [log_s]})
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda e: (
            (e + 1) / warm
            if e < warm
            else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, ARGS.epochs - warm)))
        ),
    )
    best = 1e9
    best_state = None
    pat = 0
    hist = []
    for ep in range(ARGS.epochs):
        trl = run_epoch(
            model,
            seqs,
            tr,
            opt,
            sched,
            cw,
            log_s,
            pv,
            purchase_idx,
            h_min,
            h_max,
            p_mask,
            n_et,
            past_mu,
            past_sd,
            True,
            aug_p,
        )
        vl = val_loss(model, seqs, val, purchase_idx, h_min, h_max, n_et, past_mu, past_sd)
        print(f"seed {seed} epoch {ep + 1} train {trl:.4f} val {vl:.4f}", flush=True)
        hist.append(vl)
        tol = float(np.std(hist[-5:]) / np.sqrt(max(1, len(hist[-5:])))) if len(hist) >= 3 else 1e-3
        if vl < best - tol:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 3:
                print(f"seed {seed} early stop tol {tol:.4f}", flush=True)
                break
    if best_state:
        model.load_state_dict(best_state)
    E, g = embed(model, seqs, hold, n_et, past_mu, past_sd)
    En, _ = (
        embed(model, seqs, hold, n_et, past_mu, past_sd, use_mag=False)
        if ARGS.mass == "explicit"
        else (E, g)
    )
    Xtab, labels, seg = table_features(A, hold)
    grng = np.random.default_rng(seed + 300)
    geom = {
        "E": geometry_metrics(E, labels["y_value_90"], labels["y_churn_90"], seg, grng),
        "table": geometry_metrics(
            Xtab, labels["y_value_90"], labels["y_churn_90"], seg, np.random.default_rng(seed + 301)
        ),
    }
    Sdf = seq_features(B, hold)
    Sdf = (
        pl.DataFrame({"customer_key": hold}).join(Sdf, on="customer_key", how="left").fill_null(0.0)
    )
    Xseq = Sdf.drop("customer_key").to_numpy().astype(float)
    Xh = np.hstack([Xtab, E])
    Xts = np.hstack([Xtab, Xseq])
    Xste = np.hstack([Xts, E])
    res = {}
    for name, y, kind in [
        ("value90", labels["y_value_90"], "reg"),
        ("orders90", labels["y_orders_90"], "reg"),
        ("churn90", labels["y_churn_90"], "clf"),
    ]:
        res[name] = {}
        for arm, X in [
            ("table", Xtab),
            ("E", E),
            ("E_nomag", En),
            ("hybrid", Xh),
            ("table_seq", Xts),
            ("table_seq_E", Xste),
        ]:
            pr = cv(X, y, kind)
            if kind == "clf":
                res[name][arm] = round(roc_auc_score(y, pr), 4)
            else:
                res[name][arm] = {
                    "r2": round(r2_score(y, pr), 4),
                    "mae": round(mean_absolute_error(y, pr), 2),
                }
    return {
        "seed": seed,
        "mass": ARGS.mass,
        "n_train": len(tr),
        "n_val": len(val),
        "n_hold": len(hold),
        "best_val": round(best, 4),
        "results": res,
        "gate_mean": [round(float(x), 3) for x in g.mean(0)],
        "geometry": geom,
        "E": E,
        "Xtab": Xtab,
        "labels": labels,
        "seg": seg,
        "hold": hold,
    }


if __name__ == "__main__":
    ARGS = parse_args()
    data = ARGS.data
    B, A, keys, seqs, vs, et_v, cat_v, h_min, h_max, periods_h, n_experts, taus, p_mask, num_dim = (
        load_data(data)
    )
    purchase_idx = et_v.get("purchase", -1)
    aug_p = ARGS.aug_p if ARGS.aug_p >= 0 else float(np.clip(p_mask, 0.05, 0.2))
    print(
        "data",
        len(keys),
        "events",
        B.height,
        "vocab",
        vs,
        "purchase",
        purchase_idx,
        "horizon",
        h_min,
        h_max,
        "nce",
        ARGS.nce,
        "aug_p",
        round(aug_p, 3),
        flush=True,
    )
    allres = []
    for s in [int(x) for x in ARGS.seeds.split(",")]:
        allres.append(
            one_seed(
                s,
                B,
                A,
                keys,
                seqs,
                vs,
                purchase_idx,
                h_min,
                h_max,
                n_experts,
                taus,
                p_mask,
                num_dim,
                et_v,
                cat_v,
                aug_p,
            )
        )
    Path(ARGS.out).write_text(json.dumps(allres, indent=1))
    print(json.dumps(allres, indent=1))
