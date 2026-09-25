"""CFM multi-objective training: losses, the training loop, and the registry."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from looking_glass.cfm_config import (
    AT,
    CFMConfig,
    GAMMA_MAX,
    SF_PHI,
    TIME_UNIT_SECONDS,
    _expert_biases,
    _f,
    _h,
    _seed_everything,
    _to_epoch,
)
from looking_glass.cfm_data import (
    _apply_data_revision,
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
)
from looking_glass.cfm_model import CFM, EventVocab, _scan


# ---------------------------------------------------------------------------
# training (multi-objective)
# ---------------------------------------------------------------------------
def _val_loss(model, vocab, seqs, cfg):
    """Grounded held-out metric (next-event cross-entropy; lower is better)."""
    dev = model._dev()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for seq in seqs:
            y, _ = model(seq)
            if y.shape[0] < 2:
                continue
            tgt = torch.tensor(
                [vocab.et.get(str(x), vocab.n_et) for x in seq["event_type"][1:]], device=dev
            )
            tot += float(F.cross_entropy(model.head_next(y[:-1]), tgt))
            n += 1
    return tot / max(n, 1)


def train_cfm(cfg: CFMConfig):
    _seed_everything(cfg.seed)
    df = _read_stream(cfg)
    _apply_data_revision(cfg, df)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    # ---- derive configuration from the data (no fixed values) ----
    AT.sequence_lengths(df, keys)
    all_ts = [_to_epoch(x) for x in df["event_ts"].to_list()]
    vocab_sizes = (
        df["event_type"].n_unique(),
        df["brand"].n_unique(),
        df["entity_type"].n_unique(),
    )
    res = AT.resolve_cfm(cfg, df, keys, vocab_sizes, all_ts)
    cfg.seq_len, cfg.dim, cfg.batch = res.seq_len, res.dim, res.batch
    cfg.state_half_life_days = res.half_life_days
    a_keys = [k for k in keys if split[k] == "A"]
    a_seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
    vocab = EventVocab.build(a_seqs)
    device = torch.device(cfg.device)
    K = max(1, int(cfg.n_experts))
    cfg.dim = max(K, (cfg.dim // K) * K)
    model = CFM(
        vocab, cfg.dim, n_experts=K, delta_biases=_expert_biases(K, cfg.state_half_life_days)
    ).to(device)
    model.half_life_days = cfg.state_half_life_days
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=1e-4)
    tau = 0.99  # EMA of the JEPA target encoder (documented fallback)
    # ---- train/val split of sample A (derived fraction) ----
    order = np.arange(len(a_seqs))
    np.random.default_rng(cfg.seed).shuffle(order)
    n_val = max(1, int(round(0.15 * len(order))))
    val_idx, tr_idx = order[:n_val], order[n_val:]
    rng = np.random.default_rng(cfg.seed)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def train_step(n):
        for _ in range(n):
            b = rng.choice(tr_idx, size=min(cfg.batch, len(tr_idx)), replace=False)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = _loss(model, vocab, [a_seqs[i] for i in b], cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            model.ema(tau)

    def val_metric():
        # bound eval cost (speed principle): subsample the validation set
        vi = val_idx[:256]
        v = _val_loss(model, vocab, [a_seqs[i] for i in vi], cfg)
        return v, copy.deepcopy(model.state_dict())

    gov, best_state = AT.govern(
        train_step, val_metric, res.budget_steps, res.eval_every, res.patience, cfg.seed
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "vocab": vocab.dumps(), "dim": cfg.dim, "n_experts": K},
        out / f"cfm_{cfg.tag.replace('.', '_')}.pt",
    )
    _registry(
        cfg,
        vocab,
        len(a_seqs),
        len(keys),
        derived=res.receipt,
        governor=gov,
        cfg_resolved={
            "seq_len": cfg.seq_len,
            "dim": cfg.dim,
            "batch": cfg.batch,
            "half_life_days": cfg.state_half_life_days,
        },
    )
    return model, vocab, df, keys, split


def _collate(seqs, vocab, device):
    """Pad a batch into dense GPU tensors (Polars already gave us per-group
    lists). One forward over the batch instead of one per sequence."""
    B = len(seqs)
    T = max(len(s["event_type"]) for s in seqs)
    et = torch.full((B, T), vocab.n_et, dtype=torch.long, device=device)
    br = torch.full((B, T), vocab.n_brand, dtype=torch.long, device=device)
    en = torch.full((B, T), vocab.n_ent, dtype=torch.long, device=device)
    val = torch.zeros(B, T, 1, device=device)
    dt = torch.zeros(B, T, 1, device=device)
    cov = torch.zeros(B, T, 2, device=device)
    mask = torch.zeros(B, T, device=device)
    cut = torch.ones(B, dtype=torch.long, device=device)
    for i, sq in enumerate(seqs):
        L = len(sq["event_type"])
        et[i, :L] = torch.tensor(
            [vocab.et.get(str(x), vocab.n_et) for x in sq["event_type"]], device=device
        )
        br[i, :L] = torch.tensor(
            [
                vocab.brand.get(str(b) if b is not None else "none", vocab.n_brand)
                for b in sq["brand"]
            ],
            device=device,
        )
        en[i, :L] = torch.tensor(
            [
                vocab.ent.get(str(e) if e is not None else "none", vocab.n_ent)
                for e in sq["entity_type"]
            ],
            device=device,
        )
        val[i, :L, 0] = torch.tensor(
            [_f(v) for v in sq["value"]], dtype=torch.float32, device=device
        )
        ts = np.asarray(sq.get("ts") or [_to_epoch(x) for x in sq["event_ts"]], dtype=np.float64)
        d = np.zeros(L)
        d[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
        dt[i, :L, 0] = torch.tensor(np.log1p(d), dtype=torch.float32, device=device)
        if sq.get("co"):
            cov[i, :L] = torch.tensor(sq["co"], dtype=torch.float32, device=device)
        mask[i, :L] = 1.0
        cut[i] = min(L - 1, max(1, int(0.6 * L)))
    return {"et": et, "br": br, "en": en, "val": val, "dt": dt, "co": cov, "mask": mask, "cut": cut}


def _mask_loss_batch(model, vocab, t, dev):
    B, T = t["et"].shape
    mask = t["mask"]
    rand = (torch.rand(B, T, device=dev) < 0.15) & (mask > 0)
    if rand.sum() == 0:
        return torch.zeros((), device=dev)
    et2 = t["et"].clone()
    et2[rand] = vocab.n_et
    x2 = (
        model.emb_et(et2)
        + model.emb_brand(t["br"])
        + model.emb_ent(t["en"])
        + model.w_val(t["val"])
        + model.w_dt(t["dt"])
        + model.w_co(t["co"])
    )
    y2, _ = model.ssm(x2, mask=mask)
    return F.cross_entropy(model.head_next(y2[rand]), t["et"][rand])


def _jepa_loss(model, t, y):
    """Latent future prediction: predict the EMA target encoder's embedding of
    the future suffix from the context state at the cut."""
    B, T, _ = y.shape
    cut = t["cut"]
    idx = torch.arange(B, device=y.device)
    ctx = y[idx, cut - 1]
    xt = model.target_tokens_batch(t)
    yt, _ = model.t_ssm(xt, mask=t["mask"])
    pos = torch.arange(T, device=y.device).unsqueeze(0)
    suf = (pos >= cut.unsqueeze(1)) & (t["mask"] > 0)
    cnt = suf.sum(1, keepdim=True).clamp(min=1)
    pool = (yt * suf.unsqueeze(-1)).sum(1) / cnt
    S_tgt = F.normalize(model.t_proj(pool), dim=-1)
    S_pred = F.normalize(model.pred(ctx), dim=-1)
    return (1.0 - (S_tgt * S_pred).sum(-1)).mean()


def _task_losses(model, vocab, items, cfg):
    dev = model._dev()
    t = _collate(items, vocab, dev)
    x = model.tokens_batch(t)
    y, h = model.ssm(x, mask=t["mask"])
    B, T, _ = y.shape
    # Targets at company-action positions are exogenous; never predict them.
    valid_t = t["mask"][:, :-1] * (1.0 - t["co"][:, 1:, 0])
    valid = valid_t.reshape(-1)
    nv = valid.sum().clamp(min=1)

    def mce(logits, tgt):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
        )
        return (loss * valid).sum() / nv

    def mmse(pred, tgt):
        loss = F.mse_loss(pred, tgt, reduction="none").squeeze(-1).reshape(-1)
        return (loss * valid).sum() / nv

    T_ = {
        "next": mce(model.head_next(y[:, :-1]), t["et"][:, 1:]),
        "entity": mce(model.head_ent(y[:, :-1]), t["en"][:, 1:]),
        "dt": mmse(model.head_dt(y[:, :-1]), t["dt"][:, 1:]),
        "value": mmse(model.head_val(y[:, :-1]), torch.log1p(t["val"][:, 1:].abs())),
    }
    # Occurrence horizon derived from the data (median gap) so classes balance,
    # not a fixed window that a frequent exogenous event can saturate.
    g = t["dt"][:, 1:].squeeze(-1)
    vm = valid_t > 0
    thr = torch.median(g[vm]) if vm.any() else torch.tensor(math.log1p(7 * 86400.0), device=dev)
    occ_lab = (g <= thr).float()
    ol = F.binary_cross_entropy_with_logits(
        model.head_occ(y[:, :-1]).squeeze(-1), occ_lab, reduction="none"
    ).reshape(-1)
    T_["occur"] = (ol * valid).sum() / nv
    pos = t["et"][:, 1:]
    neg = torch.randint(0, vocab.n_et, pos.shape, device=dev)
    sp = (model.order_W(y[:, :-1]) * model.emb_et(pos)).sum(-1)
    sn = (model.order_W(y[:, :-1]) * model.emb_et(neg)).sum(-1)
    opl = F.binary_cross_entropy_with_logits(
        sp - sn, torch.ones_like(sp), reduction="none"
    ).reshape(-1)
    T_["order"] = (opl * valid).sum() / nv
    S = F.normalize(model.proj(h), dim=-1)
    if B >= 4:
        logits = S @ S.t() / cfg.contrast_tau
        T_["contrast"] = F.cross_entropy(logits, torch.arange(B, device=dev))
    if B >= 2:
        zc = S - S.mean(0, keepdim=True)
        cov = (zc.t() @ zc) / max(B - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        T_["redundancy"] = (off**2).mean()
    T_["mask"] = _mask_loss_batch(model, vocab, t, dev)
    T_["jepa"] = _jepa_loss(model, t, y)
    # Successor features (self-supervised, horizon-free): from every state,
    # predict the discounted future [log1p value, count] at a continuously
    # sampled discount gamma. No fixed horizons; the model learns all scales.
    # Multi-gamma successor features (horizon-free): richer phi and several
    # discounts per step, all vectorized (one scan over B*S).
    GAMS = 4
    gamma = (torch.rand(B, GAMS, device=dev) * GAMMA_MAX).clamp(min=1e-3)  # (B,S)
    dt_days = torch.expm1(t["dt"]) / TIME_UNIT_SECONDS  # (B,T,1)
    oid = model.vocab.et.get("order_placed", -1)
    is_order = (t["et"] == oid).float().unsqueeze(-1)  # (B,T,1)
    v = torch.log1p(t["val"].abs())
    phi = torch.cat([v, torch.ones_like(v), is_order, v * is_order], dim=-1)  # (B,T,4)
    de = dt_days.unsqueeze(1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1)
    g = gamma.view(B, GAMS, 1, 1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1) ** de
    pe = phi.unsqueeze(1).expand(B, GAMS, T, SF_PHI).reshape(B * GAMS, T, SF_PHI)
    # Exact reverse affine scan: R_i = sum_{j>i} (prod g) phi_j.
    prev = torch.flip(g, dims=[1])
    dprime = torch.zeros_like(prev)
    dprime[:, 1:] = prev[:, :-1]
    phir = torch.flip(pe, dims=[1])
    bprime = torch.zeros_like(phir)
    bprime[:, 1:] = dprime[:, 1:] * phir[:, :-1]
    _, Sf = _scan(dprime, bprime)
    R = torch.flip(Sf, dims=[1])  # (B*S,T,4)
    ye = y.unsqueeze(1).expand(B, GAMS, T, y.shape[-1]).reshape(B * GAMS, T, y.shape[-1])
    gcol = gamma.view(B, GAMS, 1, 1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1)
    mask_e = t["mask"].unsqueeze(1).expand(B, GAMS, T).reshape(B * GAMS, T)
    pred = model.head_sf(torch.cat([ye, gcol], dim=-1))
    sl = F.mse_loss(pred, R, reduction="none").mean(-1)
    T_["sf"] = (sl * mask_e).sum() / mask_e.sum().clamp(min=1)
    return T_


def _combine(model, T, cfg):
    """Uncertainty-based adaptive weighting: loss = sum 0.5*exp(-s)*L + 0.5*s.
    No manual weights; tasks that are noisy/conflicting earn lower weight."""
    keys = [k for k in cfg.objectives if k in T]
    if cfg.use_uncertainty_weighting:
        return sum(0.5 * torch.exp(-model.log_var[k]) * T[k] + 0.5 * model.log_var[k] for k in keys)
    return sum(T[k] for k in keys)


def _loss(model, vocab, items, cfg):
    return _combine(model, _task_losses(model, vocab, items, cfg), cfg)


def _mask_loss(model, vocab, items):
    """Masked-event reconstruction: hide events, predict them from context."""
    dev = model._dev()
    total = 0.0
    for seq in items:
        L = len(seq["event_type"])
        if L < 4:
            continue
        k = max(1, int(0.15 * L))
        rng = np.random.default_rng(_h(seq["customer"], L) % (2**31))
        mi = list(rng.choice(L, k, replace=False))
        et = list(map(str, seq["event_type"]))
        for i in mi:
            et[i] = "<mask>"
        y, _ = model({**seq, "event_type": et})
        mi = [i for i in mi if i < len(y)]
        if not mi:
            continue
        tgt = torch.tensor(
            [vocab.et.get(str(seq["event_type"][i]), vocab.n_et) for i in mi], device=dev
        )
        total = total + F.cross_entropy(model.head_next(y[torch.tensor(mi, device=dev)]), tgt)
    return total / max(len(items), 1)


# ---------------------------------------------------------------------------
# registry + independent validation
# ---------------------------------------------------------------------------
def _registry(cfg, vocab, n_train, n_keys, derived=None, governor=None, cfg_resolved=None):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"registry_{cfg.tag.replace('.', '_')}.json").write_text(
        json.dumps(
            {
                "version": cfg.version,
                "revision": cfg.revision,
                "tag": cfg.tag,
                "data_signature": getattr(cfg, "_data_signature", ""),
                "db": cfg.db,
                "table": cfg.table,
                "n_customers": n_keys,
                "n_train_sequences": n_train,
                "config": asdict(cfg),
                "vocab_sizes": {k: len(v) for k, v in vocab.dumps().items()},
                "derived": (derived or {}).get("derived", {}),
                "overrides": (derived or {}).get("overrides", {}),
                "resolved": cfg_resolved or {},
                "governor": governor or {},
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=1,
        )
    )
