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
    LN2,
    apply_set_overrides,
    TIME_UNIT_SECONDS,
    _expert_biases,
    _f,
    _seed_everything,
    _to_epoch,
)
from looking_glass.cfm_data import (
    _apply_data_revision,
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
    draw_sample,
)
from looking_glass.cfm_model import CFM, EventVocab, _scan


# ---------------------------------------------------------------------------
# training (multi-objective)
# ---------------------------------------------------------------------------
def _pick_warm_checkpoint(out_dir, as_of, cfg=None):
    """Most-recent compatible checkpoint trained at as_of' <= as_of.

    Leak guard: a checkpoint trained on FUTURE events (as_of' > as_of) is never
    eligible; checkpoints without point-in-time info are skipped too.
    Behavior guard: checkpoints from a different encoder *version* (a behavior
    change that alters trained weights) or a different resolved architecture
    (seq_len / dim / half-life) are skipped when that info is present —
    continuing across semantics silently mixes models. Returns (tag, meta) or
    (None, None).
    """
    import glob as _glob
    import json as _json

    def _epoch(s):
        try:
            from datetime import datetime

            d = datetime.fromisoformat(str(s))
            from datetime import timezone

            return d.replace(tzinfo=timezone.utc).timestamp() if d.tzinfo is None else d.timestamp()
        except Exception:
            return -1.0

    now = _epoch(as_of) if as_of else float("inf")
    regs = sorted(
        _glob.glob(str(Path(out_dir) / "registry_*.json")),
        key=lambda q: Path(q).stat().st_mtime,
        reverse=True,
    )
    for rp in regs:
        try:
            meta = _json.loads(Path(rp).read_text())
        except Exception:
            continue
        ra = meta.get("as_of")
        if ra is None:
            continue  # no point-in-time info — cannot prove it isn't future-trained
        if _epoch(ra) > now + 1.0:  # 1s tolerance
            continue  # FUTURE-trained checkpoint — leak guard
        if cfg is not None:
            ver = getattr(cfg, "version", None)
            if ver and meta.get("version") not in (None, ver):
                continue  # different behavior version — not compatible
            rc = meta.get("config") or {}
            mismatched = any(
                f in rc and rc[f] != getattr(cfg, f, None)
                for f in ("seq_len", "dim", "state_half_life_days", "sf_mode")
            )
            if "objectives" in rc and tuple(rc["objectives"]) != tuple(
                getattr(cfg, "objectives", ())
            ):
                mismatched = True  # objective set changes heads/loss shapes
            if mismatched:
                continue  # different resolved architecture — not compatible
        return meta.get("tag"), meta
    return None, None


def train_cfm(cfg: CFMConfig):
    _seed_everything(cfg.seed)
    df = _read_stream(cfg)
    _apply_data_revision(cfg, df)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    a_keys = [k for k in keys if split[k] == "A"]
    if cfg.sample_a_customers is not None:
        a_keys = draw_sample(keys, split, "A", cfg.sample_a_customers, cfg.split_seed)
        print(f"[sample] encoder trains on {len(a_keys)} of population A", flush=True)
    # ---- derive configuration from the data (no fixed values) ----
    # Architecture identity (dim / seq_len / half-life) derives from the FULL
    # working base, so the same tag always means the same architecture; the
    # training budget and batch derive from the ACTUAL sample_A (compute
    # proportional to data) — which is exactly what makes the sample_A ladder
    # scale honestly (small samples train cheaply and stop at convergence).
    apply_set_overrides(cfg)  # explicit --set wins; resolve records the rest
    vocab_sizes = (
        df["event_type"].n_unique(),
        df["brand"].n_unique(),
        df["entity_type"].n_unique(),
    )
    res_all = AT.resolve_cfm(cfg, df, keys, vocab_sizes)
    res = AT.resolve_cfm(cfg, df, a_keys, vocab_sizes)
    cfg.seq_len, cfg.dim = res_all.seq_len, res_all.dim
    cfg.batch = res.batch
    cfg.state_half_life_days = res_all.half_life_days
    cfg.agg_horizons_days = list(res_all.agg_horizons_days)
    apply_set_overrides(cfg, res.receipt)  # record every --set key in receipts
    apply_set_overrides(cfg, res_all.receipt)
    a_seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
    vocab = EventVocab.build(a_seqs)
    device = torch.device(cfg.device)
    K = max(1, int(cfg.n_experts))
    cfg.dim = max(K, (cfg.dim // K) * K)
    model = CFM(
        vocab,
        cfg.dim,
        n_experts=K,
        delta_biases=_expert_biases(K, cfg.state_half_life_days),
        sf_mode=cfg.sf_mode,
    ).to(device)
    model.half_life_days = cfg.state_half_life_days
    # ---- warm-start / continual (same objective as scratch: data <= as_of) ----
    cfg.warm_from = None
    if str(cfg.warm_start).lower() not in ("none", "", "0"):
        out_w = Path(cfg.out_dir)
        cand_tag = cfg.warm_start if cfg.warm_start != "auto" else None
        meta = None
        if cand_tag is None:
            cand_tag, meta = _pick_warm_checkpoint(out_w, cfg.as_of, cfg)
        else:
            rp = out_w / f"registry_{cand_tag.replace('.', '_')}.json"
            if rp.exists():
                import json as _json

                meta = _json.loads(rp.read_text())
        if cand_tag is None:
            print("[warm-start] none: no compatible earlier checkpoint (scratch)", flush=True)
        else:
            ckpt = out_w / f"cfm_{cand_tag.replace('.', '_')}.pt"
            sizes_ok = meta.get("vocab_sizes") == {
                "et": len(vocab.et),
                "brand": len(vocab.brand),
                "ent": len(vocab.ent),
            }
            blob = (
                torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
            )
            if blob is None or blob.get("dim") != cfg.dim or not sizes_ok:
                why = (
                    "vocab/arch mismatch"
                    if (blob and (blob.get("dim") != cfg.dim or not sizes_ok))
                    else "checkpoint missing"
                )
                print(f"[warm-start] {cand_tag} rejected ({why}) -> scratch", flush=True)
            else:
                model.load_state_dict(blob["state"])
                cfg.warm_from = cand_tag
                print(f"[warm-start] from {cand_tag} (as_of={meta.get('as_of')})", flush=True)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=1e-4)
    tau = 0.99  # EMA of the JEPA target encoder (documented fallback)
    # ---- train/val split of sample A (shared with the portfolio grade) ----
    tr_idx, val_idx = _val_split(len(a_seqs), cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    balancer = DWA(sorted(cfg.objectives), temp=cfg.dwa_temp)
    loss_scales: dict = {}  # EMA per task — the unit system (DEC-018)

    def _update_scales(losses: dict) -> None:
        for k, v in losses.items():
            loss_scales[k] = (
                float(v) if k not in loss_scales else 0.9 * loss_scales[k] + 0.1 * float(v)
            )
            loss_scales[k] = max(loss_scales[k], 1e-8)  # documented floor

    def train_step(n):
        for _ in range(n):
            b = rng.choice(tr_idx, size=min(cfg.batch, len(tr_idx)), replace=False)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = _loss(
                    model,
                    vocab,
                    [a_seqs[i] for i in b],
                    cfg,
                    weights=balancer.weights(),
                    scales=loss_scales,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            model.ema(tau)

    def val_metric():
        # bound eval cost (speed principle): subsample the validation set;
        # metric = the SAME weighted objective training optimizes (DEC-008).
        # Per-task held-out losses update the DWA balancer AFTER scoring
        # (weights come from the past; today's losses are tomorrow's rates).
        vi = val_idx[:256]
        torch.manual_seed(cfg.seed)
        with torch.no_grad():
            T_ = _task_losses(model, vocab, [a_seqs[i] for i in vi], cfg)
        # STATIONARY selection metric: equal-weight held-out sum. DWA weights
        # change every eval, so a weighted metric would change definition
        # between evals — selection must compare like with like (DEC-016).
        # stationary selection on the SAME scale-free unit system (DEC-018)
        v = float(
            sum(T_[k] / max(loss_scales.get(k, 1.0), 1e-8) for k in cfg.objectives if k in T_)
        )
        balancer.update({k: float(x) for k, x in T_.items()})
        _update_scales({k: float(x) for k, x in T_.items()})
        if not math.isfinite(v) or v <= 0.0:
            # a sum of non-negative losses is 0 ONLY when everything collapsed
            # (fp16 blow-up / dead state) — never a breakthrough. Feed the
            # governor an invalid value so best_state is never poisoned.
            return math.inf, None
        return v, copy.deepcopy(model.state_dict())

    gov, best_state = AT.govern(
        train_step, val_metric, res.budget_steps, res.eval_every, res.patience, cfg.seed
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "vocab": vocab.dumps(),
            "dim": cfg.dim,
            "n_experts": K,
            "sf_mode": cfg.sf_mode,
        },
        out / f"cfm_{cfg.tag.replace('.', '_')}.pt",
    )
    # grade the trained encoder on its whole portfolio (DEC-008: every train
    # is graded, not claimed) — receipt lands next to the checkpoint
    from looking_glass.portfolio import evaluate as _portfolio_evaluate

    port = _portfolio_evaluate(model, vocab, cfg, a_seqs, seed=cfg.seed, tag=cfg.tag)
    port_summary = {
        "ok": bool(port["ok"]),
        "receipt": port["receipt_path"],
        "n_pass": sum(1 for r in port["rows"] if r["ok"]),
        "n_rows": len(port["rows"]),
    }
    # learned uncertainty weights (Kendall): w_k = exp(-s_k), the dynamic
    # alternative to hand-tuned loss weights — recorded so every run shows
    # where the objective balance actually landed
    cfg._weight_trajectory = [
        {k: round(w, 4) for k, w in snap.items()} for snap in balancer.history
    ]
    task_weights = {k: round(w, 4) for k, w in balancer.weights().items()}
    cfg.final_task_weights = dict(task_weights)
    cfg.final_loss_scales = {k: round(v, 6) for k, v in loss_scales.items()}
    _registry(
        cfg,
        vocab,
        len(a_seqs),
        len(keys),
        derived=res.receipt,
        governor=gov,
        portfolio=port_summary,
        task_weights=task_weights,
        cfg_resolved={
            "seq_len": cfg.seq_len,
            "dim": cfg.dim,
            "batch": cfg.batch,
            "half_life_days": cfg.state_half_life_days,
            "budget_steps": res.budget_steps,
            "eval_every": res.eval_every,
            "patience": res.patience,
            "agg_horizons_days": cfg.agg_horizons_days,
            "sf_mode": cfg.sf_mode,
            "selection": "heldout_combined_objective (all active objectives)",
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


def _masked_forward(model, vocab, t, cfg, dev):
    """One masked reconstruction forward. Returns (logits, targets, rand).

    CAUSALITY GUARANTEE (tested): the backbone is a prefix scan, so every
    masked position is predicted from LEFT context only — true future tokens
    sit later in the input and cannot influence earlier states. This is not
    BERT: there is no right context, at training or at serving.

    ALL content channels at the masked position are redacted (type, brand,
    entity, value) — feeding the true entity/value would let the model
    shortcut the task from sibling channels instead of learning dynamics.
    Arrival timing (dt) and exogenous covariates (co) stay: the task is
    "an event of unknown kind arrives after this gap", which is exactly the
    serving-relevant shape (missing/sparse events are real at inference).
    Mask rate comes from config (cfg.mask_frac), never a literal.
    """
    B, T = t["et"].shape
    mask = t["mask"]
    rand = (torch.rand(B, T, device=dev) < cfg.mask_frac) & (mask > 0)
    if rand.sum() == 0:
        return None, None, rand
    et2 = t["et"].clone()
    et2[rand] = vocab.n_et
    br2 = t["br"].clone()
    br2[rand] = vocab.n_brand
    en2 = t["en"].clone()
    en2[rand] = vocab.n_ent
    v2 = t["val"].clone()
    v2[rand] = 0.0
    x2 = (
        model.emb_et(et2)
        + model.emb_brand(br2)
        + model.emb_ent(en2)
        + model.w_val(v2)
        + model.w_dt(t["dt"])
        + model.w_co(t["co"])
    )
    y2, _ = model.ssm(x2, mask=mask)
    return y2, t["et"], rand


def _mask_loss_batch(model, vocab, t, cfg, dev):
    y2, tgt, rand = _masked_forward(model, vocab, t, cfg, dev)
    if y2 is None:
        return torch.zeros((), device=dev)
    return F.cross_entropy(model.head_next(y2[rand]), tgt[rand])


def agg_window_targets(secs, vals_lp, mask, h_sec):
    """Exact targets for `agg`: (log1p count, log1p value-sum) of events in
    (t, t+h] per position. secs: (B,T) cumulative event clock (pads flat);
    mask: (B,T); h_sec: (B,1). Returns (B,T,2). Pads never count (equal clock
    fails `cj > ci`, and mask zeroes them)."""
    ci = secs.unsqueeze(2)  # (B,T,1) query
    cj = secs.unsqueeze(1)  # (B,1,T) candidate future events
    win = (cj > ci) & (cj <= ci + h_sec.unsqueeze(-1)) & (mask > 0).unsqueeze(1)
    wf = win.float()
    cnt = wf.sum(2)
    vsum = (wf * vals_lp.unsqueeze(1)).sum(2)
    return torch.stack([torch.log1p(cnt), torch.log1p(vsum)], dim=-1)


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


def _task_losses(model, vocab, items, cfg, aux: dict | None = None):
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
    T_["mask"] = _mask_loss_batch(model, vocab, t, cfg, dev)
    T_["jepa"] = _jepa_loss(model, t, y)
    # ---- query-time readout (S1 / DEC-006): train the FADED state ---------
    # Sample a moment strictly between two events, fade the state there, and
    # grade it on predicting the next event/time — the exact path serving uses
    # (fade last-event state to the anchor), which previously saw no gradient.
    if "query" in cfg.objectives:
        rows = torch.arange(B, device=dev)
        i = torch.randint(0, max(T - 1, 1), (B,), device=dev)
        i_next = (i + 1).clamp(max=T - 1)
        m_i = t["mask"][rows, i] * t["mask"][rows, i_next]
        gap_s = torch.expm1(t["dt"][rows, i_next]).squeeze(-1)  # seconds to next event
        u = torch.rand(B, 1, device=dev)
        off_s = (u * gap_s.unsqueeze(1)).squeeze(-1)  # seconds after event i
        decay = torch.exp(-LN2 * off_s / max(cfg.state_half_life_days * 86400.0, 1.0))
        h_q = y[rows, i] * decay.unsqueeze(-1)
        tgt_next = t["et"][rows, i_next]
        ce_q = F.cross_entropy(model.head_next(h_q), tgt_next, reduction="none")
        tgt_dt = t["dt"][rows, i_next].reshape(-1)
        mse_q = F.mse_loss(model.head_dt(h_q).squeeze(-1), tgt_dt, reduction="none")
        w = m_i.clamp(min=0)
        # ignore pad targets (et == n_et) and future company actions
        w = w * (1.0 - t["co"][rows, i_next, 0])
        denom = w.sum().clamp(min=1)
        T_["query"] = ((ce_q + mse_q) * w).sum() / denom
    # ---- variance floor (VICReg-style, DEC-015) ---------------------------
    # Decorrelation (redundancy) removes correlation but not SCALE collapse:
    # the production runs measured eff-rank 0.23-0.26x null with redundancy
    # active. A hinge on per-dim std forces every channel to carry variance,
    # which is what eff-rank actually measures. Applied on the UNNORMALIZED
    # projection (per-sample L2 would erase the scale information).
    # ---- effective-rank pressure (DEC-016): train on the graded metric ----
    # The portfolio gate measures the participation ratio of the centered
    # covariance of h; making that ratio a training objective aligns pressure
    # with the yardstick. PR = (sum l)^2 / sum(l^2) in [1, dim]; loss =
    # 1 - PR/dim in [0, 1) — rank-1 covariance costs ~1, full spread costs 0.
    # fp32 eigh on purpose (fp16 eigh is unsupported/unstable).
    if "rank" in cfg.objectives and h.shape[0] >= 2:
        # autocast OFF: it downcasts even fp32 matmuls to fp16 (eigh has no
        # fp16 CUDA kernel — measured crash); this math must stay fp32.
        # Participation ratio has a CLOSED FORM — no eigh (the eigh backward
        # divides by eigengaps, which is pathological EXACTLY in the collapsed
        # regime we are escaping: tiny gaps -> exploding grads -> clipped to
        # nothing -> the pressure never landed; measured v2.5.0).
        # PR = (sum l)^2 / sum(l^2) = tr(C)^2 / ||C||_F^2  in [1, dim].
        with torch.autocast(device_type=h.device.type, enabled=False):
            hc = model.proj(h).float()  # spread pressure reaches the trunk via proj
            hc = hc - hc.mean(dim=0, keepdim=True)
            cov = (hc.T @ hc) / (hc.shape[0] - 1)
            pr = cov.diagonal().sum() ** 2 / cov.pow(2).sum().clamp(min=1e-24)
            T_["rank"] = (1.0 - pr / hc.shape[1]).clamp(min=0.0)
    if "variance" in cfg.objectives:
        # fp32 on purpose: std over a fp16-autocast batch overflows with large
        # activations and the hinge degenerates (measured: combined -> 0.0000)
        with torch.autocast(device_type=h.device.type, enabled=False):
            zu = model.proj(h).float()  # (B, D) unnormalized
            std = zu.std(dim=0)  # per-dim std across the batch
            T_["variance"] = torch.relu(1.0 - std).mean().float()
    # ---- exact multi-horizon window targets (S2 / DEC-006) ----------------
    # From the state at t, predict log1p(count) and log1p(value-sum) of the
    # events in (t, t+h] for a horizon sampled from the DERIVED gap-quantile
    # set — long-horizon integration demanded of the state as self-supervision
    # (the statistics raw RFM hand-feeds downstream).
    if "agg" in cfg.objectives and getattr(cfg, "agg_horizons_days", None):
        secs = torch.cumsum(torch.expm1(t["dt"]).squeeze(-1), dim=1)  # event clock (B,T)
        vals_lp = torch.log1p(t["val"].abs()).squeeze(-1)
        hs = torch.tensor(cfg.agg_horizons_days, device=dev, dtype=secs.dtype)
        pick = torch.randint(0, len(cfg.agg_horizons_days), (B,), device=dev)
        h_sec = (hs[pick] * 86400.0).unsqueeze(1)  # (B,1)
        tgt = agg_window_targets(secs, vals_lp, t["mask"], h_sec)  # (B,T,2)
        hin = torch.log1p(h_sec / 86400.0).unsqueeze(1).expand(B, T, 1)
        pred_agg = model.head_agg(torch.cat([y, hin], dim=-1))  # (B,T,2)
        m3 = t["mask"].unsqueeze(-1)
        T_["agg"] = ((pred_agg - tgt).pow(2) * m3).sum() / (m3.sum() * 2).clamp(min=1)
    # Successor features (self-supervised, horizon-free): from every state,
    # predict the discounted future [log1p value, count] at a continuously
    # sampled discount gamma. No fixed horizons; the model learns all scales.
    # Multi-gamma successor features (horizon-free): richer phi and several
    # discounts per step, all vectorized (one scan over B*S).
    GAMS = 4
    gamma = (torch.rand(B, GAMS, device=dev) * GAMMA_MAX).clamp(min=1e-3)  # (B,S)
    dt_days = torch.expm1(t["dt"]) / TIME_UNIT_SECONDS  # (B,T,1)
    v = torch.log1p(t["val"].abs())
    if getattr(cfg, "sf_mode", "purchase") == "event_types":
        # Agnostic phi (DEC-009): value + one discounted component per event
        # type — no objective may name a purchase event.
        et_oh = F.one_hot(t["et"].clamp(max=vocab.n_et - 1), num_classes=vocab.n_et).float()
        phi = torch.cat([v, et_oh], dim=-1)  # (B,T,1+E)
    else:
        oid = model.vocab.et.get(cfg.order_event, -1)
        is_order = (t["et"] == oid).float().unsqueeze(-1)  # (B,T,1)
        phi = torch.cat([v, torch.ones_like(v), is_order, v * is_order], dim=-1)  # (B,T,4)
    PD = phi.shape[-1]
    de = dt_days.unsqueeze(1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1)
    g = gamma.view(B, GAMS, 1, 1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1) ** de
    pe = phi.unsqueeze(1).expand(B, GAMS, T, PD).reshape(B * GAMS, T, PD)
    # Exact reverse affine scan: R_i = sum_{j>i} (prod g) phi_j.
    prev = torch.flip(g, dims=[1])
    dprime = torch.zeros_like(prev)
    dprime[:, 1:] = prev[:, :-1]
    phir = torch.flip(pe, dims=[1])
    bprime = torch.zeros_like(phir)
    bprime[:, 1:] = dprime[:, 1:] * phir[:, :-1]
    _, Sf = _scan(dprime, bprime)
    R = torch.flip(Sf, dims=[1])  # (B*S,T,PD)
    ye = y.unsqueeze(1).expand(B, GAMS, T, y.shape[-1]).reshape(B * GAMS, T, y.shape[-1])
    gcol = gamma.view(B, GAMS, 1, 1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1)
    mask_e = t["mask"].unsqueeze(1).expand(B, GAMS, T).reshape(B * GAMS, T)
    pred = model.head_sf(torch.cat([ye, gcol], dim=-1))
    sl = F.mse_loss(pred, R, reduction="none").mean(-1)
    T_["sf"] = (sl * mask_e).sum() / mask_e.sum().clamp(min=1)
    if aux is not None:
        # target variance for the R² yardstick (DEC-018): same masked entries
        # the loss averages over, across every sampled gamma and phi dim
        sel = mask_e.reshape(-1) > 0
        aux["sf_target_var"] = float(R.reshape(-1, R.shape[-1])[sel].var())
    return T_


class DWA:
    """Dynamic Weight Average (DEC-014): task weights from the RATE of loss
    change, not the loss scale — non-bounded geometric losses (redundancy,
    jepa) cannot exploit it, and a plateauing task is automatically boosted.

        r_k = L_k(prev) / L_k(prev2);  w_k = K * softmax(r_k / T)

    Warmup: the first two measurements keep all weights at 1.0. The weight
    trajectory is recorded (the frontier readout)."""

    def __init__(self, tasks, temp: float = 2.0):
        import math

        self._math = math
        self.tasks = list(tasks)
        self.temp = float(temp)
        self.prev = None
        self.prev2 = None
        self.w = {k: 1.0 for k in self.tasks}
        self.history: list[dict] = [dict(self.w)]

    def update(self, losses: dict) -> None:
        cur = {k: float(losses[k]) for k in self.tasks if k in losses}
        if self.prev is not None:
            r = {k: cur[k] / max(self.prev.get(k, cur[k]), 1e-12) for k in cur if k in self.prev}
            K = len(r)
            mx = max(r.values())
            e = {k: self._math.exp((v - mx) / self.temp) for k, v in r.items()}
            tot = sum(e.values()) or 1.0
            self.w = {k: K * e[k] / tot for k in r}
        self.prev2, self.prev = self.prev, cur
        self.history.append(dict(self.w))

    def weights(self) -> dict:
        return dict(self.w)


GEOMETRY_FAMILY = ("variance", "rank", "redundancy")


def _combine(model, T, cfg, weights: dict | None = None, scales: dict | None = None):
    """Task-balance dispatch (DEC-014). `weights` = current DWA weights.
    geometry_boost (DEC-017) scales the geometry family — the explicit
    Pareto coordinate for the skills-vs-headroom frontier."""
    keys = [k for k in cfg.objectives if k in T]
    boost = float(getattr(cfg, "geometry_boost", 1.0) or 1.0)

    def wk(k):
        w = weights.get(k, 1.0) if weights else 1.0
        return w * (boost if k in GEOMETRY_FAMILY else 1.0)

    if cfg.weight_mode == "dwa":
        sc = scales or getattr(cfg, "final_loss_scales", None) or {}

        def term(k):
            base = weights.get(k, 1.0) if weights else 1.0
            unit = T[k] / max(float(sc.get(k, 0.0)), 1e-8) if cfg.dwa_scale_free and sc else T[k]
            return base * (boost if k in GEOMETRY_FAMILY else 1.0) * unit

        return sum(term(k) for k in keys)
    if cfg.weight_mode == "uncertainty" and cfg.use_uncertainty_weighting:
        return sum(0.5 * torch.exp(-model.log_var[k]) * T[k] + 0.5 * model.log_var[k] for k in keys)
    return sum(T[k] for k in keys)


def _val_split(n_seqs: int, seed: int):
    """Deterministic 85/15 row split of the training sequences (one row per
    customer, so rows are customers). Shared by train_cfm's governor AND the
    portfolio grade — same held-out rows, comparable numbers."""
    order = np.arange(n_seqs)
    np.random.default_rng(seed).shuffle(order)
    n_val = max(1, int(round(0.15 * len(order))))
    return order[n_val:], order[:n_val]  # (train_idx, val_idx)


def held_out_objective(
    model, vocab, seqs, cfg, seed: int = 0, weights: dict | None = None
) -> float:
    """Held-out value of the SAME uncertainty-weighted objective training
    optimizes (DEC-008): the governor selects on this, not on one term of it.
    Seeded so fold/real-vs-destroyed comparisons pair exactly."""
    torch.manual_seed(seed)
    with torch.no_grad():
        T_ = _task_losses(model, vocab, seqs, cfg)
        return float(_combine(model, T_, cfg, weights))


def forward_states(model, seqs, h0s=None):
    """Batched recurrence for state advance — same math as absorb()/model(seq),
    one padded forward over a same-length bucket.

    Returns final h per sequence (B, dim) on the model's device; the caller
    handles per-sequence pre-fades (fade-to-first-event) before stacking h0s.
    """
    dev = model._dev()
    t = _collate(seqs, model.vocab, dev)
    tok = model.tokens_batch(t)
    h0 = None
    if h0s is not None:
        h0 = torch.stack([torch.as_tensor(h, dtype=torch.float32, device=dev) for h in h0s])
    with torch.no_grad():
        _y, h = model.ssm(tok, h0=h0, mask=t["mask"])
    return h


def _loss(model, vocab, items, cfg, weights: dict | None = None, scales: dict | None = None):
    return _combine(model, _task_losses(model, vocab, items, cfg), cfg, weights, scales)


def _registry(
    cfg,
    vocab,
    n_train,
    n_keys,
    derived=None,
    governor=None,
    cfg_resolved=None,
    portfolio=None,
    task_weights=None,
):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"registry_{cfg.tag.replace('.', '_')}.json").write_text(
        json.dumps(
            {
                "version": cfg.version,
                "revision": cfg.revision,
                "tag": cfg.tag,
                "data_signature": getattr(cfg, "_data_signature", ""),
                "as_of": getattr(cfg, "as_of", None),
                "split_seed": cfg.split_seed,
                "warm_from": getattr(cfg, "warm_from", None),
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
                "portfolio": portfolio or {},
                "task_weights": task_weights or {},
                "weight_mode": cfg.weight_mode,
                "weight_trajectory": getattr(cfg, "_weight_trajectory", []) or [],
                "loss_scales": dict(getattr(cfg, "final_loss_scales", {}) or {}),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=1,
        )
    )
