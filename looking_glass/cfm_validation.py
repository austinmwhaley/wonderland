"""CFM independent validation: grounded checks over the trained encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from looking_glass.cfm_config import CFMConfig, _to_epoch
from looking_glass.cfm_data import (
    _apply_data_revision,
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
)
from looking_glass.cfm_model import CFM, EventVocab
from looking_glass.cfm_state import absorb, fade


def validate(cfg: CFMConfig):
    df = _read_stream(cfg)
    _apply_data_revision(cfg, df)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    blob = torch.load(
        Path(cfg.out_dir) / f"cfm_{cfg.tag.replace('.', '_')}.pt",
        map_location=cfg.device,
        weights_only=False,
    )
    vocab = EventVocab(blob["vocab"]["et"], blob["vocab"]["brand"], blob["vocab"]["ent"])
    model = CFM(vocab, blob["dim"], n_experts=blob.get("n_experts", 1)).to(cfg.device)
    model.load_state_dict(blob["state"])
    model.eval()
    model.half_life_days = cfg.state_half_life_days

    def r(name, target, achieved, ok):
        return {"check": name, "target": target, "achieved": achieved, "ok": ok}

    rows = []
    A = {k for k, g in split.items() if g == "A"}
    B = {k for k, g in split.items() if g == "B"}
    rows.append(r("A/B disjoint", "0 overlap", len(A & B), len(A & B) == 0))
    rows.append(
        r("both samples non-empty", ">0 each", f"A={len(A)} B={len(B)}", len(A) > 0 and len(B) > 0)
    )

    bseq = [
        s
        for s in build_sequences(df, list(B), cfg, split, with_anchors=True)
        if s["anchor_epoch"] is not None
    ][:8]
    rows.append(
        r(
            "causality (future can't change past)",
            "stable",
            _causal(model, bseq),
            _causal(model, bseq),
        )
    )

    b_all = build_sequences(df, list(B), cfg, split, with_anchors=False)
    acc, base = _next_event_acc(model, vocab, b_all[:200])
    rows.append(r("next-event acc > baseline", ">", f"{acc:.3f} vs {base:.3f}", acc > base))
    mm = _objective_metrics(model, vocab, b_all[:200])
    # Baselines DERIVED from the data (doctrine #1: no magic thresholds).
    from collections import Counter as _C

    _ent = _C(
        str(s["entity_type"][i + 1]) for s in b_all[:200] for i in range(len(s["entity_type"]) - 1)
    )
    _ent_base = _ent.most_common(1)[0][1] / max(sum(_ent.values()), 1)
    rows.append(
        r(
            "objective: next-event acc > baseline",
            ">",
            f"{mm['next']:.3f} vs {base:.3f}",
            mm["next"] > base,
        )
    )
    rows.append(
        r(
            "objective: entity acc > baseline",
            ">",
            f"{mm['entity']:.3f} vs {_ent_base:.3f}",
            mm["entity"] > _ent_base,
        )
    )
    rows.append(
        r(
            "objective: occurrence acc > base",
            ">",
            f"{mm['occ']:.3f} vs {mm['occ_base']:.3f}",
            mm["occ"] > mm["occ_base"],
        )
    )
    rows.append(
        r(
            "objective: temporal-order acc > chance(0.5)",
            ">",
            round(mm["order"], 3),
            mm["order"] > 0.5,
        )
    )
    rows.append(
        r("objective: dt MAE (log1p s)", "<3.0", round(mm["dt_mae"], 3), mm["dt_mae"] < 3.0)
    )

    # state products + as_of
    import duckdb

    con = duckdb.connect(str(Path(cfg.out_dir) / "cfm_products.duckdb"), read_only=True)
    n_state = con.execute("select count(*) from customer_state").fetchone()[0]
    n_nulls = con.execute(
        "select count(*) from customer_state where as_of_epoch is null"
    ).fetchone()[0]
    n_tr = con.execute("select count(*) from anchor_embeddings").fetchone()[0]
    dims = con.execute("select distinct dim from customer_state").fetchall()
    con.close()
    rows.append(r("customer_state covers customers", len(keys), n_state, n_state == len(keys)))
    rows.append(r("state has as_of", "0 null", n_nulls, n_nulls == 0))
    rows.append(r("training embeddings present", ">0", n_tr, n_tr > 0))
    rows.append(r("dim consistent", 1, len(dims), len(dims) == 1))

    # absorb consistency: incremental == from-scratch (first sample)
    seqs = build_sequences(df, list(B)[:1], cfg, split, with_anchors=False)
    if seqs:
        s = seqs[0]
        with torch.no_grad():
            _, h_full = model(s)
            model.embed(model(s)[0])
        h_inc, emb_inc = absorb(model, s, h0=None)
        rows.append(
            r(
                "absorb == from-scratch",
                "allclose",
                float(torch.allclose(h_full, h_inc, atol=1e-4)),
                torch.allclose(h_full, h_inc, atol=1e-4),
            )
        )
        # fade decays the state
        faded = fade(h_full, 30 * 86400, cfg.state_half_life_days)
        rows.append(
            r(
                "fade decays state",
                "norm<1",
                round(float(faded.norm() / max(h_full.norm(), 1e-9)), 3),
                faded.norm() < h_full.norm(),
            )
        )
    return rows, all(x["ok"] for x in rows)


def _objective_metrics(model, vocab, seqs):
    """Held-out accuracy per objective (self-supervised; strictly causal).
    Company-action targets are excluded; occurrence is balanced (median-horizon)."""
    dev = model._dev()
    inv_et = {v: k for k, v in vocab.et.items()}
    inv_en = {v: k for k, v in vocab.ent.items()}
    m = {
        "next": 0,
        "entity": 0,
        "occ_tp": 0,
        "occ_tn": 0,
        "occ_pos": 0,
        "occ_neg": 0,
        "order": 0,
        "n": 0,
        "dt": 0.0,
    }
    for seq in seqs:
        with torch.no_grad():
            y, _ = model(seq)
        if y.shape[0] < 2:
            continue
        ts = np.array([_to_epoch(x) for x in seq["event_ts"]], dtype=np.float64)
        gap = np.maximum(ts[1:] - ts[:-1], 0.0)
        co = seq.get("co")
        co = co if co is not None else [[0.0, 0.0]] * len(seq["event_type"])
        keep = np.array([co[i + 1][0] < 0.5 for i in range(len(gap))], dtype=bool)
        thr = float(np.median(gap[keep])) if keep.any() else float(np.log1p(7 * 86400))
        nxt = model.head_next(y[:-1]).argmax(-1).detach().cpu().numpy()
        ent = model.head_ent(y[:-1]).argmax(-1).detach().cpu().numpy()
        occ = (model.head_occ(y[:-1]).squeeze(1).detach().cpu().numpy() > 0).astype(int)
        dtp = model.head_dt(y[:-1]).squeeze(1).detach().cpu().numpy()
        pos = torch.tensor(
            [vocab.et.get(str(x), vocab.n_et) for x in seq["event_type"][1:]], device=dev
        )
        neg = torch.randint(0, vocab.n_et, pos.shape, device=dev)
        sp = (model.order_W(y[:-1]) * model.emb_et(pos)).sum(-1)
        sn = (model.order_W(y[:-1]) * model.emb_et(neg)).sum(-1)
        ordv = (sp > sn).detach().cpu().numpy().astype(int)
        for i in range(len(nxt)):
            if not keep[i]:
                continue
            m["n"] += 1
            m["next"] += int(inv_et.get(int(nxt[i])) == str(seq["event_type"][i + 1]))
            m["entity"] += int(
                inv_en.get(int(ent[i]))
                == (
                    str(seq["entity_type"][i + 1])
                    if seq["entity_type"][i + 1] is not None
                    else "none"
                )
            )
            lab = int(gap[i] <= thr)
            m["occ_tp"] += int(occ[i] == 1 and lab == 1)
            m["occ_tn"] += int(occ[i] == 0 and lab == 0)
            m["occ_pos"] += lab
            m["occ_neg"] += 1 - lab
            m["order"] += int(ordv[i] == 1)
            m["dt"] += abs(float(dtp[i]) - float(np.log1p(gap[i])))
    n = max(m["n"], 1)
    occ_bal = 0.5 * (m["occ_tp"] / max(m["occ_pos"], 1) + m["occ_tn"] / max(m["occ_neg"], 1))
    return {
        "next": m["next"] / n,
        "entity": m["entity"] / n,
        "occ": occ_bal,
        "occ_base": max(m["occ_pos"], m["occ_neg"]) / n,
        "order": m["order"] / n,
        "dt_mae": m["dt"] / n,
    }


def _causal(model, seqs):
    ok = True
    for s in seqs:
        with torch.no_grad():
            y_full, _ = model(s)
        cut = {
            **s,
            "event_type": s["event_type"][:-1],
            "brand": s["brand"][:-1],
            "entity_type": s["entity_type"][:-1],
            "value": s["value"][:-1],
            "event_ts": s["event_ts"][:-1],
            "co": s["co"][:-1],
            "ts": s["ts"][:-1],
        }
        if len(cut["event_type"]) < 2:
            continue
        with torch.no_grad():
            y_cut, _ = model(cut)
        n = min(len(y_cut), len(y_full))
        if not torch.allclose(y_cut[:n], y_full[:n], atol=1e-3):
            ok = False
    return ok


def _next_event_acc(model, vocab, seqs):
    correct = total = base = 0
    from collections import Counter

    freq = Counter()
    for s in seqs:
        for x in s["event_type"][1:]:
            freq[str(x)] += 1
    top = freq.most_common(1)[0][0] if freq else None
    inv = {v: k for k, v in vocab.et.items()}
    for s in seqs:
        with torch.no_grad():
            y, _ = model(s)
            pred = model.head_next(y[:-1]).argmax(-1).detach().cpu().numpy()
        for p, t in zip(pred, [str(x) for x in s["event_type"][1:]]):
            total += 1
            correct += int(inv.get(int(p)) == t)
            base += int(t == top)
    return (correct / total if total else 0.0, base / total if total else 0.0)
