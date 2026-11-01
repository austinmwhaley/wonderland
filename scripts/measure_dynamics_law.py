#!/usr/bin/env python3
"""S3 measurement: the laws of the engine (lawful dynamics on data).

Ratifies DEC-386 / v6.0 algebraic-law enforcement with paper, on the PRODUCTION stack:
the frozen CFM checkpoint trained by the canonical command on the real demo stream.

  1 scan_associativity   : the affine scan's merge law F(g∘f)=F(g)∘F(f) on real (decay,drive)
                           tensors from the model's own modules — float64, numerically exact.
  2 ssm_merge_exact      : chunked state continuation equals one-shot encoding
                           (h0 + explicit-dt semantics of the recurrence, on real tokens).
  3 mass_additivity      : the extensive-mass algebra — scan is additive in the drive at
                           shared decay: mass(a+b) = mass(a)+mass(b), float64 exact.
  4 clock_periodicity    : shifting every event by exactly one week (lcm of the hour/day
                           cycles) leaves state and E unchanged — while a non-periodic
                           shift DOES change E (the control that keeps the test honest).
  5 step0_finite         : the architecture is finite at step 0 by construction — random
                           weights, real sequences, zero non-finite states/outputs.
  6 tau_from_spectrum    : timescales are derived from the stream, not typed: compress a
                           stream's cadence 100x and the derived taus must track it; the
                           production registry's resolved half-life/seq-len are receipted.
  7 governed_stability   : the governed training that produced this checkpoint shows all
                           epoch metrics finite, no divergence, governor.diverged=false.

Writes (merges) S3 into docs/report_cards/experiments/judge_stages.json.

    .venv/bin/python scripts/measure_dynamics_law.py --ckpt-dir /tmp/opencode/cfm_s3 \
        --train-log /tmp/opencode/cfm_train.log --db /tmp/opencode/wonderland_demo/customer_event_stream.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import _rollup, stamp_merge_write  # noqa: E402  (shared receipt stamping)

WEEK_S = 7.0 * 86400.0  # principle-ok: one week in seconds — the dow cycle definition


def _load_model(ckpt_dir: Path):
    from looking_glass.cfm_state import load_frozen_encoder

    regs = sorted(ckpt_dir.glob("registry_*.json"), key=lambda p: p.stat().st_mtime)
    if not regs:
        raise SystemExit(
            f"no registry in {ckpt_dir}; run the canonical trainer first:\n"
            f"  .venv/bin/python -m looking_glass.customer_foundation_model train "
            f"--customers 2500 --anchors 4 --as-of 2025-01-01 --db <stream.parquet> --out-dir {ckpt_dir}"
        )
    meta = json.loads(regs[-1].read_text())
    out = load_frozen_encoder(meta["tag"], ckpt_dir)
    if isinstance(out, tuple):
        model, cfg = out[0], (out[1] if len(out) > 1 else None)
    else:
        model, cfg = out, None
    model.to("cpu").eval()
    return model, cfg, meta


def _real_sequences(db: str, cfg, n=6, lo=30, hi=600):
    from looking_glass.cfm_data import assign_split, build_sequences
    from rabbit_hole.stream import read_frame

    df = read_frame(db)
    # SORTED, not "first 400 in whatever order the engine returned": `polars.unique()` does not
    # promise an order, and `build_sequences` internally does `list(set(keys))`, whose iteration
    # order is Python-hash-randomized per process. The result was that this producer claimed
    # "6 real sequences" while sampling a DIFFERENT six on every run (measured: 707840 vs 624640
    # state values between two consecutive runs of an unchanged build), which quietly made S3 —
    # and therefore any card rebuilt from it — non-reproducible (principle 10, certification
    # rule 6). Sorting is the fix that does not touch `looking_glass/`: the population and the
    # seeds are unchanged, only the arbitrary order is pinned.
    keys = sorted(df["customer_key"].unique().to_list())[:400]
    split = assign_split(keys, cfg)
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False)
    items = list(seqs.values()) if isinstance(seqs, dict) else list(seqs)
    out = sorted(
        (s for s in items if lo <= len(s["event_type"]) <= hi), key=lambda s: str(s["customer"])
    )
    if len(out) < 3:
        raise SystemExit(f"only {len(out)} usable sequences ({lo}..{hi} events) — widen the range")
    return out[:n]


def _dt_from_ts(ts: np.ndarray) -> torch.Tensor:
    d = np.zeros_like(ts, dtype=np.float64)
    d[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
    return torch.tensor(np.log1p(d), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


def _ts_seconds(seq) -> np.ndarray:
    """The MODEL'S OWN clock: the same parse tokens()/forward uses (_to_epoch of the
    raw strings), so measured and deployed time semantics cannot diverge."""
    from looking_glass.cfm_config import _to_epoch

    return np.array([_to_epoch(x) for x in seq["event_ts"]], dtype=np.float64)


def _shifted(seq, shift: float) -> dict:
    """Same events, clocks moved by `shift` seconds — encoded back through the
    inverse of the pipeline's own _to_epoch so the round trip is exact."""
    from datetime import datetime as _D

    ts = _ts_seconds(seq) + shift
    return {**seq, "event_ts": [_D.fromtimestamp(float(s)).isoformat() for s in ts]}


def _expert_drives(model, tok: torch.Tensor, dt: torch.Tensor, ex=None):
    """(decay, drive) tensors from the MODEL'S OWN modules on real tokens (no reimplementation)."""
    import torch.nn.functional as F

    ex = ex if ex is not None else model.ssm.experts[0]
    x = ex.input_norm(tok) if getattr(ex, "input_norm", None) is not None else tok
    delta = F.softplus(ex.W_delta(x) + ex.delta_bias)
    if getattr(ex, "continuous_time", False):
        decay = torch.exp(-delta * torch.expm1(dt) / 86400.0)  # principle-ok: s->days units
    else:
        decay = torch.exp(-delta)
    bx = ex.W_B(x)
    return decay, (1.0 - decay) * bx


def _double_expert(model):
    """A float64 copy of expert 0 — the model's own modules, promoted, not reimplemented."""
    ex = copy.deepcopy(model.ssm.experts[0]).cpu()
    for p in ex.parameters():
        p.data = p.data.double()
    for b in ex.buffers():
        b.data = b.data.double()
    return ex


# --------------------------------------------------------------------------- checks


def check_scan_associativity(model, seqs) -> dict:
    """Merge law on real tensors: per-segment affine maps compose to the whole-map, float64."""
    from looking_glass.cfm_model import _scan

    t0 = time.perf_counter()
    ex64 = _double_expert(model)
    worst = 0.0
    with torch.no_grad():
        for seq in seqs:
            ts = _ts_seconds(seq)
            tok = model.tokens(seq).unsqueeze(0).double()
            dt = _dt_from_ts(ts).double()
            d, b = _expert_drives(model, tok, dt, ex=ex64)
            T = d.shape[1]
            for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                k = max(1, min(T - 1, int(T * frac)))
                D, H = _scan(d, b)
                D1, H1 = _scan(d[:, :k], b[:, :k])
                D2, H2 = _scan(d[:, k:], b[:, k:])
                Dc = D1[:, -1:, :] * D2[:, -1:, :]
                Hc = D2[:, -1:, :] * H1[:, -1:, :] + H2[:, -1:, :]
                scale = max(1.0, float(H.abs().max()), float(Hc.abs().max()))
                worst = max(
                    worst,
                    float((D[:, -1:, :] - Dc).abs().max()) / scale,
                    float((H[:, -1:, :] - Hc).abs().max()) / scale,
                )
    ok = worst <= 1e-10
    ct = bool(getattr(model.ssm.experts[0], "continuous_time", False))
    return {
        "working": ok,
        "evidence": (
            f"merge law F(g∘f)=F(g)∘F(f) on {len(seqs)} real sequences x 5 split points "
            f"(float64, model's own decay/drive, continuous_time={ct}): worst relative error "
            f"{worst:.3e} (gate 1e-10)"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "worst_rel_err": worst,
    }


def check_ssm_merge(model, seqs) -> dict:
    """Chunked continuation equals one-shot when ALL recurrent states carry (main scan
    + intent-filter EMAs). Today's serving `absorb()` carries only the main state; the
    residual gap of that approximation is measured and disclosed here (O6: exactness is
    free; an approximation must be bought with a receipt)."""
    t0 = time.perf_counter()
    ex0 = model.ssm.experts[0]
    # tolerance DERIVES from the mechanism's own declared convergence criterion: a bilinear
    # expert solves its fixed point to bilinear_tol, so that (x10 float32 slack) is the law's
    # achievable floor; a non-bilinear expert holds only float32 scan noise (5e-5 floor).
    tol = max(5e-5, 10.0 * float(getattr(ex0, "bilinear_tol", 0.0)))
    ssm = model.ssm
    experts = ssm.experts
    hidden = [i for i, e in enumerate(experts) if getattr(e, "stats_proj", None) is not None]
    worst_h = worst_y = worst_h_nof = 0.0
    with torch.no_grad():
        for seq in seqs:
            ts = _ts_seconds(seq)
            tok = model.tokens(seq).unsqueeze(0)
            dt = _dt_from_ts(ts)
            T = tok.shape[1]
            k = T // 2
            y_full, h_full = ssm(tok, h0=None, dt=dt)
            _, h1 = ssm(tok[:, :k], h0=None, dt=dt[:, :k])
            # states of every auxiliary recurrence at the seam, via the model's own modules
            f1_parts = []
            for e in experts:
                xe = tok[:, :k]
                if getattr(e, "input_norm", None) is not None:
                    xe = e.input_norm(xe)
                if getattr(e, "smooth", None) is not None:
                    f1_parts.append(e.smooth(xe)[:, -1])
                else:
                    f1_parts.append(torch.zeros(tok.shape[0], ssm.chan))
            f1 = torch.cat(f1_parts, dim=-1)
            g1 = None
            if getattr(ssm, "slow_intent", False) and getattr(ssm, "intent", None) is not None:
                g1 = ssm.intent(tok[:, :k])[:, -1]
            y2, h2 = ssm(tok[:, k:], h0=h1, dt=dt[:, k:], f0=f1, g0=g1)
            scale = max(1.0, float(h_full.abs().max()))
            worst_h = max(worst_h, float((h2 - h_full).abs().max()) / scale)
            worst_y = max(worst_y, float((y2[:, -1] - y_full[:, -1]).abs().max()) / scale)
            _, h2_plain = ssm(tok[:, k:], h0=h1, dt=dt[:, k:])  # h-only: today's absorb()
            worst_h_nof = max(worst_h_nof, float((h2_plain - h_full).abs().max()) / scale)
    ok = worst_h <= tol and worst_y <= tol and not hidden
    return {
        "working": ok,
        "evidence": (
            f"one-shot vs split-at-midpoint continuation WITH all recurrent states carried "
            f"(scan h + intent-filter f + wrapper g) over {len(seqs)} real sequences: state rel "
            f"err {worst_h:.3e}, readout {worst_y:.3e} (gate {tol:.0e}); DISCLOSED: h-only "
            f"continuation (today's cfm_state.absorb) drifts {worst_h_nof:.3e} — the filter "
            f"state is not carried by serving, a bought approximation awaiting its O6 "
            f"receipt"
            + (f"; UNCONTINUABLE recurrences present at experts {hidden}" if hidden else "")
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "worst_state_rel": worst_h,
        "h_only_drift": worst_h_nof,
    }


def check_mass_additivity(model, seqs) -> dict:
    """Extensive mass: at shared decay, the scan is additive in the drive (float64 exact)."""
    from looking_glass.cfm_model import _scan

    t0 = time.perf_counter()
    ex64 = _double_expert(model)
    worst = 0.0
    with torch.no_grad():
        for seq in seqs:
            ts = _ts_seconds(seq)
            tok = model.tokens(seq).unsqueeze(0).double()
            dt = _dt_from_ts(ts).double()
            d, b = _expert_drives(model, tok, dt, ex=ex64)
            b1 = torch.zeros_like(b)
            b2 = torch.zeros_like(b)
            b1[:, ::2] = b[:, ::2]  # even-event mass
            b2[:, 1::2] = b[:, 1::2]  # odd-event mass
            _, Hs = _scan(d, b1 + b2)
            _, Ha = _scan(d, b1)
            _, Hb = _scan(d, b2)
            scale = max(1.0, float(Hs.abs().max()))
            worst = max(worst, float((Hs - (Ha + Hb)).abs().max()) / scale)
    ok = worst <= 1e-10
    return {
        "working": ok,
        "evidence": (
            f"mass additivity scan(d, b1+b2) == scan(d,b1)+scan(d,b2) on real drives "
            f"(disjoint event supports, float64): worst relative error {worst:.3e} (gate 1e-10)"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "worst_rel_err": worst,
    }


def check_clock_periodicity(model, seqs) -> dict:
    """Two time laws on real sequences: (a) shifting every event by exactly one week
    (lcm of the hour/day cycles) must not move the state; (b) STRETCHING the gaps
    (tokens fixed) must move it — the spring between events is sacred (continuous-time
    decay must be alive). A dead control would mean the engine is time-blind."""
    t0 = time.perf_counter()

    def _restring(seq, ts_new):
        # UTC-AWARE, necessarily: `_to_epoch` reads a naive ISO string as UTC, so formatting a
        # shifted epoch with the LOCAL timezone (`fromtimestamp` alone) re-parses hours off and
        # makes a correct model look time-dependent. The harness must speak the model's clock.
        from datetime import datetime as _D
        from datetime import timezone as _TZ

        return {
            **seq,
            "event_ts": [_D.fromtimestamp(float(s), tz=_TZ.utc).isoformat() for s in ts_new],
        }

    worst_periodic = 0.0
    best_control = float("inf")
    with torch.no_grad():
        for seq in seqs:
            ts = _ts_seconds(seq)
            tok = model.tokens(seq).unsqueeze(0)
            _, h_ref = model.ssm(tok, h0=None, dt=_dt_from_ts(ts))
            E_ref = model.embed(h_ref)
            # (a) week shift: identical wall-clock phases, identical gaps
            shifted = _restring(seq, ts + WEEK_S)
            tok_s = model.tokens(shifted).unsqueeze(0)
            _, h_s = model.ssm(tok_s, h0=None, dt=_dt_from_ts(_ts_seconds(shifted)))
            E_s = model.embed(h_s)
            scale = max(1.0, float(h_ref.abs().max()))
            drift = max(
                float((h_s - h_ref).abs().max()) / scale,
                float((E_s - E_ref).abs().max()) / max(1.0, float(E_ref.abs().max())),
            )
            worst_periodic = max(worst_periodic, drift)
            # (b) gaps x10 with tokens FIXED (drive dt directly): the state MUST respond
            #     — this isolates the spring (continuous-time decay) from clock features.
            d_stretched = np.concatenate(([0.0], np.diff(ts) * 10.0))
            dt_stretched = torch.tensor(np.log1p(d_stretched), dtype=torch.float32)
            dt_stretched = dt_stretched.unsqueeze(0).unsqueeze(-1)
            _, h_x = model.ssm(tok, h0=None, dt=dt_stretched)
            E_x = model.embed(h_x)
            control = max(
                float((h_x - h_ref).abs().max()) / scale,
                float((E_x - E_ref).abs().max()) / max(1.0, float(E_ref.abs().max())),
            )
            best_control = min(best_control, control)
    ok = worst_periodic <= 1e-4 and best_control > 1e-3
    # diagnostic (disclosed, not gated): two clocks in the stack + clock-feature liveness
    from looking_glass.cfm_config import _to_epoch

    clock_gap = abs(float(_to_epoch(seqs[0]["event_ts"][0])) - float(seqs[0]["ts"][0]))
    tod_norm = float(model.w_tod.weight.detach().norm())
    liveness = "used" if tod_norm > 1e-3 else "learned-into-attributes (w_tod dead)"
    return {
        "working": ok,
        "evidence": (
            f"week-shift (exact cycle multiple): worst state/E drift {worst_periodic:.3e} "
            f"(gate 1e-4); gaps-x10 control (tokens fixed): min state/E drift "
            f"{best_control:.3e} (gate >1e-3) — the spring responds to elapsed time. "
            f"DISCLOSED: absolute-clock features {liveness}, w_tod norm {tod_norm:.3g}; "
            f"cfm_data stored ts vs model _to_epoch differ {clock_gap:.0f}s (UTC vs local "
            f"frame) — reconcile before horizons depend on it"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def check_step0_finite(model, seqs) -> dict:
    """Architecture-level stability: random weights on real sequences produce no non-finite."""
    t0 = time.perf_counter()
    torch.manual_seed(0)
    m = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.5)
        bad = 0
        total = 0
        for seq in seqs:
            ts = _ts_seconds(seq)
            tok = m.tokens(seq).unsqueeze(0)
            y, h = m.ssm(tok, h0=None, dt=_dt_from_ts(ts))
            E = m.embed(h)
            for t_ in (y, h, E):
                total += int(t_.numel())
                bad += int((~torch.isfinite(t_)).sum())
    ok = bad == 0 and total > 0
    return {
        "working": ok,
        "evidence": (
            f"randomly re-initialized trunk (step-0 architecture, N(0,0.5) weights), "
            f"{len(seqs)} real sequences: {bad} non-finite out of {total} state/output/E values"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def check_tau_from_spectrum(db: str) -> dict:
    """Timescales track the stream: stretch event times 100x and the derived taus follow."""
    import datetime as _dt

    import polars as pl

    from looking_glass import stream_adapter as SA

    t0 = time.perf_counter()
    demo = Path(db).parent
    raw = pl.read_parquet(demo / "path_b" / "path_b_customer_event_stream.parquet").select(
        list(SA.REQUIRED)
    )
    raw = raw.head(200_000)
    manifest = json.loads((demo / "manifest.json").read_text())
    as_of = _dt.datetime.fromisoformat(manifest["as_of"]).timestamp()

    disc = SA.discover(raw, as_of, action_keys=[])
    taus = np.asarray(disc["taus"], dtype=np.float64)

    base = raw.with_columns(
        SA.to_ts_expr(pl.col("event_ts")).dt.epoch("us").cast(pl.Float64).alias("_t_us")
    )
    t0us = float(base["_t_us"].min())
    t0s = t0us / 1e6
    stretched = base.with_columns(
        ((base["_t_us"] - t0us) * 100.0 + t0us)
        .cast(pl.Int64)
        .cast(pl.Datetime("us"))
        .cast(pl.String)
        .alias("event_ts")
    ).drop("_t_us")
    as_of_s = t0s + (as_of - t0s) * 100.0
    disc_s = SA.discover(stretched, as_of_s, action_keys=[])
    taus_s = np.asarray(disc_s["taus"], dtype=np.float64)
    ratio = float(np.median(taus_s) / max(np.median(taus), 1e-12))
    ok = taus.std() > 0 and ratio >= 10.0
    return {
        "working": ok,
        "evidence": (
            f"taus are gap quantiles of the stream (no literal): base taus={np.round(taus, 3).tolist()}s "
            f"days -> cadence x100 taus={np.round(taus_s, 3).tolist()} (median ratio {ratio:.1f}x, "
            f"gate >=10x, non-constant={taus.std() > 0})"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def check_governed_stability(train_log: Path, meta: dict) -> dict:
    """The governed run that produced this checkpoint: finite everywhere, never diverged."""
    t0 = time.perf_counter()
    gov = meta.get("governor") or {}
    # PRIMARY source: the registry's own eval series (recorded by the governor). The training log
    # is a fallback for older artifacts — a log that may not exist cannot be the only receipt.
    hist = gov.get("history")
    text = Path(train_log).read_text() if Path(train_log).exists() else ""
    if hist:
        vals = [float(v) for v in hist]
        nan_tokens = 0 if not gov.get("diverged") else 1
        source = "registry.governor.history"
    elif text:
        vals = [float(v) for v in re.findall(r"val\s+([-+0-9.eE]+)", text)]
        vals = [v for v in vals if abs(v) != float("inf")]
        nan_tokens = len(re.findall(r"\bnan\b|\binf\b|UNSTABLE", text, flags=re.IGNORECASE))
        source = "train.log"
    else:
        return {
            "working": False,
            "evidence": f"no governor history in the registry and no usable log: {train_log}",
            "seconds": 0.0,
        }
    finite_vals = [v for v in vals if np.isfinite(v)]
    diverged = bool(gov.get("diverged", False))
    if len(finite_vals) >= 3:
        diffs = np.abs(np.diff(finite_vals))
        noise = float(np.median(diffs)) if diffs.size else 0.0
        spikes = [finite_vals[i] - finite_vals[i - 1] for i in range(1, len(finite_vals))]
        worst_spike = max(spikes) if spikes else 0.0
        spike_ok = worst_spike <= max(6.0 * noise, 1e-9)
    else:
        worst_spike, spike_ok, noise = 0.0, False, 0.0
    ok = bool(finite_vals) and nan_tokens == 0 and not diverged and spike_ok
    resolved = meta.get("resolved") or {}
    return {
        "working": ok,
        "evidence": (
            f"governed training ({source}): {len(finite_vals)} finite epoch values, non-finite tokens={nan_tokens}, "
            f"diverged={diverged}, worst epoch-to-epoch rise {worst_spike:.3g} vs 6x noise "
            f"{6 * noise:.3g}; registry resolved: half_life_days="
            f"{resolved.get('half_life_days', 'MISSING')} seq_len={resolved.get('seq_len', 'MISSING')}"
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- main


def main() -> int:
    # every run of this producer happens inside the project's one memory ceiling (DEC-440):
    # a direct launch is re-entered into the slice, so the boundary does not depend on the
    # caller remembering to use the launcher.
    from safe_run import ensure_boundary

    ensure_boundary()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="/tmp/opencode/cfm_s3")
    ap.add_argument("--train-log", default="/tmp/opencode/cfm_train.log")
    ap.add_argument("--db", default="/tmp/opencode/wonderland_demo/customer_event_stream.parquet")
    ap.add_argument("--out", default=str(ROOT / "docs/report_cards/experiments/judge_stages.json"))
    a = ap.parse_args()

    model, cfg, meta = _load_model(Path(a.ckpt_dir))
    if cfg is None:
        cfg = getattr(model, "cfg", None)
    if cfg is None:
        from looking_glass.cfm_config import CFMConfig

        cfg = CFMConfig()
        resolved = meta.get("resolved") or {}
        cfg.seq_len = int(resolved.get("seq_len", cfg.seq_len))
    seqs = _real_sequences(a.db, cfg)

    checks = {
        "scan_associativity": check_scan_associativity(model, seqs),
        "ssm_merge_exact": check_ssm_merge(model, seqs),
        "mass_additivity": check_mass_additivity(model, seqs),
        "clock_periodicity": check_clock_periodicity(model, seqs),
        "step0_finite": check_step0_finite(model, seqs),
        "tau_from_spectrum": check_tau_from_spectrum(a.db),
        "governed_stability": check_governed_stability(Path(a.train_log), meta),
    }
    existing = stamp_merge_write(
        Path(a.out),
        {"S3": _rollup(checks)},
        " ".join([Path(__file__).name] + sys.argv[1:]),
    )
    print(f"judge stages -> {a.out}")
    for name, c in checks.items():
        print(f"  S3 :: {name}: {'PASS' if c['working'] else 'FAIL'}  ({c['seconds']}s)")
    print(f"S3 working={existing['S3']['working']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
