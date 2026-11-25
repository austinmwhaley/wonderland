#!/usr/bin/env python3
"""S1/S2 measurement: the input law (schema resolution + vocabulary/numeric encoding).

Replaces the weak `measure_construction.py` claim (`Es.shape[0] > 0` on an absent dataset)
with real evidence for the two spine stages:

  S1  schema resolution — a RENAMED schema (client_id/occurred_at/event/attributes + a value
      alias) ingests through `onboard` -> canonical -> `stream_adapter.discover` -> encoder
      training, with ZERO edits to pipeline code; a NEW FIELD (one categorical, one numeric)
      injected into event_attributes is automatically resolved into the model's vocabulary
      (resolution receipt lists it).
  S2  vocabulary & numeric encoding — the unseen-value path (novel event type / category /
      huge numeric) encodes through the FROZEN schema with no crash, no NaN, and no vocab
      re-derivation; numeric encoding is the signed log1p law (monotone, finite at 1e12);
      permuting attribute-field order changes nothing (field identity is by name);
      two seed-0 training runs on identical prepared inputs are bit-identical.

Writes (merges) docs/report_cards/experiments/judge_stages.json for scripts/report_card.py.

    .venv/bin/python scripts/measure_input_law.py [--data /tmp/opencode/wonderland_demo]

Every check records what it asserted; a stage is `working` only if ALL its checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "demo"))

import report_card as RC  # noqa: E402  (behavior_hash/git: single source for the freshness stamp)

from looking_glass import stream_adapter as SA  # noqa: E402
from looking_glass.stream_adapter import to_ts_expr  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "judge_stages.json"
WORK = Path("/tmp/opencode/input_law")


# --------------------------------------------------------------------------- helpers


def _finitish(x) -> bool:
    return bool(np.all(np.isfinite(x)))


def _seqs_finite(seqs) -> bool:
    return all(_finitish(a) for arr in seqs.values() for a in arr)


def _seqs_digest(seqs) -> str:
    h = hashlib.sha256()
    for k in sorted(seqs):
        h.update(k.encode())
        for a in seqs[k]:
            h.update(np.ascontiguousarray(a).tobytes())
            h.update(str(a.dtype).encode())
    return h.hexdigest()[:16]


def _reencode(decoded: pl.DataFrame) -> pl.Series:
    """Vectorized struct-frame -> JSON-string column (NDJSON writer, one Rust pass)."""
    buf = io.StringIO()
    decoded.write_ndjson(buf)
    return pl.Series("event_attributes", buf.getvalue().splitlines())


def _attrs_with(df: pl.DataFrame, extra: dict[str, object]) -> pl.DataFrame:
    """Vectorized attribute-field injection: decode -> add columns -> re-encode as JSON."""

    decoded = SA._decode_attrs(df["event_attributes"])
    for name, values in extra.items():
        decoded = decoded.with_columns(values)
    return df.with_columns(_reencode(decoded))


def _lap(t0: float, label: str) -> float:
    """Print one phase's elapsed time and return a fresh mark (profiling, not evidence).

    The receipt keeps the single per-check `seconds` it always had; the split exists so the
    runtime receipt names WHICH phase costs (principle 16: every component reports runtime).
    """
    now = time.perf_counter()
    print(f"    [{label}] {now - t0:.1f}s", flush=True)
    return now


# --------------------------------------------------------------------------- S1 checks


def _prepare_renamed_end_to_end(data: Path) -> dict:
    """Everything the renamed-schema check does before the GPU: read -> onboard -> discover.

    Split from the training spike so the CPU-bound half can run in the worker pool while the
    spike waits for a batch budget; `check_renamed_end_to_end` is the serial composition.
    """
    from looking_glass.onboard import onboard

    t0 = time.perf_counter()
    raw = pl.read_parquet(data / "path_b" / "path_b_customer_event_stream.parquet")
    renamed = raw.rename(
        {
            "customer_key": "client_id",
            "event_ts": "occurred_at",
            "event_type": "event",
            "event_attributes": "attributes",
        }
    )
    # a value alias ("total") must also resolve; derive it from the attrs "value" field
    dec = SA._decode_attrs(renamed["attributes"])
    vals = (
        dec["value"].cast(pl.Float64, strict=False).fill_null(0.0)
        if "value" in dec.columns
        else pl.Series([0.0] * renamed.height)
    )
    renamed = renamed.with_columns(vals.alias("total"))
    src = WORK / "renamed_src.parquet"
    src.parent.mkdir(parents=True, exist_ok=True)
    renamed.write_parquet(src)
    t1 = _lap(t0, "renamed: read+rename -> renamed source")

    out_dir = WORK / "renamed_onboard"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    receipt = onboard(src, out_dir=out_dir, action_keys=[])
    t2 = _lap(t1, "renamed: onboard")

    canonical = pl.read_parquet(receipt["canonical_path"])
    missing = [c for c in SA.REQUIRED if c not in canonical.columns]
    rows_ok = receipt["n_rows"] > 0 and receipt["n_rows"] >= int(0.99 * renamed.height)

    manifest = json.loads((data / "manifest.json").read_text())
    as_of_ts = datetime.fromisoformat(manifest["as_of"]).timestamp()
    disc = SA.discover(
        canonical.select(list(SA.REQUIRED)),
        as_of_ts,
        action_keys=list(manifest.get("action_keys") or []),
    )
    # mirror the harness's prepared tuple (stream_loader._load_via_discovery)
    B = canonical.filter(
        to_ts_expr(pl.col("event_ts")) <= pl.lit(datetime.fromtimestamp(as_of_ts))
    ).sort(["customer_key", "event_ts"])
    prepared = (
        B,
        None,
        disc["keys"],
        disc["seqs"],
        disc["vs"],
        disc["et_v"],
        disc["catspec"],
        disc["num"],
        disc["taus"],
        disc["p_mask"],
        disc["as_of_ts"],
        disc["periods"],
        disc["horizons"],
    )
    _lap(t2, "renamed: discover + prepared tuple")
    return {
        "prepared": prepared,
        "receipt": receipt,
        "renamed_height": int(renamed.height),
        "missing": missing,
        "rows_ok": bool(rows_ok),
        "n_seqs": len(disc["keys"]),
        "seq_len": int(np.median([len(disc["seqs"][k][5]) for k in disc["keys"]])),
        "seconds": time.perf_counter() - t0,
    }


def _train_renamed_end_to_end(ctx: dict, budget: dict) -> dict:
    """The renamed-schema assertion: a seed-0 spike trains on the RESOLVED stream and is finite."""
    import auto_encoder as AE

    t0 = time.perf_counter()
    E, _, _, _, _, _, _, _, best, _ = AE.train_encoder(
        data=str(WORK / "renamed_e2e"),
        seed=0,
        d=64,
        layers=1,
        epochs=1,
        prepared=ctx["prepared"],
        batch=budget["batch"],
    )
    _lap(t0, f"renamed: train spike (batch {budget['batch']})")
    receipt = ctx["receipt"]
    ok = (
        not ctx["missing"]
        and ctx["rows_ok"]
        and receipt["columns_mapped"].get("customer_key") == "client_id"
        and receipt["columns_mapped"].get("event_ts") == "occurred_at"
        and receipt["columns_mapped"].get("event_type") == "event"
        and _finitish(E)
        and math.isfinite(float(best))
    )
    return {
        "working": ok,
        "evidence": (
            f"renamed client_id/occurred_at/event/attributes/total -> onboard resolved "
            f"{receipt['columns_mapped']}; {receipt['n_rows']} rows kept "
            f"(source {ctx['renamed_height']}); {ctx['n_seqs']} customers encoded; encoder "
            f"trained on the resolved stream (d=64,L=1,e=1): E finite={_finitish(E)}, "
            f"best={float(best):.4f}; zero pipeline-code edits"
        ),
        "seconds": round(ctx["seconds"] + (time.perf_counter() - t0), 1),
        "n_rows": int(receipt["n_rows"]),
        "n_customers": int(ctx["n_seqs"]),
    }


def check_renamed_end_to_end(data: Path) -> dict:
    """Serial composition of the two phases (batch derived from live hardware, as before)."""
    from looking_glass.autotune import derive_batch

    ctx = _prepare_renamed_end_to_end(data)
    batch, batch_receipt = derive_batch(ctx["n_seqs"], ctx["seq_len"])
    return _train_renamed_end_to_end(ctx, {"batch": batch, "receipt": batch_receipt})


def _prepare_new_field_ingested(data: Path) -> dict:
    """Inject one categorical + one numeric field, resolve it, before the GPU spike."""
    t0 = time.perf_counter()
    vdir = WORK / "newfield"
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(data, vdir)
    for stale in vdir.glob("prep_*.npz"):
        stale.unlink()

    raw = pl.read_parquet(vdir / "path_b" / "path_b_customer_event_stream.parquet")
    n = raw.height
    tiers = pl.Series(
        "loyalty_tier", [f"tier_{i % 4}" for i in range(n)], dtype=pl.String
    )  # values never seen by any existing vocabulary
    seconds = (
        pl.Series("session_seconds", np.abs(np.random.default_rng(7).normal(30, 20, n)))
        .cast(pl.Float64)
        .round(3)
    )
    raw = _attrs_with(raw, {"loyalty_tier": tiers, "session_seconds": seconds})
    raw.write_parquet(vdir / "path_b" / "path_b_customer_event_stream.parquet")
    t1 = _lap(t0, "newfield: copy + inject attrs")

    disc = SA.discover(
        raw.select("customer_key", "event_ts", "event_type", "event_attributes"),
        datetime.fromisoformat(
            json.loads((vdir / "manifest.json").read_text())["as_of"]
        ).timestamp(),
        action_keys=[],
    )
    _lap(t1, "newfield: discover")
    return {
        "cat_fields": sorted({g["field"] for g in disc["catspec"]}),
        "num_fields": sorted({f for f, _ in disc["num"]}),
        "n_rows": int(n),
        "n_seqs": len(disc["keys"]),
        "seq_len": int(np.median([len(disc["seqs"][k][5]) for k in disc["keys"]])),
        "seconds": time.perf_counter() - t0,
    }


def _train_new_field_ingested(ctx: dict, budget: dict) -> dict:
    """The assertion: the injected fields are in the model's vocabulary and the spike is finite."""
    import auto_encoder as AE

    t0 = time.perf_counter()
    E, _, _, _, _, _, _, _, best, _ = AE.train_encoder(
        data=str(WORK / "newfield"),
        seed=0,
        d=64,
        layers=1,
        epochs=1,
        batch=budget["batch"],
    )
    _lap(t0, f"newfield: train spike (batch {budget['batch']})")
    ok = (
        "loyalty_tier" in ctx["cat_fields"]
        and "session_seconds" in ctx["num_fields"]
        and _finitish(E)
        and math.isfinite(float(best))
    )
    return {
        "working": ok,
        "evidence": (
            "injected loyalty_tier (novel categories) + session_seconds into every event's "
            f"attributes: resolved into catspec={ctx['cat_fields']} num={ctx['num_fields']}; "
            f"encoder trained on the variant (d=64,L=1,e=1): E finite={_finitish(E)}, "
            f"best={float(best):.4f}; zero pipeline-code edits"
        ),
        "seconds": round(ctx["seconds"] + (time.perf_counter() - t0), 1),
        "n_rows": int(ctx["n_rows"]),
    }


def check_new_field_ingested(data: Path) -> dict:
    """Serial composition of the two phases (batch derived from live hardware, as before)."""
    from looking_glass.autotune import derive_batch

    ctx = _prepare_new_field_ingested(data)
    batch, batch_receipt = derive_batch(ctx["n_seqs"], ctx["seq_len"])
    return _train_new_field_ingested(ctx, {"batch": batch, "receipt": batch_receipt})


# --------------------------------------------------------------------------- S2 checks


def check_unseen_value_path(data: Path) -> dict:
    """Frozen schema must encode novel values without crash, NaN, or vocab re-derivation."""

    t0 = time.perf_counter()
    raw = pl.read_parquet(data / "path_b" / "path_b_customer_event_stream.parquet")
    manifest = json.loads((data / "manifest.json").read_text())
    as_of_ts = datetime.fromisoformat(manifest["as_of"]).timestamp()
    disc = SA.discover(
        raw.select(list(SA.REQUIRED)), as_of_ts, action_keys=list(manifest.get("action_keys") or [])
    )
    schema = disc["schema"]
    et_before, cats_before = len(schema.et_v), [len(g.get("map") or {}) for g in schema.catspec]
    t1 = _lap(t0, "unseen: discover")

    novel = pl.DataFrame(
        {
            "customer_key": ["novel_a", "novel_a", "novel_b"],
            "event_ts": [
                manifest["as_of"] + "T00:00:00",
                manifest["as_of"] + "T06:00:00",
                manifest["as_of"] + "T00:00:00",
            ],
            "event_type": ["quantum_launch", "quantum_launch", "zzz_no_such_event"],
            "event_attributes": [
                json.dumps({"loyalty_tier": "tier_zeta", "value": 1e12, "session_seconds": 1e12}),
                json.dumps({"loyalty_tier": "tier_zeta", "value": -1e12}),
                json.dumps({"brand": "brand_impossible", "session_seconds": -1e12}),
            ],
        }
    )
    eval_batch = pl.concat([raw.select(list(SA.REQUIRED)), novel])
    keys, seqs, receipt = SA.transform(eval_batch, schema)
    _lap(t1, "unseen: transform (serving)")
    ok = (
        "novel_a" in keys
        and "novel_b" in keys
        and _seqs_finite(seqs)
        and len(schema.et_v) == et_before
        and [len(g.get("map") or {}) for g in schema.catspec] == cats_before
    )
    # the novel rows must land in the reserved OOV slot 0, not a crash or NaN
    novel_idx = [i for i, k in enumerate(keys) if k.startswith("novel_")]
    oov_ok = novel_idx and all(
        int(seqs[keys[i]][0][0]) == 0
        for i in novel_idx  # event_type -> OOV index 0
    )
    return {
        "working": ok and oov_ok,
        "evidence": (
            f"frozen schema ({et_before} event types, vocab sizes {cats_before}) encoded a "
            f"batch containing unseen event types/categories/1e12 numerics: rows={receipt['rows']}, "
            f"entities={receipt['entities']}, all arrays finite={_seqs_finite(seqs)}, novel rows "
            f"-> OOV slot 0 = {oov_ok}, vocab unchanged after serving = True (never re-derived)"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "n_rows": int(receipt["rows"]),
    }


def check_numeric_transform(data: Path) -> dict:
    """Numeric encoding obeys the signed log1p law: monotone, odd-symmetric, finite at 1e12."""

    t0 = time.perf_counter()
    manifest = json.loads((data / "manifest.json").read_text())
    as_of_ts = datetime.fromisoformat(manifest["as_of"]).timestamp()
    values = [0.0, 1.0, 10.0, 1000.0, 1e12, -1.0, -1e12]
    rows = []
    for i, v in enumerate(values):
        rows.append(
            pl.DataFrame(
                {
                    "customer_key": [f"numcust_{i}"] * 3,
                    "event_ts": [
                        manifest["as_of"] + "T00:00:00",
                        manifest["as_of"] + "T01:00:00",
                        manifest["as_of"] + "T02:00:00",
                    ],
                    "event_type": ["probe", "probe", "probe"],
                    "event_attributes": [
                        json.dumps({"probe_value": v}),
                        json.dumps({"probe_value": v}),
                        json.dumps({"probe_value": v}),
                    ],
                }
            )
        )
    frame = pl.concat(rows)
    disc = SA.discover(frame, as_of_ts, action_keys=[])
    if "probe_value" not in {f for f, _ in disc["num"]}:
        return {
            "working": False,
            "evidence": "probe_value not resolved as numeric",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    seqs = disc["seqs"]
    per_cust = {
        k: float(
            seqs[k][2][:, [j for j, (f, _) in enumerate(disc["num"]) if f == "probe_value"][0]][0]
        )
        for k in disc["keys"]
    }
    encoded = [per_cust[f"numcust_{i}"] for i, _ in enumerate(values)]
    expected = [float(np.sign(v) * np.log1p(abs(v))) for v in values]
    law_ok = all(abs(a - b) < 1e-4 for a, b in zip(encoded, expected))
    order = [0, 1, 2, 3, 4]
    monotone = all(encoded[i] < encoded[j] for i, j in zip(order, order[1:]))
    symmetric = abs(encoded[1] + encoded[5]) < 1e-6  # f(1) == -f(-1)
    ok = law_ok and monotone and symmetric and _finitish(np.array(encoded))
    return {
        "working": ok,
        "evidence": (
            f"signed log1p law measured: encoded({values}) = "
            f"{[round(x, 4) for x in encoded]}; matches formula={law_ok}, monotone={monotone}, "
            f"odd-symmetric={symmetric}, finite at 1e12={bool(np.isfinite(encoded[4]))}"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def check_field_permutation(data: Path) -> dict:
    """Permuting attribute-field order must not change a single encoded number."""

    t0 = time.perf_counter()
    raw = pl.read_parquet(data / "path_b" / "path_b_customer_event_stream.parquet")
    manifest = json.loads((data / "manifest.json").read_text())
    as_of_ts = datetime.fromisoformat(manifest["as_of"]).timestamp()
    sample = raw.head(60_000).select(list(SA.REQUIRED))

    decoded = SA._decode_attrs(sample["event_attributes"])
    fields = decoded.columns
    rng = np.random.default_rng(11)
    shuffled = list(fields)
    rng.shuffle(shuffled)
    perm = sample.with_columns(_reencode(decoded.select(shuffled)).alias("event_attributes"))

    keys_a, seqs_a, _ = SA.transform(
        sample, SA.discover(sample, as_of_ts, action_keys=[])["schema"]
    )
    t1 = _lap(t0, "permutation: discover(a) + transform(a)")
    schema_b = SA.discover(perm, as_of_ts, action_keys=[])["schema"]
    keys_b, seqs_b, _ = SA.transform(perm, schema_b)
    _lap(t1, "permutation: discover(b) + transform(b)")
    ok = keys_a == keys_b and _seqs_digest(seqs_a) == _seqs_digest(seqs_b)
    return {
        "working": ok,
        "evidence": (
            f"attribute field order {fields[:4]}... vs permuted {shuffled[:4]}... over "
            f"{sample.height} rows: sequence digests "
            f"{'identical' if ok else 'DIFFER'} ({_seqs_digest(seqs_a)} vs {_seqs_digest(seqs_b)})"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "n_rows": int(sample.height),
    }


def _prepare_training_determinism(data: Path) -> dict:
    """The prepared inputs both seed-0 runs share (everything before the GPU)."""
    t0 = time.perf_counter()
    raw = pl.read_parquet(data / "path_b" / "path_b_customer_event_stream.parquet")
    manifest = json.loads((data / "manifest.json").read_text())
    as_of_ts = datetime.fromisoformat(manifest["as_of"]).timestamp()
    disc = SA.discover(
        raw.select(list(SA.REQUIRED)), as_of_ts, action_keys=list(manifest.get("action_keys") or [])
    )
    B = raw.filter(to_ts_expr(pl.col("event_ts")) <= pl.lit(datetime.fromtimestamp(as_of_ts))).sort(
        ["customer_key", "event_ts"]
    )
    prepared = (
        B,
        None,
        disc["keys"],
        disc["seqs"],
        disc["vs"],
        disc["et_v"],
        disc["catspec"],
        disc["num"],
        disc["taus"],
        disc["p_mask"],
        disc["as_of_ts"],
        disc["periods"],
        disc["horizons"],
    )
    _lap(t0, "determinism: discover + prepared")
    return {
        "prepared": prepared,
        "n_seqs": len(disc["keys"]),
        "seq_len": int(np.median([len(disc["seqs"][k][5]) for k in disc["keys"]])),
        "seconds": time.perf_counter() - t0,
    }


def _train_training_determinism(ctx: dict, budget: dict) -> dict:
    """Two seed-0 runs on identical prepared inputs produce bit-identical E and best-val.

    DEC-573 addendum: the batch is derived ONCE and shared by both runs. train_encoder's
    batch=0 default re-derives from live hardware state on every call, so under co-tenant
    GPU load the two runs can land on different batches — which violates this check's own
    "identical" precondition and confounds determinism with memory state (measured: two
    seed-0 runs diverged to best-val 23.64 vs 24.52 while O3 attempt 4 trained alongside).
    """
    import auto_encoder as AE

    t0 = time.perf_counter()
    batch, batch_receipt = budget["batch"], budget["receipt"]
    E1, _, _, _, _, _, _, _, best1, _ = AE.train_encoder(
        data=str(WORK / "det_a"),
        seed=0,
        d=64,
        layers=1,
        epochs=1,
        prepared=ctx["prepared"],
        batch=batch,
    )
    t1 = _lap(t0, "determinism: run a")
    E2, _, _, _, _, _, _, _, best2, _ = AE.train_encoder(
        data=str(WORK / "det_b"),
        seed=0,
        d=64,
        layers=1,
        epochs=1,
        prepared=ctx["prepared"],
        batch=batch,
    )
    _lap(t1, "determinism: run b")
    identical = np.ascontiguousarray(E1).tobytes() == np.ascontiguousarray(E2).tobytes()
    ok = identical and best1 == best2
    return {
        "working": ok,
        "evidence": (
            f"two seed-0 runs (d=64,L=1,e=1,batch={batch} shared: {batch_receipt}) on identical "
            f"prepared inputs: E bit-identical={identical}, "
            f"best-val equal ({float(best1):.6f} == {float(best2):.6f})"
        ),
        "seconds": round(ctx["seconds"] + (time.perf_counter() - t0), 1),
        "n_customers": int(ctx["n_seqs"]),
    }


def check_training_determinism(data: Path) -> dict:
    """Serial composition of the two phases (batch derived from live hardware, as before)."""
    from looking_glass.autotune import derive_batch

    ctx = _prepare_training_determinism(data)
    batch, batch_receipt = derive_batch(ctx["n_seqs"], ctx["seq_len"])
    return _train_training_determinism(ctx, {"batch": batch, "receipt": batch_receipt})


def products_identity(products) -> dict:
    """The identity of the products artifact a producer measured (DEC-448, one definition).

    A card whose legs were measured against DIFFERENT encoders would otherwise render as
    self-consistent (DEC-445: batch changed, tag did not). Identity is the registry — the
    artifact that describes itself: its tag and content hash. The path is recorded for the
    reader's convenience, never compared (the same artifact mounts differently in the
    container than on the host).
    """
    p = Path(products)
    d = p.parent if p.suffix else p
    out = {"path": str(p), "tag": None, "registry_sha256": None}
    # newest RUN, not last filename: `...o9e...` sorts after `...o39...` but is older (DEC-515)
    regs = sorted(d.glob("registry_*.json"), key=lambda q: q.stat().st_mtime)
    if regs:
        body = regs[-1].read_bytes()
        out["registry_sha256"] = hashlib.sha256(body).hexdigest()[:16]
        try:
            out["tag"] = json.loads(body).get("tag")
        except (ValueError, UnicodeDecodeError):
            pass  # an unreadable registry is a missing identity, which the reader reports
    return out


def stamp_merge_write(
    out_path: Path,
    stages: dict,
    command: str,
    *,
    stamp_blocks: bool = True,
    drop: tuple[str, ...] = (),
) -> dict:
    """Stamp _meta (freshness law) and merge stage blocks into the judge receipt.

    Single definition shared by every measurement script writing judge_stages.json,
    so stamps can never drift between producers.

    EACH BLOCK IS STAMPED BY THE PRODUCER THAT MEASURED IT. This file is a MERGE of five blocks
    written by four different scripts at four different times, so a file-level `_meta` alone let
    the last writer certify evidence it never produced: one re-run of `measure_views_law` made
    every other stage's rows look current, including blocks measured against `/tmp` inputs that
    have since been deleted (certification rules 6 and 7). The block's own `_stamp` is what the
    reader validates; `_meta` remains for provenance of the file as a whole.

    `stamp_blocks=False` is for a receipt written whole by ONE producer (a leg file), where the
    file-level `_meta` already describes every byte of it and per-block stamps would be noise.

    `drop` names keys this producer no longer owns: a merge that only adds keys lets a
    superseded block (e.g. a flat-era consumer description inside a bank-era leg receipt)
    survive every re-stamp, so the receipt contradicts itself. A producer writes fresh values
    for every key it owns and drops the rest (DEC-608).
    """
    existing = {}
    p = Path(out_path)
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 - rebuild rather than merge with garbage
            existing = {}
    for key in drop:
        existing.pop(key, None)
    stamp = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": RC.git("rev-parse", "HEAD"),
        "behavior_hash": RC.behavior_hash(),
        "command": command,
        "producer": Path(sys.argv[0]).name,
    }
    mine = {
        k: (
            {**v, "_stamp": dict(stamp)}
            if stamp_blocks and isinstance(v, dict) and "_stamp" not in v
            else v
        )
        for k, v in stages.items()
    }
    existing.update({"_meta": stamp, **mine})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n")
    return existing


def _rollup(checks: dict) -> dict:
    working = all(c["working"] for c in checks.values())
    return {
        "working": working,
        "evidence": " | ".join(
            f"[{name}] {'OK' if c['working'] else 'FAILED'}: {c['evidence']}"
            for name, c in checks.items()
        ),
        "checks": checks,
    }


# --------------------------------------------------------------------------- main


def _jobs() -> int:
    """Worker count — the shared compute policy (DEC-660)."""
    from compute_policy import jobs

    return jobs()


def _worker_init() -> None:
    """Give a spawned joblib worker the script's import roots (it does not inherit sys.path)."""
    for root in (ROOT, ROOT / "scripts", ROOT / "scripts" / "demo"):
        p = str(root)
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    # every run of this producer happens inside the project's one memory ceiling (DEC-440):
    # a direct launch is re-entered into the slice, so the boundary does not depend on the
    # caller remembering to use the launcher.
    from safe_run import ensure_boundary

    ensure_boundary()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/tmp/opencode/wonderland_demo")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    data = Path(a.data)
    if not (data / "path_b" / "path_b_customer_event_stream.parquet").exists():
        print(
            f"FATAL: demo stream not found under {data}; run scripts/make_demo_data.py --out {data}"
        )
        return 2
    WORK.mkdir(parents=True, exist_ok=True)

    from joblib import Parallel, delayed

    n_jobs = _jobs()
    t_run = time.perf_counter()

    # Phase A — the CPU-bound half of every check. The six are independent by construction:
    # each owns its own WORK subdir and only READS `data`, so the construction probes
    # (unseen / numeric / permutation) complete here and the three training checks return
    # their prepared inputs plus the (n_seqs, seq_len) the batch budget needs.
    prep_fns = {
        "renamed_end_to_end": _prepare_renamed_end_to_end,
        "new_field_ingested": _prepare_new_field_ingested,
        "unseen_value_path": check_unseen_value_path,
        "numeric_transform_law": check_numeric_transform,
        "field_permutation_invariance": check_field_permutation,
        "training_determinism": _prepare_training_determinism,
    }
    order = [
        "renamed_end_to_end",
        "new_field_ingested",
        "unseen_value_path",
        "numeric_transform_law",
        "field_permutation_invariance",
        "training_determinism",
    ]
    t0 = time.perf_counter()
    prepared = dict(
        zip(
            order,
            Parallel(n_jobs=n_jobs, prefer="processes", initializer=_worker_init)(
                delayed(prep_fns[name])(data) for name in order
            ),
        )
    )
    t_prep = time.perf_counter() - t0

    # Batch budget — ONE hardware snapshot for the whole run, per task's own (n_seqs, seq_len).
    # `train_encoder(batch=0)` re-derives from live free memory on every call, so under
    # co-tenant GPU load the independent spikes would land on different batches and this run
    # would confound determinism with memory state — the DEC-573 precondition, applied run-wide.
    from looking_glass.autotune import derive_batch

    budgets = {
        name: dict(
            zip(
                ("batch", "receipt"),
                derive_batch(prepared[name]["n_seqs"], prepared[name]["seq_len"]),
            )
        )
        for name in ("renamed_end_to_end", "new_field_ingested", "training_determinism")
    }
    for name, b in budgets.items():
        print(f"  batch budget :: {name}: {b['batch']} ({b['receipt']})", flush=True)

    # Phase B — only the GPU spikes remain. They are independent (separate seeds, inputs and
    # output dirs) and CUDA contexts are per-process, so the same pool runs them concurrently.
    train_fns = {
        "renamed_end_to_end": _train_renamed_end_to_end,
        "new_field_ingested": _train_new_field_ingested,
        "training_determinism": _train_training_determinism,
    }
    train_order = ["renamed_end_to_end", "new_field_ingested", "training_determinism"]
    t1 = time.perf_counter()
    trained = dict(
        zip(
            train_order,
            Parallel(n_jobs=n_jobs, prefer="processes", initializer=_worker_init)(
                delayed(train_fns[name])(prepared[name], budgets[name]) for name in train_order
            ),
        )
    )
    t_train = time.perf_counter() - t1

    s1_checks = {
        "renamed_end_to_end": trained["renamed_end_to_end"],
        "new_field_ingested": trained["new_field_ingested"],
    }
    s2_checks = {
        "unseen_value_path": prepared["unseen_value_path"],
        "numeric_transform_law": prepared["numeric_transform_law"],
        "field_permutation_invariance": prepared["field_permutation_invariance"],
        "training_determinism": trained["training_determinism"],
    }

    def rollup(checks: dict) -> dict:
        return _rollup(checks)

    out_path = Path(a.out)
    existing = stamp_merge_write(
        out_path,
        {"S1": rollup(s1_checks), "S2": rollup(s2_checks)},
        " ".join([Path(__file__).name] + sys.argv[1:]),
    )

    print(f"judge stages -> {out_path}")
    for stage in ("S1", "S2"):
        for name, c in existing[stage]["checks"].items():
            print(f"  {stage} :: {name}: {'PASS' if c['working'] else 'FAIL'}  ({c['seconds']}s)")
    print(
        f"input law: {time.perf_counter() - t_run:.1f}s wall "
        f"(cpu pool {t_prep:.1f}s, spike pool {t_train:.1f}s, jobs={n_jobs})"
    )
    print(f"S1 working={existing['S1']['working']}  S2 working={existing['S2']['working']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
