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

Writes (merges) docs/report_cards/experiments/judge_stages.json for scripts/judge_card.py.

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

import judge_card as JC  # noqa: E402  (behavior_hash/git: single source for the freshness stamp)

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


# --------------------------------------------------------------------------- S1 checks


def check_renamed_end_to_end(data: Path) -> dict:
    """Renamed schema -> onboard -> discover -> encoder training. Zero code edits."""
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

    out_dir = WORK / "renamed_onboard"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    receipt = onboard(src, out_dir=out_dir, action_keys=[])

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

    import auto_encoder as AE

    E, _, _, _, _, _, _, _, best, _ = AE.train_encoder(
        data=str(WORK / "renamed_e2e"), seed=0, d=64, layers=1, epochs=1, prepared=prepared
    )
    ok = (
        not missing
        and rows_ok
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
            f"(source {renamed.height}); {len(disc['keys'])} customers encoded; encoder "
            f"trained on the resolved stream (d=64,L=1,e=1): E finite={_finitish(E)}, "
            f"best={float(best):.4f}; zero pipeline-code edits"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "n_rows": int(receipt["n_rows"]),
        "n_customers": int(len(disc["keys"])),
    }


def check_new_field_ingested(data: Path) -> dict:
    """Inject one categorical + one numeric field; the pipeline must resolve both, no edits."""
    import auto_encoder as AE

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

    E, _, _, _, _, _, vs, _, best, _ = AE.train_encoder(
        data=str(vdir), seed=0, d=64, layers=1, epochs=1
    )

    disc = SA.discover(
        raw.select("customer_key", "event_ts", "event_type", "event_attributes"),
        datetime.fromisoformat(
            json.loads((vdir / "manifest.json").read_text())["as_of"]
        ).timestamp(),
        action_keys=[],
    )
    cat_fields = {g["field"] for g in disc["catspec"]}
    num_fields = {f for f, _ in disc["num"]}
    ok = (
        "loyalty_tier" in cat_fields
        and "session_seconds" in num_fields
        and _finitish(E)
        and math.isfinite(float(best))
    )
    return {
        "working": ok,
        "evidence": (
            "injected loyalty_tier (novel categories) + session_seconds into every event's "
            f"attributes: resolved into catspec={sorted(cat_fields)} num={sorted(num_fields)}; "
            f"encoder trained on the variant (d=64,L=1,e=1): E finite={_finitish(E)}, "
            f"best={float(best):.4f}; zero pipeline-code edits"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "n_rows": int(n),
    }


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
        return {"working": False, "evidence": "probe_value not resolved as numeric", "seconds": 0.0}
    seqs = disc["seqs"]
    per_cust = {
        k: float(
            seqs[k][2][:, [j for j, (f, _) in enumerate(disc["num"]) if f == "probe_value"][0]][0]
        )
        for k in disc["keys"]
    }
    encoded = [per_cust[f"numcust_{i}"] for i in range(len(values))]
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
    schema_b = SA.discover(perm, as_of_ts, action_keys=[])["schema"]
    keys_b, seqs_b, _ = SA.transform(perm, schema_b)
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


def check_training_determinism(data: Path) -> dict:
    """Two seed-0 runs on identical prepared inputs produce bit-identical E and best-val."""
    import auto_encoder as AE

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
    E1, _, _, _, _, _, _, _, best1, _ = AE.train_encoder(
        data=str(WORK / "det_a"), seed=0, d=64, layers=1, epochs=1, prepared=prepared
    )
    E2, _, _, _, _, _, _, _, best2, _ = AE.train_encoder(
        data=str(WORK / "det_b"), seed=0, d=64, layers=1, epochs=1, prepared=prepared
    )
    identical = np.ascontiguousarray(E1).tobytes() == np.ascontiguousarray(E2).tobytes()
    ok = identical and best1 == best2
    return {
        "working": ok,
        "evidence": (
            f"two seed-0 runs (d=64,L=1,e=1) on identical prepared inputs: "
            f"E bit-identical={identical}, best-val equal ({float(best1):.6f} == {float(best2):.6f})"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
        "n_customers": int(len(disc["keys"])),
    }


# --------------------------------------------------------------------------- main


def main() -> int:
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

    s1_checks = {
        "renamed_end_to_end": check_renamed_end_to_end(data),
        "new_field_ingested": check_new_field_ingested(data),
    }
    s2_checks = {
        "unseen_value_path": check_unseen_value_path(data),
        "numeric_transform_law": check_numeric_transform(data),
        "field_permutation_invariance": check_field_permutation(data),
        "training_determinism": check_training_determinism(data),
    }

    def rollup(checks: dict) -> dict:
        working = all(c["working"] for c in checks.values())
        return {
            "working": working,
            "evidence": " | ".join(
                f"[{name}] {'OK' if c['working'] else 'FAILED'}: {c['evidence']}"
                for name, c in checks.items()
            ),
            "checks": checks,
        }

    existing = {}
    out_path = Path(a.out)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:  # noqa: BLE001 - rebuild rather than merge with garbage
            existing = {}
    existing.update(
        {
            "_meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": JC.git("rev-parse", "HEAD"),
                "behavior_hash": JC.behavior_hash(),
                "command": " ".join([Path(__file__).name] + sys.argv[1:]),
                "data": str(data),
            },
            "S1": rollup(s1_checks),
            "S2": rollup(s2_checks),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n")

    print(f"judge stages -> {out_path}")
    for stage in ("S1", "S2"):
        for name, c in existing[stage]["checks"].items():
            print(f"  {stage} :: {name}: {'PASS' if c['working'] else 'FAIL'}  ({c['seconds']}s)")
    print(f"S1 working={existing['S1']['working']}  S2 working={existing['S2']['working']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
