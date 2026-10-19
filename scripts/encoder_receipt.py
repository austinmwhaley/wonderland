"""Measured encoder receipt: self-supervised, frozen, schema-universal + identity.

Replaces the old hardcoded `{"self_supervised": true, "frozen_donor": true}`.

  * self_supervised: the registry's training objectives contain no label/outcome
    targets (only the stream's self-supervised families).
  * frozen_donor: `freeze_encoder` makes every parameter require_grad=False and a
    second freeze leaves the state hash unchanged (no downstream mutation).
  * schema_universal: from a real degenerate-schema suite receipt.
  * identity: tag, ckpt dir, sample_a/n_train/data signature, and SHA256 of the
    checkpoint + Layer-B + serve files (resolves tag collisions).

Usage:
  python3 scripts/encoder_receipt.py --ckpt-dir <dir> --tag <t> \
      --schema-suite <schema_suite.json> --out <encoder.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SS_OBJECTIVES = {
    "agg",
    "contrast",
    "dt",
    "effect",
    "jepa",
    "mask",
    "mtp",
    "next",
    "occur",
    "order",
    "query",
    "sf",
}


def _is_self_supervised(name: str) -> bool:
    if name in _SS_OBJECTIVES:
        return True
    return name.startswith(("attr_next:", "num_next:"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--schema-suite", default=None, dest="schema_suite")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.probes import freeze_encoder

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    reg_path = Path(a.ckpt_dir) / f"registry_{a.tag.replace('.', '_')}.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    objectives = sorted((reg.get("task_weights") or {}).keys())
    bad = [o for o in objectives if not _is_self_supervised(o)]
    self_supervised = bool(objectives) and not bad

    before = _state_hash(model)
    freeze_encoder(model)
    frozen = all(not p.requires_grad for p in model.parameters())
    after = _state_hash(model)
    stable = before == after

    suite = json.loads(Path(a.schema_suite).read_text()) if a.schema_suite else {}
    schema_ok = bool(suite and int(suite.get("passed", 0)) >= int(suite.get("total", 1) or 1))

    ck = Path(a.ckpt_dir)
    files = {}
    for pat in (
        f"cfm_{a.tag.replace('.', '_')}.pt",
        f"layer_b_{a.tag.replace('.', '_')}.pkl",
        f"serve_proj_{a.tag.replace('.', '_')}.npz",
    ):
        p = ck / pat
        if p.exists():
            files[p.name] = _sha256(p)

    rec = {
        "schema_universal": schema_ok,
        "self_supervised": bool(self_supervised and not bad),
        "frozen_donor": bool(frozen and stable),
        "public_objectives": objectives,
        "unexpected_label_objectives": bad,
        "state_hash_before": before,
        "state_hash_after": after,
        "identity": {
            "tag": a.tag,
            "ckpt_dir": str(ck),
            "sample_a_customers": (reg.get("config") or {}).get("sample_a_customers"),
            "n_train_sequences": reg.get("n_train_sequences"),
            "data_signature": reg.get("data_signature"),
            "revision": reg.get("revision"),
            "file_sha256": files,
        },
    }
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "identity"}, indent=1))
    return rec


if __name__ == "__main__":
    main()
