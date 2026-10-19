"""Retag an existing model directory to a unique artifact identity.

The data revision does not encode the training sample, so two runs (e.g. 400 and
2000 sample-A customers) can share a tag. This renames the checkpoint, Layer-B,
serve projection and registry to `--new-tag` and updates the embedded tag fields;
it also copies/updates the newest portfolio receipt for the old tag.

Usage:
  python3 scripts/retag_model.py --ckpt-dir <dir> --old-tag <old> --new-tag <new>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _safe(tag: str) -> str:
    return tag.replace(".", "_")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--old-tag", required=True, dest="old_tag")
    ap.add_argument("--new-tag", required=True, dest="new_tag")
    ap.add_argument(
        "--portfolio-dir",
        default=str(ROOT / "looking_glass" / "artifacts" / "cfm" / "portfolio"),
    )
    a = ap.parse_args(argv)

    d = Path(a.ckpt_dir)
    old, new = _safe(a.old_tag), _safe(a.new_tag)
    renamed = []
    for p in sorted(d.iterdir()):
        if old in p.name and p.is_file():
            target = d / p.name.replace(old, new)
            shutil.move(str(p), str(target))
            renamed.append(target.name)
    for p in d.glob(f"registry_{new}.json"):
        rec = json.loads(p.read_text())
        rec["tag"] = a.new_tag
        p.write_text(json.dumps(rec, indent=1) + "\n")
    for p in d.glob(f"layer_b_{new}.pkl.json"):
        rec = json.loads(p.read_text())
        rec["tag"] = a.new_tag
        p.write_text(json.dumps(rec, indent=1) + "\n")
    for p in d.glob(f"serve_proj_{new}.npz"):
        rec_path = p.with_name(p.name.replace(".npz", ".json"))
        if rec_path.exists():
            rec = json.loads(rec_path.read_text())
            rec["tag"] = a.new_tag
            rec_path.write_text(json.dumps(rec, indent=1) + "\n")

    port_dir = Path(a.portfolio_dir)
    port = sorted(port_dir.glob(f"portfolio_{old}_*.json"))
    copied = []
    if port:
        src = port[-1]
        dst = port_dir / src.name.replace(old, new)
        rec = json.loads(src.read_text())
        rec["tag"] = a.new_tag
        dst.write_text(json.dumps(rec, indent=2) + "\n")
        copied.append(dst.name)
    print(
        json.dumps(
            {
                "ckpt_dir": str(d),
                "old_tag": a.old_tag,
                "new_tag": a.new_tag,
                "renamed": renamed,
                "portfolio": copied,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    main()
