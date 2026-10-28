#!/usr/bin/env python3
"""Emit the environment receipt: what runtime this repo is actually executing on.

The GPU-first policy (DEC-389) requires a measured record of the runtime before any hot path is
claimed GPU-accelerated, and the freshness law (DEC-390) requires behavior-bearing artifacts to
carry the commit that produced them. This script is that record for the environment itself.

Usage:
    python3 scripts/env_receipt.py [--out docs/report_cards/experiments/env_receipt.json]

No GPU? The receipt still emits, with gpu.available=false — the CPU fallback is kept (DEC-389),
and any row that requires GPU acceleration renders UNKNOWN rather than PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # pragma: no cover - receipt must never crash the caller
        return f"<git unavailable: {exc}>"


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return {"available": False, "raw": out.stderr.strip() or line}
        return {
            "available": True,
            "name": parts[0],
            "driver_version": parts[1],
            "compute_cap": parts[2],
            "memory_total": parts[3],
        }
    except Exception as exc:
        return {"available": False, "raw": repr(exc)}


def _torch_block() -> dict:
    block: dict = {"installed": False}
    try:
        import torch
    except Exception as exc:
        block["error"] = repr(exc)
        return block
    block.update(
        {
            "installed": True,
            "version": torch.__version__,
            "built_with_cuda": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )
    if block["cuda_available"]:
        block["device_name"] = torch.cuda.get_device_name(0)
        block["capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
        # Kernel-availability smoke: an actual matmul on device, not just a device listing.
        try:
            import time

            n = 2048
            a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            t0 = time.perf_counter()
            c = (a @ b).float().sum().item()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            block["matmul_smoke"] = {
                "ok": bool(c == c),  # not NaN
                "n": n,
                "dtype": "bfloat16",
                "seconds": round(dt, 4),
                "tflops_approx": round(2 * n**3 / dt / 1e12, 2),
            }
        except Exception as exc:
            block["matmul_smoke"] = {"ok": False, "error": repr(exc)}
    return block


def _cupy_block() -> dict:
    try:
        import cupy  # type: ignore
    except Exception as exc:
        return {"installed": False, "error": repr(exc)}
    try:
        a = cupy.random.rand(1024, 1024)
        return {"installed": True, "version": cupy.__version__, "smoke_ok": bool((a @ a).sum().item() > 0)}
    except Exception as exc:
        return {"installed": True, "version": cupy.__version__, "smoke_ok": False, "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "docs" / "report_cards" / "experiments" / "env_receipt.json"),
    )
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args()

    receipt = {
        "kind": "env_receipt",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "logical_cpus": os.cpu_count(),
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "packages": {
            name: _pkg_version(name)
            for name in (
                "numpy",
                "torch",
                "scipy",
                "scikit-learn",
                "polars",
                "pyarrow",
                "duckdb",
                "pytest",
                "ruff",
            )
        },
        "gpu": _nvidia_smi(),
        "torch": _torch_block(),
        "cupy": _cupy_block(),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"env receipt -> {out}")
    if a.do_print:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    ok = bool(receipt["torch"].get("cuda_available"))
    print(f"GPU-USABLE: {ok}")
    return 0  # informational: a CPU-only env is legal, it just cannot certify GPU rows


if __name__ == "__main__":
    raise SystemExit(main())
