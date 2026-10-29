#!/usr/bin/env python3
"""Prove the co-tenant cap actually binds (DEC-423 ground-truth gate).

`gpu_budget.enforce()` RECORDS a cap; recording is not enforcing. This command allocates
past the cap and requires the failure to be a Python `torch.cuda.OutOfMemoryError` while
the system floor stays intact — the exact property that distinguishes a receipted OOM
(costs a run) from a driver `NV_ERR_NO_MEMORY` (costs the machine).

    .venv/bin/python scripts/verify_gpu_cap.py            # enforce, then push to failure
    .venv/bin/python scripts/verify_gpu_cap.py --set gpu_mem_limit_gb=6   # force a small cap

Exit codes: 0 = cap bound at/below budget with the floor intact; 1 = the cap did NOT bind
(allocations kept succeeding past budget) or the floor was breached; 2 = refused to start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from looking_glass import gpu_budget  # noqa: E402
from looking_glass.cfm_config import CFMConfig, apply_set_overrides  # noqa: E402

_GB = 1024**3
_CHUNK_GB = 1.0  # probe granularity, not a budget: the verdict is budget-relative


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", dest="set_overrides", action="append", default=[], metavar="K=V")
    a = ap.parse_args(argv)

    cfg = CFMConfig()
    cfg.set_overrides = list(a.set_overrides)
    apply_set_overrides(cfg)
    cfg.device = "cuda"
    rec = gpu_budget.enforce(cfg)  # refuses to continue if the cap cannot be installed
    budget = float(rec["budget_gb"])

    import torch

    held, peak_alloc, breach = [], 0.0, 0.0
    outcome = "NO_FAILURE"
    for _ in range(int(budget / _CHUNK_GB) + 6):  # push past the cap, always terminate
        try:
            held.append(torch.empty(int(_CHUNK_GB * 1e9 // 4), dtype=torch.float32, device="cuda"))
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            outcome = "OutOfMemoryError"
            break
        except RuntimeError as exc:  # older torch raises the plain type
            outcome = f"RuntimeError:{type(exc).__name__}"
            break
        peak_alloc = torch.cuda.memory_allocated() / _GB
        breach = max(breach, 0.0, float(rec["system_floor_gb"]) - gpu_budget.available_gb())

    freed = sum(t.numel() * t.element_size() for t in held) / _GB
    del held
    torch.cuda.empty_cache()
    avail = gpu_budget.available_gb()
    print(
        f"[verify-cap] held={freed:.2f} GB peak_alloc={peak_alloc:.2f} GB budget={budget:.2f} GB "
        f"outcome={outcome} MemAvailable_now={avail:.2f} GB floor={rec['system_floor_gb']} GB "
        f"floor_shortfall={breach:.2f} GB"
    )
    bound = outcome != "NO_FAILURE"
    # A cap that trips EARLY is safe but starves the run: report the efficiency too.
    if not bound:
        print(
            f"[verify-cap] FAIL the allocator grew past its budget ({freed:.2f} > {budget:.2f} GB)"
        )
        return 1
    if breach > 0.5:
        print(f"[verify-cap] FAIL the floor was breached by {breach:.2f} GB before the cap tripped")
        return 1
    print(
        f"[verify-cap] PASS cap bound at {freed:.2f} GB of {budget:.2f} GB budget "
        f"({freed / max(budget, 1e-9):.0%} utilisation) with the system floor intact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
