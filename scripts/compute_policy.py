"""The compute policy — where the GPU, the cores, and the single thread each belong.

`AGENTS.md` binds the project to *maximise GPU utilisation, parallelism and multi-threading*.
This module is that directive made mechanical, and it records the measurements that decide
each question rather than asserting a preference.

    GPU (PyTorch)        MANDATORY. Training and every model forward run on CUDA under the
                         project memory slice. Measured 2026-11-25 (DEC-659): the optimizer
                         step is GPU-bound after the collate rewrite.
    CPU cores (joblib)   DEFAULT for independent fits. `jobs()` sizes the worker pool; the
                         S4 stage's per-(view,target) fits went 546 s -> 124 s (4.4x,
                         byte-identical output) on this 20-core box (DEC-660).
    GPU (Polars)         AVAILABLE, OPT-IN, NOT DEFAULT. `pl.GPUEngine` works on this
                         aarch64 GB10 (cudf-polars 26.10.2 + pylibcudf-cu12), but on the real
                         demo stream the representative aggregation measured CPU 0.060 s vs
                         GPU 0.066 s (0.9x) — the queries are too small to repay the transfer
                         and planning cost (DEC-661). Enable with LG_POLARS_GPU=1 for heavy
                         joins/aggregations; measure before adopting for a new shape.
    DuckDB               CPU ONLY. DuckDB has no GPU execution backend; its shipped
                         extensions are CPU (verified: core_functions/icu/json). Using it is
                         not a directive violation — there is no GPU option to choose.

Rule: a new producer states which tier it uses and why. "It was easier" is not a reason when
the directive names the tiers explicitly.
"""

from __future__ import annotations

import os


def jobs() -> int:
    """Worker count for independent CPU fits. LG_JOBS overrides; default min(8, cores)."""
    env = os.environ.get("LG_JOBS")
    if env:
        return max(1, int(env))
    return max(1, min(8, os.cpu_count() or 2))


def polars_engine():
    """The Polars engine to pass to `.collect(engine=...)`, or None for the CPU engine.

    Opt-in only: set LG_POLARS_GPU=1. Returns None when the GPU engine is unavailable so a
    caller never has to guard the import; `raise_on_fail=False` keeps a fallback to CPU.
    """
    if os.environ.get("LG_POLARS_GPU") != "1":
        return None
    try:
        import polars as pl

        return pl.GPUEngine(raise_on_fail=False)
    except Exception:  # noqa: BLE001 - no GPU dataframe stack installed
        return None
