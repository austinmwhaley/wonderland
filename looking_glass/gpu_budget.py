"""GPU memory budget — train on a GPU that is already spoken for (DEC-423).

The GB10 is a **unified-memory** device: the resident LLM server and this project's
trainer draw on ONE pool. Over-allocating does not fail inside the process — it fails
in the driver (`NVRM: nvCheckOkFailedNoLog: Out of memory [NV_ERR_NO_MEMORY]`) and
takes the machine down with it (measured: three hard crashes in one working session,
each ~10 min into GPU training while the LLM server was resident). An over-subscribed
GPU is therefore not a slow GPU; it is an unsafe one.

Policy (operator-set, 2026-10-29): the LLM server keeps what it needs — its
reservation is static, so it does NOT shrink when idle and must not be displaced —
the system keeps a floor for OS + desktop, and the trainer gets **the rest**, on the
GPU, and never more than that.

Three mechanisms, deliberately different in kind:

  * a HARD cap (`torch.cuda.set_per_process_memory_fraction`): a surprise then raises
    `torch.cuda.OutOfMemoryError` in Python — a receipted failure that costs a run —
    instead of a driver fault that costs the box. This is the seatbelt.
  * DERIVED demand: the budget comes from `MemAvailable`/driver-free at launch minus the
    system floor minus the device memory this process holds outside the allocator
    (`gpu_reserve_gb`), not from a constant, so the same command works whether the LLM
    server is up or down (Principle 1: no magic numbers; Principle 11: emit the receipt).
  * an EXTERNAL guardrail (`scripts/gpu_watchdog.py`): the fraction cap governs PyTorch's
    *caching allocator* only — cuDNN workspaces, Triton compile buffers and CUDA graphs
    allocate around it — so a process outside the job enforces the floor by force.

If the cap cannot be installed, a CUDA run REFUSES to start. The first version of this
module swallowed such a failure and returned silently; the trainer then ran uncapped, took
21.4 GB, and drove `MemAvailable` to 7.7 GB against an 8.0 GB floor before the watchdog
killed it. A guard that fails quietly is worse than no guard: it manufactures confidence
(Principle 7: reject, do not clamp).

    from looking_glass import gpu_budget
    gpu_budget.enforce(cfg)              # idempotent; records cfg._gpu_budget_receipt
    gpu_budget.release(cfg, "whiten")    # phase boundary: return cached segments + record peak
    gpu_budget.receipt(cfg)              # what goes into the registry (M16)
"""

from __future__ import annotations

import os
import resource

_GB = 1024**3
_MEMINFO = "/proc/meminfo"
#: cgroup v2 files. Inside a `docker --memory=20g` container these carry the container's
#: ceiling; on the bare host `memory.max` is usually absent (root cgroup) and reads as no limit.
_CGROUP_LIMIT = "/sys/fs/cgroup/memory.max"
_CGROUP_USED = "/sys/fs/cgroup/memory.current"

# principle-ok: documented conservative fallback, not a tuning knob. Below ~2% of the
# pool a CUDA context plus a single batch cannot exist, so a smaller budget means
# "nothing is coming back" — refuse with the reason instead of dying obscurely.
_MIN_VIABLE_FRACTION = 0.02

_APPLIED: dict = {}  # process-wide: the receipt from the first enforce()


def available_gb(path: str = _MEMINFO) -> float:
    """`MemAvailable` in GB — what the kernel will actually give a new allocation."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def cgroup_memory_gb(
    limit_path: str = _CGROUP_LIMIT, used_path: str = _CGROUP_USED
) -> tuple[float, float]:
    """(limit, current) of THIS process's cgroup in GB; (0, 0) when there is no finite cap.

    A container's `--memory` is a HARD ceiling on host RAM, but on the GB10 the CUDA driver
    allocates from the same pool WITHOUT being charged to that cgroup (measured with vLLM:
    the container reported 337 MiB while holding 97 GiB). So the container ceiling cannot
    cap the GPU by itself; the allocator fraction must be derived from it instead
    (DEC-450). Reading it here is what makes the budget the tighter of two real limits
    rather than the host's optimistic `MemAvailable`, which inside a container still
    reports the whole machine.
    """

    def _read(path: str) -> float | None:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            return None
        if raw == "max" or not raw.isdigit():
            return None  # unlimited / unreadable -> not a ceiling
        gb = int(raw) / _GB
        return gb if gb > 0 else None

    return (_read(limit_path) or 0.0), (_read(used_path) or 0.0)


def compute_budget(
    *,
    mem_available_gb: float,
    driver_free_gb: float,
    total_gb: float,
    system_floor_gb: float,
    reserve_gb: float,
    limit_gb: float = 0.0,
    container_limit_gb: float = 0.0,
    container_used_gb: float = 0.0,
) -> dict:
    """Pure budget arithmetic (no CUDA), so the policy is testable off-device.

    The usable amount is the *tighter* of (MemAvailable − system floor) and what the CUDA
    driver says is free, less `reserve_gb` of device memory this process holds OUTSIDE the
    capped allocator (CUDA context, cuBLAS workspaces). `reserve_gb` is a required
    argument, not a default: `CFMConfig.gpu_reserve_gb` is its single source of truth.
    `limit_gb > 0` is an operator override — taken verbatim, no deduction, because the
    operator has accepted the risk.

    `container_limit_gb > 0` adds a HARDER ceiling: the container's cgroup (DEC-450). Device
    allocations are not charged to that cgroup, so the guard must leave the container's
    current host usage (`container_used_gb`) plus this process's device overhead inside the
    limit — budget + used + reserve <= limit. An explicit `limit_gb` is clamped too: an
    operator may accept the co-tenant risk, but not override a kernel ceiling.
    """
    if limit_gb and float(limit_gb) > 0:
        budget, source = float(limit_gb), "explicit"
    else:
        headroom = min(float(mem_available_gb) - float(system_floor_gb), float(driver_free_gb))
        budget = max(0.0, headroom - float(reserve_gb))
        source = "derived"
    container_clamp = 0.0
    if container_limit_gb and float(container_limit_gb) > 0:
        limit = float(container_limit_gb)
        container_clamp = max(0.0, limit - float(container_used_gb) - float(reserve_gb))
        if container_clamp < budget:
            budget, source = container_clamp, f"{source}+cgroup"
    fraction = 0.0 if total_gb <= 0 else min(1.0, budget / float(total_gb))
    return {
        "applied": False,
        "source": source,
        "mem_total_gb": round(float(total_gb), 2),
        "mem_available_gb": round(float(mem_available_gb), 2),
        "driver_free_gb": round(float(driver_free_gb), 2),
        "system_floor_gb": round(float(system_floor_gb), 2),
        "reserve_gb": round(float(reserve_gb), 2),
        "container_limit_gb": round(float(container_limit_gb), 2),
        "container_used_gb": round(float(container_used_gb), 2),
        "container_clamp_gb": round(container_clamp, 2),
        "budget_gb": round(budget, 2),
        "fraction": round(fraction, 4),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"),
        "phases": [],
    }


def _device_index(torch, device: str) -> int:
    """CUDA device INDEX (the allocator API takes an int, not a 'cuda' string).

    Measured defect: passing the string `\"cuda\"` raises `ValueError: Expected a
    torch.device with a specified index or an integer`. A guard installed with the wrong
    signature is worse than no guard, so this normalizes explicitly rather than probing.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"not a CUDA device: {device!r}")
    return torch.cuda.current_device() if dev.index is None else int(dev.index)


def enforce(cfg, *, device: str | None = None, rebase: bool = False) -> dict:
    """Cap this process to its share of the GPU pool. Idempotent, CPU-safe.

    Returns the receipt (also stored on `cfg._gpu_budget_receipt`). A CPU-configured run
    is a quiet no-op — this policy protects a GPU co-tenant, so it must not break the
    CPU fallback (Principle 5: keep the fallback).

    A CUDA run whose cap CANNOT be installed is a hard refusal, not a warning. The first
    version of this module swallowed that failure and returned silently; the trainer then
    ran uncapped, took 21.4 GB, and drove `MemAvailable` under the system floor (7.7 GB
    against 8.0) until the external watchdog SIGTERM'd it. Reject, do not clamp (P7).

    `rebase=True` re-derives the cap from the memory free RIGHT NOW and re-installs it.
    The launch-time estimate is made before the stream is loaded, and on the GB10 a host
    allocation and a device allocation draw on the SAME pool (measured: 11.35 GiB of
    `torch.empty` moved `MemAvailable` by 11.35 GiB), so a cap that was affordable at launch
    can be ~2.5 GB too big by the time the step loop starts — and the run then dies at the
    floor instead of fitting. Rebase only at a phase boundary, after `release()` has emptied
    the allocator; never mid-phase.
    """
    receipt = getattr(cfg, "_gpu_budget_receipt", None) or dict(_APPLIED)
    if receipt.get("applied") and not rebase:
        return receipt
    try:
        import torch

        if device is None:
            device = str(getattr(cfg, "device", "cuda"))
        if not device.startswith("cuda"):
            rec = dict(receipt, applied=False, skipped=f"device={device}")
            setattr(cfg, "_gpu_budget_receipt", rec)
            return rec  # CPU run: nothing to protect, nothing to report
        if not torch.cuda.is_available():
            raise SystemExit(
                f"gpu-budget: device={device!r} but CUDA is unavailable, so the co-tenant "
                "cap cannot be installed. Train on CPU (--set device=cpu) or fix the "
                "CUDA environment; running 'cuda' without the cap is how the last three "
                "driver-level OOM crashes happened."
            )
        idx = _device_index(torch, device)
        free, total = torch.cuda.mem_get_info(idx)
        cg_limit, cg_used = cgroup_memory_gb()
        rec = compute_budget(
            mem_available_gb=available_gb(),
            driver_free_gb=free / _GB,
            total_gb=total / _GB,
            system_floor_gb=float(getattr(cfg, "gpu_system_floor_gb", 8.0)),
            reserve_gb=float(getattr(cfg, "gpu_reserve_gb", 1.0)),
            limit_gb=float(getattr(cfg, "gpu_mem_limit_gb", 0.0)),
            container_limit_gb=cg_limit,
            container_used_gb=cg_used,
        )
        if rec["fraction"] < _MIN_VIABLE_FRACTION:
            where = (
                f"the container's {rec['container_limit_gb']} GB ceiling "
                f"({rec['container_used_gb']} GB already used) and "
                if rec["container_limit_gb"] > 0
                else "the LLM server holds the rest"
            )
            raise SystemExit(
                f"gpu-budget: only {rec['budget_gb']} GB is left after {where} the "
                f"{rec['system_floor_gb']} GB system floor and {rec['reserve_gb']} GB "
                "device overhead — too little to train on. Free the co-tenant, raise "
                "WONDER_MEM, or set an explicit budget and accept the risk: "
                "  --set gpu_mem_limit_gb=<GB>"
            )
        try:
            torch.cuda.set_per_process_memory_fraction(rec["fraction"], idx)
        except TypeError:  # older torch: fraction only, applied to the current device
            torch.cuda.set_device(idx)
            torch.cuda.set_per_process_memory_fraction(rec["fraction"])
        rec["applied"] = True
        rec["rebases"] = int(receipt.get("rebases", 0)) + (1 if rebase else 0)
        # A rebase must not erase the phase history it was built on (one receipt, not two).
        rec["phases"] = list(receipt.get("phases", []))
        _APPLIED.clear()
        _APPLIED.update(rec)
        print(
            f"[gpu-budget] {'RECAP' if rebase else 'cap'} {rec['budget_gb']} GB "
            f"({rec['fraction']:.3f} of {rec['mem_total_gb']} GB) source={rec['source']} "
            f"avail={rec['mem_available_gb']} GB driver_free={rec['driver_free_gb']} GB "
            f"floor={rec['system_floor_gb']} GB reserve={rec['reserve_gb']} GB "
            f"container={rec['container_limit_gb']} GB (used {rec['container_used_gb']}, "
            f"clamp {rec['container_clamp_gb']}) "
            f"dev={idx} alloc={rec['alloc_conf']}",
            flush=True,
        )
    except SystemExit:
        raise
    except Exception as exc:
        # NEVER continue uncapped on a shared device: a guard that silently fails is
        # false confidence, and the measured cost here was a machine-killing OOM.
        reason = f"{type(exc).__name__}: {exc}"
        rec = dict(receipt, applied=False, skipped=reason)
        setattr(cfg, "_gpu_budget_receipt", rec)
        print(f"[gpu-budget] FAILED to install the cap ({reason})", flush=True)
        raise SystemExit(
            f"gpu-budget: refusing to train uncapped on a shared GPU ({reason}). "
            "Fix the cap, or run on CPU (--set device=cpu) where there is no co-tenant."
        ) from None
    setattr(cfg, "_gpu_budget_receipt", rec)
    return rec


def release(cfg, phase: str, *, reset_peak: bool = True, rebase: bool = False) -> dict | None:
    """Phase boundary: return cached segments to the driver and record the peak.

    Training, whitening, product build and portfolio grading have completely different
    allocation shapes; without this the allocator hoards the training peak right through
    the bulk-write phases (fragmentation inflates `reserved` and starves the co-tenant).

    `rebase=True` re-derives the cap for the NEXT phase from what is free NOW. It is
    legitimate at exactly one call site — the informed moment before the step loop, where
    the stream is resident and the allocator has just been emptied. Everywhere else it is
    a lie: at an *abort* boundary MemAvailable is at its lowest precisely because the
    phase blew up, so re-deriving there invents a budget the run never needed and (as
    measured 2026-10-29) can refuse the run while the real crash goes unreported.
    """
    try:
        import torch

        base = getattr(cfg, "_gpu_budget_receipt", None) or dict(_APPLIED)
        if not base.get("applied") or not torch.cuda.is_available():
            return None  # cap never engaged (CPU run / guard skipped) -> nothing to report
        dev = 0
        stats = torch.cuda.memory_stats(dev)
        # Host RSS is part of this budget: on the GB10 a device allocation and a host
        # allocation draw on ONE pool (measured 2026-10-29: 11.35 GiB of `torch.empty`
        # moved `MemAvailable` by 11.35 GiB, and driver-free tracks it minus a constant
        # ~4.4 GB). The fraction cap bounds only the device side, so the host side is
        # reported here and enforced by the watchdog.
        rec = {
            "phase": phase,
            "peak_allocated_gb": round(stats.get("allocated_bytes.all.peak", 0) / _GB, 2),
            "peak_reserved_gb": round(stats.get("reserved_bytes.all.peak", 0) / _GB, 2),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2),
            "mem_available_gb": round(available_gb(), 2),
        }
        torch.cuda.empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(dev)
        base.setdefault("phases", []).append(rec)
        setattr(cfg, "_gpu_budget_receipt", base)
        print(
            f"[gpu-budget] phase={phase} peak_alloc={rec['peak_allocated_gb']} GB "
            f"peak_reserved={rec['peak_reserved_gb']} GB peak_rss={rec['peak_rss_gb']} GB "
            f"avail={rec['mem_available_gb']} GB",
            flush=True,
        )
        # The informed moment only when the caller asked for it (see the docstring): the
        # allocator is empty and the host footprint of the phase just finished is real.
        if rebase:
            # A phase-boundary re-derivation may TIGHTEN the cap, but it may never abort a
            # run that was viable at launch: what is free here reflects what the finished
            # phase left behind, not what the pool can supply. Refusing here is how a guard
            # starts rejecting work it can genuinely do — measured 2026-10-29, a cap
            # re-derived at an abort boundary came out at 0.0 GB and killed a run that had
            # already trained 99 steps, masking the real crash. The floor is the watchdog's
            # job; report and continue.
            try:
                recap = enforce(cfg, rebase=True)
            except SystemExit as exc:
                print(f"[gpu-budget] RECAP skipped, keeping the current cap: {exc}", flush=True)
            else:
                rec["recap_budget_gb"], rec["recap_fraction"] = (
                    recap["budget_gb"],
                    recap["fraction"],
                )
                # The next phase must at least be able to reproduce this one. If it cannot,
                # say so HERE rather than discovering it as an OOM three phases later.
                if recap["budget_gb"] < rec["peak_allocated_gb"]:
                    print(
                        f"[gpu-budget] WARN re-derived cap {recap['budget_gb']} GB is below "
                        f"the {phase} peak {rec['peak_allocated_gb']} GB — the next phase will "
                        "not fit; raise the pool share or shrink the phase",
                        flush=True,
                    )
        return rec
    except Exception as exc:
        # A silent guard is indistinguishable from a dead one, and this one guards memory.
        print(f"[gpu-budget] release({phase!r}) failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def receipt(cfg) -> dict:
    """The registry-visible receipt (empty dict when the cap never applied)."""
    return dict(getattr(cfg, "_gpu_budget_receipt", None) or _APPLIED or {})
