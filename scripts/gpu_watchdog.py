#!/usr/bin/env python3
"""External guardrail for the co-tenant GPU budget (DEC-423).

`looking_glass.gpu_budget` caps PyTorch's caching allocator, but cuDNN workspaces,
Triton compile buffers and CUDA graphs allocate around that cap. On a unified-memory
GB10 those are the allocations that reach zero and fault the driver
(`NV_ERR_NO_MEMORY`), which does not kill the process — it takes the machine down.
This watchdog is the part that lives OUTSIDE the capped process and enforces the
system floor by force: if free memory falls to the floor, the repo's GPU jobs are
stopped (SIGTERM, then SIGKILL) before the driver does it for us.

    .venv/bin/python scripts/gpu_watchdog.py --floor-gb 8 --interval 5 --pidfile /tmp/guard.pid
    .venv/bin/python scripts/gpu_watchdog.py --once          # report and exit
    .venv/bin/python scripts/gpu_watchdog.py --dry-run        # observe, never kill

Stop it by pid, never by a shell job id: `kill $(cat /tmp/guard.pid)`. Five stale guards
in the wild came from `kill $!` after `cd x && rm y && python watchdog.py &`, which signals
the wrapper and leaves the guard policing jobs nobody asked it to police (and it self-ends
with its launcher via prctl, so a detached launch cannot leak one either).

Exit codes: 0 healthy (or --once report), 2 when a kill was issued (so a caller can
tell a clean finish from a memory-forced stop).
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import time
from pathlib import Path

DEFAULT_PATTERNS = (  # principle-ok: process identities, not tuning constants
    "looking_glass.customer_foundation_model",
    # every producer that can hold a large frame, model or DuckDB aggregate. A job this guard
    # does not recognise is a job it cannot stop, which is how the 2026-10-29 crash happened: the
    # runaway ran inside a script that was not on the list. `tests/test_memory_guard.py` fails if
    # a `scripts/measure_*.py` producer exists that is not covered here, so the list cannot drift
    # behind the repo again.
    "measure_battery.py",
    "measure_battery_law.py",
    "measure_b_props.py",
    "build_compact_e.py",
    "measure_ceiling.py",
    "measure_certification_law.py",
    "measure_path_a_provenance.py",
    "measure_construction.py",
    "measure_construction_law.py",
    "measure_controls.py",
    "measure_data.py",
    "measure_decision_law.py",
    "measure_decision_head_protocol.py",
    "measure_value_block.py",
    "measure_demo_plugins.py",
    "measure_donor.py",
    "measure_dynamics_law.py",
    "measure_access_axis.py",
    "measure_proxy_alignment.py",
    "measure_effect_diagnostic.py",
    "measure_identity.py",
    "measure_input_law.py",
    "measure_judge_law.py",
    "measure_minimality_law.py",
    "measure_objectives.py",
    "measure_transfer_law.py",
    "measure_robustness_law.py",
    "measure_stage_decomposition.py",
    "measure_block_serving.py",
    "measure_shape_battery.py",
    "measure_rank_attainability.py",
    "measure_regime_readout.py",
    "measure_response.py",
    "measure_served_object.py",
    "measure_sufficiency_law.py",
    "measure_rl_v2.py",
    "measure_runtime.py",
    "measure_testbed_power.py",
    "measure_views_law.py",
    "env_receipt.py",
)


def mem_available_gb(path: str = "/proc/meminfo") -> float:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("inf")  # unreadable -> never kill on a guess


def gpu_jobs(patterns: tuple[str, ...], proc_root: Path = Path("/proc")) -> list[int]:
    """Repo GPU processes, excluding this watchdog and its own shell wrapper.

    `proc_root` is a seam, not a knob: tests point it at a fixture tree so the kill
    list itself is covered (a watchdog that cannot see its targets is worse than none).
    """
    me = os.getpid()
    out: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmd = (proc_root / entry.name / "cmdline").read_bytes().decode(errors="replace")
        except OSError:  # pid vanished between listing and read, or unreadable: not a target
            continue
        argv = [a for a in cmd.split("\0") if a]
        if not argv:
            continue
        # A SHELL whose command text merely mentions a producer is not the producer. Matching the
        # command line as a string killed the launcher (`cd repo && nohup python scripts/x.py`)
        # because the wrapper is outside the slice while the job it launched is inside — punishing
        # the messenger and, worse, signalling whatever shell happened to type the command.
        # So: a `.py` pattern must be the BASENAME of an argument to a python interpreter, and a
        # module pattern (no ".py") matches any argument.
        exe = os.path.basename(argv[0])
        is_python = exe.startswith("python")
        args = argv[1:]
        hit = False
        for pat in patterns:
            if pat.endswith(".py"):
                if is_python and any(os.path.basename(a) == pat for a in args):
                    hit = True
                    break
            elif is_python and any(pat in a for a in args):
                hit = True
                break
        if hit:
            out.append(pid)
    return out


def container_jobs(name: str, proc_root: Path = Path("/proc")) -> list[int]:
    """PIDs of the `docker`/`podman run` CLIENT for container `name` (DEC-450).

    A container is outside the project slice BY DESIGN — it has its own cgroup, which is
    the whole point — so this is a FLOOR target only, never a slice stray. The engine's
    child processes are invisible here (not python), so matching the client is what lets
    the guard stop a containerized GPU job when the host floor is breached. Only an actual
    container CLI matches: a shell whose command text mentions the name must not.
    """
    me = os.getpid()
    out: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmd = (proc_root / entry.name / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        argv = [a for a in cmd.split("\0") if a]
        if not argv or os.path.basename(argv[0]) not in ("docker", "podman"):
            continue
        args = argv[1:]
        if "run" not in args:
            continue
        for i, arg in enumerate(args):
            if (arg == "--name" and i + 1 < len(args) and args[i + 1] == name) or (
                arg == f"--name={name}"
            ):
                out.append(pid)
                break
    return out


def in_slice(pid: int, slice_name: str, proc_root: Path = Path("/proc")) -> bool:
    """Is this process inside the project's memory slice? (DEC-440)

    `/proc/<pid>/cgroup` names every cgroup the process belongs to, so membership is read from the
    kernel rather than inferred from how the job was launched. A job started some other way is
    outside the boundary by definition, however well-intentioned the launch was.
    """
    try:
        cg = (proc_root / str(pid) / "cgroup").read_text()
    except OSError:
        return False  # vanished or unreadable: treat as outside, the caller may only warn
    return slice_name in cg


def process_age_s(pid: int, proc_root: Path = Path("/proc")) -> float | None:
    """Seconds since this process started, from the kernel (field 22 of /proc/<pid>/stat)."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
        # `comm` can contain spaces and parentheses; everything after the LAST ')' is positional
        fields = stat.rsplit(")", 1)[1].split()
        start_ticks = float(fields[19])
        hz = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
    except (OSError, IndexError, ValueError):
        return None
    return max(0.0, uptime - start_ticks / hz)


def should_stop_outside(age_s: float | None, grace_s: float) -> bool:
    """A producer is allowed a moment to re-enter the slice before it is called a stray.

    Measured: a direct launch calls `safe_run.ensure_boundary()`, which re-execs into the slice in
    well under a second — but the guard polls, and on its first sighting the process is outside.
    Killing it there punished the re-entry it exists to encourage (it killed the top-level launcher
    of the S6 chain one second before the job moved into the slice). So "outside" means outside for
    longer than a startup grace, not outside for an instant. An unreadable age is treated as old:
    a process we cannot age is not given the benefit of the doubt.
    """
    return age_s is None or age_s >= grace_s


def outside_slice(jobs: list[int], slice_name: str, proc_root: Path = Path("/proc")) -> list[int]:
    return [pid for pid in jobs if not in_slice(pid, slice_name, proc_root)]


def _top_rss(n: int = 6) -> list[str]:
    """Top RSS processes by (rss_gb, pid, comm) — the LOW-MEM witness line's source.

    /proc only, stdlib only: the guard must never fail to explain a stop because a
    helper (ps, psutil) is missing on the host it polices.
    """
    rows = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as fh:
                rss_kb = comm = 0
                for ln in fh:
                    if ln.startswith("VmRSS:"):
                        rss_kb = int(ln.split()[1])
                    elif ln.startswith("Name:"):
                        comm = ln.split()[1]
                    if rss_kb and comm:
                        break
            if rss_kb > 2**20:  # > 1 GB only: the witnesses that matter at an 8 GB floor
                rows.append((rss_kb / 2**20, int(d), comm))
        except OSError:
            continue  # process vanished / unreadable mid-scan — a witness list may be partial
    rows.sort(reverse=True)
    return [f"{g:.2f} GB pid={p} {c}" for g, p, c in rows[:n]]


def _cgroup_witness(root: str = "/sys/fs/cgroup") -> list[str]:
    """Container/slice memory-cgroup usage at breach time (DEC-460).

    Process RSS alone could not explain tonight's breach (the 85 GiB LLM GPU
    reservation lives OUTSIDE every memcg — cgroups summed 13 GiB while the host
    showed 96 used). Per-scope current/peak/max separates 'the LLM container grew
    past its cap' from 'the trainer ramped' from 'something else moved' — and the
    max column shows whether the cap the operator ruled is actually in force.
    A witness list may be partial (scopes vanish mid-scan); it must never fail.
    """
    out = []
    base = Path(root) / "system.slice"
    try:
        scopes = sorted(p for p in base.iterdir() if p.name.startswith(("docker-", "wonderland")))
    except OSError:
        return ["(system.slice unreadable)"]
    for sc in scopes:
        try:
            cur = int((sc / "memory.current").read_text())
            pf = sc / "memory.peak"
            peak = int(pf.read_text()) if pf.exists() else cur
            mx = (sc / "memory.max").read_text().strip()
        except OSError:
            continue
        out.append(f"{sc.name[:48]} cur={cur / 2**30:.2f}GiB peak={peak / 2**30:.2f}GiB max={mx}")
    return out


def _kill(pid: int, dry_run: bool) -> str:
    if dry_run:
        return f"would SIGTERM {pid}"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return f"{pid} already gone"
    for _ in range(20):  # 5 s grace, then force (a hung CUDA context ignores SIGTERM)
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return f"SIGTERM {pid}"
    try:
        os.kill(pid, signal.SIGKILL)
        return f"SIGKILL {pid}"
    except ProcessLookupError:
        return f"SIGTERM {pid}"


def arm_parent_death_signal() -> bool:
    """Ask the kernel to SIGTERM this guard when its parent exits (Linux `prctl`).

    Measured failure: guards launched through a shell wrapper (`cd x && rm y && python
    watchdog.py &`) OUTLIVED the wrapper — `kill $!` signals the wrapper, not the guard —
    and five stale guards went on killing unrelated GPU jobs while writing their verdicts
    to log files nobody was reading. A guard that outlives its job is a hazard, not a
    safety net, so the kernel is asked to end it when the launcher goes.
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1  # noqa: F841 - the prctl option number
        if int(libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)) != 0:
            return False
    except Exception:  # non-Linux, no ctypes: still policed, just not self-ending
        return False
    return os.getppid() != 1  # False if we were ALREADY detached (parent gone at exec)


def _write_pidfile(path: Path) -> None:
    """Exact stoppability: `kill $(cat pidfile)` targets the guard, never its wrapper."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{os.getpid()}\n")
    except OSError as exc:
        print(f"{time.strftime('%T')} WATCHDOG-WARN pidfile {path} unwritable: {exc}", flush=True)


def main(argv=None, *, arm: bool = True) -> int:
    """`arm=False` skips the parent-death signal: a test must not put a SIGTERM contract
    on the pytest process itself. The CLI entry point always arms."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--floor-gb", type=float, default=8.0, help="never let MemAvailable drop below this"
    )
    ap.add_argument("--interval", type=float, default=5.0, help="poll seconds")
    ap.add_argument("--once", action="store_true", help="sample once, report, exit")
    ap.add_argument("--dry-run", action="store_true", help="log what would be killed")
    ap.add_argument(
        "--pattern", action="append", default=None, help="cmdline substring to police (repeatable)"
    )
    ap.add_argument(
        "--container",
        default=None,
        help="also police the `docker/podman run` client of this container NAME: a containerized "
        "job is outside the slice by design, so this is a floor target only (DEC-450)",
    )
    ap.add_argument(
        "--slice",
        default="wonderland.slice",
        help="the project's memory slice; jobs outside it are not bounded by the ceiling",
    )
    ap.add_argument(
        "--slice-grace-s",
        type=float,
        default=30.0,
        help="how long a producer may be outside the slice before it is stopped: a re-entry takes "
        "well under a second, and killing on first sight punishes the thing this encourages",
    )
    ap.add_argument(
        "--outside-slice",
        choices=("kill", "warn", "ignore"),
        default="kill",
        help="what to do with a policed job that is running OUTSIDE the slice (default kill: an "
        "unbounded producer is the hazard this guard exists for)",
    )
    ap.add_argument(
        "--report-delta-gb",
        type=float,
        default=0.5,
        help="log a sample only when MemAvailable moves this much (default 0.5, material "
        "next to an 8 GB floor; keeps a 40-min run's log readable)",
    )
    ap.add_argument(
        "--pidfile",
        default=None,
        help="write this guard's pid here so the exact process can be stopped "
        "(`kill $(cat pidfile)`) rather than a shell wrapper",
    )
    a = ap.parse_args(argv)
    patterns = tuple(a.pattern) if a.pattern else DEFAULT_PATTERNS
    if a.interval <= 0:
        raise SystemExit(f"--interval must be > 0 seconds, got {a.interval}")
    if a.report_delta_gb <= 0:
        raise SystemExit(f"--report-delta-gb must be > 0, got {a.report_delta_gb}")
    self_ending = arm_parent_death_signal() if arm else False
    pidfile = Path(a.pidfile) if a.pidfile else None
    if pidfile is not None:
        _write_pidfile(pidfile)

    def _drop_pidfile():
        if pidfile is not None:
            try:
                pidfile.unlink()
            except OSError:
                pass

    # Announce itself: a silent guard is indistinguishable from a dead one, and the first
    # version of this script died at startup while a trainer was allocating under it.
    print(
        f"{time.strftime('%T')} WATCHDOG-UP floor={a.floor_gb} GB interval={a.interval}s "
        f"policing={len(patterns)} pattern(s) slice={a.slice}({a.outside_slice}) "
        f"avail={mem_available_gb():.1f} GB "
        f"pid={os.getpid()} self_ending={self_ending}"
        f"{' [dry-run]' if a.dry_run else ''}",
        flush=True,
    )

    killed = False
    low_water = float("nan")  # tracked ONLY while a job is resident: ambient idle memory is
    last_report = float("nan")  # not the job's doing; what matters is how close THIS job got

    def _farewell(signum, _frame):
        # The receipt has to survive the guard's own death: callers stop this process with
        # SIGTERM, and the low-water mark is the number the post-mortem needs.
        print(
            f"{time.strftime('%T')} WATCHDOG-STOPPED by signal {signum} "
            f"low={low_water:.1f} GB floor={a.floor_gb} GB killed={killed}",
            flush=True,
        )
        _drop_pidfile()
        raise SystemExit(2 if killed else 0)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _farewell)

    while True:
        # A guard that dies stops guarding. Any sampling defect is REPORTED and retried;
        # it must never end the watch (the first version of this script died on a Path
        # bug while the trainer it was meant to police was allocating).
        try:
            avail = mem_available_gb()
            jobs = gpu_jobs(patterns)
            cjobs = container_jobs(a.container) if a.container else []
        except Exception as exc:  # noqa: BLE001 - fail safe, loudly
            print(f"{time.strftime('%T')} WATCHDOG-FAULT {type(exc).__name__}: {exc}", flush=True)
            avail, jobs, cjobs = float("inf"), [], []
        # The floor covers everything this guard knows how to stop: matched producers AND
        # the container client. Slice policing applies only to the producers — a container
        # is outside the slice on purpose.
        targets = jobs + cjobs
        if jobs and a.outside_slice != "ignore":
            strays = [
                pid
                for pid in outside_slice(jobs, a.slice)
                if should_stop_outside(process_age_s(pid), a.slice_grace_s)
            ]
            for pid in strays:
                if a.outside_slice == "kill":
                    print(
                        f"{time.strftime('%T')} OUTSIDE-SLICE {pid} is not in {a.slice} -> "
                        f"stopping it ({_kill(pid, a.dry_run)})",
                        flush=True,
                    )
                    killed = killed or not a.dry_run
                else:
                    print(
                        f"{time.strftime('%T')} OUTSIDE-SLICE {pid} is not in {a.slice} "
                        f"(warn only)",
                        flush=True,
                    )
            jobs = [pid for pid in jobs if pid not in strays] if a.outside_slice == "kill" else jobs
        if targets:
            low_water = avail if math.isnan(low_water) else min(low_water, avail)
        if avail < a.floor_gb and targets:
            # Attribution before action: when the floor is breached, the guard must say WHO
            # holds the memory (trainer vs Qwen vs an external burst) — a bare "avail dipped"
            # line leaves the retry decision blind (measured 2026-10-30: an external 12 GB
            # burst in 6 s sacrificed the trainer before its first step, indistinguishable
            # from a trainer ramp in the old receipt).
            for line in _top_rss():
                print(f"{time.strftime('%T')} LOWMEM-WITNESS {line}", flush=True)
            for line in _cgroup_witness():
                print(f"{time.strftime('%T')} CGROUP-WITNESS {line}", flush=True)
            for pid in targets:
                # Log the DECISION before acting: `_kill` blocks for up to 5 s waiting for
                # the process to die, and a guard killed inside that window would leave no
                # record of the kill it just made (measured: a capped trainer was correctly
                # stopped at the floor and the receipt line never reached the log).
                print(
                    f"{time.strftime('%T')} LOW-MEM {avail:.1f} GB < {a.floor_gb} GB -> "
                    f"stopping GPU job {pid} (low-water so far {low_water:.1f} GB)",
                    flush=True,
                )
                print(f"{time.strftime('%T')} KILLED {pid}: {_kill(pid, a.dry_run)}", flush=True)
            killed = True
        elif targets and (math.isnan(last_report) or abs(avail - last_report) >= a.report_delta_gb):
            # Sample every `interval`, but only WRITE when memory moves materially: a 40-min
            # run at 5 s polls is 480 identical lines, and a wall of noise hides the ramp
            # that matters. `--once` always reports.
            last_report = avail
            print(
                f"{time.strftime('%T')} avail={avail:.1f} GB floor={a.floor_gb} "
                f"low={low_water:.1f} GPU jobs={len(jobs)}"
                + (f" container={len(cjobs)}" if a.container else ""),
                flush=True,
            )
        elif a.once:
            print(
                f"{time.strftime('%T')} avail={avail:.1f} GB floor={a.floor_gb} "
                f"low={low_water:.1f} GPU jobs={len(jobs)}"
                + (f" container={len(cjobs)}" if a.container else ""),
                flush=True,
            )
        if a.once:
            break
        time.sleep(a.interval)
    print(
        f"{time.strftime('%T')} WATCHDOG-DOWN low={low_water:.1f} GB floor={a.floor_gb} GB "
        f"killed={killed}",
        flush=True,
    )
    _drop_pidfile()
    return 2 if killed else 0


if __name__ == "__main__":
    raise SystemExit(main())
