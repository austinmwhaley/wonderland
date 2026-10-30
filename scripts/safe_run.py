#!/usr/bin/env python3
"""Run a job under a HARD memory cap the kernel enforces — so a bad job cannot take the box down.

    python3 scripts/safe_run.py -- <command> [args...]
    python3 scripts/safe_run.py --print-plan
    python3 scripts/safe_run.py --max-gb 24 -- .venv/bin/python scripts/measure_battery_law.py ...

Why this exists. On 2026-10-29 an unbounded DuckDB aggregation (mine) exhausted system memory on
the box that also hosts the LLM server and the machine went down. A watchdog that *observes*
memory and kills at a floor is a good second line, but it is a race: the way to make a repeat
impossible is to put the job in a cgroup with `MemoryMax` set, where the kernel kills the JOB
before it can starve anything else, and the rest of the machine never notices. `MemorySwapMax=0`
keeps the cap meaningful instead of letting the job thrash into swap.

The budget is DERIVED, never typed: the operator's policy is that the LLM server keeps what it
needs, the system keeps an 8 GB floor, and the job gets the remainder. So

    cap = min(--max-gb if given, MemAvailable - floor)

is computed at launch from the machine's own state. If that leaves less than the minimum viable
headroom, the launcher REFUSES to start the job (fail safe, principle 7) instead of starting it at
a cap that guarantees a kill mid-measurement.

This is a launcher, not a gate: it records what it did and returns the job's exit code unchanged.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: principle-ok: below this, a "capped" run is a run that will die partway through and leave a
#: half-written receipt, which is worse than refusing. It is a viability floor, not a tuning knob.
MIN_CAP_GB = 1.0

#: principle-ok: system floor from the operator's policy (LLM server keeps what it needs, the
#: system keeps 8 GB, the job gets the remainder). The default; `--floor-gb` overrides per run.
DEFAULT_FLOOR_GB = 8.0

#: The project's boundary (DEC-440): a slice with ONE memory ceiling that every job shares, so
#: the guarantee is a property of the project rather than of each launch being disciplined. It is
#: a systemd slice and not a container because the venv, the data and the GPU device live on this
#: host; the enforcement is the same kernel cgroup Docker's --memory uses.
DEFAULT_SLICE = "wonderland.slice"


def mem_available_gb(path: str = "/proc/meminfo") -> float:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0  # unreadable -> the launcher refuses rather than guessing a budget


def plan(available_gb: float, floor_gb: float = DEFAULT_FLOOR_GB, max_gb: float | None = None):
    """The derived budget: (cap_gb, ok, reason). Pure, so the arithmetic is testable.

    `ok=False` is a refusal, not an error to paper over: a job launched at a cap it cannot fit in
    dies mid-write, and a half-written receipt is worse than no receipt.
    """
    spare = available_gb - floor_gb
    if max_gb is not None:
        spare = min(spare, max_gb)
    if spare < MIN_CAP_GB:
        return (
            max(0.0, spare),
            False,
            f"only {available_gb:.1f} GB available; after the {floor_gb:.1f} GB system floor "
            f"{max(0.0, spare):.1f} GB would be left, below the {MIN_CAP_GB:.1f} GB viability floor",
        )
    return spare, True, f"MemAvailable {available_gb:.1f} GB - floor {floor_gb:.1f} GB"


def slice_cap_gb(name: str = DEFAULT_SLICE) -> float | None:
    """The slice's `MemoryMax` in GB, or None when the slice does not exist / is unlimited.

    This is the number the project is allowed to use, full stop. Reading it from the unit (rather
    than repeating it in code) keeps one definition: the ceiling is set in one place and every
    launcher and receipt quotes what the kernel is actually enforcing.
    """
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", name, "-p", "MemoryMax", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    val = r.stdout.strip()
    if r.returncode != 0 or not val.isdigit():
        return None  # "infinity" is not a cap
    gb = int(val) / 2**30
    return gb if gb > 0 else None


def cgroup_leaf_cap_gb(path: str = "/sys/fs/cgroup/memory.max") -> float | None:
    """The memory ceiling written on THIS process's cgroup, if it has one.

    Inside a container that is `docker --memory`; inside a systemd scope it is the scope's limit.
    Reading it is how a job reports the number the kernel will actually enforce instead of the
    number it computed before entering.
    """
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    if not raw.isdigit():
        return None  # "max" means unlimited
    gb = int(raw) / 2**30
    return gb if gb > 0 else None


def in_container() -> bool:
    """Docker/Podman leave /.dockerenv (or a container= env) behind."""
    return Path("/.dockerenv").exists() or os.environ.get("container") is not None


def resolve_cap(
    slice_name: str | None = DEFAULT_SLICE,
    floor_gb: float = DEFAULT_FLOOR_GB,
    max_gb: float | None = None,
) -> tuple[float, bool, str]:
    """The budget to report and enforce: the SMALLEST ceiling that applies, and why.

    Order of candidacy: the project slice's `MemoryMax`, this process's own cgroup limit (a
    container's `--memory`, or the scope it is already in), and — only when neither exists — the
    budget derived from `MemAvailable - floor`. The smallest wins, because every one of them is a
    real kernel limit and the effective one is the tightest. Reporting a computed number that is
    larger than the ceiling actually in force is how a receipt ends up claiming 108 GB for a job
    that could never use more than 60.
    """
    sliced = slice_cap_gb(slice_name) if slice_name else None
    leaf = cgroup_leaf_cap_gb()
    derived, ok_derived, why_derived = plan(mem_available_gb(), floor_gb, max_gb)
    cands = [(v, f"slice {slice_name} MemoryMax") for v in [sliced] if v]
    cands += [(v, "this process's cgroup limit") for v in [leaf] if v]
    if not cands:
        return derived, ok_derived, why_derived
    cap = min(v for v, _ in cands)
    if max_gb is not None:
        cap = min(cap, max_gb)
    ok = cap >= MIN_CAP_GB
    why = f"{min(cands)[1]} of {cap:.1f} GB is the tightest ceiling in force"
    if not ok:
        why += f" — below the {MIN_CAP_GB:.1f} GB viability floor"
    return cap, ok, why


def systemd_scope_available() -> bool:
    """`systemd-run --user --scope` needs a running user manager and cgroup v2 delegation."""
    if not shutil.which("systemd-run"):
        return False
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.stdout.strip() in ("running", "degraded")


def capped_command(
    argv: list[str], cap_gb: float, slice_name: str | None = DEFAULT_SLICE
) -> tuple[list[str], str]:
    """The argv that runs `argv` under a hard cap, plus which mechanism was chosen.

    Preference order, strongest first:
    1. **the project slice** — one ceiling shared by every job, so two modest jobs cannot add up
       to an immodest one;
    2. **a per-job cgroup** — a hard cap, but each job gets its own budget;
    3. **RLIMIT_AS** — per-process address space, the weakest of the three (a forking child gets
       its own limit), used only where there is no user systemd.

    Whichever is used is NAMED in the receipt: a weaker guarantee must never be reported as the
    same one.
    """
    gb = f"{cap_gb:.1f}G"
    if in_container():
        # Already bounded by the container's cgroup, which every child inherits. A per-job limit on
        # top would be a second, smaller ceiling nobody asked for.
        return argv, "container cgroup (docker --memory)"
    sliced = slice_cap_gb(slice_name) if slice_name else None
    if sliced is not None:
        # The slice is the ceiling for the PROJECT; a tighter `--max-gb` is a ceiling for THIS job
        # and must still apply inside it. Both limits are real cgroup limits, and the effective
        # one is the smaller — which is what the receipt reports.
        tighter = ["-p", f"MemoryMax={gb}"] if cap_gb < sliced - 1e-9 else []
        return (
            [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                f"--slice={slice_name}",
                *tighter,
                "--",
                *argv,
            ],
            f"slice {slice_name} MemoryMax"
            + (f" + job MemoryMax {cap_gb:.1f} GB" if tighter else ""),
        )
    if systemd_scope_available():
        return (
            [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "-p",
                f"MemoryMax={gb}",
                "-p",
                "MemorySwapMax=0",
                "-p",
                "MemoryAccounting=yes",
                "--",
                *argv,
            ],
            "cgroup(v2) MemoryMax",
        )
    return argv, "rlimit(RLIMIT_AS)"


def current_slice(proc_cgroup: str = "/proc/self/cgroup") -> str:
    """The cgroup path this process is in, or "" if it cannot be read."""
    try:
        for line in Path(proc_cgroup).read_text().splitlines():
            if line.startswith("0::"):
                return line[3:]
    except OSError:
        pass
    return ""


#: Set on the re-exec'd child so a host where the slice cannot be entered does not loop forever.
_BOUNDARY_ENV = "WONDERLAND_BOUNDARY"


def ensure_boundary(slice_name: str = DEFAULT_SLICE, cap_gb: float | None = None) -> None:
    """Re-enter the project's memory boundary if this process is outside it, then EXIT.

    `safe_run` only protects the launches that go through it, and the failure this whole file
    exists for was a DIRECT launch. So a producer calls this first: if the kernel says we are not
    in the slice, the process re-launches itself inside it and exits with the child's status. The
    guard is then a property of the script rather than of the operator remembering.

    It is deliberately not silent: entering a 60 GB ceiling changes what a run may do, and a
    receipt that does not say where it ran is not reproducible. It also no-ops rather than failing
    when the slice does not exist (a fresh clone, CI): a boundary that cannot be entered must not
    make the measurement impossible.
    """
    if os.environ.get(_BOUNDARY_ENV) or not slice_name:
        return
    if slice_name in current_slice():
        return
    sliced = slice_cap_gb(slice_name)
    if sliced is None or in_container():
        return
    # Re-entry needs a real script to re-exec. `python -c "..."` has no file to hand back to the
    # interpreter (`sys.argv[0]` is literally "-c"), so there is nothing to relaunch; `python -m
    # mod` and `python scripts/x.py` both expose a path and are covered.
    if not sys.argv or not sys.argv[0] or not os.path.exists(sys.argv[0]):
        print(
            "boundary: cannot re-enter the slice (no script path to re-exec); "
            "launch through scripts/safe_run.py to get the ceiling",
            flush=True,
        )
        return
    cap = min(sliced, cap_gb) if cap_gb is not None else sliced
    # re-exec through the interpreter: `sys.argv[0]` is a script path, not an executable
    run_argv, mechanism = capped_command([sys.executable, *sys.argv], cap, slice_name)
    env = {**os.environ, _BOUNDARY_ENV: "1"}
    print(
        f"boundary: re-entering {slice_name} ({mechanism}) — this process was outside it",
        flush=True,
    )
    sys.exit(subprocess.run(run_argv, env=env).returncode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--floor-gb", type=float, default=DEFAULT_FLOOR_GB)
    ap.add_argument("--max-gb", type=float, default=None, help="optional cap below the derived one")
    ap.add_argument(
        "--slice",
        default=DEFAULT_SLICE,
        help="systemd user slice holding the project's memory ceiling ('' disables)",
    )
    ap.add_argument("--print-plan", action="store_true", help="report the budget and exit")
    ap.add_argument("--quiet", action="store_true", help="suppress the receipt line")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    cap, ok, why = resolve_cap(a.slice or None, a.floor_gb, a.max_gb)
    if not ok:
        why += f" — below the {MIN_CAP_GB:.1f} GB viability floor"
    if a.print_plan or not cmd:
        print(f"safe-run plan: cap={cap:.1f} GB ({why}); ok={ok}")
        if not cmd:
            return 0 if ok else 2
    if not ok:
        print(f"safe-run: REFUSING to start — {why}", file=sys.stderr)
        return 2

    run_argv, mechanism = capped_command(cmd, cap, a.slice or None)
    if not a.quiet:
        print(
            f"safe-run: cap={cap:.1f} GB via {mechanism}; "
            f"root={ROOT.name}; cmd={' '.join(cmd[:2])}{' …' if len(cmd) > 2 else ''}",
            flush=True,
        )

    if mechanism.startswith("rlimit"):
        limit = int(cap * 2**30)
        rc = subprocess.run(
            cmd,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (limit, limit)),
        ).returncode
    else:
        rc = subprocess.run(run_argv).returncode

    if rc in (-9, 137):
        print(
            f"safe-run: the job was KILLED at the {cap:.1f} GB cap — it tried to use more than the "
            f"machine could spare. The box is fine; the job's output is not a measurement.",
            file=sys.stderr,
        )
    # `SystemExit(-9)` would map to 247 in the shell, which reads like a systemd code; normalise
    # to the conventional killed-by-signal status so the caller can test for it directly
    return 137 if rc == -9 else rc


if __name__ == "__main__":
    raise SystemExit(main())
