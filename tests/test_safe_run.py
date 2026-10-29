"""The memory cap has to actually stop a runaway job, or it is decoration.

Written after an unbounded aggregation took this machine down while the LLM server was resident.
The launcher's whole claim is that the KERNEL ends the job and the rest of the box does not
notice, so the tests here are: the arithmetic derives a budget from the machine's own state, a job
that exceeds its cap is killed and nothing else is, and the launcher refuses to start a job it
cannot fit rather than starting a doomed one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import safe_run as SR  # noqa: E402


# ---------------------------------------------------------------- the derived budget


def test_the_cap_is_what_is_left_after_the_system_floor():
    """The operator's policy, as arithmetic: LLM server first, 8 GB floor, job gets the rest."""
    cap, ok, why = SR.plan(available_gb=40.0, floor_gb=8.0)
    assert ok and cap == pytest.approx(32.0)
    assert "MemAvailable 40.0 GB - floor 8.0 GB" in why


def test_an_explicit_max_never_exceeds_the_derived_budget():
    """`--max-gb` may lower the cap; it may never promise memory the machine does not have."""
    assert SR.plan(available_gb=40.0, floor_gb=8.0, max_gb=12.0)[0] == pytest.approx(12.0)
    # asking for more than is free silently grants the derived cap, not the request
    assert SR.plan(available_gb=40.0, floor_gb=8.0, max_gb=999.0)[0] == pytest.approx(32.0)


def test_a_job_that_cannot_fit_is_refused_not_started():
    """Starting a job at a cap it cannot meet leaves a half-written receipt — worse than nothing."""
    cap, ok, why = SR.plan(available_gb=8.5, floor_gb=8.0)
    assert ok is False and cap < SR.MIN_CAP_GB
    assert "REFUSING" not in why  # `why` is the diagnosis; the refusal wording is the CLI's
    assert "below the" in why


def test_unreadable_memory_is_a_refusal_not_a_guess():
    """Fail safe: if the machine's free memory cannot be read, do not invent a budget."""
    assert SR.mem_available_gb("/nonexistent/meminfo") == 0.0
    assert SR.plan(SR.mem_available_gb("/nonexistent/meminfo"), 8.0)[1] is False


def test_the_launcher_refuses_and_reports_a_nonzero_status(tmp_path):
    """End to end through main(): a refusal must be visible to whatever launched it."""
    rc = SR.main(
        [
            "--slice",
            "",
            "--floor-gb",
            "100000",
            "--",
            sys.executable,
            "-c",
            "print('should not run')",
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------- the hard kill


def _systemd_ready() -> bool:
    return SR.systemd_scope_available()


@pytest.mark.skipif(not _systemd_ready(), reason="no user systemd/cgroup v2 delegation here")
def test_a_job_over_its_cap_is_killed_by_the_kernel_and_the_box_survives():
    """The regression test for the crash: 2 GB requested inside a 1 GB cgroup.

    If this ever allocates successfully, the cap is not real and the machine is one bad job away
    from going down again.
    """
    job = (
        "x = bytearray(2 * 1024**3)\n"
        "x[::4096] = b'x' * len(x[::4096])\n"  # touch the pages so it is real memory, not a reservation
        "print('allocated', len(x))\n"
    )
    rc = SR.main(["--max-gb", "1", "--quiet", "--", sys.executable, "-c", job])
    assert rc == 137, f"expected the cgroup kill (137), got {rc}"


@pytest.mark.skipif(not _systemd_ready(), reason="no user systemd/cgroup v2 delegation here")
def test_the_capped_run_still_returns_a_normal_job_its_own_status():
    """A cap is not a blanket failure: a small job finishes, unchanged."""
    rc = SR.main(["--max-gb", "1", "--quiet", "--", sys.executable, "-c", "print('fine')"])
    assert rc == 0


def test_the_fallback_mechanism_is_named_when_the_cgroup_is_unavailable(monkeypatch):
    """A weaker guarantee must be LABELLED as weaker, never reported as the same mechanism."""
    monkeypatch.setattr(SR, "systemd_scope_available", lambda: False)
    monkeypatch.setattr(SR, "slice_cap_gb", lambda name: None)
    argv, mechanism = SR.capped_command(
        ["python3", "-c", "pass"], 2.0, slice_name="wonderland.slice"
    )
    assert argv == ["python3", "-c", "pass"]
    assert mechanism.startswith("rlimit")


def test_the_cgroup_mechanism_caps_the_whole_tree_not_one_process(monkeypatch):
    """The reason cgroups are preferred: children inherit the limit."""
    monkeypatch.setattr(SR, "systemd_scope_available", lambda: True)
    monkeypatch.setattr(SR, "slice_cap_gb", lambda name: None)  # no project slice on this host
    argv, mechanism = SR.capped_command(["python3", "-c", "pass"], 2.0, slice_name=None)
    assert mechanism.startswith("cgroup")
    assert "--scope" in argv and any(a.startswith("MemoryMax=") for a in argv)
    # MemorySwapMax=0 keeps the cap meaningful instead of letting the job thrash into swap
    assert "MemorySwapMax=0" in argv


# ---------------------------------------------------------------- the project slice


def test_the_slice_is_the_ceiling_and_a_tighter_job_cap_still_applies(monkeypatch):
    """Two limits, both real: the project's slice, and a tighter one for this job.

    A `--max-gb` that only replaced the slice would let one job claim the project's entire budget;
    a `--max-gb` that was ignored would let a small job escape the discipline. Both are emitted,
    and the effective ceiling is the smaller.
    """
    monkeypatch.setattr(SR, "slice_cap_gb", lambda name: 60.0)
    argv, mechanism = SR.capped_command(
        ["python3", "-c", "pass"], 4.0, slice_name="wonderland.slice"
    )
    assert any(a == "--slice=wonderland.slice" for a in argv)
    assert "MemoryMax=4.0G" in argv
    assert "tightened" not in mechanism and "4.0 GB" in mechanism

    # a cap at or above the slice's own needs no second limit: the slice already enforces it
    argv2, mechanism2 = SR.capped_command(["x"], 60.0, slice_name="wonderland.slice")
    assert not any(a.startswith("MemoryMax=") for a in argv2)
    assert "slice wonderland.slice" in mechanism2


def test_without_a_slice_the_budget_is_derived_from_the_machine():
    """The fallback must stay honest: no slice means a per-job number derived at launch."""
    cap, ok, why = SR.plan(available_gb=40.0, floor_gb=8.0)
    assert ok and cap == pytest.approx(32.0) and "MemAvailable" in why


# ---------------------------------------------------------------- re-entry from a direct launch


def test_current_slice_reads_the_kernel_not_the_intent():
    """/proc/self/cgroup is the ground truth; a launcher's belief about where it put a job is not."""
    assert SR.current_slice("/proc/self/cgroup") == SR.current_slice()
    assert SR.current_slice("/nonexistent") == ""


def test_ensure_boundary_is_a_no_op_when_already_inside(monkeypatch):
    """No re-exec loop, no second launch: a job already in the slice just runs."""
    monkeypatch.setattr(
        SR, "current_slice", lambda *a, **k: "/user.slice/wonderland.slice/run-x.scope"
    )
    monkeypatch.setattr(SR, "slice_cap_gb", lambda name: 60.0)
    SR.ensure_boundary()  # returns normally, does not exit


def test_ensure_boundary_is_a_no_op_when_the_slice_does_not_exist(monkeypatch):
    """A fresh clone or CI has no unit installed; a boundary that cannot be entered must not make
    the measurement impossible."""
    monkeypatch.setattr(SR, "current_slice", lambda *a, **k: "/user.slice/app.slice/x.scope")
    monkeypatch.setattr(SR, "slice_cap_gb", lambda name: None)
    SR.ensure_boundary()


def test_ensure_boundary_does_not_recurse_when_re_entry_cannot_take_effect(monkeypatch):
    """The env marker is the loop breaker: a host where entering the slice silently fails must not
    fork a new interpreter forever."""
    monkeypatch.setattr(SR, "current_slice", lambda *a, **k: "/user.slice/app.slice/x.scope")
    monkeypatch.setattr(SR, "slice_cap_gb", lambda name: 60.0)
    monkeypatch.setenv(SR._BOUNDARY_ENV, "1")
    SR.ensure_boundary()  # returns instead of re-launching


@pytest.mark.skipif(not _systemd_ready(), reason="no user systemd/cgroup v2 delegation here")
@pytest.mark.skipif(SR.slice_cap_gb() is None, reason="no wonderland.slice on this host")
def test_a_direct_launch_re_enters_the_slice_by_itself():
    """The regression test for the crash's amplifier: the producer must bound ITSELF.

    The failure was a direct `python scripts/...` with no launcher and no guard. This runs exactly
    that shape and asserts the process ended up inside the project's ceiling.
    """
    script = ROOT / "scripts" / "_boundary_probe.py"
    script.write_text(
        "import sys\n"
        "sys.path.insert(0, 'scripts')\n"
        "import safe_run\n"
        "safe_run.ensure_boundary()\n"
        "print(safe_run.current_slice())\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=120
        )
    finally:
        script.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr
    assert SR.DEFAULT_SLICE in r.stdout
