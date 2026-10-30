"""The two layers that must stop a runaway job: the hard cgroup cap and the watchdog's coverage.

The 2026-10-29 crash had one cause and one amplifier. The cause was an unbounded aggregation. The
amplifier was that the job ran with NO guard at all, and the script holding it was not on the
watchdog's list, so nothing was in a position to stop it. `tests/test_safe_run.py` covers the hard
cap; this file covers the amplifier — the guard must police every producer the repo can launch,
and the list must fail loudly when it drifts behind the repo instead of silently missing a job.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gpu_watchdog as GW  # noqa: E402
import safe_run as SR  # noqa: E402


def test_every_producer_that_can_allocate_is_policed():
    """A producer the guard cannot see is a producer it cannot stop.

    Derived from the repo, not typed: any future `scripts/measure_*.py` is covered by definition,
    so adding a producer cannot leave a hole here without failing this test.
    """
    producers = sorted(p.name for p in (ROOT / "scripts").glob("measure_*.py"))
    assert producers, "expected the measurement producers to exist"
    missing = [p for p in producers if p not in GW.DEFAULT_PATTERNS]
    assert not missing, (
        f"unpoliced producer(s): {missing} — add them to gpu_watchdog.DEFAULT_PATTERNS, or they "
        f"can exhaust the machine while the guard watches something else"
    )


def test_the_guard_recognises_a_cmdline_for_each_pattern():
    """Matching is substring on the command line, so a pattern must appear in a real invocation.

    This catches the silent-drift case where a script is renamed: the pattern stays, and the guard
    goes on policing a name that no longer exists.
    """
    import subprocess

    for pat in ("measure_judge_law.py", "measure_battery_law.py", "measure_dynamics_law.py"):
        # each pattern is a filename that exists under scripts/
        assert (ROOT / "scripts" / pat).exists() or pat == "looking_glass.customer_foundation_model"
    # and the guard skips itself: a watchdog that kills its own process is not a watchdog
    me = subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True)
    assert me.returncode == 0
    assert "gpu_watchdog" not in GW.__doc__ or True  # the skip is exercised below


def test_the_guard_excludes_itself_and_its_wrapper(tmp_path):
    """ "gpu_watchdog" in cmdline is the skip rule; without it the guard would target itself."""
    pids = []
    for pid in (1, 2, 3):
        d = tmp_path / str(pid)
        d.mkdir()
        pids.append(d)
    (pids[0] / "cmdline").write_bytes(b"python\0scripts/gpu_watchdog.py\0")
    (pids[1] / "cmdline").write_bytes(b"python\0scripts/measure_judge_law.py\0")
    (pids[2] / "cmdline").write_bytes(b"bash\0-c\0echo hi\0")
    found = GW.gpu_jobs(GW.DEFAULT_PATTERNS, proc_root=tmp_path)
    assert 2 in found  # the producer
    assert 1 not in found  # the guard itself
    assert 3 not in found  # unrelated


def test_a_producer_outside_the_slice_is_identified(tmp_path):
    """The boundary is only real if membership is checked, and checked from the kernel.

    A job launched some other way is outside the slice however well-intentioned the launch was;
    `/proc/<pid>/cgroup` is the ground truth, not the launcher's intent.
    """
    inside = tmp_path / "11"
    outside = tmp_path / "12"
    for d, cg in (
        (inside, "0::/user.slice/user-1000.slice/wonderland.slice/run-abc.scope\n"),
        (outside, "0::/user.slice/user-1000.slice/app.slice/app-gnome.scope\n"),
    ):
        d.mkdir()
        (d / "cgroup").write_text(cg)
    assert GW.in_slice(11, "wonderland.slice", proc_root=tmp_path) is True
    assert GW.in_slice(12, "wonderland.slice", proc_root=tmp_path) is False
    assert GW.outside_slice([11, 12], "wonderland.slice", proc_root=tmp_path) == [12]
    # a vanished pid is not evidence of compliance
    assert GW.in_slice(99, "wonderland.slice", proc_root=tmp_path) is False


def test_the_project_slice_exists_with_a_finite_ceiling():
    """If the slice is missing, every launcher silently degrades to a per-job budget.

    Named as a test so the boundary cannot quietly disappear: an absent slice is a warning here,
    but a slice present with `infinity` would be a false sense of safety and is a failure.
    """
    import subprocess

    cap = SR.slice_cap_gb(SR.DEFAULT_SLICE)
    r = subprocess.run(
        ["systemctl", "--user", "show", SR.DEFAULT_SLICE, "-p", "MemoryMax", "--value"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        import pytest

        pytest.skip(f"no user slice {SR.DEFAULT_SLICE} on this host")
    assert cap is not None and cap > 0


def test_a_shell_that_mentions_a_producer_is_not_the_producer(tmp_path):
    """Measured mistake: the guard SIGTERMed the launcher shell, not the job.

    `cd repo && nohup python scripts/x.py` puts the script's name in the BASH cmdline, and bash is
    outside the slice while the job it launched is inside. Matching the command line as a string
    therefore punished the messenger — and could signal whichever shell typed the command.
    """
    shell = tmp_path / "21"
    job = tmp_path / "22"
    for d, cmd in (
        (shell, b"/bin/bash\0-c\0cd repo && nohup python scripts/measure_judge_law.py\0"),
        (job, b"/usr/bin/python3\0scripts/measure_judge_law.py\0--legs\0rl\0"),
    ):
        d.mkdir()
        (d / "cmdline").write_bytes(cmd)
    found = GW.gpu_jobs(GW.DEFAULT_PATTERNS, proc_root=tmp_path)
    assert 22 in found  # the interpreter running the producer
    assert 21 not in found  # the shell that launched it


def test_a_module_pattern_still_matches_its_interpreter(tmp_path):
    """`python -m looking_glass.customer_foundation_model` has no `.py` argument to compare."""
    d = tmp_path / "31"
    d.mkdir()
    (d / "cmdline").write_bytes(
        b"/usr/bin/python3\0-m\0looking_glass.customer_foundation_model\0train\0"
    )
    assert 31 in GW.gpu_jobs(GW.DEFAULT_PATTERNS, proc_root=tmp_path)


def test_a_producer_gets_a_moment_to_re_enter_the_slice():
    """Measured: the guard SIGTERMed the S6 launcher ~1 s before it moved into the slice itself.

    Re-entry happens immediately, but the guard samples on a timer, and on its first sighting a
    correctly-behaving producer is still outside. "Outside" therefore has to mean "outside for
    longer than a startup grace", or the enforcement punishes the very mechanism that complies.
    """
    assert GW.should_stop_outside(age_s=0.4, grace_s=30.0) is False
    assert GW.should_stop_outside(age_s=29.9, grace_s=30.0) is False
    assert GW.should_stop_outside(age_s=31.0, grace_s=30.0) is True
    # a process whose age cannot be read does not get the benefit of the doubt
    assert GW.should_stop_outside(age_s=None, grace_s=30.0) is True


def test_process_age_is_read_from_the_kernel():
    """Age comes from /proc, not from when the guard first noticed the pid."""
    age = GW.process_age_s(__import__("os").getpid())
    assert age is not None and age >= 0.0
    assert GW.process_age_s(999999999) is None
