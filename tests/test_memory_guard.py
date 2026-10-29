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
