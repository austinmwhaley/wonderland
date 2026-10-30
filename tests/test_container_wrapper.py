"""The one entry point (DEC-450): every job runs in the 20 GB container, GPU included.

`--plan` prints the exact `docker run` argv without touching Docker, so the invariants can be
graded without starting a container: the hard ceiling is `--memory` == `--memory-swap` (no swap
escape), GPU is opt-in rather than ambient, and compute is never throttled — only memory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wonder_container.sh"


def _plan(*args, env=None):
    e = dict(os.environ)
    e.pop("WONDER_MEM", None)
    e.pop("WONDER_EXTRA_ARGS", None)
    e.update(env or {})
    out = subprocess.run(
        ["bash", str(SCRIPT), "--plan", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=e,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def test_the_ceiling_is_hard_and_equal_on_swap():
    argv = _plan("--", "true")
    assert "--memory=20g" in argv
    assert "--memory-swap=20g" in argv  # equal => no swap: exceed = kill, never host thrash


def test_compute_is_not_throttled_only_memory_is():
    argv = _plan("--", "true")
    assert not any(a.startswith("--cpus") for a in argv)
    assert "--cpu-shares" not in argv
    assert not any(a.startswith("--gpus-rate") for a in argv)


def test_gpu_is_explicit_and_carries_the_cdi_request():
    assert "--gpus" not in _plan("--", "true")
    with_gpu = _plan("--gpu", "--", "true")
    i = with_gpu.index("--gpus")
    assert with_gpu[i + 1] == "all"


def test_the_out_dir_is_writable_and_inputs_stay_read_only():
    argv = _plan("--", "true")
    assert f"{ROOT}:/repo" in argv
    assert f"{os.environ.get('HOME')}/wonderland_work:/work:ro" in argv
    with_out = _plan("--out", "/tmp/run42:/out", "--", "true")
    assert "/tmp/run42:/out" in with_out


def test_mem_override_reaches_the_kernel_flag():
    argv = _plan("--", "true", env={"WONDER_MEM": "24g"})
    assert "--memory=24g" in argv and "--memory-swap=24g" in argv


def test_an_empty_cuda_visible_devices_is_omitted_not_passed():
    """An empty CUDA_VISIBLE_DEVICES means 'no GPU' to torch; passing it made the GPU path a
    silent CPU run (measured while wiring --gpu)."""
    argv = _plan("--", "true", env={"CUDA_VISIBLE_DEVICES": ""})
    assert not any(a.startswith("CUDA_VISIBLE_DEVICES=") for a in argv)
