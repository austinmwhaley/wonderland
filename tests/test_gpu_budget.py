"""DEC-423: the co-tenant GPU budget must be derived, capped, and refuse when starved.

The failure being defended against is not a slow run — it is a driver
`NV_ERR_NO_MEMORY` that reboots the machine while an LLM server shares the unified pool.
So these tests assert the POLICY, off-device: the budget is the remainder after the
co-tenant, the system floor and the device overhead; an explicit override wins; a starved
pool is REFUSED rather than attempted; and a cap that cannot be installed stops the run
instead of quietly training uncapped (the defect the first version shipped: a `ValueError`
from the allocator API was swallowed, the trainer took 21.4 GB, and the watchdog had to
kill it at 7.7 GB against an 8.0 GB floor).
"""

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gpu_watchdog  # noqa: E402
import torch  # noqa: E402

from gpu_watchdog import DEFAULT_PATTERNS, gpu_jobs, mem_available_gb  # noqa: E402

from looking_glass import gpu_budget  # noqa: E402
from looking_glass.cfm_config import CFMConfig, apply_set_overrides  # noqa: E402

_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _no_leaked_guard_state():
    """`_APPLIED` and the device fraction are PROCESS-wide; a test that caps must not make
    a later `enforce()` short-circuit (its whole point is idempotence) nor leave the
    pytest session running under a cap."""
    gpu_budget._APPLIED.clear()
    before = torch.cuda.get_per_process_memory_fraction() if torch.cuda.is_available() else None
    yield
    gpu_budget._APPLIED.clear()
    if before is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(before)
        except Exception as exc:  # a cap set before any allocation is not always liftable
            print(f"[test] could not restore the memory fraction to {before}: {exc}")


def _budget(**kw):
    base = dict(
        mem_available_gb=28.0,
        driver_free_gb=30.0,
        total_gb=121.6,
        system_floor_gb=8.0,
        reserve_gb=1.0,
        limit_gb=0.0,
    )
    base.update(kw)
    return gpu_budget.compute_budget(**base)


# ---------------------------------------------------------------- the arithmetic


def test_budget_is_the_remainder_after_floor_and_overhead_not_the_whole_pool():
    r = _budget()
    assert r["source"] == "derived"
    assert r["budget_gb"] == 19.0  # 28 available - 8 GB floor - 1 GB device overhead
    assert r["applied"] is False  # arithmetic only; enforce() is what caps


def test_tighter_of_kernel_and_driver_wins():
    assert _budget(driver_free_gb=11.0)["budget_gb"] == 10.0  # driver is the constraint
    assert _budget(mem_available_gb=14.0)["budget_gb"] == 5.0  # kernel is the constraint


def test_llm_server_takes_priority_and_the_floor_is_respected():
    # Only the floor left => nothing for the trainer, and never a negative budget.
    assert _budget(mem_available_gb=8.0)["budget_gb"] == 0.0
    assert _budget(mem_available_gb=5.0)["budget_gb"] == 0.0
    assert _budget(driver_free_gb=0.4)["budget_gb"] == 0.0


def test_overhead_is_deducted_from_a_derived_budget_but_not_an_override():
    # An explicit limit is the operator accepting the risk; it is taken verbatim.
    assert _budget(reserve_gb=0.0)["budget_gb"] == 20.0
    assert _budget(reserve_gb=4.0)["budget_gb"] == 16.0
    assert _budget(limit_gb=25.0, reserve_gb=4.0)["budget_gb"] == 25.0


def test_explicit_override_wins_and_is_labeled():
    r = _budget(limit_gb=25.0, mem_available_gb=12.0)
    assert (r["source"], r["budget_gb"]) == ("explicit", 25.0)


def test_fraction_is_a_fraction_of_total_and_never_wild():
    assert _budget()["fraction"] == pytest.approx(19.0 / 121.6, abs=1e-3)
    assert _budget(total_gb=0.0)["fraction"] == 0.0  # unknown total => no cap, not a guess
    assert _budget(mem_available_gb=999.0, driver_free_gb=999.0)["fraction"] == 1.0


# ---------------------------------------------------------------- enforcement


@_NEEDS_CUDA
def test_cap_is_installed_and_receipted():
    cfg = CFMConfig()
    rec = gpu_budget.enforce(cfg, device="cuda")
    assert rec["applied"] is True and rec["source"].startswith(
        "derived"
    )  # "derived" or "derived+unified" (machine property)
    assert 0.0 < rec["fraction"] <= 1.0
    # the cap must actually be visible to torch, not just recorded by us
    assert torch.cuda.get_per_process_memory_fraction() <= rec["fraction"] + 1e-6
    assert gpu_budget.receipt(cfg) is not rec  # a copy: the registry cannot mutate it


@_NEEDS_CUDA
def test_device_string_is_normalized_to_an_index():
    # `set_per_process_memory_fraction` rejects "cuda"; this is what used to throw.
    assert gpu_budget._device_index(torch, "cuda") == torch.cuda.current_device()
    assert gpu_budget._device_index(torch, "cuda:0") == 0
    with pytest.raises(ValueError):
        gpu_budget._device_index(torch, "cpu")


@_NEEDS_CUDA
def test_uninstallable_cap_refuses_the_run_instead_of_training_uncapped(monkeypatch):
    def boom(*a, **k):
        raise ValueError("Expected a torch.device with a specified index or an integer")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", boom)
    cfg = CFMConfig()
    with pytest.raises(SystemExit) as exc:
        gpu_budget.enforce(cfg)
    assert "uncapped" in str(exc.value)  # names the danger, not just the exception
    assert cfg._gpu_budget_receipt["applied"] is False
    assert "ValueError" in cfg._gpu_budget_receipt["skipped"]  # the reason survives


@_NEEDS_CUDA
def test_starved_pool_is_refused_not_attempted(monkeypatch):
    """A too-small budget must raise WITH the fix, not start a doomed run."""
    monkeypatch.setattr(gpu_budget, "available_gb", lambda *a, **k: 8.5)  # floor + 0.5
    with pytest.raises(SystemExit) as exc:
        gpu_budget.enforce(CFMConfig())
    assert "gpu_mem_limit_gb" in str(exc.value)  # actionable, never a bare crash


def test_cpu_box_is_never_blocked_by_the_guard():
    """The guard protects a co-tenant; it must not break the CPU fallback (Principle 5)."""
    cfg = CFMConfig(device="cpu")
    rec = gpu_budget.enforce(cfg)
    assert rec.get("applied") is False and rec["skipped"] == "device=cpu"
    assert gpu_budget.release(cfg, "train") is None


@_NEEDS_CUDA
def test_asking_for_cuda_without_cuda_is_a_refusal(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        gpu_budget.enforce(CFMConfig(), device="cuda")
    assert "device=cpu" in str(exc.value)  # the alternative is named in the error


@_NEEDS_CUDA
def test_phase_receipts_accumulate_only_when_capped():
    cfg = CFMConfig()
    cfg._gpu_budget_receipt = dict(_budget(), applied=True)
    for phase in ("train", "whiten_save"):
        assert gpu_budget.release(cfg, phase) is not None
    rec = gpu_budget.receipt(cfg)
    assert [p["phase"] for p in rec["phases"]] == ["train", "whiten_save"]
    # an ordinary phase boundary records the peak and NOTHING else: no re-derivation
    assert "recap_budget_gb" not in rec["phases"][0]


@_NEEDS_CUDA
def test_the_cap_is_re_derived_only_at_the_informed_moment(monkeypatch):
    """`release` re-basing at every boundary means the guard takes a second opinion the run
    never asked for — and at an abort boundary it invents a budget out of the wreckage of
    the crash (measured 2026-10-29: 0.0 GB, on a run that had trained 99 steps fine)."""
    cfg = CFMConfig()
    cfg._gpu_budget_receipt = dict(_budget(), applied=True)
    asked = []
    monkeypatch.setattr(
        gpu_budget, "enforce", lambda *a, **k: asked.append(k.get("rebase")) or _budget()
    )
    assert gpu_budget.release(cfg, "train") is not None
    assert asked == []
    gpu_budget.release(cfg, "pre_train", rebase=True)
    assert asked == [True]


@_NEEDS_CUDA
def test_a_crash_receipt_never_replaces_the_crash(monkeypatch, capsys):
    """`release(cfg, "train_aborted")` runs INSIDE an `except BaseException` handler. If it
    raises there, the receipt eats the failure it was written to document and the loop
    reports the wrong crash. It must record, warn, and stay silent about control flow."""
    cfg = CFMConfig()
    cfg._gpu_budget_receipt = dict(_budget(), applied=True)

    def refuse(*a, **k):  # the re-derivation decides the run is unaffordable
        raise SystemExit("gpu-budget: only 0.0 GB is left after the system floor")

    monkeypatch.setattr(gpu_budget, "enforce", refuse)
    boom = RuntimeError("the real crash (an OOM in the step loop)")
    with pytest.raises(RuntimeError) as exc:
        try:
            raise boom
        except BaseException:
            gpu_budget.release(cfg, "train_aborted", rebase=True)
            raise
    assert exc.value is boom  # the original exception survives, unmodified
    assert "RECAP skipped" in capsys.readouterr().out  # the refusal is reported, not swallowed
    assert [p["phase"] for p in gpu_budget.receipt(cfg)["phases"]] == ["train_aborted"]


@_NEEDS_CUDA
def test_release_cannot_raise(monkeypatch, capsys):
    """Total function by contract: a memory guard that can throw from a cleanup path turns
    one failure into two, and the second one is the one you see."""
    cfg = CFMConfig()
    cfg._gpu_budget_receipt = dict(_budget(), applied=True)
    monkeypatch.setattr(
        torch.cuda, "memory_stats", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("driver"))
    )
    assert gpu_budget.release(cfg, "train") is None
    assert "release('train') failed" in capsys.readouterr().out  # reported, never silent


@_NEEDS_CUDA
def test_rebase_re_derives_from_current_memory_and_keeps_history(monkeypatch):
    """The launch cap is affordable before the dataset exists; the phase cap must not be."""
    cfg = CFMConfig()
    # mock BOTH sides so the comparison is machine-independent: at launch a co-tenant holds
    # the pool (available 30), then it leaves (available 40). Real available at test time
    # varies with the co-tenant and made this assertion flip (2026-11-01: 52 GB free).
    monkeypatch.setattr(gpu_budget, "available_gb", lambda *a, **k: 30.0)
    first = gpu_budget.enforce(cfg)
    assert first["rebases"] == 0
    monkeypatch.setattr(gpu_budget, "available_gb", lambda *a, **k: 40.0)  # co-tenant left
    second = gpu_budget.enforce(cfg, rebase=True)
    assert second["rebases"] == 1
    assert second["budget_gb"] > first["budget_gb"]  # it moved with reality, not a constant
    # ... and a plain call after that is still idempotent (no silent re-derivation)
    assert gpu_budget.enforce(cfg)["budget_gb"] == second["budget_gb"]
    # the phase history survives the rebase (one receipt, not two)
    cfg._gpu_budget_receipt["phases"].append({"phase": "x"})
    assert gpu_budget.enforce(cfg, rebase=True)["phases"] == [{"phase": "x"}]


# ---------------------------------------------------------------- config surface


def test_floor_limit_and_reserve_are_config_not_code():
    cfg = CFMConfig()
    assert (cfg.gpu_system_floor_gb, cfg.gpu_mem_limit_gb, cfg.gpu_reserve_gb) == (8.0, 0.0, 0.5)
    cfg.set_overrides = ["gpu_mem_limit_gb=18", "gpu_system_floor_gb=10", "gpu_reserve_gb=2"]
    # int literals into float fields must widen — the fix command this module prints says
    # `--set gpu_mem_limit_gb=20`, and a rejection there is a dead end.
    assert apply_set_overrides(cfg) == {
        "gpu_mem_limit_gb": 18.0,
        "gpu_system_floor_gb": 10.0,
        "gpu_reserve_gb": 2.0,
    }
    assert isinstance(cfg.gpu_mem_limit_gb, float)


def test_unparseable_override_is_rejected_naming_the_field():
    bad = CFMConfig()
    bad.set_overrides = ["gpu_mem_limit_gb=lots"]
    with pytest.raises(SystemExit) as exc:
        apply_set_overrides(bad)
    assert "gpu_mem_limit_gb" in str(exc.value)  # names the field; no raw ValueError
    bad2 = CFMConfig()
    bad2.set_overrides = ["epochs=many"]  # int field, unparseable text
    with pytest.raises(SystemExit):
        apply_set_overrides(bad2)
    ok = CFMConfig()
    ok.set_overrides = ["epochs=+3"]  # literal_eval rejects "+3"; the field type coerces it
    assert apply_set_overrides(ok) == {"epochs": 3}


# ---------------------------------------------------------------- the watchdog


def test_watchdog_never_kills_on_an_unreadable_meminfo(tmp_path):
    assert mem_available_gb(str(tmp_path / "missing")) == float("inf")
    f = tmp_path / "meminfo"
    f.write_text("MemTotal: 1000000 kB\nMemAvailable: 12345678 kB\n")
    assert abs(mem_available_gb(str(f)) - 11.77) < 0.02


def _fake_proc(tmp_path, **procs):
    """A /proc fixture: {dir_name: cmdline}. The first version of `gpu_jobs` crashed on
    `Path.isdigit()` before it could kill anything — an invisible guard is no guard."""
    for name, cmdline in procs.items():
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    return tmp_path


def test_watchdog_sees_the_gpu_jobs_and_ignores_everything_else(tmp_path):
    fake = _fake_proc(
        tmp_path,
        **{
            "100": ".venv/bin/python -m looking_glass.customer_foundation_model train",
            "200": ".venv/bin/python scripts/measure_dynamics_law.py --out x.json",
            "300": ".venv/bin/python -m pytest tests/ -q",
            "not-a-pid": "should be skipped, not crash the scan",
            "400": ".venv/bin/python scripts/gpu_watchdog.py --floor-gb 8",  # itself
        },
    )
    assert sorted(gpu_jobs(DEFAULT_PATTERNS, proc_root=fake)) == [100, 200]


def test_watchdog_survives_a_proc_entry_vanishing_mid_scan(tmp_path):
    fake = _fake_proc(tmp_path, **{"111": "python measure_battery_law.py"})
    (fake / "222").mkdir()  # a pid directory with no readable cmdline (process exited)
    assert sorted(gpu_jobs(DEFAULT_PATTERNS, proc_root=fake)) == [111]


def test_watchdog_polices_every_gpu_entry_point():
    joined = " ".join(DEFAULT_PATTERNS)
    for name in ("customer_foundation_model", "measure_battery_law", "measure_dynamics_law"):
        assert name in joined


def test_guard_ends_when_its_launcher_does():
    """A guard that outlives its job is a hazard, not a safety net.

    Five stale guards survived their shell wrappers (killing the wrapper signals the
    wrapper, not the guard) and went on SIGTERM'ing an unrelated trainer at 7.8 GB while
    writing the verdict to a log nobody was reading. Probed in a CHILD process so the
    pytest session never takes on a parent-death SIGTERM contract of its own.
    """
    import subprocess

    probe = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
        "import gpu_watchdog as g; print('ARMED' if g.arm_parent_death_signal() else 'NOT')"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "ARMED", out.stderr[-400:]


def test_watchdog_pidfile_is_exact_and_cleaned_up(tmp_path):
    pf = tmp_path / "run" / "guard.pid"
    gpu_watchdog._write_pidfile(pf)
    assert pf.read_text().strip() == str(os.getpid())
    # A live GPU producer turns the --once sweep into live fire: it would SIGTERM a trainer
    # sitting outside the slice (measured 2026-11-01: this test's sweep killed two DINO runs
    # mid-training — the harness placed them in a vte scope, not wonderland.slice). The
    # pidfile mechanics are what this test pins, so refuse to sweep while a producer is up.
    if gpu_watchdog.gpu_jobs(DEFAULT_PATTERNS):
        import pytest

        pytest.skip("live GPU producer present — refusing a live-fire watchdog sweep")
    assert gpu_watchdog.main(["--once", "--pidfile", str(pf)], arm=False) in (0, 2)
    assert not pf.exists()  # the guard removes its own pidfile; no stale handles


def test_watchdog_refuses_a_silly_interval_or_delta():
    for bad in (["--interval", "0"], ["--report-delta-gb", "0"], ["--interval", "-1"]):
        with pytest.raises(SystemExit):
            gpu_watchdog.main(bad, arm=False)


def test_allocator_policy_is_pinned_unless_the_operator_set_it():
    # Pinned at import (like the triton cache pin) so it is set before CUDA init.
    from looking_glass import cfm_model  # noqa: F401  (import = the behavior under test)

    assert "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


# ---------------------------------------------------------------- activation checkpointing
# Checkpointing must buy headroom with memory, never with correctness.


def _ssm_pair():
    """Two identical MultiScaleSSM, one checkpointed, on CPU: the equivalence claim is
    device-independent, the memory claim below is measured on the GPU."""
    from looking_glass.cfm_model import MultiScaleSSM

    torch.manual_seed(0)
    plain = MultiScaleSSM(chan=8, n_experts=2, delta_biases=[-2.0, -0.5])
    ckpt = MultiScaleSSM(chan=8, n_experts=2, delta_biases=[-2.0, -0.5], grad_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    return plain, ckpt


def _loss_and_grads(model, x, dt, mask):
    model.zero_grad(set_to_none=True)
    H, _last = model(x, dt=dt, mask=mask)  # forward returns (states, final)
    loss = H.pow(2).mean()
    loss.backward()
    grads = {k: t.grad.detach().clone() for k, t in model.named_parameters() if t.grad is not None}
    return loss.item(), grads


def test_grad_checkpoint_preserves_ssm_values_and_gradients():
    """Recomputation has to yield the SAME gradients as retention. Checkpointing an
    in-place recurrent scan is exactly where a silent divergence is easy to introduce and
    impossible to see in a loss curve — so it is asserted, not assumed."""
    plain, ckpt = _ssm_pair()
    torch.manual_seed(7)
    x, dt, mask = torch.randn(4, 12, 8), torch.rand(4, 12, 2), torch.ones(4, 12)

    l0, g0 = _loss_and_grads(plain, x, dt, mask)
    l1, g1 = _loss_and_grads(ckpt, x, dt, mask)

    assert l1 == pytest.approx(l0, abs=1e-7), (l0, l1)
    assert set(g1) == set(g0), sorted(set(g0) ^ set(g1))
    worst = max((g0[k] - g1[k]).abs().max().item() for k in g0)
    assert worst < 1e-7, worst


def test_grad_checkpoint_is_off_by_default():
    """It trades step time for headroom, so adoption is a measured decision recorded in the
    registry — never a silent default that quietly changes the cost of every run."""
    from looking_glass.cfm_config import CFMConfig as _C
    from looking_glass.cfm_model import MultiScaleSSM

    assert _C().grad_checkpoint is False
    assert MultiScaleSSM(chan=8, n_experts=1, delta_biases=[-1.0]).grad_checkpoint is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak memory is a GPU claim")
def test_grad_checkpoint_lowers_the_step_peak():
    """And it has to actually pay. A flag that costs step time and buys no headroom must
    not be adopted, so the saving is asserted with a margin, not just a direction."""
    from looking_glass.cfm_model import MultiScaleSSM

    def peak(grad_checkpoint):
        torch.manual_seed(0)
        m = MultiScaleSSM(
            chan=256, n_experts=2, delta_biases=[-2.0, -0.5], grad_checkpoint=grad_checkpoint
        ).cuda()
        x, dt, mask = (
            torch.randn(8, 512, 256, device="cuda"),
            torch.rand(8, 512, 2, device="cuda"),
            torch.ones(8, 512, device="cuda"),
        )
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        m(x, dt=dt, mask=mask)[0].pow(2).mean().backward()
        return (torch.cuda.max_memory_allocated() - base) / 2**30

    plain, ckpt = peak(False), peak(True)
    assert ckpt < 0.9 * plain, f"checkpointing saved nothing: {plain:.3f} -> {ckpt:.3f} GB"


# ---------------------------------------------------------------- the container ceiling (DEC-450)


def test_container_ceiling_clamps_the_derived_budget_and_is_receipted():
    """A container's --memory is a real kernel limit; device allocations are NOT charged to
    it (measured with vLLM: 337 MiB reported, 97 GiB held), so the allocator cap must leave
    host usage + overhead INSIDE the ceiling. min(derived, limit - used - reserve)."""
    r = _budget(container_limit_gb=20.0, container_used_gb=4.0, reserve_gb=1.0)
    assert r["source"] == "derived+cgroup"
    assert r["budget_gb"] == 15.0  # 20 - 4 - 1; the host-derived 19 was looser
    assert (r["container_limit_gb"], r["container_used_gb"]) == (20.0, 4.0)
    assert r["container_clamp_gb"] == 15.0


def test_a_loose_container_ceiling_does_not_change_the_derived_budget():
    r = _budget(container_limit_gb=60.0, container_used_gb=2.0)
    assert (r["source"], r["budget_gb"]) == ("derived", 19.0)
    assert r["container_clamp_gb"] == 57.0  # recorded, but it did not bind


def test_an_explicit_override_cannot_escape_the_container():
    """The operator may accept the co-tenant risk; they cannot override the kernel."""
    r = _budget(limit_gb=25.0, container_limit_gb=20.0, container_used_gb=5.0, reserve_gb=0.5)
    assert r["source"] == "explicit+cgroup"
    assert r["budget_gb"] == 14.5  # 20 - 5 - 0.5, not 25


def test_cgroup_reader_treats_max_and_missing_as_unlimited(tmp_path):
    (tmp_path / "max").write_text("max")
    assert gpu_budget.cgroup_memory_gb(str(tmp_path / "max"), str(tmp_path / "cur")) == (0.0, 0.0)
    assert gpu_budget.cgroup_memory_gb(str(tmp_path / "nope"), str(tmp_path / "cur")) == (0.0, 0.0)
    (tmp_path / "max").write_text(str(20 * 2**30))
    (tmp_path / "cur").write_text(str(3 * 2**30))
    lim, used = gpu_budget.cgroup_memory_gb(str(tmp_path / "max"), str(tmp_path / "cur"))
    assert (lim, used) == (20.0, 3.0)


def test_watchdog_polices_a_container_client_but_never_a_shell_that_names_it(tmp_path):
    """The containerized job is invisible to /proc (it is not python), so the guard matches
    the `docker run` CLIENT by container name. A shell whose command text mentions the name
    must not match — that is the measured launcher-kill defect one layer over."""
    fake = _fake_proc(
        tmp_path,
        **{
            "500": "docker run --rm --init --name wonderland-run wonderland-runtime",
            "501": "/bin/bash -c echo --name wonderland-run",
            "502": "docker run --rm --name other image",
            "503": "docker ps",
            "504": "podman run --name=wonderland-run image",
        },
    )
    assert sorted(gpu_watchdog.container_jobs("wonderland-run", proc_root=fake)) == [500, 504]
