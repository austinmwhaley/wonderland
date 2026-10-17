"""CPU-only structural tests for scripts/apex_pipeline.py (no stages are executed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "apex_pipeline", ROOT / "scripts" / "apex_pipeline.py"
)
apex_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apex_pipeline)

KW = dict(
    n_customers=123,
    window_days=99,
    seed=7,
    as_of="2025-01-01",
    out_root="/tmp/apex_out",
    customers=45,
    sample_a=23,
    max_steps=11,
)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_stage_count_and_generation_commands():
    cmds = apex_pipeline.build_commands(**KW)
    assert len(cmds) == 5

    apex, null = cmds[0], cmds[1]
    for cmd, out in ((apex, "/tmp/apex_out/apex"), (null, "/tmp/apex_out/apex_null")):
        assert cmd[1:3] == ["-m", "rabbit_hole.generators.apex"]
        assert _flag(cmd, "--n-customers") == "123"
        assert _flag(cmd, "--window-days") == "99"
        assert _flag(cmd, "--seed") == "7"
        assert _flag(cmd, "--out") == out
    assert "--snr" not in apex
    assert _flag(null, "--snr") == "0"


def test_train_commands():
    cmds = apex_pipeline.build_commands(**KW)
    for cmd, db, out in (
        (cmds[2], "/tmp/apex_out/apex/customer_event_stream.parquet", "/tmp/apex_out/apex_cfm"),
        (
            cmds[3],
            "/tmp/apex_out/apex_null/customer_event_stream.parquet",
            "/tmp/apex_out/apex_null_cfm",
        ),
    ):
        assert cmd[1:4] == ["-m", "looking_glass.customer_foundation_model", "train"]
        assert _flag(cmd, "--db") == db
        assert _flag(cmd, "--customers") == "45"
        assert _flag(cmd, "--sample-a") == "23"
        assert _flag(cmd, "--anchors") == "3"
        assert _flag(cmd, "--as-of") == "2025-01-01"
        assert _flag(cmd, "--out-dir") == out
        sets = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--set"]
        assert "use_measure=True" in sets
        assert "measure_dim=128" in sets
        assert "effect_actions=('freq','hour','dow','channel','offer')" in sets
        assert "action_fields=('freq','hour','dow','channel','offer','discount_pct')" in sets
        assert "max_steps=11" in sets


def test_card_command_and_placeholders():
    card = apex_pipeline.build_commands(**KW)[4]
    assert card[1:3] == ["-m", "looking_glass.foundation_report"]
    assert _flag(card, "--db") == "/tmp/apex_out/apex/customer_event_stream.parquet"
    assert _flag(card, "--ckpt-dir") == "/tmp/apex_out/apex_cfm"
    assert _flag(card, "--tag") == "<TAG>"
    assert _flag(card, "--as-of") == "2025-01-01"
    assert _flag(card, "--stream") == "apex"
    assert _flag(card, "--customers") == "45"
    assert _flag(card, "--horizons") == "30,90"
    assert "--white-queen" in card
    assert _flag(card, "--truth") == "/tmp/apex_out/apex/apex_truth.json"
    assert _flag(card, "--null-db") == "/tmp/apex_out/apex_null/customer_event_stream.parquet"
    assert _flag(card, "--null-tag") == "<NULLTAG>"
    assert _flag(card, "--null-ckpt-dir") == "/tmp/apex_out/apex_null_cfm"
    assert _flag(card, "--out") == "/tmp/apex_out/card_apex.md"


def test_dry_run_prints_without_executing(monkeypatch, capsys):
    def _boom(*args, **kwargs):
        raise AssertionError("dry run must not execute stages")

    monkeypatch.setattr(apex_pipeline.subprocess, "run", _boom)
    rc = apex_pipeline.main(["--dry-run", "--as-of", "2025-01-01", "--out-root", "/tmp/dry"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.strip().splitlines()
    assert len(lines) == 5
    assert "rabbit_hole.generators.apex" in lines[0]
    assert "looking_glass.foundation_report" in lines[4]
    assert "<TAG>" in lines[4] and "<NULLTAG>" in lines[4]
    assert "/tmp/dry/card_apex.md" in lines[4]
